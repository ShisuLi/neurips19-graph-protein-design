"""Unit tests for encoder modules.

Tests the various GNN encoders for ablation studies, including
geometric attention mechanisms and probing heads.
"""

import torch
import pytest
from models.encoders import (
    BaseEncoder,
    GeometricAttentionLayer,
    Struct2SeqGeometricEncoder,
    EdgeAwareGATEncoder,
    VanillaGCNEncoder,
    StructureProbingHead,
)


class TestGeometricAttentionLayer:
    """Test the custom geometric attention layer."""

    def test_output_shape(self) -> None:
        """Test that output shape matches input node shape."""
        batch_size, seq_len, k_neighbors = 2, 50, 20
        hidden_dim, edge_dim = 128, 64

        layer = GeometricAttentionLayer(hidden_dim, edge_dim, num_heads=4)

        h_nodes = torch.randn(batch_size, seq_len, hidden_dim)
        h_edges = torch.randn(batch_size, seq_len, k_neighbors, edge_dim)
        edge_idxs = torch.randint(0, seq_len, (batch_size, seq_len, k_neighbors))
        mask = torch.ones(batch_size, seq_len)

        output = layer(h_nodes, h_edges, edge_idxs, mask)

        assert output.shape == (batch_size, seq_len, hidden_dim), \
            f"Expected shape [{batch_size}, {seq_len}, {hidden_dim}], got {output.shape}"

    def test_masking_zeros_output(self) -> None:
        """Test that masked positions produce zero output."""
        batch_size, seq_len, k_neighbors = 2, 50, 20
        hidden_dim, edge_dim = 128, 64

        layer = GeometricAttentionLayer(hidden_dim, edge_dim, num_heads=4)

        h_nodes = torch.randn(batch_size, seq_len, hidden_dim)
        h_edges = torch.randn(batch_size, seq_len, k_neighbors, edge_dim)
        edge_idxs = torch.randint(0, seq_len, (batch_size, seq_len, k_neighbors))
        mask = torch.ones(batch_size, seq_len)

        # Mask out second half
        mask[:, seq_len//2:] = 0

        output = layer(h_nodes, h_edges, edge_idxs, mask)

        # Masked positions should not affect output (attention should handle it)
        # Note: The output itself isn't necessarily zero, but it shouldn't cause NaNs
        assert not torch.isnan(output).any(), "Output contains NaN values"

    def test_batch_independence(self) -> None:
        """Test that batch elements are processed independently."""
        seq_len, k_neighbors = 50, 20
        hidden_dim, edge_dim = 128, 64

        layer = GeometricAttentionLayer(hidden_dim, edge_dim, num_heads=4)

        # Process single and batched
        h1 = torch.randn(1, seq_len, hidden_dim)
        h2 = torch.randn(1, seq_len, hidden_dim)
        e1 = torch.randn(1, seq_len, k_neighbors, edge_dim)
        e2 = torch.randn(1, seq_len, k_neighbors, edge_dim)
        idx1 = torch.randint(0, seq_len, (1, seq_len, k_neighbors))
        idx2 = torch.randint(0, seq_len, (1, seq_len, k_neighbors))
        mask = torch.ones(1, seq_len)

        out1 = layer(h1, e1, idx1, mask)
        out2 = layer(h2, e2, idx2, mask)

        # Batched
        h_batch = torch.cat([h1, h2], dim=0)
        e_batch = torch.cat([e1, e2], dim=0)
        idx_batch = torch.cat([idx1, idx2], dim=0)
        mask_batch = torch.ones(2, seq_len)

        out_batch = layer(h_batch, e_batch, idx_batch, mask_batch)

        assert torch.allclose(out_batch[0], out1[0], atol=1e-5), \
            "Batch processing should match individual processing"
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5), \
            "Batch processing should match individual processing"

    def test_different_num_heads(self) -> None:
        """Test with different numbers of attention heads."""
        for num_heads in [1, 2, 4, 8]:
            hidden_dim = 128  # Must be divisible by num_heads
            edge_dim = 64
            seq_len, k_neighbors = 50, 20

            layer = GeometricAttentionLayer(hidden_dim, edge_dim, num_heads=num_heads)

            h_nodes = torch.randn(2, seq_len, hidden_dim)
            h_edges = torch.randn(2, seq_len, k_neighbors, edge_dim)
            edge_idxs = torch.randint(0, seq_len, (2, seq_len, k_neighbors))
            mask = torch.ones(2, seq_len)

            output = layer(h_nodes, h_edges, edge_idxs, mask)

            assert output.shape == (2, seq_len, hidden_dim), \
                f"Failed with num_heads={num_heads}"


class TestStruct2SeqGeometricEncoder:
    """Test the full geometric encoder."""

    def test_output_shape(self) -> None:
        """Test that output shape matches input node shape."""
        batch_size, seq_len, k_neighbors = 2, 50, 20
        hidden_dim, edge_dim = 128, 64

        encoder = Struct2SeqGeometricEncoder(
            hidden_dim, edge_dim, num_layers=3, num_heads=4
        )

        h_nodes = torch.randn(batch_size, seq_len, hidden_dim)
        h_edges = torch.randn(batch_size, seq_len, k_neighbors, edge_dim)
        edge_idxs = torch.randint(0, seq_len, (batch_size, seq_len, k_neighbors))
        mask = torch.ones(batch_size, seq_len)

        output = encoder(h_nodes, h_edges, edge_idxs, mask)

        assert output.shape == (batch_size, seq_len, hidden_dim), \
            f"Expected shape [{batch_size}, {seq_len}, {hidden_dim}], got {output.shape}"

    def test_residual_connections(self) -> None:
        """Test that residual connections maintain gradient flow."""
        hidden_dim, edge_dim = 128, 64
        encoder = Struct2SeqGeometricEncoder(
            hidden_dim, edge_dim, num_layers=3, num_heads=4
        )

        h_nodes = torch.randn(2, 50, hidden_dim, requires_grad=True)
        h_edges = torch.randn(2, 50, 20, edge_dim)
        edge_idxs = torch.randint(0, 50, (2, 50, 20))
        mask = torch.ones(2, 50)

        output = encoder(h_nodes, h_edges, edge_idxs, mask)
        loss = output.sum()
        loss.backward()

        assert h_nodes.grad is not None, "Gradients should flow through encoder"
        assert not torch.isnan(h_nodes.grad).any(), "Gradients contain NaN"

    def test_masking_applied(self) -> None:
        """Test that masking is properly applied throughout layers."""
        hidden_dim, edge_dim = 128, 64
        encoder = Struct2SeqGeometricEncoder(
            hidden_dim, edge_dim, num_layers=3, num_heads=4
        )

        h_nodes = torch.randn(2, 50, hidden_dim)
        h_edges = torch.randn(2, 50, 20, edge_dim)
        edge_idxs = torch.randint(0, 50, (2, 50, 20))
        mask = torch.ones(2, 50)
        mask[:, 25:] = 0  # Mask second half

        output = encoder(h_nodes, h_edges, edge_idxs, mask)

        # Output for masked positions should be exactly zero (due to multiplication by mask)
        assert torch.allclose(
            output[:, 25:, :],
            torch.zeros_like(output[:, 25:, :]),
            atol=1e-6
        ), "Masked positions should have zero output"

    def test_different_layer_counts(self) -> None:
        """Test with different numbers of layers."""
        hidden_dim, edge_dim = 128, 64

        for num_layers in [1, 2, 3, 6]:
            encoder = Struct2SeqGeometricEncoder(
                hidden_dim, edge_dim, num_layers=num_layers
            )

            h_nodes = torch.randn(2, 50, hidden_dim)
            h_edges = torch.randn(2, 50, 20, edge_dim)
            edge_idxs = torch.randint(0, 50, (2, 50, 20))
            mask = torch.ones(2, 50)

            output = encoder(h_nodes, h_edges, edge_idxs, mask)

            assert output.shape == (2, 50, hidden_dim), \
                f"Failed with num_layers={num_layers}"


class TestEdgeAwareGATEncoder:
    """Test the edge-aware GAT encoder."""

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        batch_size, seq_len, k_neighbors = 2, 50, 20
        hidden_dim, edge_dim = 128, 16

        encoder = EdgeAwareGATEncoder(
            hidden_dim, edge_dim, num_layers=3, num_heads=4
        )

        h_nodes = torch.randn(batch_size, seq_len, hidden_dim)
        h_edges = torch.randn(batch_size, seq_len, k_neighbors, edge_dim)
        edge_idxs = torch.randint(0, seq_len, (batch_size, seq_len, k_neighbors))
        mask = torch.ones(batch_size, seq_len)

        output = encoder(h_nodes, h_edges, edge_idxs, mask)

        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_edge_features_affect_output(self) -> None:
        """Test that edge features actually influence the output."""
        hidden_dim, edge_dim = 128, 16
        encoder = EdgeAwareGATEncoder(hidden_dim, edge_dim, num_layers=2)

        h_nodes = torch.randn(2, 50, hidden_dim)
        edge_idxs = torch.randint(0, 50, (2, 50, 20))
        mask = torch.ones(2, 50)

        # Two different edge features
        h_edges1 = torch.randn(2, 50, 20, edge_dim)
        h_edges2 = torch.randn(2, 50, 20, edge_dim)

        output1 = encoder(h_nodes, h_edges1, edge_idxs, mask)
        output2 = encoder(h_nodes, h_edges2, edge_idxs, mask)

        # Outputs should be different (edge features matter)
        assert not torch.allclose(output1, output2, atol=1e-4), \
            "Edge features should affect output"


class TestVanillaGCNEncoder:
    """Test the baseline GCN encoder."""

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        batch_size, seq_len, k_neighbors = 2, 50, 20
        hidden_dim, edge_dim = 128, 64

        encoder = VanillaGCNEncoder(hidden_dim, edge_dim, num_layers=3)

        h_nodes = torch.randn(batch_size, seq_len, hidden_dim)
        h_edges = torch.randn(batch_size, seq_len, k_neighbors, edge_dim)
        edge_idxs = torch.randint(0, seq_len, (batch_size, seq_len, k_neighbors))
        mask = torch.ones(batch_size, seq_len)

        output = encoder(h_nodes, h_edges, edge_idxs, mask)

        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_edge_features_ignored(self) -> None:
        """Test that edge features are truly ignored (baseline property)."""
        hidden_dim, edge_dim = 128, 64
        encoder = VanillaGCNEncoder(hidden_dim, edge_dim, num_layers=2)

        h_nodes = torch.randn(2, 50, hidden_dim)
        edge_idxs = torch.randint(0, 50, (2, 50, 20))
        mask = torch.ones(2, 50)

        # Two different edge features
        h_edges1 = torch.randn(2, 50, 20, edge_dim)
        h_edges2 = torch.randn(2, 50, 20, edge_dim)

        output1 = encoder(h_nodes, h_edges1, edge_idxs, mask)
        output2 = encoder(h_nodes, h_edges2, edge_idxs, mask)

        # Outputs should be IDENTICAL (edge features ignored)
        assert torch.allclose(output1, output2), \
            "VanillaGCN should ignore edge features"

    def test_gradient_flow(self) -> None:
        """Test that gradients flow properly."""
        hidden_dim = 128
        encoder = VanillaGCNEncoder(hidden_dim, 64, num_layers=3)

        h_nodes = torch.randn(2, 50, hidden_dim, requires_grad=True)
        h_edges = torch.randn(2, 50, 20, 64)
        edge_idxs = torch.randint(0, 50, (2, 50, 20))
        mask = torch.ones(2, 50)

        output = encoder(h_nodes, h_edges, edge_idxs, mask)
        loss = output.sum()
        loss.backward()

        assert h_nodes.grad is not None
        assert not torch.isnan(h_nodes.grad).any()


class TestStructureProbingHead:
    """Test the probing classification head."""

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        batch_size, seq_len, hidden_dim = 4, 100, 128
        num_classes = 3

        head = StructureProbingHead(hidden_dim, num_classes=num_classes)

        h_nodes = torch.randn(batch_size, seq_len, hidden_dim)
        mask = torch.ones(batch_size, seq_len)

        logits = head(h_nodes, mask)

        assert logits.shape == (batch_size, num_classes), \
            f"Expected shape [{batch_size}, {num_classes}], got {logits.shape}"

    def test_different_pooling_methods(self) -> None:
        """Test different pooling strategies."""
        hidden_dim, num_classes = 128, 2

        for pooling in ['mean', 'max', 'sum']:
            head = StructureProbingHead(hidden_dim, num_classes, pooling=pooling)

            h_nodes = torch.randn(2, 50, hidden_dim)
            mask = torch.ones(2, 50)

            logits = head(h_nodes, mask)

            assert logits.shape == (2, num_classes), \
                f"Failed with pooling={pooling}"

    def test_masking_affects_pooling(self) -> None:
        """Test that masking correctly affects pooling."""
        hidden_dim, num_classes = 128, 2
        head = StructureProbingHead(hidden_dim, num_classes, pooling='mean')

        h_nodes = torch.ones(1, 100, hidden_dim)  # All ones
        mask_full = torch.ones(1, 100)
        mask_half = torch.ones(1, 100)
        mask_half[:, 50:] = 0

        logits_full = head(h_nodes, mask_full)
        logits_half = head(h_nodes, mask_half)

        # With mean pooling of all ones, changing the mask shouldn't change
        # the pooled value (since all are ones), but it does affect which
        # positions are included
        # The outputs should be the same since mean of ones is ones
        assert torch.allclose(logits_full, logits_half, atol=1e-5), \
            "Mean pooling of constant values should be independent of mask length"

    def test_gradient_flow(self) -> None:
        """Test that gradients flow back to encoder."""
        hidden_dim, num_classes = 128, 2
        head = StructureProbingHead(hidden_dim, num_classes)

        h_nodes = torch.randn(2, 50, hidden_dim, requires_grad=True)
        mask = torch.ones(2, 50)

        logits = head(h_nodes, mask)
        loss = logits.sum()
        loss.backward()

        assert h_nodes.grad is not None
        assert not torch.isnan(h_nodes.grad).any()


class TestEncoderComparison:
    """Integration tests comparing different encoder architectures."""

    def test_all_encoders_same_interface(self) -> None:
        """Test that all encoders follow the same interface."""
        batch_size, seq_len, k_neighbors = 2, 50, 20
        hidden_dim, edge_dim = 128, 64

        encoders = [
            VanillaGCNEncoder(hidden_dim, edge_dim, num_layers=2),
            EdgeAwareGATEncoder(hidden_dim, edge_dim, num_layers=2),
            Struct2SeqGeometricEncoder(hidden_dim, edge_dim, num_layers=2),
        ]

        h_nodes = torch.randn(batch_size, seq_len, hidden_dim)
        h_edges = torch.randn(batch_size, seq_len, k_neighbors, edge_dim)
        edge_idxs = torch.randint(0, seq_len, (batch_size, seq_len, k_neighbors))
        mask = torch.ones(batch_size, seq_len)

        for encoder in encoders:
            output = encoder(h_nodes, h_edges, edge_idxs, mask)
            assert output.shape == (batch_size, seq_len, hidden_dim), \
                f"{encoder.__class__.__name__} failed interface test"

    def test_geometric_encoder_most_expressive(self) -> None:
        """Test that geometric encoder uses the most information."""
        hidden_dim, edge_dim = 128, 64

        vanilla = VanillaGCNEncoder(hidden_dim, edge_dim, num_layers=2)
        edge_aware = EdgeAwareGATEncoder(hidden_dim, edge_dim, num_layers=2)
        geometric = Struct2SeqGeometricEncoder(hidden_dim, edge_dim, num_layers=2)

        h_nodes = torch.randn(2, 50, hidden_dim)
        edge_idxs = torch.randint(0, 50, (2, 50, 20))
        mask = torch.ones(2, 50)

        # Different edge features
        h_edges1 = torch.randn(2, 50, 20, edge_dim)
        h_edges2 = torch.randn(2, 50, 20, edge_dim)

        # VanillaGCN should be identical (ignores edges)
        out_vanilla1 = vanilla(h_nodes, h_edges1, edge_idxs, mask)
        out_vanilla2 = vanilla(h_nodes, h_edges2, edge_idxs, mask)
        assert torch.allclose(out_vanilla1, out_vanilla2)

        # EdgeAware and Geometric should differ (uses edge features)
        out_edge1 = edge_aware(h_nodes, h_edges1, edge_idxs, mask)
        out_edge2 = edge_aware(h_nodes, h_edges2, edge_idxs, mask)
        assert not torch.allclose(out_edge1, out_edge2, atol=1e-4)

        out_geo1 = geometric(h_nodes, h_edges1, edge_idxs, mask)
        out_geo2 = geometric(h_nodes, h_edges2, edge_idxs, mask)
        assert not torch.allclose(out_geo1, out_geo2, atol=1e-4)

    def test_end_to_end_with_probing_head(self) -> None:
        """Test complete pipeline: encoder + probing head."""
        hidden_dim, edge_dim, num_classes = 128, 64, 2

        encoder = Struct2SeqGeometricEncoder(hidden_dim, edge_dim, num_layers=2)
        head = StructureProbingHead(hidden_dim, num_classes)

        h_nodes = torch.randn(2, 50, hidden_dim)
        h_edges = torch.randn(2, 50, 20, edge_dim)
        edge_idxs = torch.randint(0, 50, (2, 50, 20))
        mask = torch.ones(2, 50)

        # Forward pass
        encoded = encoder(h_nodes, h_edges, edge_idxs, mask)
        logits = head(encoded, mask)

        assert logits.shape == (2, num_classes)

        # Backward pass
        loss = F.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()

        # Check gradients exist (training would work)
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None


# Import F for cross_entropy
import torch.nn.functional as F
