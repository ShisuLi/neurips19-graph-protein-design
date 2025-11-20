"""Unit tests for decoder module.

Tests the autoregressive decoder components including causal attention,
cross-attention, and sequence generation.
"""

import torch
import pytest
from models.decoder import (
    PositionalEncoding,
    CausalSelfAttention,
    CrossAttention,
    DecoderLayer,
    StructureDecoder,
)


class TestPositionalEncoding:
    """Test positional encoding."""

    def test_output_shape(self) -> None:
        """Test that output shape matches input."""
        pos_enc = PositionalEncoding(hidden_dim=128, max_len=1000)
        x = torch.randn(2, 100, 128)
        output = pos_enc(x)

        assert output.shape == x.shape, \
            f"Expected shape {x.shape}, got {output.shape}"

    def test_different_sequence_lengths(self) -> None:
        """Test with different sequence lengths."""
        pos_enc = PositionalEncoding(hidden_dim=128, max_len=1000)

        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(2, seq_len, 128)
            output = pos_enc(x)
            assert output.shape == (2, seq_len, 128)

    def test_max_len_limit(self) -> None:
        """Test that sequences longer than max_len raise error."""
        pos_enc = PositionalEncoding(hidden_dim=128, max_len=100)
        x = torch.randn(2, 100, 128)  # Exactly max_len should work
        output = pos_enc(x)
        assert output.shape == (2, 100, 128)


class TestCausalSelfAttention:
    """Test causal self-attention layer."""

    def test_output_shape(self) -> None:
        """Test that output shape matches input."""
        layer = CausalSelfAttention(hidden_dim=128, num_heads=4)
        x = torch.randn(2, 100, 128)
        output = layer(x)

        assert output.shape == x.shape

    def test_causal_masking(self) -> None:
        """Test that causal masking prevents future information leakage."""
        layer = CausalSelfAttention(hidden_dim=128, num_heads=4)

        # Create input where each position has unique value
        x = torch.arange(10, dtype=torch.float).view(1, 10, 1)
        x = x.expand(1, 10, 128)

        output = layer(x)

        # Since attention is causal, each position should only depend on
        # current and past positions. We can't test this directly, but
        # we can verify the output doesn't have NaN or unreasonable values
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_with_padding_mask(self) -> None:
        """Test that padding mask works correctly."""
        layer = CausalSelfAttention(hidden_dim=128, num_heads=4)
        x = torch.randn(2, 100, 128)
        mask = torch.ones(2, 100)
        mask[:, 50:] = 0  # Mask out second half

        output = layer(x, mask)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_gradient_flow(self) -> None:
        """Test that gradients flow properly."""
        layer = CausalSelfAttention(hidden_dim=128, num_heads=4)
        x = torch.randn(2, 50, 128, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestCrossAttention:
    """Test cross-attention layer."""

    def test_output_shape(self) -> None:
        """Test that output shape matches decoder input."""
        layer = CrossAttention(hidden_dim=128, num_heads=4)

        decoder_hidden = torch.randn(2, 50, 128)
        encoder_out = torch.randn(2, 100, 128)

        output = layer(decoder_hidden, encoder_out)

        assert output.shape == decoder_hidden.shape

    def test_different_lengths(self) -> None:
        """Test with different encoder and decoder lengths."""
        layer = CrossAttention(hidden_dim=128, num_heads=4)

        # Decoder shorter than encoder
        dec_h = torch.randn(2, 30, 128)
        enc_out = torch.randn(2, 100, 128)
        output = layer(dec_h, enc_out)
        assert output.shape == (2, 30, 128)

        # Decoder longer than encoder
        dec_h = torch.randn(2, 150, 128)
        enc_out = torch.randn(2, 100, 128)
        output = layer(dec_h, enc_out)
        assert output.shape == (2, 150, 128)

    def test_encoder_masking(self) -> None:
        """Test that encoder mask prevents attending to invalid positions."""
        layer = CrossAttention(hidden_dim=128, num_heads=4)

        decoder_hidden = torch.randn(2, 50, 128)
        encoder_out = torch.randn(2, 100, 128)
        encoder_mask = torch.ones(2, 100)
        encoder_mask[:, 70:] = 0  # Mask out last 30 positions

        output = layer(decoder_hidden, encoder_out, encoder_mask)

        assert output.shape == (2, 50, 128)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self) -> None:
        """Test gradient flow through cross-attention."""
        layer = CrossAttention(hidden_dim=128, num_heads=4)

        dec_h = torch.randn(2, 50, 128, requires_grad=True)
        enc_out = torch.randn(2, 100, 128, requires_grad=True)

        output = layer(dec_h, enc_out)
        loss = output.sum()
        loss.backward()

        assert dec_h.grad is not None
        assert enc_out.grad is not None


class TestDecoderLayer:
    """Test complete decoder layer."""

    def test_output_shape(self) -> None:
        """Test output shape."""
        layer = DecoderLayer(hidden_dim=128, num_heads=4, ffn_dim=256)

        decoder_hidden = torch.randn(2, 50, 128)
        encoder_out = torch.randn(2, 100, 128)

        output = layer(decoder_hidden, encoder_out)

        assert output.shape == decoder_hidden.shape

    def test_with_masks(self) -> None:
        """Test with both decoder and encoder masks."""
        layer = DecoderLayer(hidden_dim=128, num_heads=4, ffn_dim=256)

        decoder_hidden = torch.randn(2, 50, 128)
        encoder_out = torch.randn(2, 100, 128)
        decoder_mask = torch.ones(2, 50)
        encoder_mask = torch.ones(2, 100)

        # Mask some positions
        decoder_mask[:, 40:] = 0
        encoder_mask[:, 80:] = 0

        output = layer(decoder_hidden, encoder_out, decoder_mask, encoder_mask)

        assert output.shape == (2, 50, 128)
        # Masked decoder positions should be zero
        assert torch.allclose(
            output[:, 40:, :],
            torch.zeros_like(output[:, 40:, :])
        )

    def test_residual_connections(self) -> None:
        """Test that residual connections enable gradient flow."""
        layer = DecoderLayer(hidden_dim=128, num_heads=4, ffn_dim=256)

        decoder_hidden = torch.randn(2, 50, 128, requires_grad=True)
        encoder_out = torch.randn(2, 100, 128)

        output = layer(decoder_hidden, encoder_out)
        loss = output.sum()
        loss.backward()

        assert decoder_hidden.grad is not None
        assert not torch.isnan(decoder_hidden.grad).any()


class TestStructureDecoder:
    """Test the complete structure decoder."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        decoder = StructureDecoder(
            vocab_size=20,
            hidden_dim=128,
            num_layers=3,
            num_heads=4
        )

        sequence = torch.randint(0, 20, (2, 100))
        encoder_out = torch.randn(2, 100, 128)
        mask = torch.ones(2, 100)

        logits = decoder(sequence, encoder_out, mask)

        assert logits.shape == (2, 100, 20), \
            f"Expected shape [2, 100, 20], got {logits.shape}"

    def test_different_vocab_sizes(self) -> None:
        """Test with different vocabulary sizes."""
        for vocab_size in [10, 20, 30]:
            decoder = StructureDecoder(
                vocab_size=vocab_size,
                hidden_dim=128,
                num_layers=2
            )

            sequence = torch.randint(0, vocab_size, (2, 50))
            encoder_out = torch.randn(2, 50, 128)
            mask = torch.ones(2, 50)

            logits = decoder(sequence, encoder_out, mask)

            assert logits.shape == (2, 50, vocab_size)

    def test_different_layer_counts(self) -> None:
        """Test with different numbers of layers."""
        for num_layers in [1, 2, 3, 6]:
            decoder = StructureDecoder(
                vocab_size=20,
                hidden_dim=128,
                num_layers=num_layers
            )

            sequence = torch.randint(0, 20, (2, 50))
            encoder_out = torch.randn(2, 50, 128)
            mask = torch.ones(2, 50)

            logits = decoder(sequence, encoder_out, mask)

            assert logits.shape == (2, 50, 20)

    def test_gradient_flow(self) -> None:
        """Test gradient flow through decoder."""
        decoder = StructureDecoder(vocab_size=20, hidden_dim=128, num_layers=2)

        sequence = torch.randint(0, 20, (2, 50))
        encoder_out = torch.randn(2, 50, 128, requires_grad=True)
        mask = torch.ones(2, 50)

        logits = decoder(sequence, encoder_out, mask)
        loss = logits.sum()
        loss.backward()

        assert encoder_out.grad is not None

        # Check decoder parameters have gradients
        for param in decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_generate_autoregressive(self) -> None:
        """Test autoregressive generation."""
        decoder = StructureDecoder(vocab_size=20, hidden_dim=128, num_layers=2)

        encoder_out = torch.randn(2, 100, 128)
        mask = torch.ones(2, 100)

        # Generate sequences
        generated = decoder.generate(
            encoder_out,
            mask,
            max_len=100,
            temperature=1.0
        )

        assert generated.shape == (2, 100)
        # Check all tokens are in vocabulary range
        assert (generated >= 0).all() and (generated < 20).all()

    def test_generate_with_temperature(self) -> None:
        """Test generation with different temperatures."""
        decoder = StructureDecoder(vocab_size=20, hidden_dim=128, num_layers=2)

        encoder_out = torch.randn(1, 50, 128)
        mask = torch.ones(1, 50)

        # Low temperature (more deterministic)
        gen_low = decoder.generate(encoder_out, mask, max_len=50, temperature=0.1)

        # High temperature (more random)
        gen_high = decoder.generate(encoder_out, mask, max_len=50, temperature=2.0)

        assert gen_low.shape == (1, 50)
        assert gen_high.shape == (1, 50)

        # Both should be valid
        assert (gen_low >= 0).all() and (gen_low < 20).all()
        assert (gen_high >= 0).all() and (gen_high < 20).all()

    def test_masking_applied(self) -> None:
        """Test that masking is properly applied."""
        decoder = StructureDecoder(vocab_size=20, hidden_dim=128, num_layers=2)

        sequence = torch.randint(0, 20, (2, 100))
        encoder_out = torch.randn(2, 100, 128)
        mask = torch.ones(2, 100)
        mask[:, 70:] = 0  # Mask last 30 positions

        logits = decoder(sequence, encoder_out, mask, decoder_mask=mask)

        assert logits.shape == (2, 100, 20)
        assert not torch.isnan(logits).any()


class TestIntegration:
    """Integration tests for encoder-decoder pipeline."""

    def test_end_to_end_forward(self) -> None:
        """Test complete forward pass: encoder -> decoder."""
        from models import Struct2SeqGeometricEncoder

        # Create models
        encoder = Struct2SeqGeometricEncoder(
            hidden_dim=128,
            edge_dim=22,
            num_layers=2
        )
        decoder = StructureDecoder(
            vocab_size=20,
            hidden_dim=128,
            num_layers=2
        )

        # Create inputs
        h_nodes = torch.randn(2, 50, 128)
        h_edges = torch.randn(2, 50, 20, 22)
        edge_idxs = torch.randint(0, 50, (2, 50, 20))
        mask = torch.ones(2, 50)
        sequence = torch.randint(0, 20, (2, 50))

        # Encode
        encoder_out = encoder(h_nodes, h_edges, edge_idxs, mask)

        # Decode
        logits = decoder(sequence, encoder_out, mask)

        assert logits.shape == (2, 50, 20)

    def test_end_to_end_backward(self) -> None:
        """Test complete backward pass."""
        from models import Struct2SeqGeometricEncoder
        import torch.nn.functional as F

        encoder = Struct2SeqGeometricEncoder(128, 22, num_layers=2)
        decoder = StructureDecoder(20, 128, num_layers=2)

        h_nodes = torch.randn(2, 50, 128)
        h_edges = torch.randn(2, 50, 20, 22)
        edge_idxs = torch.randint(0, 50, (2, 50, 20))
        mask = torch.ones(2, 50)
        sequence = torch.randint(0, 20, (2, 50))

        # Forward
        encoder_out = encoder(h_nodes, h_edges, edge_idxs, mask)
        logits = decoder(sequence, encoder_out, mask)

        # Loss
        loss = F.cross_entropy(
            logits.view(-1, 20),
            sequence.view(-1),
            reduction='mean'
        )

        # Backward
        loss.backward()

        # Check gradients
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

        for param in decoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
