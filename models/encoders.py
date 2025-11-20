"""Graph Neural Network encoders for protein structures.

This module implements a suite of interchangeable GNN encoders for ablation studies:
- VanillaGCNEncoder: Baseline without geometric features
- EdgeAwareGATEncoder: Attention with distance features
- Struct2SeqGeometricEncoder: Full geometric attention with local frames

All encoders follow the same interface for easy swapping in experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from abc import ABC, abstractmethod

from utils import get_logger

logger = get_logger(__name__)


class BaseEncoder(ABC, nn.Module):
    """Abstract base class for protein structure encoders.

    All encoders must implement the forward method with standardized inputs
    to enable easy ablation studies and architecture comparison.

    Args:
        hidden_dim: Hidden dimension for node representations.
        edge_dim: Dimension of edge features.
        num_layers: Number of encoder layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout

    @abstractmethod
    def forward(
        self,
        h_nodes: torch.Tensor,
        h_edges: torch.Tensor,
        edge_idxs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode protein structure.

        Args:
            h_nodes: Node features with shape [Batch, Length, hidden_dim].
            h_edges: Edge features with shape [Batch, Length, K, edge_dim].
            edge_idxs: Neighbor indices with shape [Batch, Length, K].
            mask: Valid residue mask with shape [Batch, Length].

        Returns:
            Encoded node representations with shape [Batch, Length, hidden_dim].

        Shape:
            - h_nodes: [Batch, Length, hidden_dim]
            - h_edges: [Batch, Length, K, edge_dim]
            - edge_idxs: [Batch, Length, K]
            - mask: [Batch, Length]
            - output: [Batch, Length, hidden_dim]
        """
        pass


class GeometricAttentionLayer(nn.Module):
    """Custom geometric attention layer with edge feature injection.

    This implements the core geometric attention mechanism from Struct2Seq,
    where edge features (relative positions, orientations) are injected
    into the key and value computations.

    The attention mechanism:
        Q = W_q @ h_i
        K = W_k @ (h_j + edge_features)
        V = W_v @ (h_j + edge_features)
        attention = softmax(Q @ K^T / sqrt(d))
        output = attention @ V

    Args:
        hidden_dim: Dimension of node features.
        edge_dim: Dimension of edge features.
        num_heads: Number of attention heads.
        dropout: Dropout probability.

    Example:
        >>> layer = GeometricAttentionLayer(128, 64, num_heads=4)
        >>> h = torch.randn(2, 100, 128)
        >>> e = torch.randn(2, 100, 30, 64)
        >>> idx = torch.randint(0, 100, (2, 100, 30))
        >>> mask = torch.ones(2, 100)
        >>> out = layer(h, e, idx, mask)
        >>> print(out.shape)  # [2, 100, 128]
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.dropout = dropout

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = hidden_dim // num_heads

        # Project edge features to hidden dimension for injection
        self.edge_projection = nn.Linear(edge_dim, hidden_dim)

        # Q, K, V projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout_layer = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        h_nodes: torch.Tensor,
        h_edges: torch.Tensor,
        edge_idxs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply geometric attention.

        Args:
            h_nodes: Node features [Batch, Length, hidden_dim].
            h_edges: Edge features [Batch, Length, K, edge_dim].
            edge_idxs: Neighbor indices [Batch, Length, K].
            mask: Valid residue mask [Batch, Length].

        Returns:
            Attended node features [Batch, Length, hidden_dim].

        Shape:
            - h_nodes: [Batch, Length, hidden_dim]
            - h_edges: [Batch, Length, K, edge_dim]
            - edge_idxs: [Batch, Length, K]
            - mask: [Batch, Length]
            - output: [Batch, Length, hidden_dim]
        """
        batch_size, seq_len, _ = h_nodes.shape
        k_neighbors = edge_idxs.shape[2]

        # Project edge features to hidden dimension
        # Shape: [Batch, Length, K, hidden_dim]
        h_edges_proj = self.edge_projection(h_edges)

        # Gather neighbor node features
        # edge_idxs: [Batch, Length, K]
        # Expand for gathering: [Batch, Length, K, hidden_dim]
        idx_expand = edge_idxs.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        h_neighbors = torch.gather(
            h_nodes.unsqueeze(2).expand(-1, -1, seq_len, -1),
            dim=2,
            index=idx_expand
        )  # [Batch, Length, K, hidden_dim]

        # Inject edge features into neighbor representations
        # This is the key innovation: K and V include geometric information
        h_neighbors_geometric = h_neighbors + h_edges_proj

        # Compute Q, K, V
        # Q from central node only
        Q = self.W_q(h_nodes)  # [Batch, Length, hidden_dim]
        # K and V from neighbors + edge features
        K = self.W_k(h_neighbors_geometric)  # [Batch, Length, K, hidden_dim]
        V = self.W_v(h_neighbors_geometric)  # [Batch, Length, K, hidden_dim]

        # Reshape for multi-head attention
        # [Batch, Length, num_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # [Batch, Length, K, num_heads, head_dim]
        K = K.view(batch_size, seq_len, k_neighbors, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, k_neighbors, self.num_heads, self.head_dim)

        # Transpose for attention computation
        # Q: [Batch, num_heads, Length, head_dim]
        Q = Q.transpose(1, 2)
        # K, V: [Batch, num_heads, Length, K, head_dim]
        K = K.permute(0, 3, 1, 2, 4)
        V = V.permute(0, 3, 1, 2, 4)

        # Compute attention scores
        # Q: [Batch, num_heads, Length, 1, head_dim]
        # K: [Batch, num_heads, Length, K, head_dim]
        # scores: [Batch, num_heads, Length, K]
        scores = torch.matmul(Q.unsqueeze(3), K.transpose(-1, -2)).squeeze(3)
        scores = scores * self.scale

        # Create attention mask
        # mask: [Batch, Length] -> [Batch, 1, Length, 1]
        mask_2d = mask.unsqueeze(1).unsqueeze(-1)  # [Batch, 1, Length, 1]

        # Gather neighbor masks
        # [Batch, 1, Length, K]
        neighbor_mask = torch.gather(
            mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len, -1),
            dim=2,
            index=edge_idxs.unsqueeze(1).expand(-1, 1, -1, -1)
        )

        # Combined mask: central node valid AND neighbor valid
        # [Batch, 1, Length, K]
        attn_mask = mask_2d * neighbor_mask

        # Apply mask to attention scores (set invalid to large negative)
        scores = scores.masked_fill(attn_mask == 0, -1e9)

        # Softmax over neighbors
        attn_weights = F.softmax(scores, dim=-1)  # [Batch, num_heads, Length, K]
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        # attn_weights: [Batch, num_heads, Length, K, 1]
        # V: [Batch, num_heads, Length, K, head_dim]
        # output: [Batch, num_heads, Length, head_dim]
        output = torch.matmul(attn_weights.unsqueeze(-2), V).squeeze(-2)

        # Reshape back to [Batch, Length, hidden_dim]
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)

        # Final projection
        output = self.W_o(output)

        return output


class Struct2SeqGeometricEncoder(BaseEncoder):
    """Full geometric encoder with local frame features.

    This implements the complete geometric attention mechanism from
    Ingraham et al. (2019), using:
    - Relative positions in local frames
    - Relative orientations (quaternions)
    - Distance-based RBF features

    The encoder consists of multiple geometric attention layers with
    residual connections and layer normalization.

    Args:
        hidden_dim: Hidden dimension for node representations.
        edge_dim: Dimension of edge features (should include position,
                  orientation, and RBF features).
        num_layers: Number of geometric attention layers.
        num_heads: Number of attention heads per layer.
        dropout: Dropout probability.
        ffn_dim: Feed-forward network hidden dimension. If None, uses 2*hidden_dim.

    Example:
        >>> encoder = Struct2SeqGeometricEncoder(
        ...     hidden_dim=128,
        ...     edge_dim=64,
        ...     num_layers=3,
        ...     num_heads=4
        ... )
        >>> h = torch.randn(2, 100, 128)
        >>> e = torch.randn(2, 100, 30, 64)
        >>> idx = torch.randint(0, 100, (2, 100, 30))
        >>> mask = torch.ones(2, 100)
        >>> out = encoder(h, e, idx, mask)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_dim: Optional[int] = None
    ):
        super().__init__(hidden_dim, edge_dim, num_layers, dropout)

        self.num_heads = num_heads
        self.ffn_dim = ffn_dim if ffn_dim is not None else 2 * hidden_dim

        # Build layers
        self.attention_layers = nn.ModuleList([
            GeometricAttentionLayer(hidden_dim, edge_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, self.ffn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.ffn_dim, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        logger.info(
            f"Initialized Struct2SeqGeometricEncoder: "
            f"{num_layers} layers, {num_heads} heads, "
            f"hidden_dim={hidden_dim}, edge_dim={edge_dim}"
        )

    def forward(
        self,
        h_nodes: torch.Tensor,
        h_edges: torch.Tensor,
        edge_idxs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode protein structure with geometric attention.

        Args:
            h_nodes: Node features [Batch, Length, hidden_dim].
            h_edges: Edge features [Batch, Length, K, edge_dim].
                     Should contain geometric features from Phase 1:
                     - Relative positions (normalized)
                     - Relative orientations
                     - RBF distance encoding
            edge_idxs: Neighbor indices [Batch, Length, K].
            mask: Valid residue mask [Batch, Length].

        Returns:
            Encoded node representations [Batch, Length, hidden_dim].

        Shape:
            - h_nodes: [Batch, Length, hidden_dim]
            - h_edges: [Batch, Length, K, edge_dim]
            - edge_idxs: [Batch, Length, K]
            - mask: [Batch, Length]
            - output: [Batch, Length, hidden_dim]
        """
        h = h_nodes

        # Apply geometric attention layers with residual connections
        for i in range(self.num_layers):
            # Geometric attention with residual
            h_attn = self.attention_layers[i](h, h_edges, edge_idxs, mask)
            h = self.norm1_layers[i](h + h_attn)

            # Feed-forward with residual
            h_ffn = self.ffn_layers[i](h)
            h = self.norm2_layers[i](h + h_ffn)

            # Apply mask to ensure invalid positions stay zero
            h = h * mask.unsqueeze(-1)

        logger.debug(f"Encoded structure with shape: {h.shape}")

        return h


class EdgeAwareGATEncoder(BaseEncoder):
    """Graph Attention Network with edge-aware attention.

    This is an intermediate encoder that uses distance-based edge features
    to modulate attention, but doesn't use full geometric information
    (relative positions/orientations).

    The attention mechanism includes edge features as a bias term:
        attention_score = (Q @ K^T) + edge_bias

    Args:
        hidden_dim: Hidden dimension for node representations.
        edge_dim: Dimension of edge features.
        num_layers: Number of attention layers.
        num_heads: Number of attention heads per layer.
        dropout: Dropout probability.

    Example:
        >>> encoder = EdgeAwareGATEncoder(128, 16, num_layers=3)
        >>> h = torch.randn(2, 100, 128)
        >>> e = torch.randn(2, 100, 30, 16)  # Only RBF features
        >>> idx = torch.randint(0, 100, (2, 100, 30))
        >>> mask = torch.ones(2, 100)
        >>> out = encoder(h, e, idx, mask)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__(hidden_dim, edge_dim, num_layers, dropout)

        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        # Build layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                EdgeAwareGATLayer(hidden_dim, edge_dim, num_heads, dropout)
            )

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        logger.info(
            f"Initialized EdgeAwareGATEncoder: "
            f"{num_layers} layers, {num_heads} heads"
        )

    def forward(
        self,
        h_nodes: torch.Tensor,
        h_edges: torch.Tensor,
        edge_idxs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode with edge-aware attention.

        Args:
            h_nodes: Node features [Batch, Length, hidden_dim].
            h_edges: Edge features [Batch, Length, K, edge_dim].
                     Typically RBF-encoded distances.
            edge_idxs: Neighbor indices [Batch, Length, K].
            mask: Valid residue mask [Batch, Length].

        Returns:
            Encoded node representations [Batch, Length, hidden_dim].
        """
        h = h_nodes

        for i, layer in enumerate(self.layers):
            h_new = layer(h, h_edges, edge_idxs, mask)
            h = self.norm_layers[i](h + h_new)
            h = h * mask.unsqueeze(-1)

        return h


class EdgeAwareGATLayer(nn.Module):
    """Single edge-aware GAT layer.

    Args:
        hidden_dim: Hidden dimension.
        edge_dim: Edge feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Q, K, V projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Edge bias projection
        self.edge_bias = nn.Linear(edge_dim, num_heads)

        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        h_nodes: torch.Tensor,
        h_edges: torch.Tensor,
        edge_idxs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply edge-aware attention."""
        batch_size, seq_len, _ = h_nodes.shape
        k_neighbors = edge_idxs.shape[2]

        # Gather neighbor features
        idx_expand = edge_idxs.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        h_neighbors = torch.gather(
            h_nodes.unsqueeze(2).expand(-1, -1, seq_len, -1),
            dim=2,
            index=idx_expand
        )

        # Q, K, V
        Q = self.W_q(h_nodes)  # [B, L, H]
        K = self.W_k(h_neighbors)  # [B, L, K, H]
        V = self.W_v(h_neighbors)  # [B, L, K, H]

        # Compute edge bias from edge features
        edge_bias = self.edge_bias(h_edges)  # [B, L, K, num_heads]

        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, k_neighbors, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = V.view(batch_size, seq_len, k_neighbors, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # Attention scores
        scores = torch.matmul(Q.unsqueeze(3), K.transpose(-1, -2)).squeeze(3)
        scores = scores * self.scale

        # Add edge bias
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # [B, num_heads, L, K]
        scores = scores + edge_bias

        # Mask
        mask_2d = mask.unsqueeze(1).unsqueeze(-1)
        neighbor_mask = torch.gather(
            mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len, -1),
            dim=2,
            index=edge_idxs.unsqueeze(1).expand(-1, 1, -1, -1)
        )
        attn_mask = mask_2d * neighbor_mask
        scores = scores.masked_fill(attn_mask == 0, -1e9)

        # Attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        output = torch.matmul(attn_weights.unsqueeze(-2), V).squeeze(-2)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.W_o(output)

        return output


class VanillaGCNEncoder(BaseEncoder):
    """Baseline GCN encoder without geometric features.

    This encoder ignores edge features and treats the protein as a simple
    connectivity graph. It provides a lower-bound baseline for comparison.

    The aggregation is a simple mean over neighbors:
        h_i' = MLP(h_i + mean(h_j for j in neighbors(i)))

    Args:
        hidden_dim: Hidden dimension for node representations.
        edge_dim: Dimension of edge features (ignored).
        num_layers: Number of GCN layers.
        dropout: Dropout probability.

    Example:
        >>> encoder = VanillaGCNEncoder(128, 64, num_layers=3)
        >>> h = torch.randn(2, 100, 128)
        >>> e = torch.randn(2, 100, 30, 64)  # Ignored
        >>> idx = torch.randint(0, 100, (2, 100, 30))
        >>> mask = torch.ones(2, 100)
        >>> out = encoder(h, e, idx, mask)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,  # Not used, but kept for interface consistency
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__(hidden_dim, edge_dim, num_layers, dropout)

        # Build layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        logger.info(
            f"Initialized VanillaGCNEncoder (baseline): {num_layers} layers"
        )

    def forward(
        self,
        h_nodes: torch.Tensor,
        h_edges: torch.Tensor,  # Ignored
        edge_idxs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode with simple graph convolution.

        Args:
            h_nodes: Node features [Batch, Length, hidden_dim].
            h_edges: Edge features [Batch, Length, K, edge_dim]. IGNORED.
            edge_idxs: Neighbor indices [Batch, Length, K].
            mask: Valid residue mask [Batch, Length].

        Returns:
            Encoded node representations [Batch, Length, hidden_dim].
        """
        batch_size, seq_len, _ = h_nodes.shape

        h = h_nodes

        for i in range(self.num_layers):
            # Gather neighbor features
            idx_expand = edge_idxs.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
            h_neighbors = torch.gather(
                h.unsqueeze(2).expand(-1, -1, seq_len, -1),
                dim=2,
                index=idx_expand
            )  # [B, L, K, H]

            # Simple mean aggregation (ignoring edge features)
            h_aggregated = h_neighbors.mean(dim=2)  # [B, L, H]

            # MLP with residual
            h_new = self.layers[i](h + h_aggregated)
            h = self.norm_layers[i](h_new)

            # Apply mask
            h = h * mask.unsqueeze(-1)

        logger.debug("Encoded with VanillaGCN (no geometric features)")

        return h


class StructureProbingHead(nn.Module):
    """Probing head for validating encoder representations.

    This simple classification head allows us to verify that encoders
    are learning meaningful representations before building the full
    decoder architecture.

    The head performs global pooling over the sequence and predicts
    a simple property (e.g., sequence length category).

    Args:
        hidden_dim: Dimension of input node features.
        num_classes: Number of output classes.
        pooling: Pooling method ('mean', 'max', or 'sum').
        dropout: Dropout probability.

    Example:
        >>> encoder = Struct2SeqGeometricEncoder(128, 64)
        >>> head = StructureProbingHead(128, num_classes=2)
        >>> h = torch.randn(2, 100, 128)
        >>> e = torch.randn(2, 100, 30, 64)
        >>> idx = torch.randint(0, 100, (2, 100, 30))
        >>> mask = torch.ones(2, 100)
        >>> encoded = encoder(h, e, idx, mask)
        >>> logits = head(encoded, mask)
        >>> print(logits.shape)  # [2, 2]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 2,
        pooling: str = 'mean',
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.pooling = pooling

        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        logger.info(
            f"Initialized StructureProbingHead: "
            f"pooling={pooling}, num_classes={num_classes}"
        )

    def forward(
        self,
        h_nodes: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Classify protein based on encoded representations.

        Args:
            h_nodes: Encoded node features [Batch, Length, hidden_dim].
            mask: Valid residue mask [Batch, Length].

        Returns:
            Class logits [Batch, num_classes].

        Shape:
            - h_nodes: [Batch, Length, hidden_dim]
            - mask: [Batch, Length]
            - output: [Batch, num_classes]
        """
        # Global pooling over sequence
        if self.pooling == 'mean':
            # Masked mean pooling
            masked_sum = (h_nodes * mask.unsqueeze(-1)).sum(dim=1)
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = masked_sum / lengths
        elif self.pooling == 'max':
            # Masked max pooling
            masked_nodes = h_nodes.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            pooled = masked_nodes.max(dim=1)[0]
        elif self.pooling == 'sum':
            # Masked sum pooling
            pooled = (h_nodes * mask.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Classify
        logits = self.classifier(pooled)

        return logits
