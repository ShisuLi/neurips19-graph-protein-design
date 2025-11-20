"""PyTorch Geometric encoders for protein structures.

This module reimplements the geometric attention encoder using PyG's message
passing framework for memory-efficient sparse operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Batch
from typing import Optional

from utils import get_logger

logger = get_logger(__name__)


class PyGGeometricAttentionLayer(MessagePassing):
    """Geometric attention layer using PyG message passing.

    This reimplements the GeometricAttentionLayer from Phase 2 using
    PyTorch Geometric's message passing framework for sparse operations.

    The attention mechanism with edge feature injection:
        Q = W_q @ h_i
        K = W_k @ (h_j + edge_features)
        V = W_v @ (h_j + edge_features)
        attention = softmax(Q @ K^T / sqrt(d))
        output = attention @ V

    Args:
        hidden_dim: Hidden dimension for node features.
        edge_dim: Dimension of edge features.
        num_heads: Number of attention heads.
        dropout: Dropout probability.

    Example:
        >>> layer = PyGGeometricAttentionLayer(128, 22, num_heads=4)
        >>> x = torch.randn(100, 128)
        >>> edge_index = torch.randint(0, 100, (2, 300))
        >>> edge_attr = torch.randn(300, 22)
        >>> out = layer(x, edge_index, edge_attr)
        >>> print(out.shape)  # [100, 128]
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        **kwargs
    ):
        # Use 'add' aggregation, we'll handle attention weighting in message()
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.dropout = dropout

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = hidden_dim // num_heads

        # Edge feature projection
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
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Apply geometric attention via message passing.

        Args:
            x: Node features [Num_Nodes, hidden_dim].
            edge_index: Edge connectivity [2, Num_Edges].
            edge_attr: Edge features [Num_Edges, edge_dim].

        Returns:
            Updated node features [Num_Nodes, hidden_dim].

        Shape:
            - x: [Num_Nodes, hidden_dim]
            - edge_index: [2, Num_Edges]
            - edge_attr: [Num_Edges, edge_dim]
            - output: [Num_Nodes, hidden_dim]
        """
        # Project edge features
        edge_attr_proj = self.edge_projection(edge_attr)  # [E, H]

        # Compute queries (for target nodes)
        q = self.W_q(x)  # [N, H]

        # Propagate messages (computes keys, values, attention in message())
        # Note: we pass q in x_i (queries from target nodes)
        out = self.propagate(
            edge_index,
            x=x,
            q=q,
            edge_attr=edge_attr_proj
        )

        # Final output projection
        out = self.W_o(out)

        return out

    def message(
        self,
        x_j: torch.Tensor,
        q_i: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor] = None,
        size_i: Optional[int] = None
    ) -> torch.Tensor:
        """Compute messages with geometric attention.

        This is where the core attention computation happens.

        Args:
            x_j: Source node features [Num_Edges, hidden_dim].
            q_i: Target node queries [Num_Edges, hidden_dim].
            edge_attr: Projected edge features [Num_Edges, hidden_dim].
            index: Target node indices for each edge [Num_Edges].
            ptr: Pointers for segmented softmax (optional).
            size_i: Number of target nodes (optional).

        Returns:
            Attention-weighted messages [Num_Edges, hidden_dim].
        """
        # Inject edge features into source node representations
        x_j_geom = x_j + edge_attr  # [E, H]

        # Compute keys and values from geometrically-enhanced neighbors
        k = self.W_k(x_j_geom)  # [E, H]
        v = self.W_v(x_j_geom)  # [E, H]

        # Reshape for multi-head attention
        # [E, H] -> [E, num_heads, head_dim]
        q_i = q_i.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        # Compute attention scores
        # [E, num_heads, head_dim] x [E, num_heads, head_dim] -> [E, num_heads]
        scores = (q_i * k).sum(dim=-1) * self.scale

        # Softmax over edges for each target node
        # This uses PyG's built-in softmax which handles the grouping by index
        alpha = self.softmax(scores, index, ptr, size_i)  # [E, num_heads]
        alpha = self.dropout_layer(alpha)

        # Apply attention to values
        # [E, num_heads, 1] * [E, num_heads, head_dim] -> [E, num_heads, head_dim]
        out = alpha.unsqueeze(-1) * v

        # Reshape back to [E, H]
        out = out.view(-1, self.hidden_dim)

        return out

    def softmax(
        self,
        src: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None
    ) -> torch.Tensor:
        """Compute softmax over edges grouped by target node.

        Args:
            src: Attention scores [Num_Edges, num_heads].
            index: Target node indices [Num_Edges].
            ptr: Pointers for segmented operations (optional).
            num_nodes: Number of nodes (optional).

        Returns:
            Normalized attention weights [Num_Edges, num_heads].
        """
        # Use PyG's softmax for efficient grouped softmax
        from torch_geometric.utils import softmax as pyg_softmax

        # Apply softmax separately for each head
        out = []
        for head_idx in range(self.num_heads):
            head_scores = src[:, head_idx]  # [E]
            head_alpha = pyg_softmax(head_scores, index, ptr, num_nodes)
            out.append(head_alpha)

        out = torch.stack(out, dim=1)  # [E, num_heads]

        return out


class PyGStruct2SeqEncoder(nn.Module):
    """PyG-based Struct2Seq geometric encoder.

    This reimplements the Struct2SeqGeometricEncoder using PyTorch Geometric
    for memory-efficient sparse operations on long proteins.

    Args:
        hidden_dim: Hidden dimension for node representations.
        edge_dim: Dimension of edge features.
        num_layers: Number of geometric attention layers.
        num_heads: Number of attention heads per layer.
        dropout: Dropout probability.
        ffn_dim: Feed-forward network hidden dimension. If None, uses 2*hidden_dim.

    Example:
        >>> encoder = PyGStruct2SeqEncoder(128, 22, num_layers=3, num_heads=4)
        >>> x = torch.randn(100, 128)
        >>> edge_index = torch.randint(0, 100, (2, 300))
        >>> edge_attr = torch.randn(300, 22)
        >>> out = encoder(x, edge_index, edge_attr)
        >>> print(out.shape)  # [100, 128]
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
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim if ffn_dim is not None else 2 * hidden_dim

        # Input projection (from node features to hidden_dim)
        # Assumes input is dihedral angles (6 dims)
        self.input_proj = nn.Linear(6, hidden_dim)

        # Build layers
        self.attention_layers = nn.ModuleList([
            PyGGeometricAttentionLayer(hidden_dim, edge_dim, num_heads, dropout)
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
            f"Initialized PyGStruct2SeqEncoder: "
            f"{num_layers} layers, {num_heads} heads, "
            f"hidden_dim={hidden_dim}, edge_dim={edge_dim}"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode protein structure with geometric attention.

        Args:
            x: Node features [Num_Nodes, node_dim].
               For protein design, typically dihedral angles (6 dims).
            edge_index: Edge connectivity [2, Num_Edges].
            edge_attr: Edge features [Num_Edges, edge_dim].
                      Should contain geometric features from Phase 1:
                      - Relative positions (3)
                      - Relative orientations (3)
                      - RBF distance encoding (16)
                      Total: 22 dimensions
            batch: Batch assignment [Num_Nodes]. Optional, for batched graphs.

        Returns:
            Encoded node representations [Num_Nodes, hidden_dim].

        Shape:
            - x: [Num_Nodes, node_dim]
            - edge_index: [2, Num_Edges]
            - edge_attr: [Num_Edges, edge_dim]
            - batch: [Num_Nodes]
            - output: [Num_Nodes, hidden_dim]
        """
        # Project input features to hidden dimension
        if x.shape[1] != self.hidden_dim:
            h = self.input_proj(x)
        else:
            h = x

        # Apply geometric attention layers with residual connections
        for i in range(self.num_layers):
            # Geometric attention with residual
            h_attn = self.attention_layers[i](h, edge_index, edge_attr)
            h = self.norm1_layers[i](h + h_attn)

            # Feed-forward with residual
            h_ffn = self.ffn_layers[i](h)
            h = self.norm2_layers[i](h + h_ffn)

        logger.debug(f"Encoded structure with PyG: {h.shape}")

        return h

    def forward_batch(
        self,
        batch: Batch
    ) -> torch.Tensor:
        """Convenience method for PyG Batch objects.

        Args:
            batch: PyG Batch containing x, edge_index, edge_attr, batch.

        Returns:
            Encoded node representations [Num_Nodes, hidden_dim].
        """
        return self.forward(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )


class PyGToD

enseAdapter(nn.Module):
    """Adapter to convert PyG encoder output to dense format.

    This allows using PyG encoders with dense decoders that expect
    [Batch, Length, hidden_dim] format.

    Args:
        pyg_encoder: PyG encoder module.
        max_length: Maximum sequence length for padding.

    Example:
        >>> pyg_enc = PyGStruct2SeqEncoder(128, 22)
        >>> adapter = PyGToDenseAdapter(pyg_enc, max_length=500)
        >>> batch = ...  # PyG Batch
        >>> dense_out, mask = adapter(batch)
        >>> print(dense_out.shape)  # [Batch, MaxLength, 128]
    """

    def __init__(
        self,
        pyg_encoder: nn.Module,
        max_length: int = 500
    ):
        super().__init__()
        self.pyg_encoder = pyg_encoder
        self.max_length = max_length

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode with PyG and convert to dense format.

        Args:
            batch: PyG Batch object.

        Returns:
            Tuple of (dense_output, mask):
                - dense_output: [Batch, MaxLength, hidden_dim]
                - mask: [Batch, MaxLength]
        """
        # Encode with PyG
        node_embeddings = self.pyg_encoder.forward_batch(batch)

        # Convert to dense
        from .data_pyg import pyg_to_dense

        # Add dummy sequence (not used, just for conversion)
        batch.y = torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.x.device)

        # Temporarily replace x with embeddings
        original_x = batch.x
        batch.x = node_embeddings

        dense_out, _, mask = pyg_to_dense(batch, self.max_length)

        # Restore original x
        batch.x = original_x

        return dense_out, mask
