"""
Graph neural network layers for protein modeling using PyTorch Geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import numpy as np


class ProteinGATLayer(MessagePassing):
    """
    Graph Attention Network layer adapted for proteins with edge features.

    This layer performs multi-head attention over k-NN neighbors, using both
    node and edge features. It's similar to GATConv but handles edge features
    in the attention computation.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, num_heads=4, dropout=0.1):
        """
        Args:
            node_dim: dimension of node features
            edge_dim: dimension of edge features
            hidden_dim: output hidden dimension
            num_heads: number of attention heads
            dropout: dropout probability
        """
        super().__init__(aggr='add', node_dim=0)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Attention projections
        self.W_Q = nn.Linear(node_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(node_dim + edge_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(node_dim + edge_dim, hidden_dim, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Args:
            x: [num_nodes, node_dim] node features
            edge_index: [2, num_edges] graph connectivity
            edge_attr: [num_edges, edge_dim] edge features
            batch: [num_nodes] batch assignment (optional)

        Returns:
            x_out: [num_nodes, hidden_dim] updated node features
        """
        # Self-attention with residual connection
        x_res = x if x.size(-1) == self.hidden_dim else self.W_Q(x)
        x_attn = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.norm1(x_res + self.dropout(x_attn))

        # Feed-forward network with residual
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        """
        Compute messages from neighbors.

        Args:
            x_i: [num_edges, node_dim] source node features
            x_j: [num_edges, node_dim] target node features
            edge_attr: [num_edges, edge_dim] edge features
            index: source node indices for each edge
            ptr: for use in aggregation
            size_i: number of target nodes

        Returns:
            messages: [num_edges, hidden_dim] messages to aggregate
        """
        # Queries from source nodes, Keys and Values from edges
        Q = self.W_Q(x_i)  # [num_edges, hidden_dim]

        # Concatenate target node and edge features
        x_j_e = torch.cat([x_j, edge_attr], dim=-1)
        K = self.W_K(x_j_e)  # [num_edges, hidden_dim]
        V = self.W_V(x_j_e)  # [num_edges, hidden_dim]

        # Reshape for multi-head attention
        Q = Q.view(-1, self.num_heads, self.head_dim)
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        attn_logits = (Q * K).sum(dim=-1) / np.sqrt(self.head_dim)  # [num_edges, num_heads]

        # Softmax over neighbors of each node
        attn_weights = softmax(attn_logits, index, ptr, size_i)  # [num_edges, num_heads]

        # Apply attention to values
        attn_weights = attn_weights.unsqueeze(-1)  # [num_edges, num_heads, 1]
        messages = (attn_weights * V).view(-1, self.hidden_dim)  # [num_edges, hidden_dim]

        return messages

    def update(self, aggr_out):
        """
        Update node features after aggregation.

        Args:
            aggr_out: [num_nodes, hidden_dim] aggregated messages

        Returns:
            updated features
        """
        return self.W_O(aggr_out)


class ProteinMPNNLayer(MessagePassing):
    """
    Message Passing Neural Network layer for proteins.

    Uses a 3-layer MLP for computing edge messages, similar to the original
    Struct2Seq MPNN implementation.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1, scale=30):
        """
        Args:
            node_dim: dimension of node features
            edge_dim: dimension of edge features
            hidden_dim: output hidden dimension
            dropout: dropout probability
            scale: scaling factor for message aggregation
        """
        super().__init__(aggr='add', node_dim=0)

        self.hidden_dim = hidden_dim
        self.scale = scale

        # Message network (3-layer MLP)
        self.message_net = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Args:
            x: [num_nodes, node_dim] node features
            edge_index: [2, num_edges] graph connectivity
            edge_attr: [num_edges, edge_dim] edge features
            batch: [num_nodes] batch assignment (optional)

        Returns:
            x_out: [num_nodes, hidden_dim] updated node features
        """
        # Message passing with residual connection
        x_res = x if x.size(-1) == self.hidden_dim else torch.zeros(
            x.size(0), self.hidden_dim, device=x.device
        )
        x_msg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.norm1(x_res + self.dropout(x_msg))

        # Feed-forward network with residual
        x = self.norm2(x + self.dropout(self.ffn(x)))

        return x

    def message(self, x_i, x_j, edge_attr):
        """
        Compute messages from neighbors.

        Args:
            x_i: [num_edges, node_dim] source node features
            x_j: [num_edges, node_dim] target node features
            edge_attr: [num_edges, edge_dim] edge features

        Returns:
            messages: [num_edges, hidden_dim]
        """
        # Concatenate neighbor node features and edge features
        x_j_e = torch.cat([x_j, edge_attr], dim=-1)

        # Compute messages using 3-layer MLP
        messages = self.message_net(x_j_e)

        return messages

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregate messages from neighbors.

        Args:
            inputs: [num_edges, hidden_dim] messages
            index: [num_edges] source indices
            ptr: for efficient aggregation
            dim_size: number of nodes

        Returns:
            aggregated: [num_nodes, hidden_dim]
        """
        # Sum aggregation with scaling
        aggregated = super().aggregate(inputs, index, ptr, dim_size)
        return aggregated / self.scale


class AutoregressiveMask:
    """
    Helper class for creating autoregressive attention masks.

    In autoregressive decoding, position i can only attend to positions j < i.
    """

    @staticmethod
    def create_mask(edge_index, num_nodes):
        """
        Create an autoregressive mask for edges.

        Args:
            edge_index: [2, num_edges]
            num_nodes: number of nodes

        Returns:
            mask: [num_edges] - 1.0 if edge is allowed, 0.0 otherwise
        """
        src, dst = edge_index
        # Only allow edges where dst < src (attending to past)
        mask = (dst < src).float()
        return mask
