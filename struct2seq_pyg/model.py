"""
Struct2Seq model implementation using PyTorch Geometric

A graph-based protein sequence design model that uses structural information
to generate amino acid sequences autoregressively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from .layers import ProteinGATLayer, ProteinMPNNLayer, AutoregressiveMask


class Struct2SeqPyG(nn.Module):
    """
    PyTorch Geometric implementation of Struct2Seq.

    Architecture:
    1. Featurization: Protein structure → graph with node/edge features
    2. Encoder: Graph attention layers (unmasked) process structure
    3. Decoder: Autoregressive graph attention generates sequence
    """

    def __init__(
        self,
        node_feature_dim=6,
        edge_feature_dim=39,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_letters=20,
        num_heads=4,
        dropout=0.1,
        use_mpnn=False,
    ):
        """
        Args:
            node_feature_dim: dimension of input node features (default: 6 for dihedral angles)
            edge_feature_dim: dimension of input edge features (default: 39 for pos_enc + rbf + orientation)
            hidden_dim: hidden dimension for all layers
            num_encoder_layers: number of encoder layers
            num_decoder_layers: number of decoder layers
            num_letters: vocabulary size (20 amino acids)
            num_heads: number of attention heads
            dropout: dropout probability
            use_mpnn: if True, use MPNN layers instead of GAT
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_letters = num_letters
        self.use_mpnn = use_mpnn

        # Input projections
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Sequence embedding for decoder
        self.seq_embedding = nn.Embedding(num_letters, hidden_dim)

        # Choose layer type
        LayerClass = ProteinMPNNLayer if use_mpnn else ProteinGATLayer

        # Encoder: processes structure with unmasked attention
        self.encoder_layers = nn.ModuleList([
            LayerClass(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads if not use_mpnn else None,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])

        # Decoder: generates sequence autoregressively
        self.decoder_layers = nn.ModuleList([
            LayerClass(
                node_dim=hidden_dim,
                edge_dim=hidden_dim * 2,  # structure + sequence embeddings
                hidden_dim=hidden_dim,
                num_heads=num_heads if not use_mpnn else None,
                dropout=dropout
            )
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, num_letters)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, data):
        """
        Encode protein structure with graph neural network.

        Args:
            data: PyG Data object with x, edge_index, edge_attr

        Returns:
            h_V: [num_nodes, hidden_dim] encoded node features
            h_E: [num_edges, hidden_dim] encoded edge features
        """
        # Embed nodes and edges
        h_V = self.node_embedding(data.x)
        h_E = self.edge_embedding(data.edge_attr)

        # Encoder layers
        for layer in self.encoder_layers:
            h_V = layer(h_V, data.edge_index, h_E, data.batch)

        return h_V, h_E

    def decode(self, h_V, h_E, edge_index, seq_embeddings=None, autoregressive_mask=None):
        """
        Decode sequence from structure encoding.

        Args:
            h_V: [num_nodes, hidden_dim] encoded structure
            h_E: [num_edges, hidden_dim] edge features
            edge_index: [2, num_edges] graph connectivity
            seq_embeddings: [num_nodes, hidden_dim] sequence embeddings (optional)
            autoregressive_mask: [num_edges] mask for autoregressive decoding

        Returns:
            h_V_out: [num_nodes, hidden_dim] decoded node features
        """
        # If no sequence embeddings provided, use zeros
        if seq_embeddings is None:
            seq_embeddings = torch.zeros_like(h_V)

        # Concatenate structure and sequence edge features
        # For each edge, we need features from both source and target nodes
        src, dst = edge_index
        h_E_combined = torch.cat([h_E, seq_embeddings[dst]], dim=-1)

        # Apply autoregressive mask if provided
        if autoregressive_mask is not None:
            h_E_combined = h_E_combined * autoregressive_mask.unsqueeze(-1)

        # Decoder layers
        h_V_out = h_V
        for layer in self.decoder_layers:
            h_V_out = layer(h_V_out, edge_index, h_E_combined)

        return h_V_out

    def forward(self, data):
        """
        Forward pass for training.

        Args:
            data: PyG Data/Batch object with:
                - x: node features
                - edge_index: graph connectivity
                - edge_attr: edge features
                - seq: ground truth sequence
                - batch: batch assignment (if batched)

        Returns:
            log_probs: [num_nodes, num_letters] log probabilities
        """
        # Encode structure
        h_V, h_E = self.encode(data)

        # Embed sequence for decoder
        seq_embeddings = self.seq_embedding(data.seq)

        # Decode with teacher forcing
        h_V_decoded = self.decode(h_V, h_E, data.edge_index, seq_embeddings)

        # Project to amino acid vocabulary
        logits = self.output_layer(h_V_decoded)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

    @torch.no_grad()
    def sample(self, data, temperature=1.0):
        """
        Sample sequences autoregressively.

        Args:
            data: PyG Data object with structure (no sequence needed)
            temperature: sampling temperature (higher = more random)

        Returns:
            sequences: [batch_size, max_length] sampled amino acid indices
        """
        device = data.x.device

        # Encode structure
        h_V, h_E = self.encode(data)

        # Determine batch structure
        if hasattr(data, 'batch'):
            batch = data.batch
            batch_size = batch.max().item() + 1
            # Get sequence length for each protein in batch
            lengths = torch.bincount(batch)
        else:
            batch_size = 1
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            lengths = torch.tensor([data.num_nodes], device=device)

        max_length = lengths.max().item()

        # Initialize sequences
        sequences = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        seq_embeddings = torch.zeros_like(h_V)

        # Get cumulative node counts for indexing
        node_offsets = torch.cat([
            torch.tensor([0], device=device),
            torch.cumsum(lengths, dim=0)
        ])

        # Autoregressive sampling
        for t in range(max_length):
            # Create autoregressive mask (only attend to positions < t)
            src, dst = data.edge_index
            # We need to convert global node indices to per-graph positions
            # For simplicity, we'll use a basic mask
            ar_mask = AutoregressiveMask.create_mask(data.edge_index, data.num_nodes)

            # Decode current step
            h_V_decoded = self.decode(h_V, h_E, data.edge_index, seq_embeddings, ar_mask)

            # Get logits for position t in each sequence
            logits = self.output_layer(h_V_decoded) / temperature

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)

            # Update sequences and embeddings for valid positions
            for b in range(batch_size):
                if t < lengths[b]:
                    node_idx = node_offsets[b] + t
                    sequences[b, t] = sampled[node_idx]
                    seq_embeddings[node_idx] = self.seq_embedding(sampled[node_idx])

        return sequences

    def compute_perplexity(self, data):
        """
        Compute perplexity on a batch.

        Args:
            data: PyG Data/Batch with ground truth sequences

        Returns:
            perplexity: scalar tensor
        """
        log_probs = self.forward(data)

        # Gather log probs for true amino acids
        true_log_probs = log_probs.gather(-1, data.seq.unsqueeze(-1)).squeeze(-1)

        # Compute mean negative log likelihood
        nll = -true_log_probs.mean()

        # Perplexity = exp(NLL)
        perplexity = torch.exp(nll)

        return perplexity
