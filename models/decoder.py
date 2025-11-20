"""Autoregressive decoder for protein sequence generation.

This module implements a transformer-based decoder that generates amino acid
sequences conditioned on structural embeddings from the encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from utils import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence positions.

    Uses sinusoidal functions to encode absolute position information,
    following the original Transformer paper (Vaswani et al., 2017).

    Args:
        hidden_dim: Dimension of the embeddings.
        max_len: Maximum sequence length to support.
        dropout: Dropout probability.

    Example:
        >>> pos_enc = PositionalEncoding(128, max_len=1000)
        >>> x = torch.randn(2, 100, 128)
        >>> x_with_pos = pos_enc(x)
    """

    def __init__(
        self,
        hidden_dim: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, hidden_dim]

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input embeddings with shape [Batch, Length, hidden_dim].

        Returns:
            Input with positional encoding added [Batch, Length, hidden_dim].

        Shape:
            - Input: [Batch, Length, hidden_dim]
            - Output: [Batch, Length, hidden_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer for autoregressive decoding.

    This implements masked self-attention where position i can only attend
    to positions 0...i, preventing the model from "cheating" by looking
    at future positions during training.

    Args:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.

    Example:
        >>> layer = CausalSelfAttention(128, num_heads=4)
        >>> x = torch.randn(2, 100, 128)
        >>> mask = torch.ones(2, 100)
        >>> out = layer(x, mask)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Q, K, V projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply causal self-attention.

        Args:
            x: Input sequence [Batch, Length, hidden_dim].
            mask: Optional padding mask [Batch, Length].

        Returns:
            Attended sequence [Batch, Length, hidden_dim].

        Shape:
            - x: [Batch, Length, hidden_dim]
            - mask: [Batch, Length]
            - output: [Batch, Length, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # [B, L, H]
        K = self.W_k(x)  # [B, L, H]
        V = self.W_v(x)  # [B, L, H]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [B, num_heads, L, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: [B, num_heads, L, L]

        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        # causal_mask: [L, L]

        # Apply causal mask (positions can only attend to past)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, -1e9)

        # Apply padding mask if provided
        if mask is not None:
            # mask: [B, L] -> [B, 1, 1, L]
            padding_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, -1e9)

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # output: [B, num_heads, L, head_dim]

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Final projection
        output = self.W_o(output)

        return output


class CrossAttention(nn.Module):
    """Cross-attention layer for attending to encoder output.

    This allows the decoder to attend to the structural context provided
    by the encoder, enabling the model to condition sequence generation
    on 3D structure.

    Args:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.

    Example:
        >>> layer = CrossAttention(128, num_heads=4)
        >>> decoder_hidden = torch.randn(2, 100, 128)
        >>> encoder_out = torch.randn(2, 100, 128)
        >>> mask = torch.ones(2, 100)
        >>> out = layer(decoder_hidden, encoder_out, mask)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Query from decoder, Key/Value from encoder
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply cross-attention from decoder to encoder.

        Args:
            decoder_hidden: Decoder hidden states [Batch, DecLen, hidden_dim].
            encoder_out: Encoder output [Batch, EncLen, hidden_dim].
            encoder_mask: Encoder padding mask [Batch, EncLen].

        Returns:
            Attended decoder states [Batch, DecLen, hidden_dim].

        Shape:
            - decoder_hidden: [Batch, DecLen, hidden_dim]
            - encoder_out: [Batch, EncLen, hidden_dim]
            - encoder_mask: [Batch, EncLen]
            - output: [Batch, DecLen, hidden_dim]
        """
        batch_size, dec_len, _ = decoder_hidden.shape
        enc_len = encoder_out.shape[1]

        # Project
        Q = self.W_q(decoder_hidden)  # [B, DecLen, H]
        K = self.W_k(encoder_out)     # [B, EncLen, H]
        V = self.W_v(encoder_out)     # [B, EncLen, H]

        # Reshape for multi-head
        Q = Q.view(batch_size, dec_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, enc_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [B, num_heads, L, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores: [B, num_heads, DecLen, EncLen]

        # Apply encoder mask if provided
        if encoder_mask is not None:
            # encoder_mask: [B, EncLen] -> [B, 1, 1, EncLen]
            mask_expanded = encoder_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        output = torch.matmul(attn_weights, V)
        # output: [B, num_heads, DecLen, head_dim]

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, dec_len, self.hidden_dim)

        # Final projection
        output = self.W_o(output)

        return output


class DecoderLayer(nn.Module):
    """Single decoder layer with causal self-attention and cross-attention.

    Args:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward network hidden dimension.
        dropout: Dropout probability.

    Example:
        >>> layer = DecoderLayer(128, num_heads=4, ffn_dim=256)
        >>> dec_h = torch.randn(2, 100, 128)
        >>> enc_out = torch.randn(2, 100, 128)
        >>> out = layer(dec_h, enc_out)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        # Causal self-attention
        self.self_attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Cross-attention to encoder
        self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_out: torch.Tensor,
        decoder_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply decoder layer.

        Args:
            decoder_hidden: Decoder states [Batch, DecLen, hidden_dim].
            encoder_out: Encoder output [Batch, EncLen, hidden_dim].
            decoder_mask: Decoder padding mask [Batch, DecLen].
            encoder_mask: Encoder padding mask [Batch, EncLen].

        Returns:
            Updated decoder states [Batch, DecLen, hidden_dim].
        """
        # Causal self-attention with residual
        h = self.norm1(decoder_hidden + self.self_attn(decoder_hidden, decoder_mask))

        # Cross-attention with residual
        h = self.norm2(h + self.cross_attn(h, encoder_out, encoder_mask))

        # Feed-forward with residual
        h = self.norm3(h + self.ffn(h))

        # Apply decoder mask
        if decoder_mask is not None:
            h = h * decoder_mask.unsqueeze(-1)

        return h


class StructureDecoder(nn.Module):
    """Autoregressive decoder for sequence generation from structure.

    This decoder generates amino acid sequences conditioned on structural
    embeddings from the encoder. It uses:
    - Amino acid embedding
    - Positional encoding
    - Causal self-attention (prevents looking ahead)
    - Cross-attention to encoder output (structure conditioning)
    - Output projection to amino acid vocabulary

    Args:
        vocab_size: Size of amino acid vocabulary (typically 20).
        hidden_dim: Hidden dimension.
        num_layers: Number of decoder layers.
        num_heads: Number of attention heads per layer.
        ffn_dim: Feed-forward network hidden dimension.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for positional encoding.

    Example:
        >>> decoder = StructureDecoder(
        ...     vocab_size=20,
        ...     hidden_dim=128,
        ...     num_layers=3,
        ...     num_heads=4
        ... )
        >>> encoder_out = torch.randn(2, 100, 128)
        >>> sequence = torch.randint(0, 20, (2, 100))
        >>> mask = torch.ones(2, 100)
        >>> logits = decoder(sequence, encoder_out, mask)
        >>> print(logits.shape)  # [2, 100, 20]
    """

    def __init__(
        self,
        vocab_size: int = 20,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 5000
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Amino acid embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"Initialized StructureDecoder: vocab={vocab_size}, "
            f"hidden_dim={hidden_dim}, layers={num_layers}, heads={num_heads}"
        )

    def _init_weights(self) -> None:
        """Initialize weights following best practices."""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=self.hidden_dim ** -0.5)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        sequence: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate sequence logits conditioned on structure.

        Args:
            sequence: Input sequence (teacher forcing) [Batch, Length].
                     Contains amino acid indices (0-19).
            encoder_out: Encoder output (structure) [Batch, Length, hidden_dim].
            encoder_mask: Encoder padding mask [Batch, Length].
            decoder_mask: Decoder padding mask [Batch, Length]. If None,
                         uses encoder_mask.

        Returns:
            Logits for amino acid prediction [Batch, Length, vocab_size].

        Shape:
            - sequence: [Batch, Length]
            - encoder_out: [Batch, Length, hidden_dim]
            - encoder_mask: [Batch, Length]
            - decoder_mask: [Batch, Length]
            - output: [Batch, Length, vocab_size]

        Note:
            During training, sequence is the ground truth (teacher forcing).
            During inference, sequence is generated autoregressively.
        """
        if decoder_mask is None:
            decoder_mask = encoder_mask

        # Embed amino acids
        h = self.embedding(sequence)  # [B, L, H]

        # Add positional encoding
        h = self.pos_encoding(h)

        # Apply decoder layers
        for layer in self.layers:
            h = layer(h, encoder_out, decoder_mask, encoder_mask)

        # Project to vocabulary
        logits = self.output_projection(h)  # [B, L, vocab_size]

        return logits

    def generate(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
        start_token: int = 0
    ) -> torch.Tensor:
        """Generate sequences autoregressively (inference mode).

        Args:
            encoder_out: Encoder output [Batch, Length, hidden_dim].
            encoder_mask: Encoder mask [Batch, Length].
            max_len: Maximum generation length. If None, uses encoder length.
            temperature: Sampling temperature. Lower = more conservative.
            start_token: Token to start generation with.

        Returns:
            Generated sequences [Batch, Length].

        Shape:
            - encoder_out: [Batch, Length, hidden_dim]
            - encoder_mask: [Batch, Length]
            - output: [Batch, Length]

        Example:
            >>> decoder = StructureDecoder(vocab_size=20, hidden_dim=128)
            >>> encoder_out = torch.randn(2, 100, 128)
            >>> mask = torch.ones(2, 100)
            >>> sequences = decoder.generate(encoder_out, mask)
            >>> print(sequences.shape)  # [2, 100]
        """
        batch_size = encoder_out.shape[0]
        if max_len is None:
            max_len = encoder_out.shape[1]

        # Start with start token
        generated = torch.full(
            (batch_size, 1),
            start_token,
            dtype=torch.long,
            device=encoder_out.device
        )

        # Generate autoregressively
        for i in range(max_len - 1):
            # Forward pass with current sequence
            logits = self.forward(
                generated,
                encoder_out,
                encoder_mask,
                decoder_mask=None  # Will use encoder_mask
            )

            # Get logits for last position
            next_logits = logits[:, -1, :] / temperature

            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated
