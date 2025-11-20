"""Training script for protein sequence design.

This module provides training infrastructure including:
- Noam learning rate scheduler
- Training loop with teacher forcing
- Metrics: perplexity and sequence recovery
- W&B integration for experiment tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from utils import setup_logger, get_logger, TrainingConfig
from models import (
    # Data
    AA_TO_IDX,
    # Geometry
    get_local_frames,
    get_neighbor_features,
    get_rbf_features,
    compute_dihedral_angles,
    # Models
    Struct2SeqGeometricEncoder,
    StructureDecoder,
)

logger = get_logger(__name__)


class NoamLR:
    """Noam learning rate scheduler with warmup.

    Implements the learning rate schedule from "Attention Is All You Need":
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    Args:
        optimizer: Optimizer to schedule.
        hidden_dim: Model hidden dimension (d_model).
        warmup_steps: Number of warmup steps.
        factor: Scaling factor for learning rate.

    Example:
        >>> optimizer = Adam(model.parameters(), lr=1.0)
        >>> scheduler = NoamLR(optimizer, hidden_dim=128, warmup_steps=4000)
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        hidden_dim: int,
        warmup_steps: int = 4000,
        factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.hidden_dim = hidden_dim
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0
        self._rate = 0

    def step(self) -> None:
        """Update learning rate for next step."""
        self._step += 1
        rate = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate

    def rate(self, step: Optional[int] = None) -> float:
        """Compute learning rate for given step.

        Args:
            step: Step number. If None, uses current step.

        Returns:
            Learning rate value.
        """
        if step is None:
            step = self._step

        # Noam schedule
        step = max(step, 1)  # Avoid division by zero
        rate = self.factor * (
            self.hidden_dim ** (-0.5) *
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
        return rate

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self._rate


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        logits: Model predictions [Batch, Length, vocab_size].
        targets: Ground truth sequences [Batch, Length].
        mask: Valid position mask [Batch, Length].

    Returns:
        Dictionary containing:
            - 'loss': Cross-entropy loss
            - 'perplexity': Perplexity (exp(loss))
            - 'accuracy': Sequence recovery rate

    Shape:
        - logits: [Batch, Length, vocab_size]
        - targets: [Batch, Length]
        - mask: [Batch, Length]
    """
    # Flatten for loss computation
    logits_flat = logits.view(-1, logits.shape[-1])  # [B*L, vocab]
    targets_flat = targets.view(-1)  # [B*L]
    mask_flat = mask.view(-1)  # [B*L]

    # Cross-entropy loss (only on valid positions)
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        reduction='none'
    )
    loss = (loss * mask_flat).sum() / mask_flat.sum()

    # Perplexity
    perplexity = torch.exp(loss)

    # Accuracy (sequence recovery)
    predictions = torch.argmax(logits, dim=-1)  # [B, L]
    correct = (predictions == targets).float() * mask
    accuracy = correct.sum() / mask.sum()

    return {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'accuracy': accuracy.item()
    }


def train_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamLR,
    device: torch.device,
    gradient_clip: float = 1.0,
    log_interval: int = 10
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        encoder: Structure encoder model.
        decoder: Sequence decoder model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        gradient_clip: Maximum gradient norm for clipping.
        log_interval: Log every N batches.

    Returns:
        Dictionary of average metrics for the epoch.

    Example:
        >>> encoder = Struct2SeqGeometricEncoder(128, 22)
        >>> decoder = StructureDecoder(20, 128)
        >>> optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()))
        >>> scheduler = NoamLR(optimizer, 128)
        >>> metrics = train_epoch(encoder, decoder, train_loader, optimizer, scheduler, device)
    """
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        coords = batch['coords'].to(device)
        sequence = batch['sequence'].to(device)
        mask = batch['mask'].to(device)

        # Extract geometric features (Phase 1)
        R, t = get_local_frames(coords)
        rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
            coords, R, t, mask, k_neighbors=30
        )
        rbf = get_rbf_features(distances)
        dihedrals = compute_dihedral_angles(coords)

        # Combine edge features
        edge_features = torch.cat([rel_pos, rel_orient, rbf], dim=-1)

        # Initial node embeddings (from dihedrals for this example)
        # In practice, you might use amino acid type embeddings
        h_nodes = dihedrals  # [B, L, 6]

        # Encode structure (Phase 2)
        encoder_out = encoder(h_nodes, edge_features, neighbor_idx, mask)

        # Decode sequence (Phase 3)
        # Teacher forcing: input is ground truth shifted
        logits = decoder(sequence, encoder_out, mask)

        # Compute loss and metrics
        metrics = compute_metrics(logits, sequence, mask)

        loss = torch.tensor(metrics['loss'], requires_grad=True, device=device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            gradient_clip
        )

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        total_loss += metrics['loss']
        total_perplexity += metrics['perplexity']
        total_accuracy += metrics['accuracy']
        num_batches += 1

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_ppl = total_perplexity / num_batches
            avg_acc = total_accuracy / num_batches
            lr = scheduler.get_lr()

            logger.info(
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {avg_loss:.4f}, PPL: {avg_ppl:.4f}, "
                f"Acc: {avg_acc:.4f}, LR: {lr:.6f}"
            )

    # Return average metrics
    return {
        'loss': total_loss / num_batches,
        'perplexity': total_perplexity / num_batches,
        'accuracy': total_accuracy / num_batches
    }


def evaluate(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation/test set.

    Args:
        encoder: Structure encoder model.
        decoder: Sequence decoder model.
        dataloader: Evaluation data loader.
        device: Device to evaluate on.

    Returns:
        Dictionary of average metrics.

    Example:
        >>> metrics = evaluate(encoder, decoder, val_loader, device)
        >>> print(f"Val perplexity: {metrics['perplexity']:.2f}")
    """
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            coords = batch['coords'].to(device)
            sequence = batch['sequence'].to(device)
            mask = batch['mask'].to(device)

            # Extract geometric features
            R, t = get_local_frames(coords)
            rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
                coords, R, t, mask, k_neighbors=30
            )
            rbf = get_rbf_features(distances)
            dihedrals = compute_dihedral_angles(coords)

            edge_features = torch.cat([rel_pos, rel_orient, rbf], dim=-1)
            h_nodes = dihedrals

            # Encode and decode
            encoder_out = encoder(h_nodes, edge_features, neighbor_idx, mask)
            logits = decoder(sequence, encoder_out, mask)

            # Compute metrics
            metrics = compute_metrics(logits, sequence, mask)

            total_loss += metrics['loss']
            total_perplexity += metrics['perplexity']
            total_accuracy += metrics['accuracy']
            num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'perplexity': total_perplexity / num_batches,
        'accuracy': total_accuracy / num_batches
    }


def train_model(
    config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    checkpoint_dir: Optional[Path] = None
) -> Tuple[nn.Module, nn.Module]:
    """Complete training loop with W&B logging.

    Args:
        config: Training configuration.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        checkpoint_dir: Directory to save checkpoints. If None, uses config.

    Returns:
        Tuple of (trained_encoder, trained_decoder).

    Example:
        >>> from utils import TrainingConfig
        >>> config = TrainingConfig(
        ...     hidden_dim=128,
        ...     num_epochs=100,
        ...     learning_rate=1e-3
        ... )
        >>> encoder, decoder = train_model(config, train_loader, val_loader)
    """
    # Set up logging
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80)
    logger.info(f"Config: {config.to_dict()}")

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize models
    encoder = Struct2SeqGeometricEncoder(
        hidden_dim=config.hidden_dim,
        edge_dim=22,  # 3 + 3 + 16 from Phase 1
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    ).to(device)

    decoder = StructureDecoder(
        vocab_size=20,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout
    ).to(device)

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = encoder_params + decoder_params
    logger.info(f"Model parameters:")
    logger.info(f"  Encoder: {encoder_params:,}")
    logger.info(f"  Decoder: {decoder_params:,}")
    logger.info(f"  Total: {total_params:,}")

    # Initialize optimizer and scheduler
    optimizer = Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1.0,  # Will be controlled by Noam scheduler
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = NoamLR(
        optimizer,
        hidden_dim=config.hidden_dim,
        warmup_steps=config.warmup_steps
    )

    # Initialize W&B if enabled
    if config.wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.experiment_name,
                config=config.to_dict()
            )
            logger.info("W&B initialized successfully")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")
            config.wandb_enabled = False

    # Checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = config.checkpoint_dir
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_perplexity = float('inf')

    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        logger.info("-" * 80)

        # Train
        train_metrics = train_epoch(
            encoder, decoder, train_loader, optimizer, scheduler,
            device, config.gradient_clip
        )

        logger.info(f"Train metrics: {train_metrics}")

        # Validate
        if val_loader is not None:
            val_metrics = evaluate(encoder, decoder, val_loader, device)
            logger.info(f"Val metrics: {val_metrics}")

            # Log to W&B
            if config.wandb_enabled:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/perplexity': train_metrics['perplexity'],
                    'train/accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/perplexity': val_metrics['perplexity'],
                    'val/accuracy': val_metrics['accuracy'],
                    'lr': scheduler.get_lr()
                })

            # Save best model
            if val_metrics['perplexity'] < best_val_perplexity:
                best_val_perplexity = val_metrics['perplexity']
                checkpoint_path = checkpoint_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_perplexity': best_val_perplexity,
                    'config': config.to_dict()
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")

        else:
            # Log to W&B (training only)
            if config.wandb_enabled:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/perplexity': train_metrics['perplexity'],
                    'train/accuracy': train_metrics['accuracy'],
                    'lr': scheduler.get_lr()
                })

        # Periodic checkpoint
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.to_dict()
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)

    return encoder, decoder


if __name__ == '__main__':
    # Example usage
    from utils import parse_training_config

    # Parse configuration
    config = parse_training_config()

    # Setup logger
    logger = setup_logger(
        name='training',
        log_dir=Path('./logs'),
        log_level=logging.INFO
    )

    # Note: You would need to create actual data loaders here
    # This is just a skeleton showing the structure
    logger.info("Training script ready. Create data loaders and call train_model().")
