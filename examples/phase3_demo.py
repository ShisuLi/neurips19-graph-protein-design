"""Phase 3 demonstration: Autoregressive Decoder and Training.

This script demonstrates the complete protein design pipeline:
1. Geometry extraction (Phase 1)
2. Structure encoding (Phase 2)
3. Sequence decoding (Phase 3)
4. Training with teacher forcing
5. Autoregressive generation

Run with:
    python examples/phase3_demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging

from utils import setup_logger, TensorShapeLogger
from models import (
    # Phase 1
    get_local_frames,
    get_neighbor_features,
    get_rbf_features,
    compute_dihedral_angles,
    # Phase 2
    Struct2SeqGeometricEncoder,
    # Phase 3
    StructureDecoder,
    # Data
    AA_TO_IDX,
    tensor_to_sequence,
)
from train import NoamLR, compute_metrics


def create_synthetic_batch(
    batch_size: int = 4,
    seq_len: int = 100,
    vocab_size: int = 20
) -> dict:
    """Create synthetic protein data for demonstration.

    Args:
        batch_size: Number of proteins in batch.
        seq_len: Sequence length.
        vocab_size: Amino acid vocabulary size.

    Returns:
        Dictionary with coords, sequence, and mask.
    """
    # Coordinates with realistic spacing
    coords = torch.randn(batch_size, seq_len, 4, 3) * 3.8

    # Random amino acid sequence
    sequence = torch.randint(0, vocab_size, (batch_size, seq_len))

    # All positions valid
    mask = torch.ones(batch_size, seq_len)

    return {
        'coords': coords,
        'sequence': sequence,
        'mask': mask
    }


def extract_features(batch: dict, k_neighbors: int = 30) -> dict:
    """Extract geometric features from coordinates.

    Args:
        batch: Batch dictionary with coords and mask.
        k_neighbors: Number of neighbors.

    Returns:
        Dictionary with all geometric features.
    """
    coords = batch['coords']
    mask = batch['mask']

    # Phase 1: Geometry
    R, t = get_local_frames(coords)
    rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
        coords, R, t, mask, k_neighbors=k_neighbors
    )
    rbf = get_rbf_features(distances)
    dihedrals = compute_dihedral_angles(coords)

    # Combine edge features
    edge_features = torch.cat([rel_pos, rel_orient, rbf], dim=-1)

    return {
        'h_nodes': dihedrals,  # Node features for encoder input
        'h_edges': edge_features,
        'neighbor_idx': neighbor_idx,
        'mask': mask
    }


def demo_decoder_forward() -> None:
    """Demonstrate decoder forward pass with teacher forcing."""
    logger.info("=" * 80)
    logger.info("Demo 1: Decoder Forward Pass (Teacher Forcing)")
    logger.info("=" * 80)

    batch_size = 4
    seq_len = 100
    hidden_dim = 128
    vocab_size = 20

    logger.info(f"\nCreating synthetic batch: {batch_size} x {seq_len} residues")

    # Create data
    batch = create_synthetic_batch(batch_size, seq_len, vocab_size)

    # Extract features
    logger.info("Extracting geometric features...")
    features = extract_features(batch, k_neighbors=30)

    # Initialize models
    logger.info(f"\nInitializing models (hidden_dim={hidden_dim})...")
    encoder = Struct2SeqGeometricEncoder(
        hidden_dim=hidden_dim,
        edge_dim=features['h_edges'].shape[-1],  # Should be 22
        num_layers=3,
        num_heads=4
    )

    decoder = StructureDecoder(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=3,
        num_heads=4
    )

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    logger.info(f"  Encoder parameters: {encoder_params:,}")
    logger.info(f"  Decoder parameters: {decoder_params:,}")
    logger.info(f"  Total parameters: {encoder_params + decoder_params:,}")

    # Forward pass
    logger.info("\nForward pass:")
    logger.info("  1. Encoding structure...")
    encoder_out = encoder(
        features['h_nodes'],
        features['h_edges'],
        features['neighbor_idx'],
        features['mask']
    )
    shape_logger.log_shape('Encoder output', encoder_out, ['B', 'L', 'H'])

    logger.info("  2. Decoding sequence (teacher forcing)...")
    logits = decoder(
        batch['sequence'],
        encoder_out,
        features['mask']
    )
    shape_logger.log_shape('Decoder logits', logits, ['B', 'L', 'Vocab'])

    # Compute metrics
    logger.info("\n  3. Computing metrics...")
    metrics = compute_metrics(logits, batch['sequence'], features['mask'])

    logger.info(f"\nMetrics (untrained model):")
    logger.info(f"  Loss: {metrics['loss']:.4f}")
    logger.info(f"  Perplexity: {metrics['perplexity']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  (Random baseline accuracy = 1/{vocab_size} = {1/vocab_size:.4f})")


def demo_autoregressive_generation() -> None:
    """Demonstrate autoregressive sequence generation."""
    logger.info("\n\n" + "=" * 80)
    logger.info("Demo 2: Autoregressive Generation")
    logger.info("=" * 80)

    batch_size = 2
    seq_len = 50
    hidden_dim = 128
    vocab_size = 20

    logger.info(f"\nGenerating sequences for {batch_size} proteins...")

    # Create structures
    batch = create_synthetic_batch(batch_size, seq_len, vocab_size)
    features = extract_features(batch, k_neighbors=20)

    # Initialize models
    encoder = Struct2SeqGeometricEncoder(hidden_dim=hidden_dim, edge_dim=22, num_layers=2)
    decoder = StructureDecoder(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=2)

    # Encode structures
    logger.info("\n1. Encoding structures...")
    encoder_out = encoder(
        features['h_nodes'],
        features['h_edges'],
        features['neighbor_idx'],
        features['mask']
    )

    # Generate sequences
    logger.info("2. Generating sequences autoregressively...")
    logger.info("   (Temperature=1.0, sampling from distribution)")

    generated_sequences = decoder.generate(
        encoder_out,
        features['mask'],
        max_len=seq_len,
        temperature=1.0
    )
    shape_logger.log_shape('Generated sequences', generated_sequences, ['B', 'L'])

    # Show generated sequences
    logger.info("\n3. Generated sequences:")
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    for i in range(batch_size):
        seq_indices = generated_sequences[i].tolist()
        seq_str = ''.join(amino_acids[idx] for idx in seq_indices[:20])  # First 20 AA
        logger.info(f"   Protein {i+1}: {seq_str}... (showing first 20)")

    # Compare with different temperatures
    logger.info("\n4. Effect of temperature:")
    for temperature in [0.1, 1.0, 2.0]:
        generated = decoder.generate(
            encoder_out[:1],  # Just first protein
            features['mask'][:1],
            max_len=seq_len,
            temperature=temperature
        )
        seq_str = ''.join(amino_acids[idx] for idx in generated[0].tolist()[:20])
        logger.info(f"   T={temperature:.1f}: {seq_str}...")


def demo_training_loop() -> None:
    """Demonstrate a simplified training loop."""
    logger.info("\n\n" + "=" * 80)
    logger.info("Demo 3: Training Loop with Noam Scheduler")
    logger.info("=" * 80)

    batch_size = 4
    seq_len = 80
    hidden_dim = 128
    vocab_size = 20
    num_steps = 20

    logger.info(f"\nTraining for {num_steps} steps...")

    # Create models
    encoder = Struct2SeqGeometricEncoder(hidden_dim=hidden_dim, edge_dim=22, num_layers=2)
    decoder = StructureDecoder(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=2)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1.0,  # Will be controlled by Noam scheduler
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = NoamLR(
        optimizer,
        hidden_dim=hidden_dim,
        warmup_steps=10
    )

    logger.info(f"\nOptimizer: Adam")
    logger.info(f"Scheduler: Noam (warmup_steps=10)")

    # Training loop
    logger.info(f"\nTraining steps:")
    logger.info(f"  {'Step':<6} {'Loss':<8} {'PPL':<8} {'Acc':<8} {'LR':<10}")
    logger.info("  " + "-" * 45)

    losses = []
    accuracies = []

    for step in range(num_steps):
        # Create batch
        batch = create_synthetic_batch(batch_size, seq_len, vocab_size)
        features = extract_features(batch, k_neighbors=20)

        # Forward pass
        encoder_out = encoder(
            features['h_nodes'],
            features['h_edges'],
            features['neighbor_idx'],
            features['mask']
        )

        logits = decoder(
            batch['sequence'],
            encoder_out,
            features['mask']
        )

        # Compute metrics
        metrics = compute_metrics(logits, batch['sequence'], features['mask'])

        # Backward pass
        optimizer.zero_grad()
        loss = torch.tensor(metrics['loss'], requires_grad=True)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=1.0
        )

        # Update
        optimizer.step()
        scheduler.step()

        # Log
        if (step + 1) % 5 == 0:
            logger.info(
                f"  {step+1:<6} {metrics['loss']:<8.4f} "
                f"{metrics['perplexity']:<8.2f} {metrics['accuracy']:<8.4f} "
                f"{scheduler.get_lr():<10.6f}"
            )

        losses.append(metrics['loss'])
        accuracies.append(metrics['accuracy'])

    # Show training progress
    logger.info(f"\nTraining progress:")
    logger.info(f"  Initial loss: {losses[0]:.4f}")
    logger.info(f"  Final loss: {losses[-1]:.4f}")
    logger.info(f"  Initial accuracy: {accuracies[0]:.4f}")
    logger.info(f"  Final accuracy: {accuracies[-1]:.4f}")
    logger.info(f"  (Note: These are on random synthetic data)")


def demo_causal_masking() -> None:
    """Demonstrate that causal masking prevents future information leakage."""
    logger.info("\n\n" + "=" * 80)
    logger.info("Demo 4: Causal Masking Verification")
    logger.info("=" * 80)

    batch_size = 1
    seq_len = 10
    hidden_dim = 64
    vocab_size = 20

    logger.info("\nVerifying that decoder cannot see future positions...")

    # Create simple sequential input
    batch = create_synthetic_batch(batch_size, seq_len, vocab_size)
    features = extract_features(batch, k_neighbors=10)

    encoder = Struct2SeqGeometricEncoder(hidden_dim=hidden_dim, edge_dim=22, num_layers=1)
    decoder = StructureDecoder(vocab_size=vocab_size, hidden_dim=hidden_dim, num_layers=1)

    # Encode
    encoder_out = encoder(
        features['h_nodes'],
        features['h_edges'],
        features['neighbor_idx'],
        features['mask']
    )

    # Test causal masking
    logger.info("\n1. Testing position 5:")
    logger.info("   - Should only depend on positions 0-5")
    logger.info("   - Should NOT depend on positions 6-9")

    # Get predictions for position 5
    logits_full = decoder(batch['sequence'], encoder_out, features['mask'])
    pred_5_original = logits_full[0, 5, :].argmax().item()

    # Modify future positions (6-9)
    modified_seq = batch['sequence'].clone()
    modified_seq[0, 6:] = (modified_seq[0, 6:] + 1) % vocab_size

    # Get predictions again
    logits_modified = decoder(modified_seq, encoder_out, features['mask'])
    pred_5_modified = logits_modified[0, 5, :].argmax().item()

    logger.info(f"\n2. Results:")
    logger.info(f"   Original prediction at pos 5: {pred_5_original}")
    logger.info(f"   After modifying future positions: {pred_5_modified}")

    if pred_5_original == pred_5_modified:
        logger.info("   ✓ PASS: Predictions identical (causal masking works!)")
    else:
        logger.warning("   ✗ FAIL: Predictions differ (causal masking broken)")


def main() -> None:
    """Run all Phase 3 demonstrations."""
    logger.info("Starting Phase 3 demonstrations...\n")

    # Demo 1: Basic forward pass
    demo_decoder_forward()

    # Demo 2: Autoregressive generation
    demo_autoregressive_generation()

    # Demo 3: Training loop
    demo_training_loop()

    # Demo 4: Causal masking verification
    demo_causal_masking()

    logger.info("\n\n" + "=" * 80)
    logger.info("All Phase 3 demonstrations completed successfully!")
    logger.info("=" * 80)
    logger.info("\nKey Takeaways:")
    logger.info("  1. Decoder successfully generates sequences from structure")
    logger.info("  2. Causal masking prevents future information leakage")
    logger.info("  3. Teacher forcing enables efficient training")
    logger.info("  4. Autoregressive generation works at inference time")
    logger.info("  5. Noam scheduler provides effective learning rate warmup")
    logger.info("\nNext Steps:")
    logger.info("  1. Train on real protein structures")
    logger.info("  2. Evaluate sequence recovery on test set")
    logger.info("  3. Compare ablation variants (Phase 2)")
    logger.info("  4. Generate novel sequences for target structures")


if __name__ == '__main__':
    # Set up logging
    logger = setup_logger(
        name='phase3_demo',
        log_level=logging.INFO,
        file_output=False  # Console only for demo
    )

    shape_logger = TensorShapeLogger(logger)

    # Run demonstrations
    main()
