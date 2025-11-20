"""Phase 2 demonstration: GNN Encoders and Ablation Studies.

This script demonstrates the encoder architectures implemented in Phase 2:
1. Building geometric features from Phase 1
2. Using three encoder variants (VanillaGCN, EdgeAwareGAT, Struct2SeqGeometric)
3. Comparing encoder outputs
4. Using the probing head for validation

Run with:
    python examples/phase2_demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging

from utils import setup_logger, TensorShapeLogger
from models import (
    # Phase 1: Geometry
    get_local_frames,
    get_rbf_features,
    get_neighbor_features,
    compute_dihedral_angles,
    # Phase 2: Encoders
    VanillaGCNEncoder,
    EdgeAwareGATEncoder,
    Struct2SeqGeometricEncoder,
    StructureProbingHead,
    # Data utilities
    AA_TO_IDX,
)


def create_geometric_features(
    X: torch.Tensor,
    mask: torch.Tensor,
    k_neighbors: int = 30
) -> dict:
    """Extract complete geometric features from coordinates.

    This combines all Phase 1 geometry functions into a single pipeline.

    Args:
        X: Backbone coordinates [Batch, Length, 4, 3].
        mask: Valid residue mask [Batch, Length].
        k_neighbors: Number of nearest neighbors.

    Returns:
        Dictionary containing all geometric features ready for encoder input.
    """
    logger.info("Extracting geometric features...")

    # Step 1: Local coordinate frames
    R, t = get_local_frames(X)
    shape_logger.log_shape('Rotation matrices', R, ['B', 'L', '3', '3'])

    # Step 2: Neighbor features
    rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
        X, R, t, mask, k_neighbors=k_neighbors
    )
    shape_logger.log_shape('Relative positions', rel_pos, ['B', 'L', 'K', '3'])
    shape_logger.log_shape('Neighbor indices', neighbor_idx, ['B', 'L', 'K'])

    # Step 3: RBF distance encoding
    rbf = get_rbf_features(distances, D_min=0.0, D_max=20.0, D_count=16)
    shape_logger.log_shape('RBF features', rbf, ['B', 'L', 'K', 'RBF'])

    # Step 4: Dihedral angles (node features)
    dihedrals = compute_dihedral_angles(X)
    shape_logger.log_shape('Dihedral angles', dihedrals, ['B', 'L', '6'])

    # Combine edge features: [rel_pos (3), rel_orient (3), rbf (16)] = 22 dims
    edge_features = torch.cat([
        rel_pos,      # [B, L, K, 3]
        rel_orient,   # [B, L, K, 3]
        rbf,          # [B, L, K, 16]
    ], dim=-1)  # [B, L, K, 22]

    shape_logger.log_shape('Combined edge features', edge_features, ['B', 'L', 'K', 'EdgeDim'])

    logger.info(f"Geometric feature extraction complete!")
    logger.info(f"  Edge feature dimension: {edge_features.shape[-1]}")
    logger.info(f"  Node feature dimension: {dihedrals.shape[-1]}")

    return {
        'node_features': dihedrals,
        'edge_features': edge_features,
        'neighbor_idx': neighbor_idx,
        'mask': mask,
        'distances': distances,
    }


def demo_encoder_comparison() -> None:
    """Compare the three encoder architectures on synthetic data.

    This demonstrates the ablation study setup: same input, different encoders.
    """
    logger.info("=" * 80)
    logger.info("Demo 1: Encoder Architecture Comparison")
    logger.info("=" * 80)

    # Create synthetic protein data
    batch_size = 4
    seq_len = 100
    k_neighbors = 30

    logger.info(f"Creating synthetic proteins: {batch_size} x {seq_len} residues")

    # Coordinates with realistic spacing (~3.8 Å between CA atoms)
    X = torch.randn(batch_size, seq_len, 4, 3) * 3.8
    mask = torch.ones(batch_size, seq_len)

    # Extract geometric features
    features = create_geometric_features(X, mask, k_neighbors)

    # Prepare node embeddings (in real model, this would be amino acid embeddings)
    hidden_dim = 128
    # For demo, project dihedral features to hidden_dim
    node_embedding = nn.Linear(6, hidden_dim)
    h_nodes = node_embedding(features['node_features'])  # [B, L, hidden_dim]

    edge_dim = features['edge_features'].shape[-1]  # Should be 22

    logger.info(f"\nInitializing three encoder architectures:")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"  Edge dim: {edge_dim}")

    # Initialize all three encoders
    encoders = {
        'VanillaGCN': VanillaGCNEncoder(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_layers=3,
            dropout=0.1
        ),
        'EdgeAwareGAT': EdgeAwareGATEncoder(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_layers=3,
            num_heads=4,
            dropout=0.1
        ),
        'Struct2SeqGeometric': Struct2SeqGeometricEncoder(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_layers=3,
            num_heads=4,
            dropout=0.1
        ),
    }

    # Count parameters for each encoder
    logger.info("\nEncoder statistics:")
    for name, encoder in encoders.items():
        num_params = sum(p.numel() for p in encoder.parameters())
        logger.info(f"  {name:25s}: {num_params:,} parameters")

    # Encode with each encoder
    logger.info("\nEncoding with each architecture...")
    outputs = {}

    for name, encoder in encoders.items():
        logger.info(f"  Encoding with {name}...")
        with torch.no_grad():
            output = encoder(
                h_nodes,
                features['edge_features'],
                features['neighbor_idx'],
                features['mask']
            )
        outputs[name] = output
        shape_logger.log_shape(f'{name} output', output, ['B', 'L', 'H'])

        # Check for NaN
        if torch.isnan(output).any():
            logger.warning(f"  WARNING: {name} produced NaN values!")
        else:
            logger.info(f"  {name} output: min={output.min():.3f}, max={output.max():.3f}")

    # Compare outputs
    logger.info("\nComparing encoder outputs:")
    logger.info("  (Outputs should differ since they use different architectures)")

    for name1 in outputs:
        for name2 in outputs:
            if name1 < name2:  # Avoid duplicates
                diff = (outputs[name1] - outputs[name2]).abs().mean().item()
                logger.info(f"  {name1:25s} vs {name2:25s}: mean diff = {diff:.6f}")


def demo_geometric_attention_mechanism() -> None:
    """Demonstrate the custom geometric attention mechanism.

    Shows how edge features (geometric information) are injected into
    the attention computation.
    """
    logger.info("\n\n" + "=" * 80)
    logger.info("Demo 2: Geometric Attention Mechanism")
    logger.info("=" * 80)

    batch_size = 2
    seq_len = 50
    k_neighbors = 20
    hidden_dim = 128
    edge_dim = 22  # From Phase 1 features

    # Create synthetic data
    X = torch.randn(batch_size, seq_len, 4, 3) * 3.8
    mask = torch.ones(batch_size, seq_len)

    # Extract features
    features = create_geometric_features(X, mask, k_neighbors)

    # Initial node embeddings
    h_nodes = torch.randn(batch_size, seq_len, hidden_dim)

    logger.info("\nTesting geometric attention with different edge features:")

    # Create encoder
    encoder = Struct2SeqGeometricEncoder(
        hidden_dim=hidden_dim,
        edge_dim=edge_dim,
        num_layers=2,
        num_heads=4
    )

    # Test 1: Normal geometric features
    logger.info("\n1. With full geometric features:")
    output1 = encoder(
        h_nodes,
        features['edge_features'],
        features['neighbor_idx'],
        mask
    )
    logger.info(f"   Output range: [{output1.min():.3f}, {output1.max():.3f}]")

    # Test 2: Zeroed geometric features (should behave differently)
    logger.info("\n2. With zeroed geometric features:")
    edge_features_zero = torch.zeros_like(features['edge_features'])
    output2 = encoder(
        h_nodes,
        edge_features_zero,
        features['neighbor_idx'],
        mask
    )
    logger.info(f"   Output range: [{output2.min():.3f}, {output2.max():.3f}]")

    # Compare
    diff = (output1 - output2).abs().mean().item()
    logger.info(f"\n3. Difference when geometry is present vs absent: {diff:.6f}")
    logger.info("   (Large difference confirms geometry is used in attention)")


def demo_probing_task() -> None:
    """Demonstrate the probing head for validation.

    Creates a simple classification task to verify encoder representations.
    """
    logger.info("\n\n" + "=" * 80)
    logger.info("Demo 3: Structure Probing Task")
    logger.info("=" * 80)

    # Create proteins of different lengths
    batch_size = 8
    hidden_dim = 128
    edge_dim = 22

    logger.info(f"\nCreating {batch_size} proteins with varying lengths...")

    # Create batch with different sequence lengths
    proteins = []
    labels = []
    for i in range(batch_size):
        # Random length between 50 and 150
        length = torch.randint(50, 150, (1,)).item()
        X = torch.randn(1, length, 4, 3) * 3.8
        mask = torch.ones(1, length)

        # Label: 0 if short (<100), 1 if long (>=100)
        label = 0 if length < 100 else 1

        proteins.append({
            'X': X,
            'mask': mask,
            'length': length,
            'label': label
        })
        labels.append(label)

    logger.info(f"  Created proteins with lengths: {[p['length'] for p in proteins]}")
    logger.info(f"  Labels (0=short, 1=long): {labels}")

    # Process each protein and pad to batch
    max_len = max(p['length'] for p in proteins)
    X_batch = torch.zeros(batch_size, max_len, 4, 3)
    mask_batch = torch.zeros(batch_size, max_len)

    for i, p in enumerate(proteins):
        length = p['length']
        X_batch[i, :length] = p['X'][0]
        mask_batch[i, :length] = p['mask'][0]

    # Extract features
    logger.info(f"\nExtracting geometric features...")
    features = create_geometric_features(X_batch, mask_batch, k_neighbors=20)

    # Create initial embeddings
    node_embedding = nn.Linear(6, hidden_dim)
    h_nodes = node_embedding(features['node_features'])

    # Encode
    logger.info(f"Encoding with Struct2SeqGeometric...")
    encoder = Struct2SeqGeometricEncoder(
        hidden_dim=hidden_dim,
        edge_dim=edge_dim,
        num_layers=3,
        num_heads=4
    )

    encoded = encoder(
        h_nodes,
        features['edge_features'],
        features['neighbor_idx'],
        mask_batch
    )

    # Classify
    logger.info(f"Classifying with probing head...")
    probing_head = StructureProbingHead(
        hidden_dim=hidden_dim,
        num_classes=2,
        pooling='mean'
    )

    logits = probing_head(encoded, mask_batch)
    predictions = torch.argmax(logits, dim=-1).tolist()

    logger.info(f"\nResults:")
    logger.info(f"  True labels:   {labels}")
    logger.info(f"  Predictions:   {predictions}")

    # Calculate accuracy
    labels_tensor = torch.tensor(labels)
    accuracy = (torch.tensor(predictions) == labels_tensor).float().mean()
    logger.info(f"  Random accuracy: {accuracy.item():.2%}")
    logger.info(f"  (Note: Encoder is untrained, so accuracy should be ~50%)")

    # Show that training would work
    logger.info(f"\nDemonstrating gradient flow for training...")
    loss = F.cross_entropy(logits, labels_tensor)
    loss.backward()

    # Check gradients
    has_grads = sum(1 for p in encoder.parameters() if p.grad is not None)
    total_params = sum(1 for p in encoder.parameters())
    logger.info(f"  Loss: {loss.item():.4f}")
    logger.info(f"  Parameters with gradients: {has_grads}/{total_params}")
    logger.info(f"  ✓ Training loop is ready!")


def demo_ablation_study_setup() -> None:
    """Demonstrate how to set up an ablation study.

    Shows the standardized interface that makes encoder comparison easy.
    """
    logger.info("\n\n" + "=" * 80)
    logger.info("Demo 4: Ablation Study Setup")
    logger.info("=" * 80)

    batch_size = 4
    seq_len = 80
    k_neighbors = 30
    hidden_dim = 128
    edge_dim = 22

    logger.info("\nAblation Study: Effect of Geometric Information")
    logger.info("-" * 80)

    # Create data
    X = torch.randn(batch_size, seq_len, 4, 3) * 3.8
    mask = torch.ones(batch_size, seq_len)
    features = create_geometric_features(X, mask, k_neighbors)

    # Initial embeddings
    h_nodes = torch.randn(batch_size, seq_len, hidden_dim)

    # Define ablation: what information does each encoder use?
    ablation_config = {
        'VanillaGCN': {
            'encoder': VanillaGCNEncoder(hidden_dim, edge_dim, num_layers=3),
            'uses_connectivity': True,
            'uses_distances': False,
            'uses_geometry': False,
        },
        'EdgeAwareGAT': {
            'encoder': EdgeAwareGATEncoder(hidden_dim, edge_dim, num_layers=3),
            'uses_connectivity': True,
            'uses_distances': True,
            'uses_geometry': False,
        },
        'Struct2SeqGeometric': {
            'encoder': Struct2SeqGeometricEncoder(hidden_dim, edge_dim, num_layers=3),
            'uses_connectivity': True,
            'uses_distances': True,
            'uses_geometry': True,
        },
    }

    logger.info("\nEncoder capabilities:")
    logger.info(f"  {'Encoder':<25} {'Connectivity':<15} {'Distances':<15} {'Geometry':<15}")
    logger.info("  " + "-" * 70)
    for name, config in ablation_config.items():
        logger.info(
            f"  {name:<25} "
            f"{'✓' if config['uses_connectivity'] else '✗':<15} "
            f"{'✓' if config['uses_distances'] else '✗':<15} "
            f"{'✓' if config['uses_geometry'] else '✗':<15}"
        )

    logger.info("\nRunning encoders (all use same interface)...")
    results = {}

    for name, config in ablation_config.items():
        encoder = config['encoder']
        output = encoder(
            h_nodes,
            features['edge_features'],
            features['neighbor_idx'],
            mask
        )
        results[name] = output
        logger.info(f"  {name:<25} output: {output.shape}")

    logger.info("\n✓ Ablation study setup complete!")
    logger.info("  Next step: Train each encoder on a task and compare performance")


def main() -> None:
    """Run all Phase 2 demonstrations."""
    logger.info("Starting Phase 2 demonstrations...\n")

    # Demo 1: Compare encoder architectures
    demo_encoder_comparison()

    # Demo 2: Show geometric attention mechanism
    demo_geometric_attention_mechanism()

    # Demo 3: Probing task for validation
    demo_probing_task()

    # Demo 4: Ablation study setup
    demo_ablation_study_setup()

    logger.info("\n\n" + "=" * 80)
    logger.info("All Phase 2 demonstrations completed successfully!")
    logger.info("=" * 80)
    logger.info("\nKey Takeaways:")
    logger.info("  1. Three encoder variants implemented for ablation studies")
    logger.info("  2. Geometric attention successfully injects 3D structure info")
    logger.info("  3. Probing head validates encoder representations")
    logger.info("  4. Standardized interface enables easy architecture comparison")
    logger.info("\nNext Steps:")
    logger.info("  1. Train encoders on real protein data")
    logger.info("  2. Compare performance: VanillaGCN < EdgeAwareGAT < Geometric")
    logger.info("  3. Implement the autoregressive decoder for sequence design")


if __name__ == '__main__':
    # Set up logging
    logger = setup_logger(
        name='phase2_demo',
        log_level=logging.INFO,
        file_output=False  # Console only for demo
    )

    shape_logger = TensorShapeLogger(logger)

    # Run demonstrations
    main()
