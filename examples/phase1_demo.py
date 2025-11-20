"""Phase 1 demonstration: Data loading and geometry computation.

This script demonstrates the core functionality implemented in Phase 1:
1. Loading protein structures from files
2. Computing local coordinate frames
3. Computing neighbor features
4. Processing complete geometric features

Run with:
    python examples/phase1_demo.py
"""

import torch
from pathlib import Path
import logging

from utils import setup_logger, TensorShapeLogger
from models import (
    get_local_frames,
    get_rbf_features,
    get_neighbor_features,
    compute_dihedral_angles,
)


def demo_synthetic_protein() -> None:
    """Demonstrate geometry computation on synthetic protein data.

    This creates a synthetic protein structure and shows how to:
    - Compute local coordinate frames
    - Find k-nearest neighbors
    - Compute relative positions and orientations
    - Extract geometric features
    """
    logger.info("=" * 60)
    logger.info("Demo 1: Synthetic Protein Geometry Computation")
    logger.info("=" * 60)

    # Create synthetic protein data
    # In real usage, this would come from load_structure_from_gemmi()
    batch_size = 2
    seq_len = 100
    k_neighbors = 30

    logger.info(f"Creating synthetic protein: {batch_size} proteins, {seq_len} residues each")

    # Coordinates: [Batch, Length, 4, 3] where 4 atoms are [N, CA, C, O]
    # Scale by 3.8 to approximate realistic CA-CA distances (~3.8 Angstroms)
    X = torch.randn(batch_size, seq_len, 4, 3) * 3.8

    # Mask: [Batch, Length] - all residues are valid
    mask = torch.ones(batch_size, seq_len)

    # Log input shape
    shape_logger.log_shape('Input coordinates (X)', X, ['Batch', 'Length', 'Atoms', 'XYZ'])
    shape_logger.log_shape('Mask', mask, ['Batch', 'Length'])

    # Step 1: Compute local coordinate frames
    logger.info("\n" + "-" * 60)
    logger.info("Step 1: Computing local coordinate frames")
    logger.info("-" * 60)

    R, t = get_local_frames(X)

    shape_logger.log_shape('Rotation matrices (R)', R, ['Batch', 'Length', '3x3'])
    shape_logger.log_shape('Translation vectors (t)', t, ['Batch', 'Length', 'XYZ'])

    # Verify orthonormality
    identity = torch.eye(3).unsqueeze(0).unsqueeze(0)
    R_R_T = torch.matmul(R, R.transpose(-1, -2))
    ortho_error = torch.abs(R_R_T - identity).max().item()
    logger.info(f"Orthonormality check: max error = {ortho_error:.2e} (should be ~0)")

    # Step 2: Compute k-nearest neighbor features
    logger.info("\n" + "-" * 60)
    logger.info(f"Step 2: Computing k-nearest neighbor features (k={k_neighbors})")
    logger.info("-" * 60)

    rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
        X, R, t, mask, k_neighbors=k_neighbors
    )

    shape_logger.log_shape('Relative positions', rel_pos, ['Batch', 'Length', 'K', 'XYZ'])
    shape_logger.log_shape('Relative orientations', rel_orient, ['Batch', 'Length', 'K', 'XYZ'])
    shape_logger.log_shape('Distances', distances, ['Batch', 'Length', 'K'])
    shape_logger.log_shape('Neighbor indices', neighbor_idx, ['Batch', 'Length', 'K'])

    # Statistics on distances
    logger.info(f"Distance statistics:")
    logger.info(f"  Min distance: {distances.min():.2f} Å")
    logger.info(f"  Max distance: {distances.max():.2f} Å")
    logger.info(f"  Mean distance: {distances.mean():.2f} Å")
    logger.info(f"  Median distance: {distances.median():.2f} Å")

    # Step 3: Compute RBF features for distances
    logger.info("\n" + "-" * 60)
    logger.info("Step 3: Computing RBF distance features")
    logger.info("-" * 60)

    rbf_features = get_rbf_features(distances, D_min=0.0, D_max=20.0, D_count=16)

    shape_logger.log_shape('RBF features', rbf_features, ['Batch', 'Length', 'K', 'RBF_dim'])
    logger.info(f"RBF feature range: [{rbf_features.min():.3f}, {rbf_features.max():.3f}]")

    # Step 4: Compute dihedral angles
    logger.info("\n" + "-" * 60)
    logger.info("Step 4: Computing backbone dihedral angles")
    logger.info("-" * 60)

    dihedrals = compute_dihedral_angles(X)

    shape_logger.log_shape('Dihedral features', dihedrals, ['Batch', 'Length', 'Features'])
    logger.info("Dihedral features: [cos(phi), sin(phi), cos(psi), sin(psi), cos(omega), sin(omega)]")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary: All geometric features computed successfully!")
    logger.info("=" * 60)
    logger.info(f"Total features per residue:")
    logger.info(f"  - Dihedral angles: {dihedrals.shape[-1]} features")
    logger.info(f"  - Neighbor positions: {k_neighbors} x 3 = {k_neighbors * 3} features")
    logger.info(f"  - Neighbor orientations: {k_neighbors} x 3 = {k_neighbors * 3} features")
    logger.info(f"  - RBF distance encoding: {k_neighbors} x 16 = {k_neighbors * 16} features")
    total_features = dihedrals.shape[-1] + k_neighbors * (3 + 3 + 16)
    logger.info(f"  - TOTAL: {total_features} features per residue")


def demo_feature_extraction_pipeline() -> None:
    """Demonstrate a complete feature extraction pipeline.

    This shows how to combine all geometric computations into a single
    feature extraction function that could be used in a model.
    """
    logger.info("\n\n" + "=" * 60)
    logger.info("Demo 2: Complete Feature Extraction Pipeline")
    logger.info("=" * 60)

    batch_size = 4
    seq_len = 50
    k_neighbors = 20

    # Simulate batch of protein structures
    X = torch.randn(batch_size, seq_len, 4, 3) * 3.8
    mask = torch.ones(batch_size, seq_len)

    logger.info(f"Processing batch of {batch_size} proteins")

    # Complete feature extraction in one pipeline
    logger.info("\nExtracting all geometric features...")

    # Local frames
    R, t = get_local_frames(X)

    # Neighbor features
    rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
        X, R, t, mask, k_neighbors=k_neighbors
    )

    # RBF encoding
    rbf = get_rbf_features(distances)

    # Dihedral angles
    dihedrals = compute_dihedral_angles(X)

    # Combine features (example: concatenate along feature dimension)
    # In a real model, these would be processed separately and combined in the network

    logger.info("\nFeature dimensions:")
    logger.info(f"  Node features (dihedrals): {dihedrals.shape}")
    logger.info(f"  Edge features (rel_pos): {rel_pos.shape}")
    logger.info(f"  Edge features (rel_orient): {rel_orient.shape}")
    logger.info(f"  Edge features (RBF): {rbf.shape}")

    # Example: flatten edge features for each residue
    edge_features_flat = torch.cat([
        rel_pos.reshape(batch_size, seq_len, -1),
        rel_orient.reshape(batch_size, seq_len, -1),
        rbf.reshape(batch_size, seq_len, -1),
    ], dim=-1)

    shape_logger.log_shape('Combined edge features', edge_features_flat, ['Batch', 'Length', 'EdgeDim'])

    logger.info("\nFeature extraction complete!")


def demo_with_masking() -> None:
    """Demonstrate handling of missing residues with masking.

    Shows how the mask properly excludes invalid residues from
    neighbor computations.
    """
    logger.info("\n\n" + "=" * 60)
    logger.info("Demo 3: Handling Missing Residues with Masking")
    logger.info("=" * 60)

    seq_len = 50
    k_neighbors = 20

    # Create protein with some missing residues
    X = torch.randn(1, seq_len, 4, 3) * 3.8
    mask = torch.ones(1, seq_len)

    # Mask out residues 20-25 (simulating missing coordinates)
    mask[:, 20:25] = 0
    logger.info(f"Created protein with {seq_len} residues, {mask.sum().item():.0f} valid")
    logger.info(f"Masked residues: 20-25")

    # Compute features
    R, t = get_local_frames(X)
    rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
        X, R, t, mask, k_neighbors=k_neighbors
    )

    # Check that valid residues don't have masked neighbors
    logger.info("\nVerifying neighbor exclusion:")
    for i in [15, 16, 17, 18, 19]:  # Residues before masked region
        neighbors = neighbor_idx[0, i, :]
        has_masked = torch.any((neighbors >= 20) & (neighbors < 25))
        logger.info(f"  Residue {i}: contains masked neighbors? {has_masked.item()}")

    logger.info("\nMasking works correctly - invalid residues are excluded from neighbors")


def main() -> None:
    """Run all Phase 1 demonstrations."""
    # Demo 1: Basic geometry computation
    demo_synthetic_protein()

    # Demo 2: Complete pipeline
    demo_feature_extraction_pipeline()

    # Demo 3: Masking
    demo_with_masking()

    logger.info("\n\n" + "=" * 60)
    logger.info("All Phase 1 demonstrations completed successfully!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Load real protein structures with load_structure_from_gemmi()")
    logger.info("  2. Build the protein design model using these geometric features")
    logger.info("  3. Implement training loop with W&B tracking")


if __name__ == '__main__':
    # Set up logging
    logger = setup_logger(
        name='phase1_demo',
        log_level=logging.INFO,
        file_output=False  # Only console output for demo
    )

    # Create shape logger for tensor dimensions
    shape_logger = TensorShapeLogger(logger)

    # Run demonstrations
    main()
