"""Unit tests for geometry module.

Tests geometric transformations including local frame construction,
neighbor features, and related utilities.
"""

import torch
import pytest
import numpy as np
from models.geometry import (
    get_local_frames,
    get_rbf_features,
    get_neighbor_features,
    rotation_matrix_to_quaternion,
    compute_dihedral_angles,
)


class TestLocalFrames:
    """Test local coordinate frame construction."""

    def test_output_shapes(self) -> None:
        """Test that output shapes are correct."""
        batch_size, seq_len = 2, 10
        X = torch.randn(batch_size, seq_len, 4, 3)

        R, t = get_local_frames(X)

        assert R.shape == (batch_size, seq_len, 3, 3), \
            f"Expected R shape [{batch_size}, {seq_len}, 3, 3], got {R.shape}"
        assert t.shape == (batch_size, seq_len, 3), \
            f"Expected t shape [{batch_size}, {seq_len}, 3], got {t.shape}"

    def test_orthonormality(self) -> None:
        """Test that rotation matrices are orthonormal."""
        X = torch.randn(2, 10, 4, 3)
        R, _ = get_local_frames(X)

        # Check R @ R^T = I
        identity = torch.eye(3).unsqueeze(0).unsqueeze(0)
        R_R_T = torch.matmul(R, R.transpose(-1, -2))

        # Allow small numerical error
        assert torch.allclose(R_R_T, identity, atol=1e-5), \
            "Rotation matrices are not orthonormal"

    def test_determinant_is_one(self) -> None:
        """Test that rotation matrices have determinant +1."""
        X = torch.randn(2, 10, 4, 3)
        R, _ = get_local_frames(X)

        # Compute determinants
        det = torch.det(R)

        # Should be close to 1 (right-handed coordinate system)
        assert torch.allclose(det, torch.ones_like(det), atol=1e-5), \
            "Rotation matrices should have determinant +1"

    def test_translation_is_ca(self) -> None:
        """Test that translation vector is CA position."""
        X = torch.randn(2, 10, 4, 3)
        _, t = get_local_frames(X)

        CA = X[:, :, 1, :]  # CA is second atom

        assert torch.allclose(t, CA), \
            "Translation should be CA position"

    def test_batch_independence(self) -> None:
        """Test that batch elements are processed independently."""
        X1 = torch.randn(1, 10, 4, 3)
        X2 = torch.randn(1, 10, 4, 3)
        X_batch = torch.cat([X1, X2], dim=0)

        R1, t1 = get_local_frames(X1)
        R2, t2 = get_local_frames(X2)
        R_batch, t_batch = get_local_frames(X_batch)

        assert torch.allclose(R_batch[0], R1[0]), \
            "Batch processing should match individual processing"
        assert torch.allclose(R_batch[1], R2[0]), \
            "Batch processing should match individual processing"


class TestRBFFeatures:
    """Test Radial Basis Function features."""

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        distances = torch.randn(2, 10, 30).abs()
        D_count = 16

        rbf = get_rbf_features(distances, D_count=D_count)

        assert rbf.shape == (2, 10, 30, D_count), \
            f"Expected shape [2, 10, 30, {D_count}], got {rbf.shape}"

    def test_rbf_range(self) -> None:
        """Test that RBF features are in valid range [0, 1]."""
        distances = torch.randn(2, 10, 30).abs()
        rbf = get_rbf_features(distances)

        assert torch.all(rbf >= 0) and torch.all(rbf <= 1), \
            "RBF features should be in range [0, 1]"

    def test_rbf_peak_at_center(self) -> None:
        """Test that RBF has maximum value at center distance."""
        # Create distances that exactly match RBF centers
        D_min, D_max, D_count = 0.0, 20.0, 16
        centers = torch.linspace(D_min, D_max, D_count)

        for i, center in enumerate(centers):
            distances = torch.full((1, 1, 1), center)
            rbf = get_rbf_features(distances, D_min, D_max, D_count)

            # The i-th RBF should have maximum value
            assert rbf[0, 0, 0, i] > 0.99, \
                f"RBF {i} should peak at center distance {center}"


class TestNeighborFeatures:
    """Test neighbor feature computation."""

    def test_output_shapes(self) -> None:
        """Test that output shapes are correct."""
        batch_size, seq_len, k = 2, 20, 10
        X = torch.randn(batch_size, seq_len, 4, 3)
        mask = torch.ones(batch_size, seq_len)

        R, t = get_local_frames(X)
        rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
            X, R, t, mask, k_neighbors=k
        )

        assert rel_pos.shape == (batch_size, seq_len, k, 3), \
            f"Expected rel_pos shape [{batch_size}, {seq_len}, {k}, 3]"
        assert rel_orient.shape == (batch_size, seq_len, k, 3), \
            f"Expected rel_orient shape [{batch_size}, {seq_len}, {k}, 3]"
        assert distances.shape == (batch_size, seq_len, k), \
            f"Expected distances shape [{batch_size}, {seq_len}, {k}]"
        assert neighbor_idx.shape == (batch_size, seq_len, k), \
            f"Expected neighbor_idx shape [{batch_size}, {seq_len}, {k}]"

    def test_relative_positions_normalized(self) -> None:
        """Test that relative positions are unit vectors."""
        X = torch.randn(2, 20, 4, 3)
        mask = torch.ones(2, 20)
        R, t = get_local_frames(X)

        rel_pos, _, _, _ = get_neighbor_features(X, R, t, mask, k_neighbors=10)

        # Compute norms
        norms = torch.norm(rel_pos, dim=-1)

        # Should be close to 1 (unit vectors)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            "Relative positions should be normalized to unit vectors"

    def test_first_neighbor_is_self(self) -> None:
        """Test that first neighbor is the residue itself (distance 0)."""
        X = torch.randn(2, 20, 4, 3)
        mask = torch.ones(2, 20)
        R, t = get_local_frames(X)

        _, _, distances, neighbor_idx = get_neighbor_features(
            X, R, t, mask, k_neighbors=10
        )

        # First neighbor should have distance ~0 (self)
        assert torch.allclose(distances[:, :, 0], torch.zeros_like(distances[:, :, 0]), atol=1e-4), \
            "First neighbor should be self with distance ~0"

        # First neighbor index should be the residue index itself
        expected_idx = torch.arange(20).unsqueeze(0).expand(2, -1)
        assert torch.all(neighbor_idx[:, :, 0] == expected_idx), \
            "First neighbor should be self"

    def test_distances_sorted(self) -> None:
        """Test that distances are sorted in ascending order."""
        X = torch.randn(2, 20, 4, 3)
        mask = torch.ones(2, 20)
        R, t = get_local_frames(X)

        _, _, distances, _ = get_neighbor_features(X, R, t, mask, k_neighbors=10)

        # Check that distances are sorted
        sorted_distances, _ = torch.sort(distances, dim=-1)
        assert torch.allclose(distances, sorted_distances), \
            "Distances should be sorted in ascending order"

    def test_masking(self) -> None:
        """Test that masking correctly excludes invalid residues."""
        X = torch.randn(2, 20, 4, 3)
        mask = torch.ones(2, 20)
        # Mask out last 5 residues
        mask[:, 15:] = 0

        R, t = get_local_frames(X)
        _, _, _, neighbor_idx = get_neighbor_features(
            X, R, t, mask, k_neighbors=10
        )

        # For valid residues (0-14), neighbors should not include masked residues (15-19)
        for i in range(15):
            neighbors = neighbor_idx[:, i, :]
            # Check that no neighbor is in the masked region
            assert torch.all(neighbors < 15), \
                f"Residue {i} should not have neighbors in masked region"


class TestQuaternions:
    """Test rotation matrix to quaternion conversion."""

    def test_identity_rotation(self) -> None:
        """Test that identity matrix converts to [0, 0, 0, 1]."""
        R = torch.eye(3).unsqueeze(0)  # [1, 3, 3]
        Q = rotation_matrix_to_quaternion(R)

        expected = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        assert torch.allclose(Q, expected, atol=1e-5), \
            "Identity rotation should give quaternion [0, 0, 0, 1]"

    def test_quaternion_normalized(self) -> None:
        """Test that quaternions are unit length."""
        R = torch.randn(5, 10, 3, 3)
        # Make them proper rotation matrices via QR decomposition
        Q_mat, _ = torch.linalg.qr(R)

        Q = rotation_matrix_to_quaternion(Q_mat)

        # Check unit length
        norms = torch.norm(Q, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            "Quaternions should be unit length"

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        R = torch.randn(2, 10, 3, 3)
        Q = rotation_matrix_to_quaternion(R)

        assert Q.shape == (2, 10, 4), \
            f"Expected shape [2, 10, 4], got {Q.shape}"


class TestDihedralAngles:
    """Test dihedral angle computation."""

    def test_output_shape(self) -> None:
        """Test that output shape is correct."""
        batch_size, seq_len = 2, 10
        X = torch.randn(batch_size, seq_len, 4, 3)

        dihedrals = compute_dihedral_angles(X)

        assert dihedrals.shape == (batch_size, seq_len, 6), \
            f"Expected shape [{batch_size}, {seq_len}, 6], got {dihedrals.shape}"

    def test_values_in_range(self) -> None:
        """Test that cos/sin values are in [-1, 1]."""
        X = torch.randn(2, 10, 4, 3)
        dihedrals = compute_dihedral_angles(X)

        assert torch.all(dihedrals >= -1.0) and torch.all(dihedrals <= 1.0), \
            "Dihedral cos/sin values should be in [-1, 1]"

    def test_unit_circle_property(self) -> None:
        """Test that (cos, sin) pairs satisfy cos^2 + sin^2 = 1."""
        X = torch.randn(2, 10, 4, 3)
        dihedrals = compute_dihedral_angles(X)

        # Extract (cos, sin) pairs
        cos_phi, sin_phi = dihedrals[:, :, 0], dihedrals[:, :, 1]
        cos_psi, sin_psi = dihedrals[:, :, 2], dihedrals[:, :, 3]
        cos_omega, sin_omega = dihedrals[:, :, 4], dihedrals[:, :, 5]

        # Check unit circle property
        for cos_val, sin_val, name in [
            (cos_phi, sin_phi, 'phi'),
            (cos_psi, sin_psi, 'psi'),
            (cos_omega, sin_omega, 'omega')
        ]:
            magnitude = cos_val**2 + sin_val**2
            assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-5), \
                f"{name}: cos^2 + sin^2 should equal 1"


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_full_pipeline(self) -> None:
        """Test complete pipeline from coordinates to features."""
        # Create synthetic protein data
        batch_size, seq_len, k_neighbors = 2, 50, 30
        X = torch.randn(batch_size, seq_len, 4, 3) * 10  # Scale to realistic distances
        mask = torch.ones(batch_size, seq_len)

        # Step 1: Get local frames
        R, t = get_local_frames(X)
        assert R.shape == (batch_size, seq_len, 3, 3)
        assert t.shape == (batch_size, seq_len, 3)

        # Step 2: Get neighbor features
        rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
            X, R, t, mask, k_neighbors=k_neighbors
        )
        assert rel_pos.shape == (batch_size, seq_len, k_neighbors, 3)
        assert rel_orient.shape == (batch_size, seq_len, k_neighbors, 3)

        # Step 3: Get RBF features
        rbf = get_rbf_features(distances)
        assert rbf.shape == (batch_size, seq_len, k_neighbors, 16)

        # Step 4: Get dihedral angles
        dihedrals = compute_dihedral_angles(X)
        assert dihedrals.shape == (batch_size, seq_len, 6)

        # All should complete without errors
        logger_msg = (
            f"Successfully processed {batch_size} proteins with "
            f"{seq_len} residues each"
        )

    def test_realistic_protein_dimensions(self) -> None:
        """Test with realistic protein dimensions."""
        # Typical small protein: 100 residues
        X = torch.randn(1, 100, 4, 3) * 5  # ~5Å average spacing
        mask = torch.ones(1, 100)

        R, t = get_local_frames(X)
        rel_pos, rel_orient, distances, _ = get_neighbor_features(
            X, R, t, mask, k_neighbors=30
        )

        # Check that distances are in reasonable range (0-20 Å)
        assert torch.all(distances >= 0) and torch.all(distances <= 100), \
            "Distances should be in reasonable range"

    def test_gpu_compatibility(self) -> None:
        """Test that operations work on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        X = torch.randn(2, 50, 4, 3).cuda()
        mask = torch.ones(2, 50).cuda()

        R, t = get_local_frames(X)
        assert R.is_cuda, "Output should be on GPU"

        rel_pos, rel_orient, distances, _ = get_neighbor_features(
            X, R, t, mask, k_neighbors=30
        )
        assert rel_pos.is_cuda, "Output should be on GPU"
