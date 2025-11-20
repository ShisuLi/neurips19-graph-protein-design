"""Geometric transformations for protein structures.

This module implements critical geometric operations for protein design,
including local coordinate frame construction and neighbor feature computation.
All functions follow the engineering standards with type hints, docstrings,
and proper shape annotations.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from utils import get_logger

logger = get_logger(__name__)


def get_local_frames(
    X: torch.Tensor,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct local coordinate frames for each residue using Gram-Schmidt.

    This function builds a local orthonormal coordinate system (rotation matrix R
    and translation vector t) for each residue based on its backbone atoms. The
    local frame is constructed using the Gram-Schmidt process on vectors formed
    by N-CA and C-CA bonds.

    The local frame is defined as:
    - Origin: CA (alpha carbon) position
    - X-axis: Normalized CA -> C direction
    - Y-axis: Gram-Schmidt orthogonalization of CA -> N relative to X-axis
    - Z-axis: Cross product of X and Y axes

    Args:
        X: Backbone atom coordinates with shape [Batch, Length, 4, 3].
           The 4 atoms are ordered as [N, CA, C, O].
        eps: Small constant for numerical stability. Defaults to 1e-6.

    Returns:
        Tuple containing:
            - R: Rotation matrices of shape [Batch, Length, 3, 3].
                 Each R[b, i] is an orthonormal matrix defining the local frame.
            - t: Translation vectors of shape [Batch, Length, 3].
                 Each t[b, i] is the CA position (origin of local frame).

    Shape:
        - Input X: [Batch, Length, 4, 3]
        - Output R: [Batch, Length, 3, 3]
        - Output t: [Batch, Length, 3]

    Example:
        >>> X = torch.randn(2, 100, 4, 3)  # 2 proteins, 100 residues each
        >>> R, t = get_local_frames(X)
        >>> print(R.shape)  # torch.Size([2, 100, 3, 3])
        >>> print(t.shape)  # torch.Size([2, 100, 3])

    Note:
        The rotation matrices are guaranteed to be orthonormal (within
        numerical precision), meaning R @ R.T = I and det(R) = 1.
    """
    # Extract backbone atoms: [N, CA, C, O]
    # Shape: [Batch, Length, 3] for each atom
    N = X[:, :, 0, :]   # Nitrogen
    CA = X[:, :, 1, :]  # Alpha carbon
    C = X[:, :, 2, :]   # Carbonyl carbon
    # O = X[:, :, 3, :]  # Oxygen (not used for frame construction)

    # Set translation as CA position (origin of local frame)
    # Shape: [Batch, Length, 3]
    t = CA

    # Construct first basis vector: CA -> C (will become X-axis after normalization)
    # Shape: [Batch, Length, 3]
    v1 = C - CA
    e1 = F.normalize(v1, dim=-1, eps=eps)

    # Construct second basis vector before orthogonalization: CA -> N
    # Shape: [Batch, Length, 3]
    v2 = N - CA

    # Gram-Schmidt orthogonalization: remove component of v2 along e1
    # u2 = v2 - (v2 · e1) * e1
    # Shape: [Batch, Length, 3]
    u2 = v2 - torch.sum(v2 * e1, dim=-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1, eps=eps)

    # Third basis vector: cross product to ensure right-handed coordinate system
    # e3 = e1 × e2
    # Shape: [Batch, Length, 3]
    e3 = torch.cross(e1, e2, dim=-1)

    # Stack basis vectors to form rotation matrix
    # Each column is a basis vector of the local frame
    # Shape: [Batch, Length, 3, 3]
    R = torch.stack([e1, e2, e3], dim=-1)

    logger.debug(f'Constructed local frames - R: {R.shape}, t: {t.shape}')

    return R, t


def get_rbf_features(
    distances: torch.Tensor,
    D_min: float = 0.0,
    D_max: float = 20.0,
    D_count: int = 16
) -> torch.Tensor:
    """Compute Radial Basis Function (RBF) features for distances.

    RBF features encode distances using Gaussian kernels centered at evenly
    spaced values. This provides a smooth, differentiable representation of
    distance information.

    Args:
        distances: Pairwise distances with shape [Batch, Length, K] or
                  [Batch, Length, Length].
        D_min: Minimum distance for RBF centers. Defaults to 0.0 Angstroms.
        D_max: Maximum distance for RBF centers. Defaults to 20.0 Angstroms.
        D_count: Number of RBF kernels. Defaults to 16.

    Returns:
        RBF features with shape [..., D_count], where ... matches the input shape.

    Shape:
        - Input: [Batch, Length, K] or [Batch, Length, Length]
        - Output: [..., D_count] (adds RBF dimension)

    Example:
        >>> distances = torch.randn(2, 100, 30).abs()  # K=30 neighbors
        >>> rbf = get_rbf_features(distances, D_count=16)
        >>> print(rbf.shape)  # torch.Size([2, 100, 30, 16])

    Note:
        The RBF kernel is defined as:
        RBF_i(d) = exp(-((d - μ_i) / σ)^2)
        where μ_i are evenly spaced centers and σ = (D_max - D_min) / D_count.
    """
    # Create evenly spaced RBF centers
    # Shape: [D_count]
    D_mu = torch.linspace(D_min, D_max, D_count, device=distances.device)
    D_mu = D_mu.view([1] * (distances.dim()) + [-1])

    # RBF width
    D_sigma = (D_max - D_min) / D_count

    # Expand distances for broadcasting
    # Shape: [..., 1]
    D_expand = distances.unsqueeze(-1)

    # Compute RBF features: exp(-((d - μ) / σ)^2)
    # Shape: [..., D_count]
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    logger.debug(f'Computed RBF features: {RBF.shape}')

    return RBF


def get_neighbor_features(
    X: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    mask: torch.Tensor,
    k_neighbors: int = 30,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute k-nearest neighbor features in local coordinate frames.

    This function finds k-nearest neighbors based on CA-CA distance, then computes:
    1. Relative positions: Neighbor coordinates transformed into the central
       residue's local frame
    2. Relative orientations: Rotation matrices between the central residue
       and its neighbors
    3. Distance-based features using RBF encoding

    Args:
        X: Backbone atom coordinates with shape [Batch, Length, 4, 3].
           The 4 atoms are [N, CA, C, O].
        R: Rotation matrices from get_local_frames(), shape [Batch, Length, 3, 3].
        t: Translation vectors from get_local_frames(), shape [Batch, Length, 3].
        mask: Validity mask with shape [Batch, Length]. 1 for valid residues,
              0 for padding/missing residues.
        k_neighbors: Number of nearest neighbors to consider. Defaults to 30.
        eps: Small constant for numerical stability. Defaults to 1e-6.

    Returns:
        Tuple containing:
            - rel_pos: Relative positions of neighbors in local frames.
                      Shape: [Batch, Length, K, 3]
            - rel_orient: Relative orientation features (unit vectors).
                         Shape: [Batch, Length, K, 3]
            - distances: Distances to k-nearest neighbors.
                        Shape: [Batch, Length, K]
            - neighbor_idx: Indices of k-nearest neighbors.
                           Shape: [Batch, Length, K]

    Shape:
        - Input X: [Batch, Length, 4, 3]
        - Input R: [Batch, Length, 3, 3]
        - Input t: [Batch, Length, 3]
        - Input mask: [Batch, Length]
        - Output rel_pos: [Batch, Length, K, 3]
        - Output rel_orient: [Batch, Length, K, 3]
        - Output distances: [Batch, Length, K]
        - Output neighbor_idx: [Batch, Length, K]

    Example:
        >>> X = torch.randn(2, 100, 4, 3)
        >>> mask = torch.ones(2, 100)
        >>> R, t = get_local_frames(X)
        >>> rel_pos, rel_orient, dist, idx = get_neighbor_features(
        ...     X, R, t, mask, k_neighbors=30
        ... )
        >>> print(rel_pos.shape)  # torch.Size([2, 100, 30, 3])

    Note:
        The relative position is computed as:
        rel_pos[i, j] = R[i]^T @ (t[neighbor[j]] - t[i])
        This transforms the neighbor's position into residue i's local frame.
    """
    batch_size = X.shape[0]
    seq_len = X.shape[1]

    # Extract CA coordinates for distance calculation
    # Shape: [Batch, Length, 3]
    CA = X[:, :, 1, :]

    # Compute pairwise distances between CA atoms
    # Shape: [Batch, Length, Length]
    mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)  # [Batch, Length, Length]
    dX = CA.unsqueeze(1) - CA.unsqueeze(2)  # [Batch, Length, Length, 3]
    distances_all = torch.sqrt(torch.sum(dX ** 2, dim=-1) + eps)  # [Batch, Length, Length]

    # Mask invalid pairs
    distances_all = mask_2D * distances_all

    # Find k-nearest neighbors
    # Set invalid distances to large value so they're not selected
    D_max = torch.max(distances_all, dim=-1, keepdim=True)[0]
    D_adjust = distances_all + (1.0 - mask_2D) * D_max

    # Get k-nearest neighbors (including self)
    # Shape: [Batch, Length, K]
    distances, neighbor_idx = torch.topk(
        D_adjust, k_neighbors, dim=-1, largest=False
    )

    # Gather neighbor information
    # Expand dimensions for gathering
    # neighbor_idx: [Batch, Length, K] -> [Batch, Length, K, 1, 1] for R
    #                                   -> [Batch, Length, K, 1] for t

    # Gather neighbor rotation matrices
    # Shape: [Batch, Length, K, 3, 3]
    idx_expand_R = neighbor_idx.unsqueeze(-1).unsqueeze(-1).expand(
        batch_size, seq_len, k_neighbors, 3, 3
    )
    R_neighbors = torch.gather(
        R.unsqueeze(2).expand(batch_size, seq_len, seq_len, 3, 3),
        dim=2,
        index=idx_expand_R
    )

    # Gather neighbor translation vectors
    # Shape: [Batch, Length, K, 3]
    idx_expand_t = neighbor_idx.unsqueeze(-1).expand(
        batch_size, seq_len, k_neighbors, 3
    )
    t_neighbors = torch.gather(
        t.unsqueeze(2).expand(batch_size, seq_len, seq_len, 3),
        dim=2,
        index=idx_expand_t
    )

    # Compute relative positions in local frames
    # rel_pos = R^T @ (t_neighbor - t_central)
    # Shape: [Batch, Length, K, 3]
    dX_local = t_neighbors - t.unsqueeze(2)  # [Batch, Length, K, 3]
    rel_pos = torch.matmul(
        R.unsqueeze(2).transpose(-1, -2),  # [Batch, Length, 1, 3, 3]
        dX_local.unsqueeze(-1)  # [Batch, Length, K, 3, 1]
    ).squeeze(-1)  # [Batch, Length, K, 3]

    # Normalize to get unit direction vectors
    rel_pos_normalized = F.normalize(rel_pos, dim=-1, eps=eps)

    # Compute relative orientations
    # R_relative = R_central^T @ R_neighbor
    # Shape: [Batch, Length, K, 3, 3]
    R_relative = torch.matmul(
        R.unsqueeze(2).transpose(-1, -2),  # [Batch, Length, 1, 3, 3]
        R_neighbors  # [Batch, Length, K, 3, 3]
    )

    # Extract orientation features (first column of relative rotation)
    # This represents the neighbor's X-axis direction in the central frame
    # Shape: [Batch, Length, K, 3]
    rel_orient = R_relative[:, :, :, :, 0]

    logger.debug(
        f'Neighbor features - rel_pos: {rel_pos.shape}, '
        f'rel_orient: {rel_orient.shape}, distances: {distances.shape}'
    )

    return rel_pos_normalized, rel_orient, distances, neighbor_idx


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions.

    Converts a batch of 3D rotation matrices to quaternion representation.
    Quaternions provide a compact, singularity-free representation of rotations.

    Args:
        R: Rotation matrices with shape [..., 3, 3].

    Returns:
        Quaternions with shape [..., 4], ordered as [x, y, z, w].

    Shape:
        - Input: [..., 3, 3]
        - Output: [..., 4]

    Example:
        >>> R = torch.eye(3).unsqueeze(0)  # Identity rotation
        >>> q = rotation_matrix_to_quaternion(R)
        >>> print(q)  # Should be close to [0, 0, 0, 1]

    Note:
        The quaternion is normalized to unit length.
        Uses the stable conversion method from Wikipedia.
    """
    # Extract diagonal elements
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)

    # Compute magnitudes for x, y, z components
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
        Rxx - Ryy - Rzz,
        -Rxx + Ryy - Rzz,
        -Rxx - Ryy + Rzz
    ], -1)))

    # Compute signs from off-diagonal elements
    _R = lambda i, j: R[..., i, j]
    signs = torch.sign(torch.stack([
        _R(2, 1) - _R(1, 2),
        _R(0, 2) - _R(2, 0),
        _R(1, 0) - _R(0, 1)
    ], -1))

    # Compute x, y, z components
    xyz = signs * magnitudes

    # Compute w component (ensure non-negative trace)
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.0

    # Combine and normalize
    Q = torch.cat([xyz, w], dim=-1)
    Q = F.normalize(Q, dim=-1)

    return Q


def compute_dihedral_angles(
    X: torch.Tensor,
    eps: float = 1e-7
) -> torch.Tensor:
    """Compute backbone dihedral angles (phi, psi, omega).

    Computes the three backbone dihedral angles for each residue:
    - phi: rotation around N-CA bond
    - psi: rotation around CA-C bond
    - omega: rotation around C-N bond (peptide bond, usually ~180°)

    Args:
        X: Backbone atom coordinates with shape [Batch, Length, 4, 3].
           The 4 atoms are [N, CA, C, O].
        eps: Small constant for numerical stability. Defaults to 1e-7.

    Returns:
        Dihedral features with shape [Batch, Length, 6].
        Features are [cos(phi), sin(phi), cos(psi), sin(psi), cos(omega), sin(omega)].

    Shape:
        - Input: [Batch, Length, 4, 3]
        - Output: [Batch, Length, 6]

    Example:
        >>> X = torch.randn(2, 100, 4, 3)
        >>> dihedrals = compute_dihedral_angles(X)
        >>> print(dihedrals.shape)  # torch.Size([2, 100, 6])

    Note:
        Edge residues (first and last) will have padded values for missing angles.
        Angles are represented as (cos, sin) pairs to avoid discontinuities.
    """
    # Reshape to treat as sequence of atoms: N, CA, C from each residue
    # Shape: [Batch, 3*Length, 3]
    X_flat = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

    # Compute unit vectors between consecutive atoms
    # Shape: [Batch, 3*Length-1, 3]
    dX = X_flat[:, 1:, :] - X_flat[:, :-1, :]
    U = F.normalize(dX, dim=-1, eps=eps)

    # Extract shifted slices
    u_2 = U[:, :-2, :]  # Shape: [Batch, 3*Length-3, 3]
    u_1 = U[:, 1:-1, :]
    u_0 = U[:, 2:, :]

    # Compute normals to planes defined by consecutive bonds
    # Shape: [Batch, 3*Length-4, 3]
    n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1, eps=eps)
    n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1, eps=eps)

    # Compute dihedral angles
    # cosD = n_2 · n_1
    cosD = (n_2[:, :-1, :] * n_1[:, :-1, :]).sum(-1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)

    # Determine sign from triple product
    D = torch.sign((u_2[:, :-1, :] * n_1[:, :-1, :]).sum(-1)) * torch.acos(cosD)

    # Pad to match sequence length and reshape to (batch, length, 3)
    # This removes phi[0], psi[-1], omega[-1]
    D = F.pad(D, (1, 2), 'constant', 0)
    D = D.view((D.size(0), D.size(1) // 3, 3))

    # Separate into phi, psi, omega
    # Shape: [Batch, Length] for each
    phi, psi, omega = torch.unbind(D, dim=-1)

    # Convert to (cos, sin) representation to avoid discontinuities
    # Shape: [Batch, Length, 6]
    dihedral_features = torch.stack([
        torch.cos(phi), torch.sin(phi),
        torch.cos(psi), torch.sin(psi),
        torch.cos(omega), torch.sin(omega)
    ], dim=-1)

    logger.debug(f'Computed dihedral angles: {dihedral_features.shape}')

    return dihedral_features
