"""
Protein feature computation module
Computes node and edge features from protein 3D structures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProteinFeaturizer(nn.Module):
    """
    Computes protein structural features for graph neural networks.

    Features computed:
    - Node features: Backbone dihedral angles (phi, psi, omega) as sin/cos pairs
    - Edge features: Positional encodings, RBF distance features, orientation quaternions
    """

    def __init__(
        self,
        num_positional_embeddings=16,
        num_rbf=16,
        features_type='full'
    ):
        super().__init__()
        self.num_positional_embeddings = num_positional_embeddings
        self.num_rbf = num_rbf
        self.features_type = features_type

    def compute_dihedrals(self, coords):
        """
        Compute backbone dihedral angles (phi, psi, omega).

        Args:
            coords: [batch, length, 4, 3] - N, CA, C, O coordinates

        Returns:
            dihedral_features: [batch, length, 6] - cos/sin of phi, psi, omega
        """
        batch_size, length = coords.shape[:2]
        # Reshape to [batch, length*3, 3] for N, CA, C
        X = coords[:, :, :3, :].reshape(batch_size, length * 3, 3)

        # Compute unit vectors between consecutive atoms
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)

        # Shifted slices for dihedral calculation
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]

        # Compute normals to planes
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        # Dihedral angle
        cosD = torch.clamp((n_2 * n_1).sum(-1), -1 + 1e-7, 1 - 1e-7)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # Pad and reshape to get phi, psi, omega
        D = F.pad(D, (1, 2), 'constant', 0)
        D = D.view(batch_size, length, 3)

        # Convert to sin/cos representation
        D_features = torch.cat([torch.cos(D), torch.sin(D)], dim=-1)

        return D_features

    def compute_rbf(self, distances, D_min=0., D_max=20.):
        """
        Radial basis function features for distances.

        Args:
            distances: [...] - pairwise distances

        Returns:
            rbf_features: [..., num_rbf]
        """
        D_mu = torch.linspace(D_min, D_max, self.num_rbf, device=distances.device)
        D_mu = D_mu.view([1] * len(distances.shape) + [-1])
        D_sigma = (D_max - D_min) / self.num_rbf
        D_expand = distances.unsqueeze(-1)

        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def compute_positional_encodings(self, edge_index):
        """
        Sinusoidal positional encodings for sequence distance.

        Args:
            edge_index: [2, num_edges] - edge connectivity

        Returns:
            pos_enc: [num_edges, num_positional_embeddings]
        """
        # Sequence distance (i - j)
        src, dst = edge_index
        d = (dst - src).float().unsqueeze(-1)

        # Sinusoidal frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_positional_embeddings, 2,
                        dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / self.num_positional_embeddings)
        )

        angles = d * frequency
        pos_enc = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

        return pos_enc

    def compute_quaternions(self, R):
        """
        Convert rotation matrices to quaternions.

        Args:
            R: [..., 3, 3] - rotation matrices

        Returns:
            Q: [..., 4] - quaternions
        """
        # Extract diagonal
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)

        # Compute quaternion components
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
            -Rxx + Ryy - Rzz,
            -Rxx - Ryy + Rzz
        ], dim=-1)))

        signs = torch.sign(torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1]
        ], dim=-1))

        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.

        Q = torch.cat([xyz, w], dim=-1)
        Q = F.normalize(Q, dim=-1)

        return Q

    def compute_orientations(self, coords_ca, edge_index, edge_distances):
        """
        Compute relative orientations between connected residues.

        Args:
            coords_ca: [num_nodes, 3] - CA coordinates
            edge_index: [2, num_edges] - edge connectivity
            edge_distances: [num_edges] - pairwise distances

        Returns:
            orientation_features: [num_edges, 7] - direction unit vector (3) + quaternion (4)
        """
        src, dst = edge_index

        # Compute local reference frames
        # For simplicity, we use the direction vectors
        dX = coords_ca[dst] - coords_ca[src]
        dU = F.normalize(dX, dim=-1)

        # For a full implementation, compute rotation matrices and quaternions
        # Here we use a simplified version with just direction vectors
        # In the full version, you would compute full orientation quaternions

        # Placeholder quaternions (in real implementation, compute from local frames)
        # For now, we'll just use the direction vector and pad with zeros
        # A full implementation would build local coordinate frames
        Q = torch.zeros(len(src), 4, device=coords_ca.device)
        Q[:, 0] = 1.0  # Identity quaternion

        orientation_features = torch.cat([dU, Q], dim=-1)

        return orientation_features
