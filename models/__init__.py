"""Models package for protein design.

This package contains core modules for protein structure processing:
- geometry: Geometric transformations and local frame construction
- data_loader: Data loading from BinaryCIF/PDB formats
"""

from .geometry import (
    get_local_frames,
    get_rbf_features,
    get_neighbor_features,
    rotation_matrix_to_quaternion,
    compute_dihedral_angles,
)

from .data_loader import (
    ProteinStructure,
    load_structure_from_gemmi,
    load_structure_from_biopython,
    sequence_to_tensor,
    tensor_to_sequence,
    ProteinDataset,
    collate_protein_batch,
    AMINO_ACIDS,
    AA_TO_IDX,
    IDX_TO_AA,
    BACKBONE_ATOMS,
)

__all__ = [
    # Geometry functions
    'get_local_frames',
    'get_rbf_features',
    'get_neighbor_features',
    'rotation_matrix_to_quaternion',
    'compute_dihedral_angles',
    # Data loading
    'ProteinStructure',
    'load_structure_from_gemmi',
    'load_structure_from_biopython',
    'sequence_to_tensor',
    'tensor_to_sequence',
    'ProteinDataset',
    'collate_protein_batch',
    # Constants
    'AMINO_ACIDS',
    'AA_TO_IDX',
    'IDX_TO_AA',
    'BACKBONE_ATOMS',
]
