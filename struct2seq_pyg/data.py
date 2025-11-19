"""
PyTorch Geometric data loading for protein structures
"""

import json
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import knn_graph
from .features import ProteinFeaturizer


def create_protein_graph(
    coords,
    sequence,
    k_neighbors=30,
    featurizer=None,
    device='cpu'
):
    """
    Create a PyTorch Geometric Data object from protein structure.

    Args:
        coords: dict with keys 'N', 'CA', 'C', 'O' containing [length, 3] arrays
        sequence: str - amino acid sequence
        k_neighbors: int - number of nearest neighbors for graph construction
        featurizer: ProteinFeaturizer instance
        device: torch device

    Returns:
        data: PyTorch Geometric Data object with:
            - x: node features [num_nodes, node_feature_dim]
            - edge_index: graph connectivity [2, num_edges]
            - edge_attr: edge features [num_edges, edge_feature_dim]
            - seq: sequence as tensor [num_nodes]
            - coords: CA coordinates [num_nodes, 3]
            - pos: alias for coords (PyG convention)
    """
    if featurizer is None:
        featurizer = ProteinFeaturizer()

    # Convert sequence to indices
    AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
    seq_indices = torch.tensor([aa_to_idx.get(aa, 0) for aa in sequence],
                              dtype=torch.long, device=device)

    # Stack coordinates [length, 4, 3] for N, CA, C, O
    coords_tensor = torch.stack([
        torch.tensor(coords['N'], dtype=torch.float32, device=device),
        torch.tensor(coords['CA'], dtype=torch.float32, device=device),
        torch.tensor(coords['C'], dtype=torch.float32, device=device),
        torch.tensor(coords['O'], dtype=torch.float32, device=device),
    ], dim=1)

    # CA coordinates for graph construction
    coords_ca = coords_tensor[:, 1, :]  # [length, 3]

    # Build k-NN graph
    edge_index = knn_graph(coords_ca, k=k_neighbors, loop=False)

    # Compute node features (dihedral angles)
    node_features = featurizer.compute_dihedrals(coords_tensor.unsqueeze(0)).squeeze(0)

    # Compute edge features
    src, dst = edge_index
    edge_distances = torch.norm(coords_ca[dst] - coords_ca[src], dim=-1)

    # Positional encodings
    pos_enc = featurizer.compute_positional_encodings(edge_index)

    # RBF distance features
    rbf_features = featurizer.compute_rbf(edge_distances)

    # Orientation features
    orientation_features = featurizer.compute_orientations(
        coords_ca, edge_index, edge_distances
    )

    # Concatenate edge features
    edge_attr = torch.cat([pos_enc, rbf_features, orientation_features], dim=-1)

    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        seq=seq_indices,
        coords=coords_ca,
        pos=coords_ca,  # PyG convention
        num_nodes=len(sequence)
    )

    return data


class ProteinGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for protein structures.

    Reads JSONL files with protein structures and converts them to graphs.
    """

    def __init__(
        self,
        jsonl_file,
        k_neighbors=30,
        max_length=500,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            jsonl_file: path to JSONL file with protein structures
            k_neighbors: number of nearest neighbors for graph
            max_length: maximum sequence length
            transform: optional transform to apply to each graph
            pre_transform: optional pre-transform
        """
        super().__init__(None, transform, pre_transform)

        self.jsonl_file = jsonl_file
        self.k_neighbors = k_neighbors
        self.max_length = max_length
        self.featurizer = ProteinFeaturizer()

        # Load data
        self._load_data()

    def _load_data(self):
        """Load protein structures from JSONL file."""
        self.data_list = []
        self.names = []

        AA_ALPHABET = set('ACDEFGHIKLMNPQRSTVWY')

        with open(self.jsonl_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                seq = entry['seq']
                name = entry['name']

                # Filter by alphabet and length
                if set(seq).issubset(AA_ALPHABET) and len(seq) <= self.max_length:
                    # Convert coords to numpy arrays
                    coords = {k: np.array(v) for k, v in entry['coords'].items()}

                    self.data_list.append({
                        'coords': coords,
                        'seq': seq,
                        'name': name
                    })
                    self.names.append(name)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        """
        Get a protein graph.

        Returns:
            data: PyTorch Geometric Data object
        """
        item = self.data_list[idx]

        data = create_protein_graph(
            coords=item['coords'],
            sequence=item['seq'],
            k_neighbors=self.k_neighbors,
            featurizer=self.featurizer
        )

        # Add metadata
        data.name = item['name']

        return data


def batch_protein_graphs(data_list):
    """
    Custom batching function that handles variable-length proteins efficiently.

    Args:
        data_list: list of Data objects

    Returns:
        batch: Batched Data object
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(data_list)
