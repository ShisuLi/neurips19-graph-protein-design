"""PyTorch Geometric data conversion for protein structures.

This module converts dense tensor representations to PyG sparse format
for memory-efficient processing of long protein sequences.
"""

import torch
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Optional
import numpy as np

from utils import get_logger
from .geometry import (
    get_local_frames,
    get_neighbor_features,
    get_rbf_features,
    compute_dihedral_angles,
)

logger = get_logger(__name__)


def to_pyg_data(
    X: torch.Tensor,
    S: torch.Tensor,
    mask: torch.Tensor,
    k_neighbors: int = 30
) -> Data:
    """Convert protein to PyTorch Geometric Data object.

    This function converts dense tensor representations to sparse PyG format,
    enabling memory-efficient processing of long proteins.

    Args:
        X: Backbone coordinates with shape [Length, 4, 3].
           The 4 atoms are [N, CA, C, O].
        S: Amino acid sequence with shape [Length].
           Contains amino acid indices (0-19).
        mask: Valid residue mask with shape [Length].
              1 for valid residues, 0 for padding/missing.
        k_neighbors: Number of nearest neighbors per residue.

    Returns:
        PyG Data object containing:
            - x: Node features [Num_Nodes, node_dim]
            - edge_index: Sparse edges [2, Num_Edges]
            - edge_attr: Edge features [Num_Edges, edge_dim]
            - y: Target sequence [Num_Nodes]
            - pos: CA positions [Num_Nodes, 3]
            - mask: Validity mask [Num_Nodes]

    Shape:
        - Input X: [Length, 4, 3]
        - Input S: [Length]
        - Input mask: [Length]
        - Output x: [Num_Valid_Nodes, node_dim]
        - Output edge_index: [2, Num_Edges]
        - Output edge_attr: [Num_Edges, edge_dim]

    Example:
        >>> X = torch.randn(100, 4, 3)
        >>> S = torch.randint(0, 20, (100,))
        >>> mask = torch.ones(100)
        >>> data = to_pyg_data(X, S, mask, k_neighbors=30)
        >>> print(data.num_nodes, data.num_edges)
    """
    # Add batch dimension for geometry functions
    X_batched = X.unsqueeze(0)  # [1, L, 4, 3]
    mask_batched = mask.unsqueeze(0)  # [1, L]

    # Extract geometric features using Phase 1 functions
    R, t = get_local_frames(X_batched)
    rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
        X_batched, R, t, mask_batched, k_neighbors=k_neighbors
    )
    rbf = get_rbf_features(distances)
    dihedrals = compute_dihedral_angles(X_batched)

    # Remove batch dimension
    neighbor_idx = neighbor_idx[0]  # [L, K]
    rel_pos = rel_pos[0]  # [L, K, 3]
    rel_orient = rel_orient[0]  # [L, K, 3]
    rbf = rbf[0]  # [L, K, 16]
    dihedrals = dihedrals[0]  # [L, 6]

    # Filter out invalid nodes
    valid_mask = mask.bool()
    num_valid = valid_mask.sum().item()

    if num_valid == 0:
        raise ValueError("No valid residues in protein")

    # Create node index mapping (old_idx -> new_idx)
    old_to_new = torch.full((len(mask),), -1, dtype=torch.long)
    old_to_new[valid_mask] = torch.arange(num_valid)

    # Node features: use dihedrals for valid nodes
    node_features = dihedrals[valid_mask]  # [Num_Valid, 6]

    # Target sequence
    target_seq = S[valid_mask]  # [Num_Valid]

    # CA positions for visualization/analysis
    ca_positions = X[:, 1, :][valid_mask]  # [Num_Valid, 3]

    # Build sparse edge_index and edge_attr
    edge_list = []
    edge_attrs = []

    for i in range(len(mask)):
        if not valid_mask[i]:
            continue

        new_i = old_to_new[i].item()

        # Get neighbors for this node
        neighbors = neighbor_idx[i]  # [K]

        for k in range(k_neighbors):
            j = neighbors[k].item()

            # Check if neighbor is valid
            if j >= len(mask) or not valid_mask[j]:
                continue

            new_j = old_to_new[j].item()

            # Add edge i -> j
            edge_list.append([new_i, new_j])

            # Combine edge features: [rel_pos (3), rel_orient (3), rbf (16)]
            edge_feat = torch.cat([
                rel_pos[i, k],
                rel_orient[i, k],
                rbf[i, k]
            ])  # [22]
            edge_attrs.append(edge_feat)

    if len(edge_list) == 0:
        # Create self-loops if no edges
        logger.warning("No valid edges found, creating self-loops")
        for i in range(num_valid):
            edge_list.append([i, i])
            edge_attrs.append(torch.zeros(22))

    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2, E]
    edge_attr = torch.stack(edge_attrs)  # [E, 22]

    # Create PyG Data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=target_seq,
        pos=ca_positions,
        num_nodes=num_valid
    )

    logger.debug(
        f"Created PyG data: {num_valid} nodes, {edge_index.shape[1]} edges"
    )

    return data


def batch_to_pyg(
    coords_batch: torch.Tensor,
    sequence_batch: torch.Tensor,
    mask_batch: torch.Tensor,
    k_neighbors: int = 30
) -> Batch:
    """Convert batch of proteins to PyG Batch object.

    Args:
        coords_batch: Batch of coordinates [Batch, Length, 4, 3].
        sequence_batch: Batch of sequences [Batch, Length].
        mask_batch: Batch of masks [Batch, Length].
        k_neighbors: Number of neighbors per residue.

    Returns:
        PyG Batch object containing all proteins.

    Shape:
        - coords_batch: [Batch, Length, 4, 3]
        - sequence_batch: [Batch, Length]
        - mask_batch: [Batch, Length]

    Example:
        >>> coords = torch.randn(4, 100, 4, 3)
        >>> seqs = torch.randint(0, 20, (4, 100))
        >>> masks = torch.ones(4, 100)
        >>> batch = batch_to_pyg(coords, seqs, masks, k_neighbors=30)
        >>> print(batch.num_graphs, batch.num_nodes)
    """
    batch_size = coords_batch.shape[0]

    # Convert each protein to PyG Data
    data_list = []
    for i in range(batch_size):
        try:
            data = to_pyg_data(
                coords_batch[i],
                sequence_batch[i],
                mask_batch[i],
                k_neighbors=k_neighbors
            )
            data_list.append(data)
        except ValueError as e:
            logger.warning(f"Skipping protein {i} in batch: {e}")
            continue

    if len(data_list) == 0:
        raise ValueError("All proteins in batch are invalid")

    # Batch into single PyG Batch object
    batch = Batch.from_data_list(data_list)

    logger.debug(
        f"Created PyG batch: {batch.num_graphs} graphs, "
        f"{batch.num_nodes} total nodes, {batch.num_edges} total edges"
    )

    return batch


def pyg_to_dense(
    batch: Batch,
    max_length: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert PyG Batch back to dense tensors.

    This is useful for interfacing with models that expect dense inputs.

    Args:
        batch: PyG Batch object.
        max_length: Maximum sequence length for padding. If None, uses
                   the longest sequence in the batch.

    Returns:
        Tuple of (node_features, sequence, mask) as dense tensors:
            - node_features: [Batch, MaxLength, node_dim]
            - sequence: [Batch, MaxLength]
            - mask: [Batch, MaxLength]

    Example:
        >>> # After processing with PyG model
        >>> node_feat, seq, mask = pyg_to_dense(batch)
        >>> print(node_feat.shape)  # [Batch, MaxLength, NodeDim]
    """
    # Get batch assignment
    batch_assign = batch.batch  # [Total_Nodes]
    num_graphs = batch.num_graphs

    # Find sequence lengths for each graph
    lengths = []
    for i in range(num_graphs):
        graph_mask = (batch_assign == i)
        lengths.append(graph_mask.sum().item())

    if max_length is None:
        max_length = max(lengths)

    # Initialize dense tensors
    node_dim = batch.x.shape[1]
    node_features = torch.zeros(num_graphs, max_length, node_dim, device=batch.x.device)
    sequence = torch.zeros(num_graphs, max_length, dtype=torch.long, device=batch.y.device)
    mask = torch.zeros(num_graphs, max_length, device=batch.x.device)

    # Fill in data for each graph
    node_idx = 0
    for i in range(num_graphs):
        length = lengths[i]

        # Get nodes for this graph
        graph_nodes = batch.x[node_idx:node_idx + length]
        graph_seq = batch.y[node_idx:node_idx + length]

        # Fill dense tensors
        node_features[i, :length] = graph_nodes
        sequence[i, :length] = graph_seq
        mask[i, :length] = 1.0

        node_idx += length

    return node_features, sequence, mask


def compute_pyg_edge_index_knn(
    pos: torch.Tensor,
    k: int,
    batch: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute k-nearest neighbor graph in PyG format.

    This is an alternative to using geometry functions, useful when you
    only have positions and want to build the graph directly.

    Args:
        pos: Node positions [Num_Nodes, 3].
        k: Number of nearest neighbors.
        batch: Batch assignment [Num_Nodes]. If None, assumes single graph.

    Returns:
        Edge index [2, Num_Edges].

    Example:
        >>> pos = torch.randn(100, 3)
        >>> edge_index = compute_pyg_edge_index_knn(pos, k=30)
        >>> print(edge_index.shape)  # [2, ~3000]
    """
    from torch_geometric.nn import knn_graph

    edge_index = knn_graph(
        pos,
        k=k,
        batch=batch,
        loop=True  # Include self-loops
    )

    return edge_index
