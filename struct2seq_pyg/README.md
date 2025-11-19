# Struct2Seq with PyTorch Geometric

A clean, modern reimplementation of **Struct2Seq** using PyTorch Geometric.

## Overview

Struct2Seq is a graph neural network model for protein sequence design. Given a protein backbone structure (3D coordinates), it generates amino acid sequences that are likely to fold into that structure.

This refactored version provides:
- ✅ **Modern PyTorch Geometric integration** for efficient graph operations
- ✅ **Cleaner, modular architecture** with well-documented code
- ✅ **Comprehensive tutorial notebook** for easy learning
- ✅ **Flexible design** supporting both GAT and MPNN layers

## Installation

### 1. Install PyTorch

First, install PyTorch following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

```bash
# Example for CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### 2. Install PyTorch Geometric

```bash
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

Replace `${TORCH}` with your PyTorch version (e.g., `1.13.0`) and `${CUDA}` with your CUDA version (e.g., `cu117`).

### 3. Install Additional Dependencies

```bash
pip install -r requirements_pyg.txt
```

## Quick Start

### Loading Data

```python
from struct2seq_pyg import ProteinGraphDataset

# Load dataset from JSONL file
dataset = ProteinGraphDataset(
    jsonl_file='data/proteins.jsonl',
    k_neighbors=30,
    max_length=500
)

# Create data loader
from torch_geometric.loader import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### Creating a Model

```python
from struct2seq_pyg import Struct2SeqPyG

model = Struct2SeqPyG(
    node_feature_dim=6,       # Dihedral angles
    edge_feature_dim=39,      # Pos encodings + RBF + orientations
    hidden_dim=128,
    num_encoder_layers=3,
    num_decoder_layers=3,
    num_letters=20,           # 20 amino acids
    num_heads=4,
    dropout=0.1,
    use_mpnn=False            # Use GAT (set True for MPNN)
)
```

### Training

```python
import torch.nn.functional as F

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for batch in loader:
        # Forward pass
        log_probs = model(batch)

        # Compute loss
        loss = F.nll_loss(log_probs, batch.seq)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Sampling Sequences

```python
# Generate sequences for a given structure
sampled_sequences = model.sample(data, temperature=1.0)
```

## Module Structure

```
struct2seq_pyg/
├── __init__.py          # Package initialization
├── data.py              # PyG dataset and graph creation
├── features.py          # Protein feature computation
├── layers.py            # GNN layers (GAT and MPNN)
├── model.py             # Main Struct2Seq model
└── README.md            # This file
```

### Key Components

- **`ProteinFeaturizer`**: Computes structural features (dihedrals, RBF, orientations)
- **`create_protein_graph()`**: Converts protein structure to PyG Data object
- **`ProteinGraphDataset`**: PyG dataset for loading protein structures
- **`ProteinGATLayer`**: Graph attention layer with edge features
- **`ProteinMPNNLayer`**: Message passing layer
- **`Struct2SeqPyG`**: Main encoder-decoder model

## Tutorial

See [`tutorial_struct2seq_pyg.ipynb`](../tutorial_struct2seq_pyg.ipynb) for a comprehensive walkthrough covering:

1. Data format and loading
2. Graph construction
3. Model architecture
4. Training loop
5. Sequence sampling
6. Evaluation metrics

## Architecture Details

### Encoder (Structure Processing)

```
Input: Protein backbone structure (N, CA, C, O coordinates)
  ↓
k-NN Graph Construction (k=30 neighbors)
  ↓
Node Features: Dihedral angles (φ, ψ, ω) as sin/cos pairs [6D]
Edge Features: Positional encodings + RBF distances + Orientations [39D]
  ↓
Graph Attention Layers × 3 (unmasked)
  ↓
Output: Structure encoding
```

### Decoder (Sequence Generation)

```
Input: Structure encoding + Sequence embeddings
  ↓
Graph Attention Layers × 3 (autoregressive masking)
  ↓
Output Projection
  ↓
Output: Amino acid log probabilities
```

### Key Features

- **Multi-head attention** over k-NN neighbors
- **Rich edge features** encoding spatial and sequential relationships
- **Autoregressive decoding** for sequence generation
- **Layer normalization** and **residual connections**

## Data Format

Proteins are stored in JSONL format:

```json
{
    "name": "protein_id",
    "seq": "ACDEFGHIKLMNPQRSTVWY...",
    "coords": {
        "N": [[x, y, z], ...],
        "CA": [[x, y, z], ...],
        "C": [[x, y, z], ...],
        "O": [[x, y, z], ...]
    }
}
```

## Comparison with Original Implementation

| Feature | Original | PyG Version |
|---------|----------|-------------|
| Graph library | Custom | PyTorch Geometric |
| Code structure | Single file | Modular (4 files) |
| Batching | Custom clustering | PyG DataLoader |
| Message passing | Manual gather/scatter | MessagePassing API |
| Documentation | Minimal | Comprehensive |
| Tutorial | None | Jupyter notebook |

## Performance Notes

- The PyG version uses efficient sparse operations for message passing
- Automatic GPU acceleration for graph operations
- Dynamic batching handles variable-length proteins efficiently
- Memory usage scales well with protein size

## Advanced Usage

### Custom Featurization

```python
from struct2seq_pyg import ProteinFeaturizer

# Customize features
featurizer = ProteinFeaturizer(
    num_positional_embeddings=32,  # Increase positional encodings
    num_rbf=32,                     # More RBF basis functions
    features_type='full'
)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in loader:
    with autocast():
        log_probs = model(batch)
        loss = F.nll_loss(log_probs, batch.seq)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Multi-GPU Training

```python
from torch.nn.parallel import DataParallel

model = DataParallel(Struct2SeqPyG(...))
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{ingraham2019generative,
  title={Generative models for graph-based protein design},
  author={Ingraham, John and Garg, Vikas and Barzilay, Regina and Jaakkola, Tommi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

## License

This implementation follows the license of the original repository.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

- Original paper: [Generative Models for Graph-Based Protein Design (NeurIPS 2019)](https://papers.nips.cc/paper/2019/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html)
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Original implementation: https://github.com/jingraham/neurips19-graph-protein-design
