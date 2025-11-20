# Phase 1: Data & Geometry Foundation

This document describes the Phase 1 implementation of the protein design project, focusing on data loading and geometric transformations.

## Overview

Phase 1 implements the foundational components for protein structure processing:

1. **Data Loading**: Load protein structures from BinaryCIF, mmCIF, or PDB formats
2. **Geometry Library**: Compute local coordinate frames and geometric features
3. **Feature Extraction**: Extract neighbor relationships and relative positions/orientations

## Files Created

### Core Modules

```
models/
├── __init__.py           # Package exports
├── geometry.py           # Geometric transformations (510 lines)
└── data_loader.py        # Data loading utilities (445 lines)

tests/
├── __init__.py
└── test_geometry.py      # Comprehensive unit tests (330 lines)

examples/
└── phase1_demo.py        # Usage demonstrations (280 lines)
```

### Documentation

- `PHASE1_README.md` - This file
- Updated `requirements.txt` - Added gemmi and biopython

## Module Documentation

### `models/geometry.py`

Implements critical geometric operations for protein structures.

#### Key Functions

**`get_local_frames(X: Tensor) -> Tuple[Tensor, Tensor]`**
- Constructs local coordinate frames using Gram-Schmidt orthogonalization
- Input: Backbone coordinates `[Batch, Length, 4, 3]`
- Output: Rotation matrices `R [Batch, Length, 3, 3]`, translations `t [Batch, Length, 3]`
- Local frame origin: CA (alpha carbon) position
- Axes defined by N-CA and C-CA vectors

**`get_neighbor_features(X, R, t, mask, k_neighbors) -> Tuple[...]`**
- Finds k-nearest neighbors based on CA-CA distance
- Computes relative positions in local frames
- Computes relative orientations between residues
- Returns: `rel_pos`, `rel_orient`, `distances`, `neighbor_idx`

**`get_rbf_features(distances, D_min, D_max, D_count) -> Tensor`**
- Radial Basis Function encoding of distances
- Provides smooth, differentiable distance representation
- Default: 16 Gaussian kernels spanning 0-20 Angstroms

**`compute_dihedral_angles(X) -> Tensor`**
- Computes backbone dihedral angles (phi, psi, omega)
- Returns (cos, sin) pairs to avoid discontinuities
- Output: `[Batch, Length, 6]` features

**`rotation_matrix_to_quaternion(R) -> Tensor`**
- Converts rotation matrices to quaternion representation
- Useful for compact orientation encoding

### `models/data_loader.py`

Provides utilities for loading and processing protein structures.

#### Key Classes

**`ProteinStructure`** (dataclass)
- Container for protein structure data
- Fields: `name`, `sequence`, `coords`, `mask`, `chain_id`
- Automatic validation of data consistency

**`ProteinDataset`** (PyTorch Dataset)
- Loads protein structures from file paths
- Supports batching with automatic padding
- Configurable max sequence length
- Compatible with PyTorch DataLoader

#### Key Functions

**`load_structure_from_gemmi(path, chain_id) -> ProteinStructure`**
- Load from BinaryCIF, mmCIF, or PDB using gemmi
- Extracts backbone atoms: N, CA, C, O
- Handles missing atoms and invalid residues
- Returns structured data ready for model input

**`load_structure_from_biopython(path, chain_id) -> ProteinStructure`**
- Alternative loader using BioPython
- PDB format only
- Useful when gemmi is not available

**`collate_protein_batch(batch) -> Dict`**
- Collate function for DataLoader
- Handles variable-length sequences
- Proper padding and masking

## Usage Examples

### Basic Geometry Computation

```python
import torch
from models import get_local_frames, get_neighbor_features

# Load or create protein coordinates
X = torch.randn(2, 100, 4, 3)  # 2 proteins, 100 residues
mask = torch.ones(2, 100)

# Compute local frames
R, t = get_local_frames(X)
print(f"Rotation matrices: {R.shape}")  # [2, 100, 3, 3]
print(f"Translations: {t.shape}")       # [2, 100, 3]

# Compute neighbor features
rel_pos, rel_orient, distances, idx = get_neighbor_features(
    X, R, t, mask, k_neighbors=30
)
print(f"Relative positions: {rel_pos.shape}")  # [2, 100, 30, 3]
```

### Loading Protein Structures

```python
from pathlib import Path
from models import load_structure_from_gemmi, ProteinDataset
from torch.utils.data import DataLoader

# Load single structure
protein = load_structure_from_gemmi(
    Path('structures/1abc.bcif'),
    chain_id='A'
)
print(f"Loaded {protein.name}: {len(protein.sequence)} residues")

# Load dataset
paths = [Path(f'structures/protein_{i}.pdb') for i in range(100)]
dataset = ProteinDataset(paths, max_length=500)
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_protein_batch)

for batch in loader:
    coords = batch['coords']      # [8, MaxLen, 4, 3]
    sequence = batch['sequence']  # [8, MaxLen]
    mask = batch['mask']          # [8, MaxLen]
    # Process batch...
```

### Complete Feature Pipeline

```python
from models import (
    get_local_frames,
    get_neighbor_features,
    get_rbf_features,
    compute_dihedral_angles,
)

def extract_features(X, mask, k_neighbors=30):
    """Extract all geometric features for model input."""
    # Local coordinate frames
    R, t = get_local_frames(X)

    # Neighbor features
    rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
        X, R, t, mask, k_neighbors=k_neighbors
    )

    # Distance encoding
    rbf = get_rbf_features(distances)

    # Backbone angles
    dihedrals = compute_dihedral_angles(X)

    return {
        'local_frames': (R, t),
        'neighbors': {
            'rel_pos': rel_pos,
            'rel_orient': rel_orient,
            'distances': distances,
            'indices': neighbor_idx,
            'rbf': rbf,
        },
        'dihedrals': dihedrals,
    }

# Use in model
features = extract_features(X, mask)
```

## Running Tests

```bash
# Run all geometry tests
pytest tests/test_geometry.py -v

# Run specific test class
pytest tests/test_geometry.py::TestLocalFrames -v

# Run with coverage
pytest tests/test_geometry.py --cov=models --cov-report=html
```

## Running Examples

```bash
# Run Phase 1 demonstration
python examples/phase1_demo.py
```

This will show:
1. Synthetic protein geometry computation
2. Complete feature extraction pipeline
3. Handling of missing residues with masking

## Engineering Standards Applied

All Phase 1 code follows the mandatory engineering standards:

✅ **Type Hints**: Complete type annotations on all functions
✅ **Docstrings**: Google-style docstrings with Args, Returns, and Shape info
✅ **Path Handling**: Exclusive use of `pathlib.Path`
✅ **Logging**: `logging` module instead of `print()`
✅ **No Hardcoding**: All hyperparameters configurable
✅ **Error Handling**: Comprehensive validation and error messages

## Dependencies

New dependencies added in Phase 1:

```
gemmi>=0.6.0        # BinaryCIF, mmCIF, PDB loading
biopython>=1.79     # Alternative PDB parser
```

Install with:
```bash
pip install -r requirements.txt
```

## Key Insights

### Geometric Transformations

1. **Local Frames**: Each residue has its own coordinate system
   - Origin: CA position
   - X-axis: CA → C direction
   - Y-axis: Gram-Schmidt of CA → N
   - Z-axis: Cross product

2. **Neighbor Features**:
   - Relative positions are normalized to unit vectors
   - First neighbor is always self (distance ~0)
   - Neighbors sorted by distance

3. **RBF Encoding**:
   - Smooth representation of distances
   - 16 Gaussian kernels by default
   - Covers 0-20 Angstrom range

### Data Format

Input tensor format: `[Batch, Length, 4, 3]`
- Dimension 0: Batch
- Dimension 1: Sequence length
- Dimension 2: Atoms [N, CA, C, O]
- Dimension 3: Coordinates [x, y, z]

Output features maintain batch and length dimensions:
- Node features: `[Batch, Length, Features]`
- Edge features: `[Batch, Length, K, Features]`

## Next Steps (Phase 2)

With Phase 1 complete, we're ready for:

1. **Model Architecture**: Build graph neural network using geometric features
2. **Attention Mechanisms**: Implement structure-aware attention
3. **Training Loop**: Set up training with W&B tracking
4. **Evaluation**: Metrics for sequence recovery and perplexity

## Performance Considerations

- All operations are batched and vectorized
- GPU compatible (tested with CUDA)
- Efficient k-NN via PyTorch's topk
- Minimal memory footprint with proper masking

## Testing Coverage

Comprehensive tests cover:
- Shape correctness
- Mathematical properties (orthonormality, unit vectors)
- Edge cases (masking, batch independence)
- GPU compatibility
- Integration tests for complete pipeline

All tests pass with numerical tolerance of 1e-5.
