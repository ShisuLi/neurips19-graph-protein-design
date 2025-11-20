## Phase 2: The Encoder Laboratory

This document describes the Phase 2 implementation focusing on Graph Neural Network encoders for ablation studies of geometric information in protein design.

## Overview

Phase 2 implements three interchangeable encoder architectures to systematically study the impact of geometric features:

1. **VanillaGCNEncoder**: Baseline without geometric information
2. **EdgeAwareGATEncoder**: Intermediate with distance-based attention
3. **Struct2SeqGeometricEncoder**: Full geometric attention with local frames

All encoders follow a standardized interface, enabling direct comparison in ablation studies.

## Files Created

### Core Modules

```
models/
├── encoders.py           # All encoder implementations (650+ lines)
└── __init__.py          # Updated exports

tests/
└── test_encoders.py     # Comprehensive encoder tests (450+ lines)

examples/
└── phase2_demo.py       # Complete demonstrations (380+ lines)
```

### Documentation

- `PHASE2_README.md` - This file

## Architecture Details

### Base Interface

All encoders inherit from `BaseEncoder` and implement:

```python
def forward(
    h_nodes: Tensor,      # [Batch, Length, hidden_dim]
    h_edges: Tensor,      # [Batch, Length, K, edge_dim]
    edge_idxs: Tensor,    # [Batch, Length, K]
    mask: Tensor          # [Batch, Length]
) -> Tensor:              # [Batch, Length, hidden_dim]
```

This standardized interface enables:
- Easy encoder swapping for ablation studies
- Fair performance comparisons
- Consistent experiment setup

### 1. VanillaGCNEncoder (Baseline)

**Philosophy**: "Structure is just a connectivity graph."

**Key Characteristics**:
- Ignores edge features (h_edges parameter unused)
- Simple mean aggregation over neighbors
- MLP with residual connections
- Establishes lower-bound performance

**Architecture**:
```
For each layer:
  1. Gather neighbor features
  2. h_aggregated = mean(h_neighbors)
  3. h_new = MLP(h + h_aggregated)
  4. h = LayerNorm(h_new)
  5. h = h * mask
```

**Use Case**: Baseline to quantify the value of geometric information.

### 2. EdgeAwareGATEncoder (Intermediate)

**Philosophy**: "Distances matter, but direction doesn't."

**Key Characteristics**:
- Uses distance-based edge features (RBF encoding)
- Graph Attention Network with edge bias
- Edge features modulate attention scores
- More expressive than VanillaGCN

**Attention Mechanism**:
```
Q = W_q @ h_i
K = W_k @ h_j
V = W_v @ h_j

attention_score = (Q @ K^T) / sqrt(d) + edge_bias(h_edge)
attention_weights = softmax(attention_score)
output = attention_weights @ V
```

**Use Case**: Test whether distance information alone is sufficient.

### 3. Struct2SeqGeometricEncoder (Target)

**Philosophy**: "Full 3D geometry (local frames) is essential."

**Key Characteristics**:
- Custom `GeometricAttentionLayer`
- Injects geometric features into keys and values
- Uses relative positions and orientations from Phase 1
- Most expressive architecture

**Geometric Attention Mechanism**:
```
# Edge features include: relative positions, orientations, RBF distances
h_edge_proj = W_edge @ h_edges

# Inject geometry into neighbor representations
h_neighbors_geometric = h_neighbors + h_edge_proj

# Standard attention with geometric information
Q = W_q @ h_i
K = W_k @ h_neighbors_geometric
V = W_v @ h_neighbors_geometric

attention_score = (Q @ K^T) / sqrt(d)
attention_weights = softmax(attention_score)
output = attention_weights @ V
```

**Architecture**:
- Multiple geometric attention layers
- Residual connections
- Layer normalization
- Feed-forward networks (2x hidden_dim)

**Use Case**: Target architecture for protein design.

## Custom Geometric Attention Layer

The `GeometricAttentionLayer` is the core innovation:

**Key Features**:
1. **Edge Feature Injection**: Projects edge features and adds to neighbor representations
2. **Multi-head Attention**: Supports multiple attention heads
3. **Proper Masking**: Handles invalid residues in both source and targets
4. **Residual-Compatible**: Maintains input/output dimension

**Implementation Details**:
```python
class GeometricAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_heads=4, dropout=0.1):
        # Project edge features to hidden dimension
        self.edge_projection = nn.Linear(edge_dim, hidden_dim)

        # Q, K, V projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
```

**Why Custom Implementation?**

PyTorch's `nn.MultiheadAttention` doesn't support:
- Injecting features into keys/values
- Per-edge feature conditioning
- Custom masking for graph attention

## Probing Head for Validation

The `StructureProbingHead` enables validation before building the full decoder:

**Purpose**: Verify that encoders learn meaningful representations

**Architecture**:
```
1. Global pooling (mean/max/sum) over sequence
2. MLP classifier: hidden_dim -> hidden_dim//2 -> num_classes
3. Returns logits for classification
```

**Example Tasks**:
- Length classification (short vs long proteins)
- Fold classification (alpha, beta, mixed)
- Stability prediction

**Usage**:
```python
encoder = Struct2SeqGeometricEncoder(128, 64)
head = StructureProbingHead(128, num_classes=2)

# Forward pass
encoded = encoder(h_nodes, h_edges, edge_idxs, mask)
logits = head(encoded, mask)

# Training
loss = F.cross_entropy(logits, labels)
loss.backward()  # Gradients flow through encoder
```

## Usage Examples

### Basic Encoder Usage

```python
from models import Struct2SeqGeometricEncoder

# Create encoder
encoder = Struct2SeqGeometricEncoder(
    hidden_dim=128,
    edge_dim=22,      # From Phase 1: 3+3+16
    num_layers=3,
    num_heads=4,
    dropout=0.1
)

# Prepare inputs
h_nodes = ...         # [Batch, Length, 128] Initial node features
h_edges = ...         # [Batch, Length, 30, 22] Geometric features
edge_idxs = ...       # [Batch, Length, 30] Neighbor indices
mask = ...            # [Batch, Length] Valid residues

# Encode
encoded = encoder(h_nodes, h_edges, edge_idxs, mask)
# encoded: [Batch, Length, 128]
```

### Complete Pipeline (Phase 1 + Phase 2)

```python
import torch
from models import (
    # Phase 1
    get_local_frames,
    get_rbf_features,
    get_neighbor_features,
    compute_dihedral_angles,
    # Phase 2
    Struct2SeqGeometricEncoder,
)

# Input: protein coordinates
X = ...  # [Batch, Length, 4, 3]
mask = ...  # [Batch, Length]

# Phase 1: Extract geometry
R, t = get_local_frames(X)
rel_pos, rel_orient, distances, neighbor_idx = get_neighbor_features(
    X, R, t, mask, k_neighbors=30
)
rbf = get_rbf_features(distances)
dihedrals = compute_dihedral_angles(X)

# Combine edge features: [rel_pos, rel_orient, rbf]
edge_features = torch.cat([rel_pos, rel_orient, rbf], dim=-1)
# edge_features: [Batch, Length, 30, 22]

# Initial node embeddings (from dihedral angles or amino acid type)
h_nodes = embedding_layer(dihedrals)  # [Batch, Length, hidden_dim]

# Phase 2: Encode with geometric attention
encoder = Struct2SeqGeometricEncoder(128, 22, num_layers=3)
encoded = encoder(h_nodes, edge_features, neighbor_idx, mask)
```

### Ablation Study Setup

```python
# Define encoders for comparison
encoders = {
    'Baseline': VanillaGCNEncoder(128, 22, num_layers=3),
    'Distances': EdgeAwareGATEncoder(128, 22, num_layers=3),
    'FullGeometry': Struct2SeqGeometricEncoder(128, 22, num_layers=3),
}

# Train each encoder on same task
for name, encoder in encoders.items():
    # Training loop
    for batch in dataloader:
        # All encoders use same interface
        encoded = encoder(
            batch['h_nodes'],
            batch['h_edges'],
            batch['edge_idxs'],
            batch['mask']
        )

        # Task-specific head (e.g., sequence recovery)
        logits = decoder(encoded)
        loss = criterion(logits, batch['target'])

        # Optimize
        loss.backward()
        optimizer.step()

    # Evaluate and compare
    results[name] = evaluate(encoder, test_loader)

# Expected: Baseline < Distances < FullGeometry
```

## Running Tests

```bash
# Run all encoder tests
pytest tests/test_encoders.py -v

# Run specific test class
pytest tests/test_encoders.py::TestGeometricAttentionLayer -v

# Run with coverage
pytest tests/test_encoders.py --cov=models.encoders --cov-report=html
```

## Running Examples

```bash
# Run Phase 2 demonstration
python examples/phase2_demo.py
```

This will show:
1. Encoder architecture comparison
2. Geometric attention mechanism in action
3. Probing task for validation
4. Ablation study setup

## Engineering Standards Applied

All Phase 2 code follows mandatory standards:

✅ **Type Hints**: Complete annotations on all functions
✅ **Docstrings**: Google-style with Args, Returns, Shape info
✅ **Logging**: Using `logging` module, no `print()`
✅ **No Hardcoding**: All hyperparameters configurable
✅ **Abstract Base Class**: Proper inheritance hierarchy
✅ **Residual Connections**: For gradient flow
✅ **Layer Normalization**: For training stability

## Key Design Decisions

### 1. Edge Feature Injection

**Decision**: Add edge features to neighbor representations before K/V projection

**Rationale**:
- Allows geometric information to influence attention
- More flexible than just using as bias
- Matches Struct2Seq paper implementation

**Alternative considered**: Use edge features only as attention bias
- Simpler but less expressive
- Doesn't allow geometry to modify value representations

### 2. Multi-Head Attention

**Decision**: Split attention into multiple heads

**Rationale**:
- Different heads can attend to different aspects (structure, sequence, etc.)
- Standard practice in transformers
- Improves representation capacity

**Configuration**: 4 heads by default (hidden_dim=128 → 32 per head)

### 3. Residual Connections

**Decision**: Use residual connections around attention and FFN

**Rationale**:
- Essential for training deep networks
- Enables gradient flow through many layers
- Allows identity mapping if needed

**Implementation**: `h = LayerNorm(h + attention(h))`

### 4. Masking Strategy

**Decision**: Apply mask at multiple points
- Invalid attention sources: Set to -inf before softmax
- Invalid attention targets: Set to -inf before softmax
- Output masking: Multiply by mask to ensure zero

**Rationale**: Comprehensive masking prevents information leakage

## Ablation Study Hypothesis

**Expected Performance Ranking**:
```
VanillaGCN < EdgeAwareGAT < Struct2SeqGeometric
```

**Why?**

1. **VanillaGCN** (Baseline)
   - Only knows connectivity graph
   - No distance or orientation information
   - Treats all neighbors equally
   - Lower bound on performance

2. **EdgeAwareGAT** (+ Distances)
   - Knows which residues are near/far
   - Can weight nearby residues more
   - Still lacks directional information
   - Should improve over baseline

3. **Struct2SeqGeometric** (+ Full Geometry)
   - Knows relative positions in 3D
   - Knows relative orientations
   - Can leverage local coordinate frames
   - Should achieve best performance

**Validation**: Train all three on sequence recovery task and compare perplexity.

## Performance Considerations

**Parameter Counts** (hidden_dim=128, edge_dim=22, 3 layers):
- VanillaGCN: ~150K parameters
- EdgeAwareGAT: ~200K parameters
- Struct2SeqGeometric: ~250K parameters

**Memory**:
- All encoders: O(B × L × K × H) where:
  - B = batch size
  - L = sequence length
  - K = neighbors per residue
  - H = hidden dimension

**Computation**:
- VanillaGCN: O(B × L × K × H²) - mean aggregation + MLP
- EdgeAwareGAT: O(B × L × K × H²) - attention over neighbors
- Struct2SeqGeometric: O(B × L × K × H²) - same, but with edge projection

**GPU Utilization**:
- All operations fully batched
- Efficient gather operations for neighbor features
- Multi-head attention parallelizable

## Testing Coverage

Comprehensive tests cover:

**GeometricAttentionLayer**:
- Output shape correctness
- Masking behavior
- Batch independence
- Multiple head configurations

**All Encoders**:
- Interface compliance
- Gradient flow
- Residual connections
- Masking application
- Different layer counts

**Comparison Tests**:
- VanillaGCN ignores edge features (verified)
- EdgeAware/Geometric use edge features (verified)
- All follow same interface (verified)

**Integration Tests**:
- End-to-end: encoder + probing head
- Backward pass
- Loss computation

All tests pass with numerical tolerance of 1e-5.

## Next Steps (Phase 3)

With Phase 2 complete, ready for:

1. **Autoregressive Decoder**: Design sequences conditioned on structure
2. **Training Loop**: Implement with W&B tracking
3. **Evaluation Metrics**: Sequence recovery, perplexity, diversity
4. **Ablation Experiments**: Quantify geometric information value

## Common Issues & Solutions

**Issue**: Attention produces NaN values

**Solution**: Check that:
- Input features are normalized
- Mask is applied before softmax
- Scale factor (1/sqrt(d)) is used

**Issue**: Encoder ignores edge features

**Solution**: Verify:
- Edge projection is not identity
- Projected edges are actually added to neighbors
- Gradients flow through edge_projection layer

**Issue**: Out of memory

**Solution**:
- Reduce k_neighbors (30 → 20)
- Reduce hidden_dim (128 → 64)
- Use gradient checkpointing
- Reduce batch size

## References

- Ingraham et al. (2019): "Generative Models for Graph-Based Protein Design"
- Vaswani et al. (2017): "Attention Is All You Need"
- Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
- Veličković et al. (2018): "Graph Attention Networks"
