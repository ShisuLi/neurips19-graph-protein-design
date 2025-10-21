# Complete Tutorial: Graph-Based Protein Design

A comprehensive guide to using the Struct2Seq model for graph-based protein design based on the NeurIPS 2019 paper "Generative Models for Graph-Based Protein Design" by John Ingraham, Vikas Garg, Regina Barzilay, and Tommi Jaakkola.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Data Preparation](#data-preparation)
6. [Understanding the Model](#understanding-the-model)
7. [Training](#training)
8. [Testing and Evaluation](#testing-and-evaluation)
9. [Protein Redesign](#protein-redesign)
10. [Advanced Features](#advanced-features)
11. [Troubleshooting](#troubleshooting)
12. [Examples](#examples)

---

## Introduction

### What is Struct2Seq?

Struct2Seq is a deep learning model that designs protein sequences for target 3D structures. Unlike traditional methods, it uses a graph-conditioned autoregressive language model to predict amino acid sequences that would fold into a given protein backbone structure.

### Key Features

- **Graph-based encoding**: Protein structures are represented as graphs where nodes are residues and edges encode spatial relationships
- **Autoregressive generation**: Sequences are generated one residue at a time, conditioned on the structure and previously generated residues
- **Transformer architecture**: Uses self-attention mechanisms to capture long-range dependencies
- **Multiple feature types**: Supports various structural feature encodings (full, distance, hydrogen bonds, coarse)

### Applications

- De novo protein design
- Protein redesign and optimization
- Understanding sequence-structure relationships
- Generating diverse sequences for fixed backbones

---

## Installation

### Requirements

```bash
Python >= 3.0
PyTorch >= 1.0
NumPy
matplotlib (for visualization)
```

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/jingraham/neurips19-graph-protein-design.git
cd neurips19-graph-protein-design
```

2. **Set up Python environment** (recommended)
```bash
# Using conda
conda create -n struct2seq python=3.7
conda activate struct2seq

# Or using virtualenv
python3 -m venv struct2seq_env
source struct2seq_env/bin/activate
```

3. **Install dependencies**
```bash
pip install torch numpy matplotlib
```

4. **Verify installation**
```bash
cd experiments
python3 -c "import sys; sys.path.insert(0, '..'); from struct2seq import *; print('Installation successful!')"
```

---

## Project Structure

```
neurips19-graph-protein-design/
├── struct2seq/                    # Core model implementation
│   ├── __init__.py               # Package initialization
│   ├── struct2seq.py             # Main Struct2Seq model
│   ├── self_attention.py         # Transformer/attention layers
│   ├── protein_features.py       # Protein structure featurization
│   ├── data.py                   # Dataset and data loading
│   ├── seq_model.py              # Sequence-only baseline models
│   └── noam_opt.py               # Noam learning rate scheduler
│
├── experiments/                   # Training and testing scripts
│   ├── train_s2s.py              # Main training script
│   ├── test_s2s.py               # Perplexity testing
│   ├── test_redesign.py          # Sequence redesign testing
│   ├── test_rocklin_mutations.py # Mutation analysis
│   ├── seq_only_train.py         # Baseline model training
│   ├── seq_only_test.py          # Baseline model testing
│   ├── utils.py                  # Helper functions
│   ├── run_training.sh           # Training examples
│   ├── run_test.sh               # Testing examples
│   └── run_redesign.sh           # Redesign examples
│
├── data/                         # Data processing scripts
│   ├── build_chain_dataset.py    # Build CATH dataset
│   ├── mmtf_util.py              # MMTF file parsing utilities
│   ├── SPIN2/                    # SPIN2 benchmark splits
│   └── ollikainen/               # Ollikainen benchmark
│
├── README.md                     # Basic project information
└── LICENSE                       # MIT License
```

### Key Components

- **struct2seq.py**: The main model architecture implementing graph-conditioned sequence generation
- **protein_features.py**: Converts 3D protein structures into graph features (node and edge features)
- **data.py**: Handles loading and batching of protein structure data
- **utils.py**: Command-line argument parsing, model setup, loss functions

---

## Quick Start

### Download Pre-processed Data

```bash
cd data
# Download from http://people.csail.mit.edu/ingraham/graph-protein-design/data/
# Example files:
# - chain_set.jsonl (preprocessed CATH structures)
# - chain_set_splits.json (train/validation/test splits)
```

### Train a Simple Model

```bash
cd experiments

# CPU training (small dataset for testing)
python3 train_s2s.py \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --name my_first_model

# GPU training (recommended)
python3 train_s2s.py --cuda \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --batch_tokens 6000 \
    --name my_first_model
```

### Test a Trained Model

```bash
python3 test_s2s.py --cuda \
    --restore log/my_first_model/best_checkpoint_epoch*.pt \
    --file_splits ../data/chain_set_splits.json
```

---

## Data Preparation

### Understanding the Data Format

Protein structures are stored in JSONL format (one JSON object per line). Each entry contains:

```json
{
  "name": "1abc.A",
  "seq": "ACDEFGHIKLMNPQRSTVWY...",
  "coords": {
    "N": [[x1,y1,z1], [x2,y2,z2], ...],
    "CA": [[x1,y1,z1], [x2,y2,z2], ...],
    "C": [[x1,y1,z1], [x2,y2,z2], ...],
    "O": [[x1,y1,z1], [x2,y2,z2], ...]
  },
  "num_chains": 1,
  "CATH": ["3.40.50.720"]
}
```

**Fields:**
- `name`: Unique identifier (PDB ID + chain)
- `seq`: Amino acid sequence (one-letter codes)
- `coords`: 3D coordinates for backbone atoms (N, CA, C, O)
- `num_chains`: Number of chains in the structure
- `CATH`: CATH topology classification

### Building a Dataset from CATH

The repository includes a script to build datasets from the CATH database:

```bash
cd data
python3 build_chain_dataset.py
```

**This script will:**
1. Download CATH non-redundant set (40% sequence identity)
2. Download CATH domain classifications
3. Parse protein structures from MMTF files
4. Filter by length (max 500 residues)
5. Generate `chain_set.jsonl`

### Creating Train/Validation/Test Splits

Splits are defined in a JSON file:

```json
{
  "train": ["1abc.A", "1def.B", ...],
  "validation": ["2ghi.C", "2jkl.D", ...],
  "test": ["3mno.E", "3pqr.F", ...]
}
```

**Example: Create custom splits**

```python
import json
import random

# Load dataset
with open('chain_set.jsonl') as f:
    chains = [json.loads(line)['name'] for line in f]

# Shuffle and split (80/10/10)
random.shuffle(chains)
n = len(chains)
splits = {
    'train': chains[:int(0.8*n)],
    'validation': chains[int(0.8*n):int(0.9*n)],
    'test': chains[int(0.9*n):]
}

# Save splits
with open('my_splits.json', 'w') as f:
    json.dump(splits, f, indent=2)
```

### Preparing Your Own Structures

To use your own protein structures:

```python
import json

def create_data_entry(pdb_file, chain_id, name):
    """
    Parse a PDB file and create a data entry

    Args:
        pdb_file: Path to PDB file
        chain_id: Chain identifier
        name: Unique name for this structure

    Returns:
        Dictionary with required fields
    """
    # Parse PDB file (you'll need to implement this)
    coords_N, coords_CA, coords_C, coords_O = parse_pdb(pdb_file, chain_id)
    sequence = extract_sequence(pdb_file, chain_id)

    return {
        'name': name,
        'seq': sequence,
        'coords': {
            'N': coords_N.tolist(),
            'CA': coords_CA.tolist(),
            'C': coords_C.tolist(),
            'O': coords_O.tolist()
        },
        'num_chains': 1
    }

# Create dataset
entries = [
    create_data_entry('protein1.pdb', 'A', 'prot1.A'),
    create_data_entry('protein2.pdb', 'A', 'prot2.A'),
]

# Save to JSONL
with open('my_dataset.jsonl', 'w') as f:
    for entry in entries:
        f.write(json.dumps(entry) + '\n')
```

---

## Understanding the Model

### Model Architecture

The Struct2Seq model consists of three main components:

```
Input Structure → Encoder → Decoder → Output Sequence
```

#### 1. Featurization (protein_features.py:28-32)

Converts 3D structure to graph representation:
- **Nodes**: One per residue, with features derived from backbone geometry
- **Edges**: k-nearest neighbors (default k=30) based on Cα distance
- **Node features**: Position, orientation, dihedral angles
- **Edge features**: Distance, relative orientation, angles

#### 2. Encoder Layers

Unmasked graph neural network that processes the entire structure:
```
For each encoder layer:
    1. Aggregate information from neighboring nodes
    2. Self-attention over the graph
    3. Feed-forward transformation
```

#### 3. Decoder Layers

Autoregressive generation with masked attention:
```
For each position i:
    1. Attend to structure (encoder output)
    2. Attend to previously generated sequence (positions < i)
    3. Predict amino acid at position i
```

### Feature Types

The model supports different structural feature sets (controlled by `--features` flag):

| Feature Type | Description | Use Case |
|-------------|-------------|----------|
| `full` | Complete geometric features | Best performance, default |
| `dist` | Distance-based features only | Simpler, faster |
| `hbonds` | Includes hydrogen bond features | Emphasizes secondary structure |
| `coarse` | Reduced feature set | Fastest, baseline |

### Key Hyperparameters

Located in experiments/utils.py:20-42

```python
--hidden 128              # Hidden dimension size
--k_neighbors 30          # Number of nearest neighbors
--vocab_size 20           # Amino acid alphabet size
--batch_tokens 6000       # Tokens per batch (adjust for GPU memory)
--epochs 100              # Number of training epochs
--dropout 0.1             # Dropout rate
--smoothing 0.1           # Label smoothing factor
```

---

## Training

### Basic Training

```bash
cd experiments

python3 train_s2s.py \
    --cuda \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --batch_tokens 6000 \
    --features full \
    --name my_model
```

### Command-Line Arguments

**Required:**
- `--file_data`: Path to dataset JSONL file
- `--file_splits`: Path to train/val/test splits JSON

**Important optional:**
- `--cuda`: Use GPU acceleration (highly recommended)
- `--name`: Experiment name (creates log/name/ directory)
- `--batch_tokens`: Batch size in tokens (default: 2500)
- `--hidden`: Hidden dimension (default: 128)
- `--features`: Feature type (default: 'full')
- `--epochs`: Number of epochs (default: 100)
- `--dropout`: Dropout rate (default: 0.1)
- `--smoothing`: Label smoothing (default: 0.1)
- `--mpnn`: Use message passing instead of attention
- `--restore`: Resume from checkpoint

### Training Output

Training creates the following structure:

```
log/my_model/
├── args.json                    # Saved hyperparameters
├── log.txt                      # Epoch-by-epoch losses
├── results.txt                  # Final results
├── best_checkpoint_epoch*.pt    # Best model (by validation)
├── checkpoints/
│   ├── epoch1_step5000.pt      # Periodic checkpoints
│   ├── epoch2_step10000.pt
│   └── ...
└── plots/
    ├── train_*.pdf             # Training visualizations
    └── valid_*.pdf             # Validation visualizations
```

### Monitoring Training

Watch the training progress:

```bash
# View perplexity over time
tail -f log/my_model/log.txt

# Output format:
# Epoch   Train           Validation
# 0       15.234          14.892
# 1       12.456          12.134
# ...
```

Lower perplexity = better model

### Training Tips

1. **Batch size**: Adjust `--batch_tokens` based on GPU memory
   - 6000-8000 for GPUs with 12+ GB
   - 2500-4000 for GPUs with 8 GB
   - 1000-2000 for GPUs with 4 GB

2. **Early stopping**: Training automatically selects the best model based on validation perplexity

3. **Convergence**: Typical convergence in 50-100 epochs

4. **Multiple runs**: Train with different feature types:
```bash
for features in full dist hbonds coarse; do
    python3 train_s2s.py --cuda \
        --file_data ../data/chain_set.jsonl \
        --file_splits ../data/chain_set_splits.json \
        --features $features \
        --name model_$features
done
```

### Resuming Training

```bash
python3 train_s2s.py \
    --cuda \
    --restore log/my_model/checkpoints/epoch50_step34000.pt \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --name my_model_continued
```

---

## Testing and Evaluation

### Perplexity Testing

Evaluate model perplexity on test set:

```bash
python3 test_s2s.py \
    --cuda \
    --features full \
    --restore log/my_model/best_checkpoint_epoch*.pt \
    --file_splits ../data/chain_set_splits.json
```

**Output:**
```
Testing 1000 domains
Perplexity: 5.432
```

Lower perplexity indicates better sequence prediction.

### Understanding Perplexity

Perplexity measures how well the model predicts the native sequence:
- **Perplexity = 5**: Model considers ~5 amino acids equally likely on average
- **Perplexity = 20**: Random baseline (uniform over 20 amino acids)
- **Good models**: Perplexity 4-7 on test sets

### Testing on Different Benchmarks

The repository includes several test sets:

```bash
# CATH test set
python3 test_s2s.py --cuda --features full \
    --restore log/my_model/best*.pt \
    --file_splits ../data/cath/chain_set_splits.json

# SPIN2 short chains
python3 test_s2s.py --cuda --features full \
    --restore log/my_model/best*.pt \
    --file_splits ../data/SPIN2/test_split_sc.json

# SPIN2 chains with length < 100
python3 test_s2s.py --cuda --features full \
    --restore log/my_model/best*.pt \
    --file_splits ../data/SPIN2/test_split_L100.json
```

### Batch Testing Script

Use the provided script to test multiple configurations:

```bash
# Edit run_test.sh to set your checkpoint path
vim run_test.sh

# Run all tests
bash run_test.sh
```

---

## Protein Redesign

### What is Protein Redesign?

Given a protein backbone structure, generate new sequences that would fold into that structure.

### Basic Redesign

```bash
python3 test_redesign.py \
    --cuda \
    --features full \
    --restore log/my_model/best_checkpoint_epoch*.pt \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/test_structures.json \
    --seed 111
```

### Redesign Process

1. **Input**: 3D backbone structure (N, CA, C, O coordinates)
2. **Encoding**: Structure is encoded using the encoder
3. **Sampling**: Decoder generates sequence autoregressively:
   ```
   For position i = 1 to L:
       - Attend to structure
       - Attend to positions 1..i-1
       - Sample amino acid from probability distribution
   ```
4. **Output**: New protein sequence

### Generating Multiple Designs

Generate diverse sequences for the same structure:

```bash
# Run with different random seeds
for seed in 111 222 333 444; do
    python3 test_redesign.py --seed $seed \
        --cuda --features full \
        --restore log/my_model/best*.pt \
        --file_data ../data/chain_set.jsonl \
        --file_splits ../data/test.json \
        > redesign_seed_${seed}.log
done
```

### Temperature Sampling

Control diversity of generated sequences by modifying the sampling temperature in struct2seq/struct2seq.py:212:

```python
# Lower temperature (0.5-0.8): More conservative, higher native sequence recovery
logits = self.W_out(h_V_t) / 0.7

# Higher temperature (1.2-1.5): More diverse sequences
logits = self.W_out(h_V_t) / 1.3
```

### Analyzing Redesign Results

The test_redesign.py script outputs:
- **Native sequences**: Original sequences from the dataset
- **Designed sequences**: Model-generated sequences
- **Recovery rate**: Percentage of positions matching native sequence
- **Perplexity**: Model confidence in the design

---

## Advanced Features

### 1. Using MPNN Instead of Transformers

Message Passing Neural Networks (MPNNs) can be used instead of self-attention:

```bash
python3 train_s2s.py --cuda \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --mpnn \
    --name my_mpnn_model
```

**MPNN vs Transformer:**
- MPNN: Local message passing, faster, simpler
- Transformer: Global attention, more expressive, slower

### 2. Sequence Shuffling for Background Models

Train a background model by shuffling the input sequence:

```bash
python3 train_s2s.py --cuda \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --shuffle 0.5 \
    --name background_model
```

This breaks the connection between structure and sequence, useful for baselines.

### 3. Data Augmentation

Enable structure augmentation:

```bash
python3 train_s2s.py --cuda \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --augment \
    --name augmented_model
```

Augmentation applies random rotations and noise to structures during training.

### 4. Sequence-Only Baselines

Train sequence-only models (no structure information):

```bash
# Transformer-based sequence model
python3 seq_only_train.py --cuda \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --model_type sequence

# RNN-based sequence model
python3 seq_only_train.py --cuda \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --model_type rnn
```

### 5. Custom Loss Functions

The codebase includes multiple loss functions in experiments/utils.py:

- **loss_nll**: Standard negative log-likelihood (utils.py:166-173)
- **loss_smoothed**: Label smoothing for regularization (utils.py:175-185)
- **loss_smoothed_reweight**: Upweight difficult examples (utils.py:187-203)

Modify train_s2s.py to use different losses:

```python
# Line 79: Change loss function
_, loss_av_smoothed = loss_smoothed_reweight(S, log_probs, mask, weight=0.1, factor=10.0)
```

### 6. Custom Protein Features

Implement custom features by modifying struct2seq/protein_features.py.

Example: Add solvent accessibility features

```python
class ProteinFeatures(nn.Module):
    def __init__(self, ...):
        # Add new feature computation
        self.compute_sasa = True

    def forward(self, X, L, mask):
        # Existing feature computation
        V, E, E_idx = ...

        # Add SASA features
        if self.compute_sasa:
            sasa = self._compute_sasa(X)
            V = torch.cat([V, sasa], dim=-1)

        return V, E, E_idx
```

### 7. Fine-tuning on Specific Protein Families

Fine-tune a pre-trained model on specific proteins:

```bash
# First, train on general dataset
python3 train_s2s.py --cuda \
    --file_data ../data/chain_set.jsonl \
    --file_splits ../data/chain_set_splits.json \
    --name general_model

# Then fine-tune on specific family
python3 train_s2s.py --cuda \
    --restore log/general_model/best*.pt \
    --file_data ../data/my_family.jsonl \
    --file_splits ../data/my_family_splits.json \
    --epochs 20 \
    --name finetuned_model
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `--batch_tokens`: Try 4000, 2500, or 1000
- Reduce `--hidden`: Try 64 or 96 instead of 128
- Use gradient accumulation (requires code modification)
- Use CPU training (remove `--cuda`, much slower)

#### 2. File Not Found Errors

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: '../data/chain_set.jsonl'
```

**Solutions:**
- Ensure you're in the `experiments/` directory
- Check file paths are correct
- Download data from the provided URL
- Use absolute paths if needed

#### 3. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'struct2seq'
```

**Solutions:**
- Ensure you're in the `experiments/` directory
- Check that `struct2seq/` directory exists
- Verify `__init__.py` exists in `struct2seq/`

#### 4. Poor Model Performance

**Symptoms:**
- Perplexity > 15
- No improvement after many epochs

**Solutions:**
- Check data quality (are coordinates correct?)
- Verify train/val/test splits are reasonable
- Try different learning rates
- Increase model capacity (`--hidden 256`)
- Train longer (`--epochs 200`)

#### 5. Very Slow Training

**Solutions:**
- Use GPU (`--cuda`)
- Increase batch size (`--batch_tokens 8000`)
- Use MPNN instead of attention (`--mpnn`)
- Reduce `--k_neighbors` to 20 or 15

#### 6. NaN Loss During Training

**Symptoms:**
```
loss: nan
```

**Solutions:**
- Reduce learning rate (modify noam_opt.py)
- Check for corrupted data entries
- Reduce label smoothing (`--smoothing 0.0`)
- Use gradient clipping (requires code modification)

#### 7. Checkpoint Loading Errors

**Error:**
```
KeyError: 'model_state_dict'
```

**Solutions:**
- Ensure checkpoint file is complete (not corrupted)
- Verify hyperparameters match (features, hidden size, etc.)
- Use correct checkpoint file (best_*.pt vs epoch*.pt)

---

## Examples

### Example 1: Complete Training Pipeline

```bash
#!/bin/bash
# complete_pipeline.sh

# 1. Prepare workspace
cd experiments
mkdir -p ../data/my_project

# 2. Download and prepare data (assuming you have downloaded chain_set.jsonl)
cp ../data/chain_set.jsonl ../data/my_project/
cp ../data/chain_set_splits.json ../data/my_project/

# 3. Train model
python3 train_s2s.py \
    --cuda \
    --file_data ../data/my_project/chain_set.jsonl \
    --file_splits ../data/my_project/chain_set_splits.json \
    --batch_tokens 6000 \
    --features full \
    --hidden 128 \
    --epochs 100 \
    --name protein_designer_v1

# 4. Test model
python3 test_s2s.py \
    --cuda \
    --features full \
    --restore log/protein_designer_v1/best_checkpoint_epoch*.pt \
    --file_splits ../data/my_project/chain_set_splits.json \
    > results_perplexity.txt

# 5. Generate redesigns
for seed in 111 222 333; do
    python3 test_redesign.py \
        --cuda \
        --features full \
        --restore log/protein_designer_v1/best_checkpoint_epoch*.pt \
        --file_data ../data/my_project/chain_set.jsonl \
        --file_splits ../data/my_project/chain_set_splits.json \
        --seed $seed \
        > results_redesign_${seed}.txt
done

echo "Pipeline complete! Check log/protein_designer_v1/ for results."
```

### Example 2: Comparing Different Feature Types

```bash
#!/bin/bash
# compare_features.sh

FEATURES=("full" "dist" "hbonds" "coarse")

for feat in "${FEATURES[@]}"; do
    echo "Training with features: $feat"

    # Train
    python3 train_s2s.py \
        --cuda \
        --file_data ../data/chain_set.jsonl \
        --file_splits ../data/chain_set_splits.json \
        --batch_tokens 6000 \
        --features $feat \
        --epochs 100 \
        --name model_${feat}

    # Test
    python3 test_s2s.py \
        --cuda \
        --features $feat \
        --restore log/model_${feat}/best_checkpoint_epoch*.pt \
        --file_splits ../data/chain_set_splits.json \
        > results_${feat}.txt
done

# Compare results
echo "Feature Type | Test Perplexity"
echo "-------------|----------------"
for feat in "${FEATURES[@]}"; do
    perp=$(grep "Perplexity:" results_${feat}.txt | awk '{print $2}')
    echo "$feat | $perp"
done
```

### Example 3: Interactive Design Session

```python
# interactive_design.py
import sys, torch, json
sys.path.insert(0, '..')
from struct2seq import *
from experiments.utils import *

# Load model
args = type('Args', (), {
    'hidden': 128,
    'k_neighbors': 30,
    'vocab_size': 20,
    'features': 'full',
    'model_type': 'structure',
    'mpnn': False,
    'dropout': 0.1,
    'cuda': True,
    'seed': 111
})()

device = setup_device_rng(args)
model = setup_model(vars(args), device)
load_checkpoint('log/my_model/best_checkpoint_epoch90.pt', model)
model.eval()

# Load a structure
with open('../data/chain_set.jsonl') as f:
    structures = [json.loads(line) for line in f]

# Select a structure to redesign
structure = structures[0]  # First structure
print(f"Designing sequence for: {structure['name']}")
print(f"Native sequence: {structure['seq']}")

# Prepare structure
batch = [structure]
X, S, mask, lengths = featurize(batch, device)

# Generate designs
print("\nGenerating 5 designs...")
designs = []
with torch.no_grad():
    for i in range(5):
        S_designed = model.sample(X, lengths, mask, temperature=1.0)
        seq_designed = decode_sequence(S_designed[0], lengths[0])
        designs.append(seq_designed)
        print(f"Design {i+1}: {seq_designed}")

        # Calculate recovery
        recovery = sum(a==b for a,b in zip(structure['seq'], seq_designed)) / len(structure['seq'])
        print(f"  Recovery: {recovery*100:.1f}%")

def decode_sequence(S, length):
    """Convert integer sequence to amino acid string"""
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    return ''.join([alphabet[S[i].item()] for i in range(length)])
```

### Example 4: Analyzing Model Predictions

```python
# analyze_predictions.py
import sys, torch, json, numpy as np
sys.path.insert(0, '..')
from struct2seq import *
from experiments.utils import *

# Setup
args, device, model = setup_cli_model()
model.eval()

# Load test set
with open('../data/chain_set_splits.json') as f:
    splits = json.load(f)

dataset = data.StructureDataset('../data/chain_set.jsonl', truncate=None, max_length=500)
dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
test_set = Subset(dataset, [dataset_indices[name] for name in splits['test'][:10]])

# Analyze each structure
results = []
with torch.no_grad():
    for idx in range(len(test_set)):
        structure = dataset[test_set.indices[idx]]
        batch = [structure]
        X, S, mask, lengths = featurize(batch, device)

        # Get predictions
        log_probs = model(X, S, lengths, mask)
        probs = torch.exp(log_probs)

        # Per-position analysis
        S_np = S[0].cpu().numpy()
        probs_np = probs[0].cpu().numpy()

        native_seq = structure['seq']
        confidences = [probs_np[i, S_np[i]] for i in range(len(native_seq))]

        results.append({
            'name': structure['name'],
            'length': len(native_seq),
            'mean_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'uncertain_positions': sum(1 for c in confidences if c < 0.1)
        })

        print(f"\n{structure['name']}:")
        print(f"  Length: {len(native_seq)}")
        print(f"  Mean confidence: {np.mean(confidences):.3f}")
        print(f"  Positions with confidence < 0.1: {sum(1 for c in confidences if c < 0.1)}")

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Total structures: {len(results)}")
print(f"Average confidence: {np.mean([r['mean_confidence'] for r in results]):.3f}")
print(f"Structures with uncertain positions: {sum(1 for r in results if r['uncertain_positions'] > 0)}")
```

### Example 5: Batch Processing Multiple Structures

```python
# batch_redesign.py
import sys, torch, json
sys.path.insert(0, '..')
from struct2seq import *
from experiments.utils import *

def batch_redesign(checkpoint_path, input_jsonl, output_jsonl, num_designs=3):
    """
    Generate multiple sequence designs for all structures in a file

    Args:
        checkpoint_path: Path to model checkpoint
        input_jsonl: Input structures file
        output_jsonl: Output designs file
        num_designs: Number of designs per structure
    """
    # Load model
    args = type('Args', (), {
        'hidden': 128, 'k_neighbors': 30, 'vocab_size': 20,
        'features': 'full', 'model_type': 'structure',
        'mpnn': False, 'dropout': 0.1, 'cuda': True, 'seed': 111,
        'restore': checkpoint_path
    })()

    device = setup_device_rng(args)
    model = setup_model(vars(args), device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    # Load structures
    with open(input_jsonl) as f:
        structures = [json.loads(line) for line in f]

    # Generate designs
    results = []
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'

    with torch.no_grad():
        for struct in structures:
            print(f"Designing for {struct['name']}...")
            batch = [struct]
            X, S, mask, lengths = featurize(batch, device)

            designs = []
            for i in range(num_designs):
                S_designed = model.sample(X, lengths, mask, temperature=1.0)
                seq = ''.join([alphabet[S_designed[0,j].item()]
                              for j in range(lengths[0])])
                designs.append(seq)

            results.append({
                'name': struct['name'],
                'native_sequence': struct['seq'],
                'designed_sequences': designs
            })

    # Save results
    with open(output_jsonl, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Saved {len(results)} designs to {output_jsonl}")

if __name__ == '__main__':
    batch_redesign(
        checkpoint_path='log/my_model/best_checkpoint_epoch90.pt',
        input_jsonl='../data/my_structures.jsonl',
        output_jsonl='../data/designed_sequences.jsonl',
        num_designs=5
    )
```

---

## Additional Resources

### Paper and Citation

```bibtex
@inproceedings{ingraham2019generative,
  author = {Ingraham, John and Garg, Vikas K and Barzilay, Regina and Jaakkola, Tommi},
  title = {Generative Models for Graph-Based Protein Design},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2019}
}
```

### Data Downloads

- Preprocessed datasets: http://people.csail.mit.edu/ingraham/graph-protein-design/data/
- CATH database: http://www.cathdb.info/
- PDB database: https://www.rcsb.org/

### Related Work

- ProteinMPNN: More recent protein design method
- RosettaDesign: Traditional physics-based design
- AlphaFold: Structure prediction (inverse problem)

### Community and Support

- GitHub Issues: https://github.com/jingraham/neurips19-graph-protein-design/issues
- Original paper: https://papers.nips.cc/paper/9711-generative-models-for-graph-based-protein-design

---

## Glossary

- **Autoregressive**: Generating sequence one position at a time, conditioned on previous positions
- **CATH**: Class, Architecture, Topology, Homology - a protein structure classification
- **Perplexity**: Measure of model uncertainty (lower is better)
- **Redesign**: Generating new sequences for existing backbone structures
- **Recovery**: Percentage of positions matching the native sequence
- **Struct2Seq**: Structure-to-Sequence, the model name
- **Temperature**: Sampling parameter controlling diversity (higher = more diverse)

---

## Quick Reference

### Most Useful Commands

```bash
# Train a model
python3 train_s2s.py --cuda --file_data DATA --file_splits SPLITS --name NAME

# Test perplexity
python3 test_s2s.py --cuda --restore CHECKPOINT --file_splits SPLITS

# Generate designs
python3 test_redesign.py --cuda --restore CHECKPOINT --file_data DATA --file_splits SPLITS

# Resume training
python3 train_s2s.py --cuda --restore CHECKPOINT --file_data DATA --file_splits SPLITS
```

### Default Hyperparameters

```
hidden_dim: 128
k_neighbors: 30
batch_tokens: 2500
dropout: 0.1
label_smoothing: 0.1
epochs: 100
```

---

## Changelog

- **v1.0 (2019)**: Initial release with NeurIPS paper
- **Tutorial created**: Complete guide with examples and troubleshooting

---

**End of Tutorial**

For questions, issues, or contributions, please visit the GitHub repository:
https://github.com/jingraham/neurips19-graph-protein-design
