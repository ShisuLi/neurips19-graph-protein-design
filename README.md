# Graph-Based Protein Design

This repo contains code for [Generative Models for Graph-Based Protein Design](https://papers.nips.cc/paper/9711-generative-models-for-graph-based-protein-design) by John Ingraham, Vikas Garg, Regina Barzilay and Tommi Jaakkola, NeurIPS 2019.

Our approach 'designs' protein sequences for target 3D structures via a graph-conditioned, autoregressive language model:
<p align="center"><img src="data/scheme.png" width="500"></p>


## Overview
* `struct2seq/` contains original model code
* `struct2seq_pyg/` contains **NEW** PyTorch Geometric refactored version (cleaner, more modular)
* `experiments/` contains scripts for training and evaluating the original model
* `data/` contains scripts for building and processing datasets in the paper
* `tutorial_struct2seq_pyg.ipynb` comprehensive tutorial for the PyTorch Geometric version

## 🆕 PyTorch Geometric Version

A modern, clean reimplementation using PyTorch Geometric is now available in `struct2seq_pyg/`!

**Features:**
- ✅ Clean, modular architecture
- ✅ Comprehensive documentation and tutorial notebook
- ✅ Built on PyTorch Geometric for efficient graph operations
- ✅ Supports both GAT and MPNN layers
- ✅ Easy to understand and extend

**Quick Start:**
```bash
# Install dependencies
pip install torch torch-geometric

# Run example
python example_pyg.py

# Or explore the tutorial
jupyter notebook tutorial_struct2seq_pyg.ipynb
```

See [`struct2seq_pyg/README.md`](struct2seq_pyg/README.md) for detailed documentation.

## Requirements
* Python >= 3.0
* PyTorch >= 1.0
* Numpy

## Citation
```
@inproceedings{ingraham2019generative,
author = {Ingraham, John and Garg, Vikas K and Barzilay, Regina and Jaakkola, Tommi},
title = {Generative Models for Graph-Based Protein Design},
booktitle = {Advances in Neural Information Processing Systems}
year = {2019}
}
```
