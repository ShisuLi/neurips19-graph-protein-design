#!/usr/bin/env python3
"""
Simple example script demonstrating the PyTorch Geometric Struct2Seq model.
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np

from struct2seq_pyg import (
    create_protein_graph,
    Struct2SeqPyG,
    ProteinFeaturizer
)


def create_synthetic_protein(length=50):
    """Create a synthetic protein for demonstration."""
    coords = {
        'N': np.random.randn(length, 3),
        'CA': np.random.randn(length, 3),
        'C': np.random.randn(length, 3),
        'O': np.random.randn(length, 3),
    }

    AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
    sequence = ''.join(np.random.choice(list(AA_ALPHABET), length))

    return coords, sequence


def main():
    print("=" * 60)
    print("Struct2Seq with PyTorch Geometric - Example Script")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create synthetic protein data
    print("\n1. Creating synthetic protein data...")
    coords, sequence = create_synthetic_protein(length=30)
    print(f"   Sequence: {sequence}")
    print(f"   Length: {len(sequence)}")

    # Create protein graph
    print("\n2. Converting to graph representation...")
    featurizer = ProteinFeaturizer()
    data = create_protein_graph(
        coords=coords,
        sequence=sequence,
        k_neighbors=10,
        featurizer=featurizer
    ).to(device)

    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.edge_index.size(1)}")
    print(f"   Node features: {data.x.shape}")
    print(f"   Edge features: {data.edge_attr.shape}")

    # Initialize model
    print("\n3. Initializing Struct2Seq model...")
    model = Struct2SeqPyG(
        node_feature_dim=6,
        edge_feature_dim=39,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_letters=20,
        num_heads=4,
        dropout=0.1,
        use_mpnn=False
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    # Forward pass
    print("\n4. Running forward pass...")
    model.eval()
    with torch.no_grad():
        log_probs = model(data)
        print(f"   Output shape: {log_probs.shape}")
        print(f"   Output log probabilities for position 0: {log_probs[0, :5].cpu().numpy()}")

    # Sample sequences
    print("\n5. Sampling sequences with different temperatures...")
    AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'

    for temp in [0.1, 0.5, 1.0, 2.0]:
        sampled = model.sample(data, temperature=temp)
        sampled_seq = ''.join([AA_ALPHABET[i] for i in sampled[0].cpu().numpy()])
        print(f"   Temperature {temp:.1f}: {sampled_seq}")

    # Training example
    print("\n6. Training for a few steps...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(5):
        optimizer.zero_grad()
        log_probs = model(data)
        loss = F.nll_loss(log_probs, data.seq)
        loss.backward()
        optimizer.step()

        # Compute accuracy
        pred = log_probs.argmax(dim=-1)
        accuracy = (pred == data.seq).float().mean().item()

        print(f"   Step {step+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}")

    # Final sampling
    print("\n7. Sampling after training...")
    model.eval()
    with torch.no_grad():
        sampled = model.sample(data, temperature=0.5)
        sampled_seq = ''.join([AA_ALPHABET[i] for i in sampled[0].cpu().numpy()])
        print(f"   Sampled: {sampled_seq}")
        print(f"   Original: {sequence}")

        # Compute recovery
        recovery = (sampled[0].cpu().numpy() == data.seq.cpu().numpy()).mean()
        print(f"   Recovery rate: {recovery:.2%}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - See tutorial_struct2seq_pyg.ipynb for a full tutorial")
    print("  - Train on real protein datasets")
    print("  - Experiment with different hyperparameters")
    print("=" * 60)


if __name__ == '__main__':
    main()
