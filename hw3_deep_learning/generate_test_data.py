"""Generate test data for HW3: Deep Learning.

This script creates .npz fixtures used by dl_tests.py.
NOT shared with students.

Usage:
    python generate_test_data.py
"""

import json
from pathlib import Path

import numpy as np
import torch


def generate_spiral_data(n_samples=300, n_classes=3, noise=0.3, seed=42):
    """Generate 2D spiral data for classification."""
    rng = np.random.RandomState(seed)
    samples_per_class = n_samples // n_classes
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    for k in range(n_classes):
        ix = range(samples_per_class * k, samples_per_class * (k + 1))
        r = np.linspace(0.0, 1.0, samples_per_class)
        t = (
            np.linspace(k * 4, (k + 1) * 4, samples_per_class)
            + rng.randn(samples_per_class) * noise
        )
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = k
    return X, y


def generate_tiny_shakespeare():
    """Create a small text corpus for character-level LM testing."""
    text = (
        "First Citizen:\n"
        "Before we proceed any further, hear me speak.\n\n"
        "All:\n"
        "Speak, speak.\n\n"
        "First Citizen:\n"
        "You are all resolved rather to die than to famish?\n\n"
        "All:\n"
        "Resolved. resolved.\n\n"
        "First Citizen:\n"
        "First, you know Caius Marcius is chief enemy to the people.\n"
    )

    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    return text, char_to_idx, idx_to_char, vocab_size


def generate_mlp_reference(seed=42):
    """Create reference data for gradient checking."""
    rng = np.random.RandomState(seed)
    X = rng.randn(4, 2)
    y = np.array([0, 1, 2, 1])
    return X, y


def generate_attention_reference(seed=42):
    """Create reference Q, K, V for attention testing."""
    torch.manual_seed(seed)
    Q = torch.randn(2, 3, 8)
    K = torch.randn(2, 5, 8)
    V = torch.randn(2, 5, 8)
    return Q.numpy(), K.numpy(), V.numpy()


def main():
    output_dir = Path(__file__).parent / "test_data"
    output_dir.mkdir(exist_ok=True)

    # 1. Spiral data
    X, y = generate_spiral_data(n_samples=300, n_classes=3, noise=0.3, seed=42)
    np.savez(output_dir / "spirals.npz", X=X, y=y)
    print(f"spirals.npz: X {X.shape}, y {y.shape}, classes {np.unique(y)}")

    # 2. Tiny Shakespeare
    text, char_to_idx, idx_to_char, vocab_size = generate_tiny_shakespeare()
    np.savez(
        output_dir / "tiny_shakespeare.npz",
        text=np.array(list(text)),
        char_to_idx_json=json.dumps(char_to_idx),
        idx_to_char_json=json.dumps(idx_to_char),
        vocab_size=np.array(vocab_size),
    )
    print(
        f"tiny_shakespeare.npz: {len(text)} chars, vocab_size {vocab_size}"
    )

    # 3. MLP reference
    X_ref, y_ref = generate_mlp_reference(seed=42)
    np.savez(output_dir / "mlp_reference.npz", X=X_ref, y=y_ref, seed=np.array(42))
    print(f"mlp_reference.npz: X {X_ref.shape}, y {y_ref.shape}")

    # 4. Attention reference
    Q, K, V = generate_attention_reference(seed=42)
    np.savez(output_dir / "attention_reference.npz", Q=Q, K=K, V=V)
    print(f"attention_reference.npz: Q {Q.shape}, K {K.shape}, V {V.shape}")

    print(f"\nAll test data generated in {output_dir}/")


if __name__ == "__main__":
    main()
