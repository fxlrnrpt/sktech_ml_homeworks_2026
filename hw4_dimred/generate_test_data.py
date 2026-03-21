#!/usr/bin/env python3
"""
Generate test data fixtures for HW4: Dimensionality Reduction.

Creates .npz files in test_data/ that are loaded by dimred_tests.py.
This script is NOT shared with students.
"""

import numpy as np
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "test_data"


def generate_high_dim_blobs():
    """
    Generate 3 Gaussian clusters in 8D, then embed into 20D via a random rotation.
    This ensures PCA shows a gradual variance decay across ~8 components.
    """
    rng = np.random.RandomState(42)

    n_samples_per_class = 100
    n_classes = 3
    n_informative = 8
    n_features = 20

    centers = np.zeros((n_classes, n_informative))
    for i in range(n_classes):
        centers[i, i % n_informative] = 3.0

    X_low = []
    y = []
    for i, center in enumerate(centers):
        points = rng.randn(n_samples_per_class, n_informative) * 0.5 + center
        X_low.append(points)
        y.extend([i] * n_samples_per_class)

    X_low = np.vstack(X_low)
    y = np.array(y)

    # Embed into 20D: pad with zeros then apply random rotation
    X_padded = np.zeros((len(X_low), n_features))
    X_padded[:, :n_informative] = X_low

    # Random orthogonal rotation matrix via QR decomposition
    random_matrix = rng.randn(n_features, n_features)
    Q, _ = np.linalg.qr(random_matrix)
    X = X_padded @ Q.T

    # Add small noise to non-informative dimensions
    X += rng.randn(*X.shape) * 0.01

    # Shuffle
    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    np.savez(
        TEST_DATA_DIR / "high_dim_blobs.npz",
        X=X,
        y=y,
    )
    print(f"high_dim_blobs.npz: X{X.shape}, y{y.shape}")


def generate_swiss_roll():
    """Generate Swiss roll manifold data in 3D using numpy trig."""
    rng = np.random.RandomState(42)
    n_samples = 500
    noise = 0.3

    t = 1.5 * np.pi * (1 + 2 * rng.rand(n_samples))
    height = 10 * rng.rand(n_samples)

    X = np.column_stack([
        t * np.cos(t),
        height,
        t * np.sin(t),
    ])
    X += rng.randn(*X.shape) * noise

    # Color by the parameter t (for visualization)
    color = t

    np.savez(
        TEST_DATA_DIR / "swiss_roll.npz",
        X=X,
        color=color,
    )
    print(f"swiss_roll.npz: X{X.shape}, color{color.shape}")


def generate_pca_reference():
    """Pre-compute PCA reference values from high_dim_blobs for validation."""
    data = np.load(TEST_DATA_DIR / "high_dim_blobs.npz")
    X = data["X"]

    # Center
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Covariance
    cov = (X_centered.T @ X_centered) / (len(X) - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Components (rows)
    components = eigenvectors.T

    # Explained variance ratio
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    np.savez(
        TEST_DATA_DIR / "pca_reference.npz",
        eigenvalues=eigenvalues,
        components=components,
        explained_variance_ratio=explained_variance_ratio,
        mean=mean,
    )
    print(f"pca_reference.npz: eigenvalues{eigenvalues.shape}, components{components.shape}")


def main():
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    generate_high_dim_blobs()
    generate_swiss_roll()
    generate_pca_reference()
    print("\nAll test data generated successfully!")


if __name__ == "__main__":
    main()
