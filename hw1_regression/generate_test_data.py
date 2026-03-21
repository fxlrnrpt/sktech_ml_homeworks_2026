"""
Generate test data for HW1: Regression tests.

THIS FILE SHOULD NEVER BE SHARED WITH STUDENTS.
It generates the test data that is saved to .npz files and used by regression_tests.py.

Run this script to regenerate test data:
    python generate_test_data.py
"""

import numpy as np
from pathlib import Path


def generate_well_conditioned_data(
    n_samples: int = 100,
    n_features: int = 3,
    noise_std: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """Generate well-conditioned regression data."""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + noise_std * np.random.randn(n_samples)
    return X, y, true_weights


def generate_collinear_data(
    n_samples: int = 100,
    collinearity: float = 1e-10,
    noise_std: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """Generate ill-conditioned data with collinear features."""
    np.random.seed(random_state)
    x1 = np.random.randn(n_samples)
    x2 = x1 + collinearity * np.random.randn(n_samples)
    x3 = np.random.randn(n_samples)
    X = np.column_stack([x1, x2, x3])
    true_weights = np.array([1.0, 2.0, 3.0])
    y = X @ true_weights + noise_std * np.random.randn(n_samples)
    return X, y, true_weights


def generate_nonlinear_data(
    n_samples: int = 100, noise_std: float = 0.1, random_state: int = 42
) -> tuple:
    """Generate non-linear (sinusoidal) data."""
    np.random.seed(random_state)
    X = np.linspace(0, 2 * np.pi, n_samples).reshape(-1, 1)
    y = np.sin(X).ravel() + noise_std * np.random.randn(n_samples)
    return X, y


def generate_sparse_data(
    n_samples: int = 100,
    n_features: int = 20,
    n_informative: int = 5,
    noise_std: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """Generate data with sparse true weights."""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    true_weights = np.zeros(n_features)
    true_weights[:n_informative] = np.random.randn(n_informative) * 2
    y = X @ true_weights + noise_std * np.random.randn(n_samples)
    return X, y, true_weights


def generate_perfect_fit_data(random_state: int = 42) -> tuple:
    """Generate data with perfect linear relationship (no noise)."""
    np.random.seed(random_state)
    X = np.random.randn(50, 3)
    true_weights = np.array([1.0, 2.0, 3.0])
    y = X @ true_weights
    return X, y, true_weights


def generate_classification_data(
    n_samples: int = 200,
    separation: float = 1.5,
    noise_std: float = 0.5,
    random_state: int = 42,
) -> tuple:
    """Generate binary classification data (two Gaussian blobs)."""
    np.random.seed(random_state)
    n_per_class = n_samples // 2

    X0 = np.random.randn(n_per_class, 2) * noise_std - separation / 2
    X1 = np.random.randn(n_per_class, 2) * noise_std + separation / 2

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def main():
    """Generate and save all test data."""
    output_dir = Path(__file__).parent / "test_data"
    output_dir.mkdir(exist_ok=True)

    print("Generating test data...")

    # Well-conditioned data
    X, y, true_weights = generate_well_conditioned_data()
    np.savez(
        output_dir / "well_conditioned.npz", X=X, y=y, true_weights=true_weights
    )
    print("  ✓ well_conditioned.npz")

    # Collinear data
    X, y, true_weights = generate_collinear_data()
    np.savez(output_dir / "collinear.npz", X=X, y=y, true_weights=true_weights)
    print("  ✓ collinear.npz")

    # Nonlinear data
    X, y = generate_nonlinear_data()
    np.savez(output_dir / "nonlinear.npz", X=X, y=y)
    print("  ✓ nonlinear.npz")

    # Sparse data
    X, y, true_weights = generate_sparse_data()
    np.savez(output_dir / "sparse.npz", X=X, y=y, true_weights=true_weights)
    print("  ✓ sparse.npz")

    # Perfect fit data
    X, y, true_weights = generate_perfect_fit_data()
    np.savez(output_dir / "perfect_fit.npz", X=X, y=y, true_weights=true_weights)
    print("  ✓ perfect_fit.npz")

    # Classification data
    X, y = generate_classification_data()
    np.savez(output_dir / "classification.npz", X=X, y=y)
    print("  ✓ classification.npz")

    print(f"\nAll test data saved to {output_dir}/")


if __name__ == "__main__":
    main()
