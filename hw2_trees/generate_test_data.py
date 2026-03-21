"""
Generate test data for HW2: Trees & Ensembles tests.

THIS FILE SHOULD NEVER BE SHARED WITH STUDENTS.
It generates the test data that is saved to .npz files and used by trees_tests.py.

Run this script to regenerate test data:
    python generate_test_data.py
"""

import numpy as np
from pathlib import Path


def generate_binary_classification(
    n_samples: int = 300,
    separation: float = 2.0,
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


def generate_multiclass_classification(
    n_samples: int = 300,
    n_classes: int = 3,
    separation: float = 2.5,
    noise_std: float = 0.5,
    random_state: int = 42,
) -> tuple:
    """Generate multiclass classification data (Gaussian clusters)."""
    np.random.seed(random_state)
    n_per_class = n_samples // n_classes

    # Place class centers on a circle
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)]) * separation

    X_list = []
    y_list = []
    for i in range(n_classes):
        Xi = np.random.randn(n_per_class, 2) * noise_std + centers[i]
        X_list.append(Xi)
        y_list.append(np.full(n_per_class, i))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    idx = np.random.permutation(len(y))
    return X[idx], y[idx]


def generate_complex_classification(
    n_samples: int = 400,
    noise_std: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """Generate complex classification data (concentric circles)."""
    np.random.seed(random_state)
    n_per_class = n_samples // 2

    # Inner circle (class 0)
    theta_inner = np.random.uniform(0, 2 * np.pi, n_per_class)
    r_inner = 0.5 + np.random.randn(n_per_class) * noise_std
    X_inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])

    # Outer circle (class 1)
    theta_outer = np.random.uniform(0, 2 * np.pi, n_per_class)
    r_outer = 1.5 + np.random.randn(n_per_class) * noise_std
    X_outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])

    X = np.vstack([X_inner, X_outer])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def main():
    """Generate and save all test data."""
    output_dir = Path(__file__).parent / "test_data"
    output_dir.mkdir(exist_ok=True)

    print("Generating test data...")

    # Binary classification
    X, y = generate_binary_classification()
    np.savez(output_dir / "binary_classification.npz", X=X, y=y)
    print("  ✓ binary_classification.npz")

    # Multiclass classification
    X, y = generate_multiclass_classification()
    np.savez(output_dir / "multiclass_classification.npz", X=X, y=y)
    print("  ✓ multiclass_classification.npz")

    # Complex classification (concentric circles)
    X, y = generate_complex_classification()
    np.savez(output_dir / "complex_classification.npz", X=X, y=y)
    print("  ✓ complex_classification.npz")

    print(f"\nAll test data saved to {output_dir}/")


if __name__ == "__main__":
    main()
