"""Helper functions for HW4: Dimensionality Reduction.

Provides data generation and visualization utilities for the notebook.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_high_dim_blobs(n_samples=300, n_features=20, n_informative=8, n_classes=3, seed=42):
    """Generate high-dimensional Gaussian blobs with low intrinsic dimensionality.

    Creates clusters in n_informative dimensions, then embeds into n_features
    dimensions via a random rotation matrix.

    Parameters
    ----------
    n_samples : int
        Total number of samples (divided equally among classes).
    n_features : int
        Total number of features (ambient dimensionality).
    n_informative : int
        Number of informative dimensions (intrinsic dimensionality).
    n_classes : int
        Number of clusters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    """
    rng = np.random.RandomState(seed)
    samples_per_class = n_samples // n_classes

    centers = np.zeros((n_classes, n_informative))
    for i in range(n_classes):
        centers[i, i % n_informative] = 3.0

    X_low = []
    y = []
    for i, center in enumerate(centers):
        points = rng.randn(samples_per_class, n_informative) * 0.5 + center
        X_low.append(points)
        y.extend([i] * samples_per_class)

    X_low = np.vstack(X_low)
    y = np.array(y)

    X_padded = np.zeros((len(X_low), n_features))
    X_padded[:, :n_informative] = X_low

    random_matrix = rng.randn(n_features, n_features)
    Q, _ = np.linalg.qr(random_matrix)
    X = X_padded @ Q.T
    X += rng.randn(*X.shape) * 0.01

    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def generate_swiss_roll(n_samples=500, noise=0.3, seed=42):
    """Generate Swiss roll manifold data in 3D.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    noise : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 3)
    color : np.ndarray of shape (n_samples,)
        Parameter t, useful for coloring the manifold.
    """
    rng = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.rand(n_samples))
    height = 10 * rng.rand(n_samples)

    X = np.column_stack([
        t * np.cos(t),
        height,
        t * np.sin(t),
    ])
    X += rng.randn(*X.shape) * noise

    return X, t


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_2d_projection(X_2d, y, title="2D Projection"):
    """Scatter plot of 2D-projected data colored by class label.

    Parameters
    ----------
    X_2d : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
    title : str
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", edgecolors="k", s=20)
    plt.colorbar(scatter)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_explained_variance(explained_variance_ratio, title="Explained Variance Ratio"):
    """Bar chart of explained variance ratio with cumulative line.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray of shape (n_components,)
    title : str
    """
    n = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(range(n), explained_variance_ratio, alpha=0.6, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels([str(i + 1) for i in range(n)])

    ax2 = ax1.twinx()
    ax2.plot(range(n), cumulative, "ro-", linewidth=2, label="Cumulative")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_ylim([0, 1.05])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title(title)
    plt.tight_layout()
    plt.show()

    print(f"Top 3 components explain {cumulative[min(2, n-1)]:.1%} of variance")
    print(f"Top 5 components explain {cumulative[min(4, n-1)]:.1%} of variance")


def plot_reconstruction_comparison(X_original, X_reconstructed, n_samples=5, feature_indices=None):
    """Plot original vs reconstructed feature values for selected samples.

    Parameters
    ----------
    X_original : np.ndarray of shape (n_samples_total, n_features)
    X_reconstructed : np.ndarray of shape (n_samples_total, n_features)
    n_samples : int
        Number of samples to display.
    feature_indices : list of int, optional
        Which features to plot. If None, plots all.
    """
    n_features = X_original.shape[1]
    if feature_indices is None:
        feature_indices = list(range(n_features))

    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4), sharey=True)
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(feature_indices, X_original[i, feature_indices], "bo-", label="Original", markersize=4)
        ax.plot(feature_indices, X_reconstructed[i, feature_indices], "rx-", label="Reconstructed", markersize=4)
        ax.set_title(f"Sample {i}")
        ax.set_xlabel("Feature index")
        if i == 0:
            ax.set_ylabel("Value")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    mse = np.mean((X_original[:n_samples] - X_reconstructed[:n_samples]) ** 2)
    plt.suptitle(f"Original vs Reconstructed (MSE: {mse:.4f})", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_pca_eigen_vs_svd(eigenvalues_eigen, eigenvalues_svd):
    """Overlay eigenvalues from eigendecomposition and SVD to show equivalence.

    Parameters
    ----------
    eigenvalues_eigen : np.ndarray
    eigenvalues_svd : np.ndarray
    """
    n = len(eigenvalues_eigen)
    plt.figure(figsize=(10, 5))
    plt.plot(range(n), eigenvalues_eigen, "bo-", label="Eigendecomposition", markersize=8)
    plt.plot(range(n), eigenvalues_svd, "rx--", label="SVD", markersize=8)
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue (Explained Variance)")
    plt.title("Eigendecomposition vs SVD: Eigenvalue Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(n), [str(i + 1) for i in range(n)])
    plt.tight_layout()
    plt.show()

    max_diff = np.max(np.abs(eigenvalues_eigen - eigenvalues_svd))
    print(f"Maximum eigenvalue difference: {max_diff:.2e}")
    if max_diff < 1e-10:
        print("The two methods produce identical eigenvalues!")
    else:
        print("Small differences are due to floating-point arithmetic.")


def plot_autoencoder_training(losses, title="Autoencoder Training Loss"):
    """Plot autoencoder training loss curve.

    Parameters
    ----------
    losses : list of float
    title : str
    """
    plt.figure(figsize=(8, 4))
    plt.plot(losses, linewidth=2, color="teal")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if len(losses) > 1:
        plt.annotate(
            f"Final: {losses[-1]:.4f}",
            xy=(len(losses) - 1, losses[-1]),
            fontsize=10,
            ha="right",
        )
    plt.tight_layout()
    plt.show()


def plot_latent_space(z, color, title="Latent Space"):
    """2D scatter plot of autoencoder bottleneck activations.

    Parameters
    ----------
    z : np.ndarray of shape (n_samples, 2)
    color : np.ndarray of shape (n_samples,)
        Color values (can be class labels or continuous parameter).
    title : str
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=color, cmap="viridis", edgecolors="k", s=20, alpha=0.8)
    plt.colorbar(scatter)
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_swiss_roll_3d(X, color, title="Swiss Roll"):
    """3D scatter plot of Swiss roll data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 3)
    color : np.ndarray of shape (n_samples,)
    title : str
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="viridis", s=10, alpha=0.8)
    fig.colorbar(scatter, shrink=0.5, label="t parameter")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_manifold_comparison(X_pca, X_ae, color, titles=("PCA 2D", "Autoencoder 2D")):
    """Side-by-side comparison of PCA and autoencoder 2D projections.

    Parameters
    ----------
    X_pca : np.ndarray of shape (n_samples, 2)
    X_ae : np.ndarray of shape (n_samples, 2)
    color : np.ndarray of shape (n_samples,)
    titles : tuple of str
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap="viridis", edgecolors="k", s=20, alpha=0.8)
    ax1.set_title(titles[0])
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    ax1.grid(True, alpha=0.3)
    fig.colorbar(sc1, ax=ax1)

    sc2 = ax2.scatter(X_ae[:, 0], X_ae[:, 1], c=color, cmap="viridis", edgecolors="k", s=20, alpha=0.8)
    ax2.set_title(titles[1])
    ax2.set_xlabel("Latent dim 1")
    ax2.set_ylabel("Latent dim 2")
    ax2.grid(True, alpha=0.3)
    fig.colorbar(sc2, ax=ax2)

    plt.suptitle("PCA vs Autoencoder: 2D Projections", fontsize=14)
    plt.tight_layout()
    plt.show()
