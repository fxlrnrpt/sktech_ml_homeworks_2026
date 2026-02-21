"""
Helper functions for HW2: Trees & Ensembles.

Provides data generation and visualization utilities for the homework notebook.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# Data Generation (for notebook demos — NOT test data)
# ============================================================================


def generate_demo_binary_data(
    n_samples=200, separation=2.0, noise_std=0.5, seed=42
):
    """Generate 2D binary classification data for demos.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,) with values {0, 1}
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2

    X0 = np.random.randn(n_per_class, 2) * noise_std - separation / 2
    X1 = np.random.randn(n_per_class, 2) * noise_std + separation / 2

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def generate_demo_multiclass_data(n_samples=300, n_classes=3, seed=42):
    """Generate 2D multiclass classification data for demos.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,) with values {0, 1, ..., n_classes-1}
    """
    np.random.seed(seed)
    n_per_class = n_samples // n_classes
    separation = 2.5
    noise_std = 0.5

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


def generate_demo_complex_data(n_samples=600, noise_std=0.4, seed=42):
    """Generate 2D complex classification data (concentric circles) for demos.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,) with values {0, 1}
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2

    theta_inner = np.random.uniform(0, 2 * np.pi, n_per_class)
    r_inner = 0.5 + np.random.randn(n_per_class) * noise_std
    X_inner = np.column_stack(
        [r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)]
    )

    theta_outer = np.random.uniform(0, 2 * np.pi, n_per_class)
    r_outer = 1.5 + np.random.randn(n_per_class) * noise_std
    X_outer = np.column_stack(
        [r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)]
    )

    X = np.vstack([X_inner, X_outer])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


# ============================================================================
# Visualization Functions
# ============================================================================


def plot_decision_boundary_2d(
    X, y, model, title="Decision Boundary", resolution=200
):
    """Plot 2D decision boundary for a classifier.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
    model : object with .predict(X) method
    title : str
    resolution : int
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    Z = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=30)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    accuracy = np.mean(model.predict(X) == y)
    print(f"Training accuracy: {accuracy:.2%}")

    plt.show()


def plot_tree_depth_comparison(X, y, TreeClass, depths=None):
    """Plot decision boundaries at different tree depths.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
    TreeClass : class with __init__(max_depth=...) and .fit/.predict
    depths : list of int
    """
    if depths is None:
        depths = [1, 3, 5, 10]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 150),
        np.linspace(y_min, y_max, 150),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    for ax, depth in zip(axes, depths):
        model = TreeClass(max_depth=depth).fit(X, y)
        Z = model.predict(grid).reshape(xx.shape)
        accuracy = np.mean(model.predict(X) == y)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=20)
        ax.set_title(f"max_depth={depth}  (acc={accuracy:.1%})")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Decision Tree: Effect of max_depth", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_ensemble_vs_single(
    X, y, single_model, ensemble_model,
    single_name="Single Tree", ensemble_name="Ensemble",
):
    """Plot side-by-side decision boundaries comparing single model vs ensemble.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
    single_model : fitted model with .predict(X)
    ensemble_model : fitted model with .predict(X)
    single_name : str
    ensemble_name : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 150),
        np.linspace(y_min, y_max, 150),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    for ax, model, name in [
        (axes[0], single_model, single_name),
        (axes[1], ensemble_model, ensemble_name),
    ]:
        Z = model.predict(grid).reshape(xx.shape)
        accuracy = np.mean(model.predict(X) == y)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=20)
        ax.set_title(f"{name}  (acc={accuracy:.1%})")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_boosting_loss_curve(losses):
    """Plot loss curve over boosting iterations.

    Parameters
    ----------
    losses : list of float
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, "b-", linewidth=2)
    ax.set_xlabel("Boosting Iteration")
    ax.set_ylabel("Log Loss")
    ax.set_title("Gradient Boosting: Loss over Iterations")
    ax.grid(True, alpha=0.3)

    print(f"Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")

    plt.show()


def plot_bootstrap_demo(X, y, n_bootstraps=3):
    """Visualize bootstrap samples showing which points are duplicated.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
    y : ndarray of shape (n_samples,)
    n_bootstraps : int
    """
    fig, axes = plt.subplots(1, n_bootstraps + 1, figsize=(4 * (n_bootstraps + 1), 4))

    # Original data
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=30)
    axes[0].set_title(f"Original (n={len(X)})")
    axes[0].grid(True, alpha=0.3)

    for i in range(n_bootstraps):
        rng = np.random.RandomState(i)
        idx = rng.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X[idx], y[idx]

        unique_count = len(np.unique(idx))
        axes[i + 1].scatter(
            X_boot[:, 0], X_boot[:, 1], c=y_boot, cmap="RdYlBu", edgecolors="k", s=30
        )
        axes[i + 1].set_title(f"Bootstrap {i + 1}\n({unique_count} unique / {len(X)})")
        axes[i + 1].grid(True, alpha=0.3)

    plt.suptitle("Bootstrap Sampling", fontsize=14)
    plt.tight_layout()
    plt.show()
