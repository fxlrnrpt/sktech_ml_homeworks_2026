"""Helper functions for HW3: Deep Learning.

Provides data generation and visualization utilities for the notebook.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_spiral_data(n_samples=300, n_classes=3, noise=0.3, seed=42):
    """Generate 2D spiral data for classification.

    Parameters
    ----------
    n_samples : int
        Total number of samples (divided equally among classes).
    n_classes : int
        Number of spiral arms / classes.
    noise : float
        Standard deviation of Gaussian noise added to the spirals.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
    """
    rng = np.random.RandomState(seed)
    samples_per_class = n_samples // n_classes
    X = np.zeros((n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    for k in range(n_classes):
        ix = range(samples_per_class * k, samples_per_class * (k + 1))
        r = np.linspace(0.0, 1.0, samples_per_class)
        t = np.linspace(k * 4, (k + 1) * 4, samples_per_class) + rng.randn(samples_per_class) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = k
    return X, y


def generate_tiny_text_data():
    """Load or create a small text corpus for character-level LM.

    Returns
    -------
    text : str
        The text corpus (250 characters).
    char_to_idx : dict
        Character to index mapping.
    idx_to_char : dict
        Index to character mapping.
    vocab_size : int
        Number of unique characters.
    """
    data_path = Path(__file__).parent / "test_data" / "tiny_shakespeare.npz"
    data = np.load(data_path, allow_pickle=True)
    text = "".join(data["text"].tolist())
    char_to_idx = json.loads(str(data["char_to_idx_json"]))
    idx_to_char = {int(k): v for k, v in json.loads(str(data["idx_to_char_json"])).items()}
    vocab_size = int(data["vocab_size"])
    return text, char_to_idx, idx_to_char, vocab_size


def encode_text(text, char_to_idx):
    """Convert a string to a LongTensor of character indices.

    Parameters
    ----------
    text : str
    char_to_idx : dict

    Returns
    -------
    tokens : torch.LongTensor of shape (len(text),)
    """
    return torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)


def decode_text(tokens, idx_to_char):
    """Convert a LongTensor of indices back to a string.

    Parameters
    ----------
    tokens : torch.LongTensor
    idx_to_char : dict

    Returns
    -------
    text : str
    """
    return "".join(idx_to_char[t.item()] for t in tokens)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_decision_boundary_2d(X, y, model=None, title="Data", resolution=200):
    """Plot 2D data points and optional decision boundary.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
    model : object with .predict(X) -> labels, or None
    title : str
    resolution : int
    """
    plt.figure(figsize=(8, 6))

    if model is not None:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = model.predict(grid)
        preds = preds.reshape(xx.shape)
        plt.contourf(xx, yy, preds, alpha=0.3, cmap="viridis")

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k", s=20)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.show()


def plot_training_loss(losses, title="Training Loss"):
    """Plot loss curve over training epochs.

    Parameters
    ----------
    losses : list of float
    title : str
    """
    plt.figure(figsize=(8, 4))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_numpy_vs_torch_comparison(np_losses, torch_losses, np_model, torch_model, X, y):
    """Compare NumPy MLP and PyTorch MLP side by side.

    Parameters
    ----------
    np_losses : list of float
    torch_losses : list of float
    np_model : object with .predict(X) -> labels
    torch_model : object with .predict(X) -> labels
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].plot(np_losses, linewidth=2)
    axes[0, 0].set_title("NumPy MLP - Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(torch_losses, linewidth=2, color="orange")
    axes[0, 1].set_title("PyTorch MLP - Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)

    # Decision boundaries
    resolution = 150
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    for ax, model, name in [
        (axes[1, 0], np_model, "NumPy MLP"),
        (axes[1, 1], torch_model, "PyTorch MLP"),
    ]:
        preds = model.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, preds, alpha=0.3, cmap="viridis")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k", s=15)
        acc = np.mean(model.predict(X) == y)
        ax.set_title(f"{name} (acc: {acc:.1%})")

    plt.tight_layout()
    plt.show()


def plot_attention_weights(weights, title="Attention Weights"):
    """Heatmap of attention weights.

    Parameters
    ----------
    weights : np.ndarray of shape (n_queries, n_keys)
    title : str
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(weights, cmap="Blues", aspect="auto")
    plt.colorbar(label="Attention weight")
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_transformer_training(losses):
    """Plot transformer training loss curve.

    Parameters
    ----------
    losses : list of float
    """
    plt.figure(figsize=(8, 4))
    plt.plot(losses, linewidth=2, color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Transformer Language Model Training")
    plt.grid(True, alpha=0.3)
    if len(losses) > 1:
        plt.annotate(
            f"Final: {losses[-1]:.3f}",
            xy=(len(losses) - 1, losses[-1]),
            fontsize=10,
            ha="right",
        )
    plt.tight_layout()
    plt.show()
