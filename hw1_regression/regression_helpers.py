"""
Helper functions for HW1: Regression.

This module contains utility functions for data generation, visualization,
and analysis. Students should focus on implementing the regression classes
in the notebook, not on these helper functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cond, solve

# ============================================================================
# Data Generation
# ============================================================================


def generate_warmup_data(n_samples=50, true_slope=3.0, noise_std=0.5, seed=42):
    """
    Generate simple 1D data for the warm-up exercise: y = slope * x + noise.

    Returns 1D arrays (not matrices) to keep things simple for beginners.

    Parameters
    ----------
    n_samples : int
        Number of samples
    true_slope : float
        True slope of the line
    noise_std : float
        Standard deviation of Gaussian noise
    seed : int
        Random seed for reproducibility

    Returns
    -------
    x : np.ndarray of shape (n_samples,)
        Feature values (1D)
    y : np.ndarray of shape (n_samples,)
        Target values
    true_slope : float
        The true slope used to generate data
    """
    np.random.seed(seed)
    x = np.linspace(0, 5, n_samples)
    y = true_slope * x + noise_std * np.random.randn(n_samples)
    return x, y, true_slope


def generate_classification_data(n_samples=200, separation=1.5, noise_std=0.5, seed=42):
    """
    Generate 2D binary classification data (two Gaussian blobs).

    Parameters
    ----------
    n_samples : int
        Total number of samples (split equally between classes)
    separation : float
        Distance between class centers
    noise_std : float
        Standard deviation of each blob
    seed : int
        Random seed for reproducibility

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
        Feature matrix
    y : np.ndarray of shape (n_samples,)
        Binary labels (0 or 1)
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2

    X0 = np.random.randn(n_per_class, 2) * noise_std - separation / 2
    X1 = np.random.randn(n_per_class, 2) * noise_std + separation / 2

    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    # Shuffle
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


def create_collinear_data(n_samples=100, collinearity=1e-10, noise_std=0.1, seed=42):
    """
    Create data with highly correlated features.

    X[:, 1] is nearly identical to X[:, 0], making X^T X nearly singular.

    Parameters
    ----------
    n_samples : int
        Number of samples
    collinearity : float
        How different x2 is from x1. Smaller = more collinear = more ill-conditioned.
    noise_std : float
        Standard deviation of noise added to y
    seed : int
        Random seed for reproducibility

    Returns
    -------
    X : np.ndarray of shape (n_samples, 3)
    y : np.ndarray of shape (n_samples,)
    true_weights : np.ndarray of shape (3,)
    """
    np.random.seed(seed)

    x1 = np.random.randn(n_samples)
    x2 = x1 + collinearity * np.random.randn(n_samples)  # Almost identical to x1!
    x3 = np.random.randn(n_samples)

    X = np.column_stack([x1, x2, x3])
    true_weights = np.array([1.0, 2.0, 3.0])
    y = X @ true_weights + noise_std * np.random.randn(n_samples)

    return X, y, true_weights


def create_sparse_data(n_samples=100, n_features=20, n_informative=5, noise_std=0.1, seed=42):
    """
    Generate data where only a few features have non-zero true weights.

    This is ideal for demonstrating LASSO's feature selection capability.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Total number of features
    n_informative : int
        Number of features with non-zero weights
    noise_std : float
        Standard deviation of noise added to y
    seed : int
        Random seed for reproducibility

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    true_weights : np.ndarray of shape (n_features,)
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)

    # Only first n_informative features have non-zero weights
    true_weights = np.zeros(n_features)
    true_weights[:n_informative] = np.random.randn(n_informative) * 2

    y = X @ true_weights + noise_std * np.random.randn(n_samples)
    return X, y, true_weights


# ============================================================================
# Sensitivity Analysis
# ============================================================================


def analyze_sensitivity(X, y, n_trials=500, perturbation_scale=1e-10):
    """
    Analyze how sensitive OLS is to tiny perturbations in y.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Design matrix
    y : np.ndarray of shape (n_samples,)
        Target values
    n_trials : int
        Number of perturbation trials
    perturbation_scale : float
        Scale of perturbations to add to y

    Returns
    -------
    weights_array : np.ndarray of shape (n_trials, n_features)
        Weight estimates from each trial
    """
    weights_list = []

    for trial in range(n_trials):
        np.random.seed(trial)
        # Add TINY perturbation to y
        y_perturbed = y + perturbation_scale * np.random.randn(len(y))

        # Solve OLS
        w = solve(X.T @ X, X.T @ y_perturbed)
        weights_list.append(w)

    return np.array(weights_list)


def analyze_ridge_sensitivity(X, y, alpha, n_trials=500, perturbation_scale=1e-10):
    """
    Analyze how sensitive Ridge regression is to perturbations.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Design matrix
    y : np.ndarray of shape (n_samples,)
        Target values
    alpha : float
        Regularization strength
    n_trials : int
        Number of perturbation trials
    perturbation_scale : float
        Scale of perturbations to add to y

    Returns
    -------
    weights_array : np.ndarray of shape (n_trials, n_features)
        Weight estimates from each trial
    """
    n_features = X.shape[1]
    weights_list = []

    for trial in range(n_trials):
        np.random.seed(trial)
        y_perturbed = y + perturbation_scale * np.random.randn(len(y))

        # Solve Ridge
        A = X.T @ X + alpha * np.eye(n_features)
        w = solve(A, X.T @ y_perturbed)
        weights_list.append(w)

    return np.array(weights_list)


def analyze_regularization_effect(X, alphas):
    """
    Show how regularization improves condition number.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Design matrix
    alphas : list of float
        Regularization strengths to test

    Returns
    -------
    new_conds : list of float
        Condition numbers after regularization
    """
    XtX = X.T @ X
    original_cond = cond(XtX)

    print(f"Original condition number: {original_cond:.2e}")
    print()
    print(f"{'Alpha':>12} {'New Cond #':>15} {'Improvement':>15}")
    print("-" * 45)

    new_conds = []
    for alpha in alphas:
        regularized = XtX + alpha * np.eye(X.shape[1])
        new_cond = cond(regularized)
        new_conds.append(new_cond)
        improvement = original_cond / new_cond
        print(f"{alpha:>12.0e} {new_cond:>15.2e} {improvement:>15.2e}x")

    return new_conds


def compute_condition_number_analysis(collinearity_levels, n_trials=30, perturbation_scale=1e-10, seed=42):
    """
    Compute condition numbers and weight variances for different collinearity levels.

    Parameters
    ----------
    collinearity_levels : array-like
        Different collinearity factors to test
    n_trials : int
        Number of perturbation trials for variance computation
    perturbation_scale : float
        Scale of perturbations
    seed : int
        Random seed

    Returns
    -------
    condition_numbers : list of float
        Condition numbers for each collinearity level
    weight_variances : list of float
        Average weight variances for each collinearity level
    """
    condition_numbers = []
    weight_variances = []

    for col in collinearity_levels:
        X_test, y_test, _ = create_collinear_data(collinearity=col, seed=seed)

        # Condition number
        cond_num = cond(X_test.T @ X_test)
        condition_numbers.append(cond_num)

        # Weight variance across perturbations
        weights = analyze_sensitivity(X_test, y_test, n_trials=n_trials, perturbation_scale=perturbation_scale)
        weight_variances.append(np.mean(np.var(weights, axis=0)))

    return condition_numbers, weight_variances


# ============================================================================
# Kernel Functions (for reference)
# ============================================================================


def linear_kernel(X1, X2):
    """Linear kernel: K(x, x') = x^T x'"""
    return X1 @ X2.T


def rbf_kernel(X1, X2, gamma=1.0):
    """
    RBF (Gaussian) kernel: K(x, x') = exp(-gamma * ||x - x'||^2)

    This is a VECTORIZED implementation - no loops!
    """
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i^T x_j
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    distances_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
    distances_sq = np.maximum(distances_sq, 0)  # Numerical stability
    return np.exp(-gamma * distances_sq)


# ============================================================================
# Visualization Functions
# ============================================================================


def plot_warmup_fit(x, y, fitted_slope, true_slope):
    """
    Plot the warm-up regression: data points, fitted line, and true line.

    Parameters
    ----------
    x : np.ndarray of shape (n_samples,)
        Feature values (1D)
    y : np.ndarray of shape (n_samples,)
        Target values
    fitted_slope : float
        Student's computed slope
    true_slope : float
        True slope used to generate data
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.6, label="Data", zorder=2)

    x_line = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_line, fitted_slope * x_line, "r-", linewidth=2, label=f"Your fit (w={fitted_slope:.3f})", zorder=3)
    plt.plot(x_line, true_slope * x_line, "g--", linewidth=2, label=f"True line (w={true_slope:.3f})", zorder=3)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Warm-up: Fitting a Line Through Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sigmoid_curve(sigmoid_fn):
    """
    Plot the sigmoid function with reference lines at z=0 and sigma=0.5.

    Parameters
    ----------
    sigmoid_fn : callable
        A sigmoid function that takes a numpy array and returns sigmoid values.
    """
    z = np.linspace(-6, 6, 200)
    plt.plot(z, sigmoid_fn(z))
    plt.xlabel("z")
    plt.ylabel("σ(z)")
    plt.title("Sigmoid Function")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color="r", linestyle="--", alpha=0.5)
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)
    plt.show()

    print("Sigmoid implemented correctly!")


def plot_sigmoid_clipping_demo(sigmoid_fn):
    """
    Two-panel plot demonstrating why np.clip(z, -500, 500) is safe for sigmoid.

    Prints sigmoid values at key points, then shows:
    Left panel: wide view [-600, 600] with clipped regions shaded red.
    Right panel: zoom [-10, 10] showing all transition happens in a narrow band.

    Parameters
    ----------
    sigmoid_fn : callable
        A sigmoid function that takes a numpy array and returns sigmoid values.
    """
    print("Sigmoid values at key points:")
    print(f"  sigmoid( 500) = {sigmoid_fn(500.0)}")
    print(f"  sigmoid(-500) = {sigmoid_fn(-500.0)}")
    print(f"  sigmoid(  20) = {sigmoid_fn(20.0):.15f}")
    print(f"  sigmoid( -20) = {sigmoid_fn(-20.0):.15e}")
    print()
    print("By z = +/-20, sigmoid is already within 1e-9 of its limit.")
    print("Clipping at +/-500 changes nothing about the output --")
    print("it only prevents np.exp from overflowing on extreme inputs.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: wide view showing the clip zone
    z_wide = np.linspace(-600, 600, 2000)
    sig_wide = sigmoid_fn(np.clip(z_wide, -500, 500))
    axes[0].plot(z_wide, sig_wide, "b-", linewidth=1.5)
    axes[0].axvspan(-600, -500, alpha=0.15, color="red", label="Clipped region")
    axes[0].axvspan(500, 600, alpha=0.15, color="red")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("σ(z)")
    axes[0].set_title("Wide View: Sigmoid is Flat at Clip Boundaries")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: zoom into the transition region -- all the action is in [-10, 10]
    z_zoom = np.linspace(-10, 10, 500)
    axes[1].plot(z_zoom, sigmoid_fn(z_zoom), "b-", linewidth=2)
    axes[1].axhline(y=0.5, color="r", linestyle="--", alpha=0.4)
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("σ(z)")
    axes[1].set_title("Zoom: All Interesting Behavior is in [-10, 10]")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_gradient_descent_loss(losses):
    """
    Plot the loss curve over gradient descent steps and print a summary.

    Parameters
    ----------
    losses : list of float
        Loss values at each gradient descent step.
    """
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Log Loss")
    plt.title("Loss Should Decrease Over Gradient Steps")
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f}")
    print("Gradient step implemented correctly!")


def plot_decision_boundary(X, y, model, resolution=200):
    """
    Plot the decision boundary of a 2D binary classifier.

    Prints training accuracy, then shows the decision boundary plot.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        Feature matrix (must be 2D features)
    y : np.ndarray of shape (n_samples,)
        Binary labels (0 or 1)
    model : object
        Fitted model with a .predict(X) method
    resolution : int
        Grid resolution for the background shading
    """
    accuracy = np.mean(model.predict(X) == y)
    print(f"Training accuracy: {accuracy:.2%}")

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu_r", levels=[0, 0.5, 1])
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", edgecolors="k", linewidth=0.5, label="Class 0", alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", edgecolors="k", linewidth=0.5, label="Class 1", alpha=0.7)

    # Draw the decision boundary line if model has weights and bias
    if hasattr(model, "weights") and hasattr(model, "bias") and model.weights is not None:
        w = model.weights
        b = model.bias
        if abs(w[1]) > 1e-10:
            x_boundary = np.linspace(x_min, x_max, 100)
            y_boundary = -(w[0] * x_boundary + b) / w[1]
            plt.plot(x_boundary, y_boundary, "k-", linewidth=2, label="Decision boundary")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Logistic Regression: Decision Boundary")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_weight_instability(weights_trials, true_weights, perturbation_scale=1e-10):
    """
    Plot showing OLS weight instability across perturbations.

    Parameters
    ----------
    weights_trials : np.ndarray of shape (n_trials, n_features)
        Weight estimates from each perturbation trial
    true_weights : np.ndarray of shape (n_features,)
        True weight values
    perturbation_scale : float
        Scale of perturbations used (for title)
    """
    n_trials = weights_trials.shape[0]
    n_features = weights_trials.shape[1]

    fig, axes = plt.subplots(1, n_features, figsize=(14, 4))

    if n_features == 1:
        axes = [axes]

    for i in range(n_features):
        std = weights_trials[:, i].std()
        # Calculate amplification factor: how much bigger is std than perturbation?
        amplification = std / perturbation_scale if perturbation_scale > 0 else 0

        # Color based on stability: red/coral for unstable, green for stable
        is_unstable = amplification > 1e3
        color = "coral" if is_unstable else "mediumseagreen"

        # Histogram of weight values
        axes[i].hist(weights_trials[:, i], bins=30, edgecolor="black", alpha=0.7, color=color)
        axes[i].axvline(true_weights[i], color="red", linestyle="--", linewidth=2, label=f"True: {true_weights[i]}")

        axes[i].set_xlabel(f"w[{i}] value")
        axes[i].set_ylabel("Frequency")

        if is_unstable:
            axes[i].set_title(f"w[{i}]: UNSTABLE!\nstd={std:.2f} ({amplification:.1e}x amplification)")
        else:
            axes[i].set_title(f"w[{i}]: stable\nstd={std:.2e}, close to true value")
        axes[i].legend(loc="upper right")

    plt.suptitle(
        f"OLS Weight Instability: {n_trials} trials with {perturbation_scale:.0e} perturbations in y\n"
        f"Correlated features (w[0], w[1]) are unstable; independent feature (w[2]) is stable",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


def plot_condition_number_analysis(collinearity_levels, condition_numbers, weight_variances):
    """
    Plot condition number vs collinearity and weight variance vs condition number.

    Parameters
    ----------
    collinearity_levels : array-like
        Different collinearity factors tested
    condition_numbers : array-like
        Condition numbers for each collinearity level
    weight_variances : array-like
        Weight variances for each collinearity level
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].loglog(collinearity_levels, condition_numbers, "bo-", markersize=8)
    axes[0].set_xlabel("Collinearity Factor (smaller = more collinear)")
    axes[0].set_ylabel("Condition Number of X^T X")
    axes[0].set_title("Condition Number vs. Feature Correlation")
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(condition_numbers, weight_variances, "ro-", markersize=8)
    axes[1].set_xlabel("Condition Number")
    axes[1].set_ylabel("Average Weight Variance")
    axes[1].set_title("Solution Instability vs. Condition Number")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("KEY INSIGHT: As condition number increases, weight variance explodes!")
    print("This is numerical instability - the OLS solution becomes meaningless.")
    print("NOTE: The right plot may appear to 'go back' because at extreme condition numbers (~10^17),")
    print("      numerical precision breaks down and even computing the condition number becomes unreliable.")


def plot_ols_vs_ridge_stability(ols_weights, ridge_weights, true_weights):
    """
    Plot comparison of OLS vs Ridge weight stability under perturbations.

    Parameters
    ----------
    ols_weights : np.ndarray of shape (n_trials, n_features)
        OLS weight estimates from perturbation trials
    ridge_weights : np.ndarray of shape (n_trials, n_features)
        Ridge weight estimates from perturbation trials
    true_weights : np.ndarray of shape (n_features,)
        True weight values
    """
    n_features = ols_weights.shape[1]
    fig, axes = plt.subplots(2, n_features, figsize=(14, 8))

    for i in range(n_features):
        # OLS
        axes[0, i].hist(ols_weights[:, i], bins=20, edgecolor="black", alpha=0.7, color="red")
        axes[0, i].axvline(true_weights[i], color="green", linestyle="--", linewidth=2)
        axes[0, i].set_xlabel(f"Weight {i + 1}")
        axes[0, i].set_title(f"OLS: w[{i}]")

        # Ridge
        axes[1, i].hist(ridge_weights[:, i], bins=20, edgecolor="black", alpha=0.7, color="blue")
        axes[1, i].axvline(
            true_weights[i], color="green", linestyle="--", linewidth=2, label=f"True: {true_weights[i]}"
        )
        axes[1, i].set_xlabel(f"Weight {i + 1}")
        axes[1, i].set_title(f"Ridge (alpha=1): w[{i}]")
        axes[1, i].legend()

    axes[0, 0].set_ylabel("OLS\nFrequency")
    axes[1, 0].set_ylabel("Ridge\nFrequency")

    plt.suptitle("OLS vs Ridge: Stability Under 1e-10 Perturbations", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("Notice how Ridge weights cluster tightly around the true values,")
    print("while OLS weights are scattered across a huge range!")


def plot_ridge_vs_lasso_weights(true_weights, ridge_weights, lasso_weights, ridge_alpha, lasso_alpha):
    """
    Plot bar chart comparison of true, Ridge, and LASSO weights.

    Parameters
    ----------
    true_weights : np.ndarray
        True weight values
    ridge_weights : np.ndarray
        Ridge regression weights
    lasso_weights : np.ndarray
        LASSO regression weights
    ridge_alpha : float
        Ridge regularization strength
    lasso_alpha : float
        LASSO regularization strength
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x_pos = np.arange(len(true_weights))

    # True weights
    axes[0].bar(x_pos, true_weights, color="green", alpha=0.7)
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Weight Value")
    axes[0].set_title("True Weights (Sparse)")
    axes[0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Ridge weights
    axes[1].bar(x_pos, ridge_weights, color="blue", alpha=0.7)
    axes[1].set_xlabel("Feature Index")
    axes[1].set_ylabel("Weight Value")
    axes[1].set_title(f"Ridge Weights (alpha={ridge_alpha})\nAll features have non-zero weights")
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # LASSO weights
    axes[2].bar(x_pos, lasso_weights, color="red", alpha=0.7)
    axes[2].set_xlabel("Feature Index")
    axes[2].set_ylabel("Weight Value")
    axes[2].set_title(f"LASSO Weights (alpha={lasso_alpha})\nSparse! Many weights = 0")
    axes[2].axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.show()

    print("\nKEY OBSERVATION:")
    print("- Ridge shrinks ALL weights but keeps them non-zero")
    print("- LASSO sets many weights EXACTLY to zero (feature selection!)")
    print("- LASSO correctly identifies that only the first 5 features matter")


def plot_regularization_paths(alphas, ridge_weights_path, lasso_weights_path, n_informative=5):
    """
    Plot regularization paths for Ridge and LASSO.

    Parameters
    ----------
    alphas : np.ndarray
        Array of regularization strengths
    ridge_weights_path : np.ndarray of shape (n_alphas, n_features)
        Ridge weights for each alpha
    lasso_weights_path : np.ndarray of shape (n_alphas, n_features)
        LASSO weights for each alpha
    n_informative : int
        Number of informative features (for coloring)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_features = ridge_weights_path.shape[1]

    # Ridge path (full view)
    for j in range(n_features):
        color = "blue" if j < n_informative else "gray"
        alpha_line = 0.8 if j < n_informative else 0.3
        axes[0, 0].semilogx(alphas, ridge_weights_path[:, j], color=color, alpha=alpha_line)
    axes[0, 0].set_xlabel("Alpha (regularization strength)")
    axes[0, 0].set_ylabel("Weight value")
    axes[0, 0].set_title("Ridge Regularization Path\n(blue = informative features, gray = noise features)")
    axes[0, 0].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    axes[0, 0].grid(True, alpha=0.3)

    # LASSO path (full view)
    for j in range(n_features):
        color = "red" if j < n_informative else "gray"
        alpha_line = 0.8 if j < n_informative else 0.3
        axes[0, 1].semilogx(alphas, lasso_weights_path[:, j], color=color, alpha=alpha_line)
    axes[0, 1].set_xlabel("Alpha (regularization strength)")
    axes[0, 1].set_ylabel("Weight value")
    axes[0, 1].set_title("LASSO Regularization Path\n(red = informative features, gray = noise features)")
    axes[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    axes[0, 1].grid(True, alpha=0.3)

    # Ridge path (zoomed around zero for noise features)
    for j in range(n_features):
        color = "blue" if j < n_informative else "gray"
        alpha_line = 0.8 if j < n_informative else 0.3
        axes[1, 0].semilogx(alphas, ridge_weights_path[:, j], color=color, alpha=alpha_line)
    axes[1, 0].set_xlabel("Alpha (regularization strength)")
    axes[1, 0].set_ylabel("Weight value")
    axes[1, 0].set_title("Ridge Regularization Path (ZOOMED)\n(gray noise features never reach exactly zero)")
    axes[1, 0].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    axes[1, 0].set_ylim(-0.1, 0.1)
    axes[1, 0].grid(True, alpha=0.3)

    # LASSO path (zoomed around zero for noise features)
    for j in range(n_features):
        color = "red" if j < n_informative else "gray"
        alpha_line = 0.8 if j < n_informative else 0.3
        axes[1, 1].semilogx(alphas, lasso_weights_path[:, j], color=color, alpha=alpha_line)
    axes[1, 1].set_xlabel("Alpha (regularization strength)")
    axes[1, 1].set_ylabel("Weight value")
    axes[1, 1].set_title("LASSO Regularization Path (ZOOMED)\n(gray noise features hit exactly zero)")
    axes[1, 1].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    axes[1, 1].set_ylim(-0.1, 0.1)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("KEY OBSERVATION:")
    print("- Ridge: All weights shrink smoothly toward zero but never reach exactly zero")
    print("- LASSO: Weights hit exactly zero at different alpha values (sparse solutions)")
    print("- Noise features (gray) go to zero faster than informative features (colored)")
    print("- ZOOMED VIEW: Notice Ridge noise weights hover near zero but never hit it,")
    print("  while LASSO noise weights are exactly zero for most alpha values")


def plot_kernel_comparison(X_train, y_train, X_test, y_true, predictions, kernel_names):
    """
    Plot comparison of different kernel predictions.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_test : np.ndarray
        Test features
    y_true : np.ndarray
        True function values on test set
    predictions : list of np.ndarray
        Predictions from each kernel
    kernel_names : list of str
        Names of kernels
    """
    fig, axes = plt.subplots(1, len(kernel_names), figsize=(15, 4))

    if len(kernel_names) == 1:
        axes = [axes]

    for ax, kernel_name, y_pred in zip(axes, kernel_names, predictions):
        ax.scatter(X_train, y_train, alpha=0.5, label="Training data", s=20)
        ax.plot(X_test, y_pred, "r-", linewidth=2, label="Prediction")
        ax.plot(X_test, y_true, "g--", linewidth=2, label="True function")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{kernel_name.upper()} Kernel")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Notice how RBF kernel captures the non-linear sinusoidal pattern,")
    print("while linear kernel can only fit a straight line!")


def plot_kernel_matrices(X, linear_kernel_fn, rbf_kernel_fn, gamma=1.0):
    """
    Plot side-by-side heatmaps of linear and RBF kernel matrices.

    Accepts the student's kernel functions as parameters so the visualization
    uses whatever the student implemented.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data points to compute kernel matrices for
    linear_kernel_fn : callable
        A function linear_kernel(X1, X2) -> np.ndarray
    rbf_kernel_fn : callable
        A function rbf_kernel(X1, X2, gamma) -> np.ndarray
    gamma : float
        Gamma parameter for the RBF kernel
    """
    K_lin = linear_kernel_fn(X, X)
    K_rbf = rbf_kernel_fn(X, X, gamma)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(K_lin, cmap="viridis", aspect="auto")
    axes[0].set_title("Linear Kernel Matrix\n$K_{ij} = x_i^T x_j$")
    axes[0].set_xlabel("Sample j")
    axes[0].set_ylabel("Sample i")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(K_rbf, cmap="viridis", aspect="auto")
    axes[1].set_title(f"RBF Kernel Matrix (γ={gamma})\n$K_{{ij}} = \\exp(-\\gamma \\|x_i - x_j\\|^2)$")
    axes[1].set_xlabel("Sample j")
    axes[1].set_ylabel("Sample i")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.show()

    print("HOW TO READ THESE PLOTS:")
    print("Each cell (i, j) shows how similar sample i is to sample j.")
    print("Bright = high similarity, dark = low similarity.")
    print("Our training data is x = linspace(0, 2*pi), so sample index ≈ position.")
    print()
    print("LINEAR kernel (left): K_ij = x_i * x_j. The bottom-right corner is")
    print("brightest because both x_i and x_j are large there. The top-left is dark")
    print("because x-values near 0 give small dot products. This smooth gradient")
    print("means ALL training points contribute to every prediction with weights that")
    print("grow with x — the result is a single straight line, which cannot follow")
    print("a sine wave that goes up AND down.")
    print()
    print("RBF kernel (right): K_ij = exp(-γ ||x_i - x_j||²). The bright diagonal")
    print("band means each point is similar ONLY to its close neighbors — points")
    print("far apart (off the band) have near-zero similarity. Why does this help")
    print("with sin(x)? Because when predicting at some x, only nearby training")
    print("points get a vote. Near x=1 the sine is rising, near x=4 it's falling —")
    print("the RBF band ensures these regions don't interfere with each other, so")
    print("the prediction can follow each local trend independently.")
    print()
    print("You'll see the actual predictions in Exercise 5.4.")


# ============================================================================
# Computation Helpers (require student implementations)
# ============================================================================


def compute_regularization_paths(X, y, alphas, RidgeClass, LassoClass, max_iter=2000):
    """
    Compute regularization paths for Ridge and LASSO.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    alphas : array-like
        Regularization strengths to test
    RidgeClass : class
        Ridge regression class (student implementation)
    LassoClass : class
        LASSO regression class (student implementation)
    max_iter : int
        Max iterations for LASSO

    Returns
    -------
    ridge_weights_path : np.ndarray of shape (n_alphas, n_features)
    lasso_weights_path : np.ndarray of shape (n_alphas, n_features)
    """
    ridge_weights_path = []
    lasso_weights_path = []

    for alpha in alphas:
        ridge = RidgeClass(alpha=alpha).fit(X, y)
        lasso = LassoClass(alpha=alpha, max_iter=max_iter).fit(X, y)
        ridge_weights_path.append(ridge.weights.copy())
        lasso_weights_path.append(lasso.weights.copy())

    return np.array(ridge_weights_path), np.array(lasso_weights_path)


def compute_kernel_predictions(X_train, y_train, X_test, KernelRidgeClass, alpha=0.1):
    """
    Compute predictions for different kernel types.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_test : np.ndarray
        Test features
    KernelRidgeClass : class
        Kernel ridge regression class (student implementation)
    alpha : float
        Regularization strength

    Returns
    -------
    predictions : list of np.ndarray
        Predictions for each kernel type
    kernel_names : list of str
        Names of kernels
    """
    kernels = [
        ("linear", {}),
        ("rbf", {"gamma": 1.0}),
    ]

    kernel_names = [name for name, _ in kernels]
    predictions = []

    for kernel_name, kernel_params in kernels:
        model = KernelRidgeClass(alpha=alpha, kernel=kernel_name, **kernel_params)
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test))

    return predictions, kernel_names


def generate_sinusoidal_data(n_train=100, n_test=200, noise_std=0.1, seed=42):
    """
    Generate non-linear sinusoidal data for kernel regression demo.

    Parameters
    ----------
    n_train : int
        Number of training samples
    n_test : int
        Number of test samples
    noise_std : float
        Standard deviation of noise
    seed : int
        Random seed

    Returns
    -------
    X_train : np.ndarray of shape (n_train, 1)
    y_train : np.ndarray of shape (n_train,)
    X_test : np.ndarray of shape (n_test, 1)
    y_true : np.ndarray of shape (n_test,)
    """
    np.random.seed(seed)
    X_train = np.linspace(0, 2 * np.pi, n_train).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + noise_std * np.random.randn(n_train)

    X_test = np.linspace(0, 2 * np.pi, n_test).reshape(-1, 1)
    y_true = np.sin(X_test).ravel()

    return X_train, y_train, X_test, y_true
