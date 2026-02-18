"""
Test harness for HW1: Regression implementations.

Import into Jupyter notebook and run tests on student implementations.

Example usage in notebook:
    from regression_tests import run_single_test, check_linear_regression
    run_single_test(LinearRegression, check_linear_regression)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Data Loading (from pre-generated test data)
# ============================================================================

_TEST_DATA_DIR = Path(__file__).parent / "test_data"


def _load_test_data(name: str) -> dict:
    """Load pre-generated test data from .npz file."""
    return dict(np.load(_TEST_DATA_DIR / f"{name}.npz"))


def load_well_conditioned_data() -> tuple:
    """Load well-conditioned regression data."""
    data = _load_test_data("well_conditioned")
    return data["X"], data["y"], data["true_weights"]


def load_collinear_data() -> tuple:
    """Load ill-conditioned data with collinear features."""
    data = _load_test_data("collinear")
    return data["X"], data["y"], data["true_weights"]


def load_nonlinear_data() -> tuple:
    """Load non-linear (sinusoidal) data."""
    data = _load_test_data("nonlinear")
    return data["X"], data["y"]


def load_sparse_data() -> tuple:
    """Load data with sparse true weights."""
    data = _load_test_data("sparse")
    return data["X"], data["y"], data["true_weights"]


def load_perfect_fit_data() -> tuple:
    """Load data with perfect linear relationship (no noise)."""
    data = _load_test_data("perfect_fit")
    return data["X"], data["y"], data["true_weights"]


def load_classification_data() -> tuple:
    """Load binary classification data."""
    data = _load_test_data("classification")
    return data["X"], data["y"]


# ============================================================================
# Test Name Constants (single source of truth for test names)
# ============================================================================

LINEAR_REGRESSION_TESTS = (
    "fit_returns_self",
    "weights_shape",
    "predict_shape",
    "correctness",
    "perfect_fit",
)

RIDGE_REGRESSION_TESTS = (
    "fit_returns_self",
    "weights_shape",
    "shrinks_weights",
    "numerical_stability",
    "condition_improvement",
)

LASSO_REGRESSION_TESTS = (
    "fit_returns_self",
    "weights_shape",
    "sparsity",
    "more_sparsity",
    "feature_selection",
)

KERNEL_RIDGE_REGRESSION_TESTS = (
    "fit_returns_self",
    "predict_shape",
    "rbf_fit_quality",
    "linear_kernel_fit",
    "rbf_vs_linear",
    "generalization",
)

LOGISTIC_REGRESSION_TESTS = (
    "fit_returns_self",
    "weights_shape",
    "predict_proba_valid",
    "predict_labels",
    "accuracy",
    "sigmoid",
)


# ============================================================================
# Test Functions (can be called directly with class implementations)
# ============================================================================


def _plot_predictions_vs_actual(y_true, y_pred, title, ax=None):
    """Plot predictions vs actual values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)

    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² and MSE annotations
    mse = np.mean((y_true - y_pred) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMSE = {mse:.4f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return ax


def check_linear_regression(LinearRegressionClass) -> list:
    """Test LinearRegression implementation. Returns list of (test_name, passed, message)."""
    T = LINEAR_REGRESSION_TESTS
    results = []
    X, y, true_weights = load_well_conditioned_data()

    # Test 1: fit returns self
    try:
        model = LinearRegressionClass()
        result = model.fit(X, y)
        passed = result is model
        results.append((T[0], passed, "fit() should return self"))
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: weights shape
    try:
        model = LinearRegressionClass().fit(X, y)
        passed = model.weights.shape == (X.shape[1],)
        results.append((T[1], passed, f"Expected shape {(X.shape[1],)}, got {model.weights.shape}"))
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: predict shape
    try:
        model = LinearRegressionClass().fit(X, y)
        y_pred = model.predict(X)
        passed = y_pred.shape == (X.shape[0],)
        results.append((T[2], passed, f"Expected shape {(X.shape[0],)}, got {y_pred.shape}"))
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: correctness
    try:
        model = LinearRegressionClass().fit(X, y)
        passed = np.allclose(model.weights, true_weights, atol=0.5)
        results.append((T[3], passed, "Weights should be close to true weights"))
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: perfect fit (no noise)
    try:
        X_pf, y_pf, true_w = load_perfect_fit_data()
        model = LinearRegressionClass().fit(X_pf, y_pf)
        passed = np.allclose(model.weights, true_w, atol=1e-10)
        results.append((T[4], passed, "Should fit perfectly when y = Xw exactly"))
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Visualization: predictions vs actual
    try:
        model = LinearRegressionClass().fit(X, y)
        y_pred = model.predict(X)
        _plot_predictions_vs_actual(y, y_pred, "LinearRegression: Predictions vs Actual")
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


def check_ridge_regression(RidgeRegressionClass) -> list:
    """Test RidgeRegression implementation. Returns list of (test_name, passed, message)."""
    T = RIDGE_REGRESSION_TESTS
    results = []
    X, y, true_weights = load_well_conditioned_data()
    X_bad, y_bad, _ = load_collinear_data()

    # Test 1: fit returns self
    try:
        model = RidgeRegressionClass(alpha=1.0)
        result = model.fit(X, y)
        passed = result is model
        results.append((T[0], passed, "fit() should return self"))
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: weights shape
    try:
        model = RidgeRegressionClass(alpha=1.0).fit(X, y)
        passed = model.weights.shape == (X.shape[1],)
        results.append((T[1], passed, f"Expected shape {(X.shape[1],)}, got {model.weights.shape}"))
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: shrinks weights
    try:
        ridge_low = RidgeRegressionClass(alpha=0.1).fit(X, y)
        ridge_high = RidgeRegressionClass(alpha=10.0).fit(X, y)
        passed = np.linalg.norm(ridge_high.weights) < np.linalg.norm(ridge_low.weights)
        results.append((T[2], passed, "Higher alpha should shrink weights more"))
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: numerical stability
    try:
        ridge = RidgeRegressionClass(alpha=1.0).fit(X_bad, y_bad)
        passed = np.linalg.norm(ridge.weights) < 1e6
        results.append((T[3], passed, "Ridge should handle ill-conditioned data"))
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: condition number improvement
    try:
        XtX = X_bad.T @ X_bad
        original_cond = np.linalg.cond(XtX)
        regularized_cond = np.linalg.cond(XtX + 1.0 * np.eye(X_bad.shape[1]))
        passed = regularized_cond < original_cond
        results.append((T[4], passed, "Regularization should improve condition number"))
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Visualization: predictions vs actual
    try:
        model = RidgeRegressionClass(alpha=1.0).fit(X, y)
        y_pred = model.predict(X)
        _plot_predictions_vs_actual(y, y_pred, "RidgeRegression: Predictions vs Actual")
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


def check_lasso_regression(LassoRegressionClass) -> list:
    """Test LassoRegression implementation. Returns list of (test_name, passed, message)."""
    T = LASSO_REGRESSION_TESTS
    results = []
    X, y, _ = load_well_conditioned_data()
    X_sparse, y_sparse, true_weights_sparse = load_sparse_data()

    # Test 1: fit returns self
    try:
        model = LassoRegressionClass(alpha=0.1)
        result = model.fit(X, y)
        passed = result is model
        results.append((T[0], passed, "fit() should return self"))
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: weights shape
    try:
        model = LassoRegressionClass(alpha=0.1).fit(X, y)
        passed = model.weights.shape == (X.shape[1],)
        results.append((T[1], passed, f"Expected shape {(X.shape[1],)}, got {model.weights.shape}"))
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: produces sparsity
    try:
        model = LassoRegressionClass(alpha=0.1).fit(X_sparse, y_sparse)
        n_zero = np.sum(np.abs(model.weights) < 1e-6)
        passed = n_zero > 0
        results.append((T[2], passed, f"LASSO should produce sparse weights, got {n_zero} zeros"))
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: stronger regularization = more sparsity
    try:
        lasso_low = LassoRegressionClass(alpha=0.01).fit(X_sparse, y_sparse)
        lasso_high = LassoRegressionClass(alpha=0.5).fit(X_sparse, y_sparse)
        n_zero_low = np.sum(np.abs(lasso_low.weights) < 1e-6)
        n_zero_high = np.sum(np.abs(lasso_high.weights) < 1e-6)
        passed = n_zero_high >= n_zero_low
        results.append((T[3], passed, "Higher alpha should produce more zeros"))
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: identifies informative features
    try:
        model = LassoRegressionClass(alpha=0.05).fit(X_sparse, y_sparse)
        informative_weights = np.abs(model.weights[:5])
        non_informative_weights = np.abs(model.weights[5:])
        passed = informative_weights.mean() > non_informative_weights.mean()
        results.append((T[4], passed, "Should identify informative features"))
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Visualization: predictions vs actual
    try:
        model = LassoRegressionClass(alpha=0.1).fit(X_sparse, y_sparse)
        y_pred = model.predict(X_sparse)
        _plot_predictions_vs_actual(y_sparse, y_pred, "LassoRegression: Predictions vs Actual")
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


def check_kernel_ridge_regression(KernelRidgeRegressionClass) -> list:
    """Test KernelRidgeRegression implementation. Returns list of (test_name, passed, message)."""
    T = KERNEL_RIDGE_REGRESSION_TESTS
    results = []
    X, y, _ = load_well_conditioned_data()
    X_nl, y_nl = load_nonlinear_data()

    # Test 1: fit returns self
    try:
        model = KernelRidgeRegressionClass(alpha=1.0, kernel="rbf")
        result = model.fit(X, y)
        passed = result is model
        results.append((T[0], passed, "fit() should return self"))
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: predict shape
    try:
        model = KernelRidgeRegressionClass(alpha=1.0, kernel="rbf").fit(X, y)
        y_pred = model.predict(X)
        passed = y_pred.shape == (X.shape[0],)
        results.append((T[1], passed, f"Expected shape {(X.shape[0],)}, got {y_pred.shape}"))
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: RBF kernel fits training data well
    try:
        model = KernelRidgeRegressionClass(alpha=0.1, kernel="rbf", gamma=1.0).fit(X, y)
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        passed = mse < 0.1
        results.append((T[2], passed, f"RBF MSE ({mse:.4f}) should be < 0.1"))
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: linear kernel fits linear data well
    try:
        linear_model = KernelRidgeRegressionClass(alpha=1.0, kernel="linear").fit(X, y)
        y_pred_linear = linear_model.predict(X)
        linear_mse = np.mean((y - y_pred_linear) ** 2)
        passed = linear_mse < 0.1
        results.append((T[3], passed, f"Linear kernel MSE ({linear_mse:.4f}) should be < 0.1"))
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: RBF fits nonlinear data better than linear
    try:
        linear = KernelRidgeRegressionClass(alpha=0.1, kernel="linear").fit(X_nl, y_nl)
        rbf = KernelRidgeRegressionClass(alpha=0.1, kernel="rbf", gamma=1.0).fit(X_nl, y_nl)
        linear_mse = np.mean((y_nl - linear.predict(X_nl)) ** 2)
        rbf_mse = np.mean((y_nl - rbf.predict(X_nl)) ** 2)
        passed = rbf_mse < linear_mse
        results.append((T[4], passed, f"RBF MSE ({rbf_mse:.4f}) should be < linear MSE ({linear_mse:.4f})"))
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Test 6: predict on new data (generalization)
    try:
        model = KernelRidgeRegressionClass(alpha=0.1, kernel="rbf", gamma=1.0).fit(X_nl, y_nl)
        X_new = np.array([[1.0], [2.0], [3.0]])
        y_pred_new = model.predict(X_new)
        # Check shape and that predictions are reasonable (close to sin values)
        y_true_approx = np.sin(X_new).ravel()
        passed = y_pred_new.shape == (3,) and np.mean((y_pred_new - y_true_approx) ** 2) < 0.5
        results.append((T[5], passed, "Should predict reasonably on new data"))
    except Exception as e:
        results.append((T[5], False, str(e)))

    # Visualization: non-linear fit (sinusoidal data) with both linear and RBF kernels
    try:
        model_rbf = KernelRidgeRegressionClass(alpha=0.1, kernel="rbf", gamma=1.0).fit(X_nl, y_nl)
        model_linear = KernelRidgeRegressionClass(alpha=0.1, kernel="linear").fit(X_nl, y_nl)
        y_pred_rbf = model_rbf.predict(X_nl)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: predictions vs actual scatter (RBF)
        _plot_predictions_vs_actual(y_nl, y_pred_rbf, "KernelRidgeRegression (RBF): Predictions vs Actual", ax=axes[0])

        # Right: fit visualization comparing linear vs RBF
        X_plot = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
        y_plot_rbf = model_rbf.predict(X_plot)
        y_plot_linear = model_linear.predict(X_plot)
        y_plot_true = np.sin(X_plot).ravel()

        axes[1].scatter(X_nl, y_nl, alpha=0.5, label="Training data", s=20)
        axes[1].plot(X_plot, y_plot_linear, 'b-', linewidth=2, label="Linear kernel")
        axes[1].plot(X_plot, y_plot_rbf, 'r-', linewidth=2, label="RBF kernel")
        axes[1].plot(X_plot, y_plot_true, 'g--', linewidth=2, label="True function (sin)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("KernelRidgeRegression: Linear vs RBF Kernel")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Compute and display MSE for both kernels
        linear_mse = np.mean((y_nl - model_linear.predict(X_nl)) ** 2)
        rbf_mse = np.mean((y_nl - y_pred_rbf) ** 2)
        axes[1].text(0.05, 0.95, f'Linear MSE = {linear_mse:.4f}\nRBF MSE = {rbf_mse:.4f}',
                    transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


def check_logistic_regression(LogisticRegressionClass) -> list:
    """Test LogisticRegression implementation. Returns list of (test_name, passed, message)."""
    T = LOGISTIC_REGRESSION_TESTS
    results = []
    X, y = load_classification_data()

    # Test 1: fit returns self
    try:
        model = LogisticRegressionClass(lr=0.1)
        result = model.fit(X, y)
        passed = result is model
        results.append((T[0], passed, "fit() should return self"))
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: weights shape
    try:
        model = LogisticRegressionClass(lr=0.1).fit(X, y)
        passed = model.weights.shape == (X.shape[1],)
        results.append((T[1], passed, f"Expected shape {(X.shape[1],)}, got {model.weights.shape}"))
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: predict_proba shape and range
    try:
        model = LogisticRegressionClass(lr=0.1).fit(X, y)
        proba = model.predict_proba(X)
        shape_ok = proba.shape == (X.shape[0],)
        range_ok = np.all((proba >= 0) & (proba <= 1))
        passed = shape_ok and range_ok
        results.append((T[2], passed, "predict_proba should return values in [0, 1]"))
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: predict returns 0/1 labels
    try:
        model = LogisticRegressionClass(lr=0.1).fit(X, y)
        labels = model.predict(X)
        shape_ok = labels.shape == (X.shape[0],)
        values_ok = set(np.unique(labels)).issubset({0, 1})
        passed = shape_ok and values_ok
        results.append((T[3], passed, "predict should return 0/1 integer labels"))
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: accuracy on linearly separable data
    try:
        model = LogisticRegressionClass(lr=0.1, max_iter=2000).fit(X, y)
        accuracy = np.mean(model.predict(X) == y)
        passed = accuracy > 0.85
        results.append((T[4], passed, f"Training accuracy ({accuracy:.2%}) should be > 85%"))
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Test 6: sigmoid correctness
    try:
        model = LogisticRegressionClass()
        sig_0 = model.sigmoid(np.array([0.0]))[0]
        sig_pos = model.sigmoid(np.array([100.0]))[0]
        sig_neg = model.sigmoid(np.array([-100.0]))[0]
        passed = (abs(sig_0 - 0.5) < 1e-10 and sig_pos > 0.99 and sig_neg < 0.01)
        results.append((T[5], passed, "sigmoid(0)=0.5, sigmoid(large)~1, sigmoid(-large)~0"))
    except Exception as e:
        results.append((T[5], False, str(e)))

    # Visualization: decision boundary
    try:
        from regression_helpers import plot_decision_boundary
        model = LogisticRegressionClass(lr=0.1, max_iter=1000).fit(X, y)
        plot_decision_boundary(X, y, model)
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


# ============================================================================
# Test Registry (canonical list of all tests — used by grading script)
# ============================================================================

TEST_REGISTRY = {
    "LinearRegression": {
        "check_fn": check_linear_regression,
        "tests": list(LINEAR_REGRESSION_TESTS),
    },
    "RidgeRegression": {
        "check_fn": check_ridge_regression,
        "tests": list(RIDGE_REGRESSION_TESTS),
    },
    "LassoRegression": {
        "check_fn": check_lasso_regression,
        "tests": list(LASSO_REGRESSION_TESTS),
    },
    "KernelRidgeRegression": {
        "check_fn": check_kernel_ridge_regression,
        "tests": list(KERNEL_RIDGE_REGRESSION_TESTS),
    },
    "LogisticRegression": {
        "check_fn": check_logistic_regression,
        "tests": list(LOGISTIC_REGRESSION_TESTS),
    },
}

TOTAL_EXPECTED_TESTS = sum(len(v["tests"]) for v in TEST_REGISTRY.values())


# ============================================================================
# Test Runners (used by student notebook)
# ============================================================================


def run_single_test(
    cls,
    test_func,
) -> dict:
    """
    Run tests for a single class with formatted output.

    Parameters
    ----------
    cls : class
        The class implementation to test
    test_func : callable
        The test function (e.g., check_linear_regression)

    Returns
    -------
    dict : Results with 'passed', 'failed', and 'results' keys
    """
    class_name = cls.__name__
    print(f"Testing {class_name}...")
    print("=" * 50)

    try:
        results = test_func(cls)
        passed = sum(1 for _, p, _ in results if p)
        total = len(results)

        for test_name, test_passed, message in results:
            symbol = "✓" if test_passed else "✗"
            print(f"  {symbol} {test_name}")
            if not test_passed:
                print(f"      {message}")

        print(f"\n{class_name}: {passed}/{total} tests passed")
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("⚠️ Some tests failed. Review your implementation above.")

        return {"passed": passed, "failed": total - passed, "results": results}

    except NotImplementedError:
        print(f"⚠️ {class_name} not yet implemented. Complete the TODOs above first.")
        return {"passed": 0, "failed": 0, "results": [], "not_implemented": True}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"passed": 0, "failed": 1, "results": [], "error": str(e)}


def run_tests(
    LinearRegressionClass=None,
    RidgeRegressionClass=None,
    LassoRegressionClass=None,
    KernelRidgeRegressionClass=None,
    LogisticRegressionClass=None,
    verbose: bool = True,
) -> dict:
    """
    Run all tests on the provided implementations.

    Parameters
    ----------
    LinearRegressionClass : class, optional
        Your LinearRegression implementation
    RidgeRegressionClass : class, optional
        Your RidgeRegression implementation
    LassoRegressionClass : class, optional
        Your LassoRegression implementation
    KernelRidgeRegressionClass : class, optional
        Your KernelRidgeRegression implementation
    verbose : bool, default=True
        Whether to print results

    Returns
    -------
    dict : Summary with total passed/failed counts
    """
    all_results = {}
    total_passed = 0
    total_failed = 0

    # Map keyword args to registry class names
    provided_classes = {
        "LinearRegression": LinearRegressionClass,
        "RidgeRegression": RidgeRegressionClass,
        "LassoRegression": LassoRegressionClass,
        "KernelRidgeRegression": KernelRidgeRegressionClass,
        "LogisticRegression": LogisticRegressionClass,
    }

    for class_name, entry in TEST_REGISTRY.items():
        cls = provided_classes.get(class_name)
        test_func = entry["check_fn"]
        if cls is None:
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"{class_name}: SKIPPED (not provided)")
            continue

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Testing {class_name}")
            print("=" * 60)

        try:
            results = test_func(cls)
            all_results[class_name] = results

            passed = sum(1 for _, p, _ in results if p)
            failed = len(results) - passed
            total_passed += passed
            total_failed += failed

            if verbose:
                for test_name, test_passed, message in results:
                    status = "PASSED" if test_passed else "FAILED"
                    symbol = "✓" if test_passed else "✗"
                    print(f"  {symbol} {test_name}: {status}")
                    if not test_passed:
                        print(f"      {message}")
                print(f"\n  {class_name}: {passed}/{len(results)} tests passed")

        except Exception as e:
            if verbose:
                print(f"  ERROR running tests: {e}")
            total_failed += 1

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"TOTAL: {total_passed} passed, {total_failed} failed")
        print("=" * 60)

    return {
        "passed": total_passed,
        "failed": total_failed,
        "details": all_results,
    }
