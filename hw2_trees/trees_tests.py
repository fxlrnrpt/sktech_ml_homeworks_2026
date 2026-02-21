"""
Test harness for HW2: Trees & Ensembles implementations.

Import into Jupyter notebook and run tests on student implementations.

Example usage in notebook:
    from trees_tests import run_single_test, check_decision_tree
    run_single_test(DecisionTreeClassifier, check_decision_tree)
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


def load_binary_data() -> tuple:
    """Load binary classification data."""
    data = _load_test_data("binary_classification")
    return data["X"], data["y"]


def load_multiclass_data() -> tuple:
    """Load multiclass classification data."""
    data = _load_test_data("multiclass_classification")
    return data["X"], data["y"]


def load_complex_data() -> tuple:
    """Load complex classification data (concentric circles)."""
    data = _load_test_data("complex_classification")
    return data["X"], data["y"]


# ============================================================================
# Test Name Constants (single source of truth for test names)
# ============================================================================

DECISION_TREE_TESTS = (
    "fit_returns_self",
    "predict_shape",
    "pure_leaf",
    "binary_accuracy",
    "multiclass",
    "max_depth_effect",
)

BAGGING_CLASSIFIER_TESTS = (
    "fit_returns_self",
    "predict_shape",
    "n_estimators_count",
    "accuracy",
    "beats_single_estimator",
)

RANDOM_FOREST_TESTS = (
    "fit_returns_self",
    "predict_shape",
    "accuracy",
    "feature_subsampling",
    "beats_single_tree",
)

GRADIENT_BOOSTING_TESTS = (
    "fit_returns_self",
    "predict_shape",
    "predict_proba_valid",
    "accuracy",
    "learning_rate_effect",
    "n_estimators_effect",
)


# ============================================================================
# Internal test helper: simple stump for Bagging tests
# ============================================================================


class _SimpleStump:
    """Minimal depth-1 decision tree for testing Bagging independently."""

    def __init__(self, max_depth=1, random_state=None):
        self.max_depth = max_depth
        self.random_state = random_state
        self._feature = None
        self._threshold = None
        self._left_class = None
        self._right_class = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        best_gini = float("inf")
        n_features = X.shape[1]

        for f in range(n_features):
            values = np.unique(X[:, f])
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            # Sample at most 20 thresholds for speed
            if len(thresholds) > 20:
                idx = rng.choice(len(thresholds), 20, replace=False)
                thresholds = thresholds[idx]

            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                def gini(mask):
                    classes, counts = np.unique(y[mask], return_counts=True)
                    p = counts / counts.sum()
                    return 1.0 - np.sum(p ** 2)

                n = len(y)
                g = (left_mask.sum() / n) * gini(left_mask) + (
                    right_mask.sum() / n
                ) * gini(right_mask)
                if g < best_gini:
                    best_gini = g
                    self._feature = f
                    self._threshold = t

        if self._feature is not None:
            left_mask = X[:, self._feature] <= self._threshold
            right_mask = ~left_mask
            classes_l, counts_l = np.unique(y[left_mask], return_counts=True)
            self._left_class = classes_l[np.argmax(counts_l)]
            classes_r, counts_r = np.unique(y[right_mask], return_counts=True)
            self._right_class = classes_r[np.argmax(counts_r)]
        else:
            classes, counts = np.unique(y, return_counts=True)
            self._left_class = classes[np.argmax(counts)]
            self._right_class = self._left_class

        return self

    def predict(self, X):
        if self._feature is None:
            return np.full(X.shape[0], self._left_class)
        result = np.where(
            X[:, self._feature] <= self._threshold,
            self._left_class,
            self._right_class,
        )
        return result


# ============================================================================
# Visualization helpers for tests
# ============================================================================


def _plot_decision_boundary(X, y, model, title, ax=None):
    """Plot 2D decision boundary on given axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=20)

    accuracy = np.mean(model.predict(X) == y)
    ax.set_title(f"{title}\n(accuracy: {accuracy:.1%})")
    ax.grid(True, alpha=0.3)

    return ax


# ============================================================================
# Test Functions
# ============================================================================


def check_decision_tree(DecisionTreeClass) -> list:
    """Test DecisionTreeClassifier implementation. Returns list of (test_name, passed, message)."""
    T = DECISION_TREE_TESTS
    results = []
    X, y = load_binary_data()

    # Test 1: fit returns self
    try:
        model = DecisionTreeClass(max_depth=5)
        result = model.fit(X, y)
        passed = result is model
        results.append((T[0], passed, "fit() should return self"))
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: predict shape
    try:
        model = DecisionTreeClass(max_depth=5).fit(X, y)
        y_pred = model.predict(X)
        passed = y_pred.shape == (X.shape[0],)
        results.append(
            (T[1], passed, f"Expected shape {(X.shape[0],)}, got {y_pred.shape}")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: pure leaf — all same class predicts that class
    try:
        X_pure = np.random.randn(20, 2)
        y_pure = np.ones(20)
        model = DecisionTreeClass(max_depth=3).fit(X_pure, y_pure)
        preds = model.predict(X_pure)
        passed = np.all(preds == 1)
        results.append((T[2], passed, "Should predict single class for pure data"))
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: binary accuracy > 85%
    try:
        model = DecisionTreeClass(max_depth=10).fit(X, y)
        accuracy = np.mean(model.predict(X) == y)
        passed = accuracy > 0.85
        results.append(
            (T[3], passed, f"Binary accuracy ({accuracy:.2%}) should be > 85%")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: multiclass support
    try:
        X_mc, y_mc = load_multiclass_data()
        model = DecisionTreeClass(max_depth=10).fit(X_mc, y_mc)
        accuracy = np.mean(model.predict(X_mc) == y_mc)
        passed = accuracy > 0.75
        results.append(
            (T[4], passed, f"Multiclass accuracy ({accuracy:.2%}) should be > 75%")
        )
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Test 6: max_depth effect — deeper tree ≥ shallow tree accuracy on complex data
    try:
        X_c, y_c = load_complex_data()
        shallow = DecisionTreeClass(max_depth=1).fit(X_c, y_c)
        deep = DecisionTreeClass(max_depth=10).fit(X_c, y_c)
        acc_shallow = np.mean(shallow.predict(X_c) == y_c)
        acc_deep = np.mean(deep.predict(X_c) == y_c)
        passed = acc_deep >= acc_shallow
        results.append(
            (
                T[5],
                passed,
                f"Deep ({acc_deep:.2%}) should be >= shallow ({acc_shallow:.2%})",
            )
        )
    except Exception as e:
        results.append((T[5], False, str(e)))

    # Visualization
    try:
        model = DecisionTreeClass(max_depth=5).fit(X, y)
        _plot_decision_boundary(X, y, model, "DecisionTreeClassifier")
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


def check_bagging_classifier(BaggingClass) -> list:
    """Test BaggingClassifier implementation. Returns list of (test_name, passed, message)."""
    T = BAGGING_CLASSIFIER_TESTS
    results = []
    X, y = load_binary_data()

    # Test 1: fit returns self
    try:
        model = BaggingClass(
            base_estimator_class=_SimpleStump, n_estimators=5, random_state=42
        )
        result = model.fit(X, y)
        passed = result is model
        results.append((T[0], passed, "fit() should return self"))
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: predict shape
    try:
        model = BaggingClass(
            base_estimator_class=_SimpleStump, n_estimators=5, random_state=42
        ).fit(X, y)
        y_pred = model.predict(X)
        passed = y_pred.shape == (X.shape[0],)
        results.append(
            (T[1], passed, f"Expected shape {(X.shape[0],)}, got {y_pred.shape}")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: n_estimators count
    try:
        n_est = 7
        model = BaggingClass(
            base_estimator_class=_SimpleStump, n_estimators=n_est, random_state=42
        ).fit(X, y)
        passed = len(model.estimators_) == n_est
        results.append(
            (
                T[2],
                passed,
                f"Expected {n_est} estimators, got {len(model.estimators_)}",
            )
        )
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: accuracy > 80%
    try:
        model = BaggingClass(
            base_estimator_class=_SimpleStump, n_estimators=20, random_state=42
        ).fit(X, y)
        accuracy = np.mean(model.predict(X) == y)
        passed = accuracy > 0.80
        results.append(
            (T[3], passed, f"Accuracy ({accuracy:.2%}) should be > 80%")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: ensemble beats single estimator on binary data
    try:
        single = _SimpleStump(random_state=42).fit(X, y)
        single_acc = np.mean(single.predict(X) == y)

        bag = BaggingClass(
            base_estimator_class=_SimpleStump, n_estimators=20, random_state=42
        ).fit(X, y)
        bag_acc = np.mean(bag.predict(X) == y)

        passed = bag_acc >= single_acc
        results.append(
            (
                T[4],
                passed,
                f"Ensemble ({bag_acc:.2%}) should be >= single ({single_acc:.2%})",
            )
        )
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Visualization
    try:
        model = BaggingClass(
            base_estimator_class=_SimpleStump, n_estimators=10, random_state=42
        ).fit(X, y)
        _plot_decision_boundary(X, y, model, "BaggingClassifier")
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


def check_random_forest(RandomForestClass) -> list:
    """Test RandomForestClassifier implementation. Returns list of (test_name, passed, message)."""
    T = RANDOM_FOREST_TESTS
    results = []
    X, y = load_binary_data()

    # Test 1: fit returns self
    try:
        model = RandomForestClass(n_estimators=5, max_depth=5, random_state=42)
        result = model.fit(X, y)
        passed = result is model
        results.append((T[0], passed, "fit() should return self"))
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: predict shape
    try:
        model = RandomForestClass(
            n_estimators=5, max_depth=5, random_state=42
        ).fit(X, y)
        y_pred = model.predict(X)
        passed = y_pred.shape == (X.shape[0],)
        results.append(
            (T[1], passed, f"Expected shape {(X.shape[0],)}, got {y_pred.shape}")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: accuracy > 80%
    try:
        model = RandomForestClass(
            n_estimators=20, max_depth=10, random_state=42
        ).fit(X, y)
        accuracy = np.mean(model.predict(X) == y)
        passed = accuracy > 0.80
        results.append(
            (T[2], passed, f"Accuracy ({accuracy:.2%}) should be > 80%")
        )
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: feature subsampling — trees use different feature subsets
    try:
        model = RandomForestClass(
            n_estimators=10, max_depth=5, max_features="sqrt", random_state=42
        ).fit(X, y)
        feature_sets = model.feature_indices_
        passed = len(feature_sets) == 10
        # Check that not all feature sets are identical
        if passed and len(feature_sets) > 1:
            all_same = all(
                np.array_equal(feature_sets[0], fs) for fs in feature_sets[1:]
            )
            passed = not all_same
        results.append(
            (T[3], passed, "Trees should use different feature subsets")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: beats single tree on complex data
    try:
        X_c, y_c = load_complex_data()
        # Use first 80% for train, last 20% for test
        split = int(0.8 * len(X_c))
        X_train, X_test = X_c[:split], X_c[split:]
        y_train, y_test = y_c[:split], y_c[split:]

        from trees_tests import _SimpleStump  # already defined above

        single = _SimpleStump(max_depth=1, random_state=42).fit(X_train, y_train)
        rf = RandomForestClass(
            n_estimators=20, max_depth=5, random_state=42
        ).fit(X_train, y_train)

        single_acc = np.mean(single.predict(X_test) == y_test)
        rf_acc = np.mean(rf.predict(X_test) == y_test)
        passed = rf_acc >= single_acc
        results.append(
            (
                T[4],
                passed,
                f"RF ({rf_acc:.2%}) should beat single stump ({single_acc:.2%})",
            )
        )
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Visualization
    try:
        model = RandomForestClass(
            n_estimators=20, max_depth=10, random_state=42
        ).fit(X, y)
        _plot_decision_boundary(X, y, model, "RandomForestClassifier")
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


def check_gradient_boosting(GradientBoostingClass) -> list:
    """Test GradientBoostingClassifier implementation. Returns list of (test_name, passed, message)."""
    T = GRADIENT_BOOSTING_TESTS
    results = []
    X, y = load_binary_data()

    # Test 1: fit returns self
    try:
        model = GradientBoostingClass(
            n_estimators=50, learning_rate=0.1, random_state=42
        )
        result = model.fit(X, y)
        passed = result is model
        results.append((T[0], passed, "fit() should return self"))
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: predict shape
    try:
        model = GradientBoostingClass(
            n_estimators=50, learning_rate=0.1, random_state=42
        ).fit(X, y)
        y_pred = model.predict(X)
        passed = y_pred.shape == (X.shape[0],)
        results.append(
            (T[1], passed, f"Expected shape {(X.shape[0],)}, got {y_pred.shape}")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: predict_proba returns valid probabilities
    try:
        model = GradientBoostingClass(
            n_estimators=50, learning_rate=0.1, random_state=42
        ).fit(X, y)
        proba = model.predict_proba(X)
        shape_ok = proba.shape == (X.shape[0],)
        range_ok = np.all((proba >= 0) & (proba <= 1))
        passed = shape_ok and range_ok
        results.append((T[2], passed, "predict_proba should return values in [0, 1]"))
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: accuracy > 85%
    try:
        model = GradientBoostingClass(
            n_estimators=100, learning_rate=0.1, random_state=42
        ).fit(X, y)
        accuracy = np.mean(model.predict(X) == y)
        passed = accuracy > 0.85
        results.append(
            (T[3], passed, f"Accuracy ({accuracy:.2%}) should be > 85%")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: learning rate effect — lower lr with same n_estimators → lower accuracy
    try:
        model_high_lr = GradientBoostingClass(
            n_estimators=50, learning_rate=0.5, random_state=42
        ).fit(X, y)
        model_low_lr = GradientBoostingClass(
            n_estimators=50, learning_rate=0.01, random_state=42
        ).fit(X, y)
        acc_high = np.mean(model_high_lr.predict(X) == y)
        acc_low = np.mean(model_low_lr.predict(X) == y)
        passed = acc_high > acc_low
        results.append(
            (
                T[4],
                passed,
                f"High lr acc ({acc_high:.2%}) should be > low lr acc ({acc_low:.2%})",
            )
        )
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Test 6: more estimators → better accuracy
    try:
        model_few = GradientBoostingClass(
            n_estimators=10, learning_rate=0.1, random_state=42
        ).fit(X, y)
        model_many = GradientBoostingClass(
            n_estimators=100, learning_rate=0.1, random_state=42
        ).fit(X, y)
        acc_few = np.mean(model_few.predict(X) == y)
        acc_many = np.mean(model_many.predict(X) == y)
        passed = acc_many >= acc_few
        results.append(
            (
                T[5],
                passed,
                f"Many estimators acc ({acc_many:.2%}) should be >= few ({acc_few:.2%})",
            )
        )
    except Exception as e:
        results.append((T[5], False, str(e)))

    # Visualization
    try:
        model = GradientBoostingClass(
            n_estimators=100, learning_rate=0.1, random_state=42
        ).fit(X, y)
        _plot_decision_boundary(X, y, model, "GradientBoostingClassifier")
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


# ============================================================================
# Test Registry (canonical list of all tests — used by grading script)
# ============================================================================

TEST_REGISTRY = {
    "DecisionTreeClassifier": {
        "check_fn": check_decision_tree,
        "tests": list(DECISION_TREE_TESTS),
    },
    "BaggingClassifier": {
        "check_fn": check_bagging_classifier,
        "tests": list(BAGGING_CLASSIFIER_TESTS),
    },
    "RandomForestClassifier": {
        "check_fn": check_random_forest,
        "tests": list(RANDOM_FOREST_TESTS),
    },
    "GradientBoostingClassifier": {
        "check_fn": check_gradient_boosting,
        "tests": list(GRADIENT_BOOSTING_TESTS),
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
        The test function (e.g., check_decision_tree)

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
    DecisionTreeClass=None,
    BaggingClass=None,
    RandomForestClass=None,
    GradientBoostingClass=None,
    verbose: bool = True,
) -> dict:
    """
    Run all tests on the provided implementations.

    Parameters
    ----------
    DecisionTreeClass : class, optional
        Your DecisionTreeClassifier implementation
    BaggingClass : class, optional
        Your BaggingClassifier implementation
    RandomForestClass : class, optional
        Your RandomForestClassifier implementation
    GradientBoostingClass : class, optional
        Your GradientBoostingClassifier implementation
    verbose : bool, default=True
        Whether to print results

    Returns
    -------
    dict : Summary with total passed/failed counts
    """
    all_results = {}
    total_passed = 0
    total_failed = 0

    provided_classes = {
        "DecisionTreeClassifier": DecisionTreeClass,
        "BaggingClassifier": BaggingClass,
        "RandomForestClassifier": RandomForestClass,
        "GradientBoostingClassifier": GradientBoostingClass,
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
