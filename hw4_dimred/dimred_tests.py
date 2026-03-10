"""
Test harness for HW4: Dimensionality Reduction implementations.

Import into Jupyter notebook and run tests on student implementations.

Example usage in notebook:
    from dimred_tests import run_single_test, check_pca_eigen
    run_single_test(PCAEigen, check_pca_eigen)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# Data Loading (from pre-generated test data)
# ============================================================================

_TEST_DATA_DIR = Path(__file__).parent / "test_data"


def _load_test_data(name: str) -> dict:
    """Load pre-generated test data from .npz file."""
    return dict(np.load(_TEST_DATA_DIR / f"{name}.npz", allow_pickle=True))


def load_high_dim_blobs() -> tuple:
    """Load high-dimensional blob data."""
    data = _load_test_data("high_dim_blobs")
    return data["X"], data["y"]


def load_swiss_roll() -> tuple:
    """Load Swiss roll manifold data."""
    data = _load_test_data("swiss_roll")
    return data["X"], data["color"]


def load_pca_reference() -> dict:
    """Load pre-computed PCA reference values."""
    return _load_test_data("pca_reference")


# ============================================================================
# Test Name Constants (single source of truth for test names)
# ============================================================================

PCA_EIGEN_TESTS = (
    "components_shape",
    "components_orthogonal",
    "explained_variance_sorted",
    "transform_shape",
    "reconstruction_error",
)

PCA_SVD_TESTS = (
    "components_shape",
    "components_match_eigen",
    "explained_variance_matches",
    "transform_shape",
    "reconstruction_error",
)

LINEAR_AUTOENCODER_TESTS = (
    "output_shape",
    "training_loss_decreases",
    "reconstruction_error",
    "bottleneck_dimension",
    "weight_subspace_matches_pca",
)

NONLINEAR_AUTOENCODER_TESTS = (
    "output_shape",
    "training_loss_decreases",
    "reconstruction_error_blobs",
    "beats_linear_on_swiss_roll",
    "latent_preserves_structure",
    "encoder_decoder_compose",
)


# ============================================================================
# Internal helpers for self-contained tests
# ============================================================================


def _train_autoencoder_internal(model, X_np, n_epochs=300, lr=1e-3):
    """Internal training loop for autoencoders (independent of student's train function)."""
    X = torch.tensor(X_np, dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        X_recon = model(X)
        loss = nn.MSELoss()(X_recon, X)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _pca_eigen_reference(X, n_components):
    """Internal PCA via eigendecomposition for cross-checking."""
    mean = X.mean(axis=0)
    X_c = X - mean
    cov = (X_c.T @ X_c) / (len(X) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    components = eigenvectors[:, :n_components].T
    evr = eigenvalues / eigenvalues.sum()
    return components, eigenvalues[:n_components], evr[:n_components], mean


# ============================================================================
# Test Functions
# ============================================================================


def check_pca_eigen(PCAEigenClass) -> list:
    """Test PCAEigen implementation. Returns list of (test_name, passed, message)."""
    T = PCA_EIGEN_TESTS
    results = []
    X, y = load_high_dim_blobs()
    n_components = 3

    # Test 1: components shape
    try:
        model = PCAEigenClass(n_components=n_components)
        model.fit(X)
        passed = model.components_.shape == (n_components, X.shape[1])
        results.append(
            (T[0], passed, f"Expected components_ shape ({n_components}, {X.shape[1]}), got {model.components_.shape}")
        )
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: components orthogonal
    try:
        model = PCAEigenClass(n_components=n_components)
        model.fit(X)
        V = model.components_
        VVT = V @ V.T
        identity = np.eye(n_components)
        passed = np.allclose(VVT, identity, atol=1e-6)
        max_off = np.max(np.abs(VVT - identity))
        results.append(
            (T[1], passed, f"Components should be orthonormal, max deviation from I: {max_off:.2e}")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: explained variance sorted descending
    try:
        model = PCAEigenClass(n_components=n_components)
        model.fit(X)
        evr = model.explained_variance_ratio_
        sorted_desc = all(evr[i] >= evr[i + 1] - 1e-10 for i in range(len(evr) - 1))
        sums_ok = evr.sum() <= 1.0 + 1e-6
        all_positive = all(v >= -1e-10 for v in evr)
        passed = sorted_desc and sums_ok and all_positive
        results.append(
            (T[2], passed, f"EVR: {evr}, sorted_desc={sorted_desc}, sum={evr.sum():.4f}")
        )
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: transform shape
    try:
        model = PCAEigenClass(n_components=n_components)
        model.fit(X)
        X_proj = model.transform(X)
        passed = X_proj.shape == (X.shape[0], n_components)
        results.append(
            (T[3], passed, f"Expected transform shape ({X.shape[0]}, {n_components}), got {X_proj.shape}")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: reconstruction error
    try:
        model = PCAEigenClass(n_components=n_components)
        model.fit(X)
        X_proj = model.transform(X)
        X_recon = model.inverse_transform(X_proj)
        mse = np.mean((X - X_recon) ** 2)
        passed = mse < 0.1
        results.append(
            (T[4], passed, f"Reconstruction MSE: {mse:.4f} (should be < 0.1 with {n_components} components)")
        )
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Visualization
    try:
        model = PCAEigenClass(n_components=2)
        model.fit(X)
        X_2d = model.transform(X)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", edgecolors="k", s=20)
        plt.colorbar(scatter)
        plt.title("PCAEigen: 2D Projection")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not generate visualization: {e}")

    return results


def check_pca_svd(PCASVDClass) -> list:
    """Test PCASVD implementation. Returns list of (test_name, passed, message)."""
    T = PCA_SVD_TESTS
    results = []
    X, y = load_high_dim_blobs()
    n_components = 3

    # Test 1: components shape
    try:
        model = PCASVDClass(n_components=n_components)
        model.fit(X)
        passed = model.components_.shape == (n_components, X.shape[1])
        results.append(
            (T[0], passed, f"Expected components_ shape ({n_components}, {X.shape[1]}), got {model.components_.shape}")
        )
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: components match eigendecomposition (up to sign flips)
    try:
        model = PCASVDClass(n_components=n_components)
        model.fit(X)
        ref_components, _, _, _ = _pca_eigen_reference(X, n_components)
        # Check alignment: |dot product| should be ~1 for each pair
        dots = np.array([abs(np.dot(model.components_[i], ref_components[i])) for i in range(n_components)])
        passed = np.all(dots > 0.99)
        results.append(
            (T[1], passed, f"Component alignment (|dot|): {dots} (should all be > 0.99)")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: explained variance matches eigendecomposition
    try:
        model = PCASVDClass(n_components=n_components)
        model.fit(X)
        _, _, ref_evr, _ = _pca_eigen_reference(X, n_components)
        diff = np.max(np.abs(model.explained_variance_ratio_ - ref_evr))
        passed = diff < 1e-6
        results.append(
            (T[2], passed, f"Max EVR difference from eigen: {diff:.2e} (should be < 1e-6)")
        )
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: transform shape
    try:
        model = PCASVDClass(n_components=n_components)
        model.fit(X)
        X_proj = model.transform(X)
        passed = X_proj.shape == (X.shape[0], n_components)
        results.append(
            (T[3], passed, f"Expected transform shape ({X.shape[0]}, {n_components}), got {X_proj.shape}")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: reconstruction error
    try:
        model = PCASVDClass(n_components=n_components)
        model.fit(X)
        X_proj = model.transform(X)
        X_recon = model.inverse_transform(X_proj)
        mse = np.mean((X - X_recon) ** 2)
        passed = mse < 0.1
        results.append(
            (T[4], passed, f"Reconstruction MSE: {mse:.4f} (should be < 0.1 with {n_components} components)")
        )
    except Exception as e:
        results.append((T[4], False, str(e)))

    return results


def check_linear_autoencoder(LinearAutoencoderClass) -> list:
    """Test LinearAutoencoder implementation. Returns list of (test_name, passed, message)."""
    T = LINEAR_AUTOENCODER_TESTS
    results = []
    X, y = load_high_dim_blobs()
    input_dim = X.shape[1]
    latent_dim = 3

    # Standardize data for AE training
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Test 1: output shape
    try:
        torch.manual_seed(42)
        model = LinearAutoencoderClass(input_dim, latent_dim)
        X_t = torch.tensor(X_norm[:10], dtype=torch.float32)
        out = model(X_t)
        passed = out.shape == (10, input_dim)
        results.append(
            (T[0], passed, f"Expected output shape (10, {input_dim}), got {tuple(out.shape)}")
        )
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: training loss decreases
    try:
        torch.manual_seed(42)
        model = LinearAutoencoderClass(input_dim, latent_dim)
        losses = _train_autoencoder_internal(model, X_norm, n_epochs=200, lr=1e-3)
        passed = losses[-1] < losses[0] * 0.5
        results.append(
            (T[1], passed, f"Final loss {losses[-1]:.4f} should be < 50% of initial {losses[0]:.4f}")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: reconstruction error
    try:
        torch.manual_seed(42)
        model = LinearAutoencoderClass(input_dim, latent_dim)
        _train_autoencoder_internal(model, X_norm, n_epochs=1000, lr=1e-2)
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_norm, dtype=torch.float32)
            X_recon = model(X_t).numpy()
        mse = np.mean((X_norm - X_recon) ** 2)
        passed = mse < 0.3
        results.append(
            (T[2], passed, f"Reconstruction MSE: {mse:.4f} (should be < 0.3)")
        )
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: bottleneck dimension
    try:
        torch.manual_seed(42)
        model = LinearAutoencoderClass(input_dim, latent_dim)
        X_t = torch.tensor(X_norm[:10], dtype=torch.float32)
        z = model.encode(X_t)
        passed = z.shape == (10, latent_dim)
        results.append(
            (T[3], passed, f"Expected encode shape (10, {latent_dim}), got {tuple(z.shape)}")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: weight subspace matches PCA
    try:
        torch.manual_seed(42)
        model = LinearAutoencoderClass(input_dim, latent_dim)
        _train_autoencoder_internal(model, X_norm, n_epochs=2000, lr=1e-2)
        model.eval()

        # PCA reference
        ref_components, _, _, _ = _pca_eigen_reference(X_norm, latent_dim)

        # Compare projections: AE latent space vs PCA
        # If AE learned the PCA subspace, projecting data through AE encoder
        # should be equivalent to PCA projection (up to a rotation)
        with torch.no_grad():
            X_t = torch.tensor(X_norm, dtype=torch.float32)
            z_ae = model.encode(X_t).numpy()  # (n, latent_dim)

        mean_pca = X_norm.mean(axis=0)
        X_c = X_norm - mean_pca
        z_pca = X_c @ ref_components.T  # (n, latent_dim)

        # The AE and PCA projections should span the same subspace
        # Check via reconstruction: project AE latent space onto PCA latent space
        # Use least-squares to find the best rotation from z_ae to z_pca
        # If subspaces match, R² should be high
        R, _, _, _ = np.linalg.lstsq(z_ae, z_pca, rcond=None)
        z_ae_aligned = z_ae @ R
        ss_res = np.sum((z_pca - z_ae_aligned) ** 2)
        ss_tot = np.sum((z_pca - z_pca.mean(axis=0)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        passed = r_squared > 0.95
        results.append(
            (T[4], passed, f"Subspace R² = {r_squared:.4f} (should be > 0.95)")
        )
    except Exception as e:
        results.append((T[4], False, str(e)))

    return results


def check_nonlinear_autoencoder(NonlinearAutoencoderClass) -> list:
    """Test NonlinearAutoencoder implementation. Returns list of (test_name, passed, message)."""
    T = NONLINEAR_AUTOENCODER_TESTS
    results = []
    X_blobs, y_blobs = load_high_dim_blobs()
    X_swiss, color_swiss = load_swiss_roll()

    # Standardize blob data
    X_blobs_mean = X_blobs.mean(axis=0)
    X_blobs_std = X_blobs.std(axis=0) + 1e-8
    X_blobs_norm = (X_blobs - X_blobs_mean) / X_blobs_std

    # Standardize Swiss roll data
    X_swiss_mean = X_swiss.mean(axis=0)
    X_swiss_std = X_swiss.std(axis=0) + 1e-8
    X_swiss_norm = (X_swiss - X_swiss_mean) / X_swiss_std

    # Test 1: output shape
    try:
        torch.manual_seed(42)
        model = NonlinearAutoencoderClass(input_dim=20, hidden_dim=64, latent_dim=3)
        X_t = torch.tensor(X_blobs_norm[:10], dtype=torch.float32)
        out = model(X_t)
        passed = out.shape == (10, 20)
        results.append(
            (T[0], passed, f"Expected output shape (10, 20), got {tuple(out.shape)}")
        )
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: training loss decreases
    try:
        torch.manual_seed(42)
        model = NonlinearAutoencoderClass(input_dim=20, hidden_dim=64, latent_dim=3)
        losses = _train_autoencoder_internal(model, X_blobs_norm, n_epochs=200, lr=1e-3)
        passed = losses[-1] < losses[0] * 0.5
        results.append(
            (T[1], passed, f"Final loss {losses[-1]:.4f} should be < 50% of initial {losses[0]:.4f}")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: reconstruction error on blobs
    try:
        torch.manual_seed(42)
        model = NonlinearAutoencoderClass(input_dim=20, hidden_dim=64, latent_dim=3)
        _train_autoencoder_internal(model, X_blobs_norm, n_epochs=500, lr=1e-3)
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_blobs_norm, dtype=torch.float32)
            X_recon = model(X_t).numpy()
        mse = np.mean((X_blobs_norm - X_recon) ** 2)
        passed = mse < 0.3
        results.append(
            (T[2], passed, f"Blob reconstruction MSE: {mse:.4f} (should be < 0.3)")
        )
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: beats linear AE on Swiss roll
    try:
        # Train non-linear AE on Swiss roll
        torch.manual_seed(42)
        nl_model = NonlinearAutoencoderClass(input_dim=3, hidden_dim=64, latent_dim=2)
        _train_autoencoder_internal(nl_model, X_swiss_norm, n_epochs=500, lr=1e-3)
        nl_model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_swiss_norm, dtype=torch.float32)
            nl_recon = nl_model(X_t).numpy()
        nl_mse = np.mean((X_swiss_norm - nl_recon) ** 2)

        # PCA (linear) baseline on Swiss roll
        mean = X_swiss_norm.mean(axis=0)
        X_c = X_swiss_norm - mean
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        X_proj = X_c @ Vt[:2].T
        X_pca_recon = X_proj @ Vt[:2] + mean
        pca_mse = np.mean((X_swiss_norm - X_pca_recon) ** 2)

        passed = nl_mse < pca_mse
        results.append(
            (T[3], passed, f"Non-linear MSE: {nl_mse:.4f} vs PCA MSE: {pca_mse:.4f} (non-linear should be lower)")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: latent space preserves structure (k-NN accuracy)
    try:
        torch.manual_seed(42)
        model = NonlinearAutoencoderClass(input_dim=20, hidden_dim=64, latent_dim=3)
        _train_autoencoder_internal(model, X_blobs_norm, n_epochs=500, lr=1e-3)
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_blobs_norm, dtype=torch.float32)
            z = model.encode(X_t).numpy()

        # Simple k-NN (k=5) accuracy in latent space
        from collections import Counter
        k = 5
        correct = 0
        for i in range(len(z)):
            dists = np.sum((z - z[i]) ** 2, axis=1)
            dists[i] = np.inf
            nn_idx = np.argsort(dists)[:k]
            nn_labels = y_blobs[nn_idx]
            most_common = Counter(nn_labels.tolist()).most_common(1)[0][0]
            if most_common == y_blobs[i]:
                correct += 1
        knn_acc = correct / len(z)
        passed = knn_acc > 0.80
        results.append(
            (T[4], passed, f"k-NN accuracy in latent space: {knn_acc:.2%} (should be > 80%)")
        )
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Test 6: encode + decode = forward
    try:
        torch.manual_seed(42)
        model = NonlinearAutoencoderClass(input_dim=20, hidden_dim=64, latent_dim=3)
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_blobs_norm[:10], dtype=torch.float32)
            out_forward = model(X_t)
            out_compose = model.decode(model.encode(X_t))
        passed = torch.allclose(out_forward, out_compose, atol=1e-6)
        results.append(
            (T[5], passed, "decode(encode(X)) should equal forward(X)")
        )
    except Exception as e:
        results.append((T[5], False, str(e)))

    # Visualization
    try:
        torch.manual_seed(42)
        model = NonlinearAutoencoderClass(input_dim=20, hidden_dim=64, latent_dim=2)
        _train_autoencoder_internal(model, X_blobs_norm, n_epochs=500, lr=1e-3)
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_blobs_norm, dtype=torch.float32)
            z = model.encode(X_t).numpy()
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(z[:, 0], z[:, 1], c=y_blobs, cmap="viridis", edgecolors="k", s=20)
        plt.colorbar(scatter)
        plt.title("NonlinearAutoencoder: Latent Space")
        plt.xlabel("Latent dim 1")
        plt.ylabel("Latent dim 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not generate visualization: {e}")

    return results


# ============================================================================
# Test Registry (canonical list of all tests -- used by grading script)
# ============================================================================

TEST_REGISTRY = {
    "PCAEigen": {
        "check_fn": check_pca_eigen,
        "tests": list(PCA_EIGEN_TESTS),
    },
    "PCASVD": {
        "check_fn": check_pca_svd,
        "tests": list(PCA_SVD_TESTS),
    },
    "LinearAutoencoder": {
        "check_fn": check_linear_autoencoder,
        "tests": list(LINEAR_AUTOENCODER_TESTS),
    },
    "NonlinearAutoencoder": {
        "check_fn": check_nonlinear_autoencoder,
        "tests": list(NONLINEAR_AUTOENCODER_TESTS),
    },
}

TOTAL_EXPECTED_TESTS = sum(len(v["tests"]) for v in TEST_REGISTRY.values())


# ============================================================================
# Test Runners (used by student notebook)
# ============================================================================


def run_single_test(cls, test_func) -> dict:
    """
    Run tests for a single class with formatted output.

    Parameters
    ----------
    cls : class
        The class implementation to test
    test_func : callable
        The test function (e.g., check_pca_eigen)

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
    PCAEigenClass=None,
    PCASVDClass=None,
    LinearAutoencoderClass=None,
    NonlinearAutoencoderClass=None,
    verbose: bool = True,
) -> dict:
    """
    Run all tests on the provided implementations.

    Parameters
    ----------
    PCAEigenClass : class, optional
    PCASVDClass : class, optional
    LinearAutoencoderClass : class, optional
    NonlinearAutoencoderClass : class, optional
    verbose : bool, default=True

    Returns
    -------
    dict : Summary with total passed/failed counts
    """
    all_results = {}
    total_passed = 0
    total_failed = 0

    provided_classes = {
        "PCAEigen": PCAEigenClass,
        "PCASVD": PCASVDClass,
        "LinearAutoencoder": LinearAutoencoderClass,
        "NonlinearAutoencoder": NonlinearAutoencoderClass,
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
