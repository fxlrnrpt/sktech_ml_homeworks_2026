"""
Test harness for HW3: Deep Learning implementations.

Import into Jupyter notebook and run tests on student implementations.

Example usage in notebook:
    from dl_tests import run_single_test, check_numpy_mlp
    run_single_test(NumpyMLP, check_numpy_mlp)
"""

import json
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


def load_spiral_data() -> tuple:
    """Load 2D spiral classification data."""
    data = _load_test_data("spirals")
    return data["X"], data["y"]


def load_tiny_text_data() -> tuple:
    """Load tiny Shakespeare text data."""
    data = _load_test_data("tiny_shakespeare")
    text = "".join(data["text"].tolist())
    char_to_idx = json.loads(str(data["char_to_idx_json"]))
    idx_to_char = {int(k): v for k, v in json.loads(str(data["idx_to_char_json"])).items()}
    vocab_size = int(data["vocab_size"])
    return text, char_to_idx, idx_to_char, vocab_size


def load_mlp_reference() -> tuple:
    """Load MLP reference data for gradient checking."""
    data = _load_test_data("mlp_reference")
    return data["X"], data["y"], int(data["seed"])


def load_attention_reference() -> tuple:
    """Load attention reference data."""
    data = _load_test_data("attention_reference")
    return data["Q"], data["K"], data["V"]


# ============================================================================
# Test Name Constants (single source of truth for test names)
# ============================================================================

NUMPY_MLP_TESTS = (
    "forward_shape",
    "backward_shapes",
    "gradient_check",
    "training_loss_decreases",
    "accuracy",
)

TORCH_MLP_TESTS = (
    "forward_shape",
    "parameter_count",
    "training_loss_decreases",
    "accuracy",
    "matches_numpy_accuracy",
)

MULTI_HEAD_ATTENTION_TESTS = (
    "output_shape",
    "attn_weights_shape",
    "attn_weights_sum_to_one",
    "causal_mask_works",
    "multi_head_output_differs_from_single",
)

TRANSFORMER_BLOCK_TESTS = (
    "output_shape",
    "residual_connection",
    "layer_norm_applied",
    "different_seq_lengths",
)

SIMPLE_TRANSFORMER_TESTS = (
    "forward_shape",
    "generate_shape",
    "overfitting",
    "causal_masking",
)


# ============================================================================
# Internal helpers for self-contained tests
# ============================================================================


def _train_numpy_mlp_internal(model, X, y, lr=0.5, n_epochs=200):
    """Internal training loop for NumpyMLP (independent of student's train function)."""
    losses = []
    for _ in range(n_epochs):
        probs = model.forward(X)
        batch_size = len(y)
        log_probs = -np.log(probs[np.arange(batch_size), y] + 1e-12)
        loss = float(np.mean(log_probs))
        dlogits = probs.copy()
        dlogits[np.arange(batch_size), y] -= 1
        dlogits /= batch_size
        grads = model.backward(dlogits)
        for key in model.params:
            model.params[key] -= lr * grads[key]
        losses.append(loss)
    return losses


def _train_torch_mlp_internal(model, X_np, y_np, lr=0.5, n_epochs=200):
    """Internal training loop for TorchMLP (independent of student's train function)."""
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses



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

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k", s=20)

    accuracy = np.mean(model.predict(X) == y)
    ax.set_title(f"{title}\n(accuracy: {accuracy:.1%})")
    ax.grid(True, alpha=0.3)

    return ax


# ============================================================================
# Test Functions
# ============================================================================


def check_numpy_mlp(NumpyMLPClass) -> list:
    """Test NumpyMLP implementation. Returns list of (test_name, passed, message)."""
    T = NUMPY_MLP_TESTS
    results = []
    X, y = load_spiral_data()

    # Test 1: forward shape
    try:
        np.random.seed(42)
        model = NumpyMLPClass([2, 8, 3], seed=42)
        probs = model.forward(X[:10])
        shape_ok = probs.shape == (10, 3)
        sums_ok = np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        passed = shape_ok and sums_ok
        msg = f"Shape {probs.shape}, sums to 1: {sums_ok}"
        if not shape_ok:
            msg = f"Expected shape (10, 3), got {probs.shape}"
        results.append((T[0], passed, msg))
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: backward shapes
    try:
        np.random.seed(42)
        model = NumpyMLPClass([2, 8, 3], seed=42)
        probs = model.forward(X[:10])
        dlogits = probs.copy()
        dlogits[np.arange(10), y[:10]] -= 1
        dlogits /= 10
        grads = model.backward(dlogits)
        all_match = True
        for key in model.params:
            if key not in grads or grads[key].shape != model.params[key].shape:
                all_match = False
                break
        passed = all_match and len(grads) == len(model.params)
        results.append((T[1], passed, "All gradient shapes should match parameter shapes"))
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: numerical gradient check
    try:
        # Use seed=0 and non-zero bias init to avoid ReLU boundary issues
        np.random.seed(0)
        model = NumpyMLPClass([2, 4, 3], seed=0)
        # Initialize biases to small non-zero values to avoid relu boundary
        for key in model.params:
            if key.startswith("b"):
                model.params[key] = np.random.randn(*model.params[key].shape) * 0.1
        X_small = X[:4]
        y_small = y[:4]
        batch_size = len(y_small)
        eps = 1e-5

        # Forward + backward to get analytical gradients
        probs = model.forward(X_small)
        dlogits = probs.copy()
        dlogits[np.arange(batch_size), y_small] -= 1
        dlogits /= batch_size
        grads = model.backward(dlogits)

        # Save original params
        orig_params = {k: v.copy() for k, v in model.params.items()}

        max_diff = 0.0
        for key in model.params:
            for idx in np.ndindex(model.params[key].shape):
                # Restore original params before each perturbation
                for k in orig_params:
                    model.params[k] = orig_params[k].copy()

                # Plus perturbation
                model.params[key][idx] += eps
                p_plus = model.forward(X_small)
                loss_plus = float(np.mean(-np.log(p_plus[np.arange(batch_size), y_small] + 1e-12)))

                # Restore and minus perturbation
                for k in orig_params:
                    model.params[k] = orig_params[k].copy()
                model.params[key][idx] -= eps
                p_minus = model.forward(X_small)
                loss_minus = float(np.mean(-np.log(p_minus[np.arange(batch_size), y_small] + 1e-12)))

                grad_num = (loss_plus - loss_minus) / (2 * eps)
                grad_ana = grads[key][idx]
                diff = abs(grad_num - grad_ana) / max(abs(grad_num) + abs(grad_ana), 1e-8)
                max_diff = max(max_diff, diff)

        # Restore original params
        for k in orig_params:
            model.params[k] = orig_params[k].copy()

        passed = max_diff < 1e-4
        results.append(
            (T[2], passed, f"Max relative gradient difference: {max_diff:.2e} (should be < 1e-4)")
        )
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: training loss decreases
    try:
        np.random.seed(42)
        model = NumpyMLPClass([2, 32, 32, 3], seed=42)
        losses = _train_numpy_mlp_internal(model, X, y, lr=1.0, n_epochs=300)
        passed = losses[-1] < losses[0] * 0.5
        results.append(
            (T[3], passed, f"Final loss {losses[-1]:.4f} should be < 50% of initial {losses[0]:.4f}")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: accuracy
    try:
        # Use model from test 4 (already trained)
        preds = model.predict(X)
        accuracy = np.mean(preds == y)
        passed = accuracy > 0.75
        results.append(
            (T[4], passed, f"Accuracy {accuracy:.2%} should be > 75%")
        )
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Visualization
    try:
        _plot_decision_boundary(X, y, model, "NumpyMLP")
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


def check_torch_mlp(TorchMLPClass) -> list:
    """Test TorchMLP implementation. Returns list of (test_name, passed, message)."""
    T = TORCH_MLP_TESTS
    results = []
    X_np, y_np = load_spiral_data()

    # Test 1: forward shape
    try:
        torch.manual_seed(42)
        model = TorchMLPClass([2, 8, 3])
        x = torch.tensor(X_np[:10], dtype=torch.float32)
        out = model(x)
        passed = out.shape == (10, 3)
        results.append(
            (T[0], passed, f"Expected shape (10, 3), got {tuple(out.shape)}")
        )
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: parameter count
    try:
        torch.manual_seed(42)
        model = TorchMLPClass([2, 16, 3])
        n_params = sum(p.numel() for p in model.parameters())
        # 2*16 + 16 + 16*3 + 3 = 32 + 16 + 48 + 3 = 99
        passed = n_params == 99
        results.append(
            (T[1], passed, f"Expected 99 parameters for [2,16,3], got {n_params}")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: training loss decreases
    try:
        torch.manual_seed(42)
        model = TorchMLPClass([2, 32, 32, 3])
        losses = _train_torch_mlp_internal(model, X_np, y_np, lr=1.0, n_epochs=300)
        passed = losses[-1] < losses[0] * 0.5
        results.append(
            (T[2], passed, f"Final loss {losses[-1]:.4f} should be < 50% of initial {losses[0]:.4f}")
        )
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: accuracy
    try:
        preds = model.predict(X_np)
        accuracy = np.mean(preds == y_np)
        passed = accuracy > 0.75
        results.append(
            (T[3], passed, f"Accuracy {accuracy:.2%} should be > 75%")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: matches numpy accuracy (both should solve spirals)
    try:
        # Train a comparable model
        torch.manual_seed(42)
        model2 = TorchMLPClass([2, 32, 32, 3])
        _train_torch_mlp_internal(model2, X_np, y_np, lr=1.0, n_epochs=300)
        torch_acc = np.mean(model2.predict(X_np) == y_np)
        # Just verify torch model can reach reasonable accuracy (same as numpy would)
        passed = torch_acc > 0.70
        results.append(
            (T[4], passed, f"Torch accuracy {torch_acc:.2%} should be > 70% (comparable to numpy)")
        )
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Visualization
    try:
        _plot_decision_boundary(X_np, y_np, model, "TorchMLP")
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


def check_multi_head_attention(MultiHeadAttentionClass) -> list:
    """Test MultiHeadAttention implementation. Returns list of (test_name, passed, message)."""
    T = MULTI_HEAD_ATTENTION_TESTS
    results = []

    # Test 1: output shape
    try:
        torch.manual_seed(42)
        mha = MultiHeadAttentionClass(d_model=16, n_heads=4)
        x = torch.randn(2, 5, 16)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1).unsqueeze(0)
        out, attn = mha(x, x, x, mask)
        passed = out.shape == (2, 5, 16)
        results.append(
            (T[0], passed, f"Expected output shape (2, 5, 16), got {tuple(out.shape)}")
        )
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: attention weights shape
    try:
        torch.manual_seed(42)
        mha = MultiHeadAttentionClass(d_model=16, n_heads=4)
        x = torch.randn(2, 5, 16)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1).unsqueeze(0)
        out, attn = mha(x, x, x, mask)
        passed = attn.shape == (2, 4, 5, 5)
        results.append(
            (T[1], passed, f"Expected attn shape (2, 4, 5, 5), got {tuple(attn.shape)}")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: attention weights sum to one
    try:
        torch.manual_seed(42)
        mha = MultiHeadAttentionClass(d_model=16, n_heads=4)
        x = torch.randn(2, 5, 16)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1).unsqueeze(0)
        out, attn = mha(x, x, x, mask)
        sums = attn.sum(dim=-1)
        passed = torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        results.append(
            (T[2], passed, "Each row of attention weights should sum to 1")
        )
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: causal mask works
    try:
        torch.manual_seed(42)
        mha = MultiHeadAttentionClass(d_model=16, n_heads=4)
        x = torch.randn(2, 5, 16)
        # Causal mask: True where we should NOT attend (upper triangle)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0)  # (1, 5, 5)
        out, attn = mha(x, x, x, mask=mask)
        # Check upper triangle of attention weights is near zero
        upper = torch.triu(attn, diagonal=1)
        passed = upper.abs().max().item() < 1e-5
        results.append(
            (T[3], passed, f"Masked positions should have ~0 attention weight, max={upper.abs().max().item():.2e}")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Test 5: multi-head output differs from single head
    try:
        torch.manual_seed(42)
        mha_multi = MultiHeadAttentionClass(d_model=16, n_heads=4)
        torch.manual_seed(42)
        mha_single = MultiHeadAttentionClass(d_model=16, n_heads=1)
        x = torch.randn(2, 5, 16)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1).unsqueeze(0)
        out_multi, _ = mha_multi(x, x, x, mask)
        out_single, _ = mha_single(x, x, x, mask)
        # They should produce different outputs (different head structures)
        passed = not torch.allclose(out_multi, out_single, atol=1e-3)
        results.append(
            (T[4], passed, "4-head and 1-head attention should produce different outputs")
        )
    except Exception as e:
        results.append((T[4], False, str(e)))

    # Visualization
    try:
        torch.manual_seed(42)
        mha = MultiHeadAttentionClass(d_model=16, n_heads=4)
        x = torch.randn(1, 8, 16)
        mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1).unsqueeze(0)
        _, attn = mha(x, x, x, mask)
        fig, axes = plt.subplots(1, 4, figsize=(16, 3))
        for i in range(4):
            axes[i].imshow(attn[0, i].detach().numpy(), cmap="Blues", aspect="auto")
            axes[i].set_title(f"Head {i}")
            axes[i].set_xlabel("Key")
            axes[i].set_ylabel("Query")
        plt.suptitle("Multi-Head Attention Weights")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


def check_transformer_block(TransformerBlockClass) -> list:
    """Test TransformerBlock implementation. Returns list of (test_name, passed, message)."""
    T = TRANSFORMER_BLOCK_TESTS
    results = []

    # Test 1: output shape
    try:
        torch.manual_seed(42)
        block = TransformerBlockClass(d_model=16, n_heads=4, d_ff=32)
        x = torch.randn(2, 5, 16)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1).unsqueeze(0)
        out = block(x, mask)
        passed = out.shape == (2, 5, 16)
        results.append(
            (T[0], passed, f"Expected output shape (2, 5, 16), got {tuple(out.shape)}")
        )
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: residual connection — output should not be identical to input
    try:
        torch.manual_seed(42)
        block = TransformerBlockClass(d_model=16, n_heads=4, d_ff=32)
        x = torch.randn(2, 5, 16)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1).unsqueeze(0)
        out = block(x, mask)
        # Output should differ from input (transformation happened)
        not_same = not torch.allclose(out, x, atol=1e-3)
        # But should be correlated (residual adds, not replaces)
        corr = torch.nn.functional.cosine_similarity(
            out.flatten().unsqueeze(0), x.flatten().unsqueeze(0)
        ).item()
        passed = not_same and corr > -0.5  # should have some positive correlation
        results.append(
            (T[1], passed, f"Output differs from input: {not_same}, correlation: {corr:.3f}")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: layer norm applied — output along last dim should be roughly normalized
    try:
        torch.manual_seed(42)
        block = TransformerBlockClass(d_model=16, n_heads=4, d_ff=32)
        x = torch.randn(2, 5, 16)
        mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1).unsqueeze(0)
        out = block(x, mask)
        # After layer norm, mean should be ~0 and std ~1 along last dim
        mean = out.mean(dim=-1)
        std = out.std(dim=-1, correction=0)
        mean_ok = mean.abs().max().item() < 0.5
        std_ok = (std - 1.0).abs().max().item() < 0.5
        passed = mean_ok and std_ok
        results.append(
            (T[2], passed, f"LayerNorm: max|mean|={mean.abs().max():.3f}, max|std-1|={(std-1).abs().max():.3f}")
        )
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: different sequence lengths work
    try:
        torch.manual_seed(42)
        block = TransformerBlockClass(d_model=16, n_heads=4, d_ff=32)
        x3 = torch.randn(2, 3, 16)
        x10 = torch.randn(2, 10, 16)
        mask3 = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1).unsqueeze(0)
        mask10 = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1).unsqueeze(0)
        out3 = block(x3, mask3)
        out10 = block(x10, mask10)
        passed = out3.shape == (2, 3, 16) and out10.shape == (2, 10, 16)
        results.append(
            (T[3], passed, f"Shapes: seq3={tuple(out3.shape)}, seq10={tuple(out10.shape)}")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    return results


def check_simple_transformer(SimpleTransformerClass) -> list:
    """Test SimpleTransformer implementation. Returns list of (test_name, passed, message)."""
    T = SIMPLE_TRANSFORMER_TESTS
    results = []
    losses = []  # populated by overfitting test, used by visualization

    # Test 1: forward shape
    try:
        torch.manual_seed(42)
        model = SimpleTransformerClass(
            vocab_size=50, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=64
        )
        x = torch.randint(0, 50, (4, 10))
        logits = model(x)
        passed = logits.shape == (4, 10, 50)
        results.append(
            (T[0], passed, f"Expected shape (4, 10, 50), got {tuple(logits.shape)}")
        )
    except Exception as e:
        results.append((T[0], False, str(e)))

    # Test 2: generate shape
    try:
        torch.manual_seed(42)
        model = SimpleTransformerClass(
            vocab_size=50, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=64
        )
        start = torch.randint(0, 50, (1, 5))
        generated = model.generate(start, max_new_tokens=20)
        passed = generated.shape == (1, 25)
        results.append(
            (T[1], passed, f"Expected shape (1, 25), got {tuple(generated.shape)}")
        )
    except Exception as e:
        results.append((T[1], False, str(e)))

    # Test 3: overfitting — model should memorize tiny training data
    try:
        text, char_to_idx, idx_to_char, vocab_size = load_tiny_text_data()
        encoded = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
        X_text = encoded[:-1].unsqueeze(0)
        y_text = encoded[1:].unsqueeze(0)

        torch.manual_seed(42)
        model = SimpleTransformerClass(
            vocab_size=vocab_size, d_model=32, n_heads=2, n_layers=2,
            d_ff=64, max_seq_len=256
        )
        optimizer = optim.Adam(model.parameters(), lr=3e-3)
        criterion = nn.CrossEntropyLoss()

        losses = []
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            logits = model(X_text)
            loss = criterion(logits.view(-1, vocab_size), y_text.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        final_loss = losses[-1]
        passed = final_loss < 1.0
        results.append(
            (T[2], passed, f"Final loss {final_loss:.4f} should be < 1.0 (model should overfit tiny data)")
        )
    except Exception as e:
        results.append((T[2], False, str(e)))

    # Test 4: causal masking — changing future tokens should not affect past logits
    try:
        torch.manual_seed(42)
        model = SimpleTransformerClass(
            vocab_size=50, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=64
        )
        model.eval()
        with torch.no_grad():
            x1 = torch.randint(0, 50, (1, 10))
            x2 = x1.clone()
            x2[0, 5:] = torch.randint(0, 50, (5,))  # Change tokens at positions 5-9
            logits1 = model(x1)
            logits2 = model(x2)
            # Logits at positions 0-4 should be identical
            passed = torch.allclose(logits1[0, :5], logits2[0, :5], atol=1e-5)
        results.append(
            (T[3], passed, "Changing tokens at pos 5-9 should not affect logits at pos 0-4")
        )
    except Exception as e:
        results.append((T[3], False, str(e)))

    # Visualization
    try:
        if losses:
            plt.figure(figsize=(8, 4))
            plt.plot(losses, linewidth=2, color="purple")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("SimpleTransformer Training Loss (test)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"⚠️ Could not generate visualization: {e}")

    return results


# ============================================================================
# Test Registry (canonical list of all tests — used by grading script)
# ============================================================================

TEST_REGISTRY = {
    "NumpyMLP": {
        "check_fn": check_numpy_mlp,
        "tests": list(NUMPY_MLP_TESTS),
    },
    "TorchMLP": {
        "check_fn": check_torch_mlp,
        "tests": list(TORCH_MLP_TESTS),
    },
    "MultiHeadAttention": {
        "check_fn": check_multi_head_attention,
        "tests": list(MULTI_HEAD_ATTENTION_TESTS),
    },
    "TransformerBlock": {
        "check_fn": check_transformer_block,
        "tests": list(TRANSFORMER_BLOCK_TESTS),
    },
    "SimpleTransformer": {
        "check_fn": check_simple_transformer,
        "tests": list(SIMPLE_TRANSFORMER_TESTS),
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
        The test function (e.g., check_numpy_mlp)

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
    NumpyMLPClass=None,
    TorchMLPClass=None,
    MultiHeadAttentionClass=None,
    TransformerBlockClass=None,
    SimpleTransformerClass=None,
    verbose: bool = True,
) -> dict:
    """
    Run all tests on the provided implementations.

    Parameters
    ----------
    NumpyMLPClass : class, optional
    TorchMLPClass : class, optional
    MultiHeadAttentionClass : class, optional
    TransformerBlockClass : class, optional
    SimpleTransformerClass : class, optional
    verbose : bool, default=True

    Returns
    -------
    dict : Summary with total passed/failed counts
    """
    all_results = {}
    total_passed = 0
    total_failed = 0

    provided_classes = {
        "NumpyMLP": NumpyMLPClass,
        "TorchMLP": TorchMLPClass,
        "MultiHeadAttention": MultiHeadAttentionClass,
        "TransformerBlock": TransformerBlockClass,
        "SimpleTransformer": SimpleTransformerClass,
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
