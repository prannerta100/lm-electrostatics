# LM Electrostatics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Measure divergence and Jacobian asymmetry of GPT-2 layer vector fields (X_l as function of X_0) and plot against perplexity.

**Architecture:** Load pre-trained GPT-2, extract per-layer hidden states via hooks, define vector field f_l: X_0 → X_l, estimate divergence (Hutchinson trace) and asymmetry (random projection Jv vs Jᵀv) stochastically with K=50 random vectors, plot both vs perplexity for 10 sentences.

**Tech Stack:** Python 3.12, PyTorch, HuggingFace Transformers, Matplotlib, Poetry

---

### Task 1: Install dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add dependencies via poetry**

```bash
cd /Users/pranavpg/lm-electrostatics
poetry add torch transformers matplotlib numpy
```

**Step 2: Verify install**

```bash
poetry run python -c "import torch; import transformers; import matplotlib; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "chore: add torch, transformers, matplotlib, numpy dependencies"
```

---

### Task 2: Write GPT-2 equations document

**Files:**
- Create: `gpt2_equations.md`

Write the full GPT-2 forward pass equations from input_ids to P(x_{t+1}|x_{1:t}). Must include:
- Notation table (S, H, N, L, V, d_k = H/N)
- Token + positional embedding: X_0 = W_e[input_ids] + W_p[positions], both (S, H)
- Per-layer equations with pre-norm LayerNorm:
  - LayerNorm parameters (γ, β per layer)
  - Multi-head attention: Q, K, V projections (H → H), split into N heads of dim d_k, scaled dot-product attention with causal mask, concat + output projection
  - Residual connection after attention
  - FFN: two linear layers (H → 4H → H) with GELU activation
  - Residual connection after FFN
- Final LayerNorm + unembedding W_u (H → V) + softmax
- All parameter dimensions explicitly stated
- Convention: X_l is (S, H) — the hidden state after layer l

Reference the actual GPT-2 paper and HuggingFace implementation to ensure correctness (pre-norm, not post-norm; GELU, not ReLU; no bias in attention for some variants but gpt2 has bias).

**Step 2: Commit**

```bash
git add gpt2_equations.md
git commit -m "docs: add GPT-2 matrix equations with full dimensions"
```

---

### Task 3: Create package structure and vector field extraction

**Files:**
- Create: `src/lm_electrostatics/__init__.py`
- Create: `src/lm_electrostatics/equations.py`
- Create: `tests/test_equations.py`

**Step 1: Create directory structure**

```bash
mkdir -p src/lm_electrostatics tests
```

**Step 2: Write the failing test**

`tests/test_equations.py`:
```python
import torch
from lm_electrostatics.equations import load_model, get_layer_output_fn

def test_layer_output_fn_shape():
    """X_l should have same shape as X_0 (flattened S*H)."""
    model, tokenizer = load_model()
    input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
    fn = get_layer_output_fn(model, layer_idx=0)
    # Get X_0 from embedding
    with torch.no_grad():
        x0 = model.transformer.wte(input_ids) + model.transformer.wpe(
            torch.arange(input_ids.shape[1], device=input_ids.device)
        )
    x0_flat = x0.squeeze(0).detach().clone().requires_grad_(True)
    x_l = fn(x0_flat)
    assert x_l.shape == x0_flat.shape

def test_layer_output_fn_differentiable():
    """Output must be differentiable w.r.t. X_0."""
    model, tokenizer = load_model()
    input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
    fn = get_layer_output_fn(model, layer_idx=0)
    with torch.no_grad():
        x0 = model.transformer.wte(input_ids) + model.transformer.wpe(
            torch.arange(input_ids.shape[1], device=input_ids.device)
        )
    x0_flat = x0.squeeze(0).detach().clone().requires_grad_(True)
    x_l = fn(x0_flat)
    loss = x_l.sum()
    loss.backward()
    assert x0_flat.grad is not None
    assert x0_flat.grad.shape == x0_flat.shape
```

**Step 3: Run tests, verify they fail**

```bash
poetry run pytest tests/test_equations.py -v
```
Expected: FAIL (module not found)

**Step 4: Implement `src/lm_electrostatics/__init__.py`**

Empty file.

**Step 5: Implement `src/lm_electrostatics/equations.py`**

Key functions:
- `get_device() -> torch.device`: Returns "cuda" if available, else "cpu"
- `load_model(device=None) -> (model, tokenizer)`: Load pre-trained `gpt2`, set eval mode, move to device
- `get_embedding(model, input_ids) -> Tensor`: Compute X_0 = W_e[input_ids] + W_p[positions], shape (S, H), detached with requires_grad
- `get_layer_output_fn(model, layer_idx) -> Callable`: Returns a function f(x0_flat) -> x_l_flat that:
  1. Takes flattened X_0 (S*H,), reshapes to (1, S, H)
  2. Runs through the transformer layers 0..layer_idx (manually calling each block)
  3. Returns flattened X_l (S*H,)

  CRITICAL: This function must build a differentiable computation graph from X_0 to X_l. The weights are frozen (no grad), but the input X_0 flows through with gradients enabled. Do NOT use hooks — instead, manually iterate through `model.transformer.h[0:layer_idx+1]` calling each block, passing the hidden state through. This avoids hook-related gradient issues.

**Step 6: Run tests, verify they pass**

```bash
poetry run pytest tests/test_equations.py -v
```
Expected: PASS

**Step 7: Commit**

```bash
git add src/ tests/ pyproject.toml
git commit -m "feat: add model loading and differentiable layer output extraction"
```

---

### Task 4: Implement stochastic divergence and Jacobian asymmetry

**Files:**
- Create: `src/lm_electrostatics/divergence.py`
- Create: `tests/test_divergence.py`

**Step 1: Write failing tests**

`tests/test_divergence.py`:
```python
import torch
from lm_electrostatics.divergence import estimate_divergence, estimate_asymmetry

def test_divergence_of_identity():
    """div(identity) = dimension."""
    def identity(x):
        return x
    d = 100
    x = torch.randn(d, requires_grad=True)
    div = estimate_divergence(identity, x, n_samples=200)
    assert abs(div - d) < d * 0.15  # within 15%

def test_divergence_of_linear():
    """div(Ax) = Tr(A)."""
    d = 50
    A = torch.randn(d, d)
    def linear_fn(x):
        return A @ x
    x = torch.randn(d, requires_grad=True)
    div = estimate_divergence(linear_fn, x, n_samples=200)
    expected = torch.trace(A).item()
    assert abs(div - expected) < abs(expected) * 0.2 + 1.0

def test_asymmetry_of_symmetric():
    """A symmetric Jacobian (gradient field) should have low asymmetry."""
    d = 50
    A = torch.randn(d, d)
    A_sym = (A + A.T) / 2
    def sym_fn(x):
        return A_sym @ x
    x = torch.randn(d, requires_grad=True)
    asym = estimate_asymmetry(sym_fn, x, n_samples=200)
    assert asym < 0.05

def test_asymmetry_of_antisymmetric():
    """An antisymmetric Jacobian should have high asymmetry."""
    d = 50
    A = torch.randn(d, d)
    A_anti = (A - A.T) / 2
    def anti_fn(x):
        return A_anti @ x
    x = torch.randn(d, requires_grad=True)
    asym = estimate_asymmetry(anti_fn, x, n_samples=200)
    assert asym > 0.8
```

**Step 2: Run tests, verify they fail**

```bash
poetry run pytest tests/test_divergence.py -v
```

**Step 3: Implement `src/lm_electrostatics/divergence.py`**

Two functions:

```python
def estimate_divergence(fn, x, n_samples=50):
    """
    Hutchinson trace estimator: div(F) = Tr(J) ≈ (1/K) Σ vᵀ(Jv)

    Uses torch.autograd.functional.jvp for Jv computation.
    v are Rademacher random vectors (+1/-1 with equal probability).

    Args:
        fn: Callable R^d -> R^d (must be differentiable)
        x: Input tensor (d,) with requires_grad=True
        n_samples: Number of random vectors K

    Returns:
        float: Estimated divergence
    """

def estimate_asymmetry(fn, x, n_samples=50):
    """
    Jacobian asymmetry via random projections.

    asymmetry = (1/K) Σ ||Jv - Jᵀv||² / (||Jv||² + ||Jᵀv||² + eps)

    Jv via torch.autograd.functional.jvp (forward-mode)
    Jᵀv via torch.autograd.functional.vjp (reverse-mode)

    Args:
        fn: Callable R^d -> R^d
        x: Input tensor (d,)
        n_samples: Number of random vectors K

    Returns:
        float: Asymmetry in [0, 1]. 0 = perfectly conservative.
    """
```

CRITICAL implementation notes:
- Use `torch.autograd.functional.jvp(fn, x, v)` for Jv — this gives (output, Jv) tuple
- Use `torch.autograd.functional.vjp(fn, x)` which returns (output, vjp_fn), then call `vjp_fn(v)` to get Jᵀv
- Rademacher vectors: `v = torch.randint(0, 2, (d,)).float() * 2 - 1`
- Add eps=1e-8 to denominator to avoid division by zero

**Step 4: Run tests, verify they pass**

```bash
poetry run pytest tests/test_divergence.py -v
```

**Step 5: Commit**

```bash
git add src/lm_electrostatics/divergence.py tests/test_divergence.py
git commit -m "feat: add stochastic divergence and Jacobian asymmetry estimation"
```

---

### Task 5: Implement perplexity computation

**Files:**
- Modify: `src/lm_electrostatics/equations.py` (add `compute_perplexity` function)
- Create: `tests/test_perplexity.py`

**Step 1: Write failing test**

`tests/test_perplexity.py`:
```python
from lm_electrostatics.equations import load_model, compute_perplexity

def test_perplexity_coherent_vs_random():
    """Coherent English should have lower perplexity than random tokens."""
    model, tokenizer = load_model()
    ppl_coherent = compute_perplexity(model, tokenizer, "The cat sat on the mat.")
    ppl_random = compute_perplexity(model, tokenizer, "Xq7 blorft znk oompa wibble.")
    assert ppl_coherent < ppl_random
```

**Step 2: Implement `compute_perplexity`**

```python
def compute_perplexity(model, tokenizer, text):
    """
    Compute perplexity of text under the model.

    PPL = exp( -(1/T) Σ log P(x_t | x_{<t}) )

    Use model's forward pass with labels=input_ids to get cross-entropy loss,
    then exp(loss).
    """
```

**Step 3: Run tests, verify pass**

```bash
poetry run pytest tests/test_perplexity.py -v
```

**Step 4: Commit**

```bash
git add src/lm_electrostatics/equations.py tests/test_perplexity.py
git commit -m "feat: add perplexity computation"
```

---

### Task 6: Implement main script with plots

**Files:**
- Create: `src/lm_electrostatics/main.py`

**Step 1: Implement main.py**

```python
"""
Main script: compute divergence and Jacobian asymmetry per layer
for 10 sentences, plot both vs perplexity.
"""

# 5 in-distribution sentences (WebText-like English):
IN_DIST = [
    "The president announced new economic policies during the press conference on Monday.",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
    "The stock market rallied sharply after the Federal Reserve cut interest rates.",
    "Researchers at MIT developed a new algorithm for natural language processing.",
    "The city council voted to approve the construction of a new public library.",
]

# 5 out-of-distribution sentences:
OUT_DIST = [
    "Xq7 blorft znk oompa wibble flurp.",
    "def __init__(self, x): self.x = x; return None",
    "㊀㊁㊂㊃㊄㊅㊆㊇㊈㊉",
    "aaa bbb ccc ddd eee fff ggg hhh iii jjj",
    "SELECT * FROM users WHERE id = 1; DROP TABLE users;--",
]

def main():
    # 1. Load model
    # 2. For each sentence:
    #    a. Compute perplexity
    #    b. Get X_0 embedding
    #    c. For each layer l (0..L-1), or a subset like [0, 3, 6, 9, 11]:
    #       - Get layer output function
    #       - Estimate divergence(f_l, x_0)
    #       - Estimate asymmetry(f_l, x_0)
    # 3. Average divergence and asymmetry across layers per sentence
    #    (or pick last layer — try last layer first, simpler)
    # 4. Print all values as lists
    # 5. Plot 1: divergence vs perplexity (separate colors for in/out dist)
    # 6. Plot 2: asymmetry vs perplexity (separate colors for in/out dist)
    # 7. Save plots to plots/ directory

if __name__ == "__main__":
    main()
```

Key details:
- Use `get_device()` for device selection (auto)
- Compute metrics for ALL 12 layers, report per-layer AND average
- Print format: labeled lists for each sentence
- Two matplotlib scatter plots saved as PNG
- Color code: blue = in-dist, red = out-of-dist
- Label each point with sentence index
- x-axis: perplexity, y-axis: metric

**Step 2: Run the script**

```bash
poetry run python -m lm_electrostatics.main
```

Verify it produces:
- Printed lists of divergences, asymmetries, perplexities
- Two plot PNG files in `plots/`

**Step 3: Commit**

```bash
git add src/lm_electrostatics/main.py plots/
git commit -m "feat: add main script with divergence/asymmetry vs perplexity analysis"
```

---

### Task 7: End-to-end verification

**Step 1: Run all tests**

```bash
poetry run pytest tests/ -v
```
Expected: ALL PASS

**Step 2: Run full pipeline**

```bash
poetry run python -m lm_electrostatics.main
```

Verify output includes:
- Perplexity values (in-dist should be lower)
- Divergence values per layer per sentence
- Asymmetry values per layer per sentence
- Two saved plots

**Step 3: Final commit if any cleanup needed**
