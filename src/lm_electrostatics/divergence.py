import torch
from torch.func import jvp as func_jvp, vjp as func_vjp, vmap


def analyze_layers_hutchinson(blocks, H, x0, layer_indices, div_k, asym_k):
    """
    Compute Hutchinson divergence and asymmetry at multiple layers efficiently.

    JVP tangent vectors are propagated incrementally block-by-block, so all
    layers share a single forward pass instead of each layer recomputing from x0.
    VJP (for asymmetry) still requires per-layer backward passes.

    Args:
        blocks: list of transformer block modules
        H: hidden dimension
        x0: flattened embedding tensor (d,)
        layer_indices: list of layer indices to measure (must be sorted)
        div_k: number of Hutchinson samples for divergence
        asym_k: number of random vectors for asymmetry

    Returns:
        (divs, asyms): dicts mapping layer_idx -> float
    """
    x0 = x0.float()
    d = x0.shape[0]
    S = d // H
    max_layer = max(layer_indices)
    layer_set = set(layer_indices)

    K = max(div_k, asym_k)
    V = torch.randint(0, 2, (K, d), dtype=x0.dtype, device=x0.device) * 2 - 1

    # --- Incremental JVP: propagate tangents block by block ---
    hidden = x0.view(1, S, H)
    tangents = V.view(K, 1, S, H)

    jv_at_layer = {}

    for i in range(max_layer + 1):
        block = blocks[i]

        def _make_block_fn(b):
            def block_fn(h):
                out = b(h)
                return out if isinstance(out, torch.Tensor) else out[0]
            return block_fn

        block_fn = _make_block_fn(block)
        h_in = hidden  # capture current hidden for JVP

        def _single_jvp(t, _h=h_in, _fn=block_fn):
            _, jv = func_jvp(_fn, (_h,), (t,))
            return jv

        tangents = vmap(_single_jvp)(tangents)
        hidden = block_fn(hidden)

        if i in layer_set:
            jv_at_layer[i] = tangents.reshape(K, d).clone()

        if x0.is_cuda:
            torch.cuda.empty_cache()

    # --- Divergence: Tr(J) ≈ (1/K) Σ vᵀ(Jv) ---
    divs = {}
    for l in layer_indices:
        JV = jv_at_layer[l][:div_k]
        traces = (V[:div_k] * JV).sum(dim=1)
        divs[l] = traces.mean().item()

    # --- Asymmetry: need VJP per layer (no incremental shortcut) ---
    asyms = {}
    eps = 1e-8
    V_asym = V[:asym_k]

    for l in layer_indices:
        JV_l = jv_at_layer[l][:asym_k]

        def _make_layer_fn(layer_idx):
            def fn(x0_flat):
                h = x0_flat.view(1, S, H)
                for j in range(layer_idx + 1):
                    out = blocks[j](h)
                    h = out if isinstance(out, torch.Tensor) else out[0]
                return h.squeeze(0).reshape(-1)
            return fn

        fn_l = _make_layer_fn(l)
        _, vjp_fn = func_vjp(fn_l, x0)

        def _vjp_one(v, _vjp_fn=vjp_fn):
            return _vjp_fn(v)[0]

        JtV = vmap(_vjp_one)(V_asym)

        diff_sq = ((JV_l - JtV) ** 2).sum(dim=1)
        norm_sq = (JV_l ** 2).sum(dim=1) + (JtV ** 2).sum(dim=1) + eps
        asyms[l] = (diff_sq / norm_sq).mean().item()

        if x0.is_cuda:
            torch.cuda.empty_cache()

    return divs, asyms


def exact_divergence(fn, x, chunk_size=0):
    """
    Exact divergence: div(F) = Tr(J) = Σᵢ ∂Fᵢ/∂xᵢ.

    When chunk_size=0: batched computation of all diagonal elements using
    vmap over standard basis vectors. Uses O(d) memory for results but
    batches the forward-mode AD calls for GPU parallelism.

    When chunk_size>0: same batched approach but processes basis vectors
    in chunks to limit peak memory usage.

    Args:
        fn: Callable R^d -> R^d (must be differentiable)
        x: Input tensor (d,)
        chunk_size: If >0, process basis vectors in chunks of this size.
                    If 0, process all at once (maximizes GPU utilization).

    Returns:
        float: Exact divergence
    """
    x = x.float()
    d = x.shape[0]

    def _diag_jvp(e_i):
        """Compute eᵢᵀ J eᵢ = J[i,i] for one basis vector."""
        _, jv = func_jvp(fn, (x,), (e_i,))
        return torch.dot(e_i, jv)

    if chunk_size <= 0:
        I = torch.eye(d, dtype=x.dtype, device=x.device)
        diag = vmap(_diag_jvp)(I)
        return diag.sum().item()

    # Chunked batched computation
    trace_val = 0.0
    for start in range(0, d, chunk_size):
        end = min(start + chunk_size, d)
        # Build partial identity: only rows start..end
        chunk = torch.zeros(end - start, d, dtype=x.dtype, device=x.device)
        for i in range(end - start):
            chunk[i, start + i] = 1.0
        diag_chunk = vmap(_diag_jvp)(chunk)
        trace_val += diag_chunk.sum().item()
        if x.is_cuda:
            torch.cuda.empty_cache()
    return trace_val


def estimate_divergence(fn, x, n_samples=50):
    """
    Hutchinson trace estimator: div(F) = Tr(J) ≈ (1/K) Σ vᵀ(Jv)

    Batched via vmap over Rademacher random vectors — all K jvp calls
    run as one batched GPU operation instead of K sequential calls.

    Args:
        fn: Callable R^d -> R^d (must be differentiable)
        x: Input tensor (d,)
        n_samples: Number of random vectors K

    Returns:
        float: Estimated divergence
    """
    x = x.float()
    d = x.shape[0]
    V = torch.randint(0, 2, (n_samples, d), dtype=x.dtype, device=x.device) * 2 - 1

    def _trace_sample(v):
        _, jv = func_jvp(fn, (x,), (v,))
        return torch.dot(v, jv)

    traces = vmap(_trace_sample)(V)
    return traces.mean().item()


def estimate_asymmetry(fn, x, n_samples=50):
    """
    Jacobian asymmetry via random projections (batched).

    asymmetry = (1/K) Σ ||Jv - Jᵀv||² / (||Jv||² + ||Jᵀv||² + eps)

    Jv via batched forward-mode AD (vmap + jvp)
    Jᵀv via batched reverse-mode AD (vmap + vjp)

    Args:
        fn: Callable R^d -> R^d
        x: Input tensor (d,)
        n_samples: Number of random vectors K

    Returns:
        float: Asymmetry in [0, 2]. 0 = perfectly symmetric (conservative), 2 = purely antisymmetric.
    """
    x = x.float()
    d = x.shape[0]
    eps = 1e-8
    V = torch.randint(0, 2, (n_samples, d), dtype=x.dtype, device=x.device) * 2 - 1

    # Batched Jv via forward-mode
    def _jvp_one(v):
        _, jv = func_jvp(fn, (x,), (v,))
        return jv

    # Batched Jᵀv via reverse-mode
    def _vjp_one(v):
        _, vjp_fn = func_vjp(fn, x)
        return vjp_fn(v)[0]

    JV = vmap(_jvp_one)(V)    # (K, d)
    JtV = vmap(_vjp_one)(V)   # (K, d)

    diff_sq = ((JV - JtV) ** 2).sum(dim=1)           # (K,)
    norm_sq = (JV ** 2).sum(dim=1) + (JtV ** 2).sum(dim=1) + eps  # (K,)
    return (diff_sq / norm_sq).mean().item()
