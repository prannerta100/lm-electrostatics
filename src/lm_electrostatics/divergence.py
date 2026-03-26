import torch
from torch.func import jvp as func_jvp, vjp as func_vjp, vmap


def _call_block(block, hidden, position_embeddings=None):
    """Call a transformer block, passing position_embeddings for RoPE models."""
    if position_embeddings is not None:
        out = block(hidden, position_embeddings=position_embeddings)
    else:
        out = block(hidden)
    return out if isinstance(out, torch.Tensor) else out[0]


def analyze_layers_hutchinson(blocks, H, x0, layer_indices, div_k, cons_k, position_embeddings=None):
    """
    Compute Hutchinson divergence and column-sampled conservative ratio per layer.

    All computation uses forward-mode AD only (JVPs, no VJPs). One incremental
    pass propagates tangent vectors block-by-block through all layers.

    Divergence: Tr(J) ≈ (1/K) Σ vᵀ(Jv), Rademacher v.  [Hutchinson 1990]
    Conservative ratio: ||S||²_F / ||J||²_F via column sampling.

        Method: JVP with basis vector e_j gives full column j of J.
        Sample K_cons random columns. For each pair (j_a, j_b), read off
        J_{j_a,j_b} and J_{j_b,j_a} exactly. Then:
            S_{ab} = (J_{j_a,j_b} + J_{j_b,j_a}) / 2   (symmetric part)
            Ω_{ab} = (J_{j_a,j_b} - J_{j_b,j_a}) / 2   (antisymmetric part)
            cons_ratio = Σ S²_{ab} / (Σ S²_{ab} + Σ Ω²_{ab})
        Range: 1.0 = conservative, 0.5 = random, 0.0 = purely rotational.
        Convergence: O(1/K_cons), independent of d.

    Args:
        blocks: list of transformer block modules
        H: hidden dimension
        x0: flattened embedding tensor (d,)
        layer_indices: list of layer indices to measure
        div_k: number of Rademacher vectors for Hutchinson divergence
        cons_k: number of basis columns to sample for conservative ratio
        position_embeddings: (cos, sin) for RoPE models, or None for GPT-2

    Returns:
        (divs, cons_ratios): dicts mapping layer_idx -> float
    """
    d = x0.shape[0]
    S = d // H
    max_layer = max(layer_indices)
    layer_set = set(layer_indices)

    # --- Build tangent vectors: div_k random + cons_k basis ---
    V_rand = torch.randint(0, 2, (div_k, d), dtype=x0.dtype, device=x0.device) * 2 - 1
    col_indices = torch.randperm(d, device=x0.device)[:cons_k]
    E_basis = torch.zeros(cons_k, d, dtype=x0.dtype, device=x0.device)
    E_basis[torch.arange(cons_k, device=x0.device), col_indices] = 1.0

    K = div_k + cons_k
    V_all = torch.cat([V_rand, E_basis], dim=0)  # (K, d)

    # --- Incremental JVP: propagate all tangents block by block ---
    hidden = x0.view(1, S, H)
    tangents = V_all.view(K, 1, S, H)

    jv_at_layer = {}

    for i in range(max_layer + 1):
        block = blocks[i]
        pos_emb = position_embeddings

        def _make_block_fn(b, pe=pos_emb):
            def block_fn(h):
                return _call_block(b, h, pe)
            return block_fn

        block_fn = _make_block_fn(block)
        h_in = hidden

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
        JV_rand = jv_at_layer[l][:div_k]  # (div_k, d)
        traces = (V_rand * JV_rand).sum(dim=1)
        divs[l] = traces.mean().item()

    # --- Conservative ratio via column sampling ---
    # JV_basis[k] = J @ e_{col_indices[k]} = column col_indices[k] of J
    # JV_basis[k][i] = J_{i, col_indices[k]}
    cons_ratios = {}
    eps = 1e-8

    for l in layer_indices:
        JV_basis = jv_at_layer[l][div_k:]  # (cons_k, d) — each row is a column of J

        # For each pair (a, b) of sampled columns, extract:
        #   J_{j_a, j_b} = JV_basis[b][j_a] = column j_b, row j_a
        #   J_{j_b, j_a} = JV_basis[a][j_b] = column j_a, row j_b
        j = col_indices  # (cons_k,)
        # cols[a] is the full column j_a of J, so cols[a][j_b] = J_{j_b, j_a}
        # Build cross-entry matrix: M[a,b] = J_{j_b, j_a} = JV_basis[a][j[b]]
        M = JV_basis[:, j]  # (cons_k, cons_k), M[a,b] = J_{j[b], j[a]}
        # So M[a,b] = J_{j_b, j_a} and M[b,a] = J_{j_a, j_b}
        # S_ab = (M[b,a] + M[a,b]) / 2,  Ω_ab = (M[b,a] - M[a,b]) / 2

        # Use upper triangle (a < b) to avoid double-counting
        idx = torch.triu_indices(cons_k, cons_k, offset=1, device=x0.device)
        J_ab = M[idx[1], idx[0]]  # J_{j_a, j_b} = M[b, a]
        J_ba = M[idx[0], idx[1]]  # J_{j_b, j_a} = M[a, b]

        S_sq = ((J_ab + J_ba) ** 2).sum()
        Omega_sq = ((J_ab - J_ba) ** 2).sum()
        cons_ratios[l] = (S_sq / (S_sq + Omega_sq + eps)).item()

    return divs, cons_ratios


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
