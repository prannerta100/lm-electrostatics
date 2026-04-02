"""
Attention-sublayer-only analysis: isolate the attention component from full blocks.

Computes divergence and conservativeness for:
  - Per-layer: A_l = X_{l-1} + Attention(LayerNorm(X_{l-1}))
  - Composed: Stacking attention sublayers only (no MLP)
"""

import torch
from torch.func import jvp as func_jvp, vmap


def _call_attention_sublayer(block, hidden, position_embeddings=None):
    """
    Call just the attention sublayer: X + Attn(LN(X)).

    Handles different transformer architectures:
    - GPT-2: block.attn, block.ln_1
    - LLaMA/Qwen/Mistral: block.self_attn, block.input_layernorm
    - GPT-NeoX/Pythia: similar to LLaMA
    """
    # Identify attention and layernorm modules
    if hasattr(block, 'attn') and hasattr(block, 'ln_1'):
        # GPT-2 style
        ln = block.ln_1
        attn = block.attn
    elif hasattr(block, 'self_attn') and hasattr(block, 'input_layernorm'):
        # LLaMA/Qwen/Mistral style
        ln = block.input_layernorm
        attn = block.self_attn
    else:
        raise ValueError(f"Cannot find attention sublayer in {type(block).__name__}")

    # Compute: X + Attn(LN(X))
    normed = ln(hidden)

    if position_embeddings is not None:
        attn_out = attn(normed, position_embeddings=position_embeddings)
    else:
        attn_out = attn(normed)

    # Handle tuple output (some models return (output, attention_weights))
    if isinstance(attn_out, tuple):
        attn_out = attn_out[0]

    return hidden + attn_out


def analyze_attention_perlayer(blocks, H, x0, layer_indices, div_k, cons_k, position_embeddings=None):
    """
    Per-layer attention sublayer Jacobian: ∂(X + Attn(LN(X)))/∂X.

    For each layer l, compute divergence and conservativeness of just the
    attention update, not the full block (which includes MLP).

    Args:
        blocks: list of transformer block modules
        H: hidden dimension
        x0: flattened embedding tensor (d,)
        layer_indices: list of layer indices to measure
        div_k: number of Rademacher vectors for Hutchinson divergence
        cons_k: number of basis columns to sample for conservative ratio
        position_embeddings: (cos, sin) for RoPE models, or None

    Returns:
        (divs, cons_ratios): dicts mapping layer_idx -> float
    """
    d = x0.shape[0]
    S = d // H
    max_layer = max(layer_indices)
    layer_set = set(layer_indices)

    # Forward pass to get hidden states at each layer (full blocks)
    from .divergence import _call_block
    hidden = x0.view(1, S, H)
    hidden_at_layer = {-1: hidden}

    for i in range(max_layer + 1):
        block = blocks[i]
        hidden = _call_block(block, hidden, position_embeddings)
        if i in layer_set or (i + 1) in layer_set:
            hidden_at_layer[i] = hidden

    # For each target layer, do JVP of just the attention sublayer
    divs = {}
    cons_ratios = {}
    eps = 1e-8

    for l in layer_indices:
        h_in = hidden_at_layer[l - 1]  # input to block l
        h_in_flat = h_in.reshape(-1)

        block = blocks[l]
        pos_emb = position_embeddings

        def _make_attn_fn(b, pe=pos_emb):
            def attn_fn(h):
                return _call_attention_sublayer(b, h, pe)
            return attn_fn

        attn_fn = _make_attn_fn(block)

        # Build tangent vectors
        V_rand = torch.randint(0, 2, (div_k, d), dtype=x0.dtype, device=x0.device) * 2 - 1
        col_indices = torch.randperm(d, device=x0.device)[:cons_k]
        E_basis = torch.zeros(cons_k, d, dtype=x0.dtype, device=x0.device)
        E_basis[torch.arange(cons_k, device=x0.device), col_indices] = 1.0

        K = div_k + cons_k
        V_all = torch.cat([V_rand, E_basis], dim=0).view(K, 1, S, H)

        # JVP for attention sublayer only
        def _single_jvp(t, _h=h_in, _fn=attn_fn):
            _, jv = func_jvp(_fn, (_h,), (t,))
            return jv

        JV_all = vmap(_single_jvp)(V_all).reshape(K, d)

        # Divergence
        JV_rand = JV_all[:div_k]
        traces = (V_rand * JV_rand).sum(dim=1)
        divs[l] = traces.mean().item()

        # Conservative ratio
        JV_basis = JV_all[div_k:]
        j = col_indices
        M = JV_basis[:, j]
        idx = torch.triu_indices(cons_k, cons_k, offset=1, device=x0.device)
        J_ab = M[idx[1], idx[0]]
        J_ba = M[idx[0], idx[1]]
        S_sq = ((J_ab + J_ba) ** 2).sum()
        Omega_sq = ((J_ab - J_ba) ** 2).sum()
        cons_ratios[l] = (S_sq / (S_sq + Omega_sq + eps)).item()

        if x0.is_cuda:
            torch.cuda.empty_cache()

    return divs, cons_ratios


def analyze_attention_composed(blocks, H, x0, layer_indices, div_k, cons_k, position_embeddings=None):
    """
    Composed attention-only Jacobian: apply only attention sublayers, skip MLP.

    Computes ∂(AttentionChain_0...l)/∂X_0 where AttentionChain applies only
    attention sublayers sequentially.

    This shows whether attention alone is more rotational than the full network.

    Args:
        blocks: list of transformer block modules
        H: hidden dimension
        x0: flattened embedding tensor (d,)
        layer_indices: list of layer indices to measure
        div_k: number of Rademacher vectors for Hutchinson divergence
        cons_k: number of basis columns to sample for conservative ratio
        position_embeddings: (cos, sin) for RoPE models, or None

    Returns:
        (divs, cons_ratios): dicts mapping layer_idx -> float
    """
    d = x0.shape[0]
    S = d // H
    max_layer = max(layer_indices)
    layer_set = set(layer_indices)

    # Build tangent vectors
    V_rand = torch.randint(0, 2, (div_k, d), dtype=x0.dtype, device=x0.device) * 2 - 1
    col_indices = torch.randperm(d, device=x0.device)[:cons_k]
    E_basis = torch.zeros(cons_k, d, dtype=x0.dtype, device=x0.device)
    E_basis[torch.arange(cons_k, device=x0.device), col_indices] = 1.0

    K = div_k + cons_k
    V_all = torch.cat([V_rand, E_basis], dim=0)

    # Incremental JVP through attention sublayers only
    hidden = x0.view(1, S, H)
    tangents = V_all.view(K, 1, S, H)
    jv_at_layer = {}

    for i in range(max_layer + 1):
        block = blocks[i]
        pos_emb = position_embeddings

        def _make_attn_fn(b, pe=pos_emb):
            def attn_fn(h):
                return _call_attention_sublayer(b, h, pe)
            return attn_fn

        attn_fn = _make_attn_fn(block)
        h_in = hidden

        # Propagate tangents through attention sublayer
        def _single_jvp(t, _h=h_in, _fn=attn_fn):
            _, jv = func_jvp(_fn, (_h,), (t,))
            return jv

        tangents = vmap(_single_jvp)(tangents)
        hidden = attn_fn(hidden)

        if i in layer_set:
            jv_at_layer[i] = tangents.reshape(K, d).clone()

        if x0.is_cuda:
            torch.cuda.empty_cache()

    # Compute divergence and conservativeness at each layer
    divs = {}
    cons_ratios = {}
    eps = 1e-8

    for l in layer_indices:
        JV_all = jv_at_layer[l]

        # Divergence
        JV_rand = JV_all[:div_k]
        traces = (V_rand * JV_rand).sum(dim=1)
        divs[l] = traces.mean().item()

        # Conservative ratio
        JV_basis = JV_all[div_k:]
        j = col_indices
        M = JV_basis[:, j]
        idx = torch.triu_indices(cons_k, cons_k, offset=1, device=x0.device)
        J_ab = M[idx[1], idx[0]]
        J_ba = M[idx[0], idx[1]]
        S_sq = ((J_ab + J_ba) ** 2).sum()
        Omega_sq = ((J_ab - J_ba) ** 2).sum()
        cons_ratios[l] = (S_sq / (S_sq + Omega_sq + eps)).item()

    return divs, cons_ratios
