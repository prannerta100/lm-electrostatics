#!/usr/bin/env python
"""
Validate the column-sampled conservativeness estimator against exact computation.

For GPT-2 small, d is small enough to compute the full Jacobian and get the
exact ||S||^2_F / ||J||^2_F ratio. Compare with column-sampled estimates at
various K values.
"""

import torch
from torch.func import jacfwd

from lm_electrostatics.equations import (
    load_model, get_embedding, get_layer_output_fn,
    _get_embed_dim, _get_num_layers, _get_layers, _get_position_embeddings,
)
from lm_electrostatics.divergence import analyze_layers_hutchinson


def exact_conservativeness(fn, x):
    """Compute exact ||S||^2_F / ||J||^2_F from the full Jacobian."""
    J = jacfwd(fn)(x)  # (d, d)
    S = (J + J.T) / 2
    Omega = (J - J.T) / 2
    S_sq = (S ** 2).sum().item()
    Omega_sq = (Omega ** 2).sum().item()
    ratio = S_sq / (S_sq + Omega_sq + 1e-8)
    return ratio, J


def column_sampled_conservativeness(J, cons_k, seed=42):
    """Column-sampled estimate from a precomputed Jacobian (for validation)."""
    d = J.shape[0]
    g = torch.Generator(device=J.device).manual_seed(seed)
    col_indices = torch.randperm(d, generator=g, device=J.device)[:cons_k]
    j = col_indices

    # Extract sampled columns
    JV_basis = J[:, j].T  # (cons_k, d) — each row is a column of J

    # Build cross-entry matrix
    M = JV_basis[:, j]  # (cons_k, cons_k), M[a,b] = J_{j[b], j[a]}
    idx = torch.triu_indices(cons_k, cons_k, offset=1, device=J.device)
    J_ab = M[idx[1], idx[0]]
    J_ba = M[idx[0], idx[1]]

    S_sq = ((J_ab + J_ba) ** 2).sum().item()
    Omega_sq = ((J_ab - J_ba) ** 2).sum().item()
    return S_sq / (S_sq + Omega_sq + 1e-8)


def main():
    print("Loading GPT-2 small...")
    model, tokenizer = load_model("gpt2")
    device = next(model.parameters()).device

    text = "Hello world"
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=8)["input_ids"].to(device)
    x0 = get_embedding(model, ids).reshape(-1)
    d = x0.shape[0]
    print(f"d = {d} (seq_len={ids.shape[1]}, H={_get_embed_dim(model)})")

    layers_to_test = [0, 2, 5, 11]
    num_layers = _get_num_layers(model)
    layers_to_test = [l for l in layers_to_test if l < num_layers]

    print(f"\nTesting layers: {layers_to_test}")
    print(f"{'Layer':<8} {'Exact':>10} {'K=50':>10} {'K=100':>10} {'K=200':>10} {'K=500':>10}")
    print("-" * 58)

    for l in layers_to_test:
        fn = get_layer_output_fn(model, l)

        # Exact
        ratio_exact, J = exact_conservativeness(fn, x0)

        # Column-sampled at various K
        estimates = {}
        for k in [50, 100, 200, 500]:
            # Average over 5 random seeds for stability
            vals = []
            for seed in range(5):
                vals.append(column_sampled_conservativeness(J, k, seed=seed))
            estimates[k] = sum(vals) / len(vals)

        print(f"L{l:<7} {ratio_exact:>10.4f} {estimates[50]:>10.4f} {estimates[100]:>10.4f} {estimates[200]:>10.4f} {estimates[500]:>10.4f}")

        # Also validate the Hutchinson pipeline estimate
        blocks = list(_get_layers(model))
        H = _get_embed_dim(model)
        S = ids.shape[1]
        pos_emb = _get_position_embeddings(model, S)
        divs, cons = analyze_layers_hutchinson(blocks, H, x0, [l], 10, 50, position_embeddings=pos_emb)
        print(f"  {'pipeline K=50':>18}: {cons[l]:.4f}")

        if x0.is_cuda:
            torch.cuda.empty_cache()

    # Also print exact divergence vs Hutchinson for sanity
    print(f"\n{'Layer':<8} {'Exact Tr(J)':>12} {'Hutch Tr(J)':>12}")
    print("-" * 35)
    for l in layers_to_test:
        fn = get_layer_output_fn(model, l)
        J = torch.func.jacfwd(fn)(x0)
        exact_div = torch.trace(J).item()

        blocks = list(_get_layers(model))
        H = _get_embed_dim(model)
        S = ids.shape[1]
        pos_emb = _get_position_embeddings(model, S)
        divs, _ = analyze_layers_hutchinson(blocks, H, x0, [l], 50, 0, position_embeddings=pos_emb)

        print(f"L{l:<7} {exact_div:>12.2f} {divs[l]:>12.2f}")


if __name__ == "__main__":
    main()
