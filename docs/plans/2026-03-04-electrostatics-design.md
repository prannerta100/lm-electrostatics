# LM Electrostatics — Design

## Goal

Investigate the vector field structure of GPT-2 layer outputs as functions of input embeddings. Measure two quantities per layer:
1. **Divergence** (Tr(J)) — source/sink behavior
2. **Jacobian asymmetry** (||J - J^T|| metric) — how non-conservative the field is

Plot both against perplexity for 10 sentences (5 in-dist, 5 out-of-dist).

## Model

Pre-trained `gpt2` (124M, H=768, N=12, L=12) with short sequences (~10 tokens). Real weights give meaningful perplexity.

## Jacobian estimation

Full Jacobian at dim 768*S is too expensive. Use stochastic estimation with K=50 random vectors:

- **Divergence**: Hutchinson trace estimator: `div(F) = Tr(J) ≈ (1/K) Σ vᵀ(Jv)` where Jv is a JVP (forward-mode AD)
- **Asymmetry**: `(1/K) Σ ||Jv - Jᵀv||² / (||Jv||² + ||Jᵀv||²)` where Jv = JVP, Jᵀv = VJP

## Vector field definition

X_0 = Embedding(input_ids), shape (S, H), flattened to (S*H,).
X_l = output of layer l, same shape, flattened.
f_l: R^(S*H) → R^(S*H) maps X_0 to X_l with all layer weights frozen.

## Files

- `gpt2_equations.md` — Full GPT-2 matrix equations with dimensions
- `src/lm_electrostatics/equations.py` — Hook-based X_l extraction, vector field wrapper
- `src/lm_electrostatics/divergence.py` — Stochastic divergence + asymmetry
- `src/lm_electrostatics/main.py` — Load model, 10 sentences, compute, plot, print

## Data

5 in-distribution (WebText-like English), 5 out-of-distribution (nonsense, code, random).

## Output

- Two matplotlib plots: divergence vs perplexity, asymmetry vs perplexity (per layer, or averaged)
- All values printed as lists
