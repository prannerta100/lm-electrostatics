import torch


def estimate_divergence(fn, x, n_samples=50):
    """
    Hutchinson trace estimator: div(F) = Tr(J) ≈ (1/K) Σ vᵀ(Jv)

    Uses torch.autograd.functional.jvp for Jv computation.
    v are Rademacher random vectors (+1/-1 with equal probability).

    Args:
        fn: Callable R^d -> R^d (must be differentiable)
        x: Input tensor (d,) — does NOT need requires_grad (jvp handles it)
        n_samples: Number of random vectors K

    Returns:
        float: Estimated divergence
    """
    d = x.shape[0]
    trace_sum = 0.0
    for _ in range(n_samples):
        v = torch.randint(0, 2, (d,), dtype=x.dtype, device=x.device) * 2 - 1
        # jvp returns (output, Jv)
        _, jvp_val = torch.autograd.functional.jvp(fn, x, v)
        trace_sum += torch.dot(v, jvp_val).item()
    return trace_sum / n_samples


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
    d = x.shape[0]
    eps = 1e-8
    asym_sum = 0.0
    for _ in range(n_samples):
        v = torch.randint(0, 2, (d,), dtype=x.dtype, device=x.device) * 2 - 1

        # Jv via forward-mode AD
        _, jv = torch.autograd.functional.jvp(fn, x, v)

        # Jᵀv via reverse-mode AD
        _, jtv = torch.autograd.functional.vjp(fn, x, v)

        diff_sq = torch.sum((jv - jtv) ** 2).item()
        norm_sq = torch.sum(jv ** 2).item() + torch.sum(jtv ** 2).item() + eps
        asym_sum += diff_sq / norm_sq

    return asym_sum / n_samples
