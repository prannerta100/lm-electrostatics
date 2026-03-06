import torch
from lm_electrostatics.divergence import estimate_divergence, estimate_asymmetry


def test_divergence_of_identity():
    """div(identity) = dimension."""
    def identity(x):
        return x.clone()  # clone needed so autograd sees a computation
    d = 100
    x = torch.randn(d)
    div = estimate_divergence(identity, x, n_samples=200)
    assert abs(div - d) < d * 0.15, f"Expected ~{d}, got {div}"


def test_divergence_of_linear():
    """div(Ax) = Tr(A)."""
    d = 50
    A = torch.randn(d, d)
    def linear_fn(x):
        return A @ x
    x = torch.randn(d)
    div = estimate_divergence(linear_fn, x, n_samples=200)
    expected = torch.trace(A).item()
    assert abs(div - expected) < max(abs(expected) * 0.5, 8.0), f"Expected ~{expected}, got {div}"


def test_asymmetry_of_symmetric():
    """A symmetric Jacobian (gradient field) should have low asymmetry."""
    d = 50
    A = torch.randn(d, d)
    A_sym = (A + A.T) / 2
    def sym_fn(x):
        return A_sym @ x
    x = torch.randn(d)
    asym = estimate_asymmetry(sym_fn, x, n_samples=200)
    assert asym < 0.05, f"Expected near 0, got {asym}"


def test_asymmetry_of_antisymmetric():
    """An antisymmetric Jacobian should have high asymmetry."""
    d = 50
    A = torch.randn(d, d)
    A_anti = (A - A.T) / 2
    def anti_fn(x):
        return A_anti @ x
    x = torch.randn(d)
    asym = estimate_asymmetry(anti_fn, x, n_samples=200)
    assert asym > 0.8, f"Expected near 1, got {asym}"


def test_divergence_of_scaled():
    """div(2x) = 2*d."""
    def scale_fn(x):
        return 2.0 * x
    d = 80
    x = torch.randn(d)
    div = estimate_divergence(scale_fn, x, n_samples=200)
    assert abs(div - 2 * d) < 2 * d * 0.15, f"Expected ~{2*d}, got {div}"
