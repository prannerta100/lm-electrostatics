from lm_electrostatics.equations import load_model, compute_perplexity


def test_perplexity_coherent_vs_random():
    """Coherent English should have lower perplexity than random tokens."""
    model, tokenizer = load_model()
    ppl_coherent = compute_perplexity(model, tokenizer, "The cat sat on the mat.")
    ppl_random = compute_perplexity(model, tokenizer, "Xq7 blorft znk oompa wibble.")
    assert ppl_coherent < ppl_random


def test_perplexity_is_positive():
    model, tokenizer = load_model()
    ppl = compute_perplexity(model, tokenizer, "Hello world.")
    assert ppl > 0
