"""
Main script: compute divergence and Jacobian asymmetry per layer
for 10 sentences, plot both vs perplexity.
"""

import os
import torch
import matplotlib.pyplot as plt

from lm_electrostatics.equations import (
    load_model,
    get_embedding,
    get_layer_output_fn,
    compute_perplexity,
)
from lm_electrostatics.divergence import estimate_divergence, estimate_asymmetry

# 5 in-distribution sentences (WebText-like English)
IN_DIST = [
    "The president announced new economic policies during the press conference on Monday.",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
    "The stock market rallied sharply after the Federal Reserve cut interest rates.",
    "Researchers at MIT developed a new algorithm for natural language processing.",
    "The city council voted to approve the construction of a new public library.",
]

# 5 out-of-distribution sentences
OUT_DIST = [
    "Xq7 blorft znk oompa wibble flurp.",
    "def __init__(self, x): self.x = x; return None",
    "aaa bbb ccc ddd eee fff ggg hhh iii jjj",
    "SELECT * FROM users WHERE id = 1; DROP TABLE users;--",
    "12345 67890 11111 22222 33333 44444 55555",
]

# Layers to analyze (subset for speed)
LAYER_INDICES = [0, 3, 6, 9, 11]

N_SAMPLES = 50  # random vectors for stochastic estimation


def analyze_sentence(model, tokenizer, text, layer_indices, n_samples):
    """Compute perplexity, divergence, and asymmetry for a sentence."""
    device = next(model.parameters()).device
    ppl = compute_perplexity(model, tokenizer, text)

    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    x0 = get_embedding(model, input_ids)
    x0_flat = x0.reshape(-1)

    divergences = {}
    asymmetries = {}

    for l in layer_indices:
        fn = get_layer_output_fn(model, l)
        div = estimate_divergence(fn, x0_flat, n_samples=n_samples)
        asym = estimate_asymmetry(fn, x0_flat, n_samples=n_samples)
        divergences[l] = div
        asymmetries[l] = asym
        print(f"    Layer {l:2d}: div={div:10.2f}, asym={asym:.4f}")

    avg_div = sum(divergences.values()) / len(divergences)
    avg_asym = sum(asymmetries.values()) / len(asymmetries)

    return {
        "perplexity": ppl,
        "divergences": divergences,
        "asymmetries": asymmetries,
        "avg_divergence": avg_div,
        "avg_asymmetry": avg_asym,
    }


def main():
    print("Loading model...")
    model, tokenizer = load_model()
    num_layers = model.config.n_layer
    print(f"GPT-2: H={model.config.n_embd}, N={model.config.n_head}, L={num_layers}")
    print(f"Analyzing layers: {LAYER_INDICES}")
    print(f"Stochastic samples: {N_SAMPLES}")
    print()

    all_sentences = IN_DIST + OUT_DIST
    labels = ["in"] * len(IN_DIST) + ["out"] * len(OUT_DIST)
    results = []

    for i, (text, label) in enumerate(zip(all_sentences, labels)):
        print(f"[{i+1}/{len(all_sentences)}] ({label}-dist) {text[:60]}...")
        result = analyze_sentence(model, tokenizer, text, LAYER_INDICES, N_SAMPLES)
        result["text"] = text
        result["label"] = label
        results.append(result)
        print(f"    PPL={result['perplexity']:.2f}, "
              f"avg_div={result['avg_divergence']:.2f}, "
              f"avg_asym={result['avg_asymmetry']:.4f}")
        print()

    # Print all values as lists
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ppls = [r["perplexity"] for r in results]
    avg_divs = [r["avg_divergence"] for r in results]
    avg_asyms = [r["avg_asymmetry"] for r in results]
    print(f"Perplexities:       {ppls}")
    print(f"Avg Divergences:    {avg_divs}")
    print(f"Avg Asymmetries:    {avg_asyms}")
    print(f"Labels:             {labels}")
    print()

    for l in LAYER_INDICES:
        divs_l = [r["divergences"][l] for r in results]
        asyms_l = [r["asymmetries"][l] for r in results]
        print(f"Layer {l:2d} divergences:  {divs_l}")
        print(f"Layer {l:2d} asymmetries:  {asyms_l}")
    print()

    # Plots
    os.makedirs("plots", exist_ok=True)

    in_mask = [l == "in" for l in labels]
    out_mask = [l == "out" for l in labels]

    # Plot 1: Divergence vs Perplexity
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, r in enumerate(results):
        color = "blue" if r["label"] == "in" else "red"
        ax.scatter(r["perplexity"], r["avg_divergence"], c=color, s=80, zorder=3)
        ax.annotate(str(i), (r["perplexity"], r["avg_divergence"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("Average Divergence (Tr(J))")
    ax.set_title("Divergence vs Perplexity")
    ax.scatter([], [], c="blue", label="In-distribution")
    ax.scatter([], [], c="red", label="Out-of-distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/divergence_vs_perplexity.png", dpi=150)
    print("Saved plots/divergence_vs_perplexity.png")

    # Plot 2: Asymmetry vs Perplexity
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, r in enumerate(results):
        color = "blue" if r["label"] == "in" else "red"
        ax.scatter(r["perplexity"], r["avg_asymmetry"], c=color, s=80, zorder=3)
        ax.annotate(str(i), (r["perplexity"], r["avg_asymmetry"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Perplexity")
    ax.set_ylabel("Average Jacobian Asymmetry")
    ax.set_title("Jacobian Asymmetry vs Perplexity")
    ax.scatter([], [], c="blue", label="In-distribution")
    ax.scatter([], [], c="red", label="Out-of-distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/asymmetry_vs_perplexity.png", dpi=150)
    print("Saved plots/asymmetry_vs_perplexity.png")

    plt.close("all")


if __name__ == "__main__":
    main()
