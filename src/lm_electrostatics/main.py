"""
Main script: compute divergence and Jacobian asymmetry per layer
for sentences, plot both vs perplexity.
"""

import argparse
import json
import os
import time
import torch
import plotly.graph_objects as go

from lm_electrostatics.equations import (
    load_model,
    get_embedding,
    get_layer_output_fn,
    compute_perplexity,
    _get_embed_dim,
    _get_num_layers,
)
from lm_electrostatics.divergence import exact_divergence, estimate_asymmetry

# Default sentences (used when no data files are provided)
DEFAULT_IN_DIST = [
    "The president announced new economic policies during the press conference on Monday.",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
    "The stock market rallied sharply after the Federal Reserve cut interest rates.",
    "Researchers at MIT developed a new algorithm for natural language processing.",
    "The city council voted to approve the construction of a new public library.",
]

DEFAULT_OUT_DIST = [
    "Purple hospital the running fly water not because window seven jump quickly door.",
    "the the the the the the the the the the the the the",
    "One two three four five six seven eight nine ten eleven twelve thirteen fourteen.",
    "Zero zero zero one one one two two two three three three four.",
    "Banana chair quickly seven always under paper fly window table door blue.",
]


def load_sentences_from_file(path):
    """Load sentences from a text file (one per line) or JSONL (field: 'text')."""
    sentences = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                obj = json.loads(line)
                sentences.append(obj["text"])
            else:
                sentences.append(line)
    return sentences


def analyze_sentence(model, tokenizer, text, layer_indices, n_samples, chunk_size):
    """Compute perplexity, exact divergence, and stochastic asymmetry for a sentence."""
    device = next(model.parameters()).device
    H = _get_embed_dim(model)
    ppl = compute_perplexity(model, tokenizer, text)

    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    x0 = get_embedding(model, input_ids)
    x0_flat = x0.reshape(-1)
    print(f"    d = {x0_flat.shape[0]} (S={input_ids.shape[1]}, H={H})")

    divergences = {}
    asymmetries = {}

    for l in layer_indices:
        fn = get_layer_output_fn(model, l)
        t0 = time.time()
        div = exact_divergence(fn, x0_flat, chunk_size=chunk_size)
        t_div = time.time() - t0
        asym = estimate_asymmetry(fn, x0_flat, n_samples=n_samples)
        divergences[l] = div
        asymmetries[l] = asym
        print(f"    Layer {l:2d}: div={div:10.2f} (exact, {t_div:.1f}s), asym={asym:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_div = sum(divergences.values()) / len(divergences)
    avg_asym = sum(asymmetries.values()) / len(asymmetries)

    return {
        "perplexity": ppl,
        "divergences": divergences,
        "asymmetries": asymmetries,
        "avg_divergence": avg_div,
        "avg_asymmetry": avg_asym,
    }


def parse_layer_indices(spec, num_layers):
    """Parse layer index specification.

    Examples: "0,3,6,9,11", "all", "uniform:5" (5 uniformly spaced layers).
    """
    if spec == "all":
        return list(range(num_layers))
    if spec.startswith("uniform:"):
        n = int(spec.split(":")[1])
        step = max(1, (num_layers - 1) / (n - 1)) if n > 1 else num_layers
        indices = [round(i * step) for i in range(n)]
        return sorted(set(min(idx, num_layers - 1) for idx in indices))
    return [int(x) for x in spec.split(",")]


def make_plots(results, layer_indices, output_dir):
    """Generate and save interactive (HTML) and static (PNG) plots."""
    os.makedirs(output_dir, exist_ok=True)

    in_results = [r for r in results if r["label"] == "in"]
    out_results = [r for r in results if r["label"] == "out"]

    def _hover_text(r):
        return f"{r['text'][:80]}...<br>PPL={r['perplexity']:.2f}, div={r['avg_divergence']:.2f}"

    # --- Plot 1: Divergence vs Perplexity (interactive) ---
    fig = go.Figure()
    for label, group, color in [("In-distribution", in_results, "blue"), ("Out-of-distribution", out_results, "red")]:
        fig.add_trace(go.Scatter(
            x=[r["perplexity"] for r in group],
            y=[r["avg_divergence"] for r in group],
            mode="markers+text",
            marker=dict(size=10, color=color),
            text=[str(results.index(r)) for r in group],
            textposition="top right",
            textfont=dict(size=9),
            hovertext=[_hover_text(r) for r in group],
            hoverinfo="text",
            name=label,
        ))
    fig.update_layout(
        title="Exact Divergence vs Perplexity",
        xaxis_title="Perplexity",
        yaxis_title="Average Divergence Tr(J) [exact]",
        hovermode="closest",
    )
    fig.write_html(os.path.join(output_dir, "divergence_vs_perplexity.html"))
    print(f"Saved {output_dir}/divergence_vs_perplexity.html")

    # --- Plot 2: Asymmetry vs Perplexity (interactive) ---
    fig = go.Figure()
    for label, group, color in [("In-distribution", in_results, "blue"), ("Out-of-distribution", out_results, "red")]:
        fig.add_trace(go.Scatter(
            x=[r["perplexity"] for r in group],
            y=[r["avg_asymmetry"] for r in group],
            mode="markers+text",
            marker=dict(size=10, color=color),
            text=[str(results.index(r)) for r in group],
            textposition="top right",
            textfont=dict(size=9),
            hovertext=[_hover_text(r) for r in group],
            hoverinfo="text",
            name=label,
        ))
    fig.update_layout(
        title="Jacobian Asymmetry vs Perplexity",
        xaxis_title="Perplexity",
        yaxis_title="Average Jacobian Asymmetry",
        hovermode="closest",
    )
    fig.write_html(os.path.join(output_dir, "asymmetry_vs_perplexity.html"))
    print(f"Saved {output_dir}/asymmetry_vs_perplexity.html")

    # --- Plot 3: Divergence vs Layer (interactive) ---
    fig = go.Figure()
    for i, r in enumerate(results):
        color = "blue" if r["label"] == "in" else "red"
        divs = [r["divergences"][l] for l in layer_indices]
        fig.add_trace(go.Scatter(
            x=layer_indices,
            y=divs,
            mode="lines+markers",
            marker=dict(size=6, color=color),
            line=dict(color=color),
            opacity=0.6,
            name=f"[{i}] {r['label']} (PPL={r['perplexity']:.1f})",
            hovertext=r["text"][:60],
        ))
    fig.update_layout(
        title="Divergence vs Layer",
        xaxis_title="Layer Index",
        yaxis_title="Divergence Tr(J) [exact]",
        hovermode="x unified",
    )
    fig.write_html(os.path.join(output_dir, "divergence_vs_layer.html"))
    print(f"Saved {output_dir}/divergence_vs_layer.html")


def save_results_json(results, output_dir):
    """Save all results to a JSON file for later analysis."""
    os.makedirs(output_dir, exist_ok=True)
    # Convert int keys to strings for JSON
    serializable = []
    for r in results:
        sr = {
            "text": r["text"],
            "label": r["label"],
            "perplexity": r["perplexity"],
            "avg_divergence": r["avg_divergence"],
            "avg_asymmetry": r["avg_asymmetry"],
            "divergences": {str(k): v for k, v in r["divergences"].items()},
            "asymmetries": {str(k): v for k, v in r["asymmetries"].items()},
        }
        serializable.append(sr)
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="LM Electrostatics: divergence & asymmetry analysis")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"],
                        help="Model dtype (default: model default). Note: autograd runs in float32 regardless.")
    parser.add_argument("--in-dist", default=None, help="Path to in-distribution sentences file (one per line or JSONL)")
    parser.add_argument("--out-dist", default=None, help="Path to out-of-distribution sentences file")
    parser.add_argument("--layers", default=None,
                        help="Layer indices: comma-separated (0,3,6), 'all', or 'uniform:N' (default: uniform:5)")
    parser.add_argument("--n-samples", type=int, default=50, help="Random vectors for asymmetry estimation (default: 50)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Chunk size for exact divergence (0=full Jacobian, >0=chunked for memory savings)")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots and results (default: plots)")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype] if args.dtype else None

    print("Loading model...")
    model, tokenizer = load_model(args.model, dtype=dtype)
    H = _get_embed_dim(model)
    num_layers = _get_num_layers(model)
    print(f"{args.model}: H={H}, L={num_layers}")

    if args.layers is None:
        layer_indices = parse_layer_indices(f"uniform:5", num_layers)
    else:
        layer_indices = parse_layer_indices(args.layers, num_layers)
    print(f"Analyzing layers: {layer_indices}")
    print(f"Divergence: EXACT (chunk_size={args.chunk_size})")
    print(f"Asymmetry: stochastic ({args.n_samples} samples)")

    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        print(f"GPU: {dev.name}, {dev.total_mem / 1e9:.1f} GB")
    print()

    # Load sentences
    if args.in_dist:
        in_dist = load_sentences_from_file(args.in_dist)
    else:
        in_dist = DEFAULT_IN_DIST
    if args.out_dist:
        out_dist = load_sentences_from_file(args.out_dist)
    else:
        out_dist = DEFAULT_OUT_DIST

    all_sentences = in_dist + out_dist
    labels = ["in"] * len(in_dist) + ["out"] * len(out_dist)
    results = []

    for i, (text, label) in enumerate(zip(all_sentences, labels)):
        print(f"[{i+1}/{len(all_sentences)}] ({label}-dist) {text[:60]}...")
        result = analyze_sentence(model, tokenizer, text, layer_indices, args.n_samples, args.chunk_size)
        result["text"] = text
        result["label"] = label
        results.append(result)
        print(f"    PPL={result['perplexity']:.2f}, "
              f"avg_div={result['avg_divergence']:.2f}, "
              f"avg_asym={result['avg_asymmetry']:.4f}")
        print()

    # Print summary
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

    for l in layer_indices:
        divs_l = [r["divergences"][l] for r in results]
        asyms_l = [r["asymmetries"][l] for r in results]
        print(f"Layer {l:2d} divergences:  {divs_l}")
        print(f"Layer {l:2d} asymmetries:  {asyms_l}")
    print()

    make_plots(results, layer_indices, args.output_dir)
    save_results_json(results, args.output_dir)


if __name__ == "__main__":
    main()
