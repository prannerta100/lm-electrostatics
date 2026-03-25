#!/usr/bin/env python
"""
Large-scale divergence + asymmetry analysis on OpenWebText.

Usage:
  python run_analysis.py --n-samples 10           # CPU test
  python run_analysis.py --n-samples 100000       # GPU production run
"""

import argparse
import json
import os
import random
import time

import torch
import plotly.graph_objects as go
from datasets import load_dataset
from tqdm import tqdm

from lm_electrostatics.equations import (
    load_model, get_embedding, get_layer_output_fn,
    compute_perplexity, _get_embed_dim, _get_num_layers, _get_layers,
)
from lm_electrostatics.divergence import exact_divergence, estimate_divergence, estimate_asymmetry, analyze_layers_hutchinson


# ── data ──────────────────────────────────────────────────────

def sample_sentences(n, dataset="wikitext", seed=42):
    """Sample n usable sentences from the specified dataset."""
    rng = __import__("random").Random(seed)
    if dataset == "openwebtext":
        ds = load_dataset("openwebtext", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        sentences = []
        for ex in ds:
            for line in ex["text"].split("\n"):
                line = line.strip()
                if 30 < len(line) < 200 and len(line.split()) >= 6:
                    sentences.append(line)
                    break
            if len(sentences) >= n:
                break
        print(f"Sampled {len(sentences)} in-distribution sentences from OpenWebText")
        return sentences[:n]
    else:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        candidates = [t for t in ds["text"] if 30 < len(t.strip()) < 200 and len(t.split()) >= 6]
        rng.shuffle(candidates)
        sentences = candidates[:n]
        print(f"Sampled {len(sentences)} in-distribution sentences from wikitext-103")
        return sentences


def make_ood(sentences, seed=42):
    """Create OOD by replacing 50% of words with random words from pool + a swap."""
    rng = random.Random(seed)
    pool = [w for s in sentences for w in s.split()]
    ood = []
    for s in sentences:
        words = s.split()
        words = [rng.choice(pool) if rng.random() < 0.5 else w for w in words]
        if len(words) > 3:
            i, j = rng.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        ood.append(" ".join(words))
    return ood


# ── analysis ──────────────────────────────────────────────────

def analyze_one(model, tokenizer, text, layer_indices, asym_k, div_method, div_k):
    device = next(model.parameters()).device
    ppl = compute_perplexity(model, tokenizer, text)
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(device)
    x0 = get_embedding(model, ids).reshape(-1)

    if div_method == "hutchinson":
        blocks = list(_get_layers(model))
        H = _get_embed_dim(model)
        divs, asyms = analyze_layers_hutchinson(blocks, H, x0, layer_indices, div_k, asym_k)
    else:
        divs = {}
        asyms = {}
        for l in layer_indices:
            fn = get_layer_output_fn(model, l)
            divs[l] = exact_divergence(fn, x0, chunk_size=0)
            asyms[l] = estimate_asymmetry(fn, x0, n_samples=asym_k)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {"perplexity": ppl, "divergences": divs, "asymmetries": asyms}


# ── plots ─────────────────────────────────────────────────────

def plot_violin_div_vs_layer(results, layer_indices, out_dir):
    """Violin: divergence distribution per layer, blue=in / red=out."""
    fig = go.Figure()
    for label, color, side in [("in", "blue", "negative"), ("out", "red", "positive")]:
        group = [r for r in results if r["label"] == label]
        for l in layer_indices:
            vals = [r["divergences"][l] for r in group]
            fig.add_trace(go.Violin(
                x=[f"L{l}"] * len(vals), y=vals,
                legendgroup=label, scalegroup=label,
                name=f"{label}-dist" if l == layer_indices[0] else None,
                showlegend=(l == layer_indices[0]),
                side=side, line_color=color,
                meanline_visible=True,
            ))
    fig.update_layout(
        title="Exact Divergence Tr(J) per Layer",
        xaxis_title="Layer", yaxis_title="Divergence Tr(J)",
        violinmode="overlay", hovermode="closest",
    )
    path = os.path.join(out_dir, "divergence_vs_layer_violin.html")
    fig.write_html(path)
    print(f"Saved {path}")


def plot_asymmetry_histogram(results, layer_indices, out_dir):
    """Histogram: Frobenius asymmetry (avg across layers) for in vs out."""
    fig = go.Figure()
    for label, color in [("in", "blue"), ("out", "red")]:
        group = [r for r in results if r["label"] == label]
        avg_asyms = [sum(r["asymmetries"].values()) / len(r["asymmetries"]) for r in group]
        fig.add_trace(go.Histogram(
            x=avg_asyms, name=f"{label}-dist",
            marker_color=color, opacity=0.6,
            nbinsx=50,
        ))
    fig.update_layout(
        title="Jacobian Asymmetry Distribution (stochastic, avg over layers)",
        xaxis_title="Asymmetry (0=conservative, 2=antisymmetric)",
        yaxis_title="Count",
        barmode="overlay", hovermode="closest",
    )
    path = os.path.join(out_dir, "asymmetry_histogram.html")
    fig.write_html(path)
    print(f"Saved {path}")


def plot_violin_asym_vs_layer(results, layer_indices, out_dir):
    """Violin: asymmetry distribution per layer, blue=in / red=out."""
    fig = go.Figure()
    for label, color, side in [("in", "blue", "negative"), ("out", "red", "positive")]:
        group = [r for r in results if r["label"] == label]
        for l in layer_indices:
            vals = [r["asymmetries"][l] for r in group]
            fig.add_trace(go.Violin(
                x=[f"L{l}"] * len(vals), y=vals,
                legendgroup=label, scalegroup=label,
                name=f"{label}-dist" if l == layer_indices[0] else None,
                showlegend=(l == layer_indices[0]),
                side=side, line_color=color,
                meanline_visible=True,
            ))
    fig.update_layout(
        title="Jacobian Asymmetry per Layer",
        xaxis_title="Layer", yaxis_title="Asymmetry (0=conservative, 2=antisymmetric)",
        violinmode="overlay", hovermode="closest",
    )
    path = os.path.join(out_dir, "asymmetry_vs_layer_violin.html")
    fig.write_html(path)
    print(f"Saved {path}")


def plot_div_vs_ppl(results, layer_indices, out_dir):
    """Scatter: last-layer divergence vs perplexity, blue=in / red=out."""
    last_layer = max(layer_indices)
    fig = go.Figure()
    for label, color in [("in", "blue"), ("out", "red")]:
        group = [r for r in results if r["label"] == label]
        fig.add_trace(go.Scatter(
            x=[r["perplexity"] for r in group],
            y=[r["divergences"][last_layer] for r in group],
            mode="markers", marker=dict(size=5, color=color, opacity=0.5),
            name=f"{label}-dist",
            hovertext=[r["text"][:80] for r in group], hoverinfo="text+x+y",
        ))
    fig.update_layout(
        title=f"Divergence (Layer {last_layer}) vs Perplexity",
        xaxis_title="Perplexity", yaxis_title=f"Divergence Tr(J) [Layer {last_layer}]",
        hovermode="closest",
    )
    path = os.path.join(out_dir, "divergence_vs_perplexity.html")
    fig.write_html(path)
    print(f"Saved {path}")


# ── main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--div-method", default="exact", choices=["exact", "hutchinson"],
                    help="Divergence method: 'exact' (full Jacobian) or 'hutchinson' (stochastic)")
    ap.add_argument("--div-k", type=int, default=50, help="Hutchinson samples for divergence (only used if --div-method hutchinson)")
    ap.add_argument("--asym-k", type=int, default=20, help="Samples for stochastic asymmetry (default: 20)")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    ap.add_argument("--dataset", default="wikitext", choices=["wikitext", "openwebtext"])
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype] if args.dtype else None

    print(f"Loading {args.model}...")
    model, tokenizer = load_model(args.model, dtype=dtype)
    num_layers = _get_num_layers(model)
    if args.layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]
    print(f"Layers: {layer_indices}")
    if args.div_method == "exact":
        print(f"Divergence: EXACT (full Jacobian trace)")
    else:
        print(f"Divergence: HUTCHINSON (K={args.div_k} jvp samples)")
    print(f"Asymmetry: stochastic (K={args.asym_k} jvp+vjp pairs)")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_properties(0).name}")

    # ── data ──
    print(f"\nSampling {args.n_samples} sentences from OpenWebText...")
    in_sents = sample_sentences(args.n_samples, dataset=args.dataset, seed=args.seed)
    ood_sents = make_ood(in_sents, seed=args.seed)

    # ── run ──
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    total = len(in_sents) + len(ood_sents)
    all_sents = [(s, "in") for s in in_sents] + [(s, "out") for s in ood_sents]

    checkpoint_path = os.path.join(args.output_dir, "results_checkpoint.json")
    pbar = tqdm(all_sents, desc="Analyzing", unit="sent")

    for idx, (text, label) in enumerate(pbar):
        try:
            r = analyze_one(model, tokenizer, text, layer_indices, args.asym_k, args.div_method, args.div_k)
        except Exception as e:
            tqdm.write(f"  SKIP [{idx+1}]: {e}")
            continue
        r["text"] = text
        r["label"] = label
        r["divergences"] = {str(k): v for k, v in r["divergences"].items()}
        r["asymmetries"] = {str(k): v for k, v in r["asymmetries"].items()}
        results.append(r)
        avg_asym = sum(float(v) for v in r["asymmetries"].values()) / len(r["asymmetries"])
        pbar.set_postfix(label=label, ppl=f"{r['perplexity']:.0f}", asym=f"{avg_asym:.3f}")

        if (idx + 1) % max(1, total // 20) == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f)

    # ── save final ──
    final_path = os.path.join(args.output_dir, "results.json")
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {final_path}")

    # Convert str keys back to int for plotting
    for r in results:
        r["divergences"] = {int(k): v for k, v in r["divergences"].items()}
        r["asymmetries"] = {int(k): v for k, v in r["asymmetries"].items()}

    # ── plots ──
    plot_violin_div_vs_layer(results, layer_indices, args.output_dir)
    plot_violin_asym_vs_layer(results, layer_indices, args.output_dir)
    plot_asymmetry_histogram(results, layer_indices, args.output_dir)
    plot_div_vs_ppl(results, layer_indices, args.output_dir)

    print(f"Done in {pbar.format_dict['elapsed']:.0f}s")


if __name__ == "__main__":
    main()
