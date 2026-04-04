#!/usr/bin/env python
"""
Per-layer Jacobian analysis: divergence and conservativeness of individual
transformer blocks ∂f_l/∂x_{l-1}.

Unlike the composed Jacobian ∂x_l/∂x_0, per-layer Jacobians can reveal
whether individual blocks act as gradient fields (conservative) or have
rotational structure.

Usage:
  python run_analysis.py --n-samples 10           # CPU test
  python run_analysis.py --n-samples 1000 --model Qwen/Qwen2.5-14B --dtype bfloat16 \
    --div-method hutchinson --div-k 50 --cons-k 50 \
    --layers 0,4,8,12,16,20,24,28,32,36,40,44,47
"""

import argparse
import json
import math
import os
import random

import torch
import plotly.graph_objects as go
from datasets import load_dataset
from tqdm import tqdm

from lm_electrostatics.equations import (
    load_model, get_embedding,
    compute_perplexity, _get_embed_dim, _get_num_layers, _get_layers,
    _get_position_embeddings, _call_block,
)
from lm_electrostatics.divergence import analyze_layers_perlayer, exact_divergence
from lm_electrostatics.divergence_attention import analyze_attention_perlayer, analyze_attention_composed


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

def analyze_one(model, tokenizer, text, layer_indices, cons_k, div_method, div_k):
    device = next(model.parameters()).device
    ppl = compute_perplexity(model, tokenizer, text)
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(device)
    x0 = get_embedding(model, ids).reshape(-1)

    blocks = list(_get_layers(model))
    H = _get_embed_dim(model)
    S = ids.shape[1]
    pos_emb = _get_position_embeddings(model, S)

    if div_method == "hutchinson":
        divs, cons_ratios = analyze_layers_perlayer(blocks, H, x0, layer_indices, div_k, cons_k, position_embeddings=pos_emb)
        # Also compute attention-only metrics (per-layer and composed separately)
        try:
            attn_divs_perlayer, attn_cons_perlayer = analyze_attention_perlayer(blocks, H, x0, layer_indices, div_k, cons_k, position_embeddings=pos_emb)
        except Exception as attn_err:
            analyze_one._attn_perlayer_fail = getattr(analyze_one, '_attn_perlayer_fail', 0) + 1
            if analyze_one._attn_perlayer_fail <= 3:
                tqdm.write(f"  ATTN_PERLAYER_WARN: {attn_err}")
            attn_divs_perlayer = {l: float('nan') for l in layer_indices}
            attn_cons_perlayer = {l: float('nan') for l in layer_indices}

        try:
            attn_divs_composed, attn_cons_composed = analyze_attention_composed(blocks, H, x0, layer_indices, div_k, cons_k, position_embeddings=pos_emb)
        except Exception as attn_err:
            analyze_one._attn_composed_fail = getattr(analyze_one, '_attn_composed_fail', 0) + 1
            if analyze_one._attn_composed_fail <= 3:
                tqdm.write(f"  ATTN_COMPOSED_WARN: {attn_err}")
            attn_divs_composed = {l: float('nan') for l in layer_indices}
            attn_cons_composed = {l: float('nan') for l in layer_indices}
    else:
        # Exact: full single-block Jacobian per layer
        from torch.func import jacfwd
        hidden = x0.view(1, S, H)
        divs = {}
        cons_ratios = {}
        eps = 1e-8
        for l in layer_indices:
            # Forward pass to get input to this layer
            h = x0.view(1, S, H)
            for i in range(l):
                h = _call_block(blocks[i], h, pos_emb)
            h_in = h

            def block_fn(h_flat, _b=blocks[l], _pe=pos_emb):
                return _call_block(_b, h_flat.view(1, S, H), _pe).reshape(-1)

            h_flat = h_in.reshape(-1)
            J = jacfwd(block_fn)(h_flat)  # (d, d)
            divs[l] = torch.trace(J).item()
            Sym = (J + J.T) / 2
            Asym = (J - J.T) / 2
            S_sq = (Sym ** 2).sum().item()
            O_sq = (Asym ** 2).sum().item()
            cons_ratios[l] = S_sq / (S_sq + O_sq + eps)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Attention-only not implemented for exact method
        attn_divs_perlayer = {l: float('nan') for l in layer_indices}
        attn_cons_perlayer = {l: float('nan') for l in layer_indices}
        attn_divs_composed = {l: float('nan') for l in layer_indices}
        attn_cons_composed = {l: float('nan') for l in layer_indices}

    return {
        "perplexity": ppl,
        "divergences": divs,
        "conservativeness": cons_ratios,
        "attn_divergences_perlayer": attn_divs_perlayer,
        "attn_conservativeness_perlayer": attn_cons_perlayer,
        "attn_divergences_composed": attn_divs_composed,
        "attn_conservativeness_composed": attn_cons_composed,
    }


# ── plots ─────────────────────────────────────────────────────

def plot_violin_div_vs_layer(results, layer_indices, out_dir):
    """Violin: per-layer divergence distribution, blue=in / red=out."""
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
        title="Per-Layer Divergence Tr(∂f_l/∂x_{l-1})",
        xaxis_title="Layer", yaxis_title="Divergence Tr(J_l)",
        violinmode="overlay", hovermode="closest",
    )
    path = os.path.join(out_dir, "perlayer_divergence_violin.html")
    fig.write_html(path)
    print(f"Saved {path}")


def plot_conservativeness_vs_layer(results, layer_indices, out_dir):
    """Violin: per-layer conservative ratio, blue=in / red=out."""
    fig = go.Figure()
    for label, color, side in [("in", "blue", "negative"), ("out", "red", "positive")]:
        group = [r for r in results if r["label"] == label]
        for l in layer_indices:
            vals = [r["conservativeness"][l] for r in group]
            fig.add_trace(go.Violin(
                x=[f"L{l}"] * len(vals), y=vals,
                legendgroup=label, scalegroup=label,
                name=f"{label}-dist" if l == layer_indices[0] else None,
                showlegend=(l == layer_indices[0]),
                side=side, line_color=color,
                meanline_visible=True,
            ))
    fig.update_layout(
        title="Per-Layer Conservativeness ||S_l||²/||J_l||² (1=conservative, 0.5=random, 0=rotational)",
        xaxis_title="Layer", yaxis_title="||S_l||²_F / ||J_l||²_F",
        violinmode="overlay", hovermode="closest",
    )
    path = os.path.join(out_dir, "perlayer_conservativeness_violin.html")
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
        title=f"Per-Layer Divergence (Layer {last_layer}) vs Perplexity",
        xaxis_title="Perplexity", yaxis_title=f"Divergence Tr(J_{last_layer})",
        hovermode="closest",
    )
    path = os.path.join(out_dir, "perlayer_divergence_vs_perplexity.html")
    fig.write_html(path)
    print(f"Saved {path}")


def plot_attention_conservativeness(results, layer_indices, out_dir):
    """Violin: attention-only conservativeness (per-layer and composed)."""
    # Per-layer attention conservativeness
    fig = go.Figure()
    for label, color, side in [("in", "blue", "negative"), ("out", "red", "positive")]:
        group = [r for r in results if r["label"] == label]
        for l in layer_indices:
            vals = [r["attn_conservativeness_perlayer"][l] for r in group if not math.isnan(r["attn_conservativeness_perlayer"][l])]
            if vals:  # Only plot if we have non-zero values
                fig.add_trace(go.Violin(
                    x=[f"L{l}"] * len(vals), y=vals,
                    legendgroup=label, scalegroup=label,
                    name=f"{label}-dist" if l == layer_indices[0] else None,
                    showlegend=(l == layer_indices[0]),
                    side=side, line_color=color,
                    meanline_visible=True,
                ))
    fig.update_layout(
        title="Attention-Only Per-Layer Conservativeness (1=conservative, 0.5=random, 0=rotational)",
        xaxis_title="Layer", yaxis_title="||S||²_F / ||J||²_F (Attention Sublayer)",
        violinmode="overlay", hovermode="closest",
    )
    path = os.path.join(out_dir, "attention_perlayer_conservativeness_violin.html")
    fig.write_html(path)
    print(f"Saved {path}")

    # Composed attention conservativeness
    fig = go.Figure()
    for label, color, side in [("in", "blue", "negative"), ("out", "red", "positive")]:
        group = [r for r in results if r["label"] == label]
        for l in layer_indices:
            vals = [r["attn_conservativeness_composed"][l] for r in group if not math.isnan(r["attn_conservativeness_composed"][l])]
            if vals:
                fig.add_trace(go.Violin(
                    x=[f"L{l}"] * len(vals), y=vals,
                    legendgroup=label, scalegroup=label,
                    name=f"{label}-dist" if l == layer_indices[0] else None,
                    showlegend=(l == layer_indices[0]),
                    side=side, line_color=color,
                    meanline_visible=True,
                ))
    fig.update_layout(
        title="Attention-Only Composed Conservativeness (stacked attention sublayers)",
        xaxis_title="Layer", yaxis_title="||S||²_F / ||J||²_F (Composed Attention)",
        violinmode="overlay", hovermode="closest",
    )
    path = os.path.join(out_dir, "attention_composed_conservativeness_violin.html")
    fig.write_html(path)
    print(f"Saved {path}")


def plot_attention_divergence(results, layer_indices, out_dir):
    """Violin: attention-only divergence (per-layer and composed)."""
    # Per-layer attention divergence
    fig = go.Figure()
    for label, color, side in [("in", "blue", "negative"), ("out", "red", "positive")]:
        group = [r for r in results if r["label"] == label]
        for l in layer_indices:
            vals = [r["attn_divergences_perlayer"][l] for r in group if not math.isnan(r["attn_divergences_perlayer"][l])]
            if vals:
                fig.add_trace(go.Violin(
                    x=[f"L{l}"] * len(vals), y=vals,
                    legendgroup=label, scalegroup=label,
                    name=f"{label}-dist" if l == layer_indices[0] else None,
                    showlegend=(l == layer_indices[0]),
                    side=side, line_color=color,
                    meanline_visible=True,
                ))
    fig.update_layout(
        title="Attention-Only Per-Layer Divergence",
        xaxis_title="Layer", yaxis_title="Divergence Tr(J) (Attention Sublayer)",
        violinmode="overlay", hovermode="closest",
    )
    path = os.path.join(out_dir, "attention_perlayer_divergence_violin.html")
    fig.write_html(path)
    print(f"Saved {path}")

    # Composed attention divergence
    fig = go.Figure()
    for label, color, side in [("in", "blue", "negative"), ("out", "red", "positive")]:
        group = [r for r in results if r["label"] == label]
        for l in layer_indices:
            vals = [r["attn_divergences_composed"][l] for r in group if not math.isnan(r["attn_divergences_composed"][l])]
            if vals:
                fig.add_trace(go.Violin(
                    x=[f"L{l}"] * len(vals), y=vals,
                    legendgroup=label, scalegroup=label,
                    name=f"{label}-dist" if l == layer_indices[0] else None,
                    showlegend=(l == layer_indices[0]),
                    side=side, line_color=color,
                    meanline_visible=True,
                ))
    fig.update_layout(
        title="Attention-Only Composed Divergence",
        xaxis_title="Layer", yaxis_title="Divergence Tr(J) (Composed Attention)",
        violinmode="overlay", hovermode="closest",
    )
    path = os.path.join(out_dir, "attention_composed_divergence_violin.html")
    fig.write_html(path)
    print(f"Saved {path}")


# ── main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--div-method", default="hutchinson", choices=["exact", "hutchinson"],
                    help="'exact' (full Jacobian) or 'hutchinson' (stochastic)")
    ap.add_argument("--div-k", type=int, default=50, help="Hutchinson samples for divergence (only used if --div-method hutchinson)")
    ap.add_argument("--cons-k", type=int, default=50, help="Basis columns for conservative ratio (exact: full Jacobian used instead)")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    ap.add_argument("--dataset", default="wikitext", choices=["wikitext", "openwebtext"])
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-init", action="store_true", help="Use randomly initialized weights (no pretrained)")
    args = ap.parse_args()

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype] if args.dtype else None

    if args.random_init:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {args.model} with RANDOM weights...")
        config = AutoConfig.from_pretrained(args.model)
        config._attn_implementation = "eager"
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager", dtype=dtype)
        if dtype is not None:
            model = model.to(dtype)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        for param in model.parameters():
            param.requires_grad = False
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        print(f"Loading {args.model}...")
        model, tokenizer = load_model(args.model, dtype=dtype)
    num_layers = _get_num_layers(model)
    if args.layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]
    print(f"Layers: {layer_indices}")
    if args.div_method == "exact":
        print(f"Per-layer Jacobian: EXACT (full Jacobian)")
    else:
        print(f"Per-layer Jacobian: HUTCHINSON (div_k={args.div_k}, cons_k={args.cons_k})")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_properties(0).name}")

    # ── data ──
    print(f"\nSampling {args.n_samples} sentences from {args.dataset}...")
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
            r = analyze_one(model, tokenizer, text, layer_indices, args.cons_k, args.div_method, args.div_k)
        except Exception as e:
            tqdm.write(f"  SKIP [{idx+1}]: {e}")
            continue
        r["text"] = text
        r["label"] = label
        r["divergences"] = {str(k): v for k, v in r["divergences"].items()}
        r["conservativeness"] = {str(k): v for k, v in r["conservativeness"].items()}
        r["attn_divergences_perlayer"] = {str(k): v for k, v in r["attn_divergences_perlayer"].items()}
        r["attn_conservativeness_perlayer"] = {str(k): v for k, v in r["attn_conservativeness_perlayer"].items()}
        r["attn_divergences_composed"] = {str(k): v for k, v in r["attn_divergences_composed"].items()}
        r["attn_conservativeness_composed"] = {str(k): v for k, v in r["attn_conservativeness_composed"].items()}
        results.append(r)
        avg_cons = sum(float(v) for v in r["conservativeness"].values()) / len(r["conservativeness"])
        pbar.set_postfix(label=label, ppl=f"{r['perplexity']:.0f}", cons=f"{avg_cons:.3f}")

        if (idx + 1) % max(1, total // 20) == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f)

    # Report attention analysis failures
    perlayer_fails = getattr(analyze_one, '_attn_perlayer_fail', 0)
    composed_fails = getattr(analyze_one, '_attn_composed_fail', 0)
    if perlayer_fails or composed_fails:
        print(f"\nAttention analysis failures: perlayer={perlayer_fails}/{len(results)}, composed={composed_fails}/{len(results)}")

    # ── save final ──
    def _nan_to_none(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_nan_to_none(v) for v in obj]
        return obj

    final_path = os.path.join(args.output_dir, "results.json")
    with open(final_path, "w") as f:
        json.dump(_nan_to_none(results), f, indent=2)
    print(f"\nSaved {len(results)} results to {final_path}")

    # Convert str keys back to int for plotting (None -> NaN for filtering)
    def _none_to_nan(v):
        return float('nan') if v is None else v

    for r in results:
        r["divergences"] = {int(k): v for k, v in r["divergences"].items()}
        r["conservativeness"] = {int(k): v for k, v in r["conservativeness"].items()}
        r["attn_divergences_perlayer"] = {int(k): _none_to_nan(v) for k, v in r["attn_divergences_perlayer"].items()}
        r["attn_conservativeness_perlayer"] = {int(k): _none_to_nan(v) for k, v in r["attn_conservativeness_perlayer"].items()}
        r["attn_divergences_composed"] = {int(k): _none_to_nan(v) for k, v in r["attn_divergences_composed"].items()}
        r["attn_conservativeness_composed"] = {int(k): _none_to_nan(v) for k, v in r["attn_conservativeness_composed"].items()}

    # ── plots ──
    plot_violin_div_vs_layer(results, layer_indices, args.output_dir)
    plot_conservativeness_vs_layer(results, layer_indices, args.output_dir)
    plot_div_vs_ppl(results, layer_indices, args.output_dir)
    plot_attention_conservativeness(results, layer_indices, args.output_dir)
    plot_attention_divergence(results, layer_indices, args.output_dir)

    print(f"Done in {pbar.format_dict['elapsed']:.0f}s")


if __name__ == "__main__":
    main()
