#!/usr/bin/env python
"""
Test _call_attention_sublayer and full attention analysis pipeline
on dummy (random-init) models for all 5 architectures.

Tests both forward pass and JVP (forward-mode AD) compatibility.
"""

import torch
import traceback
from transformers import AutoConfig, AutoModelForCausalLM

from lm_electrostatics.divergence_attention import (
    _call_attention_sublayer,
    analyze_attention_perlayer,
    analyze_attention_composed,
)
from lm_electrostatics.equations import (
    _get_layers, _get_embed_dim, _get_position_embeddings,
    _get_transformer_backbone,
)
from lm_electrostatics.divergence import _call_block


MODELS = {
    "GPT-2": "gpt2",
    "GPT-NeoX (Pythia)": "EleutherAI/pythia-160m",
    "Qwen2": "Qwen/Qwen2.5-0.5B",
    "Mistral": "mistralai/Mistral-7B-v0.1",
    "LLaMA": "meta-llama/Llama-3.2-1B",
}

# Fall back to locally cached models only — skip if not available
import os
os.environ["HF_HUB_OFFLINE"] = "1"

SEQ_LEN = 8
DIV_K = 4
CONS_K = 4


def test_architecture(name, model_id):
    print(f"\n{'='*60}")
    print(f"Testing: {name} ({model_id})")
    print(f"{'='*60}")

    # Step 1: Load config and create tiny random model
    print("  Loading config...", end=" ", flush=True)
    config = AutoConfig.from_pretrained(model_id)
    # Shrink to make it fast
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = 2
    if hasattr(config, "n_layer"):
        config.n_layer = 2
    config._attn_implementation = "eager"
    print("OK")

    print("  Creating random model...", end=" ", flush=True)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model = model.float()  # ensure all weights are float32 for CPU test
    model.eval()
    print("OK")

    # Step 2: Get components
    print("  Getting backbone/layers...", end=" ", flush=True)
    backbone = _get_transformer_backbone(model)
    blocks = list(_get_layers(model))
    H = _get_embed_dim(model)
    print(f"OK (H={H}, {len(blocks)} layers)")

    # Step 3: Create dummy input
    d = SEQ_LEN * H
    x0 = torch.randn(d, dtype=torch.float32)
    hidden = x0.view(1, SEQ_LEN, H)

    # Step 4: Get position embeddings
    print("  Getting position embeddings...", end=" ", flush=True)
    pos_emb = _get_position_embeddings(model, SEQ_LEN)
    print(f"{'(cos,sin) tuple' if pos_emb is not None else 'None'}")

    # Step 5: Test _call_block (full block forward)
    print("  Testing _call_block...", end=" ", flush=True)
    with torch.no_grad():
        out = _call_block(blocks[0], hidden, pos_emb)
    print(f"OK (shape={out.shape})")

    # Step 6: Test _call_attention_sublayer
    print("  Testing _call_attention_sublayer...", end=" ", flush=True)
    with torch.no_grad():
        attn_out = _call_attention_sublayer(blocks[0], hidden, pos_emb)
    print(f"OK (shape={attn_out.shape})")

    # Step 7: Test analyze_attention_perlayer (includes JVP/vmap)
    print("  Testing analyze_attention_perlayer (JVP)...", end=" ", flush=True)
    layer_indices = [0, 1]
    divs, cons = analyze_attention_perlayer(
        blocks, H, x0, layer_indices, DIV_K, CONS_K, position_embeddings=pos_emb
    )
    print(f"OK (divs={divs}, cons={cons})")

    # Step 8: Test analyze_attention_composed (includes JVP/vmap)
    print("  Testing analyze_attention_composed (JVP)...", end=" ", flush=True)
    divs_c, cons_c = analyze_attention_composed(
        blocks, H, x0, layer_indices, DIV_K, CONS_K, position_embeddings=pos_emb
    )
    print(f"OK (divs={divs_c}, cons={cons_c})")

    print(f"  *** {name}: ALL TESTS PASSED ***")
    return True


if __name__ == "__main__":
    results = {}
    for name, model_id in MODELS.items():
        try:
            results[name] = test_architecture(name, model_id)
        except Exception as e:
            print(f"\n  *** {name}: FAILED ***")
            traceback.print_exc()
            results[name] = False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    if all(results.values()):
        print("\nAll architectures passed!")
    else:
        failed = [n for n, p in results.items() if not p]
        print(f"\nFailed: {', '.join(failed)}")
