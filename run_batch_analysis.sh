#!/bin/bash
# Batch analysis across multiple models
# Usage: ./run_batch_analysis.sh

set -e

# Common args
N_SAMPLES=1000
DIV_METHOD="hutchinson"
DIV_K=50
CONS_K=50
DATASET="wikitext"
SEED=42

# Model configurations: "model_name|dtype|output_suffix|layers"
# Adjust layer sampling based on model depth
# MODELS=(
#     "gpt2|float32|gpt2|0,2,4,6,8,10,11"
#     "gpt2-xl|float32|gpt2-xl|0,6,12,18,24,30,36,42,47"
#     "openai/gpt-oss-20b|bfloat16|gpt-oss-20b|0,4,8,12,16,20,24,28,32,36,40,43"
#     "Qwen/Qwen2.5-7B|bfloat16|qwen25-7b|0,4,8,12,16,20,24,27"
#     "Qwen/Qwen2.5-14B|bfloat16|qwen25-14b|0,4,8,12,16,20,24,28,32,36,40,44,47"
#     "Qwen/Qwen2.5-7B-Instruct|bfloat16|qwen25-7b-instruct|0,4,8,12,16,20,24,27"
#     "Qwen/Qwen2.5-14B-Instruct|bfloat16|qwen25-14b-instruct|0,4,8,12,16,20,24,28,32,36,40,44,47"
#     "meta-llama/Llama-3.2-3B|bfloat16|llama32-3b|0,4,8,12,16,20,24,27"
#     "meta-llama/Llama-3.2-3B-Instruct|bfloat16|llama32-3b-instruct|0,4,8,12,16,20,24,27"
# )

MODELS=(
    "openai/gpt-oss-20b|bfloat16|gpt-oss-20b|0,4,8,12,16,20,24,28,32,36,40,43"
)

echo "Starting batch analysis for ${#MODELS[@]} model configurations"
echo "=================================================="

for config in "${MODELS[@]}"; do
    IFS='|' read -r model dtype suffix layers <<< "$config"

    echo ""
    echo "Running: $model (dtype=$dtype, layers=$layers)"
    echo "--------------------------------------------------"

    # Per-layer Jacobian analysis
    OUTPUT_DIR_PERLAYER="results/perlayer_${suffix}"
    echo "  [1/2] Per-layer Jacobian..."
    poetry run python run_analysis.py \
        --model "$model" \
        --dtype "$dtype" \
        --n-samples "$N_SAMPLES" \
        --div-method "$DIV_METHOD" \
        --div-k "$DIV_K" \
        --cons-k "$CONS_K" \
        --layers "$layers" \
        --dataset "$DATASET" \
        --output-dir "$OUTPUT_DIR_PERLAYER" \
        --seed "$SEED" \
        || echo "FAILED: $model (per-layer)"

    # Composed Jacobian analysis
    OUTPUT_DIR_COMPOSED="results/composed_${suffix}"
    echo "  [2/2] Composed Jacobian..."
    poetry run python run_composed_jacobian_analysis.py \
        --model "$model" \
        --dtype "$dtype" \
        --n-samples "$N_SAMPLES" \
        --div-method "$DIV_METHOD" \
        --div-k "$DIV_K" \
        --cons-k "$CONS_K" \
        --layers "$layers" \
        --dataset "$DATASET" \
        --output-dir "$OUTPUT_DIR_COMPOSED" \
        --seed "$SEED" \
        || echo "FAILED: $model (composed)"

    echo "Completed: $model → $OUTPUT_DIR_PERLAYER, $OUTPUT_DIR_COMPOSED"
done

# Random init baselines
echo ""
echo "=================================================="
echo "Running random-init baselines"
echo "=================================================="

RANDOM_MODELS=(
    "gpt2|float32|gpt2-random|0,2,4,6,8,10,11"
    "Qwen/Qwen2.5-14B|bfloat16|qwen25-14b-random|0,4,8,12,16,20,24,28,32,36,40,44,47"
)

for config in "${RANDOM_MODELS[@]}"; do
    IFS='|' read -r model dtype suffix layers <<< "$config"

    echo ""
    echo "Running: $model RANDOM INIT (dtype=$dtype, layers=$layers)"
    echo "--------------------------------------------------"

    # Per-layer Jacobian (random init)
    OUTPUT_DIR_PERLAYER="results/perlayer_${suffix}"
    echo "  [1/2] Per-layer Jacobian (random)..."
    poetry run python run_analysis.py \
        --model "$model" \
        --dtype "$dtype" \
        --n-samples 100 \
        --div-method "$DIV_METHOD" \
        --div-k "$DIV_K" \
        --cons-k "$CONS_K" \
        --layers "$layers" \
        --dataset "$DATASET" \
        --output-dir "$OUTPUT_DIR_PERLAYER" \
        --seed "$SEED" \
        --random-init \
        || echo "FAILED: $model (per-layer random)"

    # Composed Jacobian (random init)
    OUTPUT_DIR_COMPOSED="results/composed_${suffix}"
    echo "  [2/2] Composed Jacobian (random)..."
    poetry run python run_composed_jacobian_analysis.py \
        --model "$model" \
        --dtype "$dtype" \
        --n-samples 100 \
        --div-method "$DIV_METHOD" \
        --div-k "$DIV_K" \
        --cons-k "$CONS_K" \
        --layers "$layers" \
        --dataset "$DATASET" \
        --output-dir "$OUTPUT_DIR_COMPOSED" \
        --seed "$SEED" \
        --random-init \
        || echo "FAILED: $model (composed random)"

    echo "Completed: $model (random) → $OUTPUT_DIR_PERLAYER, $OUTPUT_DIR_COMPOSED"
done

echo ""
echo "=================================================="
echo "All runs complete!"
echo "Results saved to results/perlayer_*/"
