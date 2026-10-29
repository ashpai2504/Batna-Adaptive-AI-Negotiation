#!/usr/bin/env bash
# LoRA fine-tuning of a small open model on the rejection-sampled Dynamic-BATNA SFT corpus (mlx-lm, Apple Silicon).
# Usage: bash dynbatna/lora.sh [train|serve|serve-base]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}
ADAPTERS=${ADAPTERS:-results/dynbatna/lora/adapters}
DATA=${DATA:-data/dynbatna/sft}
ITERS=${ITERS:-600}
PORT=${PORT:-8080}

case "${1:-train}" in
  train)
    mkdir -p "$ADAPTERS"
    .venv/bin/python -m mlx_lm lora --model "$MODEL" --train --data "$DATA" --adapter-path "$ADAPTERS" \
      --iters "$ITERS" --batch-size 1 --num-layers 8 --learning-rate 1e-4 --max-seq-length 3072 \
      --grad-checkpoint --mask-prompt --steps-per-report 20 --steps-per-eval 100 --val-batches 25 --seed 0 \
      2>&1 | tee results/dynbatna/lora/train.log
    ;;
  serve)
    .venv/bin/python -m mlx_lm server --model "$MODEL" --adapter-path "$ADAPTERS" --port "$PORT" --max-tokens 512
    ;;
  serve-base)
    .venv/bin/python -m mlx_lm server --model "$MODEL" --port "$PORT" --max-tokens 512
    ;;
esac
