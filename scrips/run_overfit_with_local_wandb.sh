#!/usr/bin/env bash
set -euo pipefail

# 使用 W&B offline 模式运行 10 条视频 overfit 训练（不依赖 docker / wandb local）。
# 用法：
#   bash scrips/run_overfit_with_local_wandb.sh
# 可选环境变量：
#   WANDB_PROJECT=videonsa-overfit
#   WANDB_RUN_NAME=qwen3vl-overfit-10samples
#   MODEL_TYPE=videonsa_qwen3
#   ATTN_IMPL=flash_attention_2
#   TRAIN_TYPE=full
#   FREEZE_VIT=false
#   FREEZE_ALIGNER=false
#   FREEZE_LLM=false

export WANDB_PROJECT=${WANDB_PROJECT:-videonsa-overfit}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-qwen3vl-overfit-10samples}
export MODEL_TYPE=${MODEL_TYPE:-videonsa_qwen3}
export ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
export TRAIN_TYPE=${TRAIN_TYPE:-full}
export FREEZE_VIT=${FREEZE_VIT:-false}
export FREEZE_ALIGNER=${FREEZE_ALIGNER:-false}
export FREEZE_LLM=${FREEZE_LLM:-false}

export WANDB_MODE=offline
echo "[INFO] 使用 W&B offline 模式: WANDB_MODE=$WANDB_MODE"

echo "[INFO] 开始 overfit 训练..."
bash scrips/overfit_qwen3vl_10samples.sh
