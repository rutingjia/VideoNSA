#!/usr/bin/env bash
set -euo pipefail

# 一键启动本地 W&B（wandb local）并运行 10 条视频 overfit 训练。
# 用法：
#   bash scrips/run_overfit_with_local_wandb.sh
# 可选环境变量：
#   WANDB_LOCAL_PORT=8080
#   WANDB_PROJECT=videonsa-overfit
#   WANDB_RUN_NAME=qwen3vl-overfit-10samples
#   WANDB_API_KEY=<your_key_if_needed>
#   MODEL_TYPE=videonsa_qwen3
#   ATTN_IMPL=flash_attention_2
#   TRAIN_TYPE=full
#   FREEZE_VIT=false
#   FREEZE_ALIGNER=false
#   FREEZE_LLM=false

WANDB_LOCAL_PORT=${WANDB_LOCAL_PORT:-8080}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://127.0.0.1:${WANDB_LOCAL_PORT}}
export WANDB_PROJECT=${WANDB_PROJECT:-videonsa-overfit}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-qwen3vl-overfit-10samples}
export MODEL_TYPE=${MODEL_TYPE:-videonsa_qwen3}
export ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
export TRAIN_TYPE=${TRAIN_TYPE:-full}
export FREEZE_VIT=${FREEZE_VIT:-false}
export FREEZE_ALIGNER=${FREEZE_ALIGNER:-false}
export FREEZE_LLM=${FREEZE_LLM:-false}

if ! command -v wandb >/dev/null 2>&1; then
  echo "[ERROR] wandb CLI 未安装，请先安装：pip install wandb"
  exit 1
fi

USE_WANDB_LOCAL=${USE_WANDB_LOCAL:-auto}
if [[ "$USE_WANDB_LOCAL" == "auto" ]]; then
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    USE_WANDB_LOCAL=true
  else
    USE_WANDB_LOCAL=false
  fi
fi

if [[ "$USE_WANDB_LOCAL" == "true" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login --host "$WANDB_BASE_URL" "$WANDB_API_KEY"
  else
    echo "[INFO] 未设置 WANDB_API_KEY，跳过自动登录。若训练时报鉴权错误，请手动执行："
    echo "       wandb login --host $WANDB_BASE_URL"
  fi

  echo "[INFO] 启动 wandb local（端口 ${WANDB_LOCAL_PORT}）..."
  wandb local --port "$WANDB_LOCAL_PORT" >/tmp/wandb-local.log 2>&1 &
  WLOCAL_PID=$!

  cleanup() {
    if kill -0 "$WLOCAL_PID" >/dev/null 2>&1; then
      kill "$WLOCAL_PID" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT

  # 等待本地服务可访问
  for _ in $(seq 1 30); do
    if curl -fsS "$WANDB_BASE_URL" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  echo "[INFO] W&B Local 地址: $WANDB_BASE_URL"
  echo "[INFO] 现在可在浏览器打开该地址查看训练指标。"
else
  export WANDB_MODE=${WANDB_MODE:-offline}
  echo "[WARN] docker 不可用，跳过 wandb local，使用 WANDB_MODE=$WANDB_MODE 继续训练。"
fi

echo "[INFO] 开始 overfit 训练..."
bash scrips/overfit_qwen3vl_10samples.sh
