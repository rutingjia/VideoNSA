#!/usr/bin/env bash
set -euo pipefail

# 单卡、10条数据过拟合脚本（Qwen3-VL + MixNSA/flash_attention_2链路）
# 用法:
#   bash scrips/overfit_qwen3vl_10samples.sh
# 可选环境变量:
#   MODEL=Qwen/Qwen3-VL-2B-Instruct
#   DATASET=/data/vjuicefs_ai_camera_album_ql/public_data/rutingjia/qwen3vl_video_sft/train.jsonl
#   OUT_DIR=output/qwen3vl_overfit_10
#   MAX_STEPS=2000
#   MODEL_TYPE=videonsa_qwen3
#   ATTN_IMPL=flash_attention_2
#   LR=1e-5
#   WANDB_BASE_URL=http://127.0.0.1:8080
#   WANDB_PROJECT=videonsa-overfit
#   WANDB_RUN_NAME=qwen3vl-overfit-10samples

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-16384}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-4}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://127.0.0.1:8080}
export WANDB_PROJECT=${WANDB_PROJECT:-videonsa-overfit}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-qwen3vl-overfit-10samples}

MODEL=${MODEL:-Qwen/Qwen3-VL-2B-Instruct}
DATASET=${DATASET:-/data/vjuicefs_ai_camera_album_ql/public_data/rutingjia/qwen3vl_video_sft/train.jsonl}
OUT_DIR=${OUT_DIR:-output/qwen3vl_overfit_10}
MAX_STEPS=${MAX_STEPS:-2000}
MODEL_TYPE=${MODEL_TYPE:-videonsa_qwen3}
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}
LR=${LR:-1e-5}

swift sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --model_type "$MODEL_TYPE" \
  --attn_impl "$ATTN_IMPL" \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate "$LR" \
  --freeze_vit true \
  --split_dataset_ratio 0 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 20 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --report_to wandb \
  --run_name "$WANDB_RUN_NAME" \
  --max_length 2048 \
  --warmup_ratio 0.0 \
  --dataloader_num_workers 0 \
  --output_dir "$OUT_DIR"
