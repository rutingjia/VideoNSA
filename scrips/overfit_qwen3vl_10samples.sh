#!/usr/bin/env bash
set -euo pipefail

# 单卡、10条数据过拟合脚本（Qwen3-VL + MixNSA/flash_attention_2链路）
# 用法:
#   bash scrips/overfit_qwen3vl_10samples.sh
# 可选环境变量:
#   MODEL=Qwen/Qwen3-VL-8B-Instruct
#   DATASET=/data/vjuicefs_ai_camera_album_ql/public_data/rutingjia/qwen3vl_video_sft/train.jsonl
#   OUT_DIR=output/qwen3vl_overfit_10
#   MAX_STEPS=200
#   LR=1e-5

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-12}

MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
DATASET=${DATASET:-/data/vjuicefs_ai_camera_album_ql/public_data/rutingjia/qwen3vl_video_sft/train.jsonl}
OUT_DIR=${OUT_DIR:-output/qwen3vl_overfit_10}
MAX_STEPS=${MAX_STEPS:-200}
LR=${LR:-1e-5}

swift sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --train_type lora \
  --attn_impl flash_attention_2 \
  --torch_dtype bfloat16 \
  --num_train_epochs 20 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate "$LR" \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit true \
  --split_dataset_ratio 0 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 20 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --max_length 2048 \
  --warmup_ratio 0.0 \
  --dataloader_num_workers 2 \
  --output_dir "$OUT_DIR"
