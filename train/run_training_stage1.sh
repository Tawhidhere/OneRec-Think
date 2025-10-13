#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_DIR="../basemodel/Qwen3-1-7B-expand"
TRAIN_DATA="../data/training_align_data_train.parquet"
VAL_DATA="../data/training_align_data_val.parquet"

nohup deepspeed --hostfile=./scripts/hostfile \
    --num_gpus 8 ./scripts/train_beauty_align.py \
    --model_dir "${MODEL_DIR}" \
    --train_data_path "${TRAIN_DATA}" \
    --val_data_path "${VAL_DATA}" \
    --per_device_train_batch_size 8 \
    --num_train_epochs 15 \
    --gradient_checkpointing True \
    --bf16 True \
    --deepspeed ./scripts/ds_config_zero2.json \
    --output_dir ./results/beauty_align \
    --logging_dir ./logs/beauty_sid_align \
    --logging_steps 10 \
    --eval_strategy epoch \
    --eval_on_start False \
    --save_strategy epoch \
    --save_total_limit 15 \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --load_best_model_at_end True \
    --optim adamw_torch \
    --learning_rate 1e-4 \
    --warmup_ratio 0.0 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 4 \
    --remove_unused_columns False >> beauty_align.log 2>&1 &
