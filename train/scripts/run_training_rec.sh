#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

MODEL_DIR="../basemodel/merged_beauty_model_1-1"
TRAIN_DATA="../data/training_prediction_sid_data_train.parquet"
VAL_DATA="../data/training_prediction_sid_data_val.parquet"

USE_LORA=false

DEEPSPEED_CMD=(
    deepspeed
    --hostfile=./scripts/hostfile
    --num_gpus 8
    ./scripts/train_beauty_sid_rec.py
    --model_name_or_path "${MODEL_DIR}"
    --train_data_path "${TRAIN_DATA}"
    --val_data_path "${VAL_DATA}"
)

if [ "${USE_LORA}" = "true" ]; then
    DEEPSPEED_CMD+=(
        --use_lora True
        --lora_r 128
        --lora_alpha 128
        --lora_dropout 0.05
        --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
else
    DEEPSPEED_CMD+=(--use_lora False)
fi

DEEPSPEED_CMD+=(
    --per_device_train_batch_size 2
    --num_train_epochs 6
    --gradient_checkpointing True
    --bf16 True
    --deepspeed ./scripts/ds_config_zero2.json
    --output_dir ./results/beauty_sid_rec
    --logging_dir ./logs/beauty_sid_rec
    --logging_steps 10
    --eval_strategy epoch
    --eval_on_start False
    --save_strategy epoch
    --save_total_limit 10
    --metric_for_best_model eval_loss
    --greater_is_better False
    --load_best_model_at_end True
    --optim adamw_torch
    --learning_rate 1e-5
    --warmup_ratio 0.1
    --weight_decay 0.01
    --adam_beta1 0.9
    --adam_beta2 0.999
    --adam_epsilon 1e-8
    --max_grad_norm 1.0
    --dataloader_num_workers 4
    --remove_unused_columns False
)

nohup "${DEEPSPEED_CMD[@]}" >> beauty_sid_rec.log 2>&1 &
