#!/bin/bash
export CONFIG_NAME=ReasoningActivation
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

NUM_TOTAL_EPOCHS=10
INITIAL_MODEL_PATH=$1
INITIAL_DATA_PATH='../data/training_RA_train.parquet'

OUTPUT_DIR_BASE="./results/${CONFIG_NAME}"

CURRENT_MODEL_PATH=${INITIAL_MODEL_PATH}
CURRENT_DATA_PATH=${INITIAL_DATA_PATH}
export NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


# --- Training Loop ---
for (( i=1; i<=${NUM_TOTAL_EPOCHS}; i++ ))
do
    mkdir -p ${OUTPUT_DIR_BASE}/epoch_${i}
    LOG_FILE="${OUTPUT_DIR_BASE}/epoch_${i}/logs.log"
    echo "=================================" >> ${LOG_FILE}
    echo "          EPOCH ${i} / ${NUM_TOTAL_EPOCHS}" >> ${LOG_FILE}
    echo "=================================" >> ${LOG_FILE}

    echo "[$(date)] Starting training for epoch ${i}..." >> ${LOG_FILE}
    echo "--> Model Path: ${CURRENT_MODEL_PATH}" >> ${LOG_FILE}
    echo "--> Data Path: ${CURRENT_DATA_PATH}" >> ${LOG_FILE}

    deepspeed --hostfile=./scripts/hostfile \
        --num_gpus $NUM_GPUS ./scripts/train_beauty_RA.py \
        --model_name_or_path ${CURRENT_MODEL_PATH} \
        --use_lora False \
        --per_device_train_batch_size 2 \
        --num_train_epochs 1 \
        --gradient_checkpointing True \
        --bf16 True \
        --deepspeed ./scripts/ds_config_zero2.json \
        --output_dir ${OUTPUT_DIR_BASE}/epoch_${i} \
        --logging_dir ${OUTPUT_DIR_BASE}/epoch_${i} \
        --data_path ${CURRENT_DATA_PATH} \
        --logging_steps 1 \
        --eval_strategy "no" \
        --eval_on_start False \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --load_best_model_at_end False \
        --optim adamw_torch \
        --learning_rate 1e-5 \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --adam_beta1 0.9 \
        --adam_beta2 0.999 \
        --adam_epsilon 1e-8 \
        --max_grad_norm 1.0 \
        --dataloader_num_workers 4 \
        --remove_unused_columns False >> ${LOG_FILE} 2>&1 &

    TRAIN_PID=$!
    wait $TRAIN_PID

    # Check if training was successful
    if [ $? -ne 0 ]; then
        echo "[$(date)] Training for epoch ${i} FAILED. Check log for details." >> ${LOG_FILE}
        exit 1
    fi

    echo "[$(date)] Training for epoch ${i} finished successfully." >> ${LOG_FILE}

    CURRENT_MODEL_PATH=${OUTPUT_DIR_BASE}/epoch_${i}

    # --- Data Reconstruction Step ---
    if [ "$i" -eq ${NUM_TOTAL_EPOCHS} ]; then
        break
    fi

    python3 -u ./scripts/reconstruct_data_parallel.py \
        ${CURRENT_MODEL_PATH} \
        ${INITIAL_DATA_PATH} \
        ${CONFIG_NAME} \
        ${i} \
        ${NUM_GPUS} \
        2 >> ${LOG_FILE} 2>&1

    if [ $? -ne 0 ]; then
        echo "[$(date)] Data reconstruction FAILED." >> ${LOG_FILE}
        exit 1
    fi
    echo "[$(date)] Parallel data reconstruction with model ${CURRENT_MODEL_PATH} complete." >> ${LOG_FILE}
    CURRENT_DATA_PATH=${OUTPUT_DIR_BASE}/epoch_${i}/reconstructed_data.parquet
    echo "CURRENT_DATA_PATH: ${CURRENT_DATA_PATH}" >> ${LOG_FILE}

    echo "[$(date)] Data reconstruction complete." >> ${LOG_FILE}
    echo "---------------------------------\n" >> ${LOG_FILE}
done

echo "[$(date)] All ${NUM_TOTAL_EPOCHS} epochs completed." >> ${LOG_FILE}
echo "Final model is available at: ${CURRENT_MODEL_PATH}" >> ${LOG_FILE}
