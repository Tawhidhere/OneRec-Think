#!/usr/bin/env bash

# Usage: ./eval_parallel_8gpu_cot.sh [EXTRA_ARGS]

echo "[INFO] Starting 8-GPU Parallel CoT Evaluation..."

MERGED_MODEL_PATH="../train/results/ReasoningActivation/epoch_2/checkpoint-125"
ADDITIONAL_LORA_PATH=""
TEST_PARQUET="../data/training_RA_test.parquet"
GLOBAL_TRIE_FILE="./exact_trie.pkl"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_0930/parallel_cot_eval_${TS}"
mkdir -p "$LOG_DIR"

echo "[INFO] Log directory: $LOG_DIR"
echo "[INFO] Started at: $(date)"

TOTAL_SAMPLES=22363
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / 8))
BATCH_SIZE=4
NUM_THINKING_SAMPLES=5
NUM_BEAMS_PER_SAMPLE=10
THINK_TOKENS=128
SID_TOKENS=8

echo "[INFO] Precomputing exact trie tree for CoT evaluation..."
if [ ! -f "$GLOBAL_TRIE_FILE" ]; then
    python3 precompute_global_trie.py \
        --test_parquet_file "$TEST_PARQUET" \
        --model_path "$MERGED_MODEL_PATH" \
        --output_file "$GLOBAL_TRIE_FILE"
    echo "[OK] Exact trie precomputed: $GLOBAL_TRIE_FILE"
else
    echo "[OK] Exact trie already exists: $GLOBAL_TRIE_FILE"
fi

echo "[INFO] 8-GPU Parallel CoT Configuration:"
echo "  Merged model: $MERGED_MODEL_PATH"
echo "  Additional LoRA: $ADDITIONAL_LORA_PATH"
echo "  Exact trie: $GLOBAL_TRIE_FILE"
echo "  Total samples: $TOTAL_SAMPLES"
echo "  Samples per GPU: $SAMPLES_PER_GPU"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Thinking samples: $NUM_THINKING_SAMPLES"
echo "  Beams per sample: $NUM_BEAMS_PER_SAMPLE"
echo "  Total beams: $((NUM_THINKING_SAMPLES * NUM_BEAMS_PER_SAMPLE))"
echo "  Think tokens: $THINK_TOKENS"
echo "  SID tokens: $SID_TOKENS"
echo "  Print generations: ENABLED"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

pids=()
for gpu_id in {0..7}; do
    offset=$((gpu_id * SAMPLES_PER_GPU))
    log_file="${LOG_DIR}/gpu_${gpu_id}.log"
    
    echo "[INFO] Starting CoT GPU $gpu_id: samples $offset-$((offset + SAMPLES_PER_GPU - 1))"

    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 -u test_model_hitrate_cot.py \
        --merged_model_path "${MERGED_MODEL_PATH}" \
        --additional_lora_path "${ADDITIONAL_LORA_PATH}" \
        --test_parquet_file "${TEST_PARQUET}" \
        --global_trie_file "${GLOBAL_TRIE_FILE}" \
        --test_batch_size ${BATCH_SIZE} \
        --num_thinking_samples ${NUM_THINKING_SAMPLES} \
        --num_beams_per_sample ${NUM_BEAMS_PER_SAMPLE} \
        --metrics "hit@1,hit@5,hit@10,ndcg@5,ndcg@10" \
        --think_max_tokens ${THINK_TOKENS} \
        --sid_max_tokens ${SID_TOKENS} \
        --think_temperature 1.5 \
        --think_top_p 0.95 \
        --sid_temperature 0.6 \
        --sid_top_p 1 \
        --print_generations \
        --sample_num ${SAMPLES_PER_GPU} \
        --sample_offset ${offset} \
        --gpu_id ${gpu_id} \
        --log_file "$log_file" \
        "$@" > "$log_file" 2>&1 &

    pids+=($!)
    sleep 3
done

echo ""
echo "[INFO] All 8 CoT processes started:"
for i in {0..7}; do
    echo "  GPU $i: PID ${pids[$i]} -> ${LOG_DIR}/gpu_${i}.log"
done

echo ""
echo "[INFO] CoT Monitor commands:"
echo "  # View GPU status"
echo "  nvidia-smi"
echo "  # View specific GPU output (recommended)"
echo "  tail -f ${LOG_DIR}/gpu_0.log"
echo "  # View all GPU outputs"
echo "  tail -f ${LOG_DIR}/gpu_*.log"
echo "  # View process status"
echo "  ps aux | grep test_model_hitrate_cot"

echo ""
echo "[INFO] Waiting for all CoT processes to complete..."
wait "${pids[@]}"

# Merge results and save to summary log
python3 -c "
import re
import os
from glob import glob
from datetime import datetime

log_dir = '${LOG_DIR}'
hit_metrics = ['hit@1', 'hit@5', 'hit@10', 'ndcg@5', 'ndcg@10']
total_metrics = {m: 0.0 for m in hit_metrics}
total_samples = 0

summary_log = f'{log_dir}/summary_cot_results.log'
with open(summary_log, 'w', encoding='utf-8') as f:
    f.write('[INFO] 8-GPU Parallel CoT Evaluation Summary\\n')
    f.write('=' * 80 + '\\n')
    f.write(f'Evaluation Time: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\\n')
    f.write(f'Log Directory: {log_dir}\\n')
    f.write('\\n')

    f.write('[CONFIG] Model & Training Configuration:\\n')
    f.write('-' * 50 + '\\n')
    f.write(f'Merged Model Path: ${MERGED_MODEL_PATH}\\n')
    if '${ADDITIONAL_LORA_PATH}':
        f.write(f'Additional LoRA: ${ADDITIONAL_LORA_PATH}\\n')
    else:
        f.write('Additional LoRA: None\\n')
    f.write(f'Test Data: ${TEST_PARQUET}\\n')
    f.write(f'Global Trie File: ${GLOBAL_TRIE_FILE}\\n')
    f.write('\\n')

    f.write('[PARAMS] CoT Reasoning Parameters:\\n')
    f.write('-' * 50 + '\\n')
    f.write(f'Total Test Samples: ${TOTAL_SAMPLES}\\n')
    f.write(f'Samples per GPU: ${SAMPLES_PER_GPU}\\n')
    f.write(f'Batch Size per GPU: ${BATCH_SIZE}\\n')
    f.write(f'Thinking Samples per Input: ${NUM_THINKING_SAMPLES}\\n')
    f.write(f'Beams per Thinking Sample: ${NUM_BEAMS_PER_SAMPLE}\\n')
    f.write(f'Total Initial Candidates: $((${NUM_THINKING_SAMPLES} * ${NUM_BEAMS_PER_SAMPLE}))\\n')
    f.write(f'Final Unique Top-K: 10\\n')
    f.write(f'Think Max Tokens: ${THINK_TOKENS}\\n')
    f.write(f'SID Max Tokens: ${SID_TOKENS}\\n')
    f.write('\\n')

    f.write('[RESULTS] Individual GPU Results:\\n')
    f.write('-' * 50 + '\\n')

    found_gpus = 0
    gpu_details = []

    for gpu_id in range(8):
        log_file = f'{log_dir}/gpu_{gpu_id}.log'
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as gpu_f:
                    content = gpu_f.read()

                final_marker = '🎯 Final CoT Hit Rate Results:'
                marker_pos = content.rfind(final_marker)

                if marker_pos != -1:
                    results_section = content[marker_pos:]
                    gpu_metrics = {}

                    for line in results_section.split('\\n'):
                        line = line.strip()
                        for metric in hit_metrics:
                            pattern = rf'.*{re.escape(metric)}:\\s*([0-9]*\\.?[0-9]+)'
                            match = re.search(pattern, line)
                            if match:
                                value = float(match.group(1))
                                gpu_metrics[metric] = value
                                total_metrics[metric] += value
                                break

                    sample_patterns = [
                        r'Total samples:\\s*(\\d+)',
                        r'Test data size:\\s*(\\d+)',
                        r'samples.*?(\\d+)'
                    ]
                    samples = 0
                    for pattern in sample_patterns:
                        sample_match = re.search(pattern, content, re.IGNORECASE)
                        if sample_match:
                            samples = int(sample_match.group(1))
                            break

                    if samples == 0:
                        samples = ${SAMPLES_PER_GPU}

                    if gpu_metrics:
                        found_gpus += 1
                        total_samples += samples
                        gpu_details.append((gpu_id, samples, gpu_metrics))

                        f.write(f'[OK] GPU {gpu_id}: {samples} samples\\n')
                        for metric in hit_metrics:
                            value = gpu_metrics.get(metric, 0.0)
                            f.write(f'    {metric}: {value:.4f}\\n')
                    else:
                        f.write(f'[WARN] GPU {gpu_id}: Results format not recognized\\n')
                else:
                    f.write(f'[ERROR] GPU {gpu_id}: Final results marker not found\\n')

            except Exception as e:
                f.write(f'[ERROR] GPU {gpu_id}: Error reading log - {str(e)}\\n')
        else:
            f.write(f'[ERROR] GPU {gpu_id}: Log file not found\\n')

    f.write('\\n')

    if found_gpus > 0:
        avg_metrics = {m: total_metrics[m] / found_gpus for m in hit_metrics}

        f.write('[FINAL] FINAL AVERAGED CoT RESULTS:\\n')
        f.write('=' * 60 + '\\n')
        for metric in hit_metrics:
            value = avg_metrics[metric]
            if metric.startswith('hit'):
                f.write(f'{metric:>10}: {value:.4f} ({value*100:.2f}%)\\n')
            else:
                f.write(f'{metric:>10}: {value:.4f}\\n')
        f.write('=' * 60 + '\\n')

        f.write('\\n[SUMMARY] Evaluation Summary:\\n')
        f.write('-' * 40 + '\\n')
        f.write(f'Total Samples Processed: {total_samples:,}\\n')
        f.write(f'Successful GPUs: {found_gpus}/8\\n')
        f.write(f'Average Samples per GPU: {total_samples//found_gpus if found_gpus > 0 else 0}\\n')

        f.write('\\n📈 Performance Interpretation:\\n')
        f.write('-' * 40 + '\\n')
        hit1 = avg_metrics.get('hit@1', 0) * 100
        hit5 = avg_metrics.get('hit@5', 0) * 100
        hit10 = avg_metrics.get('hit@10', 0) * 100
        ndcg5 = avg_metrics.get('ndcg@5', 0)
        ndcg10 = avg_metrics.get('ndcg@10', 0)

        if hit1 > 0:
            f.write(f'• Top-1 Hit Rate: {hit1:.2f}%\\n')
        if hit5 > 0:
            f.write(f'• Top-5 Hit Rate: {hit5:.2f}%\\n')
        if hit10 > 0:
            f.write(f'• Top-10 Hit Rate: {hit10:.2f}%\\n')
        if ndcg5 > 0:
            f.write(f'• NDCG@5 Score: {ndcg5:.4f}\\n')
        if ndcg10 > 0:
            f.write(f'• NDCG@10 Score: {ndcg10:.4f}\\n')

        total_beams = ${NUM_THINKING_SAMPLES} * ${NUM_BEAMS_PER_SAMPLE}
        f.write(f'\\n⚡ Inference Efficiency:\\n')
        f.write(f'• Thinking + SID generation per sample\\n')
        f.write(f'• {total_beams} candidate generations per input\\n')
        f.write(f'• CoT reasoning enabled\\n')

        f.write('\\n✅ CoT-Enhanced Evaluation Completed Successfully!\\n')

    else:
        f.write('❌ No valid CoT results found from any GPU!\\n')
        f.write('🔧 Troubleshooting tips:\\n')
        f.write('1. Check if all GPU processes completed successfully\\n')
        f.write('2. Verify log files contain \"🎯 Final CoT Hit Rate Results:\" marker\\n')
        f.write('3. Check for OOM or other runtime errors in individual GPU logs\\n')

print(f'[INFO] CoT Summary Report: {summary_log}')
" >> "${LOG_DIR}/summary_cot_results.log"

echo ""
echo "🎯 CoT Evaluation Summary:"
echo "  📁 Results directory: ${LOG_DIR}"
echo "  📊 Summary file: ${LOG_DIR}/summary_cot_results.log"
echo "  🔍 View summary: cat ${LOG_DIR}/summary_cot_results.log"
echo ""
echo "✅ 8-GPU Parallel CoT Evaluation completed!"