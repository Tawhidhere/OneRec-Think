#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REC_SCRIPT="${SCRIPT_DIR}/scripts/run_training_rec.sh"
RA_SCRIPT="${SCRIPT_DIR}/scripts/run_training_RA.sh"
REC_RESULTS_DIR="${SCRIPT_DIR}/results/beauty_sid_rec"
REC_PROCESS_PATTERN="train_beauty_sid_rec.py"

if [[ ! -x "${REC_SCRIPT}" ]]; then
    echo "Error: ${REC_SCRIPT} not found or not executable." >&2
    exit 1
fi

if [[ ! -x "${RA_SCRIPT}" ]]; then
    echo "Error: ${RA_SCRIPT} not found or not executable." >&2
    exit 1
fi

echo "=== Stage 1: Starting recommendation training (run_training_rec.sh) ==="
bash "${REC_SCRIPT}"

echo "Waiting for recommendation training process to complete..."
sleep 10
while pgrep -f "${REC_PROCESS_PATTERN}" > /dev/null; do
    sleep 60
done
echo "Recommendation training finished."

if [[ ! -d "${REC_RESULTS_DIR}" ]]; then
    echo "Error: results directory ${REC_RESULTS_DIR} not found." >&2
    exit 1
fi

last_checkpoint=$(ls -d "${REC_RESULTS_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ -z "${last_checkpoint}" ]]; then
    echo "Error: no checkpoint directories found under ${REC_RESULTS_DIR}." >&2
    exit 1
fi

echo "Identified final checkpoint for RA stage: ${last_checkpoint}"

echo "=== Stage 2: Starting reasoning activation training (run_training_RA.sh) ==="
bash "${RA_SCRIPT}" "${last_checkpoint}"

echo "Pipeline completed successfully."
