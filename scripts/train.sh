#!/usr/bin/env bash
set -euo pipefail

# Run this script from anywhere; paths below are resolved from the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -------------------------
# Train config
# -------------------------
DATA_PATH="${DATA_PATH:-../data/dateset_v1}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/aide_train}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"

MODEL="${MODEL:-AIDE}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-20}"
BLR="${BLR:-1e-4}"
UPDATE_FREQ="${UPDATE_FREQ:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Leave these empty to use the default model-zoo fallbacks in models/AIDE.py.
RESNET_PATH="${RESNET_PATH:-}"
CONVNEXT_PATH="${CONVNEXT_PATH:-}"
RESUME="${RESUME:-}"

DEVICE="${DEVICE:-cuda}"
USE_AMP="${USE_AMP:-False}"
DISABLE_EVAL="${DISABLE_EVAL:-False}"

# -------------------------
# Distributed config
# -------------------------
GPU_NUM="${GPU_NUM:-1}"
WORLD_SIZE="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29512}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

PY_ARGS=(
    main_finetune.py
    --model "${MODEL}"
    --data_path "${DATA_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --log_dir "${LOG_DIR}"
    --device "${DEVICE}"
    --batch_size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --blr "${BLR}"
    --update_freq "${UPDATE_FREQ}"
    --num_workers "${NUM_WORKERS}"
    --use_amp "${USE_AMP}"
    --disable_eval "${DISABLE_EVAL}"
)

if [[ -n "${EVAL_DATA_PATH}" ]]; then
    PY_ARGS+=(--eval_data_path "${EVAL_DATA_PATH}")
fi

if [[ -n "${RESNET_PATH}" ]]; then
    PY_ARGS+=(--resnet_path "${RESNET_PATH}")
fi

if [[ -n "${CONVNEXT_PATH}" ]]; then
    PY_ARGS+=(--convnext_path "${CONVNEXT_PATH}")
fi

if [[ -n "${RESUME}" ]]; then
    PY_ARGS+=(--resume "${RESUME}")
fi

# Extra CLI args override or extend the defaults above.
PY_ARGS+=("$@")

python -m torch.distributed.launch \
    --nproc_per_node "${GPU_NUM}" \
    --nnodes "${WORLD_SIZE}" \
    --node_rank "${RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${PY_ARGS[@]}"
