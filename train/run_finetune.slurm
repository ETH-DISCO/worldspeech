#!/bin/bash
#SBATCH --job-name=ws_ft
#SBATCH --output=logs/%x_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Required env: CONFIG LANG FLEURS BASE
# Optional: CONFIG2, EXTRA_CONFIGS, TEXT_COL, MODEL_SUFFIX, REPO_ROOT, CONDA_ENV, CONDA_HOOK, MODELS_NFS_DIR
: "${CONFIG:?}" "${LANG:?}" "${FLEURS:?}" "${BASE:?}"
: "${HF_TOKEN:?HF_TOKEN env var must be set (HuggingFace access token)}"
TEXT_COL="${TEXT_COL:-gt_transcript}"
REPO_ROOT="${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_ENV="${CONDA_ENV:-worldspeech}"
MODELS_NFS_DIR="${MODELS_NFS_DIR:-$REPO_ROOT/models_out}"

# Auto-detect suffix from base model name if not set
if [ -z "${MODEL_SUFFIX:-}" ]; then
    case "$BASE" in
        *v3-turbo*) MODEL_SUFFIX="v3t" ;;
        *large-v3*)  MODEL_SUFFIX="v3"  ;;
        *)           MODEL_SUFFIX="v2"  ;;
    esac
fi

if [ -n "${CONDA_HOOK:-}" ]; then
    eval "$("$CONDA_HOOK" shell.bash hook)"
else
    eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV"

export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME"

NFS_OUT="$MODELS_NFS_DIR/whisper-${CONFIG}-${MODEL_SUFFIX}"
LOCAL_OUT="/scratch/$USER/models/whisper-${CONFIG}-${MODEL_SUFFIX}"
mkdir -p "$LOCAL_OUT" "$NFS_OUT"
export OUTPUT_DIR_OVERRIDE="$LOCAL_OUT"

echo "[FT] config=${CONFIG} lang=${LANG} fleurs=${FLEURS} base=${BASE}"
echo "[FT] local out=${LOCAL_OUT}, nfs out=${NFS_OUT}"

cd "$REPO_ROOT/train"

EXTRA_ARGS=()
if [ -n "${REPO:-}" ]; then EXTRA_ARGS+=("--repo" "$REPO"); fi
if [ -n "${CONFIG2:-}" ]; then EXTRA_ARGS+=("--config2" "$CONFIG2"); fi
if [ -n "${EXTRA_CONFIGS:-}" ]; then EXTRA_ARGS+=("--extra_configs" "$EXTRA_CONFIGS"); fi
if [ -n "${MAX_TRAIN_HOURS:-}" ]; then EXTRA_ARGS+=("--max_train_hours" "$MAX_TRAIN_HOURS"); fi
if [ -n "${SEED:-}" ]; then EXTRA_ARGS+=("--seed" "$SEED"); fi
if [ -n "${OUT_JSON:-}" ]; then EXTRA_ARGS+=("--out_json" "$OUT_JSON"); fi
if [ -n "${BASE_WER:-}" ]; then EXTRA_ARGS+=("--base_wer" "$BASE_WER"); fi
if [ -n "${BASE_CER:-}" ]; then EXTRA_ARGS+=("--base_cer" "$BASE_CER"); fi
if [ -n "${NUM_EPOCHS:-}" ]; then EXTRA_ARGS+=("--num_train_epochs" "$NUM_EPOCHS"); fi

export SKIP_R0="${SKIP_R0:-0}"

python finetune_whisper_generic.py \
    --config "$CONFIG" \
    --lang "$LANG" \
    --text_col "$TEXT_COL" \
    --fleurs "$FLEURS" \
    --model "$BASE" \
    "${EXTRA_ARGS[@]}" || echo "[FT] Python exited with error, rsync anyway"

echo "[FT] rsync ${LOCAL_OUT} -> ${NFS_OUT}"
rsync -a --info=stats1 --exclude 'checkpoint-*' "$LOCAL_OUT/" "$NFS_OUT/"

if [ -d "$LOCAL_OUT" ]; then
    cp -r "$LOCAL_OUT/results.json" "$NFS_OUT/" 2>/dev/null || true
fi
echo "[FT] done"
