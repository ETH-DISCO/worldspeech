#!/bin/bash
#SBATCH --job-name=ws_fleval
#SBATCH --output=logs/%x_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
set -euo pipefail
export PYTHONUNBUFFERED=1

# Required env: LABEL BASE FT_NFS BENCHMARK SUBSET WLANG
# Optional: CV_VER (default 17), BS (default 8), REPO_ROOT, CONDA_ENV, CONDA_HOOK
: "${LABEL:?}" "${BASE:?}" "${FT_NFS:?}" "${BENCHMARK:?}" "${SUBSET:?}" "${WLANG:?}"
: "${HF_TOKEN:?HF_TOKEN env var must be set (HuggingFace access token)}"
CV_VER="${CV_VER:-17}"
BS="${BS:-8}"
REPO_ROOT="${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
CONDA_ENV="${CONDA_ENV:-worldspeech}"

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

# Stage FT model from NFS to local /scratch
LOCAL_FT="/scratch/$USER/fleval_models/$LABEL"
mkdir -p "$(dirname "$LOCAL_FT")"
echo "[stage] rsync $FT_NFS -> $LOCAL_FT"
rsync -a --info=stats1 --exclude 'checkpoint-*' "$FT_NFS/" "$LOCAL_FT/"

OUT_DIR="$REPO_ROOT/eval/results"
mkdir -p "$OUT_DIR"
OUT_JSON="$OUT_DIR/${LABEL}.json"

cd "$REPO_ROOT/eval"
python eval_whisper_benchmark.py \
    --base "$BASE" \
    --ft "$LOCAL_FT" \
    --benchmark "$BENCHMARK" \
    --subset "$SUBSET" \
    --whisper_lang "$WLANG" \
    --cv_version "$CV_VER" \
    --label "$LABEL" \
    --out "$OUT_JSON" \
    --batch_size "$BS"
