#!/bin/bash
#SBATCH --job-name=ws_baseval
#SBATCH --output=logs/%x_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
export PYTHONUNBUFFERED=1

# Required env: LABEL BASE SUBSET WLANG
# Optional: REPO_ROOT, CONDA_ENV, CONDA_HOOK
: "${LABEL:?}" "${BASE:?}" "${SUBSET:?}" "${WLANG:?}"
: "${HF_TOKEN:?HF_TOKEN env var must be set (HuggingFace access token)}"
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

OUT="$REPO_ROOT/eval/results/${LABEL}_base_only.json"
mkdir -p "$(dirname "$OUT")"
cd "$REPO_ROOT/eval"
python eval_base_only.py --base "$BASE" --subset "$SUBSET" --whisper_lang "$WLANG" --out "$OUT"
