# FLEURS / Common Voice Evaluation

Scripts to evaluate Whisper checkpoints on a public benchmark and to compare a fine-tuned checkpoint against its base, reporting Δ WER / Δ CER. FLEURS by default, Common Voice where FLEURS coverage is missing.

## Files

| File | Purpose |
|------|---------|
| `eval_whisper_benchmark.py` | Evaluate base + FT checkpoints on FLEURS or CV, report WER / CER / Δ |
| `eval_base_only.py` | Evaluate a single Whisper checkpoint (no FT comparison) |
| `run_eval.slurm` | SLURM wrapper for `eval_whisper_benchmark.py` |
| `run_base_eval.slurm` | SLURM wrapper for `eval_base_only.py` |
| `../train/run_finetune.slurm` | SLURM wrapper for `../train/finetune_whisper_generic.py` |

The SLURM wrappers are reference templates. All paths are parameterized via env vars (`REPO_ROOT`, `CONDA_ENV`, `CONDA_HOOK`, `MODELS_NFS_DIR`, `HF_TOKEN`, `HF_HOME`); set them to match your environment.

## Requirements

```bash
pip install torch transformers datasets jiwer evaluate accelerate
export HF_TOKEN=<your_huggingface_token>     # required
export HF_HOME=/scratch/$USER/hf_cache       # recommended (avoid network home dirs)
```

## Usage

### Evaluate a fine-tuned checkpoint against its base

```bash
python eval_whisper_benchmark.py \
    --base openai/whisper-large-v2 \
    --ft   /path/to/whisper-lu_lb-v2 \
    --benchmark fleurs \
    --subset    lb_lu \
    --whisper_lang luxembourgish \
    --label lu_lb_v2 \
    --out   results/lu_lb_v2.json
```

Output is a JSON summary with WER / CER for both checkpoints and the relative Δ.

### Evaluate a single base checkpoint

```bash
python eval_base_only.py \
    --base openai/whisper-large-v2 \
    --subset lb_lu \
    --whisper_lang luxembourgish \
    --out results/lu_lb_base.json
```

### Fine-tune Whisper on a WorldSpeech config

```bash
sbatch \
  --export=ALL,CONFIG=lb_lu,LANG=luxembourgish,FLEURS=lb_lu,BASE=openai/whisper-large-v2 \
  run_finetune.slurm
```

Set `MAX_TRAIN_HOURS`, `NUM_EPOCHS`, `SEED` etc. as additional `--export` keys to override defaults.

## Decoding settings

All evaluation runs use the same fixed protocol:

- Greedy decoding (`num_beams=1`)
- Forced decoder language token via `language="<name>"` (or `forced_decoder_ids` fallback)
- Canonical Whisper `suppress_tokens` re-injected for community fine-tunes that cleared them
- `BasicTextNormalizer` (from `transformers.models.whisper.english_normalizer`) applied to both references and hypotheses
- `jiwer.wer` / `jiwer.cer` for metrics
- Empty references stripped before metric computation

## Notes

- The scripts read `HF_HOME` / `HF_DATASETS_CACHE` to locate the cache. Point these at fast local storage.
- Some fine-tuned checkpoints suffer from cleared `suppress_tokens` or repetition pathologies. The eval script auto-restores the canonical `suppress_tokens`. The optional `WHISPER_NO_REPEAT_NGRAM` env var enables n-gram blocking but should be used with caution since it can degrade results on languages with character-level tokenization.
