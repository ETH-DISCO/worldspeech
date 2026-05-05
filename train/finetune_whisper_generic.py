#!/usr/bin/env python3
"""
Generic Whisper large-v3 fine-tune on WS2 data.
Usage: python finetune_whisper_generic.py --config lu_lb --lang lb --text_col gt_transcript --fleurs lb_lu
"""

import argparse, os, json, hashlib, random, numpy as np, torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline as hf_pipeline,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN env var must be set (HuggingFace access token)")
MODEL = "openai/whisper-large-v3"
SCRATCH = f"/scratch/{os.environ.get('USER')}"
CACHE_DIR = f"{SCRATCH}/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_DIR}/transformers"
os.makedirs(CACHE_DIR, exist_ok=True)

normalizer = BasicTextNormalizer()

@dataclass
class DataCollator:
    processor: Any
    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def eval_pipe(pipe, dataset, text_col, name, max_n=500):
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    preds, refs = [], []
    n = min(len(dataset), max_n)
    for i in range(n):
        try:
            row = dataset[i]
            out = pipe(row["audio"]["array"])
            preds.append(normalizer(out["text"]))
            refs.append(normalizer(row[text_col]))
        except:
            continue
    pairs = [(p, r) for p, r in zip(preds, refs) if r.strip()]
    if not pairs:
        return {"wer": -1, "cer": -1}
    p, r = zip(*pairs)
    wer = wer_metric.compute(predictions=p, references=r)
    cer = cer_metric.compute(predictions=p, references=r)
    print(f"  {name}: WER={wer:.4f}, CER={cer:.4f} ({len(p)} samples)")
    return {"wer": wer, "cer": cer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="WS2 config e.g. lu_lb")
    parser.add_argument("--repo", default="AAdonis/WS2", help="HF dataset repo (default AAdonis/WS2, can be AAdonis/WorldSpeech)")
    parser.add_argument("--config2", default=None, help="Second config to merge e.g. xk_sq")
    parser.add_argument("--lang", required=True, help="Whisper lang code e.g. lb")
    parser.add_argument("--text_col", default="gt_transcript", help="GT text column name")
    parser.add_argument("--fleurs", default=None, help="FLEURS config e.g. lb_lu")
    parser.add_argument("--model", default=None, help="Override model e.g. openai/whisper-large-v3-turbo")
    parser.add_argument("--extra_configs", default=None, help="Comma-separated extra configs to merge e.g. tn_ar,ma_ar,iq_ar")
    parser.add_argument("--max_train_hours", type=float, default=None,
                        help="Subsample CER-filtered train set to first N hours under a locked-seed shuffle. "
                             "If None, use the full filtered train set (legacy behaviour).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Locked seed for subsample shuffle and Trainer.")
    parser.add_argument("--out_json", default=None,
                        help="Write per-cut JSON {hours, lang, ft_wer, ft_cer, base_wer, base_cer, "
                             "trainer_config_hash, ...} to this path.")
    parser.add_argument("--base_wer", type=float, default=None,
                        help="Pre-recorded base (zero-shot) FLEURS WER. Used to fill out-json when SKIP_R0=1.")
    parser.add_argument("--base_cer", type=float, default=None,
                        help="Pre-recorded base (zero-shot) FLEURS CER. Used to fill out-json when SKIP_R0=1.")
    parser.add_argument("--num_train_epochs", type=float, default=1.0,
                        help="Trainer num_train_epochs. Override only for smoke tests.")
    args = parser.parse_args()

    # Lock random seeds for the shuffle. Trainer also uses its own seed (training_args.seed below).
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_id = args.model or MODEL
    # Allow SLURM wrapper to override the per-run output dir (so concurrent
    # ablation cuts of the same config don't collide on /scratch).
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR_OVERRIDE") or f"{SCRATCH}/models/whisper-{args.config}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lang_arg = None if args.lang.lower() == "none" else args.lang
    use_bf16 = "v3" in model_id
    train_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Config: {args.config}, Lang: {lang_arg or 'NONE'}, Text col: {args.text_col}, bf16={use_bf16}")
    print(f"Loading {model_id}...")
    processor = WhisperProcessor.from_pretrained(model_id, language=lang_arg, task="transcribe", cache_dir=CACHE_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=CACHE_DIR, torch_dtype=torch.float32)
    # Clear forced decoder ids and suppress tokens — model learns from labels
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    # Patch generate to use language+task at eval/inference time
    if lang_arg:
        from functools import partial as _partial
        model.generate = _partial(model.generate, language=lang_arg, task="transcribe", use_cache=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ─── R0 Baseline ──────────────────────────────────────────────────────
    skip_r0 = os.environ.get("SKIP_R0", "0") == "1"
    def _load_repo(repo, config):
        # For WorldSpeech (legacy repo), load directly from parquets to bypass script schema issues
        if "WorldSpeech" in repo:
            base = f"hf://datasets/{repo}/data/{config}"
            return load_dataset("parquet",
                                data_files={"train": f"{base}/train-*.parquet",
                                            "test":  f"{base}/test-*.parquet"},
                                cache_dir=CACHE_DIR,
                                token=HF_TOKEN)
        try:
            return load_dataset(repo, config, token=HF_TOKEN, cache_dir=CACHE_DIR)
        except Exception:
            return load_dataset(repo, config, token=HF_TOKEN, cache_dir=CACHE_DIR,
                                download_mode="force_redownload")

    if skip_r0:
        print("\n========== ROUND 0: SKIPPED (SKIP_R0=1) ==========")
        ws2 = _load_repo(args.repo, args.config)
        ws2_audio = ws2.cast_column("audio", Audio(sampling_rate=16000))
        r0_ws2 = {"wer": -1, "cer": -1}
        r0_fleurs = {"wer": -1, "cer": -1}
    else:
        print("\n========== ROUND 0: BASELINE ==========")
        pipe = hf_pipeline("automatic-speech-recognition", model=model,
            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
            device=device, torch_dtype=train_dtype,
            generate_kwargs={"language": lang_arg, "task": "transcribe"} if lang_arg else {"task": "transcribe"}, chunk_length_s=30)

        ws2 = _load_repo(args.repo, args.config)
        ws2_audio = ws2.cast_column("audio", Audio(sampling_rate=16000))
        r0_ws2 = eval_pipe(pipe, ws2_audio["test"], args.text_col, "WS2")

        r0_fleurs = {"wer": -1, "cer": -1}
        if args.fleurs:
            try:
                fleurs = load_dataset("google/fleurs", args.fleurs, split="test", cache_dir=CACHE_DIR, trust_remote_code=True)
                fleurs = fleurs.cast_column("audio", Audio(sampling_rate=16000))
                r0_fleurs = eval_pipe(pipe, fleurs, "transcription", "FLEURS")
            except Exception as e:
                print(f"  FLEURS failed: {e}")
        del pipe

    # ─── R1 Fine-tune ─────────────────────────────────────────────────────
    print("\n========== ROUND 1: FINE-TUNE ==========")

    cer_thr = float(os.environ.get("CER_THRESHOLD", "0.3"))
    def fast_cer_filter(ds, threshold=cer_thr):
        """HF datasets .filter() with audio columns reads full rows (incl audio bytes)
        from parquet per row -> hours on 100K+ segs. Reading only the `cer` column
        and using .select() is ~1000x faster."""
        cer_col = ds["cer"]
        keep = [i for i, c in enumerate(cer_col) if c is not None and c < threshold]
        return ds.select(keep)

    ws2["train"] = fast_cer_filter(ws2["train"])
    print(f"Train (CER<{cer_thr}): {len(ws2['train'])}")

    # Optionally merge second config
    if args.config2:
        ws2_2 = _load_repo(args.repo, args.config2)
        ws2_2["train"] = fast_cer_filter(ws2_2["train"])
        print(f"Config2 {args.config2} train (CER<{cer_thr}): {len(ws2_2['train'])}")
        ws2["train"] = concatenate_datasets([ws2["train"], ws2_2["train"]])
        print(f"Combined train: {len(ws2['train'])}")

    # Optionally merge extra configs (comma-separated)
    if args.extra_configs:
        for ec in args.extra_configs.split(","):
            ec = ec.strip()
            if not ec:
                continue
            try:
                ws2_e = _load_repo(args.repo, ec)
                ws2_e["train"] = fast_cer_filter(ws2_e["train"])
                print(f"Extra {ec} train (CER<{cer_thr}): {len(ws2_e['train'])}")
                ws2["train"] = concatenate_datasets([ws2["train"], ws2_e["train"]])
            except Exception as e:
                print(f"Extra {ec} failed: {e}")
        print(f"Combined train: {len(ws2['train'])}")

    # ─── Cumulative-hours subsample (CER ablation) ───────────────────────
    # Only the 50/100/200/... cuts use this. The legacy code path (no flag)
    # is unchanged so the existing 1919736 reference is bit-identical w.r.t.
    # data ordering.
    cumulative_hours = None
    n_segments_kept = len(ws2["train"])
    if args.max_train_hours is not None:
        target_secs = float(args.max_train_hours) * 3600.0
        # Read only `duration` + `segment_id` columns -> O(rows), no audio bytes.
        durs = ws2["train"]["duration"]
        seg_ids = ws2["train"]["segment_id"]
        n = len(durs)
        rng = random.Random(args.seed)
        perm = list(range(n))
        rng.shuffle(perm)
        cum = 0.0
        keep_perm = []
        for i in perm:
            d = durs[i]
            if d is None:
                continue
            cum += float(d)
            keep_perm.append(i)
            if cum >= target_secs:
                break
        cumulative_hours = cum / 3600.0
        n_segments_kept = len(keep_perm)
        print(f"\nSubsample: target {args.max_train_hours}h, seed {args.seed} "
              f"-> kept {n_segments_kept} segs, cumulative {cumulative_hours:.3f}h")
        # Persist segment list for reproducibility.
        if args.out_json:
            seg_path = Path(args.out_json).with_name(Path(args.out_json).stem + "_segments.json")
            seg_path.parent.mkdir(parents=True, exist_ok=True)
            seg_path.write_text(json.dumps({
                "config": args.config,
                "lang": args.lang,
                "seed": args.seed,
                "max_train_hours": args.max_train_hours,
                "cer_threshold": cer_thr,
                "n_segments": n_segments_kept,
                "cumulative_hours": round(cumulative_hours, 4),
                "segment_ids": [seg_ids[i] for i in keep_perm],
                "indices_in_filtered_train": keep_perm,
            }, indent=2))
            print(f"Saved segment list -> {seg_path}")
        ws2["train"] = ws2["train"].select(keep_perm)
        print(f"Train after subsample: {len(ws2['train'])}")

    ws2_audio = ws2.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        labels = processor.tokenizer(text=normalizer(batch[args.text_col])).input_ids
        if len(labels) > 440:
            batch["labels"] = None
        else:
            batch["labels"] = labels
        return batch

    print("Preprocessing train...")
    train_ds = ws2_audio["train"].map(prepare, remove_columns=ws2_audio["train"].column_names, num_proc=1)
    # fast filter: drop rows where prepare() set labels=None (token len > 440).
    # plain .filter() reads full rows (incl 1MB mel spectrograms) at ~12/s -> hours.
    labels_col = train_ds["labels"]
    keep_idx = [i for i, l in enumerate(labels_col) if l is not None]
    train_ds = train_ds.select(keep_idx)
    print(f"Train: {len(train_ds)}")

    test_sub = ws2_audio["test"].select(range(min(300, len(ws2_audio["test"]))))
    test_ds = test_sub.map(prepare, remove_columns=test_sub.column_names, num_proc=1)
    test_ds = test_ds.filter(lambda x: x["labels"] is not None)
    print(f"Test subset: {len(test_ds)}")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = [normalizer(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
        label_str = [normalizer(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
        pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        p, l = zip(*pairs)
        return {"wer": wer_metric.compute(predictions=p, references=l), "cer": cer_metric.compute(predictions=p, references=l)}

    # batch sizing: env-overridable so we can drop to 1 on small GPUs (32GB V100)
    pdt = int(os.environ.get("PER_DEVICE_BATCH", "2"))
    ga = int(os.environ.get("GRAD_ACCUM", "16"))
    total_steps = (len(train_ds) // (pdt * ga)) * args.num_train_epochs
    warmup_steps = min(500, max(50, total_steps // 10))
    # For small datasets, use epoch-based eval/save so model is always checkpointed
    use_grad_ckpt = bool(int(os.environ.get("GRAD_CKPT", "0")))
    if use_grad_ckpt:
        model.config.use_cache = False
    eval_steps = max(100, min(1000, total_steps // 4))
    print(f"total_steps={total_steps}, warmup_steps={warmup_steps}, eval_steps={eval_steps}")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=pdt, per_device_eval_batch_size=pdt,
        gradient_accumulation_steps=ga, learning_rate=float(os.environ.get("LR", "1e-5")),
        warmup_steps=warmup_steps, num_train_epochs=args.num_train_epochs,
        fp16=(not use_bf16), bf16=use_bf16,
        bf16_full_eval=use_bf16, fp16_full_eval=(not use_bf16),
        gradient_checkpointing=use_grad_ckpt,
        eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps, save_total_limit=2,
        logging_steps=max(1, total_steps // 20), load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=225,
        dataloader_num_workers=4, report_to="none",
        seed=args.seed, data_seed=args.seed,
    )

    # Recipe hash: SHA256 of TrainingArguments + generation_config, with
    # cut-specific fields stripped so every cut for any language yields the
    # same hash. This proves all 28 runs share one recipe.
    ta_dict = training_args.to_dict()
    for k in ("output_dir", "logging_dir", "run_name", "hub_model_id"):
        ta_dict.pop(k, None)
    gen_cfg_dict = model.generation_config.to_dict() if model.generation_config else {}
    # forced_decoder_ids is language-specific; strip from hash so cross-lang hashes match.
    gen_cfg_dict.pop("forced_decoder_ids", None)
    recipe_payload = json.dumps(
        {"training_args": ta_dict, "generation_config": gen_cfg_dict},
        sort_keys=True, default=str,
    )
    trainer_config_hash = hashlib.sha256(recipe_payload.encode()).hexdigest()
    print(f"trainer_config_hash: {trainer_config_hash}")

    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=test_ds,
        data_collator=DataCollator(processor=processor),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    # Re-set forced_decoder_ids after training (Trainer may have cleared them)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang_arg, task="transcribe") if lang_arg else None
    model.generation_config.forced_decoder_ids = model.config.forced_decoder_ids

    # ─── R1 Eval ──────────────────────────────────────────────────────────
    # SKIP_R1=1 -> bypass the in-script R1 eval. Used by the ablation pipeline
    # which calls eval_whisper_benchmark.py post-FT for authoritative numbers
    # (full FLEURS test split, batched generate, jiwer). The hf_pipeline-based
    # eval_pipe here is unbatched and adds 30-50 min/cut for no benefit.
    skip_r1 = os.environ.get("SKIP_R1", "0") == "1"
    if skip_r1:
        print("\n========== ROUND 1: SKIPPED (SKIP_R1=1) ==========")
        r1_ws2 = {"wer": -1, "cer": -1}
        r1_fleurs = {"wer": -1, "cer": -1}
    else:
        print("\n========== ROUND 1: EVALUATION ==========")
        pipe = hf_pipeline("automatic-speech-recognition", model=model,
            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
            device=model.device, torch_dtype=torch.float16,
            generate_kwargs={"language": lang_arg, "task": "transcribe"} if lang_arg else {"task": "transcribe"}, chunk_length_s=30)

        r1_ws2 = eval_pipe(pipe, ws2_audio["test"], args.text_col, "WS2")

        r1_fleurs = {"wer": -1, "cer": -1}
        if args.fleurs:
            try:
                r1_fleurs = eval_pipe(pipe, fleurs, "transcription", "FLEURS")
            except Exception as e:
                print(f"  FLEURS failed: {e}")

    print(f"\n{'='*60}")
    print(f"  R0 WS2:    WER={r0_ws2['wer']:.4f}, CER={r0_ws2['cer']:.4f}")
    print(f"  R1 WS2:    WER={r1_ws2['wer']:.4f}, CER={r1_ws2['cer']:.4f}")
    if args.fleurs:
        print(f"  R0 FLEURS: WER={r0_fleurs['wer']:.4f}, CER={r0_fleurs['cer']:.4f}")
        print(f"  R1 FLEURS: WER={r1_fleurs['wer']:.4f}, CER={r1_fleurs['cer']:.4f}")
    print(f"{'='*60}")

    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump({"config": args.config, "r0_ws2": r0_ws2, "r1_ws2": r1_ws2,
                    "r0_fleurs": r0_fleurs, "r1_fleurs": r1_fleurs,
                    "trainer_config_hash": trainer_config_hash,
                    "max_train_hours": args.max_train_hours,
                    "cumulative_hours": cumulative_hours,
                    "n_segments_kept": n_segments_kept,
                    "seed": args.seed}, f, indent=2)

    # Per-cut ablation JSON (--out-json) -- canonical schema for the figure renderer.
    if args.out_json:
        # Prefer pre-recorded base numbers (faster: skip R0 per cut).
        base_wer = args.base_wer if args.base_wer is not None else r0_fleurs.get("wer", -1)
        base_cer = args.base_cer if args.base_cer is not None else r0_fleurs.get("cer", -1)
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({
            "lang": args.config,
            "fleurs_subset": args.fleurs,
            "max_train_hours": args.max_train_hours,
            "cumulative_hours": cumulative_hours,
            "n_segments_kept": n_segments_kept,
            "base_wer": base_wer,
            "base_cer": base_cer,
            "ft_wer": r1_fleurs.get("wer", -1),
            "ft_cer": r1_fleurs.get("cer", -1),
            "n_eval": 500,  # eval_pipe max_n
            "trainer_config_hash": trainer_config_hash,
            "seed": args.seed,
            "model": model_id,
        }, indent=2))
        print(f"Wrote ablation JSON -> {out_json}")


if __name__ == "__main__":
    main()
