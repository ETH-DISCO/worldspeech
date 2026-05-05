#!/usr/bin/env python3
"""
Eval a base Whisper model + a fine-tuned Whisper model on a public benchmark
(FLEURS or Common Voice). Reports WER/CER for both, plus delta.

Args:
  --base PATH-OR-HFID    base / starting Whisper checkpoint
  --ft PATH              fine-tuned Whisper checkpoint dir (already rsynced to local /scratch)
  --benchmark fleurs|cv  which benchmark
  --subset CODE          fleurs subset (e.g. sw_ke) or cv lang (e.g. tn)
  --whisper_lang LANG    forced-decoder language token (e.g. swahili / sw / nepali)
  --cv_version N         CV version, default 17
  --label LABEL          short label for output JSON
  --out PATH             output json path
"""
import argparse, json, os, sys, time
import torch
import jiwer
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def get_normalizer():
    return BasicTextNormalizer()


_CANON_SUPPRESS = None
def _canonical_suppress_tokens():
    global _CANON_SUPPRESS
    if _CANON_SUPPRESS is None:
        from transformers import GenerationConfig
        gc = GenerationConfig.from_pretrained("openai/whisper-large-v3")
        _CANON_SUPPRESS = list(gc.suppress_tokens or [])
    return _CANON_SUPPRESS


def _patch_generation_config(model):
    # FT pathologies seen in WS2 community-FT'd Whispers:
    # (1) suppress_tokens cleared (Dragneel/whisper-medium-nepali-openslr) → hallucinated
    #     boilerplate. Restore canonical list only when missing. Always safe.
    # (2) EOS calibration lost from training on long parliamentary segments
    #     (Chillarmo/whisper-large-v3-turbo-armenian on top of which whisper-hy was FT'd):
    #     model transcribes correctly then loops on a single word. Was patched here
    #     with no_repeat_ngram_size=4, but on tests of normal mm_my Whisper-large-v2
    #     FT (job 1920759) this *catastrophically* worsens results: WER goes from
    #     0.26 (clean greedy) to 0.77 (with 4-gram block). Because Burmese tokens
    #     are character-level and many real n-grams are legitimate repeats. So the
    #     n-gram block is now opt-in via WHISPER_NO_REPEAT_NGRAM env var; default off.
    gc = model.generation_config
    if not gc.suppress_tokens:
        gc.suppress_tokens = _canonical_suppress_tokens()
        print(f"  [patch] injected canonical suppress_tokens (n={len(gc.suppress_tokens)})", flush=True)
    nr = os.environ.get("WHISPER_NO_REPEAT_NGRAM")
    if nr:
        gc.no_repeat_ngram_size = int(nr)
        print(f"  [patch] no_repeat_ngram_size={gc.no_repeat_ngram_size} (opt-in via env)", flush=True)


def transcribe(model, processor, audio_arrays, sr, whisper_lang, batch_size=8, device="cuda"):
    _patch_generation_config(model)
    out = []
    forced = None
    for i in range(0, len(audio_arrays), batch_size):
        batch = audio_arrays[i:i + batch_size]
        inputs = processor(batch, sampling_rate=sr, return_tensors="pt", padding=True)
        feats = inputs.input_features.to(device, dtype=torch.float16)
        with torch.no_grad():
            try:
                ids = model.generate(
                    feats,
                    language=whisper_lang,
                    task="transcribe",
                    max_new_tokens=440,
                    num_beams=1,
                )
            except (TypeError, ValueError):
                forced = forced or processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")
                ids = model.generate(
                    feats,
                    forced_decoder_ids=forced,
                    max_new_tokens=440,
                    num_beams=1,
                )
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        out.extend(texts)
        if (i // batch_size) % 10 == 0:
            print(f"    transcribed {min(i + batch_size, len(audio_arrays))}/{len(audio_arrays)}", flush=True)
    return out


def _safe_load(repo, subset, split, cache_dir, **kwargs):
    """Load with cache-corruption recovery. HF cache races between concurrent
    jobs leave broken arrow files; on FileNotFoundError, force re-download."""
    try:
        return load_dataset(repo, subset, split=split, cache_dir=cache_dir,
                            trust_remote_code=True, **kwargs)
    except FileNotFoundError as e:
        print(f"  [cache] corrupted cache, forcing redownload: {e}", flush=True)
        return load_dataset(repo, subset, split=split, cache_dir=cache_dir,
                            trust_remote_code=True,
                            download_mode="force_redownload", **kwargs)


def load_benchmark(benchmark, subset, cv_version, cache_dir):
    if benchmark == "fleurs":
        ds = _safe_load("google/fleurs", subset, "test", cache_dir)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ref_col = "transcription"
    elif benchmark == "cv":
        repo = f"mozilla-foundation/common_voice_{cv_version}_0"
        ds = _safe_load(repo, subset, "test", cache_dir,
                        token=os.environ.get("HF_TOKEN"))
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ref_col = "sentence"
    else:
        raise ValueError(f"unknown benchmark {benchmark}")
    audios = [r["audio"]["array"] for r in ds]
    refs = [r[ref_col] for r in ds]
    return audios, refs


def eval_model(model_id_or_path, audios, refs, whisper_lang, normalizer, label, batch_size=8):
    print(f"\n=== eval [{label}] {model_id_or_path} ===", flush=True)
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(model_id_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id_or_path, torch_dtype=torch.float16
    ).to("cuda").eval()
    hyps = transcribe(model, processor, audios, 16000, whisper_lang, batch_size=batch_size)
    norm_hyps = [normalizer(h) for h in hyps]
    norm_refs = [normalizer(r) for r in refs]
    keep = [(h, r) for h, r in zip(norm_hyps, norm_refs) if r.strip()]
    norm_hyps = [h for h, _ in keep]
    norm_refs = [r for _, r in keep]
    wer = jiwer.wer(norm_refs, norm_hyps)
    cer = jiwer.cer(norm_refs, norm_hyps)
    dt = time.time() - t0
    print(f"  [{label}] n={len(norm_refs)} WER={wer:.4f} CER={cer:.4f}  ({dt:.0f}s)", flush=True)
    print(f"  sample hyp: {norm_hyps[0][:140]!r}", flush=True)
    print(f"  sample ref: {norm_refs[0][:140]!r}", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"n": len(norm_refs), "wer": wer, "cer": cer,
            "sample_hyps": norm_hyps[:3], "sample_refs": norm_refs[:3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--benchmark", required=True, choices=["fleurs", "cv"])
    ap.add_argument("--subset", required=True)
    ap.add_argument("--whisper_lang", required=True)
    ap.add_argument("--cv_version", type=int, default=17)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--skip_base", action="store_true",
                    help="Skip base eval (for ablation cuts where base is constant per language).")
    ap.add_argument("--base_wer", type=float, default=None,
                    help="Pre-recorded base WER to fill summary when --skip_base.")
    ap.add_argument("--base_cer", type=float, default=None,
                    help="Pre-recorded base CER to fill summary when --skip_base.")
    args = ap.parse_args()

    cache_dir = os.environ.get("HF_HOME", f"/scratch/{os.environ['USER']}/hf_cache")
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"base={args.base}", flush=True)
    print(f"ft={args.ft}", flush=True)
    print(f"benchmark={args.benchmark} subset={args.subset} whisper_lang={args.whisper_lang}", flush=True)

    print("\n--- loading benchmark ---", flush=True)
    audios, refs = load_benchmark(args.benchmark, args.subset, args.cv_version, cache_dir)
    print(f"loaded n={len(audios)} utterances", flush=True)

    normalizer = get_normalizer()

    if args.skip_base:
        if args.base_wer is None or args.base_cer is None:
            raise SystemExit("--skip_base requires --base_wer and --base_cer.")
        base_res = {"n": -1, "wer": args.base_wer, "cer": args.base_cer,
                    "sample_hyps": [], "sample_refs": [], "skipped": True}
        print(f"  [base] skipped, using pre-recorded WER={args.base_wer:.4f} CER={args.base_cer:.4f}", flush=True)
    else:
        base_res = eval_model(args.base, audios, refs, args.whisper_lang, normalizer,
                              "base", batch_size=args.batch_size)
    ft_res = eval_model(args.ft, audios, refs, args.whisper_lang, normalizer,
                        "ft", batch_size=args.batch_size)

    summary = {
        "label": args.label,
        "benchmark": args.benchmark,
        "subset": args.subset,
        "whisper_lang": args.whisper_lang,
        "base_model": args.base,
        "ft_model": args.ft,
        "base": base_res,
        "ft": ft_res,
        "wer_delta_pct": (base_res["wer"] - ft_res["wer"]) / max(base_res["wer"], 1e-9) * 100,
        "cer_delta_pct": (base_res["cer"] - ft_res["cer"]) / max(base_res["cer"], 1e-9) * 100,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {args.out}", flush=True)
    print(f"\n=== {args.label} on {args.benchmark} {args.subset} ===", flush=True)
    print(f"  Base: WER={base_res['wer']:.4f} CER={base_res['cer']:.4f}", flush=True)
    print(f"  FT:   WER={ft_res['wer']:.4f} CER={ft_res['cer']:.4f}", flush=True)
    print(f"  ΔWER={summary['wer_delta_pct']:+.1f}%  ΔCER={summary['cer_delta_pct']:+.1f}%", flush=True)


if __name__ == "__main__":
    main()
