#!/usr/bin/env python3
"""Base-only Whisper FLEURS eval (no FT). Args: --base --subset --whisper_lang --out"""
import argparse, json, os, time
import torch, jiwer
from datasets import load_dataset, Audio
from transformers import pipeline as hf_pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def _safe_load(subset, cache_dir):
    try:
        return load_dataset("google/fleurs", subset, split="test",
                            cache_dir=cache_dir, trust_remote_code=True)
    except FileNotFoundError as e:
        print(f"  [cache] redownloading: {e}", flush=True)
        return load_dataset("google/fleurs", subset, split="test",
                            cache_dir=cache_dir, trust_remote_code=True,
                            download_mode="force_redownload")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--whisper_lang", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cache = os.environ.get("HF_HOME", f"/scratch/{os.environ['USER']}/hf_cache")
    ds = _safe_load(args.subset, cache).cast_column("audio", Audio(sampling_rate=16000))
    print(f"loaded n={len(ds)}", flush=True)
    norm = BasicTextNormalizer()
    t0 = time.time()
    pipe = hf_pipeline("automatic-speech-recognition", model=args.base,
                       device="cuda:0" if torch.cuda.is_available() else "cpu",
                       torch_dtype=torch.float16,
                       generate_kwargs={"language": args.whisper_lang, "task": "transcribe"},
                       chunk_length_s=30)
    bs = int(os.environ.get("WHISPER_EVAL_BS", "8"))
    n = len(ds)
    refs = [norm(ds[i]["transcription"]) for i in range(n)]
    audios = (ds[i]["audio"]["array"] for i in range(n))
    preds, done = [], 0
    try:
        for out in pipe(audios, batch_size=bs):
            preds.append(norm(out["text"]))
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{n}", flush=True)
    except Exception as e:
        print(f"  batch fail at {done}/{n}: {e}", flush=True)
    while len(preds) < n:
        preds.append("")
    pairs = [(p if p and p.strip() else " ", r if r and r.strip() else " ")
             for p, r in zip(preds, refs) if r and r.strip()]
    p, r = zip(*pairs)
    wer = jiwer.wer(list(r), list(p))
    cer = jiwer.cer(list(r), list(p))
    dt = time.time() - t0
    print(f"\n=== {args.subset} base={args.base} ===\n  n={len(p)} WER={wer:.4f} CER={cer:.4f} ({dt:.0f}s)", flush=True)
    with open(args.out, "w") as f:
        json.dump({"subset": args.subset, "base": args.base, "n": len(p),
                   "wer": wer, "cer": cer}, f, indent=2)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
