# WorldSpeech

This repository contains the build and evaluation pipeline for [**disco-eth/WorldSpeech**](https://huggingface.co/datasets/disco-eth/WorldSpeech) on HuggingFace, a multilingual ASR dataset of over **65k hours** of transcribed speech across **127 language-region variants**, drawn from national parliaments, public broadcasters, public-domain audiobooks, and international institutions. Each row is a 24 kHz speech utterance paired with a human-provided transcript, an aligned ASR transcript, character error rate (CER), a WADA-SNR estimate, and four DNSMOS-P.835 quality scores.

Use this repo to:

- evaluate a Whisper checkpoint on a single config (FLEURS or Common Voice)
- fine-tune Whisper on one or more configs and report Δ WER vs. the base checkpoint
- extend the dataset with a new language using the five-step alignment pipeline

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total hours | 65,072 |
| Distinct languages | 88 |
| Language variations | 127 |
| Languages >= 1,000 h | 24 |
| Languages >= 500 h | 28 |
| Languages >= 200 h | 37 |
| Languages >= 50 h | 53 |
| Avg DNSMOS-P.835 OVR | 2.83 |
| Sample rate | 24 kHz |

---

## How to use

```python
from datasets import load_dataset

ds = load_dataset("disco-eth/WorldSpeech", "nl_nl", split="train")

row = ds[0]
wav    = row["audio"]["array"]
sr     = row["audio"]["sampling_rate"]
text   = row["human_transcript"]
snr    = row["snr"]
dnsmos = row["dnsmos_ovr"]
```

### Streaming

```python
from datasets import load_dataset

ds = load_dataset("disco-eth/WorldSpeech", "nl_nl", split="train", streaming=True)

for row in ds:
    wav = row["audio"]["array"]
    break
```

---

## Schema

| Field | Description |
|-------|-------------|
| `audio` | OGG Opus decoded to `{"array": np.float32[N], "sampling_rate": 24000}` |
| `human_transcript` | Human-provided transcript |
| `asr_transcript` | ASR output used for alignment and CER computation |
| `cer` | Character error rate between `asr_transcript` and `human_transcript` |
| `snr` | WADA-SNR estimate, dB |
| `dnsmos_sig` | DNSMOS-P.835 signal quality |
| `dnsmos_bak` | DNSMOS-P.835 background noise |
| `dnsmos_ovr` | DNSMOS-P.835 overall MOS |
| `dnsmos_p808` | DNSMOS-P.808 MOS |
| `duration` | Segment duration in seconds |
| `source` | Source identifier (e.g. `parliament_nl`, `librivox`, `voa_hausa`) |
| `source_url` | URL to the original recording |
| `source_start_s` | Start offset within the source recording (seconds) |
| `source_end_s` | End offset within the source recording (seconds) |
| `session_date` | ISO-8601 date of the original recording |
| `segment_id` | Unique identifier for this segment |
| `language` | BCP-47 language tag |
| `country` | ISO 3166-1 alpha-2 country code (`un` / `va` for international) |

---

## Language Coverage

> **Note:** Five configs (`kh_km` Khmer, `la_lo` Lao, `mm_my` Burmese, `vn_vi` Vietnamese, `cn_ug` Uyghur), sourced from Radio Free Asia, include only metadata, transcripts, and source URLs. Audio must be downloaded separately from `source_url`.

| Config | Country | Language | Hours | DNSMOS OVR |
|--------|---------|----------|------:|----------:|
| `af_za` | South Africa | Afrikaans | 20.2 | 2.79 |
| `am_et` | Ethiopia | Amharic | 39.6 | 3.11 |
| `ar_bh` | Bahrain | Arabic | 272.5 | 2.72 |
| `ar_dz` | Algeria | Arabic | 92.9 | 2.66 |
| `ar_eg` | Egypt | Arabic | 22.0 | 2.86 |
| `ar_iq` | Iraq | Arabic | 291.9 | 2.94 |
| `ar_kw` | Kuwait | Arabic | 175.5 | 2.82 |
| `ar_ma` | Morocco | Arabic | 78.3 | 2.68 |
| `ar_sa` | Saudi Arabia | Arabic | 6.1 | 2.01 |
| `ar_tn` | Tunisia | Arabic | 50.9 | 2.74 |
| `ar_un` | United Nations | Arabic | 11.1 | 3.03 |
| `as_in` | India | Assamese | 54.5 | 3.06 |
| `az_az` | Azerbaijan | Azerbaijani | 305.4 | 2.76 |
| `be_by` | Belarus | Belarusian | 24.2 | 3.10 |
| `bn_bd` | Bangladesh | Bengali | 46.1 | 3.03 |
| `bn_in` | India | Bengali | 26.7 | 2.83 |
| `ca_es` | Spain | Catalan | 1,171.0 | 2.99 |
| `ca_fr` | Canada | French | 5,989.3 | 2.88 |
| `ckb_iq` | Iraq | Sorani Kurdish | 35.3 | 2.93 |
| `cn_ug` | China | Uyghur | 200.0 | 2.86 |
| `cnr_me` | Montenegro | Montenegrin | 47.9 | 2.98 |
| `crs_sc` | Seychelles | Kreol Seselwa | 1,602.3 | 3.15 |
| `cs_cz` | Czech Republic | Czech | 2,689.5 | 2.87 |
| `cz_cs` | Czech Republic | Czech (Senate) | 1,027.9 | 2.85 |
| `de_at` | Austria | German | 1,077.5 | 2.93 |
| `de_li` | Liechtenstein | German | 829.7 | 2.79 |
| `dgo_in` | India | Dogri | 34.5 | 3.17 |
| `dv_mv` | Maldives | Dhivehi | 20.0 | 2.92 |
| `el_cy` | Cyprus | Greek | 394.4 | 2.90 |
| `el_gr` | Greece | Greek | 35.9 | 3.37 |
| `en_au` | Australia | English | 568.4 | 3.02 |
| `en_jm` | Jamaica | English | 9.6 | 2.52 |
| `en_ke` | Kenya | English | 170.0 | 2.97 |
| `en_nz` | New Zealand | English | 435.7 | 2.68 |
| `en_pk` | Pakistan | English | 8.5 | 2.28 |
| `en_sl` | Sierra Leone | English | 102.4 | 2.70 |
| `en_us` | USA | English | 3,725.9 | 2.85 |
| `en_zm` | Zambia | English | 291.6 | 2.70 |
| `eo` | International | Esperanto | 15.0 | 3.27 |
| `es_ar` | Argentina | Spanish | 251.7 | 2.78 |
| `es_cl` | Chile | Spanish | 1,740.5 | 2.94 |
| `es_co` | Colombia | Spanish | 156.8 | 2.75 |
| `es_es` | Spain | Spanish | 2,097.4 | 2.52 |
| `es_mx` | Mexico | Spanish | 901.6 | 2.63 |
| `es_pe` | Peru | Spanish | 387.4 | 2.63 |
| `es_pr` | Puerto Rico | Spanish | 228.6 | 2.73 |
| `es_py` | Paraguay | Spanish | 133.2 | 3.14 |
| `es_uy` | Uruguay | Spanish | 894.4 | 2.94 |
| `fa_ir` | Iran | Persian | 27.6 | 2.94 |
| `fr_cd` | DR Congo | French | 28.0 | 3.00 |
| `fr_ci` | Côte d'Ivoire | French | 11.9 | 2.97 |
| `ga_ie` | Ireland | Irish | 60.6 | 3.00 |
| `grc_gr` | Greece | Ancient Greek | 1.3 | 3.25 |
| `gu_in` | India | Gujarati | 27.2 | 3.09 |
| `ha_ng` | Nigeria | Hausa | 54.9 | 2.88 |
| `ha_td` | Chad | Hausa | 71.5 | 2.87 |
| `he_il` | Israel | Hebrew | 41.8 | 3.21 |
| `hi_in` | India | Hindi | 1,706.7 | 2.68 |
| `hu_hu` | Hungary | Hungarian | 1,350.1 | 2.96 |
| `hy_am` | Armenia | Armenian | 1,138.9 | 2.59 |
| `id_id` | Indonesia | Indonesian | 340.0 | 2.95 |
| `ig_ng` | Nigeria | Igbo | 40.7 | 3.31 |
| `iu_ca` | Canada | Inuktitut | 33.8 | 2.81 |
| `ja_jp` | Japan | Japanese | 1,387.2 | 2.86 |
| `ka_ge` | Georgia | Georgian | 206.4 | 2.37 |
| `kh_km` | Cambodia | Khmer | 1,323.0 | 2.85 |
| `kk_kz` | Kazakhstan | Kazakh | 179.0 | 2.84 |
| `kn_in` | India | Kannada | 30.2 | 3.19 |
| `ko_kr` | South Korea | Korean | 1,454.1 | 3.07 |
| `kok_in` | India | Konkani | 0.7 | 3.19 |
| `la_lo` | Laos | Lao | 827.0 | 2.82 |
| `la_va` | Vatican | Latin | 34.8 | 3.37 |
| `lb_lu` | Luxembourg | Luxembourgish | 1,805.3 | 2.87 |
| `mai_in` | India | Maithili | 9.6 | 3.19 |
| `mfe_mu` | Mauritius | Mauritian Creole | 44.3 | 3.45 |
| `mi_nz` | New Zealand | Māori | 1.6 | 2.85 |
| `ml_in` | India | Malayalam | 56.7 | 3.10 |
| `mm_my` | Myanmar | Burmese | 865.0 | 2.88 |
| `mn_mn` | Mongolia | Mongolian | 181.0 | 2.80 |
| `mr_in` | India | Marathi | 114.3 | 3.17 |
| `ms_my` | Malaysia | Malay | 432.0 | 2.91 |
| `ne_in` | India | Nepali | 5.7 | 2.94 |
| `ne_np` | Nepal | Nepali | 58.0 | 2.79 |
| `nl_be` | Belgium | Dutch (Flemish) | 960.5 | 2.59 |
| `nl_nl` | Netherlands | Dutch | 4,497.6 | 3.00 |
| `nr_za` | South Africa | Southern Ndebele | 0.2 | 2.60 |
| `nso_za` | South Africa | Northern Sotho | 0.7 | 2.70 |
| `om_et` | Ethiopia | Oromo | 16.3 | 3.06 |
| `or_in` | India | Odia | 57.8 | 3.05 |
| `pa_in` | India | Punjabi | 4.0 | 2.61 |
| `pl_pl` | Poland | Polish | 2,731.9 | 2.70 |
| `pt_br` | Brazil | Portuguese | 1,763.5 | 2.68 |
| `rm_ch` | Switzerland | Romansh | 163.1 | 3.14 |
| `ro_md` | Moldova | Romanian | 1,059.5 | 2.66 |
| `ro_ro` | Romania | Romanian | 686.3 | 2.75 |
| `ru_by` | Belarus | Russian | 2.9 | 2.79 |
| `ru_ru` | Russia | Russian | 1,534.2 | 3.03 |
| `rw_rw` | Rwanda | Kinyarwanda | 17.6 | 2.63 |
| `rw_voa` | Rwanda | Kinyarwanda (VOA) | 14.5 | 2.77 |
| `si_lk` | Sri Lanka | Sinhala | 154.0 | 2.40 |
| `sm_ws` | Samoa | Samoan | 55.9 | 2.79 |
| `sn_zw` | Zimbabwe | Shona | 18.2 | 2.88 |
| `sq_al` | Albania | Albanian | 257.4 | 2.69 |
| `sq_xk` | Kosovo | Albanian | 176.6 | 2.69 |
| `ss_za` | South Africa | Swati | 0.1 | 2.84 |
| `st_za` | South Africa | Sesotho | 0.4 | 2.67 |
| `sv_ax` | Åland Islands | Swedish | 66.0 | 2.73 |
| `sw_ke` | Kenya | Swahili | 257.4 | 2.93 |
| `sw_tz` | Tanzania | Swahili | 748.8 | 2.33 |
| `ta_in` | India | Tamil | 36.1 | 3.06 |
| `ta_lk` | Sri Lanka | Tamil | 204.0 | 2.80 |
| `te_in` | India | Telugu | 77.0 | 3.13 |
| `th_th` | Thailand | Thai | 1,175.6 | 3.23 |
| `ti_et` | Ethiopia | Tigrinya | 14.0 | 3.10 |
| `tl_ph` | Philippines | Tagalog | 219.0 | 2.88 |
| `tn_bw` | Botswana | Tswana | 49.7 | 2.88 |
| `tn_za` | South Africa | Tswana | 0.9 | 2.69 |
| `tr_tr` | Turkey | Turkish | 1,007.6 | 2.22 |
| `ts_za` | South Africa | Tsonga | 0.2 | 2.75 |
| `ur_in` | India | Urdu | 12.6 | 3.01 |
| `ur_pk` | Pakistan | Urdu | 73.1 | 2.44 |
| `uz_uz` | Uzbekistan | Uzbek | 33.7 | 2.84 |
| `ve_za` | South Africa | Venda | 0.1 | 2.72 |
| `vn_vi` | Vietnam | Vietnamese | 726.0 | 2.90 |
| `xh_za` | South Africa | Xhosa | 10.1 | 2.74 |
| `yue_hk` | Hong Kong | Cantonese | 1,943.5 | 3.02 |
| `zh_tw` | Taiwan | Mandarin | 1,482.1 | 2.47 |
| `zu_za` | South Africa | Zulu | 19.0 | 2.71 |
| **Total** | | | **65,072.4** | **2.83** |

---

## ASR Results

Fine-tuning open-source ASR models (Whisper-large family) on WorldSpeech consistently lowers word error rate across typologically diverse languages. A representative slice of the FLEURS evaluation:

| Config | Language | Family | WER base | WER + WS FT | Δ |
|---|---|---|---:|---:|---:|
| `lb_lu` | Luxembourgish | Germanic | 0.95 | 0.29 | -70% |
| `mm_my` | Burmese | Sino-Tibetan | 1.07 | 0.25 | -77% |
| `sw_tz` | Swahili | Bantu | 0.58 | 0.15 | -75% |
| `tn_bw` | Tswana | Bantu | 1.76 | 0.64 | -63% |
| `hu_hu` | Hungarian | Uralic | 0.28 | 0.18 | -37% |

Gains are largest where the pretrained model is weakest. For languages with a strong pretrained baseline or a smaller WS subset, gains are correspondingly smaller.

---

## Evaluating Whisper on a config

Scripts in `eval/` evaluate a Whisper checkpoint on FLEURS or Common Voice and report WER / CER. They also support comparing a fine-tuned checkpoint against its base, reporting Δ WER. Decoding is fixed:

- Greedy decoding (no beam search)
- Forced decoder language token where applicable
- Canonical Whisper `suppress_tokens` re-injected for community fine-tunes
- `BasicTextNormalizer` from `transformers.models.whisper.english_normalizer` applied to references and hypotheses
- Metrics computed with `jiwer` (WER, CER)

Minimal usage:

```bash
export HF_TOKEN=<your_huggingface_token>

# Base + FT comparison
python eval/eval_whisper_benchmark.py \
    --base openai/whisper-large-v2 \
    --ft   /path/to/whisper-lb_lu-v2 \
    --benchmark fleurs --subset lb_lu \
    --whisper_lang luxembourgish \
    --label lb_lu_v2 --out results/lb_lu_v2.json

# Base only
python eval/eval_base_only.py \
    --base openai/whisper-large-v2 \
    --subset lb_lu --whisper_lang luxembourgish \
    --out results/lb_lu_base.json
```

See [`eval/README.md`](eval/README.md) for SLURM wrappers and the full protocol.

## Fine-tuning Whisper on a config

`train/finetune_whisper_generic.py` fine-tunes Whisper on a single WorldSpeech config (or a union of configs) and tracks WER on FLEURS. Driven via the `train/run_finetune.slurm` wrapper:

```bash
sbatch \
  --export=ALL,CONFIG=lb_lu,LANG=luxembourgish,FLEURS=lb_lu,BASE=openai/whisper-large-v2 \
  train/run_finetune.slurm
```

Override paths and conda env via `REPO_ROOT`, `CONDA_ENV`, `CONDA_HOOK`, `MODELS_NFS_DIR`, `HF_HOME`. Override training knobs via `MAX_TRAIN_HOURS`, `NUM_EPOCHS`, `SEED`.

---

## Building a new config

Adding a language follows a five-step pipeline:

```
01_build_csv      Map sessions to audio + transcript URLs
02_download       Download audio, convert to 24 kHz mono WAV
03_verify         Spot-check pairs, run ASR, compute CER (target < 0.3)
04_align          VAD -> ASR -> sliding-window CER alignment
05_upload         Cut segments, build parquet, push to HuggingFace
```

Alignment uses the [EuroSpeech parliament_transcript_aligner](https://github.com/SamuelPfisterer/EuroSpeech). Each segment's `human_transcript` is always sourced from official records, never from ASR output.

### Repository layout

```
.
├── README.md
├── eval/
│   ├── README.md
│   ├── eval_whisper_benchmark.py    Base + FT eval, Δ WER / Δ CER
│   ├── eval_base_only.py            Single-checkpoint eval
│   ├── run_eval.slurm
│   └── run_base_eval.slurm
└── train/
    ├── finetune_whisper_generic.py  Whisper fine-tune driver
    └── run_finetune.slurm
```

---

## Citation

```bibtex
Coming soon.
```

---

## Dataset License

WorldSpeech is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Individual source licenses apply per config and are listed in the [Sources and Licenses](#sources-and-licenses) section. Where a source license is more restrictive than CC BY-NC 4.0, the more restrictive terms govern that config.

---

## Sources and Licenses

| Config | Country / Language | License | Notes |
|---|---|---|---|
| `af_za` | South Africa / Afrikaans | [Parliamentary Proceedings](https://www.parliament.gov.za/legal) | |
| `am_et` | Ethiopia / Amharic | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `ar_bh` | Bahrain / Arabic | [Parliamentary Proceedings](https://www.nuwab.bh/) | |
| `ar_dz` | Algeria / Arabic | [Parliamentary Proceedings](https://www.apn.dz/) | |
| `ar_eg` | Egypt / Arabic | [Parliamentary Proceedings](https://www.parliament.gov.eg/) | |
| `ar_iq` | Iraq / Arabic | [Parliamentary Proceedings](https://parliament.iq/) | |
| `ar_kw` | Kuwait / Arabic | [Parliamentary Proceedings](https://www.kna.kw/) | |
| `ar_ma` | Morocco / Arabic | [Parliamentary Proceedings](https://www.parlement.ma/) | |
| `ar_sa` | Saudi Arabia / Arabic | [Government Archive (Public Record)](https://www.gph.gov.sa/) | |
| `ar_tn` | Tunisia / Arabic | [Parliamentary Proceedings](https://www.parliament.tn/) | |
| `ar_un` | United Nations / Arabic | [UN Official Records](https://www.un.org/en/about-us/copyright) | UN official records, non-commercial / research use per UN terms. Contact permissions@un.org for commercial use. |
| `as_in` | India / Assamese | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `az_az` | Azerbaijan / Azerbaijani | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `be_by` | Belarus / Belarusian | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | |
| `bn_bd` | Bangladesh / Bengali | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `bn_in` | India / Bengali | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `ca_es` | Spain / Catalan | [Parliamentary Proceedings](https://www.parlament.cat/) | |
| `ca_fr` | Canada / French | [Parliamentary Proceedings](https://www.assnat.qc.ca/) | |
| `ckb_iq` | Iraq / Sorani Kurdish | [Parliamentary Proceedings](https://parliament.iq/) | |
| `cn_ug` | China / Uyghur | [RFA Terms of Use (non-commercial)](https://www.rfa.org/english/about/terms-of-use/) | Metadata only, audio not embedded. Download audio from `source_url`. Non-commercial research use, see RFA terms. |
| `cnr_me` | Montenegro / Montenegrin | [Parliamentary Proceedings](https://www.skupstina.me/) | |
| `crs_sc` | Seychelles / Kreol Seselwa | [Parliamentary Proceedings](https://www.nationalassembly.sc/) | |
| `cs_cz` | Czech Republic / Czech | [Parliamentary Proceedings](https://www.psp.cz/) | |
| `cz_cs` | Czech Republic / Czech | [Parliamentary Proceedings](https://www.senat.cz/) | Czech Senate (Senát Parlamentu ČR). |
| `de_at` | Austria / German | [Parliamentary Proceedings](https://www.parlament.gv.at/rechtliches) | |
| `de_li` | Liechtenstein / German | [Parliamentary Proceedings](https://www.landtag.li/) | |
| `dgo_in` | India / Dogri | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `dv_mv` | Maldives / Dhivehi | [Parliamentary Proceedings](https://majlis.gov.mv/) | |
| `el_cy` | Cyprus / Greek | [Parliamentary Proceedings](https://www.parliament.cy/) | |
| `el_gr` | Greece / Greek | [Parliamentary Proceedings](https://www.hellenicparliament.gr/) | |
| `en_au` | Australia / English | [Parliamentary Proceedings](https://www.aph.gov.au/Help/Disclaimer_Privacy_Copyright) | |
| `en_jm` | Jamaica / English | [Parliamentary Proceedings](https://www.japarliament.gov.jm/) | |
| `en_ke` | Kenya / English | [Parliamentary Proceedings](https://www.parliament.go.ke/) | |
| `en_nz` | New Zealand / English | [Parliamentary Proceedings](https://www.parliament.nz/en/footer/copyright/) | |
| `en_pk` | Pakistan / English | [Parliamentary Proceedings](https://www.na.gov.pk/) | |
| `en_sl` | Sierra Leone / English | [Parliamentary Proceedings](https://www.parliament.gov.sl/) | |
| `en_us` | USA / English | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal proceedings (Congress.gov / C-SPAN public domain content). |
| `en_zm` | Zambia / English | [Parliamentary Proceedings](https://www.parliament.gov.zm/) | |
| `eo` | International / Esperanto | [CC0 1.0 (Public Domain)](https://librivox.org/pages/public-domain/) | |
| `es_ar` | Argentina / Spanish | [Parliamentary Proceedings](https://www.hcdn.gob.ar/) | |
| `es_cl` | Chile / Spanish | [Parliamentary Proceedings](https://www.camara.cl/) | |
| `es_co` | Colombia / Spanish | [Parliamentary Proceedings](https://www.camara.gov.co/) | |
| `es_es` | Spain / Spanish | [Parliamentary Proceedings](https://www.congreso.es/) | |
| `es_mx` | Mexico / Spanish | [Parliamentary Proceedings](https://www.congreso.gob.mx/) | |
| `es_pe` | Peru / Spanish | [Parliamentary Proceedings](https://www.congreso.gob.pe/) | |
| `es_pr` | Puerto Rico / Spanish | [Parliamentary Proceedings](https://www.camaraderepresentantes.org/) | |
| `es_py` | Paraguay / Spanish | [Parliamentary Proceedings](https://www.congreso.gov.py/) | |
| `es_uy` | Uruguay / Spanish | [Parliamentary Proceedings](https://parlamento.gub.uy/) | |
| `fa_ir` | Iran / Persian | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `fr_cd` | DR Congo / French | [ICC Court Records](https://www.icc-cpi.int/disclaimer) | Public judicial proceedings of an international court, non-commercial / educational use per ICC terms. |
| `fr_ci` | Côte d'Ivoire / French | [ICC Court Records](https://www.icc-cpi.int/disclaimer) | Public judicial proceedings of an international court, non-commercial / educational use per ICC terms. |
| `ga_ie` | Ireland / Irish | [CC BY 4.0 (Oireachtas PSI Licence)](https://www.oireachtas.ie/en/open-data/license/) | |
| `grc_gr` | Greece / Ancient Greek | [CC0 1.0 (Public Domain)](https://librivox.org/pages/public-domain/) | |
| `gu_in` | India / Gujarati | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `ha_ng` | Nigeria / Hausa | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `ha_td` | Chad / Hausa | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `he_il` | Israel / Hebrew | [CC0 1.0 (Public Domain)](https://librivox.org/pages/public-domain/) | LibriVox recordings (CC0) + Ben-Yehuda Project texts (public domain per Israeli copyright law). |
| `hi_in` | India / Hindi | [Parliamentary Proceedings](https://sansad.in/) | |
| `hu_hu` | Hungary / Hungarian | [Parliamentary Proceedings](https://www.parlament.hu/) | |
| `hy_am` | Armenia / Armenian | [Parliamentary Proceedings](https://www.parliament.am/) | |
| `id_id` | Indonesia / Indonesian | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `ig_ng` | Nigeria / Igbo | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `iu_ca` | Canada / Inuktitut | [Parliamentary Proceedings](https://www.assembly.nu.ca/) | |
| `ja_jp` | Japan / Japanese | [CC0 1.0 (Public Domain)](https://librivox.org/pages/public-domain/) | LibriVox recordings (CC0) + Aozora Bunko texts (public domain, 50yr Japanese law). |
| `ka_ge` | Georgia / Georgian | [Parliamentary Proceedings](https://parliament.ge/) | |
| `kh_km` | Cambodia / Khmer | [RFA Terms of Use (non-commercial)](https://www.rfa.org/english/about/terms-of-use/) | Metadata only, audio not embedded. Download audio from `source_url`. Non-commercial research use, see RFA terms. |
| `kk_kz` | Kazakhstan / Kazakh | [Parliamentary Proceedings](https://www.parlam.kz/) | |
| `kn_in` | India / Kannada | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `ko_kr` | South Korea / Korean | [Parliamentary Proceedings](https://www.assembly.go.kr/) | |
| `kok_in` | India / Konkani | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `la_lo` | Laos / Lao | [RFA Terms of Use (non-commercial)](https://www.rfa.org/english/about/terms-of-use/) | Metadata only, audio not embedded. Download audio from `source_url`. Non-commercial research use, see RFA terms. |
| `la_va` | Vatican / Latin | [Vatican Radio Archive](https://www.vaticannews.va/en/others/terms-and-conditions.html) | Vatican Radio historical archive, contact Vatican Media for commercial use. |
| `lb_lu` | Luxembourg / Luxembourgish | [Parliamentary Proceedings](https://www.chd.lu/) | |
| `mai_in` | India / Maithili | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `mfe_mu` | Mauritius / Mauritian Creole | [Parliamentary Proceedings](https://govmu.org/) | |
| `mi_nz` | New Zealand / Māori | [Parliamentary Proceedings](https://www.parliament.nz/en/footer/copyright/) | |
| `ml_in` | India / Malayalam | [Parliamentary Proceedings](https://niyamasabha.org/) | |
| `mm_my` | Myanmar / Burmese | [RFA Terms of Use (non-commercial)](https://www.rfa.org/english/about/terms-of-use/) | Metadata only, audio not embedded. Download audio from `source_url`. Non-commercial research use, see RFA terms. |
| `mn_mn` | Mongolia / Mongolian | [Parliamentary Proceedings / CC0](https://www.parliament.mn/) | Parliament sessions (public record) + Latter-day Saints addresses (CC0 per lds.org). |
| `mr_in` | India / Marathi | [Parliamentary Proceedings](https://www.vidhan.maharashtra.gov.in/) | |
| `ms_my` | Malaysia / Malay | [Parliamentary Proceedings](https://www.parlimen.gov.my/) | |
| `ne_in` | India / Nepali | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `ne_np` | Nepal / Nepali | [Parliamentary Proceedings](https://www.parliament.gov.np/) | |
| `nl_be` | Belgium / Dutch | [Parliamentary Proceedings](https://www.vlaamsparlement.be/) | |
| `nl_nl` | Netherlands / Dutch | [Parliamentary Proceedings](https://www.tweedekamer.nl/) | |
| `nr_za` | South Africa / Southern Ndebele | [Parliamentary Proceedings](https://www.parliament.gov.za/legal) | |
| `nso_za` | South Africa / Northern Sotho | [Parliamentary Proceedings](https://www.parliament.gov.za/legal) | |
| `om_et` | Ethiopia / Oromo | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `or_in` | India / Odia | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `pa_in` | India / Punjabi | [Parliamentary Proceedings](https://vidhanpb.gov.in/) | |
| `pl_pl` | Poland / Polish | [Parliamentary Proceedings](https://www.sejm.gov.pl/) | |
| `pt_br` | Brazil / Portuguese | [Parliamentary Proceedings](https://www.senado.leg.br/) | |
| `rm_ch` | Switzerland / Romansh | [SRG SSR Terms of Use](https://www.srgssr.ch/en/who-we-are/public-service/) | Corpus compiled from publicly broadcast RTR programming, contact SRG SSR for commercial use. |
| `ro_md` | Moldova / Romanian | [Parliamentary Proceedings](https://www.parlament.md/) | |
| `ro_ro` | Romania / Romanian | [Parliamentary Proceedings](https://www.senat.ro/) | |
| `ru_by` | Belarus / Russian | [Government Publication (Public Record)](https://president.gov.by/) | |
| `ru_ru` | Russia / Russian | [Parliamentary Proceedings](https://duma.gov.ru/) | |
| `rw_rw` | Rwanda / Kinyarwanda | [Parliamentary Proceedings](https://www.parliament.gov.rw/) | |
| `rw_voa` | Rwanda / Kinyarwanda (VOA) | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `si_lk` | Sri Lanka / Sinhala | [Parliamentary Proceedings](https://www.parliament.lk/) | |
| `sm_ws` | Samoa / Samoan | [Parliamentary Proceedings](https://parliament.ws/) | |
| `sn_zw` | Zimbabwe / Shona | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `sq_al` | Albania / Albanian | [Parliamentary Proceedings](https://www.parlament.al/) | |
| `sq_xk` | Kosovo / Albanian | [Parliamentary Proceedings](https://www.assembly-kosova.org/) | |
| `ss_za` | South Africa / Swati | [Parliamentary Proceedings](https://www.parliament.gov.za/legal) | |
| `st_za` | South Africa / Sesotho | [Parliamentary Proceedings](https://www.parliament.gov.za/legal) | |
| `sv_ax` | Åland Islands / Swedish | [Parliamentary Proceedings](https://www.lagting.ax/) | |
| `sw_ke` | Kenya / Swahili | [Parliamentary Proceedings](https://www.parliament.go.ke/) | |
| `sw_tz` | Tanzania / Swahili | [Parliamentary Proceedings](https://www.bunge.go.tz/) | |
| `ta_in` | India / Tamil | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `ta_lk` | Sri Lanka / Tamil | [Parliamentary Proceedings](https://www.parliament.lk/) | |
| `te_in` | India / Telugu | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `th_th` | Thailand / Thai | [Parliamentary Proceedings](https://www.parliament.go.th/) | |
| `ti_et` | Ethiopia / Tigrinya | [Public Domain (17 USC §105)](https://www.law.cornell.edu/uscode/text/17/105) | US federal agency content (VOA). |
| `tl_ph` | Philippines / Tagalog | [Parliamentary Proceedings](https://www.congress.gov.ph/) | |
| `tn_bw` | Botswana / Tswana | [Parliamentary Proceedings](https://www.parliament.gov.bw/) | |
| `tn_za` | South Africa / Tswana | [Parliamentary Proceedings](https://www.parliament.gov.za/legal) | |
| `tr_tr` | Turkey / Turkish | [Parliamentary Proceedings](https://www.tbmm.gov.tr/) | |
| `ts_za` | South Africa / Tsonga | [Parliamentary Proceedings](https://www.parliament.gov.za/legal) | |
| `ur_in` | India / Urdu | [All India Radio Archive (Prasar Bharati)](https://newsonair.gov.in/) | Prasar Bharati public broadcaster archive. |
| `ur_pk` | Pakistan / Urdu | [Parliamentary Proceedings](https://www.na.gov.pk/) | |
| `uz_uz` | Uzbekistan / Uzbek | [RFE/RL Terms of Use (non-commercial)](https://about.rferl.org/use-our-content/) | US government-funded broadcaster, non-commercial research use. Original Ozodlik (RFE/RL Uzbek) content. |
| `ve_za` | South Africa / Venda | [Parliamentary Proceedings](https://www.parliament.gov.za/legal) | |
| `vn_vi` | Vietnam / Vietnamese | [RFA Terms of Use (non-commercial)](https://www.rfa.org/english/about/terms-of-use/) | Metadata only, audio not embedded. Download audio from `source_url`. Non-commercial research use, see RFA terms. |
| `xh_za` | South Africa / Xhosa | [Parliamentary Proceedings](https://www.parliament.gov.za/legal) | |
| `yue_hk` | Hong Kong / Cantonese | [Parliamentary Proceedings](https://www.legco.gov.hk/en/general/disclaimer.html) | |
| `zh_tw` | Taiwan / Mandarin | [Parliamentary Proceedings](https://www.ly.gov.tw/) | |
| `zu_za` | South Africa / Zulu | [Parliamentary Proceedings](https://www.parliament.gov.za/legal) | |

---

## Acknowledgments

Pipeline built on the [EuroSpeech alignment framework](https://github.com/SamuelPfisterer/EuroSpeech). ASR uses [Whisper](https://github.com/openai/whisper) (OpenAI), [MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) (Meta), and community fine-tunes from HuggingFace. VAD uses [Silero VAD](https://github.com/snakers4/silero-vad).
