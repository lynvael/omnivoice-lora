# MEMORY

## Goal
Fine-tune OmniVoice to speak as Mita (MiSide), **Russian dub (DreamCast)**. Personal use only —
do not publish weights/dataset. Full plan in `PLAN.md`.

## Machine (verified 2026-11-15)
- GPU: RTX 3060 Ti, **8 GB VRAM** (Ampere → bf16 ok, flex_attention not used)
- CPU: Ryzen 7 5700G, 32 GB RAM
- OS: Windows, bash available; ffmpeg 9.0.1 on PATH
- Python 3.13 global; project `.venv` = 3.12.14 (faster-whisper only, **no torch**)
- Scratch files go in `./tmp/` (user preference — NOT C:\Temp)

## Training environment (stage 0 done 2026-11-15)
- Repo cloned to `vendor/OmniVoice` at **`08be0b4`** (v0.2.1). **Separate** venv
  `vendor/OmniVoice/.venv` (7.6 GB) — never install torch into the project `.venv`
  (keeps faster-whisper/ctranslate2 CUDA isolated).
- `uv sync --extra lora` → torch **2.8.0+cu128**, `cuda.is_available() True`, bf16 ok,
  peft 0.21.3, transformers 5.3.0. Run long installs detached via
  `Start-Process cmd /c tmp/uv_sync.cmd` with the log in `logs/`.
- `k2-fsa/OmniVoice` is already in the HF cache (3.0 GB) and ships its own `audio_tokenizer/`
  subfolder. **Tokenization needs no extra download**: `--tokenizer_path
  <snapshot>/audio_tokenizer` loads `DacFeatureExtractor` (sr 24000, hop 960) and
  `HiggsAudioV2TokenizerModel` (`frame_rate 25`, `num_quantizers 8`) — verified by loading it.
  The `eustlb/higgs-audio-v2-tokenizer` default in the run script can be ignored.
- Inference smoke: `omnivoice-infer --instruct "female, young adult" --language ru` →
  2.08 s wav; ~17 s model load + ~2.5 s generation on the 3060 Ti.
- **Instruct patch confirmed twice**: stock CLI raises `ValueError` for `mita`;
  `vd._INSTRUCT_ALL_VALID |= {"mita"}` works because `|=` mutates the *same set object* that
  `omnivoice.models.omnivoice` imported by reference (a plain `=` would rebind and fail).
  End-to-end generation with `instruct="mita"` produces audio. No site-packages edits.

## Data source — no asset ripping needed
Game at `E:\SteamLibrary\steamapps\common\MiSide`. Voice lines ship as plain ogg:
- `Data/LanguagesVoice/Russian-Dream Cast/<Category>/<id>.ogg`
- `Data/Languages/Russian/<Category>.txt` — **line N (1-based, blank lines counted) = `<id>.ogg`**
- Speaker comes from `// SpeakerName` markers, applies until the next marker
- 348 text lines have no audio → always intersect with the actual ogg set

### Measured volume
- Mita family = **1,515 clips / 42.51 min** (mita 705/19.01, mita_kind 399/10.84,
  mita_cap 228/7.35, mita_shorthairs 129/4.03, mita_dream 54/1.28)
- Excluded: Player + other characters = 1,154 clips / 30.26 min
- Audio: Vorbis 44.1 kHz stereo 112 kbps; max clip 5.65 s
- The old "survives `min_sample_tokens=50` (=0.67 s): 94.7%" line was built on the wrong
  75 fps rate — at 25 fps a 50-token floor would cut 2.0 s, not 0.67 s. Current configs use
  `min_sample_tokens: 11`, which drops nothing (shortest clip 0.42 s ≈ 10–11 frames).

## Training math (CRITICAL — corrected 2026-11-15)
- Higgs-Audio-v2 tokenizer runs at **25 frames/sec, not 75**.
  `hop_length = prod(acoustic.downsampling_ratios) = 8·5·4·2·3 = 960`;
  `frame_rate = ceil(24000 / 960) = 25`. Cross-check that settles it:
  `num_quantizers = int(1000·target_bandwidths[-1] // (frame_rate · codebook_nbits))
  = int(2000 // 250) = 8` — matches `num_audio_codebook: 8`. At 75 tok/s it would be 2.
- **The old "24000 / downsample_factor 320 = 75 frames/sec" note in PLAN.md is wrong by 3×.**
  Every step count derived from it is wrong; do not reuse them.
- Corpus 612 clips / 16.47 min ≈ **24.7k audio frames**. Mean processed sample ≈ 60 tokens
  (style+text prefix ≈19 + audio ≈41). 20 length buckets fragment batches to ~21 samples.
- SDPA path: **~29 steps/epoch** single-worker, **~35** with `num_workers: 2` (shard split).
- `steps: 1200` ≈ 34–41 epochs. Checkpoint = full fp32 accelerate state ≈ **2.85 GB**.
- `num_workers` must stay > 0: `builder.py` hardcodes `prefetch_factor=4`, which PyTorch
  rejects when `num_workers == 0`.

## Gotchas
- **`_resolve_instruct` in `omnivoice/models/omnivoice.py` validates `instruct` against a closed
  whitelist and raises `ValueError` for `mita_kind`.** Training is fine (processor injects the raw
  string into `<|instruct_start|>...`), inference is blocked until patched.
- `extract_audio_tokens` copies the whole label dict → extra JSONL fields (`speaker`, `instruct`)
  survive tokenization into shard JSONLs.
- Transcript parser traps: blank lines count; `//` markers appear mid-file; some text lines carry
  trailing ` // comment`; `[player]` placeholder needs substitution.
- Default `attn_implementation` is `flex_attention` → use SDPA on this GPU.
- **`id` in `data/manifest.jsonl` is NOT unique** — it is the ogg number within a category and
  restarts at 1 in every category (2,669 rows, only 711 distinct ids). Always key on
  `audio_relpath`. A resume keyed on `id` silently drops ~2,434 rows.
- **Whisper hallucinates on short/silent Russian clips.** Confirmed: `LocationDialogue Location1/13`
  (0.9 s, source "Хм-м-м, игрок...") transcribed as «Субтитры сделал DimaTorzok». When comparing
  `stt_result` against subtitles, filter known artifact phrases and very short clips, otherwise
  the mismatch report is dominated by noise.
- Model choice, same weights different runtime: `openai/whisper-large-v3` (safetensors,
  transformers/torch) vs `Systran/faster-whisper-large-v3` (model.bin, CTranslate2) — both
  2,944 MB, 1.55B params. `large-v3-turbo` is a different, smaller model (809M, 1,543 MB).

## Decisions
- LoRA fine-tune, not from scratch, not a different base model.
- **`data/manifest_mita.jsonl` — the future reference ("эталон") dataset**, 1,515 Mita rows.
  Fields: `id, character, instruct, location, audio_path, duration_s, text, text_raw,
  text_stt, sim, status, has_player, text_final, note`. `text_final` is what training must use;
  the user edits corrections by hand, `note` is free text for their comments.
- **`data/review_mita.md` — human review doc**, 1,515 rows, disjoint sections A→B→C→D
  (7 problems / 25 `[player]` / 67 mismatches / 1,416 matches), empty «правка» column.
- **`data/finetune/` — the actual training corpus**, built by `scripts/build_finetune_manifest.py`
  from `data/manifest.jsonl` (the authoritative source, NOT the derived view):
  `character == "mita"` (705) → `status == "match"` (**612 clips / 16.47 min**) → **582 train +
  30 dev**. Text = original `text` verbatim (`--text-source manifest`, fails if `text_final`
  diverges; switch to `--text-source final` after applying review corrections). Adds
  `language_id: "ru"` and `instruct: "mita"`; audio copied to `data/finetune/audio/<id>.ogg`
  with absolute space-free paths, so no GPU stage ever reads `E:`.
  93 excluded rows with reasons in `manifest_excluded.jsonl` (near 55, mismatch 28,
  mismatch_strong 7, halluc 3) — recoverable later via review. Dev is a loss/NaN monitor,
  not a checkpoint-ranking set (the eval processor re-randomises mask ratio and `drop_cond`).
- STT agreement: **94.1%** of Mita subtitles match the audio (1,297 exact, 124 near, 4 numbers).
  Only 90 rows need attention.
- `[player]` is the single biggest systemic issue: 26 Mita rows contain it, **17 mismatch** —
  the dub actress simply omits the address. Do not substitute «игрок»; remove it.
- STT for subtitle-vs-audio comparison: **faster-whisper (Systran CTranslate2 conversion of
  openai/whisper-large-v3), `language="ru"`, beam 5**. Runs **on CUDA fp16** at 0.59 s/file
  (vs ~7.5 s/file on CPU int8) after fixing the missing runtime DLLs — see below.
- CUDA fix for CTranslate2 on this box (no admin, no system CUDA toolkit needed):
  `uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12`
  then prepend `.venv/Lib/site-packages/nvidia/{cublas,cudnn,cuda_runtime,cuda_nvrtc}/bin` to PATH.
  `scripts/transcribe_stt.py` does this itself on win32, so it is a single command.
  Diagnose with `ctranslate2.get_supported_compute_types('cuda')` — it returns a set even when the
  DLLs are missing, so that alone does not prove it works; run a real transcribe.
- `data/manifest_stt.jsonl` contract is **deliberately minimal, 4 fields only**:
  `id` (= `<location with underscores>_<ogg_id>`), `location`, `audio_path`, `stt_result`.
  The script uses `data/manifest.jsonl` **only as a selector** (`is_mita == true` → 1,515 rows:
  mita 705, mita_kind 399, mita_cap 228, mita_shorthairs 129, mita_dream 54). Do not add fields.
- Only one writer at a time: the resume logic is not multi-writer safe.
- Launch detached via PowerShell `Start-Process cmd.exe /c <script.cmd> -WindowStyle Hidden`,
  **without** `-RedirectStandardOutput` — that flag forces a console-less CreateProcess and the
  child dies with the parent's console (observed after ~15 s, twice). Let the `.cmd` redirect
  itself with `>> logs\x.log`.
- Extracted 2026-11-15: 2,669 ogg + 19 transcripts in `data/`, manifest in `data/manifest.jsonl`.
  Mita family = **1,515 clips / 42.51 min** (mita 705/19.01, mita_kind 399/10.84, mita_cap
  228/7.35, mita_shorthairs 129/4.03, mita_dream 54/1.28). 1,154 clips / 30.26 min excluded.
- **Inline `// Speaker` at the end of a transcript line is a per-line speaker OVERRIDE**, not a
  comment — dialogue alternates inside one marker block. 81 manifest rows use `speaker_source:
  "inline"`. Getting this wrong silently mislabels 41 clips.
- Two runs: (2a) single `mita` over all 42 min as control, (2b) multi-speaker with `mita_kind`
  as the priority instruction.
- Fixed eval set of **20 Russian phrases**, frozen in `eval/phrases.jsonl` (sha256
  `d5110c09…c51b3c`, guarded by `eval/frozen.json`). Never edit it.
- Steps come from epochs, not from the config. Real numbers (computed by simulating
  `StreamLengthGroupDataset`, not by formula): train = 582 clips / 15.61 min / **23,703 audio
  frames**; processed length mean 58.8, max 177, min 23; total 34,244 tokens; **29 steps/epoch**
  at `batch_tokens: 2048` (batch-size cap 32 dominates; 45 at 1024, 27 at 4096). `steps: 1200`
  = **41.4 epochs**. The old "188k frames / 92 steps / 2750 steps" line was 75 fps arithmetic.

## Eval harness (Phase 0, built + reviewed 2026-11-15)
- `scripts/mita_voice/{runtime,generate}.py` — inference wrapper. Whitelist patch
  `vd._INSTRUCT_ALL_VALID |= {"mita"}` in memory; **per-phrase seed `seed*1000+i`** via upstream
  `fix_random_seed` → generation is **bit-reproducible** on this box (verified twice: identical
  tokens, max abs diff 0.0). `--lora-adapter` uses `load_lora_adapter(..., merge=True)` and records
  `adapter_sha256` (content, not basename) so two `lora-600` dirs cannot be confused.
- `eval/run_eval.py` = one command per variant → render → free model → WER (project venv
  subprocess) → SIM (vendor venv) → `eval/results/<variant>.json` + `index.json`.
  `--reuse-audio` skips render but verifies wav set, phrase sha, per-phrase seeds, instruct,
  adapter sha, gen params and every wav sha256. A failed stage is fatal and never overwrites a
  good result; `--limit` rows are marked `limit`/`n_phrases` and are refused by `make_baseline.py`.
- **WER** (`eval/wer.py`, project venv, faster-whisper large-v3 exactly as `manifest_stt.jsonl`):
  NFKC → lower → ё→е → punctuation→space → integer tokens verbalised with `num2words` (ru) on
  both sides → word-level Levenshtein DP → **pooled** (S+D+I)/ref_words. Saturated here: 19/20
  phrases are 0.00%; use it as a **veto** (`pass` iff pooled ≤ 3% and no phrase > 50%), not a ranker.
- **SIM** (`eval/speaker_sim.py`): upstream `ECAPA_TDNN_WAVLM` (WavLM-large + ECAPA head) from
  `k2-fsa/TTS_eval_models` cached at `vendor/eval_models` (2.4 GB). Anchor = frozen 60-clip
  sample of `manifest_train.jsonl` (`eval/sim_refs.json`, seed 42, 87.4 s). **Primary metric is
  `sim_best` = max cosine over the 60 refs**; centroid kept for continuity. Leave-one-out ceiling
  `sim_best` 0.6061 (p10 0.4204), centroid ceiling 0.6121 (p10 0.3124).
- Requires `s3prl==0.4.18` in the vendor venv (`uv pip install --no-deps`, torch untouched) and
  `num2words` in the project venv (`--no-deps`). Both installed. HF repo ships `hubconf.py` one
  level deeper than upstream expects; a cache-local copy at
  `vendor/eval_models/speaker_similarity/hubconf.py` fixes it (not a vendor-source edit).
- **Baseline** (`eval/baseline.json`, machine-generated by `eval/make_baseline.py`):
  `base_native` (`female, young adult`) WER 4.46 % → **fail** (one 0.73 s truncated p17), sim_best
  **0.378 ± 0.071**; `base_mita_tag` (`mita`) WER 0.00 % → pass, sim_best **0.272 ± 0.086**.
  The untrained `mita` tag is *worse* than the native voice → a blank canvas for LoRA; compare
  checkpoints against `base_mita_tag`, same tag and seeds.
- **Selection rule for the 12 checkpoints:** veto on WER, rank on `sim_best_mean`; declare a
  winner only if the paired bootstrap CI vs the **runner-up** excludes 0 **and** Δ ≥ 0.05; at
  n=20 paired SE ≈ 0.026 (per-phrase correlation ≈ −0.1, so pairing buys nothing), MDE ≈ 0.074 —
  anything below ~0.05 is noise, listen to the top two instead. The 60-ref anchor is drawn from
  the train split, so a SIM climb is partly in-sample fit, not proof of generalization.
- `tensorboard` package is **not** installed; `tensorboardX 2.6.4` writes the events, and
  `runs/<run>/train.log` carries `Step N | train/loss | ...`. Install `tensorboard` only if the
  dashboard is actually wanted.

## Training launch (prepared 2026-11-15, not yet started)
- **`keep_last_n_checkpoints: -1`** in both configs: all 12 checkpoints are kept (≈2.7 GB each,
  ~33 GB). Do not lower it — early checkpoints are the informative end of the ladder.
- **Training does NOT need the `mita` whitelist patch.** `OmniVoiceSampleProcessor` does
  `label.get("instruct", "None")` and wraps it directly; `_resolve_instruct` lives only in the
  generation path. Verified in `data/processor.py:80-106`.
- **webdataset cannot open `C:\...` paths.** `urlparse` sees scheme `c` → `no gopen handler
  defined`, on both the writer (tokenization) and the reader (training). Fix:
  `set GOPEN_REWRITE=([A-Za-z]):\\=file:///\1:/` plus `set WDS_SECURE=0` in every launcher.
  Without it training dies on the first shard read. The env var reaches DataLoader workers
  (spawn) and only rewrites `X:\` URLs — HF downloads/checkpoints are unaffected.
- **Detached launch pattern that survives:** `Start-Process cmd.exe /c <tmp.cmd> -WindowStyle
  Hidden` and let the `.cmd` self-redirect with `>> logs\x.log`. Do **not** use
  `-RedirectStandardOutput`: that forces CreateProcess without a new console and the child dies
  with the parent's console (observed after 15 s).
- **Tokenized shards are built and verified** (`data/finetune/shards/`): train 582 rows /
  4 shards / 23,703 tokens / 15.61 min; dev 30 rows / 5 shards / 1,310 tokens. Every shard's
  `npy` is `(8, N)` int16, `N == num_tokens`, `|num_tokens − round(dur·25)| ≤ 1`, key sets equal
  the jsonl ids. `data.lst` has 4 whitespace-free fields per line.
- Operator scripts (all with the **vendor** venv): `scripts/tokenize_shards.py`
  (`--calibrate|--split train|dev|all`), `scripts/check_checkpoint.py`
  (`--render|--export|--all`), `scripts/launch_train.py` (`--smoke|--full`, dry-run by default,
  `--yes`, `--resume`), `scripts/eval_ladder.py` (`--run-dir`, `--reuse`, `--self-test`).
  The launcher's preflight reads one real shard through `WebDatasetReader` — the check that
  catches a missing `GOPEN_REWRITE` before a 20-minute run.
- Known non-blocking gaps: `check_checkpoint --all` hits a `KeyError` on a *partially written*
  checkpoint (triage-only); `--resume` does not verify accelerate state and accelerate
  `load_state` skips missing files silently (restarts the optimizer while logging "Resumed from
  step N") — so never resume from an `--export` dir, and inspect `optimizer.bin` first.
- **Smoke run passed 2026-11-15.** Windows `DataLoader` uses spawn, so `builder.py`'s
  `length_fn=lambda s: s["length"]` (lines 207/250) is unpicklable — training died at
  `iter(dataloader)` with `Can't get local object 'build_dataloaders.<locals>.<lambda>'`.
  Fixed **without vendor edits** by `scripts/train_entry.py`: a subclass of
  `StreamLengthGroupDataset` that replaces any `<locals>` callable with a module-level
  `_length_fn`, patched onto `omnivoice.training.builder`; launched as
  `-m scripts.train_entry` with `PYTHONPATH=<project root>` (spawn children re-import the module
  to unpickle `_length_fn`). `scripts/__init__.py` makes the package importable.
  Result: 60 steps, `trainable params 26,886,144 / 639,463,424 = 4.20%` (higher than the docs'
  19.3M because `audio_embeddings`/`audio_heads` are trained fully), eval loss 4.334 → 4.252,
  both checkpoints `resume-capable: yes`, adapter 107.6 MB, render p05 = 2.80 s,
  5/5 wavs differ from `base_mita_tag` (adapter is not a no-op).
- **Wall time is the planning risk:** 60 steps took **~9 min**, not 2–5. Startup + 2 eval passes
  + 2 × 2.7 GB saves + an epoch-1 worker respawn dominate the short run; steady state after
  step 39 is ~1.5–2 s/step. Full 1200 steps (41 epochs) ≈ **45–75 min** if that pace holds,
  worst case 2–3 h if the disk/AV stalls seen at steps 34–38 recur. Do not promise 40 min.

## Soup + deployment (verified 2026-11-15)
- **`instruct="mita"` is NOT baked into a merged model.** It is a text prefix:
  `<|lang_start|>ru<|lang_end|><|instruct_start|>mita<|instruct_end|>` (`data/processor.py:105`,
  `models/omnivoice.py:1225`). `mita` tokenizes as `['mit','a']` — not a vocab token, not special.
  `_INSTRUCT_ALL_VALID` is a Python frozenset of 48 strings (`utils/voice_design.py:89`), not a
  weight; merge cannot change it. A merged dir therefore still needs the in-memory whitelist patch
  (`runtime.patch_instruct`) — use `generate.py --model <dir> --instruct mita`, never stock
  `omnivoice-infer`. Passing a different tag after merge gives non-Mita output.
- **`scripts/soup_lora.py`** — uniform mean of adapter tensors in float64 (394 tensors: 392 LoRA +
  `audio_embeddings`/`audio_heads`), `--steps >= 2`, `--out` default `<run-dir>\soups`. Rationale:
  same run, `A` drift 1–10% adjacent / ~20% across run, `B` carries ~85% of ΔdW. Canonical shas:
  `soup-246` (200+400+600) = `bcd84776…`, `soup-345` = `d75bcfce…`, both rebuild identically.
- **`scripts/merge_lora.py`** — dry-run by default, `--yes` executes; 11 preflight + 10 verify rows
  (key set == base, no `lora_` keys, `modules_to_save` **bit-identical to the adapter**, `q_proj`
  differs); `MERGE_IN_PROGRESS` marker is refused by `generate.py` and `run_eval --reuse-audio
  --model`; `merge_meta.json` written last. Upstream fp32 merge → ~3.2 GB dir; loaded as fp16, so
  audio is not bit-identical to the adapter path (expected).
- Current deployment artifact: **`models\` (root, not `models\<name>`) = merged `soup-345`**,
  sha256 `f3882ab46ce2…`, 2,450,344,144 B, 313 keys, no adapter keys, both `modules_to_save`
  tensors bit-identical to `runs\mita_v1\soups\soup-345`; no marker. Verified 2026-11-15.
- **Eval render is deterministic given the full recipe** — re-rendering `soup-246` `p01` at the
  same `num_step` reproduced `905e7214…` byte-for-byte. The earlier apparent "drift" was a
  **recipe mismatch**: `eval\run_eval.py` defaults to `num_step=32`, while
  `scripts\mita_voice\generate.py`'s default is **64** (file mtime 21:12), and the
  `soup-345`/`soup-3458` evals were run with `--num-step 64`. Never compare `sim_best` across
  variants rendered at different `num_step` (same artifact: 0.6679 @32 vs 0.6800 @64). Always
  record `generation.params.num_step` next to a number.
- **`scripts/run_mita.py`** — 74-line no-argument example: loads `models\`, applies
  `patch_instruct("mita")`, generates one phrase at 32 steps, writes `tmp\mita_example.wav`.
  Smoke-tested 2026-11-15 (2.20 s, rms 0.090, 24 kHz).

## Compact inference model (2026-11-15/16)
- **`models_compact/` = the deployable artifact.** `models/` is untouched (sha `f3882ab4…`).
  3,267,461,681 → 1,327,144,606 B (−59.4 %): LLM re-saved fp16 (2.45 → 1.225 GB) and the audio
  tokenizer pruned to its decode path (805 → 90.5 MB, 201.4M → 22.6M params). Built by
  `scripts/build_compact_model.py --src models --dst models_compact` (has `--verify`, 27 checks).
- **fp16 is provably lossless here, not an approximation**: `scripts/mita_voice/runtime.py`
  already loads with `dtype=torch.float16`, so saving `.half()` applies the same RNE cast the
  loader would. Gate: **20/20 wav sha256 identical**, WER pass, `sim_best_mean 0.665419`
  identical to 6 dp (`eval/results/merged_fp32.json` vs `merged_compact*.json`).
- **The audio tokenizer encoder is dead code on the Mita path** — `generate()` only builds a
  voice-clone prompt when `voice_clone_prompt is None`, and the Mita calls pass no ref audio
  (`scripts/mita_voice/generate.py:116-125`). Dropped: `semantic_model`, `acoustic_encoder`,
  `encoder_semantic`, `fc`, `decoder_semantic`, `fc1` (178.8M params). Kept: `acoustic_decoder.*`,
  `fc2.*`, `quantizer.*`. `decoder_semantic`/`fc1` are constructed but never referenced at all.
- **HF `from_pretrained` only WARNS on missing keys and random-initialises** — that is why the
  shim exists. Marker `audio_tokenizer/decode_only.json` lists the dropped keys; the runtime
  asserts `state_dict keys - shipped == drop.keys` and refuses the dir otherwise (`encode()` raises).
- **CPU-first tokenizer load is required, not an optimization**: the vendor passes
  `device_map=model.device` (`omnivoice.py:364-366`), so popping the encoder *after* load strands
  682 MiB in the CUDA caching allocator (`memory_reserved` stays 1954 MiB; `empty_cache()` cannot
  return it). `runtime._install_cpu_first_tokenizer` patches the module-global class, loads on CPU,
  strips, then `.to(device)`: **reserved 1954 → 1260 MiB, allocated 1256 MiB**. Non-compact dirs are
  a strict no-op. Do not "simplify" this back to post-load deletion.
- `models_compact` is **inference-only**: never a `merge_lora --base_model`, never a training base
  (it would upcast the fp16 masters and copy the pruned tokenizer + marker). `tokenize_shards.py`
  dies if handed `models_compact/audio_tokenizer`.
- **int8/int4 quantization is not available yet**: `torchao` and `bitsandbytes` are both absent
  from `vendor/OmniVoice/.venv` (torch 2.8.0+cu128). `torch.ao.quantization.quantize_dynamic`
  has no CUDA kernels. Any int8 attempt needs a wheel install + a blind A/B — `sim_best` cannot
  hear frying, and a flipped audio code is a click, not a word error.
- **Never hardlink `models_compact/**` into `tmp/` for tests.** A hardlink shares the inode, so
  writing the "copy" rewrites the live artifact. This actually happened: a review test flipped
  `decode_only.json` to `format_version: 2` through a hardlink and the runtime (correctly) refused
  to load — `RuntimeError: unsupported format_version 2 (this runtime understands 1)`. Use
  `shutil.copytree`; treat `models_compact/**` as read-only. Recovery: rebuild from `models/`
  (`build_compact_model.py --src models --dst models_compact --force`, deterministic) then
  `--verify` — 27 checks, weights come back `001394c1…` / `d134810801…`.
- Measured floor on this box: **~1.05 GB of the 8 GB is already used before torch loads**
  (Windows desktop/other processes), so `device_used` deltas are the honest number, not `nvidia-smi`.

## Open follow-ups
- **`scripts/extract_audio.py` has no argparse — running it re-extracts everything.** A
  documentation pass accidentally invoked it on 2026-11-15; `data/manifest.jsonl`,
  `data/audio/ru/**` and `data/extract_report.*` were rewritten. Proven content-neutral for the
  training corpus: the 612 `data/finetune/audio/*.ogg` copies (built earlier and sha-verified)
  still hash-match the rewritten sources 612/612, and the 612 texts still equal
  `manifest.jsonl.text`. Funnel still 705 → 612 → 582+30. `manifest_mita.jsonl` and
  `review_mita.md` untouched. Do not run it "just to check `--help`".
- `scripts/README.md` — the old 905-line operator reference was **deleted by the user on
  2026-11-22 12:01** (still in the Windows Recycle Bin). Replaced by a short root `README.md`
  (install → data → train → soup → merge → compact → run). Keep the root README in sync when a
  flag changes.
- Extraction failure gate is all-or-nothing: a category with 0 manifest rows but present oggs
  passes silently. Duration mapping also lacks a test. Both non-blocking, data verified good.

## Rename + environment state (2026-11-22)
- Project root renamed `C:\develop\mita_voice_clone` → `C:\develop\omnivoice-lora`. Every absolute
  reference in project-owned text files was rewritten (24 files / 2,141 occurrences: manifests,
  shard `data.lst`, `configs/data_config.json`, `eval/sim_refs.json`, `runs/mita_v1/*.json`,
  `models_compact/merge_meta.json`, logs). Pre-change copies: `tmp/pre_rename_backup.zip`.
- `.venv` and `vendor/**` were deliberately NOT rewritten (user instruction). Consequence: the
  vendor venv is currently **broken** — `_editable_impl_omnivoice.pth` and all 33
  `Scripts\*.exe` uv trampolines still embed the old path, so `import omnivoice` fails and
  `accelerate.exe` dies with "uv trampoline failed to canonicalize script path". Repair:
  re-run `setup.cmd` (root), which re-syncs both venvs and regenerates the shims.
- `setup.cmd` / `setup.sh` (root) = the same install/repair script for cmd and bash
  (Git Bash/MSYS2): project `.venv` (faster-whisper 1.2.1, num2words 0.5.14, four nvidia cu12
  DLL wheels) + `vendor/OmniVoice/.venv` (`uv sync --extra lora --inexact`, then
  `s3prl==0.4.18 --no-deps`) + import smoke checks. Idempotent; `setup.cmd` is CRLF,
  `setup.sh` is LF-only. `README.md` section 0 documents both.
- `models/` and all `runs/mita_v1/checkpoint-*` are deleted (disk freed); only `models_compact/`
  remains as the deployable artifact. `runs/mita_v1/ladder.json`, `soup_meta.json` and
  `merge_meta.json` still point at the removed checkpoints — historical metadata, expected.
- `scripts/run_mita.py` now loads `models_compact/` (not `models/`).
