# ADR-0001: Compressing the merged Mita checkpoint

- **Status:** Accepted · **Date:** 2026-11-15 · **Decided:** 2026-11-15
- **Decision:** ship **S1a + S4** — LLM re-saved fp16 and the audio tokenizer pruned to its decode
  path, with a CPU-first tokenizer load so the encoder never touches VRAM. Built artifact
  `models_compact/` (3.267 → 1.327 GB; 1976 → 1260 MiB of process VRAM; 20/20 eval wavs
  byte-identical, WER pass, `sim_best_mean` unchanged). **S2 (int8/int4) deferred** by the owner:
  1.26 GB on an 8 GB card is enough, and quantisation would need a listening A/B. **S3 (vocab
  prune) not done** — 9 % for the only stage that can force a retrain. **S0** replaced by the
  `merged_fp32` variant (`sim_best 0.665419` at the harness default 32 steps).
- **Scope:** `models/` (merged soup-345, 612.6M params) + `models/audio_tokenizer/` (201.4M params)
- **Related:** `MEMORY.md` (eval harness/baseline), `PLAN.md` §5

## Context

Asked: *"Can we compress this model — throw out what's not needed, e.g. everything not related to ru and en — so it takes less memory?"*

Measured on this box (probe `tmp/vram_probe.py`, throwaway):

| Quantity | Value |
|---|---|
| `models/` on disk | 2.462 GB (weights 2.450 GB + `tokenizer.json` 11.4 MB) |
| `models/audio_tokenizer/` on disk | 0.806 GB |
| **Total on disk** | **3.268 GB** |
| VRAM after load | **2.031 GB** = LLM 1.225 GB (fp16) + tokenizer **0.801 GB (fp32)** |
| VRAM peak, 2 s `generate()` | 2.068 GB (+37 MB) |
| Language handling | text, `omnivoice.py:1219-1228` |

**Deciding number: the model uses 25 % of the 8 GB card and peaks 37 MB above its weights. VRAM is not the constraint.** Disk (3.27 GB) and portability are the only real wins, so every option below is judged on bytes-per-risk.

## 1. Verdict on the literal ask

**"Delete everything not related to ru/en" is not a thing in this architecture.** The premise assumes per-language parameters; there are none:

- Text side is **Qwen3 byte-level BPE** (151,643 merges + 33 added tokens): one shared table, and `tie_word_embeddings: true` means `lm_head` *is* `llm.embed_tokens` — 155.3M params (25.4 %), one tensor, no separate `lm_head` in the safetensors.
- Language is **a string spliced into the prompt**, not a module: `style_text += f"<|lang_start|>{lang_str}<|lang_end|>"` (`omnivoice.py:1219-1228`). "ru" costs 3 tokens of context.
- The audio tokenizer is **language-agnostic**: HuBERT semantic encoder + DAC codec (`modeling_higgs_audio_v2_tokenizer.py:487-587`).

What *is* per-language deletable: **vocab rows ru/en text can never emit** — an alphabet closure, not a language module. Measured 57,147 of 151,676 rows (37.7 %) unreachable from {ASCII, Latin-1/Ext, Cyrillic, Greek, punctuation} = 58.5M params = **117 MB fp16 / 234 MB fp32**. See option B.

Honest framing: a 0.6B dense model with a 151k-tied embedding. The only compressible blocks are **(1) dtype** — 2.45 GB fp32 on disk that is already fp16 in VRAM, **(2) the embedding table**, **(3) the audio-tokenizer encoder**, which the target use case never runs.

## 2. Options

| # | Option | Deletes | Disk saved | VRAM saved | Risk | Effort | Retrain |
|---|---|---|---|---|---|---|---|
| **A** | fp32→fp16 re-save | redundant precision | **1.63 GB** (3.27→1.63) | 0.40 GB (tok only) | **Very low** | 1 script | no |
| **D** | Tokenizer decode-only split | `semantic_model`, `acoustic_encoder`, `encoder_semantic`, `fc`, dead `decoder_semantic`/`fc1` | 0.72 GB fp32 / 0.36 GB fp16 | 0.72 / 0.36 GB | Low (cached-prompt UX) | ~100 lines | no |
| **C** | int8 / int4 weight-only (torchao) | precision on LLM Linears | ~0.6 / ~1.0 GB | 0.61 / 0.92 GB | **Med-high** | script + listening | no |
| **B** | Vocab prune + id remap | 57.1k embedding rows + merges | 117 MB fp16 | 117 MB | Medium (id remap) | ~150 lines + QC | **only if QC fails** |
| **E** | Housekeeping | `chat_template.jinja`, dead merges, `max_position_embeddings` | ~5 MB | 0 | None | minutes | no |
| **F** | Layer prune / MoE / distill | — | — | — | **Rejected** | weeks | from scratch |

Ranked bytes-per-risk: **A ≫ D > E > C > B ≫ F**.

### A. fp32 → fp16 re-save — the only free win

The loader already casts (`runtime.py:58-60`, `dtype=torch.float16`), so fp32 on disk buys nothing; re-saving performs the *same* round-to-nearest cast once, at save time. **LLM output is bit-identical** — a file-format change, not a compression.

- **fp16 vs bf16:** **fp16 for the LLM** (reproduces the frozen baseline exactly; bf16 stacks a second numerics change on a compression and would also need `runtime.py:59` changed — do not mix). For the **audio tokenizer**, if ever cast, use **bf16**: the DAC decoder is ConvTranspose1d + snake, and bf16 keeps fp32's exponent range (no 65504 overflow) at 8 mantissa bits.
- **Config consistency:** `dtype: "float16"` in *both* the top level and `llm_config.dtype` (currently `"float32"` in both), same in `audio_tokenizer/config.json`; v5 prefers `dtype=` over the deprecated `torch_dtype=`. Then assert `next(m.parameters()).dtype is torch.float16` after load.
- **Nothing breaks on non-fp32 LLM weights.** The `_autocast_flex_attention` docstring (`omnivoice.py:92-113`) about q_norm/k_norm and RoPE promoting to fp32 is a *mixed-precision training* note — it needs fp32 master weights. Under `dtype=float16` the q_norm weights are fp16 too, and the flex path is not installed for SDPA (`omnivoice.py:295-299`).
- **The audio tokenizer is the exception and does break**, because it is loaded *without* a dtype hint (`omnivoice.py:364-366`) and stays fp32: (1) `encode` gets fp32 input (`omnivoice.py:820`, `torch.from_numpy(ref_wav)`) against fp16 weights → `RuntimeError: Input type (float) and bias type (Half)`; (2) `decode` returns fp16, then `omnivoice.py:852-865` calls `.cpu().numpy()` → `TypeError: Got unsupported ScalarType Half`. Both are fixable in a project-side wrapper (cast in, cast out, no vendor edit) — but that is why "add `dtype=float16`" is not a one-liner for the tokenizer.

### B. Vocab / embedding pruning

Build the drop set by **alphabet closure** over `tokenizer.json` (seed the 256 byte tokens, walk merges in order, keep a merge only if both parts are kept and the merged payload's chars are all allowed), remap ids densely, `embed[new] = old[perm]`, and rewrite `tokenizer.json` (vocab *and* merges), `config.json` (`vocab_size`, `pad/eos/bos_token_id`), `tokenizer_config.json` (`pad_token`, `eos_token`, `added_tokens_decoder`).

- **It is a row permutation plus compaction, so it should be exactly lossless — on one checkable condition.** Ids reach the model *only* as `embed_tokens` rows (head tied), so copying survivors and remapping yields an identical model, **no retrain**, *provided* re-tokenizing the corpus gives exactly `perm[old_ids]`. Gate below.
- Build the drop set by a **character-class heuristic** instead and that proof is gone: dropping a merge can change how the surviving list applies to rare words → tail-word tokenization shifts → fresh LoRA (~45–75 min) + full re-eval mandatory. **Three naive rules gave 3, 44,703 and 57,147 rows — a 19,000× spread.** The rule is the hard part; the script is not.
- Trap: ids 0–255 must always survive. They look non-Latin (`Ġ`, `Ċ`) but they are the byte alphabet; drop one and BPE fallback breaks for *every* unknown character.
- Trap: `audio_vocab_size: 1025`, `audio_mask_id: 1024`, `audio_embeddings`/`audio_heads` are a **separate id space** — never touched by the text remap.
- Bounded upside: all 2,689 corpus texts use **1,743 distinct ids (1.1 %)** — but pruning to that destroys open-vocabulary Russian. Corpus-driven pruning beyond the closure adds ≤ ~10M params. Not worth the risk.
- Saving is **117 MB at fp16 = 9 % of the model** for the most dangerous change here. Note `builder.py:113-116` loads the LLM in **fp32** for training and `builder.py:120-125` already resizes embeddings when `len(tokenizer)` changes, so a retrain of a pruned model needs no vendor edit.

### C. Weight-only quantization

- **int8** (torchao `Int8WeightOnlyConfig`, Linear-only) 612.6M → ~0.61 GB, usually transparent. **int4** (`Int4WeightOnlyConfig`, group 32) → ~0.31 GB + ~6 % scales ≈ 0.33 GB.
- **Not** `torch.ao.quantization.quantize_dynamic`: no quantized-Linear kernels on CUDA (fbgemm/qnnpack are CPU) — it would silently move the target to CPU.
- **bitsandbytes on Windows:** wheels exist but pinning against `torch 2.8.0+cu128` is fragile, needs an explicit load path (the vendor calls `super().from_pretrained`, `omnivoice.py:347`), and cannot share one weight between `embed_tokens` and a 4-bit `lm_head` — it would break the tie and duplicate 155M params. **Treat as unavailable.**
- **Quality risk is TTS-specific:** the head emits 8×1025 discrete codes per step and guidance runs at 2.0 (`eval/run_eval.py:60`), so logit error is amplified and a flipped code is an audible click, not a slightly wrong word. int8 is normally inaudible; int4 must be listened to. **`sim_best` measures timbre and will not catch frying.** Eval gates plus a blind A/B are mandatory; WER-only gating would pass a frying model.

### D. Audio-tokenizer surgery (best bytes-per-risk after A)

| Module | Params | Used by |
|---|---|---|
| `semantic_model` (HuBERT) | 94.4M | encode only (`:497-498`) |
| `acoustic_encoder` | 51.3M | encode only (`:547-549`) |
| `encoder_semantic` | 14.7M | encode only (`:542`) |
| `fc` | 1.05M | encode only (`:552`) |
| **`acoustic_decoder`** | 20.2M | **decode (`:582`)** |
| `decoder_semantic` | 16.5M | **nothing — never referenced** (`:463`) |
| `fc1` | 0.79M | **nothing** (`:466`) |
| `quantizer` | 2.11M | both (`:553`, `:580`) |
| `fc2` | 0.26M | decode (`:581`) |

<sup>[D]</sup> **Measured 2026-11-15 (RTX 3060 Ti, `models_compact/`):** the saving is in *live allocation* (1937.2 → 1254.7 MiB) and in the post-`decode` peak (2035.2 → 1955.2 MiB); the **load peak is identical** (1955.2 MiB both) because transformers random-initialises the missing modules on GPU before the runtime shim can delete them, and the allocator keeps those blocks reserved (1960 → 1954 MiB `memory_reserved`, non-releasable under expandable segments).

- Decode-only need = **22.6M**. Encoder side = 161.5M. Plus 17.3M of dead weight that goes regardless. Saving: **178.8M = 715 MB fp32 / 358 MB fp16**. Measured 2026-11-15: that 715 MB is *live* allocation; the load peak does not drop — see footnote [D].
- **Inference-safe?** `encode` is called only from `create_voice_clone_prompt` (`omnivoice.py:821`), which `generate` calls only when `voice_clone_prompt is None` (`omnivoice.py:1078-1092`). The Mita path is `generate(text, language="ru", instruct="mita")` with **no ref audio** (`scripts/mita_voice/generate.py:116-125`) → `encode` is dead code. Nothing else touches `semantic_model`.
- **UX consequence:** zero-shot cloning of a new voice is lost unless the reference is tokenized ahead of time into a cached `VoiceClonePrompt` (`ref_audio_tokens`, `:125-170`). Keep the encoder weights as a **cold spare outside the model dir** (~645 MB) — dropping them is reversible only if kept.
- **Silent-failure trap:** `from_pretrained` only *warns* on missing keys and leaves modules randomly initialized, so `encode` would return garbage instead of raising. Delete the modules, replace `encode` with a raiser, assert on `missing_keys`.

### E. Cheaper things that were missed

- `chat_template.jinja` (4.3 KB) is **unused** — the style prompt is string concatenation (`omnivoice.py:1219-1228`). Drop it. But `tokenizer_config.json` and `preprocessor_config.json` are required (`:350`, `:367-369`).
- `models/audio_tokenizer/` **cannot be deleted** even with cached prompts: if the subfolder is missing, `omnivoice.py:352-357` silently falls back to downloading 805 MB from HF.
- `codebook_layer_offsets` is an 8-element I64 buffer used at runtime (`:305-308`) — keep. `merge_meta.json` (2.1 KB) is the audit trail — keep.
- `max_position_embeddings: 40960` saves no bytes (it is an int) and bounds nothing: the theoretical fp16 KV cache at 40960 tokens is ~4.7 GB. Lowering it to 4096 is an OOM guard, not compression.

### F. Rejected

- **Layer/depth pruning.** 28 layers = 440M = 881 MB fp16; dropping 25 % saves 220 MB and perturbs an iterative codec decoder whose per-layer contribution WER and SIM cannot measure. Recovery needs a full retrain — not worth 220 MB.
- **MoE-ification.** More total params, pays off only at batch > 1; this is single-user single-stream. Strictly slower here.
- **Distillation.** Needs a teacher forward, a large corpus and training headroom this 8 GB box lacks, and it degrades exactly the timbre/prosody the LoRA produced.
- **GGUF / llama.cpp / GPTQ / AWQ.** Not applicable: `generate` is a custom iterative + chunked sampler (`omnivoice.py:704-714`, `:849-865`) over `audio_embeddings`/`audio_heads`; no runtime supports this graph and there is no calibration data for the 8×1025 head.

## 3. Recommended sequence

Every stage writes a **new directory**; `models/` is never modified, so rollback is `rm -rf`.

| Stage | Action | Artifact | VRAM | Gate | Independent? |
|---|---|---|---|---|---|
| **S0** | Measure the merged dir as its own variant | — | 2.031 GB | `--variant merged_mita --model models` must reproduce soup-345: `wer_verdict: pass`, `sim_best_mean 0.680 ± 0.054`; record wav sha256 + VRAM probe | prerequisite |
| **S1a** | Re-save **LLM only** to fp16 | 1.225 GB | 1.225 GB | **audio bit-identical** (20 wav sha256). If not, stop — the cast is not where you think | yes |
| **S1b** | Tokenizer to bf16 + cast wrapper | 0.403 GB | −0.40 GB | not bit-identical by construction: WER veto + `sim_best` Δ<0.05 + listen | yes |
| **S2** | torchao int8 (then int4) on LLM Linears | 0.61 / 0.33 GB | ~1.0 / 0.7 GB | WER veto + `sim_best` Δ<0.05 + **blind A/B vs S0**; watch the p17 short-clip flag | yes |
| **S3** | Vocab prune + id remap (closure rule) | −117 MB | −0.12 GB | re-tokenize all 2,689 texts + 20 phrases → ids must equal `perm[old]`, then bit-identical audio; else retrain + full eval | independent |
| **S4** | Tokenizer decode-only + dead modules | −0.36 GB bf16 | −0.36 GB | push one frozen code tensor through `decode`, compare wav sha256 vs S0 — zero-variance test | independent |

```bash
# S0 — baseline of the merged dir (also records model_sha256/model_bytes)
python eval/run_eval.py --variant merged_mita --instruct mita --model models
# S1a / S1b — scripts proposed by this ADR, they do not exist yet
python tools/compress/resave.py --src models --dst models_fp16 --dtype float16 --parts llm
python tools/compress/resave.py --src models --dst models_bf16 --dtype bfloat16 --parts audio_tokenizer
python eval/run_eval.py --variant merged_fp16 --instruct mita --model models_fp16
# S2 / S3 / S4
python tools/compress/quantize.py --src models_fp16 --dst models_int8 --scheme int8-weight-only
python tools/compress/prune_vocab.py --src models_fp16 --dst models_ru_en --closure alphabet --verify-corpus
python tools/compress/split_tokenizer.py --src models_fp16 --dst models_tiny --keep decode-only
```

S1a and S1b are ~30-line scripts. **S0 + S1a fit in an hour and capture 75 % of the available saving with a bit-identical-output proof.** S3 is the only stage that can cost a training run, and it saves the least.

Remap pseudo-code (the part that must not be improvised):

```
keep  = closure(tokenizer.json, alphabet={ascii, latin1ext, cyrillic, greek, punct})
assert set(range(256)) <= keep and added_tokens <= keep
rows  = sorted(keep); perm = {old: new for new, old in enumerate(rows)}
embed = W["llm.embed_tokens.weight"][rows]        # (len(rows), 1024)
cfg["llm_config"]["vocab_size"] = len(rows)
cfg["pad_token_id"] = perm[cfg["pad_token_id"]]; cfg["eos_token_id"] = perm[cfg["eos_token_id"]]
# never touch: audio_vocab_size, audio_mask_id, audio_embeddings, audio_heads
for t in corpus_texts + eval_phrases:             # the gate
    assert tok_new(t) == [perm[i] for i in tok_old(t)]
```

## 4. Risk register

| Risk | Where it bites | Detection |
|---|---|---|
| **Prosody / frying passes every automatic metric** | int4, any tokenizer numerics change | `sim_best` measures timbre only — blind A/B vs S0 is the only catch |
| **Short-clip truncation** (early EOS) | quantized or pruned model | p17 already flags `too_fast` in `base_native`; gate on per-phrase duration ratio vs S0 |
| **Silent random-init modules** | any partial state-dict load (warns, never raises) | assert `missing_keys == expected`; stub `encode` to raise |
| **`added_tokens` id drift** | pad/eos/bos remapped in `config.json` but not `tokenizer_config.json` (or vice versa) → wrong stop token, silent truncation | corpus re-tokenization gate; assert `tok.pad_token_id == config.pad_token_id` |
| **Audio id-space collision** | remap shifting `audio_mask_id`/`audio_vocab_size` | separate config keys; assert unchanged |
| **fp16 overflow in the DAC decoder** | transposed-conv stack, wide dynamic range | bf16 for the tokenizer, never fp16; decode-sha test |
| **`.numpy()` on half tensors** | `omnivoice.py:852-865` with an fp16 tokenizer | `.float()` in the wrapper; fails loudly, so low risk |
| **Missing `audio_tokenizer/` re-downloads 805 MB** | `omnivoice.py:352-357` | assert the resolved path is local, log its size |
| **RU normalisation masks regressions** (ё, numbers, `[player]`) | `eval/wer.py` normalises both sides, so a dropped `ё` hides behind it | keep `ё` in the alphabet; `ё`-only smoke string before/after |
| **Eval set saturation** | 19/20 phrases already 0.00 % WER — WER can only veto | keep it as a gate; rank on `sim_best` + listening |
| **GGUF / ollama / LM Studio expectations** | users assume any HF dir is portable | state plainly: this graph is not exportable |

## 5. Open questions for the human

1. **What is the real target — disk, VRAM, load time, or a weaker machine?** Measured: 2.03 GB VRAM (25 % of this card), peak 2.07 GB, 3.27 GB disk, ~4 s load. Nothing is under pressure, so "make it smaller" needs a destination: 2 GB laptop, CPU box, 4 GB card? That picks the stage — S1 for disk, S2 for a weaker GPU, S2+S4 for CPU-ish.
2. **Acceptance threshold?** Bit-identical (only S1a promises it), Δ`sim_best` < 0.05 with no audible change in a blind A/B, or "sounds fine to me"?
3. **Is zero-shot cloning to an unseen voice still required?** Yes → S4 costs it (or a 645 MB cold spare). No → S4 is the second-best win here.
4. **Is a 45–75 min retrain acceptable** if the S3 gate fails? If not, skip S3 — it is 9 %.
5. **Is fp32 kept as an archive** or is the fp16 dir the new master? The fp32 source is reproducible from `merge_meta.json` + the adapter, so it is safe to drop.

## 6. Recommendation

Do **S0 then S1a** now: a 30-line re-save that removes 1.2 GB the loader was already discarding on every load, and it is provable — the 20 wavs must hash-match, because fp32→fp16 at save time is the same cast the loader already performs. Then **S4** if footprint still matters: the encoder half of the audio tokenizer (161.5M params, 645 MB) is dead code for `instruct="mita"`, and the decode path can be validated by hashing one waveform. **S2 (int8)** is the only option that changes what hardware can run this, so it is right if "less memory" means "a weaker machine" — but it needs a listening test, not just the harness, because these metrics cannot hear frying. **S3** buys 9 % and is the only stage that can force a retrain; I would not do it, and I would not do F at all. What I would *not* do is compress against a premise ("remove the non-ru/en parts") that neither the tokenizer nor the codec supports.

**If you only do one thing:** re-save `models/` to fp16 (LLM only, `dtype` updated in both config places), verify the 20 wavs are bit-identical, delete the fp32 dir — 3.27 → 2.07 GB on disk, zero quality risk, no retraining, ~30 minutes.

## Appendix — verified vs. assumed

Verified here: per-module param counts (safetensors), fp32 audio tokenizer in VRAM, 2.031/2.068 GB VRAM, 3.268 GB disk, 256 byte tokens at ids 0-255, 1,743 distinct ids across the 2,689-text corpus, the 715 MB encoder saving. Assumed/inherited: that `models/` behaves identically to the `soup-345` adapter (**S0 measures it — if it fails, every delta is against the wrong reference**); the brief's 44,703 "non-Latin script" vocab figure (my closure rule gives 57,147, a byte-decoding rule gives 3 — the ADR uses the closure number and treats the spread as a finding); 1 GB = 10⁹ B.
