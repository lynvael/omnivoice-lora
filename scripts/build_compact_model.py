r"""Build a compact deploy dir: fp16 LLM + decode-only fp32 audio tokenizer.

Usage (vendor venv):
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\build_compact_model.py --src models --dst models_compact
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\build_compact_model.py --src models --dst models_compact --force
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\build_compact_model.py --src models --dst models_compact --verify

Implements ADR-0001 options A + D, and nothing else:

* ``model.safetensors`` — every key kept, every float tensor cast F32 -> F16
  (the same round-to-nearest cast ``runtime.load_model`` already performs at
  load time with ``dtype=torch.float16``), non-float tensors (the I64 buffer
  ``codebook_layer_offsets``) kept as-is.  Refuses to write if any source
  tensor is non-finite or has ``max|w| >= 60000`` (fp16 ceiling is 65504).
* ``audio_tokenizer/model.safetensors`` — keep only ``acoustic_decoder.*``,
  ``fc2.*``, ``quantizer.*`` (+ ``codebook_layer_offsets`` if it were in this
  file; it is not — it belongs to the LLM, see note below).  Nothing is cast:
  the decode path stays bit-identical fp32.  A ``decode_only.json`` marker
  records the dropped keys, so ``scripts/mita_voice/runtime.py`` can delete
  the encoder modules after ``from_pretrained`` instead of letting
  transformers silently random-init them.
* ``config.json`` — only ``dtype`` (top level and ``llm_config``) -> "float16".
  ``audio_tokenizer/config.json`` is copied byte-for-byte.

The build is written into ``<dst>.partial`` and renamed onto ``--dst`` only
after verification passes, so an interrupted run can never leave a
half-written directory that looks like a finished model.  ``--src`` is only
ever read: a pre/post size+mtime manifest proves it.

``--verify`` re-checks an existing ``--dst`` against ``--src`` (full tensor
comparison) and writes nothing.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "OmniVoice"
VENV_PY = VENDOR / ".venv" / "Scripts" / "python.exe"

PARTIAL_SUFFIX = ".partial"
OLD_SUFFIX = ".old"  # previous artifact, renamed aside during commit
LLM_WEIGHTS = "model.safetensors"
AUDIO_DIR = "audio_tokenizer"
AUDIO_WEIGHTS = "audio_tokenizer/model.safetensors"
DECODE_MARKER = "audio_tokenizer/decode_only.json"
COMPACT_META = "compact_meta.json"

# --- policy (ADR-0001 A + D) -------------------------------------------------
LLM_TARGET_DTYPE = "float16"
FP16_MAX_ABS = 60000.0  # fp16 tops out at 65504; a weight above this is a bug

AUDIO_KEEP_PREFIXES = ("acoustic_decoder.", "fc2.", "quantizer.")
AUDIO_EXTRA_KEEP_KEYS = ("codebook_layer_offsets",)  # buffer; not in this file
AUDIO_DROP_PREFIXES = (
    "semantic_model.",
    "acoustic_encoder.",
    "encoder_semantic.",
    "fc.",
    "fc1.",
    "decoder_semantic.",
)

COPY_TOP = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merge_meta.json")
COPY_AUDIO = ("config.json", "preprocessor_config.json", "README.md", "LICENSE", ".gitattributes")

# measured on models/ 2026-11-15; recorded, and a mismatch is a loud warning
REF_AUDIO_KEEP_PARAMS = 22_614_569
REF_AUDIO_DROP_PARAMS = 178_785_984
REF_LLM_F32_PARAMS = 612_577_280
DISK_HEADROOM = 1.20


def die(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def fmt_bytes(n: int) -> str:
    return f"{n:,} B"


def fmt_gb(n: int) -> str:
    return f"{n / 1e9:.3f} GB"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())


def rel_files(d: Path) -> dict[str, int]:
    return {p.relative_to(d).as_posix(): p.stat().st_size for p in d.rglob("*") if p.is_file()}


def manifest(d: Path) -> dict[str, tuple[int, int]]:
    """{relpath: (bytes, mtime_ns)} — cheap proof that --src was not written."""
    return {
        p.relative_to(d).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in d.rglob("*")
        if p.is_file()
    }


def free_bytes(p: Path) -> int:
    q = p
    while not q.exists() and q.parent != q:
        q = q.parent
    return shutil.disk_usage(str(q)).free


def overlaps(a: Path, b: Path) -> bool:
    return a == b or b in a.parents or a in b.parents


def audio_key_kept(key: str) -> bool:
    return key in AUDIO_EXTRA_KEEP_KEYS or key.startswith(AUDIO_KEEP_PREFIXES)


def audio_key_dropped(key: str) -> bool:
    return key.startswith(AUDIO_DROP_PREFIXES)


def group_by_top(key: str) -> str:
    return key.split(".", 1)[0]


def tensor_numel(t) -> int:
    n = 1
    for s in t.shape:
        n *= int(s)
    return n


# ---------------------------------------------------------------- preflight
def preflight(args) -> dict:
    """Print the PASS/FAIL table; die on the first FAIL. Returns the run context."""

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46} {detail}")
        if not ok:
            die(f"{name}: {detail}")

    try:
        import safetensors  # noqa: F401
        import torch

        have_torch = True
        py_detail = f"{sys.executable} (torch {torch.__version__}, safetensors {safetensors.__version__})"
    except Exception as e:  # pragma: no cover - wrong interpreter
        have_torch = False
        py_detail = f"{sys.executable}: {type(e).__name__}: {e}"
    check("interpreter has torch+safetensors", have_torch, py_detail)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    check("src dir", src.is_dir(), str(src))
    check("src/dst do not overlap", not overlaps(src, dst),
          f"src={src} dst={dst}" + (" (dst would be written inside src)" if overlaps(src, dst) else ""))

    required = [LLM_WEIGHTS, "config.json", *COPY_TOP, AUDIO_WEIGHTS,
                *(f"{AUDIO_DIR}/{n}" for n in COPY_AUDIO)]
    missing = [rel for rel in required if not (src / rel).is_file()]
    check("src file set", not missing,
          f"{len(required)} required" + (f"; MISSING: {missing}" if missing else ""))

    empty = not dst.exists() or not any(dst.iterdir())
    if args.verify:
        check("dst dir exists (--verify)", dst.is_dir(),
              str(dst) + ("" if dst.is_dir() else " — nothing to verify"))
    elif empty:
        state = "does not exist" if not dst.exists() else "empty"
        check("dst empty (or --force)", True, state)
    elif args.force:
        check("dst empty (or --force)", True, "non-empty, allowed by --force")
    else:
        check("dst empty (or --force)", False,
              f"non-empty ({sorted(p.name for p in dst.iterdir())[:5]}) — use --force")

    partial = dst.with_name(dst.name + PARTIAL_SUFFIX)
    if args.verify:
        check("partial dir absent (--verify)", True, "not checked")
    elif partial.exists():
        state = f"{partial} (stale partial build)"
        if not args.force:
            check("partial dir absent", False, state + " — use --force")
        check("partial dir absent", True, state + " — will be deleted")
    else:
        check("partial dir absent", True, str(partial))

    llm_bytes = (src / LLM_WEIGHTS).stat().st_size
    audio_bytes = (src / AUDIO_WEIGHTS).stat().st_size
    copies = sum((src / rel).stat().st_size for rel in COPY_TOP)
    need = int((llm_bytes / 2 + audio_bytes * 0.15 + copies) * DISK_HEADROOM)
    free = free_bytes(dst)
    check("free disk >= estimate + 20%", free >= need,
          f"free {fmt_gb(free)}, need {fmt_gb(need)}")

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    bad = []
    if cfg.get("dtype") != "float32":
        bad.append(f"dtype={cfg.get('dtype')!r}")
    if (cfg.get("llm_config") or {}).get("dtype") != "float32":
        bad.append(f"llm_config.dtype={(cfg.get('llm_config') or {}).get('dtype')!r}")
    check("src config dtype == float32", not bad, "both places" if not bad else f"unexpected: {bad}")

    src_manifest = manifest(src)
    print(f"  [INFO] src {src}: {len(src_manifest)} files, {fmt_bytes(dir_bytes(src))} ({fmt_gb(dir_bytes(src))})")

    return {
        "src": src,
        "dst": dst,
        "partial": partial,
        "src_manifest": src_manifest,
    }


# ------------------------------------------------------------------- build
def write_llm_weights(src_w: Path, dst_w: Path) -> dict:
    """Re-save every key; cast float tensors to fp16, keep everything else."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out = {}
    stats = {
        "n_keys": 0,
        "n_float": 0,
        "n_nonfloat": 0,
        "params_f32": 0,
        "params_other": 0,
        "max_abs_f32": 0.0,
        "nonfloat_dtypes": {},
    }
    with safe_open(str(src_w), framework="pt", device="cpu") as f:
        meta = f.metadata()
        keys = list(f.keys())
        total = len(keys)
        for i, k in enumerate(keys, 1):
            t = f.get_tensor(k)
            stats["n_keys"] += 1
            if t.is_floating_point():
                n = t.numel()
                stats["n_float"] += 1
                stats["params_f32"] += n
                if n and not torch.isfinite(t).all():
                    die(f"non-finite values in {k} ({src_w}) — refusing to cast")
                mx = float(t.abs().max()) if n else 0.0
                stats["max_abs_f32"] = max(stats["max_abs_f32"], mx)
                if mx >= FP16_MAX_ABS:
                    die(f"max|w| = {mx:.1f} in {k} >= {FP16_MAX_ABS:.0f} — fp16 would overflow")
                out[k] = t.to(torch.float16)
            else:
                stats["n_nonfloat"] += 1
                stats["params_other"] += t.numel()
                dt = str(t.dtype).replace("torch.", "")
                stats["nonfloat_dtypes"][dt] = stats["nonfloat_dtypes"].get(dt, 0) + 1
                out[k] = t.contiguous()
            del t
            if i % 50 == 0 or i == total:
                print(f"  [llm] {i:>4}/{total} tensors  {stats['max_abs_f32']:.3f} max|w|", flush=True)

    save_file(out, str(dst_w), metadata=meta or {"format": "pt"})
    out.clear()
    print(f"  [llm] wrote {dst_w} ({fmt_bytes(dst_w.stat().st_size)})")
    return stats


def write_audio_weights(src_w: Path, dst_w: Path) -> dict:
    """Keep the decode path only; no dtype change at all."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out = {}
    kept, dropped = [], []
    by_prefix: dict[str, dict[str, int]] = {}
    keep_params = 0
    with safe_open(str(src_w), framework="pt", device="cpu") as f:
        meta = f.metadata()
        keys = list(f.keys())
        for k in keys:
            if audio_key_kept(k):
                kept.append(k)
            elif audio_key_dropped(k):
                dropped.append(k)
            else:
                die(f"unclassified audio-tokenizer key {k!r} — refusing to guess (add a prefix to the policy)")
        for i, k in enumerate(kept, 1):
            t = f.get_tensor(k)
            if str(t.dtype) != "torch.float32":
                die(f"{k}: expected float32, got {t.dtype} — the decode path must stay fp32")
            if not torch.isfinite(t).all():
                die(f"non-finite values in {k}")
            out[k] = t.contiguous()
            keep_params += t.numel()
            del t
            if i % 50 == 0 or i == len(kept):
                print(f"  [audio] {i:>4}/{len(kept)} kept tensors", flush=True)
        for k in dropped:
            sh = f.get_slice(k)
            n = 1
            for s in sh.get_shape():
                n *= int(s)
            g = by_prefix.setdefault(group_by_top(k), {"keys": 0, "params": 0})
            g["keys"] += 1
            g["params"] += n

    save_file(out, str(dst_w), metadata=meta or {"format": "pt"})
    out.clear()
    drop_params = sum(g["params"] for g in by_prefix.values())
    print(f"  [audio] wrote {dst_w} ({fmt_bytes(dst_w.stat().st_size)}), "
          f"kept {len(kept)} keys / {keep_params:,} params, "
          f"dropped {len(dropped)} keys / {drop_params:,} params")
    return {
        "kept": kept,
        "dropped": dropped,
        "keep_params": keep_params,
        "drop_params": drop_params,
        "by_prefix": by_prefix,
        "n_src_keys": len(keys),
    }


def rewrite_config(src_cfg: Path, dst_cfg: Path) -> dict:
    """Change exactly two dtype values; every other byte of the JSON is kept."""
    raw = src_cfg.read_bytes()
    text = raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"  # keep the source's line endings
    cfg = json.loads(text)
    cfg["dtype"] = LLM_TARGET_DTYPE
    cfg["llm_config"]["dtype"] = LLM_TARGET_DTYPE
    new_text = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
    dst_cfg.write_text(new_text, encoding="utf-8", newline=newline)

    old_lines, new_lines = text.splitlines(), new_text.splitlines()
    if len(old_lines) != len(new_lines):
        die(f"config.json: rewriting changed the line count {len(old_lines)} -> {len(new_lines)}")
    changed = [(i + 1, o.strip(), n.strip())
               for i, (o, n) in enumerate(zip(old_lines, new_lines)) if o != n]
    if len(changed) != 2 or any(n != '"dtype": "float16",' for _, _, n in changed):
        die(f"config.json: expected exactly 2 dtype edits, got {changed}")
    # byte-level: the file must be the source with `float32` -> `float16` twice, nothing else
    want = raw.replace(b'"dtype": "float32"', b'"dtype": "float16"')
    if dst_cfg.read_bytes() != want:
        die("config.json: output is not the source with exactly the 2 dtype values replaced")
    print(f"  [config] 2 lines changed: " + "; ".join(f"{i}:{o} -> {n}" for i, o, n in changed)
          + f" ({len(raw):,} B -> {len(want):,} B, newline={newline!r})")
    return {"changed_lines": changed, "src_bytes": len(raw), "dst_bytes": len(want)}


def write_marker(dst: Path, src: Path, audio: dict) -> None:
    marker = dst / DECODE_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    dst_w = dst / AUDIO_WEIGHTS
    doc = {
        "format_version": 1,
        "kind": "higgs_audio_v2_tokenizer_decode_only",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dst_sha256": sha256_file(dst_w),
        "dst_bytes": dst_w.stat().st_size,
        "source": {
            "path": AUDIO_WEIGHTS,
            "sha256": sha256_file(src / AUDIO_WEIGHTS),
            "bytes": (src / AUDIO_WEIGHTS).stat().st_size,
            "n_keys": audio["n_src_keys"],
            "params": audio["keep_params"] + audio["drop_params"],
        },
        "keep": {
            "prefixes": list(AUDIO_KEEP_PREFIXES),
            "extra_keys": list(AUDIO_EXTRA_KEEP_KEYS),
            "extra_keys_present": [k for k in AUDIO_EXTRA_KEEP_KEYS if k in audio["kept"]],
            "n_keys": len(audio["kept"]),
            "params": audio["keep_params"],
            "dtype": "float32 unchanged (decode must stay bit-identical)",
        },
        "drop": {
            "prefixes": list(AUDIO_DROP_PREFIXES),
            "n_keys": len(audio["dropped"]),
            "params": audio["drop_params"],
            "by_prefix": audio["by_prefix"],
            "keys": audio["dropped"],
        },
        "reason": (
            "encode() is dead code for text-only generation (instruct='mita', no ref_audio): "
            "OmniVoice.create_voice_clone_prompt is the only caller and generate() skips it when "
            "voice_clone_prompt is None. decode() needs only fc2 + quantizer + acoustic_decoder. "
            "decoder_semantic and fc1 are constructed but referenced by no forward path."
        ),
        "runtime": (
            "scripts/mita_voice/runtime.py strips these modules from loaded.audio_tokenizer and "
            "replaces encode() with a RuntimeError raiser when this file exists. Required because "
            "HiggsAudioV2TokenizerModel.from_pretrained only WARNS on missing weights and leaves "
            "them randomly initialised, which would run garbage and hold ~645 MB of VRAM."
        ),
        "note": (
            "codebook_layer_offsets is NOT in this file: it is an I64 buffer of the top-level "
            "OmniVoice model and lives in ../model.safetensors, where it is preserved unchanged."
        ),
    }
    marker.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"  [marker] {marker} ({len(audio['dropped'])} dropped keys recorded)")


def write_compact_meta(dst: Path, src: Path) -> dict:
    """Provenance for the compact dir: what it was built from and what it is.

    Deliberately separate from ``merge_meta.json`` (which is a byte copy of the
    upstream merge record and describes ``models/``, not this directory).
    """

    def entry(p: Path) -> dict:
        return {"sha256": sha256_file(p), "bytes": p.stat().st_size}

    doc = {
        "format_version": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "scripts/build_compact_model.py",
        "built_from": {
            LLM_WEIGHTS: entry(src / LLM_WEIGHTS),
            AUDIO_WEIGHTS: entry(src / AUDIO_WEIGHTS),
        },
        "policy": {
            "llm_dtype": LLM_TARGET_DTYPE,
            "keep_prefixes": list(AUDIO_KEEP_PREFIXES),
            "drop_prefixes": list(AUDIO_DROP_PREFIXES),
        },
        "output": {
            LLM_WEIGHTS: entry(dst / LLM_WEIGHTS),
            AUDIO_WEIGHTS: entry(dst / AUDIO_WEIGHTS),
        },
        "note": (
            "merge_meta.json here is a byte copy of the upstream merge record: its 'output' "
            "block describes models/ (the fp32 source), NOT this directory. Use this file for "
            "compact-dir provenance. Inference-only artifact: never use it as a --base_model "
            "for merge_lora or as a training base (the audio-tokenizer encoder is missing)."
        ),
    }
    path = dst / COMPACT_META
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"  [meta] {path}")
    return doc


def build(ctx: dict) -> dict:
    src, out = ctx["src"], ctx["partial"]
    if out.exists():
        shutil.rmtree(out)
    (out / AUDIO_DIR).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("\nbuild:")
    llm = write_llm_weights(src / LLM_WEIGHTS, out / LLM_WEIGHTS)
    audio = write_audio_weights(src / AUDIO_WEIGHTS, out / AUDIO_WEIGHTS)
    cfg = rewrite_config(src / "config.json", out / "config.json")

    for rel in COPY_TOP:
        shutil.copyfile(src / rel, out / rel)
    for rel in COPY_AUDIO:
        shutil.copyfile(src / AUDIO_DIR / rel, out / AUDIO_DIR / rel)
    print(f"  [copy] {len(COPY_TOP) + len(COPY_AUDIO)} files copied byte-for-byte "
          "(audio_tokenizer/config.json included unchanged)")
    # the LLM config is rewritten, not copied: put it back on the copy list for verify
    ctx["copied"] = [r for r in COPY_TOP if r != "config.json"] + [f"{AUDIO_DIR}/{r}" for r in COPY_AUDIO]
    write_marker(out, src, audio)
    meta = write_compact_meta(out, src)
    print(f"build wall {time.time() - t0:.1f}s")
    return {"llm": llm, "audio": audio, "config": cfg, "compact_meta": meta}


# ------------------------------------------------------------------ verify
DECODE_GATE_FRAMES = 50
DECODE_GATE_CODEBOOKS = 8


def decode_gate(src: Path, dst: Path, marker: dict) -> dict:
    """ADR-0001 S4: run both tokenizers and compare the decoded waveform.

    Deliberately *not* a key-set argument: the destination tokenizer is built
    by transformers from its own config and actually executes ``decode()``,
    so a wrong fp32 weight, a mis-shaped module or a bad config would show up
    here.  Also re-runs after popping the marker's dropped modules, i.e. the
    exact state ``runtime._strip_audio_tokenizer_encoder`` leaves behind.

    CPU-only and deterministic; ~10 s, dominated by loading the 805 MB source
    tokenizer.

    The codes are deliberately different per codebook (``(7·i + 3) % 1024``
    over the flattened (codebook, frame) grid): RVQ decode sums each
    codebook's contribution, so if every codebook were fed the same ids, a
    swap of two codebooks' weights would pass vacuously (the sum is
    unchanged), and only embedding rows 0–49 would ever be exercised.
    """
    import torch
    from transformers import HiggsAudioV2TokenizerModel
    from transformers.utils import logging as hf_logging

    codes = (
        (torch.arange(DECODE_GATE_FRAMES * DECODE_GATE_CODEBOOKS)
         .view(1, DECODE_GATE_CODEBOOKS, DECODE_GATE_FRAMES) * 7 + 3) % 1024
    ).to(torch.long)

    def run(tok):
        with torch.no_grad():
            wav = tok.decode(codes.clone()).audio_values.detach().float().cpu().reshape(-1)
        return hashlib.sha256(wav.numpy().astype("float32").tobytes()).hexdigest(), wav

    prev = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()  # the MISSING keys are asserted by the audio key-set row above
    try:
        src_tok = HiggsAudioV2TokenizerModel.from_pretrained(str(src / AUDIO_DIR), device_map="cpu")
        dst_tok = HiggsAudioV2TokenizerModel.from_pretrained(str(dst / AUDIO_DIR), device_map="cpu")
        src_tok.eval()
        dst_tok.eval()
        src_sha, src_wav = run(src_tok)
        dst_sha, dst_wav = run(dst_tok)
        fresh_equal = torch.equal(src_wav, dst_wav)
        for prefix in marker["drop"]["prefixes"]:
            dst_tok._modules.pop(str(prefix).split(".", 1)[0], None)
        pruned_sha, pruned_wav = run(dst_tok)
        pruned_equal = torch.equal(src_wav, pruned_wav)
    finally:
        hf_logging.set_verbosity(prev)
    return {
        "src_sha256": src_sha,
        "dst_sha256": dst_sha,
        "pruned_sha256": pruned_sha,
        "samples": int(src_wav.numel()),
        "fresh_equal": fresh_equal,
        "pruned_equal": pruned_equal,
    }


def verify(src: Path, dst: Path, ctx: dict | None = None) -> dict:
    """Full structural + numerical verification; die on the first FAIL."""
    import torch
    from safetensors import safe_open

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46} {detail}")
        if not ok:
            die(f"verify failed: {name}: {detail}")

    def keys_of(p: Path) -> list[str]:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            return list(f.keys())

    print("\nverify:")
    # 1. file set exactly the policy set
    expected = set(COPY_TOP) | {"config.json", LLM_WEIGHTS, AUDIO_WEIGHTS, DECODE_MARKER,
                                COMPACT_META} | {f"{AUDIO_DIR}/{n}" for n in COPY_AUDIO}
    got = set(rel_files(dst))
    extra, missing = sorted(got - expected), sorted(expected - got)
    check("file set == policy", not extra and not missing,
          f"{len(got)} files" + (f"; MISSING={missing} EXTRA={extra}" if (missing or extra) else ""))

    # 2-5. LLM weights
    src_w, dst_w = src / LLM_WEIGHTS, dst / LLM_WEIGHTS
    src_keys, dst_keys = keys_of(src_w), keys_of(dst_w)
    check("LLM key set == src", src_keys == dst_keys,
          f"{len(dst_keys)} keys (src {len(src_keys)})"
          + ("" if src_keys == dst_keys else f"; sym-diff {sorted(set(src_keys) ^ set(dst_keys))[:5]}"))

    bad_dtype, n_f16, n_other = [], 0, 0
    with safe_open(str(dst_w), framework="pt", device="cpu") as f:
        for k in f.keys():
            dt = str(f.get_slice(k).get_dtype())
            if dt == "F16":
                n_f16 += 1
            else:
                n_other += 1
                if dt != "I64":
                    bad_dtype.append(f"{k}:{dt}")
    check("LLM dtypes F16 (+ I64 buffer)", not bad_dtype,
          f"{n_f16} F16, {n_other} non-float" + (f"; unexpected {bad_dtype[:5]}" if bad_dtype else ""))

    max_abs = 0.0
    max_err = 0.0
    n_cmp = 0
    n_float_params = 0
    bad = []
    with safe_open(str(src_w), framework="pt", device="cpu") as fs, \
            safe_open(str(dst_w), framework="pt", device="cpu") as fd:
        for k in src_keys:
            a, b = fs.get_tensor(k), fd.get_tensor(k)
            if a.shape != b.shape:
                bad.append(f"{k} shape {tuple(a.shape)} -> {tuple(b.shape)}")
                continue
            if a.is_floating_point():
                n_float_params += a.numel()
                want = a.to(torch.float16)
                if not torch.equal(want, b):
                    n = int((want != b).sum())
                    bad.append(f"{k}: {n} elements differ from fp16(src)")
                else:
                    n_cmp += 1
                max_err = max(max_err, float((a - b.float()).abs().max()) if a.numel() else 0.0)
                max_abs = max(max_abs, float(b.float().abs().max()) if b.numel() else 0.0)
            else:
                if not torch.equal(a, b):
                    bad.append(f"{k}: non-float tensor not bit-identical")
                else:
                    n_cmp += 1
            del a, b
    check("LLM values == fp16(src), bit-exact", not bad,
          f"{n_cmp} tensors compared" + (f"; {bad[:5]}" if bad else ""))
    check("LLM F32 params == ADR reference", n_float_params == REF_LLM_F32_PARAMS,
          f"{n_float_params:,} (ref {REF_LLM_F32_PARAMS:,})")
    check("LLM finite, max|w| < 60000", max_abs < FP16_MAX_ABS and torch.isfinite(torch.tensor(max_abs)),
          f"max|w| {max_abs:.3f}, max|f32-f16| {max_err:.6f}")

    # 6-8. audio tokenizer
    asrc, adst = src / AUDIO_WEIGHTS, dst / AUDIO_WEIGHTS
    src_ak, dst_ak = keys_of(asrc), keys_of(adst)
    want_keep = [k for k in src_ak if audio_key_kept(k)]
    want_drop = [k for k in src_ak if audio_key_dropped(k)]
    check("audio key set == keep prefixes", dst_ak == want_keep,
          f"{len(dst_ak)} kept keys (src {len(src_ak)}, dropped {len(want_drop)})"
          + ("" if dst_ak == want_keep else
             f"; sym-diff {sorted(set(dst_ak) ^ set(want_keep))[:5]}"))
    check("audio keep+drop partitions src", len(want_keep) + len(want_drop) == len(src_ak),
          f"{len(want_keep)} + {len(want_drop)} == {len(src_ak)}")

    bad, n = [], 0
    params = 0
    with safe_open(str(asrc), framework="pt", device="cpu") as fs, \
            safe_open(str(adst), framework="pt", device="cpu") as fd:
        for k in dst_ak:
            a, b = fs.get_tensor(k), fd.get_tensor(k)
            n += 1
            params += a.numel()
            if str(b.dtype) != "torch.float32":
                bad.append(f"{k}: dtype {b.dtype}")
            elif not torch.equal(a, b):
                bad.append(f"{k}: not bit-identical")
            del a, b
    check("audio values bit-identical (fp32 untouched)", not bad,
          f"{n} tensors, {params:,} params" + (f"; {bad[:5]}" if bad else ""))
    if params != REF_AUDIO_KEEP_PARAMS:
        print(f"  [WARN] audio keep params {params:,} != ADR reference {REF_AUDIO_KEEP_PARAMS:,}")
    dropped_params = 0
    with safe_open(str(asrc), framework="pt", device="cpu") as fs:
        for k in want_drop:
            sh = fs.get_slice(k)
            q = 1
            for s in sh.get_shape():
                q *= int(s)
            dropped_params += q
    check("audio param counts as recorded", dropped_params == REF_AUDIO_DROP_PARAMS,
          f"dropped {dropped_params:,} (ref {REF_AUDIO_DROP_PARAMS:,}), kept {params:,} (ref {REF_AUDIO_KEEP_PARAMS:,})")

    # 8b. S4: actually run decode() on both tokenizers (independent of the key sets)
    marker = json.loads((dst / DECODE_MARKER).read_text(encoding="utf-8"))
    gate = decode_gate(src, dst, marker)
    check("S4 decode(src) == decode(dst)", gate["fresh_equal"],
          f"{gate['samples']} samples, sha {gate['dst_sha256'][:16]}…")
    check("S4 decode(dst) == decode(src) after popping drops", gate["pruned_equal"],
          f"sha {gate['pruned_sha256'][:16]}…")

    # 9. config
    src_cfg_text = (src / "config.json").read_text(encoding="utf-8")
    dst_cfg_text = (dst / "config.json").read_text(encoding="utf-8")
    src_cfg = json.loads(src_cfg_text)
    dst_cfg = json.loads(dst_cfg_text)
    check("config dtype float16 in both places",
          dst_cfg.get("dtype") == "float16" and dst_cfg["llm_config"].get("dtype") == "float16",
          f"top {dst_cfg.get('dtype')!r}, llm {dst_cfg['llm_config'].get('dtype')!r}")
    src_cfg_un = json.loads(src_cfg_text)
    dst_cfg_un = json.loads(dst_cfg_text)
    src_cfg_un["dtype"] = dst_cfg_un["dtype"]
    src_cfg_un["llm_config"]["dtype"] = dst_cfg_un["llm_config"]["dtype"]
    check("config: no other value changed", src_cfg_un == dst_cfg_un,
          "identical apart from the 2 dtype fields")
    src_cfg_raw = (src / "config.json").read_bytes()
    check("config: byte-exact apart from 2 dtypes",
          (dst / "config.json").read_bytes() == src_cfg_raw.replace(b'"dtype": "float32"', b'"dtype": "float16"'),
          f"{len(src_cfg_raw):,} B -> {len((dst / 'config.json').read_bytes()):,} B")
    # audio ids are a separate id space: assert they survived untouched
    same_ids = all(src_cfg.get(k) == dst_cfg.get(k)
                   for k in ("audio_vocab_size", "audio_mask_id", "audio_codebook_weights", "num_audio_codebook"))
    check("audio_vocab_size / audio_mask_id unchanged", same_ids,
          f"audio_vocab_size {dst_cfg.get('audio_vocab_size')}, audio_mask_id {dst_cfg.get('audio_mask_id')}")

    # 10. copied files byte-identical
    copied = (ctx or {}).get("copied") or [r for r in COPY_TOP if r != "config.json"] + [
        f"{AUDIO_DIR}/{r}" for r in COPY_AUDIO]
    diff = []
    for rel in copied:
        a, b = src / rel, dst / rel
        if not b.is_file():
            diff.append(f"{rel}: missing")
        elif a.stat().st_size != b.stat().st_size or sha256_file(a) != sha256_file(b):
            diff.append(f"{rel}: differs")
    check("copied files byte-identical", not diff,
          f"{len(copied)} files" + (f"; {diff[:5]}" if diff else ""))
    check("audio_tokenizer/config.json unchanged", (src / AUDIO_WEIGHTS).with_name("config.json").read_bytes()
          == (dst / AUDIO_WEIGHTS).with_name("config.json").read_bytes(), "byte-for-byte")

    # 11. decode_only.json
    rec = marker["drop"]["keys"]
    check("decode_only.json dropped list == src diff", rec == want_drop,
          f"{len(rec)} keys, {marker['drop']['params']:,} params"
          + ("" if rec == want_drop else "; list does not match the actual key diff"))
    prefix_tops = {str(p).split(".", 1)[0] for p in marker["drop"]["prefixes"]}
    key_tops = {str(k).split(".", 1)[0] for k in rec}
    check("decode_only.json prefixes cover drop.keys", prefix_tops == key_tops,
          f"prefixes {sorted(prefix_tops)} vs keys {sorted(key_tops)}"
          + ("" if prefix_tops == key_tops
             else " — a marker may under-declare prefixes and leave a random module resident"))
    check("decode_only.json counts", marker["keep"]["params"] == params
          and marker["drop"]["params"] == dropped_params,
          f"keep {marker['keep']['params']:,} / drop {marker['drop']['params']:,}")
    check("decode_only.json source sha", marker["source"]["sha256"] == sha256_file(src / AUDIO_WEIGHTS),
          marker["source"]["sha256"][:16] + "…")
    check("decode_only.json dst sha", marker.get("dst_sha256") == sha256_file(adst)
          and marker.get("dst_bytes") == adst.stat().st_size,
          f"{marker.get('dst_sha256', 'MISSING')[:16]}… {marker.get('dst_bytes', 0):,} B")

    # 11b. compact_meta.json (provenance: what it was built from / what it is)
    meta = json.loads((dst / COMPACT_META).read_text(encoding="utf-8"))
    policy = {
        "llm_dtype": LLM_TARGET_DTYPE,
        "keep_prefixes": list(AUDIO_KEEP_PREFIXES),
        "drop_prefixes": list(AUDIO_DROP_PREFIXES),
    }
    check("compact_meta.json: format/tool/policy",
          meta.get("format_version") == 1
          and meta.get("tool") == "scripts/build_compact_model.py"
          and meta.get("policy") == policy,
          f"format_version={meta.get('format_version')}, tool={meta.get('tool')!r}")
    built_from = meta.get("built_from") or {}
    check("compact_meta.json: built_from == src",
          (built_from.get(LLM_WEIGHTS) or {}).get("sha256") == sha256_file(src / LLM_WEIGHTS)
          and (built_from.get(LLM_WEIGHTS) or {}).get("bytes") == (src / LLM_WEIGHTS).stat().st_size
          and (built_from.get(AUDIO_WEIGHTS) or {}).get("sha256") == marker["source"]["sha256"]
          and (built_from.get(AUDIO_WEIGHTS) or {}).get("bytes") == (src / AUDIO_WEIGHTS).stat().st_size,
          f"{LLM_WEIGHTS} {(built_from.get(LLM_WEIGHTS) or {}).get('sha256', 'MISSING')[:16]}…")
    out_rec = meta.get("output") or {}
    check("compact_meta.json: output == dst",
          (out_rec.get(LLM_WEIGHTS) or {}).get("sha256") == sha256_file(dst_w)
          and (out_rec.get(LLM_WEIGHTS) or {}).get("bytes") == dst_w.stat().st_size
          and (out_rec.get(AUDIO_WEIGHTS) or {}).get("sha256") == marker["dst_sha256"]
          and (out_rec.get(AUDIO_WEIGHTS) or {}).get("bytes") == adst.stat().st_size,
          f"{LLM_WEIGHTS} {(out_rec.get(LLM_WEIGHTS) or {}).get('sha256', 'MISSING')[:16]}…")

    # 12. src untouched (size + mtime manifest from preflight, or live hash-free check)
    if ctx and ctx.get("src_manifest"):
        now = manifest(src)
        changed = sorted(k for k in set(now) | set(ctx["src_manifest"])
                         if now.get(k) != ctx["src_manifest"].get(k))
        check("src untouched (size+mtime)", not changed,
              f"{len(now)} files" + (f"; CHANGED {changed[:5]}" if changed else ""))

    # 13. sizes
    src_total, dst_total = dir_bytes(src), dir_bytes(dst)
    print(f"\n  src {src}: {fmt_bytes(src_total)} ({fmt_gb(src_total)})")
    print(f"  dst {dst}: {fmt_bytes(dst_total)} ({fmt_gb(dst_total)})")
    print(f"  saving: {fmt_bytes(src_total - dst_total)} "
          f"({(1 - dst_total / src_total) * 100:.1f}% smaller, {src_total / dst_total:.2f}x)")
    print(f"  model.safetensors: {fmt_bytes((src / LLM_WEIGHTS).stat().st_size)} -> "
          f"{fmt_bytes((dst / LLM_WEIGHTS).stat().st_size)}")
    print(f"  audio_tokenizer/model.safetensors: {fmt_bytes((src / AUDIO_WEIGHTS).stat().st_size)} -> "
          f"{fmt_bytes((dst / AUDIO_WEIGHTS).stat().st_size)}")
    return {"src_bytes": src_total, "dst_bytes": dst_total}


def commit(ctx: dict) -> None:
    """Install the verified build with rename-aside, so a crash cannot lose the old artifact.

    ``dst`` -> ``dst.old`` -> ``dst`` are two renames on the same volume: if the
    second one dies, the exception handler renames the old artifact back.
    """
    dst, partial = ctx["dst"], ctx["partial"]
    old = dst.with_name(dst.name + OLD_SUFFIX)
    if dst.exists():
        if old.exists():
            shutil.rmtree(old)  # stale aside from an earlier crash; dst is still the live artifact
        os.replace(str(dst), str(old))
    try:
        os.replace(str(partial), str(dst))
    except Exception:
        if old.exists() and not dst.exists():
            os.replace(str(old), str(dst))  # restore the previous artifact
            print(f"commit failed: restored {dst} from {old}", file=sys.stderr)
        raise
    had_old = old.exists()
    if had_old:
        shutil.rmtree(old)
    print(f"\ncommitted: {partial} -> {dst}" + (" (previous artifact replaced)" if had_old else ""))


# -------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="models",
                    help="source model dir, read-only (default: %(default)s)")
    ap.add_argument("--dst", default="models_compact",
                    help="output model dir (default: %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="allow a non-empty --dst / delete a stale <dst>.partial")
    ap.add_argument("--verify", action="store_true",
                    help="verify an existing --dst against --src; write nothing")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    print(f"=== build_compact_model ({'VERIFY' if args.verify else 'BUILD'}) ===")
    print("preflight:")
    ctx = preflight(args)

    if args.verify:
        if not ctx["dst"].is_dir():
            die(f"--verify: {ctx['dst']} is not a directory")
        verify(ctx["src"], ctx["dst"])
        print("\nVERIFY OK: nothing written.")
        return

    info = build(ctx)
    verify(ctx["src"], ctx["partial"], ctx)
    commit(ctx)

    llm, audio = info["llm"], info["audio"]
    print("\nsummary:")
    print(f"  LLM: {llm['n_keys']} keys kept, {llm['n_float']} cast F32->F16 "
          f"({llm['params_f32']:,} params), {llm['n_nonfloat']} non-float "
          f"({llm['nonfloat_dtypes']}, {llm['params_other']} params), "
          f"max|w| {llm['max_abs_f32']:.3f}")
    print(f"  audio tokenizer: {len(audio['kept'])} keys / {audio['keep_params']:,} params kept fp32, "
          f"{len(audio['dropped'])} keys / {audio['drop_params']:,} params dropped")
    for name in sorted(audio["by_prefix"]):
        g = audio["by_prefix"][name]
        print(f"    dropped {name:<18} {g['keys']:>4} keys {g['params']:>12,} params")
    print("\nnext (vendor venv, from the repo root):")
    print(f"  {VENV_PY} eval\\run_eval.py --variant merged_fp32 --instruct mita --model models")
    print(f"  {VENV_PY} eval\\run_eval.py --variant merged_compact --instruct mita --model models_compact")
    print("expect the 20 wav_sha256 values to be identical.")


if __name__ == "__main__":
    main()
