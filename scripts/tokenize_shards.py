r"""Tokenize the finetune manifests into WebDataset shards, one command per split.

Usage (vendor venv — NOT the project .venv):
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\tokenize_shards.py --calibrate
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\tokenize_shards.py --split train
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\tokenize_shards.py --split dev
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\tokenize_shards.py --split all

Runs the upstream `omnivoice.scripts.extract_audio_tokens` CLI as a subprocess
(same interpreter, PYTHONUTF8=1) and then verifies the output:
  [1] data.lst: exactly 4 whitespace-free fields per line; every tar/jsonl exists
  [2] union of ids across txts/*.jsonl == manifest id set (no silent row loss)
  [3] every label row keeps id/text/language_id=="ru"/instruct=="mita"
      plus num_tokens and audio_duration
  [4] |num_tokens - round(audio_duration * frame_rate)| <= 1
      (frame_rate read from the loaded tokenizer object; abort if != 25)
  [5] 3 sampled tars (first/middle/last): every npy entry decodes to
      shape (8, N), dtype int16, N == num_tokens, values in [0, 1024)
  [6] sum(audio_duration) within 0.5% of the manifest's total duration

--calibrate tokenizes a 20-row train subset to tmp/shards_calib/ and then
pushes 3 shard samples through OmniVoiceSampleProcessor (instruct_ratio=1.0,
prompt_ratio_range=(0,0), language_ratio=1.0, fixed seed, real text
tokenizer) and decodes input_ids — the only proof that the
<|instruct_start|>mita<|instruct_end|> and <|lang_start|>ru<|lang_end|> tags
actually reach training.
"""

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
FINETUNE = ROOT / "data" / "finetune"
MANIFESTS = {"train": FINETUNE / "manifest_train.jsonl", "dev": FINETUNE / "manifest_dev.jsonl"}
SHARDS = FINETUNE / "shards"
CALIB_DIR = ROOT / "tmp" / "shards_calib"
CALIB_ROWS = 20
CALIB_SEED = 20261115
EXPECTED_FRAME_RATE = 25
TOKENIZER_REPO = "k2-fsa/OmniVoice"
HF_HUB = Path.home() / ".cache" / "huggingface" / "hub" / "models--k2-fsa--OmniVoice" / "snapshots"

UPSTREAM_ARGS = [
    "--nj_per_gpu", "2",
    "--samples_per_shard", "150",
    "--min_num_shards", "4",
    "--loader_workers", "4",
    "--shuffle", "True",
]

# webdataset's gopen dispatches on urlparse().scheme, and urlparse treats a
# Windows drive letter as a scheme ("C:\..." -> scheme "c"), so both
# wds.TarWriter (here) and wds.WebDataset (training) refuse C:\ paths.
# GOPEN_REWRITE rewrites "<drive>:\..." to "file:///<drive>:/" which gopen
# opens via url2pathname (gopen's rewrite_url prepends "^" to the pattern).
# Verified: write+read round-trip works with it, reading fails without it.
# Must be set for the tokenizer subprocess (write side) AND in the training
# .cmd (read side, see launch_train.py).
GOPEN_REWRITE = r"([A-Za-z]):\\=file:///\1:/"


def die(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def reject_decode_only(tok: Path) -> None:
    """Refuse a decode-only audio tokenizer (scripts/build_compact_model.py).

    The encoder-pruning shim lives only in scripts/mita_voice/runtime.py, so
    this script's HiggsAudioV2TokenizerModel would silently run encode() on
    randomly initialised HuBERT/DAC weights and tokenize hours of GPU time
    into a garbage corpus.  The full tokenizer lives next to the compact
    model dir, e.g. models/audio_tokenizer.
    """
    if (tok / "decode_only.json").is_file():
        die(
            f"{tok} is a decode-only audio tokenizer (decode_only.json present): its encoder "
            "weights are pruned, so encode() would run on random init and produce a garbage "
            "corpus. Pass the full tokenizer dir instead, e.g. --tokenizer-path "
            "models\\audio_tokenizer"
        )


def discover_tokenizer(cli_override: str | None) -> Path:
    if cli_override:
        p = Path(cli_override).expanduser().resolve()
        if not (p / "config.json").is_file() or not (p / "preprocessor_config.json").is_file():
            die(f"--tokenizer-path {p} missing config.json / preprocessor_config.json")
        reject_decode_only(p)
        return p
    cands = sorted(HF_HUB.glob("*/audio_tokenizer"))
    if not cands:
        die(f"no audio_tokenizer found under {HF_HUB}; pass --tokenizer-path")
    tok = cands[0]
    for f in ("config.json", "preprocessor_config.json"):
        if not (tok / f).is_file():
            die(f"tokenizer dir {tok} missing {f}")
    reject_decode_only(tok)
    return tok


def load_frame_rate(tokenizer_dir: Path) -> int:
    """Load the tokenizer object and read config.frame_rate (verified 25)."""
    from transformers import HiggsAudioV2TokenizerModel

    model = HiggsAudioV2TokenizerModel.from_pretrained(str(tokenizer_dir))
    frame_rate = int(model.config.frame_rate)
    if frame_rate != EXPECTED_FRAME_RATE:
        die(f"tokenizer frame_rate={frame_rate}, expected {EXPECTED_FRAME_RATE} — aborting")
    return frame_rate


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        die(f"manifest missing: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                die(f"{path}:{ln} is not valid JSON: {e}")
            rows.append(row)
    if not rows:
        die(f"manifest {path} is empty")
    return rows


def preflight(rows: list[dict], split: str, tokenizer_dir: Path) -> None:
    errs = []
    for i, row in enumerate(rows):
        rid = row.get("id", f"row{i}")
        ap = row.get("audio_path", "")
        if not ap or not os.path.isfile(ap):
            errs.append(f"{rid}: audio_path missing or not a file: {ap!r}")
        elif any(c.isspace() for c in ap):
            errs.append(f"{rid}: audio_path contains whitespace: {ap!r}")
        if row.get("language_id") != "ru":
            errs.append(f"{rid}: language_id={row.get('language_id')!r} != 'ru'")
        if row.get("instruct") != "mita":
            errs.append(f"{rid}: instruct={row.get('instruct')!r} != 'mita'")
        if not (row.get("text") or "").strip():
            errs.append(f"{rid}: empty text")
    try:
        import omnivoice  # noqa: F401
    except ImportError as e:
        errs.append(f"omnivoice not importable: {e}")
    free_gb = shutil.disk_usage(str(SHARDS if SHARDS.exists() else ROOT)).free / 1e9
    if free_gb < 2.0:
        errs.append(f"free disk {free_gb:.1f} GB < 2 GB")
    if errs:
        for e in errs:
            print(f"  preflight FAIL: {e}", file=sys.stderr)
        die(f"preflight failed for {split}: {len(errs)} error(s)")
    print(f"  preflight OK: {len(rows)} rows, tokenizer {tokenizer_dir}, free disk {free_gb:.1f} GB")


# ---------------------------------------------------------------------------
# Stale-output guard
# ---------------------------------------------------------------------------

def stale_guard(split_dir: Path, force: bool) -> None:
    if not split_dir.exists():
        return
    stale = [p for p in split_dir.rglob("*") if p.is_file() and p.suffix in (".tar", ".jsonl")]
    stale += [split_dir / "data.lst", split_dir / "errors.jsonl"]
    stale = [p for p in stale if p.is_file()]
    if not stale:
        return
    if not force:
        die(
            f"{split_dir} already contains output ({len(stale)} files, e.g. "
            f"{stale[0].name}); pass --force to delete audios/, txts/, data.lst, errors.jsonl"
        )
    for name in ("audios", "txts"):
        d = split_dir / name
        if d.is_dir():
            shutil.rmtree(d)
    for name in ("data.lst", "errors.jsonl"):
        f = split_dir / name
        if f.is_file():
            f.unlink()
    print(f"  --force: cleared {split_dir} (audios/, txts/, data.lst, errors.jsonl)")


# ---------------------------------------------------------------------------
# Upstream run
# ---------------------------------------------------------------------------

def run_upstream(split: str, manifest: Path, out_dir: Path, tokenizer_dir: Path) -> None:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"tokenize_{split}.log"
    cmd = [
        sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
        "--input_jsonl", str(manifest),
        "--tar_output_pattern", f"{out_dir}/audios/shard-%06d.tar",
        "--jsonl_output_pattern", f"{out_dir}/txts/shard-%06d.jsonl",
        "--tokenizer_path", str(tokenizer_dir),
        *UPSTREAM_ARGS,
    ]
    print(f"  running upstream tokenizer -> {log_path}")
    env = {**os.environ, "PYTHONUTF8": "1", "GOPEN_REWRITE": GOPEN_REWRITE}
    with open(log_path, "ab") as logf:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  upstream tokenizer FAILED (exit {proc.returncode}); last lines of {log_path}:", file=sys.stderr)
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-40:]:
            print(f"    {line}", file=sys.stderr)
        die(f"tokenization failed for {split}; see {log_path}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def read_data_lst(out_dir: Path) -> list[tuple[str, str, int, float]]:
    data_lst = out_dir / "data.lst"
    if not data_lst.is_file():
        die(f"data.lst missing: {data_lst}")
    entries = []
    for ln, line in enumerate(data_lst.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 4:
            die(f"data.lst:{ln} has {len(parts)} fields, expected 4: {line!r}")
        for i, p in enumerate(parts):
            if any(c.isspace() for c in p):
                die(f"data.lst:{ln} field {i} contains whitespace")
        tar, jsonl, count, seconds = parts[0], parts[1], int(parts[2]), float(parts[3])
        if not Path(tar).is_file():
            die(f"data.lst:{ln} tar missing: {tar}")
        if not Path(jsonl).is_file():
            die(f"data.lst:{ln} jsonl missing: {jsonl}")
        entries.append((tar, jsonl, count, seconds))
    return entries


def read_shard_labels(out_dir: Path) -> list[dict]:
    rows = []
    for f in sorted((out_dir / "txts").glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def manifest_total_duration(rows: list[dict]) -> float:
    import soundfile as sf

    total = 0.0
    for row in rows:
        info = sf.info(row["audio_path"])
        total += info.frames / info.samplerate
    return total


def verify(out_dir: Path, manifest_rows: list[dict], frame_rate: int) -> list[tuple[str, bool, str]]:
    results = []

    # [1] data.lst format + referenced files
    entries = read_data_lst(out_dir)
    results.append(("1 data.lst: 4 fields/line, all tar+jsonl exist", True, f"{len(entries)} shards"))

    # [2] id-set equality
    shard_rows = read_shard_labels(out_dir)
    manifest_ids = {r["id"] for r in manifest_rows}
    shard_ids = {r["id"] for r in shard_rows}
    dropped = sorted(manifest_ids - shard_ids)
    added = sorted(shard_ids - manifest_ids)
    ok = not dropped and not added
    detail = f"{len(shard_ids)} shard ids == {len(manifest_ids)} manifest ids"
    if dropped:
        detail += f"; DROPPED {len(dropped)}: {dropped[:5]}"
    if added:
        detail += f"; ADDED {len(added)}: {added[:5]}"
    results.append(("2 id set match (no silent row loss)", ok, detail))

    # [3] label fields
    bad = []
    for r in shard_rows:
        for field in ("id", "text", "num_tokens", "audio_duration"):
            if field not in r:
                bad.append(f"{r.get('id', '?')}: missing {field}")
        if r.get("language_id") != "ru":
            bad.append(f"{r.get('id', '?')}: language_id={r.get('language_id')!r}")
        if r.get("instruct") != "mita":
            bad.append(f"{r.get('id', '?')}: instruct={r.get('instruct')!r}")
    results.append(("3 label rows keep id/text/ru/mita + num_tokens + audio_duration",
                    not bad, f"{len(shard_rows)} rows checked" + (f"; {bad[:3]}" if bad else "")))

    # [4] num_tokens vs duration * frame_rate
    devs = []
    for r in shard_rows:
        expected = round(r["audio_duration"] * frame_rate)
        devs.append((abs(r["num_tokens"] - expected), r["id"], r["num_tokens"], expected))
    n0 = sum(1 for d in devs if d[0] == 0)
    n1 = sum(1 for d in devs if d[0] == 1)
    worst = sorted(devs, reverse=True)[:5]
    ok = all(d[0] <= 1 for d in devs)
    detail = f"dev0={n0} dev1={n1} dev>1={len(devs) - n0 - n1}; worst: " + ", ".join(
        f"{i}({t} vs {e})" for _, i, t, e in worst
    )
    results.append((f"4 |num_tokens - round(duration*{frame_rate})| <= 1", ok, detail))

    # [5] EVERY shard (not a sample): npy (8, N) int16, N==num_tokens,
    #     values in [0, 1024), and the tar's key set == the jsonl's id set
    import numpy as np

    problems = []
    checked = 0
    for i, (tar_path, jsonl_path, count, _) in enumerate(entries):
        num_tokens_by_id = {}
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    num_tokens_by_id[r["id"]] = r["num_tokens"]
        try:
            with tarfile.open(tar_path) as tf:
                npy_names = [m.name for m in tf.getmembers() if m.name.endswith(".npy")]
                npy_keys = {Path(n).stem for n in npy_names}
                if npy_keys != set(num_tokens_by_id):
                    only_tar = sorted(npy_keys - set(num_tokens_by_id))[:3]
                    only_jsonl = sorted(set(num_tokens_by_id) - npy_keys)[:3]
                    problems.append(
                        f"{Path(tar_path).name}: key set mismatch "
                        f"(only in tar: {only_tar}, only in jsonl: {only_jsonl})")
                if len(npy_names) != count:
                    problems.append(f"{Path(tar_path).name}: {len(npy_names)} npy entries != count {count}")
                for name in npy_names:
                    key = Path(name).stem
                    arr = np.load(io.BytesIO(tf.extractfile(name).read()))
                    if arr.shape[0] != 8:
                        problems.append(f"{key}: shape {arr.shape} != (8, N)")
                        continue
                    if arr.dtype != np.int16:
                        problems.append(f"{key}: dtype {arr.dtype} != int16")
                    if arr.min() < 0 or arr.max() >= 1024:
                        problems.append(f"{key}: values outside [0, 1024) (min={arr.min()} max={arr.max()})")
                    if key in num_tokens_by_id and arr.shape[1] != num_tokens_by_id[key]:
                        problems.append(f"{key}: N={arr.shape[1]} != num_tokens={num_tokens_by_id[key]}")
                    checked += 1
        except (tarfile.TarError, EOFError, OSError) as e:
            problems.append(f"{Path(tar_path).name}: unreadable (shard {i}): {type(e).__name__}: {e}")
    results.append(("5 all shards: npy (8, N) int16, N==num_tokens, keys==jsonl ids, values in [0, 1024)",
                    not problems, f"{checked} npy entries in {len(entries)} tars" + (f"; {problems[:3]}" if problems else "")))

    # [6] duration sum vs manifest
    shard_dur = sum(r["audio_duration"] for r in shard_rows)
    manifest_dur = manifest_total_duration(manifest_rows)
    rel = abs(shard_dur - manifest_dur) / manifest_dur
    ok = rel <= 0.005
    results.append(("6 sum(audio_duration) within 0.5% of manifest",
                    ok, f"shards {shard_dur / 60:.2f} min vs manifest {manifest_dur / 60:.2f} min (rel {rel * 100:.3f}%)"))
    return results


# ---------------------------------------------------------------------------
# Calibration: push shard samples through the real training processor
# ---------------------------------------------------------------------------

def run_calibration(out_dir: Path) -> None:
    import random as _random

    os.environ["GOPEN_REWRITE"] = GOPEN_REWRITE  # wds.WebDataset reads the shard tars via gopen
    import torch
    from transformers import AutoTokenizer

    from omnivoice.data.dataset import WebDatasetReader, webdataset_manifest_reader
    from omnivoice.data.processor import OmniVoiceSampleProcessor
    from omnivoice.models.omnivoice import _resolve_model_path

    tok_path = _resolve_model_path(TOKENIZER_REPO)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    new_tokens = ["<|denoise|>", "<|lang_start|>", "<|lang_end|>",
                  "<|instruct_start|>", "<|instruct_end|>", "<|text_start|>", "<|text_end|>"]
    to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    manifests = webdataset_manifest_reader(str(out_dir / "data.lst"))
    reader = WebDatasetReader(manifests=manifests, evaluation=True)
    samples = []
    for s in reader:
        samples.append(s)
        if len(samples) == 3:
            break
    if len(samples) < 3:
        die(f"calibration: only {len(samples)} samples readable from {out_dir}")

    processor = OmniVoiceSampleProcessor(
        text_tokenizer=tokenizer,
        num_channels=8,
        audio_mask_id=1024,
        prompt_ratio_range=(0.0, 0.0),
        mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=0.0,
        language_ratio=1.0,
        use_pinyin_ratio=0.0,
        instruct_ratio=1.0,
        only_instruct_ratio=0.0,
    )

    for i, s in enumerate(samples):
        _random.seed(CALIB_SEED + i)
        torch.manual_seed(CALIB_SEED + i)
        out = processor(s)
        ids = out["input_ids"][0]
        audio_mask = out["audio_mask"]
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        prompt = tokenizer.decode(ids[~audio_mask], skip_special_tokens=False)
        label = s["label"]
        missing = [
            tag for tag in ("<|instruct_start|>mita<|instruct_end|>", "<|lang_start|>ru<|lang_end|>", label["text"])
            if tag not in decoded
        ]
        if missing:
            die(f"calibration sample {i} ({label['id']}): decoded input_ids missing {missing}")
        print(f"  calib[{i}] {label['id']}: decoded OK (instruct tag, lang tag, text all present)")
        if i == 0:
            print(f"  decoded prompt (sample 0):\n    {prompt}")
    print("  calibration PASS: <|instruct_start|>mita<|instruct_end|> and <|lang_start|>ru<|lang_end|> reach training")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def tokenize_split(split: str, manifest: Path, out_dir: Path, tokenizer_dir: Path, force: bool) -> None:
    print(f"=== tokenize {split}: {manifest.name} -> {out_dir} ===")
    rows = read_manifest(manifest)
    preflight(rows, split, tokenizer_dir)
    stale_guard(out_dir, force)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_upstream(split, manifest, out_dir, tokenizer_dir)
    frame_rate = load_frame_rate(tokenizer_dir)
    results = verify(out_dir, rows, frame_rate)
    shard_rows = read_shard_labels(out_dir)
    data_lst = out_dir / "data.lst"
    total_tokens = sum(r["num_tokens"] for r in shard_rows)
    total_min = sum(r["audio_duration"] for r in shard_rows) / 60
    print(f"  summary: rows={len(shard_rows)} shards={len(list((out_dir / 'audios').glob('*.tar')))} "
          f"minutes={total_min:.2f} tokens={total_tokens} data.lst={data_lst.stat().st_size} bytes")
    all_ok = True
    for name, ok, detail in results:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    if not all_ok:
        die(f"verification failed for {split}")
    print(f"  {split}: ALL CHECKS PASSED")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--calibrate", action="store_true",
                      help=f"tokenize {CALIB_ROWS} train rows -> {CALIB_DIR} + processor calibration")
    mode.add_argument("--split", choices=["train", "dev", "all"])
    ap.add_argument("--force", action="store_true", help="delete existing shard output in the target split dir")
    ap.add_argument("--tokenizer-path", default=None, help="override the auto-discovered audio_tokenizer dir")
    args = ap.parse_args()

    tokenizer_dir = discover_tokenizer(args.tokenizer_path)
    print(f"tokenizer: {tokenizer_dir}")

    if args.calibrate:
        rows = read_manifest(MANIFESTS["train"])[:CALIB_ROWS]
        calib_manifest = ROOT / "tmp" / "manifest_calib.jsonl"
        calib_manifest.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        tokenize_split("calib", calib_manifest, CALIB_DIR, tokenizer_dir, args.force)
        run_calibration(CALIB_DIR)
        return

    splits = ["train", "dev"] if args.split == "all" else [args.split]
    for split in splits:
        tokenize_split(split, MANIFESTS[split], SHARDS / split, tokenizer_dir, args.force)


if __name__ == "__main__":
    main()
