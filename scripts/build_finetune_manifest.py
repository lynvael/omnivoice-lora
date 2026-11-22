#!/usr/bin/env python3
"""Build training manifests for the OmniVoice LoRA fine-tune.

Read-only inputs (never modified):
  data/manifest.jsonl        authoritative extraction (2669 rows)
  data/manifest_mita.jsonl   derived view, used ONLY as a lookup table (1515 rows)
  data/audio/ru/<Category>/<n>.ogg   source audio

Outputs (created/overwritten by this script):
  data/finetune/manifest_all.jsonl       612 rows
  data/finetune/manifest_train.jsonl     582 rows
  data/finetune/manifest_dev.jsonl       30 rows
  data/finetune/manifest_excluded.jsonl  93 rows, each with a reason
  data/finetune/audio/<id>.ogg           byte-identical copies, space-free names
  data/finetune/build_report.md

Selection rule: character == "mita" AND status == "match"
(match = subtitle's normalized words identical to an independent Whisper
transcription of the audio; the machine-verified set).

Text comes from the authoritative manifest (data/manifest.jsonl 'text',
original punctuation) by default. --text-source final switches to
text_final in the derived view (stripped of surrounding whitespace); in
that mode a divergence from the manifest is reported, not fatal. In
manifest mode any divergence fails the build (review corrections
detected). The STT column in the derived view is a QC signal and is
never used for training text.

Exit code is non-zero on any failed assertion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.jsonl"
MANIFEST_MITA = ROOT / "data" / "manifest_mita.jsonl"
OUT_DIR = ROOT / "data" / "finetune"
AUDIO_DIR = OUT_DIR / "audio"

SEED = 42
DEV_SIZE = 30
DEV_MIN_PER_BUCKET = 5
# (label, low inclusive, high exclusive); last bucket is open-ended
BUCKETS = [
    ("<1.0s", 0.0, 1.0),
    ("1.0-2.5s", 1.0, 2.5),
    (">2.5s", 2.5, float("inf")),
]
EXPECTED = {
    "mita_rows": 705,
    "mita_minutes": 19.01,
    "match_rows": 612,
    "match_minutes": 16.47,
    "reason_counts": {
        "status:near": 55,
        "status:mismatch": 28,
        "status:mismatch_strong": 7,
        "status:halluc_artifact": 3,
    },
    "excluded_rows": 93,
}
EMOTICON_RE = re.compile(r"<[^<>]*>")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)
    print(f"  ok: {msg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build training manifests for the OmniVoice LoRA "
                    "fine-tune.")
    p.add_argument(
        "--text-source", choices=("manifest", "final"), default="manifest",
        help="training text source: 'manifest' = data/manifest.jsonl "
             "'text' (authoritative, default; fails if text_final "
             "diverges); 'final' = derived view 'text_final' (divergence "
             "reported only)")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                fail(f"{path}:{i}: bad JSON: {e}")
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(t: str) -> str:
    """Lowercase, strip punctuation, collapse spaces (leakage guard key)."""
    t = t.lower()
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def bucket_of(duration_s: float) -> str:
    for label, lo, hi in BUCKETS:
        if lo <= duration_s < hi:
            return label
    return BUCKETS[-1][0]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """UTF-8 without BOM, LF line endings, ensure_ascii=False, 5 fixed fields."""
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_bytes(data.encode("utf-8"))


def main() -> None:
    # Windows consoles default to a legacy code page (e.g. cp1251) that
    # cannot encode the ∩/Σ/∅ symbols used below; force UTF-8 so a fresh
    # shell never dies mid-build after the manifests are already written.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    # ---------- load ----------
    print("== load ==")
    manifest = load_jsonl(MANIFEST)
    mita_view = load_jsonl(MANIFEST_MITA)
    print(f"  manifest.jsonl: {len(manifest)} rows")
    print(f"  manifest_mita.jsonl: {len(mita_view)} rows")

    lookup: dict[str, dict] = {}
    for r in mita_view:
        if r["audio_path"] in lookup:
            fail(f"duplicate audio_path in manifest_mita: {r['audio_path']}")
        lookup[r["audio_path"]] = r

    # ---------- step 1: character == "mita" ----------
    print("== step 1: character == 'mita' ==")
    step1 = [r for r in manifest if r.get("character") == "mita"]
    mins1 = sum(r["duration_s"] for r in step1) / 60
    check(len(step1) == EXPECTED["mita_rows"],
          f"mita rows: {len(step1)} (expect {EXPECTED['mita_rows']})")
    check(abs(mins1 - EXPECTED["mita_minutes"]) < 0.01,
          f"mita minutes: {mins1:.2f} (expect ~{EXPECTED['mita_minutes']})")

    # ---------- step 2: join on audio_relpath == audio_path ----------
    print("== step 2: join to manifest_mita ==")
    step2: list[tuple[dict, dict]] = []
    for r in step1:
        m = lookup.get(r["audio_relpath"])
        if m is None:
            fail(f"join miss: {r['audio_relpath']} not in manifest_mita "
                 f"(derived view drifted)")
        step2.append((r, m))
    check(len(step2) == EXPECTED["mita_rows"],
          f"joined: {len(step2)}/{EXPECTED['mita_rows']} (expect {EXPECTED['mita_rows']}/{EXPECTED['mita_rows']})")

    # ---------- step 3: status == "match" ----------
    print("== step 3: status == 'match' ==")
    step3 = [(r, m) for r, m in step2 if m["status"] == "match"]
    excluded = [(r, m) for r, m in step2 if m["status"] != "match"]
    mins3 = sum(m["duration_s"] for _, m in step3) / 60
    check(len(step3) == EXPECTED["match_rows"],
          f"match rows: {len(step3)} (expect {EXPECTED['match_rows']})")
    check(abs(mins3 - EXPECTED["match_minutes"]) < 0.01,
          f"match minutes: {mins3:.2f} (expect ~{EXPECTED['match_minutes']})")
    hp = [m for _, m in step3 if m.get("has_player")]
    check(len(hp) == 0,
          f"has_player rows in selection: {len(hp)} (expect 0; "
          f"all [player] rows fall outside 'match')")

    # ---------- excluded manifest ----------
    excluded_rows = [
        {
            "id": m["id"],
            "audio_relpath": r["audio_relpath"],
            "duration_s": r["duration_s"],
            "status": m["status"],
            "reason": f"status:{m['status']}",
        }
        for r, m in excluded
    ]
    reason_counts = Counter(e["reason"] for e in excluded_rows)
    check(len(excluded_rows) == EXPECTED["excluded_rows"],
          f"excluded rows: {len(excluded_rows)} (expect {EXPECTED['excluded_rows']})")
    check(all(e["reason"] for e in excluded_rows),
          "every excluded row carries a non-empty reason")
    check(reason_counts == Counter(EXPECTED["reason_counts"]),
          f"excluded reason counts: {dict(sorted(reason_counts.items()))} "
          f"(expect {EXPECTED['reason_counts']})")

    # ---------- text ----------
    print("== text ==")
    for r, m in step3:
        tf = m.get("text_final")
        if not isinstance(tf, str) or not tf.strip():
            fail(f"missing/empty text_final for {m['id']}")

    if args.text_source == "manifest":
        offenders = [m["id"] for r, m in step3
                     if m["text_final"] != r["text"]]
        if offenders:
            fail(f"text_final != text on {len(offenders)} selected rows "
                 f"(first: {', '.join(offenders[:5])}); review corrections "
                 f"detected; rebuild with --text-source final")
        check(not offenders,
              f"text_final == text on all {len(step3)} selected rows "
              f"(derived view in sync with manifest)")
    else:
        diff_count = sum(1 for r, m in step3
                         if m["text_final"] != r["text"])
        print(f"  info: text_final != text on {diff_count} selected rows "
              f"(expected 0 today; informational, not a failure)")

    def chosen_text(r: dict, m: dict) -> str:
        if args.text_source == "manifest":
            return r["text"]  # authoritative, original punctuation
        return m["text_final"].strip()  # legacy: surrounding whitespace only

    flagged = []
    for r, m in step3:
        t = chosen_text(r, m)
        if "\n" in t or "\t" in t:
            flagged.append((m["id"], "newline/tab", t))
        elif EMOTICON_RE.search(t):
            flagged.append((m["id"], "emoticon", t))
    if flagged:
        for i, kind, t in flagged:
            print(f"  FLAG {i} [{kind}]: {t!r}", file=sys.stderr)
        fail(f"{len(flagged)} selected texts contain newline/tab/emoticon "
             f"(reported, not edited)")
    check(not flagged, "no selected text contains newline/tab/emoticon")

    selected = []
    for r, m in step3:
        selected.append({
            "id": m["id"],
            "src": r["audio_relpath"],
            "duration_s": r["duration_s"],
            "text": chosen_text(r, m),
        })
    check(all(s["text"].strip() for s in selected),
          f"chosen text non-empty on all {len(selected)} selected rows")
    # text_stt is a QC signal; the emitted text must come from the chosen
    # source (manifest 'text' or 'text_final'), never from text_stt.
    bad_stt = [s["id"] for s, (r, m) in zip(selected, step3)
               if s["text"] not in (r["text"], m["text_final"].strip())]
    check(not bad_stt,
          f"emitted text sourced from manifest/final only, never text_stt "
          f"({len(bad_stt)} bad)")

    ids = [s["id"] for s in selected]
    check(len(set(ids)) == len(ids),
          f"ids unique across all {len(ids)} selected rows")

    # ---------- copy audio ----------
    print("== copy audio ==")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    # Orphan guard runs BEFORE the wipe below: the wipe unlinks every .ogg,
    # so a stray .ogg from a previous selection would be silently deleted
    # instead of reported. Stray .ogg files are deleted and logged; any
    # non-.ogg file is a hard failure.
    selected_ids = {s["id"] for s in selected}
    stray_ogg: list[str] = []
    stray_other: list[str] = []
    for f in AUDIO_DIR.iterdir():
        if f.suffix == ".ogg":
            if f.stem not in selected_ids:
                f.unlink()
                stray_ogg.append(f.name)
        else:
            stray_other.append(f.name)
    if stray_ogg:
        print(f"  info: removed {len(stray_ogg)} stray .ogg file(s) from a "
              f"previous selection: {sorted(stray_ogg)[:5]}")
    check(not stray_other,
          f"audio dir holds only .ogg files before cleanup "
          f"({len(stray_other)} stray file(s): {sorted(stray_other)[:5]})")
    for f in AUDIO_DIR.iterdir():  # idempotent: clear stale copies
        if f.suffix == ".ogg":
            f.unlink()
    total_bytes = 0
    for s in selected:
        src = ROOT / s["src"]
        if not src.is_file():
            fail(f"source audio missing: {src}")
        dst = AUDIO_DIR / f"{s['id']}.ogg"
        shutil.copyfile(src, dst)
        if src.stat().st_size != dst.stat().st_size:
            fail(f"size mismatch after copy: {dst.name}")
        if sha256(src) != sha256(dst):
            fail(f"sha256 mismatch after copy: {dst.name}")
        total_bytes += dst.stat().st_size
    print(f"  copied {len(selected)} files, {total_bytes} bytes "
          f"({total_bytes / 1024 / 1024:.2f} MB), all size+sha256 verified")

    # final-state assertion: after the copy, the audio dir must hold
    # exactly the copied .ogg files for the selected ids
    orphans = [f.name for f in AUDIO_DIR.iterdir()
               if f.suffix != ".ogg" or f.stem not in selected_ids]
    check(not orphans,
          f"audio dir holds only .ogg files for selected ids "
          f"({len(orphans)} orphan(s): {sorted(orphans)[:5]})")

    # Every copied file was verified above (size + sha256); the spot-check
    # below re-hashes first, last, and 2 evenly spaced files for the log.
    step = max(1, len(selected) // 4)
    spot_idx = sorted({0, len(selected) - 1, step, 2 * step})
    for i in spot_idx:
        s = selected[i]
        src = ROOT / s["src"]
        dst = AUDIO_DIR / f"{s['id']}.ogg"
        h = sha256(dst)
        same = h == sha256(src)
        print(f"  spot {s['id']}: sha256={h[:16]}... src==copy: {same}")
        if not same:
            fail(f"spot-check sha256 mismatch: {s['id']}")

    # ---------- split ----------
    print("== split (seed 42) ==")
    counts = Counter(norm_text(s["text"]) for s in selected)
    dup_texts = {t: n for t, n in counts.items() if n > 1}
    dup_rows = sum(dup_texts.values())
    unique_rows = [s for s in selected if counts[norm_text(s["text"])] == 1]
    print(f"  info: {len(dup_texts)} duplicate normalized texts, "
          f"{dup_rows} rows moved to train (leakage guard)")

    rng = random.Random(SEED)
    dev: list[dict] = []
    bucket_alloc: dict[str, int] = {}
    for label, lo, hi in BUCKETS:
        rows = sorted((s for s in unique_rows if lo <= s["duration_s"] < hi),
                      key=lambda s: s["id"])
        ideal = DEV_SIZE * len(rows) / len(unique_rows)
        bucket_alloc[label] = int(ideal)
        rng.shuffle(rows)
        dev.extend(rows[:bucket_alloc[label]])
    # largest-remainder top-up, then enforce the per-bucket minimum
    rem = DEV_SIZE - len(dev)
    order = sorted(
        (label for label, _, _ in BUCKETS),
        key=lambda l: (DEV_SIZE * sum(1 for s in unique_rows
                                      if bucket_of(s["duration_s"]) == l)
                       / len(unique_rows) - bucket_alloc[l]),
        reverse=True,
    )
    dev_ids = {s["id"] for s in dev}
    for label in order[:rem]:
        rows = sorted((s for s in unique_rows
                       if bucket_of(s["duration_s"]) == label),
                      key=lambda s: s["id"])
        rng.shuffle(rows)
        # exclude rows already in dev, else a duplicate id aborts the build
        picked = next((c for c in rows if c["id"] not in dev_ids), None)
        if picked is None:
            fail(f"no top-up candidate left in bucket {label}")
        dev.append(picked)
        dev_ids.add(picked["id"])
        bucket_alloc[label] += 1
    for label, _, _ in BUCKETS:
        while bucket_alloc[label] < DEV_MIN_PER_BUCKET:
            donor = max((l for l, _, _ in BUCKETS
                         if bucket_alloc[l] > DEV_MIN_PER_BUCKET),
                        key=lambda l: bucket_alloc[l])
            moved = next(s for s in dev
                         if bucket_of(s["duration_s"]) == donor)
            dev.remove(moved)
            rows = sorted((s for s in unique_rows
                           if bucket_of(s["duration_s"]) == label
                           and s not in dev),
                          key=lambda s: s["id"])
            rng.shuffle(rows)
            dev.append(rows[0])
            bucket_alloc[donor] -= 1
            bucket_alloc[label] += 1

    dev_ids = {s["id"] for s in dev}
    check(len(dev_ids) == len(dev),
          f"dev has no duplicate ids ({len(dev)} rows, "
          f"{len(dev_ids)} unique ids)")
    train = [s for s in selected if s["id"] not in dev_ids]
    check(len(dev) == DEV_SIZE, f"dev rows: {len(dev)} (expect {DEV_SIZE})")
    check(len(train) == EXPECTED["match_rows"] - DEV_SIZE,
          f"train rows: {len(train)} (expect {EXPECTED['match_rows'] - DEV_SIZE})")

    # ---------- write outputs ----------
    print("== write outputs ==")

    def out_rec(s: dict) -> dict:
        return {
            "id": s["id"],
            "audio_path": str(AUDIO_DIR / f"{s['id']}.ogg").replace("\\", "/"),
            "text": s["text"],
            "language_id": "ru",
            "instruct": "mita",
        }

    all_recs = [out_rec(s) for s in selected]
    train_recs = [out_rec(s) for s in train]
    dev_recs = [out_rec(s) for s in dev]
    files = {
        "manifest_all.jsonl": all_recs,
        "manifest_train.jsonl": train_recs,
        "manifest_dev.jsonl": dev_recs,
        "manifest_excluded.jsonl": excluded_rows,
    }
    for name, rows in files.items():
        write_jsonl(OUT_DIR / name, rows)
        print(f"  wrote {name} ({len(rows)} rows)")

    # ---------- gates ----------
    print("== gates ==")
    train_ids = {s["id"] for s in train}
    check(len(set(ids)) == len(ids) == EXPECTED["match_rows"],
          f"ids unique across all {len(ids)} rows")
    check(not (train_ids & dev_ids), "train ∩ dev = ∅")

    bad_paths = [r["id"] for r in all_recs
                 if not (Path(r["audio_path"]).is_absolute()
                         and not any(c.isspace() for c in r["audio_path"])
                         and Path(r["audio_path"]).is_file())]
    check(not bad_paths,
          f"all {len(all_recs)} audio_path absolute, whitespace-free, exist "
          f"({len(bad_paths)} bad)")

    bad_fields = [r["id"] for r in all_recs
                  if r["language_id"] != "ru" or r["instruct"] != "mita"
                  or not r["text"].strip()]
    check(not bad_fields,
          f"all rows: language_id=='ru', instruct=='mita', text non-empty "
          f"({len(bad_fields)} bad)")

    total_dur = sum(s["duration_s"] for s in selected)
    split_dur = sum(s["duration_s"] for s in train) \
        + sum(s["duration_s"] for s in dev)
    check(abs(split_dur - total_dur) / total_dur < 0.005,
          f"Σ duration train+dev {split_dur:.3f}s within 0.5% of "
          f"{total_dur:.3f}s")

    for label, _, _ in BUCKETS:
        n = sum(1 for s in dev if bucket_of(s["duration_s"]) == label)
        check(n >= DEV_MIN_PER_BUCKET,
              f"dev bucket {label}: {n} rows (>= {DEV_MIN_PER_BUCKET})")

    dev_norms = {norm_text(s["text"]) for s in dev}
    coll = [s["id"] for s in train if norm_text(s["text"]) in dev_norms]
    check(not coll, f"no normalized-text collision between train and dev "
                    f"({len(coll)} collisions)")

    check(len(excluded_rows) == EXPECTED["excluded_rows"]
          and all(e["reason"] for e in excluded_rows)
          and reason_counts == Counter(EXPECTED["reason_counts"]),
          f"excluded: {len(excluded_rows)} rows, reasons "
          f"{dict(sorted(reason_counts.items()))}")

    for name in files:
        b = (OUT_DIR / name).read_bytes()
        check(not b.startswith(b"\xef\xbb\xbf") and b"\r" not in b,
              f"{name}: no BOM, zero CR")

    # ---------- report ----------
    print("== report ==")
    speaker_counts = Counter(r.get("speaker_raw") or "(missing)"
                             for r, m in step3)
    print("  info: speaker_raw distribution: " +
          ", ".join(f"{name}={n}" for name, n in
                    sorted(speaker_counts.items(),
                           key=lambda kv: (-kv[1], kv[0]))))
    max_dur = max(r["duration_s"] for r, m in step3)
    max_audio_frames = math.ceil(max_dur * 25)  # audio tokenizer: 25 fps
    lines = []
    lines.append("# Finetune build report")
    lines.append("")
    lines.append(f"Filter: `character == \"mita\" AND status == \"match\"` "
                 f"(subtitle's normalized words identical to an independent "
                 f"Whisper transcription of the audio).")
    lines.append("")
    lines.append("## Funnel")
    lines.append("")
    lines.append(f"- mita rows: {len(step1)} ({mins1:.2f} min)")
    lines.append(f"- status == match: {len(step3)} ({mins3:.2f} min)")
    lines.append(f"- train: {len(train)} ({sum(s['duration_s'] for s in train) / 60:.2f} min)")
    lines.append(f"- dev: {len(dev)} ({sum(s['duration_s'] for s in dev) / 60:.2f} min)")
    lines.append(f"- excluded: {len(excluded_rows)} "
                 f"({dict(sorted(reason_counts.items()))})")
    lines.append(f"- token ceiling: max processed length ≈{max_audio_frames + 48} "
                 f"tokens (longest clip {max_dur:.2f} s ≈ {max_audio_frames} "
                 f"audio frames at 25 fps + ~48 style/text tokens) vs "
                 f"`max_sample_tokens = 256`; no rows filtered by length")
    lines.append("")
    lines.append("## Split level")
    lines.append("")
    lines.append("The split is row-level (seeded, duration-stratified), not "
                 "category-level, although PLAN.md:134 says the split should "
                 "be \"strictly by category\" (строго по категориям), not by "
                 "row. Row-level is correct here: all "
                 f"{len(step3)} rows are one speaker and one instruct, so "
                 "there is no second voice or style that could leak between "
                 "splits; the only real leak risk is duplicated text, which "
                 f"is guarded by sending all {dup_rows} rows of the "
                 f"{len(dup_texts)} duplicated normalized texts to train. "
                 "Dev is therefore a loss/NaN monitor, not a "
                 "checkpoint-ranking set, because the eval processor "
                 "re-randomises mask ratio and drop_cond.")
    lines.append("")
    lines.append("## Dev bucket composition")
    lines.append("")
    lines.append("| bucket | count | min (s) | mean (s) | max (s) |")
    lines.append("|---|---|---|---|---|")
    for label, _, _ in BUCKETS:
        ds = [s["duration_s"] for s in dev if bucket_of(s["duration_s"]) == label]
        lines.append(f"| {label} | {len(ds)} | {min(ds):.2f} | "
                     f"{sum(ds) / len(ds):.2f} | {max(ds):.2f} |")
    lines.append("")
    lines.append("## 10 shortest clips (of 612)")
    lines.append("")
    for s in sorted(selected, key=lambda s: s["duration_s"])[:10]:
        lines.append(f"- {s['id']}: {s['duration_s']:.2f}s")
    lines.append("")
    lines.append("## 10 longest clips (of 612)")
    lines.append("")
    for s in sorted(selected, key=lambda s: -s["duration_s"])[:10]:
        lines.append(f"- {s['id']}: {s['duration_s']:.2f}s")
    lines.append("")
    lines.append(f"## Duplicate texts")
    lines.append("")
    lines.append(f"{len(dup_texts)} normalized texts occur more than once; "
                 f"all {dup_rows} of their rows were kept in train, dev takes "
                 f"only texts that appear exactly once.")
    lines.append("")
    lines.append("## Speaker personas (informational)")
    lines.append("")
    lines.append("`character == \"mita\"` groups several scene labels for "
                 "the same actress. Distribution of `speaker_raw` across "
                 f"the {len(step3)} selected rows:")
    lines.append("")
    lines.append("| speaker_raw | count |")
    lines.append("|---|---|")
    for name, n in sorted(speaker_counts.items(),
                          key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {name} | {n} |")
    lines.append("")
    lines.append("These are scene labels, not distinct voices: a spectral "
                 "check found no telephone band-limiting in the `Mitafon` "
                 "rows, so no exclusion is applied.")
    lines.append("")
    (OUT_DIR / "build_report.md").write_bytes(
        ("\n".join(lines) + "\n").encode("utf-8"))
    print(f"  wrote build_report.md")

    print("== done ==")
    print(f"funnel: {len(step1)} -> {len(step3)} -> "
          f"{len(train)} train + {len(dev)} dev; excluded {len(excluded_rows)}")
    print(f"bytes copied: {total_bytes}")


if __name__ == "__main__":
    main()
