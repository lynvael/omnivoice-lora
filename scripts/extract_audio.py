"""One-shot extraction of the MiSide Russian voice pack into the project.

Copies all ogg audio + LocationDialogue transcripts out of the (read-only)
game install into data/, probes durations with ffprobe, and writes
data/manifest.jsonl with text/speaker/instruction labels for the later
OmniVoice LoRA fine-tune.

Speaker resolution per text line:
  1. A '//' marker line sets the current speaker for the lines that follow.
     'X - [not]' means X speaks but the block is unvoiced; a bare '[not]'
     marker has no speaker.
  2. A trailing inline annotation ('text // Note') is stripped from the text.
     If the annotation normalises to a KNOWN SPEAKER LABEL (corpus-derived,
     see build_known_speakers), it OVERRIDES the marker for that line only.
     Otherwise it is a stage direction / description: the marker's speaker
     is kept and the annotation is preserved in `inline_note`.
  3. Unrecognised annotations never relabel a line (safe fallback).

Idempotent: safe to re-run; outputs are overwritten in place.
Source is never modified.

Usage:  python scripts/extract_audio.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_AUDIO = Path(r"E:\SteamLibrary\steamapps\common\MiSide\Data\LanguagesVoice\Russian-Dream Cast")
SOURCE_TEXTS = Path(r"E:\SteamLibrary\steamapps\common\MiSide\Data\Languages\Russian")

OUT_AUDIO = PROJECT_ROOT / "data" / "audio" / "ru"
OUT_TEXTS = PROJECT_ROOT / "data" / "transcripts" / "ru"
OUT_MANIFEST = PROJECT_ROOT / "data" / "manifest.jsonl"
OUT_REPORT_JSON = PROJECT_ROOT / "data" / "extract_report.json"
OUT_REPORT_MD = PROJECT_ROOT / "data" / "extract_report.md"

FFPROBE_WORKERS = 16
PLAYER_PLACEHOLDER = "[player]"
PLAYER_REPLACEMENT = "игрок"
COMMENT_DELIMITER = " // "  # trailing inline annotation marker in text lines
NOT_TAG = "[not]"  # "unvoiced" tag, not a speaker

# Corpus-derived gap: real non-Mita speakers appear as marker speakers 71-289
# times (Mila, Player); the most frequent non-speaker clean name is a stage
# direction with 3 (Lamping). 5 sits in that gap.
MIN_SPEAKER_COUNT = 5

# ---------------------------------------------------------------------------
# Pure parsing / normalization (unit-tested in test_extract_audio.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawLine:
    """One physical transcript line (1-based, blanks counted)."""

    line_no: int
    kind: str  # "marker" | "text" | "blank"
    marker_body: str | None  # marker lines: text after '//'
    text: str | None  # text lines: annotation-stripped, trimmed
    annotation: str | None  # text lines: trailing inline annotation, if any


@dataclass(frozen=True)
class ResolvedLine:
    """A raw line with speaker resolution applied."""

    line_no: int
    kind: str
    text: str | None
    speaker_raw: str | None
    speaker_source: str | None  # "marker" | "inline" (text lines only)
    inline_note: str | None  # annotation classified as description, else None
    unvoiced: bool  # line sits in a [not] block / carries a [not] annotation


def iter_lines(text: str) -> list[str]:
    """Split transcript text into physical lines.

    Splits on \n, strips a trailing \r (CRLF files), and does not invent a
    phantom line for a trailing newline. A missing final newline still
    yields the last line.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [ln[:-1] if ln.endswith("\r") else ln for ln in lines]


def strip_trailing_comment(line: str) -> str:
    """Strip a trailing ' // ...' inline annotation from a text line."""
    idx = line.find(COMMENT_DELIMITER)
    return line[:idx] if idx != -1 else line


def extract_speaker(body: str) -> tuple[str | None, bool]:
    """(speaker, unvoiced) from a marker body or inline annotation.

    - 'a - b - c'  -> speaker is the part after the LAST ' - ' (c)
    - 'X - [not]'  -> speaker is X, block is unvoiced ([not] is a tag,
      not a speaker; the rule applies recursively to the preceding segment)
    - '[not]'      -> no speaker, unvoiced
    """
    body = body.strip()
    if " - " in body:
        head, tail = body.rsplit(" - ", 1)
        if tail.strip() == NOT_TAG:
            speaker, _ = extract_speaker(head)
            return speaker, True
        return (tail.strip() or None), False
    if body == NOT_TAG:
        return None, True
    return (body or None), False


def is_clean_name(label: str) -> bool:
    """A single alphabetic token: no spaces, digits, or punctuation."""
    return bool(label) and " " not in label and label.isalpha()


# Longest prefix first: 'mita' is a prefix of every other Mita-family label.
_CHARACTER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("mita_shorthairs", "mitashorthairs"),
    ("mita_dream", "mitadream"),
    ("mita_kind", "mitakind"),
    ("mita_cap", "mitacap"),  # case-insensitive: covers MitaCap and Mitacap
    ("mita", "mita"),
)


def normalize_character(speaker: str | None) -> str | None:
    """Map a raw speaker label to a character bucket (longest prefix wins)."""
    if not speaker:
        return None
    s = speaker.strip().lower()
    if not s:
        return None
    for bucket, prefix in _CHARACTER_PREFIXES:
        if s.startswith(prefix):
            return bucket
    return None


def build_known_speakers(
    marker_freq: dict[str, int],
    inline_annotations: set[str],
) -> set[str]:
    """Corpus-derived set of labels allowed to act as inline speaker overrides.

    A label qualifies only if it (a) actually appears as an inline annotation
    and (b) appears as a marker-derived speaker in the corpus, either as a
    Mita-family label (any Mita-prefix) or as a frequent clean name. The
    frequency filter separates real non-Mita speakers (Player, Mila) from
    stage directions that merely look name-like (Lamping). Unrecognised
    annotations fall back to the marker's speaker and never relabel a line.
    """
    known: set[str] = set()
    for label in inline_annotations:
        if label not in marker_freq:
            continue
        if normalize_character(label) is not None:
            known.add(label)
        elif is_clean_name(label) and marker_freq[label] >= MIN_SPEAKER_COUNT:
            known.add(label)
    return known


def clean_text(raw: str) -> str:
    """Replace the [player] placeholder with the Russian word for 'player'."""
    return raw.replace(PLAYER_PLACEHOLDER, PLAYER_REPLACEMENT)


def parse_transcript(text: str) -> list[RawLine]:
    """Physical parse: line N (1-based, blanks counted) <-> <Category>/<N>.ogg.

    A line starting with '//' is a marker; other non-blank lines are text
    with an optional trailing ' // ...' annotation split off.
    """
    out: list[RawLine] = []
    for i, raw in enumerate(iter_lines(text), start=1):
        stripped = raw.strip()
        if stripped.startswith("//"):
            out.append(RawLine(i, "marker", stripped[2:].strip(), None, None))
        elif not stripped:
            out.append(RawLine(i, "blank", None, None, None))
        else:
            idx = raw.find(COMMENT_DELIMITER)
            if idx != -1:
                text_part = raw[:idx].strip()
                annotation = raw[idx + len(COMMENT_DELIMITER):].strip() or None
            else:
                text_part, annotation = raw.strip(), None
            if text_part:
                out.append(RawLine(i, "text", None, text_part, annotation))
            else:
                out.append(RawLine(i, "blank", None, None, None))
    return out


def resolve_lines(raw_lines: list[RawLine], known_speakers: set[str]) -> list[ResolvedLine]:
    """Speaker resolution: marker carry-forward + per-line inline overrides."""
    out: list[ResolvedLine] = []
    current_speaker: str | None = None
    current_unvoiced = False
    for ln in raw_lines:
        if ln.kind == "marker":
            current_speaker, current_unvoiced = extract_speaker(ln.marker_body)
            out.append(
                ResolvedLine(ln.line_no, "marker", None, current_speaker, None, None, current_unvoiced)
            )
        elif ln.kind == "blank":
            out.append(ResolvedLine(ln.line_no, "blank", None, None, None, None, False))
        else:
            speaker, source, note, unvoiced = current_speaker, "marker", None, current_unvoiced
            if ln.annotation is not None:
                ann_speaker, ann_unvoiced = extract_speaker(ln.annotation)
                if ann_speaker is not None and ann_speaker in known_speakers:
                    speaker, source, note = ann_speaker, "inline", None
                    unvoiced = current_unvoiced or ann_unvoiced
                else:
                    note = ln.annotation  # description: keep marker's speaker
            out.append(ResolvedLine(ln.line_no, "text", ln.text, speaker, source, note, unvoiced))
    return out


# ---------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------


def probe_duration(path: Path) -> float | None:
    """Duration in seconds via ffprobe, or None on failure."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            return None
        return float(proc.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def probe_durations(paths: list[Path]) -> dict[Path, float | None]:
    """Probe many files concurrently (sequential ffprobe is too slow)."""
    durations: dict[Path, float | None] = {}
    with ThreadPoolExecutor(max_workers=FFPROBE_WORKERS) as pool:
        for path, dur in zip(paths, pool.map(probe_duration, paths)):
            durations[path] = dur
    return durations


# ---------------------------------------------------------------------------
# Copy + verify
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    """Real byte copy (no symlinks/hardlinks), overwriting any previous run."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def verify_copy(src: Path, dst: Path) -> bool:
    """Byte-identity check: size + sha256."""
    if src.stat().st_size != dst.stat().st_size:
        return False
    return sha256_of(src) == sha256_of(dst)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@dataclass
class CategoryStats:
    category: str
    source_ogg: int = 0
    copied_ogg: int = 0
    verified_ogg: int = 0
    failed_ogg: list[str] = field(default_factory=list)
    non_numeric_ogg: int = 0
    transcript_copied: bool = False
    manifest_rows: int = 0
    ogg_skipped_missing_text: int = 0
    text_skipped_missing_ogg: int = 0
    inline_speaker_rows: int = 0
    unvoiced_with_audio: list[tuple[int, str]] = field(default_factory=list)


def discover_categories() -> list[tuple[str, list[Path]]]:
    """(category name, ogg paths) for every source dir with >=1 ogg.

    Numeric stems sort by number first; non-numeric stems (unexpected) sort
    by name after them, so a stray file can never crash the numeric sort.
    """
    cats: list[tuple[str, list[Path]]] = []
    for d in sorted(SOURCE_AUDIO.iterdir()):
        if not d.is_dir():
            continue
        oggs = [p for p in d.iterdir() if p.suffix.lower() == ".ogg"]
        if not oggs:
            continue
        numeric = sorted((p for p in oggs if p.stem.isdigit()), key=lambda p: int(p.stem))
        non_numeric = sorted((p for p in oggs if not p.stem.isdigit()), key=lambda p: p.name)
        cats.append((d.name, numeric + non_numeric))
    return cats


def build_manifest_rows(
    category: str,
    oggs: list[Path],
    resolved_lines: list[ResolvedLine],
    durations: dict[Path, float | None],
) -> tuple[list[dict], int, int, list[tuple[int, str]]]:
    """Intersect ogg files with non-empty text lines.

    Returns (rows, ogg_skipped_missing_text, text_skipped_missing_ogg,
    unvoiced_with_audio).
    """
    text_by_line: dict[int, ResolvedLine] = {
        ln.line_no: ln for ln in resolved_lines if ln.kind == "text"
    }
    ogg_ids = {int(p.stem) for p in oggs}
    rows: list[dict] = []
    ogg_skipped = 0
    unvoiced_with_audio: list[tuple[int, str]] = []
    for p in oggs:
        n = int(p.stem)
        tl = text_by_line.get(n)
        if tl is None:
            ogg_skipped += 1
            continue
        dst = OUT_AUDIO / category / p.name
        character = normalize_character(tl.speaker_raw)
        rows.append(
            {
                "id": n,
                "category": category,
                "audio_relpath": f"data/audio/ru/{category}/{p.name}",
                "text": clean_text(tl.text),
                "text_raw": tl.text,
                "speaker_raw": tl.speaker_raw,
                "character": character,
                "instruct": character,
                "duration_s": durations.get(dst),
                "is_mita": character is not None,
                "speaker_source": tl.speaker_source,
                "inline_note": tl.inline_note,
            }
        )
        if tl.unvoiced:
            unvoiced_with_audio.append((n, tl.text))

    text_skipped = sum(1 for n in text_by_line if n not in ogg_ids)
    return rows, ogg_skipped, text_skipped, unvoiced_with_audio


# ---------------------------------------------------------------------------
# Self-test (required assertions, checked against real data)
# ---------------------------------------------------------------------------


def run_selftest(rows: list[dict]) -> list[tuple[str, bool]]:
    by_key = {(r["category"], r["id"]): r for r in rows}
    results: list[tuple[str, bool]] = []

    # A1: line-10 alignment (Location1 line 10 == 10.ogg)
    r = by_key.get(("LocationDialogue Location1", 10))
    ok = (
        r is not None
        and r["text"] == "Привет, я Мита!"
        and r["speaker_raw"] == "Mita"
        and r["character"] == "mita"
        and r["speaker_source"] == "marker"
        and (OUT_AUDIO / "LocationDialogue Location1" / "10.ogg").is_file()
    )
    results.append(("A1 line-10 alignment (10.ogg == line 10, 'Привет, я Мита!', Mita)", ok))

    # A2: trailing-annotation strip (parser + real row: Location4 line 1
    #     'Мита? // Player' -> text 'Мита?', inline override to Player)
    parser_ok = strip_trailing_comment("Мита // is name of Mita (Check for game) [not]") == "Мита"
    r4 = by_key.get(("LocationDialogue Location4", 1))
    data_ok = (
        r4 is not None
        and r4["text"] == "Мита?"
        and r4["speaker_raw"] == "Player"
        and r4["speaker_source"] == "inline"
    )
    no_annotations_left = all(" // " not in r["text"] for r in rows)
    ok = parser_ok and data_ok and no_annotations_left
    results.append(("A2 trailing-annotation strip (' // ...' removed from text lines)", ok))

    # A3: '- Speaker' marker rule (parser + real row: Location15 line 193 under
    #     'core console - nope programm - Player'; annotation '1' is a description)
    parser_ok = extract_speaker("core console - nope programm - Player") == ("Player", False)
    r15 = by_key.get(("LocationDialogue Location15", 193))
    data_ok = (
        r15 is not None
        and r15["text"] == "Наверное, это не то."
        and r15["speaker_raw"] == "Player"
        and r15["character"] is None
        and r15["speaker_source"] == "marker"
        and r15["inline_note"] == "1"
    )
    ok = parser_ok and data_ok
    results.append(("A3 '- Speaker' marker (speaker = part after LAST ' - ')", ok))

    # A4: inline speaker overrides (reviewer-cited regions)
    a4_checks = [
        (("LocationDialogue Location6", 26), "Player", "inline"),
        (("LocationDialogue Location6", 27), "MitaKind", "inline"),
        (("LocationDialogue Location6", 28), "Player", "inline"),
        (("LocationDialogue Location6", 29), "MitaKind", "inline"),
        (("LocationDialogue Location3", 27), "Player", "inline"),
        (("LocationDialogue Location7", 409), "MitaCap", "inline"),
        (("LocationDialogue Location7", 433), "Player", "inline"),
        (("LocationDialogue Location7", 444), "Player", "inline"),
        (("LocationDialogue Location7", 446), "MitaCap", "inline"),
    ]
    ok = all(
        (row := by_key.get(key)) is not None
        and row["speaker_raw"] == speaker
        and row["speaker_source"] == source
        for key, speaker, source in a4_checks
    )
    results.append(
        ("A4 inline overrides (Location6:26-29, Location3:27, Location7:409-446)", ok)
    )

    # A5: [not] is a tag, never a speaker
    ok = all(r["speaker_raw"] != NOT_TAG for r in rows)
    results.append(("A5 no '[not]' speaker labels in manifest", ok))

    return results


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def minutes(seconds: float | None) -> float:
    return round((seconds or 0.0) / 60.0, 2)


def write_reports(
    cats: list[CategoryStats],
    rows: list[dict],
    selftest: list[tuple[str, bool]],
    known_speakers: set[str],
    annotation_stats: dict[str, int],
    started: float,
) -> None:
    total_source = sum(c.source_ogg for c in cats)
    total_copied = sum(c.copied_ogg for c in cats)
    total_verified = sum(c.verified_ogg for c in cats)
    total_failed = [f"{c.category}/{f}" for c in cats for f in c.failed_ogg]
    total_rows = len(rows)
    total_secs = sum(r["duration_s"] or 0.0 for r in rows)
    unvoiced_with_audio = [
        {"category": c.category, "line": n, "text": t}
        for c in cats
        for n, t in c.unvoiced_with_audio
    ]

    # Per-bucket stats
    buckets: dict[str, dict[str, float]] = {}
    for r in rows:
        key = r["character"] or "(none)"
        b = buckets.setdefault(key, {"clips": 0, "seconds": 0.0})
        b["clips"] += 1
        b["seconds"] += r["duration_s"] or 0.0

    mismatched = [c.category for c in cats if c.copied_ogg != c.source_ogg]
    empty_categories = [
        d.name for d in sorted(SOURCE_AUDIO.iterdir())
        if d.is_dir() and not any(p.suffix.lower() == ".ogg" for p in d.iterdir())
    ]

    report = {
        "source_audio_dir": str(SOURCE_AUDIO),
        "source_texts_dir": str(SOURCE_TEXTS),
        "totals": {
            "source_ogg": total_source,
            "copied_ogg": total_copied,
            "verified_ogg_size_and_sha256": total_verified,
            "copy_failures": total_failed,
            "transcript_files_copied": sum(1 for c in cats if c.transcript_copied),
            "manifest_rows": total_rows,
            "manifest_minutes": minutes(total_secs),
            "mita_rows": sum(1 for r in rows if r["is_mita"]),
            "mita_minutes": minutes(sum(r["duration_s"] or 0.0 for r in rows if r["is_mita"])),
        },
        "per_bucket": {
            k: {"clips": int(v["clips"]), "minutes": minutes(v["seconds"])}
            for k, v in sorted(buckets.items())
        },
        "epsilon": {
            "ogg_skipped_missing_text": sum(c.ogg_skipped_missing_text for c in cats),
            "text_lines_skipped_missing_ogg": sum(c.text_skipped_missing_ogg for c in cats),
            "per_category": {
                c.category: {
                    "source_ogg": c.source_ogg,
                    "copied_ogg": c.copied_ogg,
                    "transcript_copied": c.transcript_copied,
                    "manifest_rows": c.manifest_rows,
                    "ogg_skipped_missing_text": c.ogg_skipped_missing_text,
                    "text_lines_skipped_missing_ogg": c.text_skipped_missing_ogg,
                }
                for c in cats
            },
        },
        "speaker_sources": {
            "rows_from_inline_override": sum(1 for r in rows if r["speaker_source"] == "inline"),
            "rows_from_marker": sum(1 for r in rows if r["speaker_source"] == "marker"),
            "annotated_lines_total": annotation_stats["annotated"],
            "annotations_classified_as_speaker": annotation_stats["speaker"],
            "annotations_classified_as_description": annotation_stats["description"],
        },
        "known_speakers": sorted(known_speakers),
        "unvoiced_lines_with_audio": unvoiced_with_audio,
        "categories_copied_ne_source": mismatched,
        "empty_categories": empty_categories,
        "selftest": [{"name": name, "passed": passed} for name, passed in selftest],
        "elapsed_s": round(time.monotonic() - started, 1),
    }

    OUT_REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # Markdown report
    md: list[str] = []
    md.append("# MiSide Russian voice pack — extraction report\n")
    md.append(f"- Source audio: `{SOURCE_AUDIO}`")
    md.append(f"- Source transcripts: `{SOURCE_TEXTS}`")
    md.append(f"- Elapsed: {report['elapsed_s']} s\n")
    md.append("## Totals\n")
    t = report["totals"]
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| Source ogg | {t['source_ogg']} |")
    md.append(f"| Copied ogg | {t['copied_ogg']} |")
    md.append(f"| Verified byte-identical (size + sha256) | {t['verified_ogg_size_and_sha256']} |")
    md.append(f"| Copy failures | {len(t['copy_failures'])} |")
    md.append(f"| Transcript files copied | {t['transcript_files_copied']} |")
    md.append(f"| Manifest rows | {t['manifest_rows']} |")
    md.append(f"| Manifest audio (minutes) | {t['manifest_minutes']} |")
    md.append(f"| Mita-family rows | {t['mita_rows']} |")
    md.append(f"| Mita-family audio (minutes) | {t['mita_minutes']} |\n")

    md.append("## Per-bucket clips / minutes\n")
    md.append("| character | clips | minutes |")
    md.append("|---|---|---|")
    for k, v in report["per_bucket"].items():
        md.append(f"| {k} | {v['clips']} | {v['minutes']} |")
    md.append("")

    ss = report["speaker_sources"]
    md.append("## Inline speaker overrides\n")
    md.append(
        f"- Annotated text lines: **{ss['annotated_lines_total']}** "
        f"({ss['annotations_classified_as_speaker']} classified as speaker overrides, "
        f"{ss['annotations_classified_as_description']} classified as descriptions)"
    )
    md.append(f"- Manifest rows whose speaker came from an inline override: **{ss['rows_from_inline_override']}**")
    md.append(f"- Manifest rows whose speaker came from marker carry-forward: **{ss['rows_from_marker']}**")
    md.append(f"- Known-speaker set (corpus-derived): {', '.join(report['known_speakers'])}\n")

    md.append("## Unvoiced ([not]) lines that DO have audio\n")
    if unvoiced_with_audio:
        md.append(
            "The [not] tag marks unvoiced blocks; the lines below sit in such "
            "blocks yet have an ogg file, so they remain in the manifest "
            "(audio + text both exist). Flagged for review:\n"
        )
        for u in unvoiced_with_audio:
            md.append(f"- `{u['category']}` line {u['line']}: {u['text']}")
        md.append("")
    else:
        md.append("None — every [not]-marked line lacks audio, as expected.\n")

    md.append("## Categories where copied != source count\n")
    if mismatched:
        for c in mismatched:
            md.append(f"- {c}")
    else:
        md.append("None — every category copied all of its source ogg files.\n")

    md.append("## Epsilon (intersection losses)\n")
    e = report["epsilon"]
    md.append(
        f"- ogg skipped for missing text: **{e['ogg_skipped_missing_text']}** "
        "(audio files with no non-empty transcript line at their line index. "
        "In this pack the ogg side of the intersection is complete — every "
        "ogg file has a text line, so no audio was dropped.)"
    )
    md.append(
        f"- text lines skipped for missing ogg: **{e['text_lines_skipped_missing_ogg']}** "
        "(transcript lines whose N.ogg does not exist in the category — "
        "unvoiced lines, e.g. the [not]-marked ones, or lines never recorded)"
    )
    md.append("")

    md.append("## Notes\n")
    md.append("- Transcript scope: only `LocationDialogue <X>.txt` files are copied "
              "(the other .txt files under Languages/Russian are UI strings, not "
              "voiced dialogue).")
    md.append(f"- Empty source category skipped: "
              + (", ".join(report["empty_categories"]) or "none")
              + " (directory exists but contains no ogg files).")
    md.append("- Source transcripts are UTF-8 **without** BOM (CRLF); read with "
              "`utf-8-sig` for robustness. All outputs are UTF-8 without BOM, LF.")
    md.append("- `[player]` placeholder replaced with `игрок` in `text`; the "
              "pre-replacement form is kept in `text_raw`.")
    md.append("- Inline annotation semantics: a trailing `// Note` is stripped "
              "from `text`; if Note normalises to a known speaker label it "
              "overrides the marker for that line only (`speaker_source: "
              "\"inline\"`), otherwise it is kept in `inline_note` and the "
              "marker's speaker stands.")
    md.append("")

    md.append("## Self-test\n")
    for name, passed in selftest:
        md.append(f"- [{'PASS' if passed else 'FAIL'}] {name}")
    md.append("")

    OUT_REPORT_MD.write_text("\n".join(md), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    started = time.monotonic()
    if not SOURCE_AUDIO.is_dir():
        print(f"ERROR: source audio dir not found: {SOURCE_AUDIO}", file=sys.stderr)
        return 1
    if not SOURCE_TEXTS.is_dir():
        print(f"ERROR: source texts dir not found: {SOURCE_TEXTS}", file=sys.stderr)
        return 1

    categories = discover_categories()
    print(f"Discovered {len(categories)} categories with ogg files")

    cats: list[CategoryStats] = []
    stats_by_name: dict[str, CategoryStats] = {}
    all_rows: list[dict] = []
    copied_pairs: list[tuple[Path, Path]] = []  # (src, dst) for verification
    raw_texts: dict[str, str | None] = {}

    # 1) Copy audio + transcripts
    for name, oggs in categories:
        stats = CategoryStats(
            category=name,
            source_ogg=len(oggs),
            non_numeric_ogg=sum(1 for p in oggs if not p.stem.isdigit()),
        )
        cats.append(stats)
        stats_by_name[name] = stats
        for p in oggs:
            dst = OUT_AUDIO / name / p.name
            copy_file(p, dst)
            stats.copied_ogg += 1
            copied_pairs.append((p, dst))
        txt_src = SOURCE_TEXTS / f"{name}.txt"
        if txt_src.is_file():
            copy_file(txt_src, OUT_TEXTS / f"{name}.txt")
            stats.transcript_copied = True
            raw_texts[name] = txt_src.read_text(encoding="utf-8-sig")
        else:
            raw_texts[name] = None

    # 2) Verify copies (size + sha256, all files)
    print(f"Verifying {len(copied_pairs)} copied ogg files (size + sha256)...")
    for src, dst in copied_pairs:
        stats = stats_by_name[dst.parent.name]
        if verify_copy(src, dst):
            stats.verified_ogg += 1
        else:
            stats.failed_ogg.append(dst.name)

    # 3) Pass 1: physical parse + corpus speaker statistics
    raw_lines: dict[str, list[RawLine]] = {}
    marker_freq: Counter[str] = Counter()
    inline_annotations: set[str] = set()
    for name, text in raw_texts.items():
        if text is None:
            continue
        lines = parse_transcript(text)
        raw_lines[name] = lines
        for ln in lines:
            if ln.kind == "marker":
                speaker, _ = extract_speaker(ln.marker_body)
                if speaker:
                    marker_freq[speaker] += 1
            elif ln.kind == "text" and ln.annotation is not None:
                ann_speaker, _ = extract_speaker(ln.annotation)
                if ann_speaker:
                    inline_annotations.add(ann_speaker)
    known_speakers = build_known_speakers(marker_freq, inline_annotations)
    print(f"Known speaker set (corpus-derived): {sorted(known_speakers)}")

    # 4) Pass 2: speaker resolution
    resolved: dict[str, list[ResolvedLine]] = {
        name: resolve_lines(lines, known_speakers) for name, lines in raw_lines.items()
    }

    # 5) Probe durations (thread pool)
    dst_paths = [dst for _, dst in copied_pairs]
    print(f"Probing durations of {len(dst_paths)} files with {FFPROBE_WORKERS} threads...")
    durations = probe_durations(dst_paths)
    n_none = sum(1 for v in durations.values() if v is None)
    if n_none:
        print(f"WARNING: {n_none} files failed duration probing", file=sys.stderr)

    # 6) Build manifest rows (intersect ogg x text)
    annotation_stats = Counter()
    for lines in raw_lines.values():
        for ln in lines:
            if ln.kind == "text" and ln.annotation is not None:
                annotation_stats["annotated"] += 1
    for name, lines in raw_lines.items():
        for ln in lines:
            if ln.kind == "text" and ln.annotation is not None:
                ann_speaker, _ = extract_speaker(ln.annotation)
                if ann_speaker is not None and ann_speaker in known_speakers:
                    annotation_stats["speaker"] += 1
                else:
                    annotation_stats["description"] += 1

    for name, oggs in categories:
        stats = stats_by_name[name]
        numeric_oggs = [p for p in oggs if p.stem.isdigit()]
        rows, eps1, eps2, unv = build_manifest_rows(
            name, numeric_oggs, resolved.get(name, []), durations
        )
        stats.manifest_rows = len(rows)
        stats.ogg_skipped_missing_text = eps1
        stats.text_skipped_missing_ogg = eps2
        stats.inline_speaker_rows = sum(1 for r in rows if r["speaker_source"] == "inline")
        stats.unvoiced_with_audio = unv
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["category"], r["id"]))

    # 7) Write manifest (UTF-8 no BOM, LF)
    with OUT_MANIFEST.open("w", encoding="utf-8", newline="\n") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_rows)} rows to {OUT_MANIFEST.relative_to(PROJECT_ROOT)}")

    # 8) Self-test
    selftest = run_selftest(all_rows)
    for name, passed in selftest:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    # 9) Reports
    write_reports(cats, all_rows, selftest, known_speakers, dict(annotation_stats), started)
    print(f"Wrote {OUT_REPORT_JSON.relative_to(PROJECT_ROOT)} and {OUT_REPORT_MD.relative_to(PROJECT_ROOT)}")

    # 10) Failure gate
    problems: list[str] = []
    for c in cats:
        if c.failed_ogg:
            problems.append(f"{c.category}: {len(c.failed_ogg)} copy verification failures")
        if not c.transcript_copied and c.source_ogg > 0:
            problems.append(
                f"{c.category}: {c.source_ogg} ogg files but no transcript found"
            )
    if not all_rows:
        problems.append("manifest is empty")
    failed_tests = [name for name, passed in selftest if not passed]
    if failed_tests:
        problems.append("self-test failures: " + "; ".join(failed_tests))

    n_unv_audio = sum(len(c.unvoiced_with_audio) for c in cats)
    if n_unv_audio:
        print(
            f"WARNING: {n_unv_audio} [not]-marked line(s) have audio — "
            "kept in manifest, listed in report",
            file=sys.stderr,
        )

    if problems:
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        print("ERROR: extraction completed with failures — see report", file=sys.stderr)
        return 1
    print("OK: extraction complete, all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
