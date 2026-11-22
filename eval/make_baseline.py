"""Derive eval/baseline.json from the two baseline variant result JSONs.

Machine-generated: baseline.json is a pure function of
results/base_native.json + results/base_mita_tag.json — never hand-edited.
Called automatically by eval/run_eval.py when both baseline results exist;
can also be run standalone:

    vendor\\OmniVoice\\.venv\\Scripts\\python.exe eval\\make_baseline.py

Refuses (raising BaselineError, leaving any existing baseline.json untouched)
when a variant result is a partial run (n_phrases != 20) or carries null
headline values — a baseline must never be downgraded to nulls.
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("base_native", "base_mita_tag")
FULL_N_PHRASES = 20  # the frozen phrase set size


class BaselineError(RuntimeError):
    """A baseline variant result is incomplete; baseline.json must not be (re)written."""


def headline(v: dict) -> dict:
    wer = v.get("wer") or {}
    sim = v.get("sim") or {}
    gen = v.get("generation") or {}
    return {
        "instruct": v["instruct"],
        "wer_pct": wer.get("wer_pct"),
        "S": wer.get("S"),
        "D": wer.get("D"),
        "I": wer.get("I"),
        "ref_words": wer.get("ref_words"),
        "wer_verdict": v.get("wer_verdict"),
        "sim_best_mean": sim.get("sim_best_mean"),
        "sim_best_std": sim.get("sim_best_std"),
        "sim_best_ceiling": (sim.get("sim_best_ceiling") or {}).get("mean"),
        "sim_centroid_mean": sim.get("sim_centroid_mean"),
        "sim_centroid_std": sim.get("sim_centroid_std"),
        "sim_centroid_ceiling": (sim.get("sim_centroid_ceiling") or {}).get("mean"),
        "audio_seconds_total": round(sum(p["seconds_audio"] for p in gen.get("per_phrase", [])), 3),
        "flags": v.get("flags", []),
    }


def validate(name: str, v: dict) -> None:
    """Refuse partial runs and null headline values (baseline.json must never be downgraded)."""
    if not v.get("instruct"):
        raise BaselineError(f"{name}: missing instruct")
    n = v.get("n_phrases")
    if n != FULL_N_PHRASES:
        raise BaselineError(f"{name}: n_phrases={n!r} != {FULL_N_PHRASES} — partial run, not a baseline")
    nulls = [k for k, val in headline(v).items() if val is None]
    if nulls:
        raise BaselineError(f"{name}: null headline values {nulls} — refusing to downgrade baseline.json")


def build(results_dir: Path) -> dict:
    results_dir = Path(results_dir)
    variants = {}
    created = ""
    for name in VARIANTS:
        p = results_dir / f"{name}.json"
        if not p.is_file():
            raise BaselineError(f"missing {p} — run the baseline job first")
        v = json.loads(p.read_text(encoding="utf-8"))
        validate(name, v)
        variants[name] = headline(v)
        created = max(created, v.get("created", ""))  # ISO timestamps: string max == time max
    doc = {
        "created": created,
        "phrases_sha256": json.loads((ROOT / "eval" / "frozen.json").read_text(encoding="utf-8"))["phrases_sha256"],
        "seed": 1234,
        "model": "k2-fsa/OmniVoice",
        "lora_adapter": None,
        "sim_primary": "max_over_refs",
        "variants": variants,
    }
    out = ROOT / "eval" / "baseline.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return doc


if __name__ == "__main__":
    try:
        doc = build(ROOT / "eval" / "results")
    except BaselineError as e:
        sys.exit(f"make_baseline: {e}")
    print(json.dumps(doc, ensure_ascii=False, indent=2))
