r"""Score every checkpoint of a run with the frozen eval harness and apply
the agreed decision rule.

Usage (vendor venv):
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\eval_ladder.py --run-dir runs\mita_v1
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\eval_ladder.py --run-dir runs\mita_v1 --reuse
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\eval_ladder.py --self-test

Discovers checkpoint-* dirs (numeric order), names variants lora-<step
zero-padded to 4>, and calls eval\run_eval.py --variant <name>
--lora-adapter <abs ckpt> --instruct mita via subprocess (vendor python);
aborts on non-zero exit. --reuse skips variants whose
eval/results/<variant>.json already exists.

Decision rule:
  (1) a variant with wer_verdict == "fail" is not rankable (veto);
  (2) rank survivors by sim_best_mean;
  (3) declare a winner only if the paired bootstrap CI vs the runner-up
      excludes 0 AND |delta| >= 0.05 — otherwise "TIE: listen to <A>, <B>";
  (4) fewer than 2 survivors -> say so.

Writes runs/<run>/ladder.json and runs/<run>/ladder.md (table + verdict +
exact commands used).

--self-test exercises the ranking logic on synthetic result JSONs in
tmp/ladder_selftest/ (copies of a real result with tweaked per-phrase
sim_best values): veto, tie at delta < 0.05, winner at delta >= 0.05 with
CI excluding 0. No GPU.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
TEMPLATE_RESULT = RESULTS / "base_mita_tag.json"
SELFTEST_DIR = ROOT / "tmp" / "ladder_selftest"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261115
DELTA_MIN = 0.05
FROZEN_N = 20  # the frozen phrase set size; partial runs must never enter the ladder


def die(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def step_of(name: str) -> int:
    m = re.search(r"checkpoint-(\d+)", name)
    if not m:
        die(f"cannot parse step from {name}")
    return int(m.group(1))


def variant_of(step: int) -> str:
    return f"lora-{step:04d}"


def discover_checkpoints(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        die(f"run dir does not exist: {run_dir}")
    ckpts = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    ckpts.sort(key=lambda p: step_of(p.name))
    return ckpts


def run_eval_for(ckpt: Path, variant: str, reuse: bool) -> Path:
    result_path = RESULTS / f"{variant}.json"
    if reuse and result_path.is_file():
        print(f"[reuse] {variant}: {result_path} already exists, skipping run_eval")
        return result_path
    cmd = [
        sys.executable,
        str(ROOT / "eval" / "run_eval.py"),
        "--variant", variant,
        "--lora-adapter", str(ckpt),
        "--instruct", "mita",
    ]
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        die(f"run_eval.py failed (exit {proc.returncode}) for {variant}")
    if not result_path.is_file():
        die(f"run_eval.py exited 0 but {result_path} was not written")
    return result_path


# ---------------------------------------------------------------------------
# Decision rule (pure — also used by --self-test)
# ---------------------------------------------------------------------------

def _paired_ci(a, b, n_boot: int, seed: int):
    import numpy as np

    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean()
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def check_full_results(variants: dict[str, dict]) -> None:
    """Refuse any variant with n_phrases != FROZEN_N (a --limit run is not rankable)."""
    bad = {n: d.get("n_phrases") for n, d in variants.items() if d.get("n_phrases") != FROZEN_N}
    if bad:
        die(f"partial run(s) detected (n_phrases != {FROZEN_N}): {bad} — refusing to rank")


def decide(variants: dict[str, dict], n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict:
    """Apply the decision rule to {variant_name: result_doc}.

    Pure function of the result docs; no filesystem, no GPU.
    """
    check_full_results(variants)
    table = []
    for name in sorted(variants):
        doc = variants[name]
        sim = doc.get("sim") or {}
        wer = doc.get("wer") or {}
        m = re.search(r"lora-(\d+)", name)
        table.append({
            "variant": name,
            "step": int(m.group(1)) if m else None,
            "wer_pct": wer.get("wer_pct"),
            "wer_verdict": doc.get("wer_verdict"),
            "sim_best_mean": sim.get("sim_best_mean"),
            "sim_best_std": sim.get("sim_best_std"),
            "sim_centroid_mean": sim.get("sim_centroid_mean"),
            "flags": doc.get("flags", []),
        })

    survivors = [n for n in variants if (variants[n].get("wer_verdict") or "fail") != "fail"]
    if len(survivors) < 2:
        return {
            "verdict": "insufficient",
            "message": f"fewer than 2 WER-passing survivors ({len(survivors)}): {sorted(survivors)} — cannot rank",
            "ranked": survivors,
            "table": table,
        }

    ranked = sorted(survivors, key=lambda n: (variants[n].get("sim") or {}).get("sim_best_mean") or -1.0, reverse=True)
    top, runner = ranked[0], ranked[1]
    for n in (top, runner):
        if not (variants[n].get("sim") or {}).get("per_phrase"):
            die(f"variant {n} has no sim per-phrase data — cannot bootstrap (was it run with --skip-sim?)")
    tp = {p["id"]: p["sim_best"] for p in variants[top]["sim"]["per_phrase"]}
    rp = {p["id"]: p["sim_best"] for p in variants[runner]["sim"]["per_phrase"]}
    ids = sorted(set(tp) & set(rp))
    if len(ids) < 5:
        die(f"paired bootstrap needs >= 5 common phrase ids, have {len(ids)}")
    import numpy as np

    delta, lo, hi = _paired_ci(np.array([tp[i] for i in ids]), np.array([rp[i] for i in ids]), n_boot, seed)
    ci_excludes_zero = bool(lo > 0 or hi < 0)
    detail = {
        "top": top, "runner_up": runner, "n_phrases": len(ids),
        "delta_vs_runner_up": round(delta, 6), "ci95": [round(lo, 6), round(hi, 6)],
        "ci_excludes_zero": ci_excludes_zero, "delta_ge_0.05": abs(delta) >= DELTA_MIN,
    }
    if ci_excludes_zero and abs(delta) >= DELTA_MIN:
        return {
            "verdict": "winner", "detail": detail, "ranked": ranked, "table": table,
            "message": f"WINNER: {top} (delta {delta:+.4f} vs {runner}, CI95 [{lo:+.4f}, {hi:+.4f}])",
        }
    return {
        "verdict": "tie", "detail": detail, "ranked": ranked, "table": table,
        "message": f"TIE: listen to {top}, {runner} (delta {delta:+.4f}, CI95 [{lo:+.4f}, {hi:+.4f}])",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(table: list[dict]) -> None:
    print(f"{'step':>6}  {'variant':<12} {'WER %':>7} {'verdict':<6} {'sim_best_mean ± std':<22} {'sim_centroid':>12}  flags")
    for r in table:
        sbm = f"{r['sim_best_mean']:.4f}" if r["sim_best_mean"] is not None else "  n/a"
        sbs = f" ± {r['sim_best_std']:.4f}" if r["sim_best_std"] is not None else ""
        scm = f"{r['sim_centroid_mean']:.4f}" if r["sim_centroid_mean"] is not None else "n/a"
        wer = f"{r['wer_pct']:.2f}" if r["wer_pct"] is not None else "n/a"
        print(f"{r['step'] or '?':>6}  {r['variant']:<12} {wer:>7} {str(r['wer_verdict']):<6} "
              f"{sbm + sbs:<22} {scm:>12}  {','.join(r['flags']) or '-'}")


def write_ladder(run_dir: Path, decision: dict, commands: list[str]) -> tuple[Path, Path]:
    doc = {
        "run_dir": str(run_dir),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_rule": "WER veto; rank by sim_best_mean; winner iff paired bootstrap CI vs runner-up excludes 0 AND |delta| >= 0.05",
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
        "table": decision["table"],
        "ranked": decision["ranked"],
        "detail": decision.get("detail"),
        "verdict": decision["verdict"],
        "message": decision["message"],
        "commands": commands,
    }
    json_path = run_dir / "ladder.json"
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    lines = [
        f"# Eval ladder — {run_dir}",
        "",
        f"Generated: {doc['generated']}",
        "",
        "| step | variant | WER % | WER verdict | sim_best_mean ± std | sim_centroid_mean | flags |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in decision["table"]:
        sbm = f"{r['sim_best_mean']:.4f}" if r["sim_best_mean"] is not None else "n/a"
        sbs = f" ± {r['sim_best_std']:.4f}" if r["sim_best_std"] is not None else ""
        scm = f"{r['sim_centroid_mean']:.4f}" if r["sim_centroid_mean"] is not None else "n/a"
        wer = f"{r['wer_pct']:.2f}" if r["wer_pct"] is not None else "n/a"
        lines.append(
            f"| {r['step'] or '?'} | {r['variant']} | {wer} | {r['wer_verdict']} | {sbm}{sbs} | {scm} | {','.join(r['flags']) or '-'} |"
        )
    lines += ["", "## Verdict", "", decision["message"], ""]
    d = decision.get("detail")
    if d:
        lines += [
            f"- delta (top − runner-up): {d['delta_vs_runner_up']:+.4f} over {d['n_phrases']} phrases",
            f"- paired bootstrap CI95: [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] (n={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED})",
            f"- CI excludes 0: {d['ci_excludes_zero']}; |delta| >= 0.05: {d['delta_ge_0.05']}",
            "",
        ]
    lines += ["## Commands used", "", "```"]
    lines += commands
    lines += ["```", ""]
    md_path = run_dir / "ladder.md"
    md_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return json_path, md_path


# ---------------------------------------------------------------------------
# Self-test on synthetic results (no GPU)
# ---------------------------------------------------------------------------

def _synth_doc(template: dict, name: str, sim_shift: float = 0.0,
               sim_shifts: list[float] | None = None, wer_fail: bool = False,
               n_phrases: int | None = None) -> dict:
    import copy

    doc = copy.deepcopy(template)
    doc["variant"] = name
    doc["lora_adapter"] = f"<synthetic {name}>"
    per = doc["sim"]["per_phrase"]
    if n_phrases is not None:
        per = per[:n_phrases]
        doc["sim"]["per_phrase"] = per
        doc["wer"]["per_phrase"] = doc["wer"]["per_phrase"][:n_phrases]
        doc["n_phrases"] = n_phrases
    shifts = sim_shifts if sim_shifts is not None else [sim_shift] * len(per)
    for p, s in zip(per, shifts):
        p["sim_best"] = round(p["sim_best"] + s, 6)
    vals = [p["sim_best"] for p in per]
    import numpy as np

    doc["sim"]["sim_best_mean"] = round(float(np.mean(vals)), 6)
    doc["sim"]["sim_best_std"] = round(float(np.std(vals)), 6)
    if wer_fail:
        doc["wer"]["wer_pct"] = 12.5
        doc["wer"]["per_phrase"][0]["wer_pct"] = 60.0
        doc["wer_verdict"] = "fail"
    else:
        doc["wer_verdict"] = "pass"
    return doc


def self_test() -> None:
    if not TEMPLATE_RESULT.is_file():
        die(f"self-test template missing: {TEMPLATE_RESULT}")
    template = json.loads(TEMPLATE_RESULT.read_text(encoding="utf-8"))
    SELFTEST_DIR.mkdir(parents=True, exist_ok=True)

    # per-phrase noise with zero mean and +/-0.30 spread: keeps delta at 0.10
    # but makes the paired bootstrap CI include 0 (a deleted CI check would
    # crown this a winner)
    noise = [0.30 if i % 2 == 0 else -0.30 for i in range(FROZEN_N)]
    scenarios = {
        # veto: one variant fails WER -> only 1 survivor -> cannot rank
        "veto": {"lora-0100": _synth_doc(template, "lora-0100", wer_fail=True),
                 "lora-0200": _synth_doc(template, "lora-0200")},
        # tie: both pass, constant +0.01 shift -> CI excludes 0 but delta 0.01 < 0.05
        "tie": {"lora-0200": _synth_doc(template, "lora-0200"),
                "lora-0300": _synth_doc(template, "lora-0300", sim_shift=0.01)},
        # winner: both pass, constant +0.10 shift -> delta 0.10 >= 0.05, CI [0.10, 0.10]
        "winner": {"lora-0400": _synth_doc(template, "lora-0400"),
                   "lora-0500": _synth_doc(template, "lora-0500", sim_shift=0.10)},
        # noisy: delta 0.10 >= 0.05 but per-phrase spread +/-0.30 -> CI includes 0 -> TIE
        "noisy": {"lora-0800": _synth_doc(template, "lora-0800"),
                  "lora-0900": _synth_doc(template, "lora-0900", sim_shifts=[0.10 + s for s in noise])},
        # partial: a 5-phrase result vs a full 20-phrase one must be refused outright
        "partial": {"lora-0600": _synth_doc(template, "lora-0600", n_phrases=5),
                    "lora-0700": _synth_doc(template, "lora-0700")},
    }
    expected = {
        "veto": ("verdict", lambda d: d["verdict"] == "insufficient"),
        "tie": ("verdict", lambda d: d["verdict"] == "tie" and "TIE: listen to" in d["message"]
                              and d["detail"]["ci_excludes_zero"]
                              and abs(d["detail"]["delta_vs_runner_up"]) < DELTA_MIN),
        "winner": ("verdict", lambda d: d["verdict"] == "winner" and d["detail"]["top"] == "lora-0500"
                               and d["detail"]["ci_excludes_zero"] and d["detail"]["delta_ge_0.05"]),
        "noisy": ("verdict", lambda d: d["verdict"] == "tie" and not d["detail"]["ci_excludes_zero"]
                               and d["detail"]["delta_ge_0.05"]),
        "partial": ("exit", None),
    }

    all_ok = True
    for scen, variants in scenarios.items():
        scen_dir = SELFTEST_DIR / scen / "results"
        scen_dir.mkdir(parents=True, exist_ok=True)
        for name, doc in variants.items():
            (scen_dir / f"{name}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
        kind, check = expected[scen]
        try:
            d = decide(variants)
            exited = False
        except SystemExit:
            d, exited = None, True
        ok = exited if kind == "exit" else (not exited and check(d))
        all_ok &= ok
        if d is not None:
            print(f"[self-test] {scen:<8} -> {d['verdict']:<13} {d['message']}")
            if d.get("detail"):
                print(f"            delta={d['detail']['delta_vs_runner_up']:+.4f} "
                      f"CI95=[{d['detail']['ci95'][0]:+.4f}, {d['detail']['ci95'][1]:+.4f}] "
                      f"(ci_excludes_zero={d['detail']['ci_excludes_zero']}, delta>=0.05={d['detail']['delta_ge_0.05']})")
        else:
            print(f"[self-test] {scen:<8} -> REFUSED (SystemExit) — partial run rejected before ranking")
        print(f"            expected: {'OK' if ok else 'MISMATCH — self-test FAILED'}")
    if not all_ok:
        die("self-test failed")
    print(f"self-test PASS (synthetic results in {SELFTEST_DIR})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", help="training run dir containing checkpoint-* (e.g. runs\\mita_v1)")
    ap.add_argument("--reuse", action="store_true", help="skip variants whose eval/results/<variant>.json already exists")
    ap.add_argument("--self-test", action="store_true", help="exercise the ranking logic on synthetic results (no GPU)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.run_dir:
        die("--run-dir is required (or use --self-test)")
    run_dir = Path(args.run_dir).resolve()
    ckpts = discover_checkpoints(run_dir)
    if not ckpts:
        die(f"no checkpoint-* dirs under {run_dir}")

    variants = {}
    commands = []
    for ckpt in ckpts:
        variant = variant_of(step_of(ckpt.name))
        result_path = run_eval_for(ckpt, variant, args.reuse)
        commands.append(f"{sys.executable} eval\\run_eval.py --variant {variant} --lora-adapter {ckpt} --instruct mita")
        variants[variant] = json.loads(result_path.read_text(encoding="utf-8"))

    # covers --reuse too: a stale partial result must not be crowned
    check_full_results(variants)
    decision = decide(variants)
    print_table(decision["table"])
    print(f"\nverdict: {decision['verdict']}")
    print(decision["message"])
    json_path, md_path = write_ladder(run_dir, decision, commands)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
