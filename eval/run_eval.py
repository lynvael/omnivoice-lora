r"""One driver, one command per variant.

Usage (vendor venv):
    vendor\OmniVoice\.venv\Scripts\python.exe eval\run_eval.py --variant base_native --instruct "female, young adult"
    vendor\OmniVoice\.venv\Scripts\python.exe eval\run_eval.py --variant base_mita_tag --instruct mita
    vendor\OmniVoice\.venv\Scripts\python.exe eval\run_eval.py --variant lora-600 --lora-adapter <ckpt dir> --instruct mita
    vendor\OmniVoice\.venv\Scripts\python.exe eval\run_eval.py --variant base_native --instruct "female, young adult" --reuse-audio

Steps: (1) render with scripts.mita_voice.generate (or verify + reuse frozen
audio with --reuse-audio), (2) free the TTS model and torch.cuda.empty_cache(),
(3) shell out to eval/wer.py with the PROJECT venv (no torch there), (4) run
speaker similarity in-process, (5) merge into results/<variant>.json, update
results/index.json, and regenerate eval/baseline.json (machine-generated).

Failure policy: a failed WER or SIM stage is FATAL — the driver exits
non-zero, writes no results/<variant>.json, and does not touch index.json.
Only --skip-wer / --skip-sim produce a partial JSON, and the skipped field is
absent (not null). Stage logs live in <results-dir>/logs/.

The JSON shape is identical for every variant (base or LoRA): present stages
carry values, skipped stages are absent.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROJECT_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
BASELINE_VARIANTS = ("base_native", "base_mita_tag")
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261115


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, help="variant name, e.g. base_native, lora-600")
    ap.add_argument("--instruct", required=True, help='voice-design instruct, e.g. "female, young adult" or "mita"')
    ap.add_argument("--phrases", default=str(ROOT / "eval" / "phrases.jsonl"))
    ap.add_argument("--audio-root", default=str(ROOT / "eval" / "audio"))
    ap.add_argument("--results-dir", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--seed", type=int, default=1234)
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--lora-adapter", default=None, help="LoRA adapter checkpoint dir")
    m.add_argument("--model", default=None,
                   help="merged model dir (from scripts/merge_lora.py); mutually exclusive with --lora-adapter")
    ap.add_argument("--limit", type=int, default=0, help="render/score only the first N phrases (propagated to wer.py and speaker_sim.py)")
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--guidance-scale", type=float, default=2.0)
    ap.add_argument("--t-shift", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty audio dir")
    ap.add_argument("--reuse-audio", action="store_true",
                    help="skip rendering; verify <audio-root>/<variant>/ holds exactly the expected wavs "
                         "(sha256, phrase sha, seeds, instruct, adapter, gen params) and reuse it")
    ap.add_argument("--skip-wer", action="store_true")
    ap.add_argument("--skip-sim", action="store_true")
    return ap


def stage_render(args, audio_dir: Path):
    from scripts.mita_voice.generate import run as render

    t0 = time.time()
    meta = render(
        phrases=args.phrases,
        out_dir=str(audio_dir),
        instruct=args.instruct,
        lora_adapter=args.lora_adapter,
        model=args.model,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        seed=args.seed,
        limit=args.limit,
        device=args.device,
        force=args.force,
    )
    # (2) free the TTS model (render() already dropped it) and release CUDA memory
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    return meta, time.time() - t0


def stage_reuse(args, audio_dir: Path, results_dir: Path) -> dict:
    """--reuse-audio: prove the existing audio belongs to THIS variant config.

    Checks: exact wav set, phrases sha256, per-phrase seeds, instruct,
    lora-adapter identity (content hash when the meta carries adapter_sha256,
    loud basename-only fallback otherwise), generation params, per-wav sha256
    vs disk, and the SIM reference-set hash vs the value recorded in previous
    baseline results.
    """
    from eval.speaker_sim import REFS_PATH, refs_sha256
    from scripts.merge_lora import MARKER
    from scripts.mita_voice.generate import MODEL_NAME, adapter_weights_path

    meta_path = audio_dir / "generation_meta.json"
    if not meta_path.is_file():
        sys.exit(f"--reuse-audio: missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    phrases_path = Path(args.phrases)
    rows = [json.loads(l) for l in phrases_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]
    expected_ids = [r["id"] for r in rows]

    wav_ids = {w.stem for w in audio_dir.glob("*.wav")}
    missing, extra = set(expected_ids) - wav_ids, wav_ids - set(expected_ids)
    if missing or extra:
        sys.exit(f"--reuse-audio: wav set mismatch in {audio_dir}: missing={sorted(missing)} extra={sorted(extra)}")

    sha = hashlib.sha256(phrases_path.read_bytes()).hexdigest()
    if meta.get("phrases_sha256") != sha:
        sys.exit(f"--reuse-audio: phrases_sha256 mismatch: meta {meta.get('phrases_sha256')!r} != current phrases {sha!r}")

    seeds = {p["id"]: p["seed"] for p in meta.get("per_phrase", [])}
    for i, rid in enumerate(expected_ids):
        want = args.seed * 1000 + i
        if seeds.get(rid) != want:
            sys.exit(f"--reuse-audio: seed mismatch for {rid}: meta {seeds.get(rid)!r} != expected {want}")

    # generation parameters must match this invocation exactly
    if meta.get("instruct") != args.instruct:
        sys.exit(f"--reuse-audio: instruct mismatch: meta {meta.get('instruct')!r} != CLI {args.instruct!r}")
    # merged model identity: the audio must have been rendered with THIS merged
    # model dir (same resolved path, same model.safetensors content)
    if args.model:
        # a dir carrying the merge marker is a half-written model (see merge_lora.py);
        # mirrors the check in generate.py so a stale dir is never reusable
        if (Path(args.model) / MARKER).is_file():
            sys.exit(
                f"--reuse-audio: refusing to reuse {args.model}: {MARKER} marker present — "
                "a stale or interrupted merge dir is not reusable. Rerun scripts/merge_lora.py "
                "(it verifies the artifact and removes the marker) or delete the marker if the dir is known-good."
            )
        if meta.get("lora_adapter") is not None:
            sys.exit(
                f"--reuse-audio: --model given but meta has lora_adapter={meta.get('lora_adapter')!r} "
                "(a merged model must not carry an adapter)"
            )
        meta_model = meta.get("model")
        want = str(Path(args.model).resolve())
        if not meta_model or str(Path(str(meta_model)).resolve()) != want:
            sys.exit(f"--reuse-audio: model mismatch: meta {meta_model!r} != CLI {args.model!r} (resolved {want!r})")
        meta_sha = meta.get("model_sha256")
        if meta_sha is not None:
            weights = Path(args.model) / "model.safetensors"
            if not weights.is_file():
                sys.exit(f"--reuse-audio: meta has model_sha256 but {weights} is missing")
            cur_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
            if cur_sha != meta_sha:
                sys.exit(f"--reuse-audio: model_sha256 mismatch: meta {meta_sha} != disk {cur_sha} ({weights})")
        else:
            print(
                f"[warn] WEAK MODEL CHECK: {meta_path} has no model_sha256 (predates the content-hash field); "
                f"model identity verified by PATH ONLY ({meta_model!r}) — a different merged model saved to "
                "the same directory is indistinguishable. Re-render with the current generate.py to record model_sha256."
            )
    elif meta.get("model") not in (None, MODEL_NAME):
        # a merged-model render must never be reused as a base/adapter variant:
        # without --model the adapter fallback below would compare "None" == "None"
        sys.exit(
            f"--reuse-audio: meta says model={meta.get('model')!r} but CLI has no --model "
            "(this audio was rendered with a merged model; pass --model <that dir>)"
        )
    # adapter identity: content-based (sha256 of the weights file) when the meta
    # carries adapter_sha256; loud basename-only fallback for metas that predate it
    meta_adapter = meta.get("lora_adapter")
    meta_adapter_sha = meta.get("adapter_sha256")
    if meta_adapter_sha is not None:
        if not args.lora_adapter:
            sys.exit(
                f"--reuse-audio: lora_adapter mismatch: meta has adapter sha256 {meta_adapter_sha} "
                f"({meta_adapter!r}) but CLI has no --lora-adapter"
            )
        cur_sha = hashlib.sha256(adapter_weights_path(args.lora_adapter).read_bytes()).hexdigest()
        if cur_sha != meta_adapter_sha:
            sys.exit(
                f"--reuse-audio: adapter_sha256 mismatch: meta {meta_adapter_sha} != disk {cur_sha} "
                f"(adapter {args.lora_adapter!r})"
            )
    else:
        if Path(str(meta_adapter)).name != Path(str(args.lora_adapter)).name:
            sys.exit(
                f"--reuse-audio: lora_adapter mismatch: meta {meta_adapter!r} != CLI {args.lora_adapter!r}"
            )
        if meta_adapter is not None:
            print(
                f"[warn] WEAK ADAPTER CHECK: {meta_path} has no adapter_sha256 (predates the content-hash field); "
                f"adapter identity verified by BASENAME ONLY ({Path(str(meta_adapter)).name!r}) — two different "
                "checkpoints with the same directory name are indistinguishable. Re-render with the current "
                "generate.py to record adapter_sha256."
            )
    for field in ("num_step", "guidance_scale", "t_shift"):
        cli_val = getattr(args, field.replace("-", "_"))
        if (meta.get("params") or {}).get(field) != cli_val:
            sys.exit(f"--reuse-audio: {field} mismatch: meta {(meta.get('params') or {}).get(field)!r} != CLI {cli_val!r}")

    # content hashes: every wav on disk must match the hash of the file that was written
    for p in meta.get("per_phrase", []):
        if p["id"] not in expected_ids:
            continue
        wav = audio_dir / f"{p['id']}.wav"
        disk_sha = hashlib.sha256(wav.read_bytes()).hexdigest()
        if p.get("wav_sha256") != disk_sha:
            sys.exit(f"--reuse-audio: wav_sha256 mismatch for {p['id']}: meta {p.get('wav_sha256')!r} != disk {disk_sha!r}")

    # SIM anchor integrity: the 60 ref files must match the hash recorded in previous results
    if REFS_PATH.is_file():
        refs_doc = json.loads(REFS_PATH.read_text(encoding="utf-8"))
        cur_refs_sha = refs_sha256(refs_doc["refs"])
        recorded = set()
        for f in results_dir.glob("*.json"):
            if f.name == "index.json":
                continue
            try:
                sim = json.loads(f.read_text(encoding="utf-8")).get("sim") or {}
            except json.JSONDecodeError:
                continue
            rec = (sim.get("refs") or {}).get("refs_sha256")
            if rec:
                recorded.add(rec)
        if recorded and cur_refs_sha not in recorded:
            sys.exit(
                f"--reuse-audio: SIM reference set changed: current refs_sha256 {cur_refs_sha!r} "
                f"!= recorded {sorted(recorded)!r} — the frozen 60-ref anchor was modified"
            )
    return meta


def stage_wer(args, audio_dir: Path, results_dir: Path, variant: str) -> dict:
    """Shell out to eval/wer.py with the PROJECT venv (no torch). Fatal on failure."""
    wer_json = results_dir / f"{variant}.wer.json"
    log_path = results_dir / "logs" / f"{variant}.wer.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if wer_json.exists():
        wer_json.unlink()  # a stale wer.json must never survive a failed run
    cmd = [
        str(PROJECT_PYTHON),
        str(ROOT / "eval" / "wer.py"),
        "--phrases", args.phrases,
        "--audio-dir", str(audio_dir),
        "--out", str(wer_json),
        "--variant", variant,
    ]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.time() - t0
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n")
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(f"WER stage failed (exit {proc.returncode}); see {log_path} — no results written, index untouched")
    return json.loads(wer_json.read_text(encoding="utf-8")), wall


def stage_sim(args, audio_dir: Path, variant: str):
    """Run speaker similarity in-process. Fatal on any failure."""
    from eval.speaker_sim import compute

    t0 = time.time()
    try:
        result = compute(
            audio_dir=str(audio_dir),
            variant=variant,
            device=args.device,
            phrases=args.phrases,
            limit=args.limit,
        )
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        sys.exit(f"SIM stage failed: {type(e).__name__}: {e} — no results written, index untouched")
    return result, time.time() - t0


def wer_verdict(wer: dict) -> str:
    """WER is a veto, not a ranker: pass iff pooled <= 3% AND no phrase > 50%."""
    worst = max(p["wer_pct"] for p in wer["per_phrase"])
    return "pass" if (wer["wer_pct"] <= 3.0 and worst <= 50.0) else "fail"


def _bootstrap_ci(diff, n_boot: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_bootstrap(variant_sim: dict, results_dir: Path, variant: str):
    """Paired bootstrap of (variant - base_mita_tag) over phrase ids.

    Skipped when the variant IS base_mita_tag or the baseline result is
    missing/incomplete. A partial run (n_phrases != 20) is only paired when
    the base variant has the same n_phrases; otherwise it is skipped with a
    recorded reason. Pure numpy.
    """
    if variant == "base_mita_tag":
        return None
    base_path = results_dir / "base_mita_tag.json"
    if not base_path.is_file():
        print("[note] paired bootstrap skipped: results/base_mita_tag.json not found")
        return None
    base_doc = json.loads(base_path.read_text(encoding="utf-8"))
    base_sim = base_doc.get("sim") or {}
    base_by_id = {p["id"]: p for p in base_sim.get("per_phrase", [])}
    ids = [p["id"] for p in variant_sim["per_phrase"]]
    n = len(ids)
    base_n = base_doc.get("n_phrases")
    if base_n is None:
        base_n = len(base_by_id)  # legacy result without the n_phrases field
    if n != 20 and n != base_n:  # 20 = the frozen phrase set size
        return {"skipped_reason": f"partial run (n={n})"}
    missing = [i for i in ids if i not in base_by_id]
    if missing:
        print(f"[note] paired bootstrap skipped: base_mita_tag lacks ids {missing}")
        return None
    if any(k not in base_by_id[ids[0]] for k in ("sim_best", "sim_centroid")):
        print("[note] paired bootstrap skipped: base_mita_tag result predates the sim_best/sim_centroid schema")
        return None
    import numpy as np

    out = {"against": "base_mita_tag", "n": len(ids), "n_bootstrap": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED}
    for metric in ("sim_best", "sim_centroid"):
        v = np.array([next(p for p in variant_sim["per_phrase"] if p["id"] == i)[metric] for i in ids])
        b = np.array([base_by_id[i][metric] for i in ids])
        diff = v - b
        lo, hi = _bootstrap_ci(diff, BOOTSTRAP_N, BOOTSTRAP_SEED)
        out[metric] = {
            "delta_mean": round(float(diff.mean()), 6),
            "ci95_lo": round(lo, 6),
            "ci95_hi": round(hi, 6),
        }
    return out


def update_index(results_dir: Path, variant: str, merged: dict) -> None:
    """Update index.json; never replace a non-null headline number with null."""
    index_path = results_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else []
    prev = next((e for e in index if e.get("variant") == variant), {})

    wer = merged.get("wer") or {}
    sim = merged.get("sim") or {}
    fields = {
        "wer_pct": wer.get("wer_pct"),
        "wer_verdict": merged.get("wer_verdict"),
        "sim_best_mean": sim.get("sim_best_mean"),
        "sim_best_ceiling": (sim.get("sim_best_ceiling") or {}).get("mean"),
        "sim_centroid_mean": sim.get("sim_centroid_mean"),
        "sim_centroid_ceiling": (sim.get("sim_centroid_ceiling") or {}).get("mean"),
    }
    entry = {
        "variant": variant,
        "instruct": merged["instruct"],
        "lora_adapter": merged["lora_adapter"],
        "model": merged.get("model"),
        "seed": merged["seed"],
        "limit": merged.get("limit"),
        "n_phrases": merged.get("n_phrases"),
        "created": merged["created"],
        "flags": merged.get("flags", []),  # union of sanity_flags across scored phrases
    }
    for key, new in fields.items():
        entry[key] = new if new is not None else prev.get(key)  # keep old non-null value
    index = [e for e in index if e.get("variant") != variant]
    index.append(entry)
    index.sort(key=lambda e: e["variant"])
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def maybe_rebuild_baseline(results_dir: Path) -> None:
    """Regenerate eval/baseline.json (machine-generated) when both baselines exist
    AND are complete (full n_phrases, non-null wer+sim headline values).
    Otherwise the existing baseline.json is kept untouched — it must never be
    downgraded to nulls.
    """
    if not all((results_dir / f"{n}.json").is_file() for n in BASELINE_VARIANTS):
        print("[note] eval/baseline.json not regenerated (both baseline variant results required)")
        return
    from eval.make_baseline import BaselineError, build

    try:
        doc = build(results_dir)
    except BaselineError as e:
        print(f"[note] baseline.json not regenerated: {e}")
        return
    print(f"regenerated eval/baseline.json (created={doc['created']})")


def main() -> None:
    args = build_parser().parse_args()
    if args.limit > 0 and args.reuse_audio:
        sys.exit("--limit and --reuse-audio cannot be combined (reused audio is verified as the full frozen set)")
    audio_dir = Path(args.audio_root) / args.variant
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== variant {args.variant} | instruct={args.instruct!r} | adapter={args.lora_adapter} | model={args.model} ===")

    # (1) render (or verify + reuse existing audio)
    if args.reuse_audio:
        gen_meta = stage_reuse(args, audio_dir, results_dir)
        render_s = 0.0
        print(f"[reuse] verified {audio_dir}: {len(gen_meta['per_phrase'])} wavs (sha256, phrase sha, seeds, instruct, adapter, params)")
    else:
        gen_meta, render_s = stage_render(args, audio_dir)

    # (3) WER in the project venv
    if args.skip_wer:
        wer, wer_s = None, 0.0
        print("[skip] WER")
    else:
        wer, wer_s = stage_wer(args, audio_dir, results_dir, args.variant)

    # (4) speaker similarity in-process
    if args.skip_sim:
        sim, sim_s = None, 0.0
        print("[skip] SIM")
    else:
        sim, sim_s = stage_sim(args, audio_dir, args.variant)

    # (5) merge — skipped stages are ABSENT, never null
    flags = [f"{p['id']}:{','.join(p['flags'])}" for p in gen_meta["per_phrase"] if p["flags"]]
    merged = {
        "variant": args.variant,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instruct": args.instruct,
        "lora_adapter": args.lora_adapter,
        # resolved value from the generation meta: MODEL_NAME for base runs,
        # absolute path for --model runs (args.model is the raw CLI string)
        "model": gen_meta.get("model", args.model),
        "seed": args.seed,
        "limit": args.limit if args.limit > 0 else None,
        "n_phrases": len(gen_meta["per_phrase"]),
        "phrases_sha256": gen_meta["phrases_sha256"],
        "generation": gen_meta,
        "reused_audio": bool(args.reuse_audio),
        "flags": flags,
    }
    if wer is not None:
        merged["wer"] = wer
        merged["wer_verdict"] = wer_verdict(wer)
    if sim is not None:
        merged["sim"] = sim
        paired = paired_bootstrap(sim, results_dir, args.variant)
        if paired is not None:
            merged["paired_vs_base_mita_tag"] = paired
    merged["wall_s"] = {
        "render": round(render_s, 2),
        "wer": round(wer_s, 2),
        "sim": round(sim_s, 2),
    }
    out = results_dir / f"{args.variant}.json"
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    update_index(results_dir, args.variant, merged)
    maybe_rebuild_baseline(results_dir)
    print(f"merged -> {out}")
    print(f"wall: render {render_s:.1f}s, wer {wer_s:.1f}s, sim {sim_s:.1f}s")


if __name__ == "__main__":
    main()
