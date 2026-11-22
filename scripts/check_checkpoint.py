r"""Answer "is this checkpoint usable?" before spending GPU time on it.

Usage (vendor venv):
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\check_checkpoint.py runs\mita_v1\checkpoint-600
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\check_checkpoint.py runs\mita_v1\checkpoint-600 --render
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\check_checkpoint.py runs\mita_v1 --all
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\check_checkpoint.py runs\mita_v1\checkpoint-600 --export C:\ship

Static checks (fail -> non-zero exit): dir exists; adapter_config.json +
adapter_model.safetensors present; tokenizer.json + tokenizer_config.json
present; adapter_config r/lora_alpha/target_modules/modules_to_save match
configs/train_config_mita.json (compared as sets); weights size + sha256;
step parsed from the dir name. optimizer.bin/scheduler.bin/random_states_*.pkl
are reported as "resume-capable: yes/no" (informational, not a failure).

--render: load base + adapter via scripts.mita_voice.runtime/generate, render
p05 from eval/phrases.jsonl with the frozen-eval per-phrase seed 1234004
(seed_base 1234, phrase index 4 — same seed as eval/audio/base_mita_tag/p05.wav),
write tmp/check_<step>.wav, assert non-silent (RMS >= 1e-4) and duration
0.3-20 s. Catches a corrupt adapter that load_lora_adapter would happily merge.

--export DIR: copy the 4 shipping files into DIR/<checkpoint-name>/ (~110 MB,
not the 2.7 GB accelerate state).

Never writes into eval/ or data/.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN_CONFIG = ROOT / "configs" / "train_config_mita.json"
PHRASES = ROOT / "eval" / "phrases.jsonl"
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json")
RESUME_FILES = ("optimizer.bin", "scheduler.bin")
# generate.run() computes per-phrase seed = seed_base * 1000 + index; p05 is
# index 4, so seed_base 1234 gives the frozen-eval per-phrase seed 1234004
# (comparable to eval/audio/base_mita_tag/p05.wav).
RENDER_SEED = 1234
RENDER_PHRASE = "p05"
RENDER_PHRASE_SEED = RENDER_SEED * 1000 + 4
RMS_MIN = 1e-4
DUR_MIN, DUR_MAX = 0.3, 20.0


def die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def base_model_ok(value) -> bool:
    """Accept the raw repo id or a resolved path (local dir or HF snapshot dir)."""
    if value == "k2-fsa/OmniVoice":
        return True
    if not isinstance(value, str) or not value:
        return False
    parts = value.replace("\\", "/").rstrip("/").split("/")
    return parts[-1] == "OmniVoice" or "models--k2-fsa--OmniVoice" in parts


def static_check(ckpt: Path) -> dict:
    """Run all static checks. Returns a result dict; raises SystemExit on failure."""
    res = {"ckpt": str(ckpt), "ok": True, "errors": [], "notes": []}

    if not ckpt.is_dir():
        die(f"checkpoint dir does not exist: {ckpt}")
    for name in REQUIRED_FILES:
        if not (ckpt / name).is_file():
            res["ok"] = False
            res["errors"].append(f"missing {name}")
    if res["errors"]:
        return res

    adapter_cfg = json.loads((ckpt / "adapter_config.json").read_text(encoding="utf-8"))
    train_cfg = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
    pairs = [
        ("r", adapter_cfg.get("r"), train_cfg.get("lora_r")),
        ("lora_alpha", adapter_cfg.get("lora_alpha"), train_cfg.get("lora_alpha")),
        ("target_modules", set(adapter_cfg.get("target_modules") or []), set(train_cfg.get("lora_target_modules") or [])),
        ("modules_to_save", set(adapter_cfg.get("modules_to_save") or []), set(train_cfg.get("lora_modules_to_save") or [])),
    ]
    res["lora"] = {}
    for name, got, want in pairs:
        match = got == want
        res["lora"][name] = {"got": got, "want": want, "match": match}
        if not match:
            res["ok"] = False
            res["errors"].append(f"adapter_config {name}={got!r} != train_config {want!r}")

    bmn = adapter_cfg.get("base_model_name_or_path")
    res["base_model"] = bmn
    if not base_model_ok(bmn):
        res["ok"] = False
        res["errors"].append(f"base_model_name_or_path={bmn!r} is not k2-fsa/OmniVoice")

    weights = ckpt / "adapter_model.safetensors"
    res["weights_bytes"] = weights.stat().st_size
    res["weights_sha256"] = sha256_of(weights)

    m = re.search(r"checkpoint-(\d+)", ckpt.name)
    res["step"] = int(m.group(1)) if m else None
    if res["step"] is None:
        res["notes"].append("step not parseable from dir name")

    resume = [ckpt / p for p in RESUME_FILES if (ckpt / p).is_file()] + sorted(ckpt.glob("random_states_*.pkl"))
    res["resume_capable"] = bool(resume)
    res["resume_files"] = [p.name for p in resume]
    return res


def print_static(res: dict) -> None:
    ckpt = Path(res["ckpt"])
    print(f"=== check_checkpoint: {ckpt} ===")
    print(f"  files: " + ", ".join(f"{n}={'yes' if (ckpt / n).is_file() else 'NO'}" for n in REQUIRED_FILES))
    for name, d in res.get("lora", {}).items():
        print(f"  lora {name}: {'MATCH' if d['match'] else 'MISMATCH'} (adapter={d['got']!r} config={d['want']!r})")
    bmn = res.get("base_model")
    if bmn is not None:
        print(f"  base_model_name_or_path: {'OK' if base_model_ok(bmn) else 'WRONG'} ({bmn!r})")
    if "weights_bytes" in res:
        print(f"  weights: {res['weights_bytes'] / 1e6:.1f} MB  sha256={res['weights_sha256']}")
    print(f"  step: {res['step']}")
    print(f"  resume-capable: {'yes' if res['resume_capable'] else 'no'} ({', '.join(res['resume_files']) or 'no optimizer/scheduler/random_states'})")
    for note in res["notes"]:
        print(f"  note: {note}")
    for err in res["errors"]:
        print(f"  ERROR: {err}", file=sys.stderr)


def render(ckpt: Path, step: int) -> None:
    """Render p05 with base+adapter; assert non-silent, sane duration."""
    import soundfile as sf

    from scripts.mita_voice.generate import run

    render_dir = ROOT / "tmp" / f"check_{step}_render"
    if render_dir.exists():
        shutil.rmtree(render_dir)
    print(f"  rendering {RENDER_PHRASE} (seed_base {RENDER_SEED}, per-phrase seed {RENDER_PHRASE_SEED}) ...")
    meta = run(
        phrases=str(PHRASES),
        out_dir=str(render_dir),
        instruct="mita",
        lora_adapter=str(ckpt),
        seed=RENDER_SEED,
        limit=5,  # p05 is the 5th phrase (index 4); per-phrase seed = seed_base*1000+4 = 1234004
    )
    entry = next((p for p in meta["per_phrase"] if p["id"] == RENDER_PHRASE), None)
    if entry is None:
        die(f"{RENDER_PHRASE} not rendered")
    wav_src = render_dir / f"{RENDER_PHRASE}.wav"
    wav_dst = ROOT / "tmp" / f"check_{step}.wav"
    shutil.copyfile(wav_src, wav_dst)

    audio, sr = sf.read(str(wav_dst), dtype="float64")
    rms = float((audio ** 2).mean() ** 0.5)
    peak = float(abs(audio).max())
    dur = audio.size / sr
    print(f"  {wav_dst.name}: duration={dur:.2f}s rms={rms:.5f} peak={peak:.3f} (sr={sr})")
    problems = []
    if rms < RMS_MIN:
        problems.append(f"silent (rms {rms:.2e} < {RMS_MIN})")
    if not (DUR_MIN <= dur <= DUR_MAX):
        problems.append(f"duration {dur:.2f}s outside [{DUR_MIN}, {DUR_MAX}]")
    if problems:
        die(f"render check failed for {ckpt}: {'; '.join(problems)}")
    print(f"  render PASS: non-silent, duration in range -> {wav_dst}")


def export(ckpt: Path, dest: Path) -> None:
    target = dest / ckpt.name
    if target.exists():
        die(f"export target already exists: {target}")
    target.mkdir(parents=True)
    total = 0
    for name in REQUIRED_FILES:
        shutil.copyfile(ckpt / name, target / name)
        total += (target / name).stat().st_size
    print(f"  exported {len(REQUIRED_FILES)} files -> {target} ({total / 1e6:.1f} MB)")


def discover_checkpoints(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        die(f"run dir does not exist: {run_dir}")
    ckpts = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    ckpts.sort(key=lambda p: int(re.search(r"checkpoint-(\d+)", p.name).group(1))
               if re.search(r"checkpoint-(\d+)", p.name) else -1)
    return ckpts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="checkpoint dir, or run dir with --all")
    ap.add_argument("--all", action="store_true", help="iterate checkpoint-* under the run dir (numeric order)")
    ap.add_argument("--render", action="store_true", help=f"render {RENDER_PHRASE} with the adapter (GPU)")
    ap.add_argument("--export", metavar="DIR", help="copy the 4 shipping files into DIR/<checkpoint-name>/")
    args = ap.parse_args()

    if args.all:
        if not args.path:
            die("--all requires the run dir as positional argument")
        ckpts = discover_checkpoints(Path(args.path))
        if not ckpts:
            die(f"no checkpoint-* dirs under {args.path}")
        print(f"=== check_checkpoint --all: {len(ckpts)} checkpoints under {args.path} ===")
        print(f"{'checkpoint':<24} {'step':>6} {'weights MB':>10} {'lora match':>10} {'resume':>8}  ok")
        all_ok = True
        for ckpt in ckpts:
            res = static_check(ckpt)
            lora_ok = all(d["match"] for d in res.get("lora", {}).values())
            print(f"{ckpt.name:<24} {str(res['step']):>6} "
                  f"{res.get('weights_bytes', 0) / 1e6:>10.1f} "
                  f"{'yes' if lora_ok else 'NO':>10} "
                  f"{'yes' if res['resume_capable'] else 'no':>8}  {'PASS' if res['ok'] else 'FAIL'}")
            for err in res["errors"]:
                print(f"    ERROR: {err}", file=sys.stderr)
            all_ok &= res["ok"]
        sys.exit(0 if all_ok else 1)

    if not args.path:
        die("checkpoint path required (or --all with a run dir)")
    ckpt = Path(args.path).resolve()
    res = static_check(ckpt)
    print_static(res)
    if not res["ok"]:
        die(f"static checks failed for {ckpt}")
    if args.render:
        render(ckpt, res["step"])
    if args.export:
        export(ckpt, Path(args.export).resolve())
    print(f"  {ckpt.name}: STATIC CHECKS PASSED")


if __name__ == "__main__":
    main()
