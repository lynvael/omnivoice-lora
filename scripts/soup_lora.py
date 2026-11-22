r"""Average the LoRA adapters of several checkpoints from one run (LoRA soup).

Usage (vendor venv, from the project root):
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\soup_lora.py ^
      --name soup-2468 --steps 200 400 600 800

    # then score it on the frozen phrase set
    vendor\OmniVoice\.venv\Scripts\python.exe eval\run_eval.py ^
      --variant soup-2468 --lora-adapter runs\mita_v1\soups\soup-2468 --instruct mita

Why uniform averaging is legitimate here: every checkpoint comes from one run,
so ``lora_A`` was initialised once and drifts only 1-10% between adjacent
checkpoints (~20% across the whole run), while ``lora_B`` accounts for ~85% of
the change in ``dW = B @ A`` between adjacent checkpoints. Averaging (A, B)
therefore approximates averaging the deltas ``dW`` instead of averaging
unrelated bases; it is safe for neighbours and degrades gracefully with
distance. ``modules_to_save`` tensors (audio_embeddings / audio_heads) are
full replacements from the same basin (1-2% drift) and are averaged too.

The result is a normal adapter dir (adapter_config.json +
adapter_model.safetensors) consumable by generate.py / eval/run_eval.py via
``--lora-adapter``. CPU only, ~10 s, ~110 MB per variant.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runs" / "mita_v1"

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Windows reserved device names (con, prn, aux, nul, com1-9, lpt1-9):
# mkdir/open on them fail with OSError even though they look like plain names.
RESERVED_NAME_RE = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _q(p) -> str:
    """Quote a path for a Windows command line when it needs it (spaces/quotes)."""
    s = str(p)
    return f'"{s}"' if re.search(r'[\s"]', s) else s


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="variant name, e.g. soup-2468 (becomes the dir name)")
    ap.add_argument("--steps", required=True, type=int, nargs="+", help="checkpoint steps to average, e.g. 200 400 600 800")
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="training run dir containing checkpoint-<step>")
    ap.add_argument("--out", default=None, help="parent dir for soup adapters (default: <run-dir>\\soups)")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty output dir")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    import torch
    from safetensors.torch import load_file, save_file

    if not NAME_RE.match(args.name):
        sys.exit(f"invalid --name {args.name!r}: must match {NAME_RE.pattern} (plain dir name, no spaces or separators)")
    if args.name.endswith("."):
        sys.exit(f"invalid --name {args.name!r}: trailing '.' is stripped by Windows, so the dir name and soup_meta.json would disagree")
    if RESERVED_NAME_RE.match(args.name):
        sys.exit(f"invalid --name {args.name!r}: Windows reserved device name")
    if args.name.startswith("checkpoint-"):
        sys.exit(f"invalid --name {args.name!r}: must not start with 'checkpoint-' (collides with training checkpoint dirs)")
    if len(set(args.steps)) != len(args.steps):
        sys.exit(f"duplicate steps in {args.steps}")
    if len(args.steps) < 2:
        sys.exit("--steps needs at least 2 checkpoints (a 1-step soup is just a copy)")

    run_dir = Path(args.run_dir)
    out_parent = Path(args.out) if args.out else run_dir / "soups"
    dest = out_parent / args.name

    if out_parent.exists() and not out_parent.is_dir():
        sys.exit(f"{out_parent} exists and is not a directory")
    if dest.exists() and not dest.is_dir():
        sys.exit(f"{dest} exists and is not a directory")

    ckpts = [run_dir / f"checkpoint-{s}" for s in args.steps]
    dest_r = dest.resolve()
    for c in ckpts:
        c_r = c.resolve()
        if dest_r == c_r:
            sys.exit(f"refusing to write the soup onto ingredient checkpoint {c_r}")
        if c_r in dest_r.parents:
            sys.exit(f"refusing to write the soup inside ingredient checkpoint {c_r}")
    if dest.exists() and any(dest.iterdir()) and not args.force:
        sys.exit(f"refusing to overwrite non-empty {dest} (pass --force)")

    # Pre-flight the output parent: a nested ancestor that is a file would
    # otherwise surface as a raw FileExistsError from dest.mkdir() only after
    # every ingredient has been hashed and loaded. dest itself is NOT created
    # here — the "no write before validation" property must survive.
    try:
        out_parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"cannot create {out_parent}: {e}")

    print(f"=== soup {args.name}: steps {args.steps} ===")
    print(f"run dir: {run_dir.resolve()}")
    print(f"output:  {dest_r}")

    configs = []
    for c in ckpts:
        w = c / "adapter_model.safetensors"
        cfg = c / "adapter_config.json"
        if not w.is_file() or not cfg.is_file():
            sys.exit(f"missing adapter files in {c}: expected adapter_config.json + adapter_model.safetensors")
        configs.append(cfg.read_bytes())
    if any(c != configs[0] for c in configs[1:]):
        sys.exit(
            "adapter_config.json differs between checkpoints: refusing to average adapters with "
            "different r / lora_alpha / target_modules / modules_to_save"
        )
    cfg_doc = json.loads(configs[0].decode("utf-8"))
    for field in ("r", "lora_alpha"):
        if cfg_doc.get(field) is None:
            sys.exit(f"adapter_config.json lacks {field!r}: cannot verify the soup is well-formed")

    # Configs that break the A/B-averaging rationale: byte-equality of the
    # configs only proves the fields match across checkpoints, not that
    # averaging raw A/B tensors is meaningful. Warn loudly, record, continue.
    assumption_warnings = []
    if cfg_doc.get("use_dora"):
        assumption_warnings.append(
            "use_dora=true in adapter_config.json: DoRA splits each adapter into a learned "
            "magnitude and a direction, so the A/B averaging assumption does not hold"
        )
    if cfg_doc.get("use_rslora"):
        assumption_warnings.append(
            "use_rslora=true in adapter_config.json: rank-stable scaling rescales B per layer, "
            "so the A/B averaging assumption does not hold"
        )
    if cfg_doc.get("rank_pattern"):
        assumption_warnings.append(
            f"rank_pattern={cfg_doc['rank_pattern']} in adapter_config.json: per-layer rank "
            "overrides, so the A/B averaging assumption does not hold"
        )
    if cfg_doc.get("alpha_pattern"):
        assumption_warnings.append(
            f"alpha_pattern={cfg_doc['alpha_pattern']} in adapter_config.json: per-layer alpha "
            "overrides, so the A/B averaging assumption does not hold"
        )
    for msg in assumption_warnings:
        print(f"[warn] {msg}")

    print(
        f"config: r={cfg_doc['r']} lora_alpha={cfg_doc['lora_alpha']} "
        f"targets={len(cfg_doc.get('target_modules') or [])} "
        f"modules_to_save={cfg_doc.get('modules_to_save')}"
    )

    # Hash the ingredient bytes once, before loading: the meta must record the
    # bytes that were actually loaded, not a second read of the file.
    ing_sha = {str(c.resolve()): sha256_file(c / "adapter_model.safetensors") for c in ckpts}
    weights = []
    for c in ckpts:
        print(f"  load {c.name}: adapter sha256 {ing_sha[str(c.resolve())][:16]}…")
        weights.append(load_file(str(c / "adapter_model.safetensors")))

    ref_keys = list(weights[0].keys())
    for c, w in zip(ckpts, weights):
        if set(w.keys()) != set(ref_keys):
            missing = sorted(set(ref_keys) - set(w.keys()))[:5]
            extra = sorted(set(w.keys()) - set(ref_keys))[:5]
            sys.exit(f"key set mismatch in {c}: missing={missing} extra={extra}")
        for k in ref_keys:
            if w[k].shape != weights[0][k].shape or w[k].dtype != weights[0][k].dtype:
                sys.exit(
                    f"shape/dtype mismatch in {c}: {k} "
                    f"{tuple(w[k].shape)}/{w[k].dtype} vs {tuple(weights[0][k].shape)}/{weights[0][k].dtype}"
                )
    print(f"tensors: {len(ref_keys)} ({sum('lora_' in k for k in ref_keys)} lora + "
          f"{sum('lora_' not in k for k in ref_keys)} full) across {len(weights)} checkpoints")

    n = len(weights)
    soup = {}
    for k in ref_keys:
        acc = torch.zeros_like(weights[0][k], dtype=torch.float64)
        for w in weights:
            acc += w[k].to(torch.float64)
        soup[k] = (acc / n).to(weights[0][k].dtype).contiguous()
        if not torch.isfinite(soup[k]).all():
            sys.exit(f"non-finite values in averaged tensor {k}")

    norm = sum(soup[k].to(torch.float64).pow(2).sum() for k in ref_keys).sqrt()
    drift = {
        str(s): round(
            float(
                sum((soup[k].to(torch.float64) - w[k].to(torch.float64)).pow(2).sum() for k in ref_keys).sqrt()
                / norm
            ),
            5,
        )
        for s, w in zip(args.steps, weights)
    }

    dest.mkdir(parents=True, exist_ok=True)
    out_w = dest / "adapter_model.safetensors"
    # Atomic write: dump to a per-PID temp file in the same dir (same volume),
    # then replace(). A failed run can never destroy the previous good soup.
    tmp_w = dest / f".adapter_model.safetensors.{os.getpid()}.tmp"
    try:
        save_file(soup, str(tmp_w))
        tmp_w.replace(out_w)
    finally:
        if tmp_w.exists():
            tmp_w.unlink()

    (dest / "adapter_config.json").write_bytes(configs[0])
    meta = {
        "name": args.name,
        "steps": args.steps,
        "run_dir": str(run_dir.resolve()),
        "checkpoints": [str(c.resolve()) for c in ckpts],
        "ingredient_sha256": ing_sha,
        "soup_sha256": sha256_file(out_w),
        "soup_bytes": out_w.stat().st_size,
        "tensors": len(ref_keys),
        "lora_tensors": sum("lora_" in k for k in ref_keys),
        "r": cfg_doc["r"],
        "lora_alpha": cfg_doc["lora_alpha"],
        "target_modules": cfg_doc.get("target_modules"),
        "modules_to_save": cfg_doc.get("modules_to_save"),
        "relative_drift_to_soup": drift,
        "dtypes": sorted({str(v.dtype) for v in weights[0].values()}),
        "method": "uniform-mean-fp64",
        "n": n,
        "assumption_warnings": assumption_warnings,
    }
    (dest / "soup_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print()
    print(f"wrote {out_w.resolve()}")
    print(f"  {meta['soup_bytes'] / 1e6:.1f} MB  sha256 {meta['soup_sha256']}")
    print(f"  relative drift to soup: {json.dumps(drift)}")
    print()
    print("next:")
    py = _q(ROOT / "vendor" / "OmniVoice" / ".venv" / "Scripts" / "python.exe")
    out_q = _q(dest_r)
    print(
        f"  {py} "
        f"eval\\run_eval.py --variant {args.name} --lora-adapter {out_q} --instruct mita"
    )
    print(
        f"  {py} "
        f"scripts\\mita_voice\\generate.py --phrases eval\\phrases.jsonl "
        f"--out-dir tmp\\{args.name}_check --instruct mita --lora-adapter {out_q} --limit 1"
    )


if __name__ == "__main__":
    main()
