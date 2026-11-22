r"""Merge a LoRA checkpoint into a self-contained, deployable OmniVoice dir.

Usage (vendor venv):
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\merge_lora.py --checkpoint runs\mita_v1\checkpoint-200 --out-dir tmp\merge_test
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\merge_lora.py --checkpoint runs\mita_v1\checkpoint-200 --out-dir tmp\merge_test --yes

Default is DRY-RUN: preflight table + the exact upstream command, nothing
written. --yes runs the upstream merge (omnivoice.cli.merge_lora, CPU-only,
same interpreter) as a subprocess with live output, then structurally
verifies the artifact: required files, config, key set == base key set,
no lora_/base_model. keys, tensor shapes+dtypes, weights actually moved vs
base, modules_to_save bit-identical to the adapter, audio_tokenizer file
set. Before the upstream run, any pre-existing model.safetensors and
merge_meta.json in the out-dir are deleted, so a crash can never leave a
dir whose stale weights verify() would attest as this checkpoint's.
merge_meta.json is written before the marker is dropped, so a failing meta
write still leaves the MERGE_IN_PROGRESS marker in the out-dir: on any
failure the marker remains and a half-written dir is never mistaken for a
model (generate.py / run_eval refuse to load a dir that carries the marker).
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "OmniVoice"
VENV_PY = VENDOR / ".venv" / "Scripts" / "python.exe"

MARKER = "MERGE_IN_PROGRESS"
DISK_HEADROOM = 1.15  # +15% over base weights + audio_tokenizer
DEFAULT_BASE = "k2-fsa/OmniVoice"


def die(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def fmt_bytes(n: int) -> str:
    return f"{n:,} B"


def sha256_file(p: Path) -> tuple[str, int]:
    """(sha256, byte size) of a file, streamed in 16 MB chunks."""
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def checkpoint_step(ckpt: Path) -> str:
    """Step number from a checkpoint-<N> dir name; the dir name itself otherwise."""
    m = re.fullmatch(r"checkpoint-(\d+)", ckpt.name)
    return m.group(1) if m else ckpt.name


def free_bytes(p: Path) -> int:
    """Free disk on the volume that will hold p.

    Windows' GetDiskFreeSpaceExW needs an existing path, so walk up to the
    nearest existing ancestor (the out-dir usually does not exist yet).
    """
    q = p
    while not q.exists() and q.parent != q:
        q = q.parent
    return shutil.disk_usage(str(q)).free


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True,
                    help="LoRA checkpoint dir (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--out-dir", required=True,
                    help="target dir for the merged, deployable model")
    ap.add_argument("--base-model", default=DEFAULT_BASE,
                    help="base model: HF repo id or local dir (default: %(default)s)")
    ap.add_argument("--yes", action="store_true",
                    help="execute the merge (default is a dry-run plan)")
    ap.add_argument("--force", action="store_true",
                    help="allow a non-empty --out-dir")
    return ap


def preflight(args) -> dict:
    """Print the PASS/FAIL table; die on the first FAIL. Returns the run context."""

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<42} {detail}")
        if not ok:
            die(f"{name}: {detail}")

    # the merge runs as a subprocess of THIS interpreter; without omnivoice it
    # would pass every other row and die in the child with ModuleNotFoundError
    have_omni = importlib.util.find_spec("omnivoice") is not None
    check("interpreter has omnivoice", have_omni,
          f"sys.executable={sys.executable}"
          + ("" if have_omni else f" — omnivoice not importable; rerun with {VENV_PY}"))

    ckpt = Path(args.checkpoint)
    ckpt = ckpt.resolve()
    check("checkpoint dir", ckpt.is_dir(), str(ckpt))
    adapter_cfg_path = ckpt / "adapter_config.json"
    adapter_w = ckpt / "adapter_model.safetensors"
    check("checkpoint adapter files",
          adapter_cfg_path.is_file() and adapter_w.is_file(),
          f"adapter_config.json={'yes' if adapter_cfg_path.is_file() else 'MISSING'} "
          f"adapter_model.safetensors={'yes' if adapter_w.is_file() else 'MISSING'}")

    try:
        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        parse_detail = (f"r={adapter_cfg.get('r')} lora_alpha={adapter_cfg.get('lora_alpha')} "
                        f"targets={len(adapter_cfg.get('target_modules') or [])} "
                        f"modules_to_save={adapter_cfg.get('modules_to_save')}")
    except Exception as e:
        adapter_cfg = {}
        parse_detail = f"{type(e).__name__}: {e}"
    check("adapter_config.json parses", bool(adapter_cfg), parse_detail)
    missing = [k for k in ("r", "lora_alpha") if k not in adapter_cfg]
    check("adapter r/lora_alpha present", not missing,
          "both present" if not missing else f"missing: {missing}")

    # resolve base: local dir as-is, else HF cache (offline)
    base_spec = args.base_model
    if os.path.isdir(base_spec):
        base, how = Path(base_spec).resolve(), "local dir"
    else:
        try:
            from huggingface_hub import snapshot_download

            base, how = Path(snapshot_download(base_spec, local_files_only=True)), "HF cache"
        except Exception as e:
            check("base model resolves", False,
                  f"{base_spec}: not in HF cache ({type(e).__name__}); "
                  "run once online or pass a local dir")
    base_w = base / "model.safetensors"
    check("base model resolves", base_w.is_file(),
          f"{how}: {base} (model.safetensors "
          f"{fmt_bytes(base_w.stat().st_size) if base_w.is_file() else 'MISSING'})")

    # the adapter was trained from a specific snapshot; merging into another
    # revision is silently wrong, so require an exact match
    trained_from = adapter_cfg.get("base_model_name_or_path")
    want = os.path.normcase(os.path.normpath(trained_from)) if trained_from else None
    got = os.path.normcase(os.path.normpath(str(base)))
    base_match = want is not None and want == got
    detail = "match" if base_match else f"adapter says {trained_from!r} but base is {str(base)!r}"
    if not base_match and trained_from:
        detail += f" (or pass --base-model {_q(trained_from)} to merge into the snapshot it was trained from)"
    check("base match (trained-from == base)", base_match, detail)

    # out-dir must not collide with the base or the checkpoint
    out = Path(args.out_dir).resolve()
    base_r, ckpt_r = base.resolve(), ckpt.resolve()

    def overlaps(p: Path, q: Path) -> bool:
        return p == q or q in p.parents or p in q.parents

    if overlaps(out, base_r) or overlaps(out, ckpt_r):
        check("out-dir isolated from base/ckpt", False,
              f"{out} equals/contains/is contained by the base or checkpoint dir")
    else:
        check("out-dir isolated from base/ckpt", True, str(out))
    if out.exists() and any(out.iterdir()) and not args.force:
        check("out-dir empty (or --force)", False,
              f"non-empty: {sorted(p.name for p in out.iterdir())[:5]} (use --force)")
    else:
        state = "does not exist" if not out.exists() else (
            "empty" if not any(out.iterdir()) else "non-empty, allowed by --force")
        check("out-dir empty (or --force)", True, state)

    # free disk on the out-dir volume: base weights + audio_tokenizer + 15%
    audio_dir = base / "audio_tokenizer"
    audio_bytes = (sum(p.stat().st_size for p in audio_dir.rglob("*") if p.is_file())
                   if audio_dir.is_dir() else 0)
    base_bytes = base_w.stat().st_size
    need = int((base_bytes + audio_bytes) * DISK_HEADROOM)
    free = free_bytes(out)
    check("free disk >= base + audio_tok + 15%", free >= need,
          f"free {free / 1e9:.1f} GB, need {need / 1e9:.1f} GB"
          + (f" (shortfall {(need - free) / 1e9:.1f} GB)" if free < need else ""))

    adapter_sha, adapter_bytes = sha256_file(adapter_w)
    check("adapter weights sha256 (record)", True, f"{adapter_sha[:16]}… {fmt_bytes(adapter_bytes)}")

    return {
        "checkpoint": ckpt,
        "adapter_cfg": adapter_cfg,
        "adapter_sha256": adapter_sha,
        "adapter_bytes": adapter_bytes,
        "base_spec": base_spec,
        "base_path": base,
        "base_model_bytes": base_bytes,
        "audio_tokenizer_bytes": audio_bytes,
        "out_dir": out,
    }


def upstream_cmd(ctx: dict) -> list[str]:
    return [
        sys.executable, "-m", "omnivoice.cli.merge_lora",
        "--base_model", str(ctx["base_path"]),
        "--lora_adapter", str(ctx["checkpoint"]),
        "--output_dir", str(ctx["out_dir"]),
    ]


def run_upstream(ctx: dict) -> list[str]:
    """Run the upstream merge as a subprocess (same interpreter, live output)."""
    out = ctx["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    marker = out / MARKER
    marker.write_text(
        f"merge started {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )
    # A stale model.safetensors / merge_meta.json from a previous merge must not
    # survive an interrupted rerun: verify() compares against the base only, so a
    # pre-existing complete merge would otherwise be attested as THIS checkpoint's.
    for stale in (out / "model.safetensors", out / "merge_meta.json"):
        if stale.is_file():
            stale.unlink()
    cmd = upstream_cmd(ctx)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    print(f"$ {subprocess.list2cmdline(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)  # no capture: logs stream live
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"FAILED: exit {proc.returncode} after {wall:.0f}s")
        print(f"  $ {subprocess.list2cmdline(cmd)}")
        print(f"partial output left in {out}; delete the directory, then rerun — "
              f"upstream skips copying audio_tokenizer/ and chat_template.jinja when the "
              f"destination already exists, so a partial or stale copy from a previous run "
              f"would survive; --force only lets this script start into a non-empty dir, it does not clean it")
        sys.exit(proc.returncode or 1)
    print(f"upstream merge done in {wall:.0f}s")
    return cmd


def verify(ctx: dict) -> dict:
    """Structural verification of the merged dir (CPU; torch/safetensors only).

    Dies on the first FAIL; the MERGE_IN_PROGRESS marker is left in place by
    the caller on any failure.
    """
    import torch
    from safetensors import safe_open

    out, base = ctx["out_dir"], ctx["base_path"]

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<42} {detail}")
        if not ok:
            die(f"verify failed: {name}: {detail}")

    # 1. required files non-empty
    required = ["config.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json",
                "audio_tokenizer/config.json", "audio_tokenizer/model.safetensors"]
    if (base / "chat_template.jinja").is_file():
        required.append("chat_template.jinja")
    bad = [rel for rel in required
           if not (out / rel).is_file() or (out / rel).stat().st_size == 0]
    check("required files non-empty", not bad,
          f"{len(required)} required" + (f"; MISSING/EMPTY: {bad}" if bad else ""))

    # special_tokens_map.json: record only (tokenizer source mirrors upstream:
    # the adapter dir when it carries tokenizer_config.json, else the base)
    tok_src = ctx["checkpoint"] if (ctx["checkpoint"] / "tokenizer_config.json").is_file() else base
    check("special_tokens_map.json (record)", True,
          f"source {tok_src.name}: {'present' if (tok_src / 'special_tokens_map.json').is_file() else 'absent'}; "
          f"output: {'present' if (out / 'special_tokens_map.json').is_file() else 'absent'}")

    # 2. config.json
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    base_cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    ok = ("OmniVoice" in (cfg.get("architectures") or [])
          and cfg.get("model_type") == base_cfg.get("model_type"))
    check("config.json arch/model_type", ok,
          f"architectures={cfg.get('architectures')} model_type={cfg.get('model_type')!r} "
          f"(base {base_cfg.get('model_type')!r})")

    base_w, out_w = base / "model.safetensors", out / "model.safetensors"

    def keys_of(p: Path) -> set[str]:
        with safe_open(p, framework="pt", device="cpu") as f:
            return set(f.keys())

    base_keys, out_keys = keys_of(base_w), keys_of(out_w)

    # 3. key set exactly equal to the base
    sym = sorted(base_keys ^ out_keys)
    check("key set == base key set", not sym,
          f"{len(out_keys)} keys" + (f"; sym-diff {len(sym)}: {sym[:5]}" if sym else ""))
    stray = sorted(k for k in out_keys if "lora_" in k or k.startswith("base_model."))
    check("no lora_/base_model. keys", not stray,
          "clean" if not stray else f"stray: {stray[:5]}")

    # 4. shapes AND dtypes match on the intersection (headers only). A dtype change
    #    must fail here: casting to fp32 in differs() would let a no-op merge that
    #    merely re-quantised the base report "differs from base".
    mism = []
    with safe_open(base_w, framework="pt", device="cpu") as fb, \
         safe_open(out_w, framework="pt", device="cpu") as fo:
        for k in sorted(base_keys & out_keys):
            sb, so = fb.get_slice(k), fo.get_slice(k)
            if sb.get_shape() != so.get_shape():
                mism.append(f"{k} shape base={sb.get_shape()} out={so.get_shape()}")
            elif sb.get_dtype() != so.get_dtype():
                mism.append(f"{k} dtype base={sb.get_dtype()} out={so.get_dtype()}")
    check("tensor shapes+dtypes match base", not mism,
          f"{len(base_keys & out_keys)} compared" + (f"; {mism[:5]}" if mism else ""))

    # 5. weights actually moved (a merge that reproduces the base is a no-op)
    qk = "llm.layers.0.self_attn.q_proj.weight"
    if qk not in out_keys:
        cands = sorted(k for k in out_keys if "q_proj" in k)
        if not cands:
            die("verify failed: no q_proj key in the merged weights")
        qk = cands[0]

    def differs(fb, fo, k: str) -> bool:
        # dtype equality is enforced by the header check above, so a straight
        # torch.equal is the right no-op signal (no cast: it would erase it)
        return not torch.equal(fb.get_tensor(k), fo.get_tensor(k))

    with safe_open(base_w, framework="pt", device="cpu") as fb, \
         safe_open(out_w, framework="pt", device="cpu") as fo:
        check("q_proj weight differs from base", differs(fb, fo, qk), qk)

    # 5b. modules_to_save are full replacements, not merges: the merged tensors must
    # be bit-identical to the adapter's copies (the adapter stores them at
    # base_model.model.<key>). This binds the artifact to THIS adapter and covers
    # every modules_to_save entry, not just one.
    mts = ctx["adapter_cfg"].get("modules_to_save") or []
    bad = []
    with safe_open(ctx["checkpoint"] / "adapter_model.safetensors",
                   framework="pt", device="cpu") as fa:
        ad_keys = [k for k in fa.keys() if k.startswith("base_model.model.")]
        with safe_open(out_w, framework="pt", device="cpu") as fo:
            for mod in mts:
                mods = [k for k in ad_keys
                        if k == f"base_model.model.{mod}"
                        or k.startswith(f"base_model.model.{mod}.")]
                if not mods:
                    bad.append(f"{mod}: no tensors in adapter")
                    continue
                for ak in mods:
                    mk = ak[len("base_model.model."):]
                    if mk not in out_keys:
                        bad.append(f"{mod}: {mk} missing in merged")
                    elif not torch.equal(fo.get_tensor(mk), fa.get_tensor(ak)):
                        bad.append(f"{mod}: {mk} not bit-identical to adapter")
    check("modules_to_save bit-identical to adapter", not bad,
          f"modules={mts}" + (f"; {bad[:5]}" if bad else ""))

    # 6. audio_tokenizer file set (relative names + sizes) equals the base's
    def fileset(d: Path) -> dict[str, int]:
        return {p.relative_to(d).as_posix(): p.stat().st_size
                for p in d.rglob("*") if p.is_file()}

    f_out, f_base = fileset(out / "audio_tokenizer"), fileset(base / "audio_tokenizer")
    missing = sorted(set(f_base) - set(f_out))
    extra = sorted(set(f_out) - set(f_base))
    size_diff = sorted(k for k in set(f_base) & set(f_out) if f_base[k] != f_out[k])
    check("audio_tokenizer file set == base", not (missing or extra or size_diff),
          f"{len(f_out)} files"
          + (f"; missing={missing[:5]} extra={extra[:5]} size-diff={size_diff[:5]}"
             if (missing or extra or size_diff) else ""))

    # 7. output hash
    sha, size = sha256_file(out_w)
    check("output model.safetensors sha256", True, f"{sha} {fmt_bytes(size)}")
    return {"sha256": sha, "bytes": size}


def write_meta(ctx: dict, cmd: list[str], out_info: dict) -> None:
    """Write merge_meta.json last, inside the out-dir."""
    import importlib.metadata as md

    def ver(dist: str):
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            return None

    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {
            "path": str(ctx["checkpoint"]),
            "adapter_sha256": ctx["adapter_sha256"],
            "adapter_bytes": ctx["adapter_bytes"],
            "r": ctx["adapter_cfg"]["r"],
            "lora_alpha": ctx["adapter_cfg"]["lora_alpha"],
            "target_modules": ctx["adapter_cfg"].get("target_modules"),
            "modules_to_save": ctx["adapter_cfg"].get("modules_to_save"),
            "base_model_name_or_path": ctx["adapter_cfg"].get("base_model_name_or_path"),
        },
        "base": {
            "spec": ctx["base_spec"],
            "resolved_path": str(ctx["base_path"]),
            "model_safetensors_bytes": ctx["base_model_bytes"],
        },
        "output": {
            "path": str(ctx["out_dir"]),
            "model_safetensors_sha256": out_info["sha256"],
            "model_safetensors_bytes": out_info["bytes"],
            "files": [{"name": p.name, "bytes": p.stat().st_size}
                      for p in sorted(ctx["out_dir"].iterdir())
                      if p.is_file() and p.name != MARKER],  # the marker is dropped right after
        },
        "upstream": {
            "command": cmd,
            "omnivoice_version": ver("omnivoice"),
            "peft_version": ver("peft"),
            "torch_version": ver("torch"),
            "python": platform.python_version(),
        },
    }
    path = ctx["out_dir"] / "merge_meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")
    print(f"meta -> {path}")


def _q(p: str) -> str:
    """Quote a path for a Windows command line when it needs it (spaces/quotes)."""
    s = str(p)
    return f'"{s}"' if re.search(r'[\s"]', s) else s


def print_next_steps(ctx: dict) -> None:
    out = _q(ctx["out_dir"])
    py = _q(VENV_PY)
    step = checkpoint_step(ctx["checkpoint"])
    print()
    print("next steps (run from the repo root):")
    print(f"{py} scripts\\mita_voice\\generate.py --phrases eval\\phrases.jsonl "
          f"--out-dir tmp\\merged_check --instruct mita --model {out} --limit 1")
    print(f"{py} eval\\run_eval.py --variant merged-{step} --model {out} --instruct mita")


def main() -> None:
    args = build_parser().parse_args()
    print(f"=== merge_lora ({'MERGE' if args.yes else 'DRY-RUN'}) ===")
    print("preflight:")
    ctx = preflight(args)

    cmd = upstream_cmd(ctx)
    print("\nupstream command (CPU-only, same interpreter):")
    print(f"  $ {subprocess.list2cmdline(cmd)}")

    if not args.yes:
        print("\nDRY RUN: nothing written. Re-run with --yes to merge.")
        print_next_steps(ctx)
        return

    cmd = run_upstream(ctx)
    print("\nverify:")
    out_info = verify(ctx)
    write_meta(ctx, cmd, out_info)  # provenance first: a failing meta write must keep the marker
    (ctx["out_dir"] / MARKER).unlink()  # dropped last: "no marker" now means "fully finished"
    print(f"\nmerged model ready: {ctx['out_dir']}")
    print_next_steps(ctx)


if __name__ == "__main__":
    main()
