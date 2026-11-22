r"""Launch OmniVoice LoRA training detached, with preflight that catches
config/data/GPU errors before a 20-minute run.

Usage (vendor venv):
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\launch_train.py --smoke --dry-run
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\launch_train.py --smoke --yes
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\launch_train.py --full --yes
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\launch_train.py --full --resume runs\mita_v1\checkpoint-600 --yes

Default is DRY-RUN; --yes actually launches. Smoke -> runs/mita_smoke with
configs/train_config_smoke.json; full -> runs/mita_v1 with
configs/train_config_mita.json (separate dirs so smoke can never pollute the
real run). --resume writes tmp/train_resume_config.json (only the
resume_from_checkpoint line changes; init_from_checkpoint and all lora_*
fields stay byte-identical) and passes that as --train_config.

Launch: tmp/train_<mode>.cmd is started via
powershell -NoProfile -Command "Start-Process cmd.exe /c <cmd>" and never
blocked on; the PID, log path, and monitoring commands are printed.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenize_shards import GOPEN_REWRITE  # noqa: E402  single source of truth

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "OmniVoice"
VENV_PY = VENDOR / ".venv" / "Scripts" / "python.exe"
ACCELERATE = VENDOR / ".venv" / "Scripts" / "accelerate.exe"
DATA_CONFIG = ROOT / "configs" / "data_config.json"
LOGS = ROOT / "logs"
TMP = ROOT / "tmp"

MODES = {
    "smoke": {"config": ROOT / "configs" / "train_config_smoke.json", "out": ROOT / "runs" / "mita_smoke"},
    "full": {"config": ROOT / "configs" / "train_config_mita.json", "out": ROOT / "runs" / "mita_v1"},
}
MIN_FREE_GB = 40.0
MIN_VRAM_GB = 6.0


def die(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def print_table(rows: list[tuple[str, bool, str]]) -> bool:
    all_ok = True
    for name, ok, detail in rows:
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<42} {detail}")
    return all_ok


# Values the frozen finetune contract requires; a silent fallback to a
# TrainingConfig default (e.g. steps -> 300000) must fail preflight.
EXPECTED = {
    "smoke": {"steps": 60, "eval_steps": 30, "save_steps": 30},
    "full": {"steps": 1200, "eval_steps": 100, "save_steps": 100},
}


def check_config(cfg_path: Path, mode: str) -> list[tuple[str, bool, str]]:
    from omnivoice.training.config import TrainingConfig

    rows = []
    if not cfg_path.is_file():
        return [("config file exists", False, str(cfg_path))]
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    fields = set(TrainingConfig.__annotations__)
    dropped = sorted(set(raw.keys()) - fields)
    missing = sorted(fields - set(raw.keys()) - {"data_config", "output_dir", "deepspeed_config"})
    rows.append(("config: no unknown key dropped by TrainingConfig", not dropped,
                 f"{len(raw)} keys" + (f"; DROPPED: {dropped}" if dropped else "")))
    rows.append(("config: no dataclass key missing (no silent default)", not missing,
                 "all keys present" if not missing else f"MISSING: {missing}"))
    if dropped or missing:
        return rows
    cfg = TrainingConfig.from_json(str(cfg_path))
    for key, want in EXPECTED[mode].items():
        got = getattr(cfg, key)
        rows.append((f"{key} == {want} ({mode})", got == want, str(got)))
    rows += [
        ("steps > 0", cfg.steps > 0, str(cfg.steps)),
        ("eval_steps > 0", cfg.eval_steps > 0, str(cfg.eval_steps)),
        ("save_steps > 0", cfg.save_steps > 0, str(cfg.save_steps)),
        ("use_lora is True", cfg.use_lora is True, str(cfg.use_lora)),
        ("instruct_ratio == 1.0", cfg.instruct_ratio == 1.0, str(cfg.instruct_ratio)),
        ("keep_last_n_checkpoints == -1", cfg.keep_last_n_checkpoints == -1, str(cfg.keep_last_n_checkpoints)),
        ("attn_implementation == 'sdpa'", cfg.attn_implementation == "sdpa", str(cfg.attn_implementation)),
        ("prompt_ratio_range == [0.0, 0.0]", list(cfg.prompt_ratio_range) == [0.0, 0.0],
         str(cfg.prompt_ratio_range)),
    ]
    return rows


def check_shard_readable() -> list[tuple[str, bool, str]]:
    """Open one real sample through the exact reader path the trainer uses.

    A missing GOPEN_REWRITE (or a corrupt first shard) is a first-shard death
    sentence, so preflight must prove the read actually works.
    """
    data_lst = ROOT / "data" / "finetune" / "shards" / "train" / "data.lst"
    if not data_lst.is_file():
        return [("shards/train: 1 sample reads via WebDatasetReader", False, f"missing {data_lst}")]
    try:
        os.environ["GOPEN_REWRITE"] = GOPEN_REWRITE
        from omnivoice.data.dataset import WebDatasetReader, prepare_data_manifests_from_json

        train_manifests, _ = prepare_data_manifests_from_json(str(DATA_CONFIG))
        reader = WebDatasetReader(manifests=train_manifests, evaluation=True)
        sample = next(iter(reader))
        toks = sample["audio_tokens"]
        label = sample["label"]
        ok = (toks.ndim == 2 and toks.shape[0] == 8
              and label.get("instruct") == "mita"
              and label.get("language_id") == "ru")
        return [("shards/train: 1 sample reads via WebDatasetReader", ok,
                 f"id={label.get('id')} tokens={tuple(toks.shape)} "
                 f"instruct={label.get('instruct')!r} language_id={label.get('language_id')!r}")]
    except Exception as e:
        return [("shards/train: 1 sample reads via WebDatasetReader", False,
                 f"{type(e).__name__}: {str(e)[:160]}")]


def check_data() -> list[tuple[str, bool, str]]:
    rows = []
    if not DATA_CONFIG.is_file():
        return [("data_config.json exists", False, str(DATA_CONFIG))]
    data = json.loads(DATA_CONFIG.read_text(encoding="utf-8"))
    lsts = []
    for split in ("train", "dev"):
        for item in data.get(split, []):
            for mp in item["manifest_path"]:
                lsts.append((split, mp))
    ok = True
    detail = []
    for split, mp in lsts:
        p = Path(mp)
        if not p.is_file():
            ok = False
            detail.append(f"{split}: MISSING {mp}")
            continue
        n = 0
        for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            if len(line.split()) != 4:
                ok = False
                detail.append(f"{split}: {mp}:{ln} has {len(line.split())} fields")
            else:
                n += 1
        detail.append(f"{split}: {p.name} {n} lines x 4 fields")
    rows.append(("data_config.json paths exist, data.lst 4 fields/line", ok, "; ".join(detail)))
    for split in ("train", "dev"):
        p = ROOT / "data" / "finetune" / "shards" / split / "data.lst"
        rows.append((f"shards/{split}/data.lst exists", p.is_file(), str(p)))
    return rows


def check_output_dir(out: Path, resume: bool, force: bool) -> tuple[bool, str]:
    if not out.exists():
        return True, "does not exist (will be created empty)"
    if any(out.iterdir()):
        if resume:
            return True, "non-empty, allowed by --resume"
        if force:
            return True, "non-empty, allowed by --force (contents kept, trainer appends)"
        return False, f"non-empty: {sorted(p.name for p in out.iterdir())[:5]} (use --resume or --force)"
    return True, "exists and is empty"


def check_system() -> list[tuple[str, bool, str]]:
    rows = []
    free_gb = shutil.disk_usage(str(ROOT)).free / 1e9
    rows.append((f"free disk >= {MIN_FREE_GB:.0f} GB", free_gb >= MIN_FREE_GB, f"{free_gb:.1f} GB free"))
    try:
        import torch
        if torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info()
            rows.append(("torch.cuda.is_available()", True, torch.cuda.get_device_name(0)))
            rows.append((f"free VRAM >= {MIN_VRAM_GB:.0f} GB", free_vram / 1e9 >= MIN_VRAM_GB,
                         f"{free_vram / 1e9:.2f} / {total_vram / 1e9:.2f} GB"))
        else:
            rows.append(("torch.cuda.is_available()", False, "CUDA not available"))
    except Exception as e:  # torch import failure
        rows.append(("torch importable", False, f"{type(e).__name__}: {e}"))
    rows.append(("accelerate executable resolves", ACCELERATE.is_file(), str(ACCELERATE)))
    return rows


def make_resume_config(cfg_path: Path, ckpt: Path) -> Path:
    """Set resume_from_checkpoint by rewriting that single JSON line only,
    so init_from_checkpoint and every lora_* field stay byte-identical."""
    text = cfg_path.read_text(encoding="utf-8")
    m = re.search(r'"resume_from_checkpoint":\s*[^\n,]+', text)
    if not m:
        die(f"resume_from_checkpoint line not found in {cfg_path}")
    new_text = text[: m.start()] + f'"resume_from_checkpoint": {json.dumps(str(ckpt))}' + text[m.end():]
    diffs = [i for i, (a, b) in enumerate(zip(text.splitlines(), new_text.splitlines())) if a != b]
    if len(diffs) != 1:
        die(f"resume-config rewrite changed {len(diffs)} lines, expected exactly 1")
    a, b = json.loads(text), json.loads(new_text)
    for k in a:
        if k != "resume_from_checkpoint" and a[k] != b[k]:
            die(f"resume-config rewrite changed key {k!r}")
    out = TMP / "train_resume_config.json"
    out.write_text(new_text, encoding="utf-8")
    return out


def build_cmd(mode: str, train_cfg: Path, out: Path) -> tuple[Path, Path]:
    log_path = LOGS / f"train_{mode}.log"
    lines = [
        "@echo off",
        # Project root, not vendor/OmniVoice: -m scripts.train_entry must be
        # importable, and PYTHONPATH carries it into the spawn workers too
        # (they re-import the main module to unpickle the dataset).
        f"cd /d {ROOT}",
        "set PYTHONUTF8=1",
        "set TOKENIZERS_PARALLELISM=false",
        f"set PYTHONPATH={ROOT}",
        # webdataset gopen treats "C:\..." as scheme "c" and refuses it; rewrite
        # to file:/// so wds.WebDataset can read the shard tars (see tokenize_shards.py)
        r"set GOPEN_REWRITE=([A-Za-z]):\\=file:///\1:/",
        # pin gopen behaviour: with WDS_SECURE=1, rewrite_url raises instead of rewriting
        "set WDS_SECURE=0",
        (f"{ACCELERATE} launch --num_processes 1 -m scripts.train_entry "
         f"--train_config {train_cfg} --data_config {DATA_CONFIG} --output_dir {out} "
         f">> {log_path} 2>&1"),
        f"echo EXIT=%ERRORLEVEL% >> {log_path}",
    ]
    cmd_path = TMP / f"train_{mode}.cmd"
    # "\n" + newline="\r\n" -> CRLF (writing "\r\n".join with newline="\r\n" would double the CR)
    cmd_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\r\n")
    return cmd_path, log_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="runs/mita_smoke + train_config_smoke.json (60 steps)")
    mode.add_argument("--full", action="store_true", help="runs/mita_v1 + train_config_mita.json (1200 steps)")
    ap.add_argument("--resume", metavar="CKPT", help="resume from this checkpoint dir")
    ap.add_argument("--force", action="store_true", help="allow a non-empty output dir")
    ap.add_argument("--config", metavar="PATH",
                    help="override the mode's train config (for testing mutated configs)")
    ap.add_argument("--dry-run", action="store_true", help="preflight + print the exact command (default)")
    ap.add_argument("--yes", action="store_true", help="actually launch (detached)")
    args = ap.parse_args()
    if args.dry_run and args.yes:
        die("--dry-run and --yes are mutually exclusive")
    launch = args.yes
    mode_name = "smoke" if args.smoke else "full"
    spec = MODES[mode_name]
    cfg_path = Path(args.config).resolve() if args.config else spec["config"]

    print(f"=== launch_train {mode_name} ({'LAUNCH' if launch else 'DRY-RUN'}) ===")
    rows = []
    rows += check_config(cfg_path, mode_name)
    rows += check_data()
    rows += check_shard_readable()
    out_ok, out_detail = check_output_dir(spec["out"], args.resume is not None, args.force)
    rows.append((f"output dir {spec['out'].relative_to(ROOT)}", out_ok, out_detail))
    rows += check_system()
    all_ok = print_table(rows)

    if not all_ok:
        die("preflight failed — nothing launched, no .cmd written. Fix the FAIL lines above and re-run.")

    train_cfg = cfg_path
    if args.resume:
        ckpt = Path(args.resume).resolve()
        if not (ckpt / "adapter_model.safetensors").is_file():
            die(f"--resume {ckpt}: adapter_model.safetensors not found")
        train_cfg = make_resume_config(cfg_path, ckpt)
        print(f"  resume config: {train_cfg} (only resume_from_checkpoint changed)")

    # .cmd is written only after preflight passes
    cmd_path, log_path = build_cmd(mode_name, train_cfg, spec["out"])
    print(f"\n--- {cmd_path} ---")
    print(cmd_path.read_text(encoding="utf-8").rstrip())
    print(f"-----------------------")

    if not launch:
        print("DRY RUN: nothing launched. Re-run with --yes to launch.")
        return

    LOGS.mkdir(parents=True, exist_ok=True)
    spec["out"].mkdir(parents=True, exist_ok=True)
    # No -RedirectStandardOutput here: redirection forces CreateProcess without
    # a new console, so the child shares this console and dies with a
    # window-CLOSE event when powershell exits (MKL/forrtl aborts on it).
    # Without redirection, ShellExecute gives the child its own console; the
    # .cmd self-redirects everything to the log file.
    ps = (f"Start-Process cmd.exe -ArgumentList '/c','{cmd_path}' "
          f"-WindowStyle Hidden -PassThru | Select-Object -ExpandProperty Id")
    proc = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                          capture_output=True, text=True, cwd=str(ROOT))
    pid = proc.stdout.strip()
    if not pid.isdigit():
        die(f"Start-Process failed: {proc.stderr.strip() or proc.stdout.strip()}")
    print(f"launched detached: PID {pid}")
    print(f"log:      {log_path}")
    print("monitoring:")
    print(f"  Get-Content {log_path} -Wait -Tail 20")
    print(f"  dir {spec['out']}\\checkpoint-*")
    # /T kills the whole process tree — the PID is cmd.exe; the accelerate
    # child would survive a plain Stop-Process -Id
    print(f"  taskkill /PID {pid} /T /F")
    print("not waiting for completion.")


if __name__ == "__main__":
    main()
