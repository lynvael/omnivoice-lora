r"""Render frozen phrases with OmniVoice (base or LoRA) — deterministic per-phrase seeds.

Usage (vendor venv):
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\mita_voice\generate.py ^
      --phrases eval\phrases.jsonl --out-dir eval\audio\base_native --instruct "female, young adult"

Importable by eval/run_eval.py: ``from scripts.mita_voice.generate import run``.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.merge_lora import MARKER  # noqa: E402
from scripts.mita_voice.runtime import (  # noqa: E402
    MODEL_NAME,
    free_model,
    load_model,
    patch_instruct,
)

DEFAULTS = dict(
    num_step=64,
    guidance_scale=2.0,
    t_shift=0.1,
    seed=1234,
    limit=0,
    device="cuda:0",
)

# Sanity gates (flag, never drop)
RMS_SILENCE = 1e-4
DUR_MIN_S = 0.3
DUR_MAX_S = 20.0
WPS_MAX = 5.0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phrases", required=True, help="phrases JSONL file")
    ap.add_argument(
        "--out-dir",
        required=True,
        help="output dir for <id>.wav + generation_meta.json",
    )
    ap.add_argument(
        "--instruct",
        required=True,
        help='voice-design instruct, e.g. "female, young adult" or "mita"',
    )
    m = ap.add_mutually_exclusive_group()
    m.add_argument(
        "--lora-adapter",
        default=None,
        help="LoRA adapter dir (adapter_config.json + adapter_model.safetensors)",
    )
    m.add_argument(
        "--model",
        default=None,
        help="merged model dir from scripts/merge_lora.py (model.safetensors + tokenizer + audio_tokenizer); "
        "mutually exclusive with --lora-adapter",
    )
    ap.add_argument("--num-step", type=int, default=DEFAULTS["num_step"])
    ap.add_argument("--guidance-scale", type=float, default=DEFAULTS["guidance_scale"])
    ap.add_argument("--t-shift", type=float, default=DEFAULTS["t_shift"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument(
        "--limit",
        type=int,
        default=DEFAULTS["limit"],
        help="render only the first N phrases",
    )
    ap.add_argument("--device", default=DEFAULTS["device"])
    ap.add_argument(
        "--force", action="store_true", help="overwrite a non-empty out-dir"
    )
    return ap


def read_phrases(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                assert "id" in row and "text" in row, (
                    f"phrase row missing id/text: {row}"
                )
                rows.append(row)
    return rows


def check_out_dir(out_dir: Path, force: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        sys.exit(f"refusing to overwrite non-empty out-dir {out_dir} (pass --force)")
    out_dir.mkdir(parents=True, exist_ok=True)


def render_one(model, phrase, instruct, gen_params):
    """Seed + generate one phrase. Returns (audio_np, wall_s, per_seed)."""
    import numpy as np
    from omnivoice.utils.common import fix_random_seed

    seed = gen_params["seed_base"] * 1000 + phrase["_index"]
    fix_random_seed(seed)  # immediately before the stochastic generate()
    t0 = time.time()
    audios = model.generate(
        text=[phrase["text"]],
        language="ru",
        instruct=instruct,
        num_step=gen_params["num_step"],
        guidance_scale=gen_params["guidance_scale"],
        t_shift=gen_params["t_shift"],
        postprocess_output=True,
    )
    wall_s = time.time() - t0
    audio = np.asarray(audios[0], dtype="float64")
    return audio, wall_s, seed


def sanity_flags(audio, sr: int, text: str) -> list[str]:
    """Flag (never drop) clips that cannot be valid speech.

    Gates: silence (RMS), duration, and speech rate. The old clipping gate
    (peak > 0.999) was removed: upstream post-processing peak-normalises to
    exactly 0.5, so it can never fire. No low-rate gate either: 16.7% of the
    612 real Mita clips are < 1.5 words/s because of silences.
    """
    import numpy as np

    flags = []
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    dur = audio.size / sr
    if rms < RMS_SILENCE:
        flags.append("silence")
    if dur < DUR_MIN_S:
        flags.append("too_short")
    if dur > DUR_MAX_S:
        flags.append("too_long")
    # Speech-rate gate: over the 612 real Mita clips words/s is mean 2.16,
    # median 2.12, p95 3.39, max 4.17, and 0.00% exceed 5.0 — so a clip above
    # 5.0 wps (e.g. the native p17 at 9.6 wps) is a truncation, not speech.
    words = len(text.split())
    if dur > 0 and words / dur > WPS_MAX:
        flags.append("too_fast")
    return flags


def adapter_weights_path(adapter_dir: str | Path) -> Path:
    """The LoRA weights file in an adapter dir.

    Prefers ``adapter_model.safetensors``; accepts ``adapter_model.bin`` if
    that is the only weights file present.
    """
    d = Path(adapter_dir)
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = d / name
        if p.is_file():
            return p
    sys.exit(
        f"lora adapter weights not found in {d}: expected adapter_model.safetensors or adapter_model.bin"
    )


def adapter_identity(adapter_dir: str | Path) -> dict:
    """Content identity of a LoRA adapter dir: hash of its weights file.

    Recorded in generation_meta.json so ``--reuse-audio`` can prove the audio
    was rendered with THIS checkpoint — not just a directory with the same
    name (two different checkpoints can both be named ``lora-600``).
    """
    weights = adapter_weights_path(adapter_dir)
    data = weights.read_bytes()
    return {
        "adapter_sha256": hashlib.sha256(data).hexdigest(),
        "adapter_path_resolved": str(Path(adapter_dir).resolve()),
        "adapter_bytes": len(data),
    }


def _sha256_file(p: Path) -> tuple[str, int]:
    """(sha256, byte size) of a file, streamed in 16 MB chunks."""
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def model_identity(model_dir: str | Path) -> dict:
    """Content identity of a merged model dir: hash of its model.safetensors.

    Recorded in generation_meta.json so ``--reuse-audio`` can prove the audio
    was rendered with THIS merged model. Empty dict when the path is not a
    local dir containing model.safetensors (e.g. an HF repo id).
    """
    weights = Path(model_dir) / "model.safetensors"
    if not weights.is_file():
        return {}
    sha, size = _sha256_file(weights)
    return {"model_sha256": sha, "model_bytes": size}


def run(
    phrases: str,
    out_dir: str,
    instruct: str,
    lora_adapter: str | None = None,
    model: str | None = None,
    num_step: int = DEFAULTS["num_step"],
    guidance_scale: float = DEFAULTS["guidance_scale"],
    t_shift: float = DEFAULTS["t_shift"],
    seed: int = DEFAULTS["seed"],
    limit: int = DEFAULTS["limit"],
    device: str = DEFAULTS["device"],
    force: bool = False,
) -> dict:
    """Render all phrases; write wavs + generation_meta.json; return the meta dict."""
    import numpy as np
    import peft
    import soundfile as sf
    import torch
    import transformers

    # argparse's mutually exclusive group does not protect programmatic callers
    if lora_adapter and model:
        raise ValueError("lora_adapter and model are mutually exclusive")
    # a dir carrying the merge marker is a half-written model (see merge_lora.py);
    # refusing here keeps an interrupted merge from being rendered as a finished one
    if model and (Path(model) / MARKER).is_file():
        sys.exit(
            f"refusing to render from {model}: {MARKER} marker present — a merge is in "
            "progress or was interrupted. Rerun scripts/merge_lora.py (it verifies the "
            f"artifact and removes the marker) or delete {Path(model) / MARKER} if the dir is known-good."
        )

    phrases_path = Path(phrases)
    out = Path(out_dir)
    check_out_dir(out, force)

    rows = read_phrases(phrases_path)
    if limit > 0:
        rows = rows[:limit]
    for i, row in enumerate(rows):
        row["_index"] = i  # index in the phrase file (stable across --limit runs)

    phrases_sha = hashlib.sha256(phrases_path.read_bytes()).hexdigest()
    patch_instruct(instruct)  # before any generation
    adapter_meta = (
        adapter_identity(lora_adapter) if lora_adapter else {}
    )  # fail fast if weights missing
    model_meta = (
        model_identity(model) if model else {}
    )  # model_sha256 / model_bytes (only when --model dir has weights)

    t_load = time.time()
    loaded = load_model(device=device, lora_adapter=lora_adapter, model=model)
    load_s = time.time() - t_load
    sr = int(loaded.sampling_rate)
    print(f"model loaded in {load_s:.1f}s, sampling_rate={sr}, device={device}")

    gen_params = dict(
        seed_base=seed,
        num_step=num_step,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
        postprocess_output=True,
        language="ru",
        batch=1,
    )

    per_phrase = []
    t_all = time.time()
    for row in rows:
        audio, wall_s, per_seed = render_one(loaded, row, instruct, gen_params)
        wav_path = out / f"{row['id']}.wav"
        sf.write(
            str(wav_path), audio.astype("float32"), sr
        )  # WAV default subtype: PCM_16
        wav_bytes = wav_path.read_bytes()  # hash the file actually written
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        dur = audio.size / sr
        flags = sanity_flags(audio, sr, row["text"])
        per_phrase.append(
            dict(
                id=row["id"],
                seed=per_seed,
                seconds_audio=round(dur, 4),
                rms=round(rms, 6),
                peak=round(peak, 6),
                wall_s=round(wall_s, 2),
                wav_sha256=hashlib.sha256(wav_bytes).hexdigest(),
                wav_bytes=len(wav_bytes),
                flags=flags,
            )
        )
        flag_txt = f"  FLAG[{','.join(flags)}]" if flags else ""
        print(
            f"{row['id']}  {dur:.2f}s  rms {rms:.3f}  peak {peak:.2f}  {wall_s:.1f}s wall{flag_txt}"
        )

    free_model(loaded)

    meta = {
        "phrases": str(phrases_path),
        "phrases_sha256": phrases_sha,
        "instruct": instruct,
        "lora_adapter": lora_adapter,
        **adapter_meta,  # adapter_sha256 / adapter_path_resolved / adapter_bytes (only when --lora-adapter)
        "seed": seed,
        "params": {
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "t_shift": t_shift,
            "postprocess_output": True,
            "language": "ru",
            "batch": 1,
            "dtype": "float16",
        },
        "model": str(Path(model).resolve()) if model else MODEL_NAME,
        **model_meta,  # model_sha256 / model_bytes (only when --model points at a dir with model.safetensors)
        "sampling_rate": sr,
        "n_phrases": len(rows),
        "load_s": round(load_s, 2),
        "wall_s_total": round(time.time() - t_all, 2),
        "per_phrase": per_phrase,
        "env": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    meta_path = out / "generation_meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"meta -> {meta_path}")
    return meta


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    return run(
        phrases=args.phrases,
        out_dir=args.out_dir,
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


if __name__ == "__main__":
    main()
