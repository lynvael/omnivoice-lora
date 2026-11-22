"""Transcribe Mita-family clips (data/manifest.jsonl, is_mita=true) with faster-whisper (large-v3, ru).
Output: data/manifest_stt.jsonl, one line per clip: id, location, audio_path, stt_result."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.jsonl"
OUT = ROOT / "data" / "manifest_stt.jsonl"
MODEL_NAME = "large-v3"
LANGUAGE = "ru"
BEAM_SIZE = 5
PROGRESS_EVERY = 100
PROBE = np.zeros(1600, dtype=np.float32)  # 0.1 s of silence at 16 kHz

if sys.platform == "win32":  # ctranslate2 CUDA DLLs are not on PATH by default
    for _pkg in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
        _d = ROOT / ".venv" / "Lib" / "site-packages" / "nvidia" / _pkg / "bin"
        if _d.is_dir():
            os.environ["PATH"] = str(_d) + os.pathsep + os.environ["PATH"]


def select():
    rows = []
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["is_mita"]:
                    rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()
    done = {json.loads(l)["id"] for l in open(OUT, encoding="utf-8") if l.strip()} if OUT.exists() else set()
    print(f"already done: {len(done)}")

    rows = select()
    if args.limit:
        rows = rows[: args.limit]
    todo = [r for r in rows if f"{r['category'].replace(' ', '_')}_{r['id']}" not in done]
    print(f"total: {len(rows)}, to do: {len(todo)}")
    if not todo:
        return

    from faster_whisper import WhisperModel
    device_used = args.device
    if device_used == "cuda":
        try:
            model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
            list(model.transcribe(PROBE, language=LANGUAGE, beam_size=1)[0])  # probe first GPU op
        except Exception as e:
            print(f"[warn] CUDA failed ({type(e).__name__}: {e}); falling back to cpu/int8", file=sys.stderr)
            device_used = "cpu"
    if device_used == "cpu":
        model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8", cpu_threads=8)
    print(f"device: {device_used}")

    t0, n = time.time(), 0
    with open(OUT, "a", encoding="utf-8", newline="\n") as out:
        for r in todo:
            try:
                segments, _ = model.transcribe(str(ROOT / r["audio_relpath"]), language=LANGUAGE, beam_size=BEAM_SIZE,
                                                temperature=0, vad_filter=False, condition_on_previous_text=False)
                stt = "".join(seg.text for seg in segments).strip()
            except Exception as e:
                print(f"[error] {r['category']}/{r['id']}: {type(e).__name__}: {e}", file=sys.stderr)
                stt = None
            row = {"id": f"{r['category'].replace(' ', '_')}_{r['id']}", "location": r["category"],
                   "audio_path": r["audio_relpath"], "stt_result": stt}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n += 1
            if n % PROGRESS_EVERY == 0:
                print(f"{n}/{len(todo)}  {n / (time.time() - t0):.2f} items/s  eta {(len(todo) - n) / (n / (time.time() - t0)) / 60:.1f} min")
    print(f"done: {n} transcribed in {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    main()
