"""Intelligibility (WER) for a rendered variant — runs under the PROJECT venv (no torch).

Usage:
    C:\\develop\\omnivoice-lora\\.venv\\Scripts\\python.exe eval\\wer.py ^
      --phrases eval\\phrases.jsonl --audio-dir eval\\audio\\base_native ^
      --out eval\\results\\base_native.wer.json --variant base_native

ASR settings are exactly those used to build data/manifest_stt.jsonl
(faster-whisper large-v3, ru, beam 5, temperature 0, no VAD, no previous-text
conditioning), with the ctranslate2 CUDA DLL PATH workaround from
scripts/transcribe_stt.py.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from importlib.metadata import version
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import num2words  # pure Python; verbalises pure-integer tokens (upstream-sanctioned, omnivoice[tn])

NUM2WORDS_VERSION = version("num2words")

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "large-v3"
LANGUAGE = "ru"
BEAM_SIZE = 5
PROBE = None  # built after numpy import

if sys.platform == "win32":  # ctranslate2 CUDA DLLs are not on PATH by default
    for _pkg in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
        _d = ROOT / ".venv" / "Lib" / "site-packages" / "nvidia" / _pkg / "bin"
        if _d.is_dir():
            os.environ["PATH"] = str(_d) + os.pathsep + os.environ["PATH"]


def verify_frozen(phrases_path: Path, allow_drift: bool) -> None:
    """Refuse to run if the phrases file drifted from the frozen set."""
    frozen_path = ROOT / "eval" / "frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    sha = hashlib.sha256(phrases_path.read_bytes()).hexdigest()
    if sha != frozen["phrases_sha256"]:
        if allow_drift:
            print(f"[warn] phrases drifted (sha256 {sha[:12]}... != frozen {frozen['phrases_sha256'][:12]}...); --allow-drift set")
        else:
            sys.exit(
                f"phrases file drifted from frozen set: sha256 {sha} != {frozen['phrases_sha256']}. "
                "Pass --allow-drift to override."
            )


def normalize(text: str) -> str:
    """NFKC, lowercase, ё->е, keep only Cyrillic/Latin letters and digits,
    then verbalise pure-integer tokens (num2words, ru) — applied to BOTH
    hypothesis and reference so «42» == «сорок два»."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^a-zа-я0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    def verbalise(tok: str) -> str:
        return num2words.num2words(int(tok), lang="ru") if re.fullmatch(r"\d+", tok) else tok

    text = " ".join(verbalise(t) for t in text.split())
    return re.sub(r"\s+", " ", text).strip()


def word_levenshtein(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Word-level Levenshtein DP with backtrace. Returns (S, D, I)."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        ri = ref[i - 1]
        for j in range(1, n + 1):
            if ri == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    i, j = m, n
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            S += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I


def load_asr():
    import numpy as np
    from faster_whisper import WhisperModel

    global PROBE
    PROBE = np.zeros(1600, dtype=np.float32)  # 0.1 s of silence at 16 kHz
    try:
        model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
        list(model.transcribe(PROBE, language=LANGUAGE, beam_size=1)[0])  # probe first GPU op
        return model, "cuda", "float16"
    except Exception as e:
        print(f"[warn] CUDA failed ({type(e).__name__}: {e}); falling back to cpu/int8", file=sys.stderr)
    model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8", cpu_threads=8)
    return model, "cpu", "int8"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phrases", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default=None, help="variant name recorded in the JSON")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N phrases (same slice as generate.py)")
    ap.add_argument("--allow-drift", action="store_true", help="run even if phrases.jsonl drifted from the frozen set")
    args = ap.parse_args()

    phrases_path = Path(args.phrases)
    audio_dir = Path(args.audio_dir)
    verify_frozen(phrases_path, args.allow_drift)

    rows = [json.loads(l) for l in phrases_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]

    t0 = time.time()
    model, device, compute = load_asr()
    print(f"ASR: {MODEL_NAME} on {device}/{compute} (loaded in {time.time() - t0:.1f}s)")

    per_phrase = []
    tot_S = tot_D = tot_I = tot_ref = 0
    for row in rows:
        wav = audio_dir / f"{row['id']}.wav"
        if not wav.is_file():
            sys.exit(f"missing audio: {wav}")
        segments, info = model.transcribe(
            str(wav),
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            temperature=0,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        hyp_raw = "".join(seg.text for seg in segments).strip()
        ref_n = normalize(row["text_ref"])
        hyp_n = normalize(hyp_raw)
        S, D, I = word_levenshtein(ref_n.split(), hyp_n.split())
        ref_words = len(ref_n.split())
        per_phrase.append(
            dict(
                id=row["id"],
                ref=ref_n,
                hyp=hyp_n,
                hyp_raw=hyp_raw,
                S=S,
                D=D,
                I=I,
                ref_words=ref_words,
                wer_pct=round(100.0 * (S + D + I) / ref_words, 2) if ref_words else 0.0,
                duration_s=round(float(info.duration), 3),
            )
        )
        tot_S += S
        tot_D += D
        tot_I += I
        tot_ref += ref_words
        print(f"{row['id']}  ref_words={ref_words}  S={S} D={D} I={I}  wer {per_phrase[-1]['wer_pct']:.1f}%  hyp «{hyp_raw}»")

    wer_pct = round(100.0 * (tot_S + tot_D + tot_I) / tot_ref, 2) if tot_ref else 0.0
    result = {
        "variant": args.variant or audio_dir.name,
        "phrases_sha256": hashlib.sha256(phrases_path.read_bytes()).hexdigest(),
        "asr": {
            "model": MODEL_NAME,
            "device": device,
            "compute_type": compute,
            "language": LANGUAGE,
            "beam_size": BEAM_SIZE,
            "temperature": 0,
            "vad_filter": False,
            "condition_on_previous_text": False,
        },
        "normalisation": {
            "scheme": "NFKC, lowercase, ё->е, non-letter/digit->space, pure-integer tokens verbalised via num2words(lang=ru) on both sides",
            "num2words_version": NUM2WORDS_VERSION,
        },
        "wer_pct": wer_pct,
        "S": tot_S,
        "D": tot_D,
        "I": tot_I,
        "ref_words": tot_ref,
        "wall_s": round(time.time() - t0, 2),
        "per_phrase": per_phrase,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"AGGREGATE: WER {wer_pct:.2f}%  (S={tot_S} D={tot_D} I={tot_I}) / ref_words={tot_ref}")
    print(f"json -> {out}")


if __name__ == "__main__":
    main()
