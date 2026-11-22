#!/usr/bin/env bash
# ============================================================================
#  omnivoice-lora - install / repair Python environments (bash)
#
#  Same as setup.cmd, for Git Bash / MSYS2 on Windows (the venvs are Windows
#  venvs, so the interpreters are Scripts/python.exe).
#
#  Two separate venvs (never mix them):
#    .venv                          Python 3.12, faster-whisper only, NO torch
#    vendor/OmniVoice/.venv         Python 3.12, torch 2.8.0+cu128, omnivoice, peft
#
#  Idempotent: safe to re-run. Also repairs console-script shims and the
#  editable install after the project folder is moved or renamed.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv not found on PATH." >&2
    echo "        Install it first: https://docs.astral.sh/uv/" >&2
    exit 1
fi

VENV_PY=.venv/Scripts/python.exe
VENDOR_PY=vendor/OmniVoice/.venv/Scripts/python.exe

echo "=== [1/3] project venv (.venv): STT / WER ==============================="
uv venv .venv --python 3.12 --allow-existing
uv pip install --python "$VENV_PY" faster-whisper==1.2.1 num2words==0.5.14

# CTranslate2 (faster-whisper) needs the CUDA 12 runtime DLLs on Windows;
# scripts/transcribe_stt.py puts these on PATH itself.
uv pip install --python "$VENV_PY" \
    nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12

echo "=== [2/3] vendor venv (vendor/OmniVoice/.venv): torch + omnivoice ======"
(
    cd vendor/OmniVoice
    # --inexact: keep packages installed outside the lockfile (e.g. s3prl below).
    uv sync --extra lora --inexact
)

# Speaker-similarity metric needs s3prl pinned; install it without deps so it
# cannot drag in another torch/numpy build.
uv pip install --python "$VENDOR_PY" --no-deps s3prl==0.4.18

echo "=== [3/3] verify ======================================================="
"$VENDOR_PY" -c "import torch, peft, transformers, omnivoice; print('vendor venv ok: torch', torch.__version__, '| cuda', torch.cuda.is_available())"
"$VENV_PY" -c "import importlib.metadata as m; import faster_whisper, num2words; print('project venv ok: faster-whisper', m.version('faster-whisper'), '| num2words', m.version('num2words'))"

echo
echo "Done. Both environments are ready."
