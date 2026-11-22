@echo off
REM ============================================================================
REM  omnivoice-lora - install / repair Python environments
REM
REM  Two separate venvs (never mix them):
REM    .venv                          Python 3.12, faster-whisper only, NO torch
REM    vendor\OmniVoice\.venv         Python 3.12, torch 2.8.0+cu128, omnivoice, peft
REM
REM  Idempotent: safe to re-run. Also repairs console-script shims and the
REM  editable install after the project folder is moved or renamed.
REM ============================================================================
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] "uv" not found on PATH.
    echo         Install it first: https://docs.astral.sh/uv/
    exit /b 1
)

echo === [1/3] project venv (.venv): STT / WER ===============================
uv venv .venv --python 3.12 --allow-existing
if errorlevel 1 goto :fail

uv pip install --python .venv\Scripts\python.exe faster-whisper==1.2.1 num2words==0.5.14
if errorlevel 1 goto :fail

REM CTranslate2 (faster-whisper) needs the CUDA 12 runtime DLLs on Windows;
REM scripts\transcribe_stt.py puts these on PATH itself.
uv pip install --python .venv\Scripts\python.exe ^
    nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
if errorlevel 1 goto :fail


echo === [2/3] vendor venv (vendor\OmniVoice\.venv): torch + omnivoice ======
pushd vendor\OmniVoice
if errorlevel 1 goto :fail
REM --inexact: keep packages installed outside the lockfile (e.g. s3prl below).
uv sync --extra lora --inexact
if errorlevel 1 (popd & goto :fail)
popd

REM Speaker-similarity metric needs s3prl pinned; install it without deps so it
REM cannot drag in another torch/numpy build.
uv pip install --python vendor\OmniVoice\.venv\Scripts\python.exe --no-deps s3prl==0.4.18
if errorlevel 1 goto :fail


echo === [3/3] verify =======================================================
vendor\OmniVoice\.venv\Scripts\python.exe -c "import torch, peft, transformers, omnivoice; print('vendor venv ok: torch', torch.__version__, '| cuda', torch.cuda.is_available())"
if errorlevel 1 goto :fail

.venv\Scripts\python.exe -c "import importlib.metadata as m; import faster_whisper, num2words; print('project venv ok: faster-whisper', m.version('faster-whisper'), '| num2words', m.version('num2words'))"
if errorlevel 1 goto :fail

echo.
echo Done. Both environments are ready.
exit /b 0

:fail
echo.
echo [FAILED] see the error above.
exit /b 1
