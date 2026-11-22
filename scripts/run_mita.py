r"""Minimal end-to-end example: speak as Mita with the deployed model in models\.

No arguments. Edit the constants below for another phrase or output path.

Usage (vendor venv):
    vendor\OmniVoice\.venv\Scripts\python.exe scripts\run_mita.py

What this shows:
  * ``models\`` is a plain merged OmniVoice checkpoint — no PEFT, no adapter flag;
  * ``instruct="mita"`` is a text condition, not part of the weights, and upstream's
    whitelist rejects the tag unless we patch it in memory first;
  * ``language="ru"`` is what the adapter was trained on;
  * output is 24 kHz mono PCM16 in ``tmp\mita_example.wav``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mita_voice.runtime import free_model, load_model, patch_instruct  # noqa: E402

MODEL_DIR = ROOT / "models_compact"
OUT_PATH = ROOT / "tmp" / "mita_example.wav"

TEXT = "«Выкинуть всё, что не относится к ru/en» в этой архитектуре нельзя — не потому что сложно, а потому что такого блока не существует. Но сжать модель можно заметно, и самый большой выигрыш вообще не связан с языками."
INSTRUCT = "mita"
LANGUAGE = "ru"
SEED = 1234

NUM_STEP = 32  # eval recipe; 48-64 sounds cleaner but is slower
GUIDANCE_SCALE = 2
T_SHIFT = 0.1


def main() -> None:
    import numpy as np
    import soundfile as sf
    from omnivoice.utils.common import fix_random_seed

    if not MODEL_DIR.is_dir():
        sys.exit(f"model dir not found: {MODEL_DIR}")

    patch_instruct(INSTRUCT)  # the stock whitelist does not know "mita"
    model = load_model(model=str(MODEL_DIR), device="cuda:0")
    print(f"model: {MODEL_DIR}")
    print(f"instruct={INSTRUCT!r} language={LANGUAGE!r} num_step={NUM_STEP}")

    fix_random_seed(SEED)  # right before the stochastic generate()
    audios = model.generate(
        text=[TEXT],
        language=LANGUAGE,
        instruct=INSTRUCT,
        num_step=NUM_STEP,
        guidance_scale=GUIDANCE_SCALE,
        t_shift=T_SHIFT,
        postprocess_output=True,
        speed=1,
    )

    audio = np.asarray(audios[0], dtype="float32")
    sr = int(model.sampling_rate)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(OUT_PATH), audio, sr)

    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    print(f"text: {TEXT}")
    print(f"wrote {OUT_PATH} | {audio.size / sr:.2f}s | rms {rms:.3f} | sr {sr}")

    free_model(model)


if __name__ == "__main__":
    main()
