# OmniVoice LoRA: голос Миты

LoRA-дообучение OmniVoice под Миту из MiSide (русская озвучка DreamCast). Датасет, шарды и обученный soup-адаптер уже собраны. Для инференса готова `models_compact\` (fp16, ~1.33 GB). Тренировочные чекпоинты и `models\` удалены ради диска, разделы 3-6 описывают, как их получить заново. Стек: OmniVoice 0.2.1 (commit `08be0b4`), LoRA r=16 на `Qwen/Qwen3-0.6B`, bf16 + SDPA, RTX 3060 Ti 8 GB.

## Структура

```
configs\          train_config_smoke.json, train_config_mita.json, data_config.json
data\             manifest.jsonl (2669 строк), finetune\{manifest_*, audio\, shards\}
scripts\          extract / build / tokenize / launch / soup / merge / compact / run_mita
eval\             run_eval.py, phrases.jsonl, wer.py, speaker_sim.py
setup.cmd/.sh     установка и починка обоих venv
vendor\OmniVoice\ исходники + отдельный venv
```

Цепочка артефактов: `runs\mita_v1\checkpoint-*` → `runs\mita_v1\soup-345` → `models\` → `models_compact\`.

## 0. Установка пакетов

Установка — один идемпотентный скрипт в двух вариантах: `setup.cmd` для cmd и `setup.sh` для bash (Git Bash / MSYS2). Запуск из корня проекта:

```cmd
setup.cmd
```

```bash
bash setup.sh
```

Что делает:

- `.venv`: Python 3.12, `faster-whisper==1.2.1`, `num2words==0.5.14`, CUDA 12 DLL (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvrtc-cu12`).
- `vendor\OmniVoice\.venv`: `uv sync --extra lora --inexact` плюс `s3prl==0.4.18` с `--no-deps`.
- Smoke-импорт torch и CUDA для vendor-venv, faster-whisper и num2words для project-venv.

Требования: Windows, `uv` в PATH, ffmpeg в PATH, Python 3.12 (uv скачает сам), ~8 GB свободного диска под vendor-venv. Torch живёт только в vendor-venv, окружения не смешивать. Повторный запуск любого из двух скриптов это штатный способ починить окружение после переноса или переименования папки проекта: скрипт пересоздаёт editable install и console-script шимы.

Ручной эквивалент: команды внутри `setup.cmd` и `setup.sh`.

Все команды ниже выполняются из корня проекта и интерпретатором vendor-venv:

```cmd
cd /d C:\develop\omnivoice-lora
set PY=vendor\OmniVoice\.venv\Scripts\python.exe
```

## 1. Подготовка моделей

Модели лежат в HF-кэше `C:\Users\<user>\.cache\huggingface\hub`:

- `k2-fsa/OmniVoice` (~3 GB): база, внутри снапшота свой `audio_tokenizer\`, отдельно токенизатор не скачивается.
- `Qwen/Qwen3-0.6B`: LLM-бэкбон из `configs\train_config_mita.json`.
- `k2-fsa/TTS_eval_models`: кэш в `vendor\eval_models` (~2.4 GB), нужен для SIM.

Порог из `scripts\launch_train.py`: свободно ≥ 40 GB на диске (`MIN_FREE_GB`) и ≥ 6 GB VRAM (`MIN_VRAM_GB`).

## 2. Подготовка данных

**Извлечение** (`scripts\extract_audio.py`) тянет ogg и тексты из MiSide. У скрипта нет argparse: запуск перезаписывает `data\manifest.jsonl`, `data\audio\ru\**` и `data\extract_report.*` целиком. Сейчас манифест на 2,669 строк уже собран, так что запускать только при переустановке игры.

**Сборка корпуса** (`scripts\build_finetune_manifest.py`, флаг `--text-source manifest|final`):

```cmd
%PY% scripts\build_finetune_manifest.py
```

Фильтр `character == "mita"` → `status == "match"` даёт 612 клипов: 582 train + 30 dev. Остальные 93 попадают в `manifest_excluded.jsonl` с причиной. Аудио копируется в `data\finetune\audio\<id>.ogg` с абсолютными путями без пробелов.

Строка `data\finetune\manifest_train.jsonl` (набор ключей фиксирован):

```json
{"id": "LocationDialogue_Location1_11", "audio_path": "C:/develop/omnivoice-lora/data/finetune/audio/LocationDialogue_Location1_11.ogg", "text": "Купишь мне ингредиенты для куриного супа?", "language_id": "ru", "instruct": "mita"}
```

**Токенизация** в webdataset-шарды:

```cmd
%PY% scripts\tokenize_shards.py --calibrate
%PY% scripts\tokenize_shards.py --split train
%PY% scripts\tokenize_shards.py --split dev
```

`--calibrate` прогоняет 20 строк через процессор и проверяет, что теги `<|instruct_start|>mita<|instruct_end|>` реально доходят до обучения. `--force` пересобирает сплит, `--tokenizer-path DIR` задаёт токенизатор вместо автоизвлечённого. Результат: `data\finetune\shards\{train,dev}\` с `audios\`, `txts\` и `data.lst`; строки в `txts` дополнительно несут `audio_duration` и `num_tokens`. `configs\data_config.json` ссылается на оба `data.lst`.

Одна особенность: webdataset не открывает пути вида `C:\...`. Скрипты сами выставляют `GOPEN_REWRITE` и `WDS_SECURE` (константа `GOPEN_REWRITE` в `scripts/tokenize_shards.py`, её же переиспользует `launch_train.py`). Вручную окружение настраивать не нужно.

## 3. Запуск тренировки

```cmd
%PY% scripts\launch_train.py --smoke
%PY% scripts\launch_train.py --smoke --yes
%PY% scripts\launch_train.py --full --yes
```

По умолчанию это dry-run: preflight проверяет конфиг, данные, реальное чтение шарда, диск и VRAM, печатает готовую команду и ничего не запускает. `--yes` стартует detached, лог пишется в `logs\train_smoke.log` или `logs\train_full.log`.

| Режим | Шаги | Время | Каталог |
| --- | --- | --- | --- |
| `--smoke`, `configs\train_config_smoke.json` | 60 | ~9 мин | `runs\mita_smoke` |
| `--full`, `configs\train_config_mita.json` | 1200 ≈ 41 эпоха | ~45-75 мин | `runs\mita_v1` |

Дополнительно: `--resume CKPT`, `--force` (непустой выходной каталог), `--config PATH` (подменить конфиг режима). Каждый чекпоинт это полное fp32-состояние accelerate, около 2.85 GB. С `keep_last_n_checkpoints: -1` сохраняются все 12 штук, это ~33 GB. Место освобождается удалением `runs\mita_v1\checkpoint-*` после выбора адаптера.

## 4. LoRA soup

Усреднение адаптеров нескольких соседних чекпоинтов одного прогона:

```cmd
%PY% scripts\soup_lora.py --name soup-345 --steps 300 400 500 --out runs\mita_v1
```

`--run-dir` по умолчанию `runs\mita_v1`, `--out` по умолчанию `<run-dir>\soups`. Зачем это работает: `lora_A` инициализировался один раз и почти не дрейфует, а `lora_B` несёт основную часть изменения, поэтому среднее по (A, B) близко к среднему по дельтам весов. Нужно минимум два чекпоинта. Результат это обычный adapter dir (`adapter_config.json` + `adapter_model.safetensors` + `soup_meta.json`), считается на CPU за ~10 с. Готовый `runs\mita_v1\soup-345` уже лежит в репозитории, sha256 `d75bcfce…`.

## 5. Merge LoRA

Слияние адаптера с базой в самодостаточный каталог модели:

```cmd
%PY% scripts\merge_lora.py --checkpoint runs\mita_v1\soup-345 --out-dir models --yes
```

Тоже dry-run по умолчанию, `--yes` выполняет merge, `--force` разрешает непустой `--out-dir`. `--base-model` задаёт базу (по умолчанию `k2-fsa/OmniVoice`), её ревизия обязана совпадать с той, на которой учился адаптер. `merge_meta.json` пишется последним, а маркер `MERGE_IN_PROGRESS` заставляет инференс и eval отказаться от каталога. Итог ~3.2 GB fp32, грузится как fp16.

## 6. Компактизация модели

Урезанная копия смерженной модели для деплоя:

```cmd
%PY% scripts\build_compact_model.py --src models --dst models_compact
```

Аргументы `--src` и `--dst` уже имеют эти значения по умолчанию. `--force` разрешает непустой `--dst` (и удаляет протухший `<dst>.partial`), `--verify` сверяет готовый `--dst` с `--src` по всем тензорам и ничего не пишет.

Что получается:

- `model.safetensors` в fp16, все ключи сохранены. Потерь нет: рантайм и так грузит с `dtype=torch.float16`.
- Токенизатор урезан до декодера: остаются `acoustic_decoder.*`, `fc2.*`, `quantizer.*`, энкодер выброшен. Маркер `audio_tokenizer/decode_only.json` перечисляет удалённые ключи.
- `config.json`: `dtype` переведён в `float16`. Сборка пишется в `<dst>.partial` и переименовывается только после проверки.

Размер: ~3.27 GB → ~1.33 GB (-59.4 %).

Каталог `models_compact\` только для инференса. Его нельзя передавать в `merge_lora --base-model`, брать базой для тренировки или указывать в `tokenize_shards --tokenizer-path`: fp16-веса поднимутся обратно, а урезанный токенизатор и маркер переедут в новый артефакт. Мастер-копия остаётся в `models\`, деплоится `models_compact\`. В репозитории лежит уже проверенный экземпляр, раздел нужен для пересборки.

## 7. Запуск модели

`instruct="mita"` это текстовый префикс `<|instruct_start|>mita<|instruct_end|>`, а не веса, поэтому в смерженную модель он не попадает. Апстрим сверяет тег с frozenset `_INSTRUCT_ALL_VALID` в `omnivoice/utils/voice_design.py` и падает с `ValueError`. Патч живёт в памяти:

```python
vd._INSTRUCT_ALL_VALID |= {"mita"}
```

`|=` меняет тот же объект множества, на который ссылается модель. Обычное `=` пересвязало бы имя и оставило модель со старым набором. Штатный `omnivoice-infer --instruct mita` использовать нельзя.

Готовый пример без аргументов: `scripts\run_mita.py`, грузит `models_compact\`, патчит whitelist, пишет одну фразу в `tmp\mita_example.wav` на `NUM_STEP = 32`. Запуск:

```cmd
%PY% scripts\run_mita.py
```

Ядро скрипта:

```python
from scripts.mita_voice.runtime import load_model, patch_instruct, free_model
patch_instruct("mita")
model = load_model(model=str(MODEL_DIR), device="cuda:0")
model.generate(text=[TEXT], language="ru", instruct="mita", num_step=32, ...)
```

Батч по списку фраз:

```cmd
%PY% scripts\mita_voice\generate.py --phrases eval\phrases.jsonl --out-dir tmp\gen --instruct mita --model models_compact --num-step 32
```

`--model` и `--lora-adapter` взаимоисключающие. `--num-step` меняет результат: у `generate.py` дефолт 64, у `eval\run_eval.py` 32. Записывайте значение рядом с любым сравнением качества.

## 8. Оценка

```cmd
%PY% eval\run_eval.py --variant merged --instruct mita --model models_compact --num-step 32
```

WER по замороженным 20 фразам (`eval\phrases.jsonl`) через faster-whisper в project venv, SIM через ECAPA из `vendor\eval_models`. Результат: `eval\results\<variant>.json` и `index.json`. `--reuse-audio` переиспользует готовые wav и сверяет их sha256.
