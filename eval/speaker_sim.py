"""Speaker similarity (timbre match) for a rendered variant — vendor venv only.

Embedding model: the upstream ECAPA-TDNN/WavLM from
``omnivoice/eval/models/ecapa_tdnn_wavlm.py`` (weights: k2-fsa/TTS_eval_models,
cached under vendor/eval_models). This is the metric that answers "does it
sound like Mita".

Two anchors are reported:
- ``sim_centroid_*`` — cosine to the mean of the 60 frozen reference
  embeddings (kept for continuity with the first baseline).
- ``sim_best_*`` (PRIMARY) — max cosine over the 60 individual references.
  The centroid dilutes the anchor; the max-over-refs statistic is the
  reviewer-sanctioned selection metric. Ceiling = leave-one-out max (each
  ref vs the other 59).

Reference set (frozen once): 60 clips sampled deterministically (seed 42,
sorted by audio_path) from data/finetune/manifest_train.jsonl ->
eval/sim_refs.json. Later checkpoints are compared against the same anchor.

Usage:
    vendor\\OmniVoice\\.venv\\Scripts\\python.exe eval\\speaker_sim.py --make-refs
    vendor\\OmniVoice\\.venv\\Scripts\\python.exe eval\\speaker_sim.py ^
      --audio-dir eval\\audio\\base_native --out eval\\results\\base_native.sim.json ^
      --variant base_native

Importable by eval/run_eval.py: ``from eval.speaker_sim import compute, SimUnavailableError``.

NOTE: the upstream feature extractor is loaded via
``torch.hub.load(..., "wavlm_local", source="local")`` whose hubconf.py imports
``s3prl`` (OmniVoice's optional ``eval`` extra). s3prl==0.4.18 IS installed in
this vendor venv (2026-11-15, ``uv pip install --no-deps s3prl==0.4.18``) —
SIM works. If a future session sees ``ModuleNotFoundError: s3prl``, reinstall
it the same way (no-deps; it must not pull other packages). If it is missing
and cannot be installed, this script raises ``SimUnavailableError`` with the
exact error instead of substituting another embedding model.
"""

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FINETUNE_MANIFEST = ROOT / "data" / "finetune" / "manifest_train.jsonl"
REFS_PATH = ROOT / "eval" / "sim_refs.json"
PHRASES_PATH = ROOT / "eval" / "phrases.jsonl"
EVAL_MODELS_DIR = ROOT / "vendor" / "eval_models"
SV_WEIGHTS = "speaker_similarity/wavlm_large_finetune.pth"
SSL_DIR = "speaker_similarity/wavlm_large"
N_REFS = 60
REF_SEED = 42


class SimUnavailableError(RuntimeError):
    """Raised when the upstream embedding model cannot be loaded in this environment."""


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


def ffprobe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def build_refs(out_path: Path, force: bool = False) -> dict:
    """Deterministic seeded sample of 60 clips from the finetune train manifest."""
    if out_path.exists() and not force:
        sys.exit(f"refusing to overwrite frozen reference set {out_path} (pass --force)")
    rows = [json.loads(l) for l in FINETUNE_MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["audio_path"])  # sort by audio_path BEFORE sampling
    picked = random.Random(REF_SEED).sample(rows, N_REFS)
    refs = []
    for r in picked:
        p = Path(r["audio_path"])
        if not p.is_file():
            sys.exit(f"reference audio missing: {p}")
        refs.append(dict(id=r["id"], audio_path=r["audio_path"], duration_s=round(ffprobe_duration(str(p)), 3)))
    refs.sort(key=lambda r: r["id"])
    doc = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(FINETUNE_MANIFEST.relative_to(ROOT)),
        "selection": {
            "n": N_REFS,
            "seed": REF_SEED,
            "method": "rows sorted by audio_path, then random.Random(seed).sample(rows, n)",
        },
        "refs": refs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out_path} ({len(refs)} refs, total {sum(r['duration_s'] for r in refs):.1f}s)")
    return doc


def refs_sha256(refs: list[dict]) -> str:
    """sha256 over sorted (path, sha256(file)) of the reference clips."""
    h = hashlib.sha256()
    for r in sorted(refs, key=lambda r: r["audio_path"]):
        p = Path(r["audio_path"])
        if not p.is_file():
            sys.exit(f"reference audio missing: {p}")
        fsha = hashlib.sha256(p.read_bytes()).hexdigest()
        h.update(f"{r['audio_path']}\n{fsha}\n".encode("utf-8"))
    return h.hexdigest()


def sim_refs_sha256(refs_path: Path) -> str:
    return hashlib.sha256(refs_path.read_bytes()).hexdigest()


def _ensure_hubconf(eval_models_dir: Path) -> None:
    """Cache-local fix (idempotent, no vendor-tree edits).

    Upstream ecapa_tdnn_wavlm.py calls
        torch.hub.load(os.path.dirname(ssl_model_path), "wavlm_local", source="local", ckpt=...)
    which resolves to <eval_models>/speaker_similarity/hubconf.py, but the HF
    repo ships hubconf.py one level deeper (inside wavlm_large/). Copy it up.
    """
    src = eval_models_dir / SSL_DIR / "hubconf.py"
    dst = eval_models_dir / "speaker_similarity" / "hubconf.py"
    if src.is_file() and not dst.is_file():
        shutil.copyfile(src, dst)
        print(f"[cache fix] copied {src} -> {dst}")


def load_sv_model(eval_models_dir: Path, device: str):
    """Load the upstream ECAPA-TDNN/WavLM embedding model. Raises SimUnavailableError on failure."""
    import torch

    from omnivoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM

    sv_path = eval_models_dir / SV_WEIGHTS
    ssl_dir = eval_models_dir / SSL_DIR
    for p in (sv_path, ssl_dir / "wavlm_large.pt"):
        if not p.is_file():
            raise SimUnavailableError(f"eval model weights missing: {p} (run the k2-fsa/TTS_eval_models download first)")
    _ensure_hubconf(eval_models_dir)
    try:
        model = ECAPA_TDNN_WAVLM(
            feat_dim=1024,
            channels=512,
            emb_dim=256,
            sr=16000,
            ssl_model_path=str(ssl_dir),
        )
    except Exception as e:
        raise SimUnavailableError(
            f"upstream ECAPA_TDNN_WAVLM load failed: {type(e).__name__}: {e}\n"
            "Root cause on this box: the local hubconf.py imports s3prl, which must be installed "
            "in the vendor venv (OmniVoice's optional 'eval' extra): "
            "uv pip install --python vendor/OmniVoice/.venv/Scripts/python.exe --no-deps s3prl==0.4.18"
        ) from e
    state = torch.load(sv_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def _free_model(model) -> None:
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute(
    audio_dir: str,
    variant: str,
    refs_path: str | None = None,
    eval_models_dir: str | None = None,
    device: str = "cuda:0",
    phrases: str | None = None,
    allow_drift: bool = False,
    limit: int = 0,
) -> dict:
    """Embed the frozen refs (centroid + max-over-refs ceilings) and the variant
    clips (exactly the ids listed in the phrases file); return the result dict."""
    import numpy as np
    import torch

    from omnivoice.eval.utils import load_eval_waveform

    refs_path = Path(refs_path or REFS_PATH)
    phrases_path = Path(phrases or PHRASES_PATH)
    eval_models_dir = Path(eval_models_dir or EVAL_MODELS_DIR)
    audio_dir = Path(audio_dir)
    verify_frozen(phrases_path, allow_drift)
    if not refs_path.is_file():
        sys.exit(f"reference set missing: {refs_path} (run: python eval/speaker_sim.py --make-refs)")
    refs_doc = json.loads(refs_path.read_text(encoding="utf-8"))
    refs = refs_doc["refs"]

    # Score exactly the ids listed in the phrases file (same slice as wer.py).
    rows = [json.loads(l) for l in phrases_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit > 0:
        rows = rows[:limit]
    phrase_ids = [r["id"] for r in rows]

    model = load_sv_model(eval_models_dir, device)
    dev = torch.device(device)

    @torch.no_grad()
    def embed(path: str) -> torch.Tensor:
        speech = load_eval_waveform(path, 16000, device=dev, max_seconds=120)
        emb = model([speech])[0]
        return torch.nn.functional.normalize(emb, dim=0)

    # --- reference embeddings (60 x 256, L2-normalised) ---
    ref_embs = torch.stack([embed(r["audio_path"]) for r in refs])
    centroid = torch.nn.functional.normalize(ref_embs.mean(dim=0), dim=0)

    # --- centroid anchor + its ceiling ---
    ref_cos_c = torch.nn.functional.cosine_similarity(ref_embs, centroid.unsqueeze(0), dim=1)
    ceiling_c = {
        "mean": float(ref_cos_c.mean()),
        "p10": float(np.percentile(ref_cos_c.cpu().numpy(), 10)),
        "p90": float(np.percentile(ref_cos_c.cpu().numpy(), 90)),
    }

    # --- max-over-refs anchor + leave-one-out ceiling ---
    cos_matrix = ref_embs @ ref_embs.T  # (60, 60), diagonal = 1.0
    loo = cos_matrix.clone()
    loo.fill_diagonal_(float("-inf"))
    loo_max = loo.max(dim=1).values  # each ref vs the other 59
    ceiling_b = {
        "mean": float(loo_max.mean()),
        "p10": float(np.percentile(loo_max.cpu().numpy(), 10)),
        "p90": float(np.percentile(loo_max.cpu().numpy(), 90)),
    }
    print(
        f"centroid: n={len(refs)}  ceiling mean={ceiling_c['mean']:.4f}  p10={ceiling_c['p10']:.4f}  p90={ceiling_c['p90']:.4f}"
    )
    print(
        f"best(refs): n={len(refs)}  ceiling mean={ceiling_b['mean']:.4f}  p10={ceiling_b['p10']:.4f}  p90={ceiling_b['p90']:.4f}"
    )

    # --- variant clips: exactly the phrase ids ---
    per_phrase = []
    for pid in phrase_ids:
        wav = audio_dir / f"{pid}.wav"
        if not wav.is_file():
            sys.exit(f"missing audio: {wav}")
        emb = embed(str(wav))
        sim_c = float(torch.nn.functional.cosine_similarity(emb, centroid, dim=0))
        sims_b = (ref_embs @ emb).cpu().numpy()  # 60 cosines, both sides L2-normalised
        sim_b = float(sims_b.max())
        per_phrase.append(dict(id=pid, sim_centroid=round(sim_c, 6), sim_best=round(sim_b, 6)))
        print(f"{pid}  centroid {sim_c:.4f}  best {sim_b:.4f}")

    _free_model(model)
    c_arr = np.array([p["sim_centroid"] for p in per_phrase])
    b_arr = np.array([p["sim_best"] for p in per_phrase])
    return {
        "variant": variant,
        "refs": {
            "n": len(refs),
            "refs_sha256": refs_sha256(refs),
            "sim_refs_sha256": sim_refs_sha256(refs_path),
        },
        "sim_primary": "max_over_refs",
        "sim_centroid_mean": round(float(c_arr.mean()), 6),
        "sim_centroid_std": round(float(c_arr.std(ddof=1)), 6),
        "sim_centroid_min": round(float(c_arr.min()), 6),
        "sim_centroid_max": round(float(c_arr.max()), 6),
        "sim_centroid_ceiling": {k: round(v, 6) for k, v in ceiling_c.items()},
        "sim_best_mean": round(float(b_arr.mean()), 6),
        "sim_best_std": round(float(b_arr.std(ddof=1)), 6),
        "sim_best_min": round(float(b_arr.min()), 6),
        "sim_best_max": round(float(b_arr.max()), 6),
        "sim_best_ceiling": {k: round(v, 6) for k, v in ceiling_b.items()},
        "per_phrase": per_phrase,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--make-refs", action="store_true", help="build the frozen reference set (eval/sim_refs.json)")
    ap.add_argument("--force", action="store_true", help="allow overwriting (refs or out)")
    ap.add_argument("--audio-dir", default=None, help="variant dir with <id>.wav")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--refs", default=None, help=f"reference set JSON (default {REFS_PATH})")
    ap.add_argument("--phrases", default=str(PHRASES_PATH), help="phrases JSONL; exactly its ids are scored")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N phrases (same slice as wer.py)")
    ap.add_argument("--allow-drift", action="store_true")
    ap.add_argument("--eval-models-dir", default=str(EVAL_MODELS_DIR))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.make_refs:
        build_refs(REFS_PATH, force=args.force)
        return
    if not args.audio_dir or not args.out:
        sys.exit("--audio-dir and --out are required (or use --make-refs)")
    result = compute(
        audio_dir=args.audio_dir,
        variant=args.variant or Path(args.audio_dir).name,
        refs_path=args.refs,
        eval_models_dir=args.eval_models_dir,
        device=args.device,
        phrases=args.phrases,
        allow_drift=args.allow_drift,
        limit=args.limit,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"json -> {out}")


if __name__ == "__main__":
    main()
