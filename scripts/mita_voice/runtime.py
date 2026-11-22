"""Shared runtime helpers for Mita eval inference.

Vendor-venv only (torch/omnivoice). No site-packages edits: the instruct
whitelist patch is in-memory only.

Also applies the decode-only shim for compact dirs built by
``scripts/build_compact_model.py`` (ADR-0001 option D); a normal dir is
untouched. For a compact dir the audio tokenizer is loaded on CPU first,
stripped of its encode-only modules, and only then moved to the target
device, so the ~682 MiB of encoder weights never occupy VRAM (the vendor
loads the tokenizer on the model device, which would allocate the encoder
before the shim could pop it).
"""

import gc
import json
import re
import sys
import types
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import omnivoice.utils.voice_design as vd
from omnivoice.models import omnivoice as om
from omnivoice.models.omnivoice import OmniVoice

MODEL_NAME = "k2-fsa/OmniVoice"

# A compact dir (scripts/build_compact_model.py, ADR-0001 option D) ships only
# the audio tokenizer's decode path and carries this marker.
DECODE_ONLY_MARKER = ("audio_tokenizer", "decode_only.json")
DECODE_ONLY_WEIGHTS = ("audio_tokenizer", "model.safetensors")
DECODE_ONLY_FORMAT_VERSION = 1


def _marker_path(model_dir):
    """The decode-only marker for ``model_dir``, or None when the dir is not compact."""
    d = Path(model_dir) if model_dir else None
    if d is None or not d.is_dir():
        return None
    p = d.joinpath(*DECODE_ONLY_MARKER)
    return p if p.is_file() else None


def _strip_audio_tokenizer_encoder(tok, marker, weights_path, marker_path):
    """Delete encode-only audio-tokenizer modules from ``tok`` per ``marker``.

    Pure and device-agnostic: it reads only the safetensors *header* (no
    tensor bytes), inspects ``tok.state_dict()`` key names, and pops
    registered submodules. It can therefore run on a CPU-resident tokenizer
    (the compact load path) or a GPU-resident one, and it never allocates
    model memory itself.

    Why this is not optional: ``HiggsAudioV2TokenizerModel.from_pretrained``
    only *warns* about missing weights and leaves the modules randomly
    initialised (transformers 5.3: the model is built from config, then the
    state dict is loaded non-strictly).  In a decode-only dir that means
    ``encode()`` would run on garbage, and the random tensors would sit in
    memory for the whole session (~645 MB).

    Guard rails, all driven by ``marker``:

    1. ``format_version`` must be the one this runtime understands;
    2. ``drop.prefixes`` and ``drop.keys`` must name the same top-level
       modules (a marker that under-declares prefixes would leave a random
       module resident while still reporting success);
    3. the **kept** weights must really have been loaded — the shipped
       safetensors header is compared against ``state_dict()`` and the
       missing set must equal ``drop.keys``.  Without this a corrupt or
       wrongly-built dir would decode noise while this function still
       reported success;
    4. the modules to pop come from ``drop.prefixes`` (not a hard-coded
       list), and every one of them must actually be registered before the
       pop;
    5. after the pop, the removed set must equal the declared set.

    ``_modules.pop`` really unregisters the parameters (``setattr(..., None)``
    would leave them registered and keep the memory).  ``encode`` is replaced
    on the instance, never on the class, so non-compact dirs are untouched.

    Returns the removed module names.
    """
    import torch
    from safetensors import safe_open

    fmt = marker.get("format_version")
    if fmt != DECODE_ONLY_FORMAT_VERSION:
        raise RuntimeError(
            f"{marker_path}: unsupported format_version {fmt!r} "
            f"(this runtime understands {DECODE_ONLY_FORMAT_VERSION}); refusing to guess"
        )
    drop = marker.get("drop") or {}
    declared_keys, prefixes = drop.get("keys"), drop.get("prefixes")
    if not isinstance(declared_keys, list) or not isinstance(prefixes, list) or not prefixes:
        raise RuntimeError(f"{marker_path}: malformed drop.keys / drop.prefixes; refusing to guess")

    # drop.prefixes and drop.keys must agree on the top-level modules; a marker
    # that under-declares prefixes would leave a randomly-initialised module
    # resident while the shim still reported success.
    prefix_tops = {str(p).split(".", 1)[0] for p in prefixes}
    key_tops = {str(k).split(".", 1)[0] for k in declared_keys}
    if prefix_tops != key_tops:
        raise RuntimeError(
            f"{marker_path}: drop.prefixes top-level modules {sorted(prefix_tops)} do not "
            f"match drop.keys top-level modules {sorted(key_tops)}; refusing to guess"
        )

    weights = Path(weights_path)
    if not weights.is_file():
        raise RuntimeError(f"{weights} is missing, but {marker_path} says this is a decode-only build")
    with safe_open(str(weights), framework="pt", device="cpu") as f:
        shipped = set(f.keys())
    state_keys = set(tok.state_dict())
    missing = state_keys - shipped
    declared = set(declared_keys)
    if missing != declared:
        unexpected = sorted(missing - declared)
        absent = sorted(declared - missing)
        raise RuntimeError(
            f"{weights}: {len(missing)} model tensors have no shipped weight, but the marker "
            f"declares {len(declared)} dropped keys. Unexpected missing ({len(unexpected)}): "
            f"{unexpected[:8]}. Declared but present ({len(absent)}): {absent[:8]}. "
            "Refusing to run: transformers random-initialises missing weights, so decode() "
            "would emit noise."
        )

    modules = []
    for prefix in prefixes:
        name = str(prefix).split(".", 1)[0]
        if name not in modules:
            modules.append(name)
    absent_mods = sorted(n for n in modules if n not in tok._modules)
    if absent_mods:
        raise RuntimeError(
            f"{marker_path} declares drops for modules {sorted(modules)}, but these are not "
            f"registered on the loaded audio tokenizer: {absent_mods[:8]}"
        )

    removed, params, bytes_freed = [], 0, 0
    for name in modules:
        mod = tok._modules.pop(name, None)  # unregisters parameters + buffers
        if mod is None:
            continue
        n = sum(p.numel() for p in mod.parameters())
        b = sum(p.numel() * p.element_size() for p in mod.parameters())
        b += sum(t.numel() * t.element_size() for t in mod.buffers())
        removed.append(name)
        params += n
        bytes_freed += b
        del mod
    if set(removed) != set(modules):
        raise RuntimeError(
            f"removed {sorted(removed)} but {marker_path} says to drop {sorted(modules)}"
        )

    model_dir = marker_path.parent.parent

    def _encode_unavailable(self, *args, **kwargs):
        raise RuntimeError(
            "audio_tokenizer/encode() is not available in this decode-only build "
            f"({model_dir}): the HuBERT/DAC encoder weights were pruned to save disk and VRAM. "
            "Zero-shot voice cloning from a reference wav needs the full audio_tokenizer/ "
            "directory (see docs/adr/0001-model-compression.md, option D); "
            "a pre-tokenised VoiceClonePrompt still works."
        )

    tok.encode = types.MethodType(_encode_unavailable, tok)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[compact] decode-only audio tokenizer: {len(shipped)} kept weights verified, "
        f"removed {', '.join(removed)} ({params:,} params, {bytes_freed / 2**20:.0f} MiB); "
        "encode() now raises",
        flush=True,
    )
    return removed


def _verify_tokenizer_bytes(weights_path, marker, marker_path):
    """Stat + sha256 the shipped tokenizer file and compare it to the marker.

    Catches a dir with the right key set but different tensor bytes (which
    would decode noise). Fails when the marker's ``dst_sha256``/``dst_bytes``
    are present and mismatched; warns (does not fail) when they are absent.
    """
    import hashlib

    weights = Path(weights_path)
    want_sha = marker.get("dst_sha256")
    want_bytes = marker.get("dst_bytes")
    st = weights.stat()
    if want_bytes is not None and st.st_size != want_bytes:
        raise RuntimeError(
            f"{weights}: size {st.st_size:,} B != marker dst_bytes {want_bytes:,} B — "
            "the tokenizer file does not match the marker; decode() would emit noise"
        )
    if want_sha is None:
        print(
            f"[compact] warning: {marker_path} has no dst_sha256; "
            "tokenizer bytes not verified",
            flush=True,
        )
        return
    h = hashlib.sha256()
    with open(weights, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != want_sha:
        raise RuntimeError(
            f"{weights}: sha256 {got[:16]}… != marker dst_sha256 {want_sha[:16]}… — "
            "the tokenizer bytes do not match the marker; decode() would emit noise"
        )


def _install_cpu_first_tokenizer(model_dir, device):
    """If ``model_dir`` is compact, temporarily replace
    ``omnivoice.models.omnivoice.HiggsAudioV2TokenizerModel`` with a subclass
    whose ``from_pretrained`` loads the tokenizer on CPU, verifies its bytes,
    strips the encode-only modules there, and only then moves the remainder to
    ``device``.

    The vendor resolves the class name from the module's globals at call time
    (``omnivoice.py:364``), so patching the attribute is enough and no vendor
    edit is needed. The encoder is never allocated on the target device, so
    its ~682 MiB never enters VRAM.

    Returns the original class when the patch is installed (so the caller can
    restore it in a ``finally``), else ``None`` (non-compact dir: strict no-op).
    """
    marker_path = _marker_path(model_dir)
    if marker_path is None:
        return None
    import torch

    base = om.HiggsAudioV2TokenizerModel

    class _CpuFirstTokenizer(base):
        @classmethod
        def from_pretrained(cls, path, *args, **kwargs):
            kwargs.pop("device_map", None)
            tok = base.from_pretrained(path, *args, device_map="cpu", **kwargs)
            p = Path(path)
            m_path = p / "decode_only.json"
            marker = json.loads(m_path.read_text(encoding="utf-8"))
            weights_path = p / "model.safetensors"
            _verify_tokenizer_bytes(weights_path, marker, m_path)
            _strip_audio_tokenizer_encoder(tok, marker, weights_path, m_path)
            tok = tok.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return tok

    om.HiggsAudioV2TokenizerModel = _CpuFirstTokenizer
    return base


def _assert_compact_stripped(loaded, model_dir):
    """Read-only post-load check for a compact dir.

    Asserts that the modules declared in ``drop.prefixes`` are gone from
    ``loaded.audio_tokenizer`` and that ``encode()`` raises.

    Deliberately does NOT re-run the ``missing == declared`` header check:
    by this point the state dict is already stripped, so that check would
    read 0 missing vs 351 declared and raise on a healthy model.
    """
    marker_path = _marker_path(model_dir)
    if marker_path is None:
        return
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    tok = getattr(loaded, "audio_tokenizer", None)
    if tok is None:
        return
    prefixes = (marker.get("drop") or {}).get("prefixes") or []
    modules = []
    for prefix in prefixes:
        name = str(prefix).split(".", 1)[0]
        if name not in modules:
            modules.append(name)
    resident = sorted(n for n in modules if n in tok._modules)
    if resident:
        raise RuntimeError(
            f"{marker_path}: compact dir loaded but these encoder modules are still "
            f"resident: {resident} — the CPU-first strip did not run"
        )
    try:
        tok.encode()
    except RuntimeError:
        return
    raise RuntimeError(f"{marker_path}: compact dir loaded but encode() did not raise")


def patch_instruct(instruct: str) -> None:
    """Add the instruct items to the shared whitelist, in memory only.

    ``vd._INSTRUCT_ALL_VALID |= {item}`` mutates the *same set object* that
    ``omnivoice.models.omnivoice`` imported by reference (a plain ``=`` would
    rebind the name in this module and the patch would not be visible there).

    Asserts the patch is visible through ``omnivoice.models.omnivoice`` and
    that the stock native instruct still resolves.
    """
    items = [x.strip().lower() for x in re.split(r"[\uFF0C,]", instruct) if x.strip()]
    for item in items:
        vd._INSTRUCT_ALL_VALID |= {item}
    for item in items:
        assert om._resolve_instruct(item) == item, (
            f"instruct patch not visible through omnivoice.models.omnivoice for {item!r}"
        )
    assert om._resolve_instruct("female, young adult") == "female, young adult", (
        "native instruct broken after patch"
    )


def load_model(
    device: str = "cuda:0",
    dtype=None,
    lora_adapter: str | None = None,
    model: str | None = None,
):
    """Load OmniVoice once; optionally attach a LoRA adapter (merged).

    ``model``: HF repo id or local dir passed to ``from_pretrained``; defaults
    to ``MODEL_NAME``. A merged model dir (see scripts/merge_lora.py) is
    passed here as ``model`` — it is self-contained, so no adapter is
    involved.  A compact dir (scripts/build_compact_model.py) additionally
    gets its dead audio-tokenizer encoder stripped — see
    ``_strip_audio_tokenizer_encoder``. For a compact dir the tokenizer is
    loaded on CPU first and stripped before reaching the target device, so
    the encoder never occupies VRAM.
    """
    import torch

    orig_tok_class = _install_cpu_first_tokenizer(model, device)
    try:
        loaded = OmniVoice.from_pretrained(
            model or MODEL_NAME, device_map=device, dtype=dtype or torch.float16
        )
    finally:
        if orig_tok_class is not None:
            om.HiggsAudioV2TokenizerModel = orig_tok_class
    if orig_tok_class is not None:
        _assert_compact_stripped(loaded, model)
    if lora_adapter:
        from omnivoice.utils.lora import load_lora_adapter

        loaded = load_lora_adapter(loaded, lora_adapter)
    return loaded


def free_model(model) -> None:
    """Drop the model and release CUDA memory so the next model can load."""
    del model
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
