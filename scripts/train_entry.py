r"""Training entry shim: makes OmniVoice training work on Windows (spawn).

What breaks
    ``omnivoice/training/builder.py`` (``build_dataloaders``, lines 207 and
    250) passes ``length_fn=lambda s: s["length"]`` into
    ``StreamLengthGroupDataset``. The dataset stores the callable, and when
    ``DataLoader`` spawns its workers the dataset is pickled into each one.
    A lambda defined inside ``build_dataloaders`` has no module-level name,
    so unpickling fails with::

        AttributeError: Can't get local object
            'build_dataloaders.<locals>.<lambda>'

Why spawn
    On Windows the ``multiprocessing`` start method is always **spawn**:
    workers are fresh interpreter processes that receive the dataset as a
    pickle over a pipe. On Linux the default is **fork**, which inherits
    memory without pickling, so the un-picklable lambda never surfaces
    there and upstream (v0.2.1, commit ``08be0b4``) is not aware of the bug.

Why a shim and not a vendor patch
    ``vendor/OmniVoice`` is a pinned, read-only checkout of upstream
    v0.2.1 ``08be0b4``; we must not edit it. Instead we subclass
    ``StreamLengthGroupDataset`` here, swap the local lambda for a
    module-level ``_length_fn`` (picklable by reference), and patch
    ``omnivoice.training.builder.StreamLengthGroupDataset`` — the builder
    imported the class by name, so ``build_dataloaders`` resolves the
    patched attribute. Launch with ``-m scripts.train_entry`` instead of
    ``-m omnivoice.cli.train`` (see scripts/launch_train.py).

Spawn re-import note
    The ``if __name__ == "__main__"`` guard below is MANDATORY: spawn
    re-imports the main module in each worker as ``__mp_main__`` to
    unpickle objects defined here. Without the guard, every worker would
    re-run ``main()`` and start training recursively. ``scripts`` must be
    an importable package (hence ``scripts/__init__.py``) and the project
    root must be on ``PYTHONPATH`` so workers can import
    ``scripts.train_entry`` by name.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import omnivoice.training.builder as builder
from omnivoice.data.batching import StreamLengthGroupDataset


def _length_fn(sample):
    return sample["length"]


class _PicklableStreamLengthGroupDataset(StreamLengthGroupDataset):
    """Drop-in replacement that swaps a local (unpicklable) ``length_fn``
    for the module-level ``_length_fn``. Any other callable is left
    untouched."""

    def __init__(self, *args, **kwargs):
        lf = kwargs.get("length_fn")
        if lf is not None and "<locals>" in getattr(lf, "__qualname__", ""):
            kwargs["length_fn"] = _length_fn
            print(
                "[train_entry] replaced build_dataloaders' local lambda length_fn "
                "with module-level _length_fn: Windows spawn workers pickle the "
                "dataset, and a local lambda is not picklable"
            )
        super().__init__(*args, **kwargs)


# build_dataloaders resolves StreamLengthGroupDataset as a module global of
# omnivoice.training.builder, so patching the attribute is what takes effect.
builder.StreamLengthGroupDataset = _PicklableStreamLengthGroupDataset


def main():
    from omnivoice.cli.train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
