#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Action layout of a dataset (command envelope and grip clamp), as YAML snippets for the configs.

    python scripts/compute_action_envelope.py data/g1/new_wm_h5_segmented_small/world_model \\
        --grip-dims 20 21 [--include 'chunk_*.h5'] [--decimals 2] [--rounding nearest|up]

The G1 environment clamps every integrated absolute command to the symmetric envelope
``[-action_scale, action_scale]``, and the ``grip_dims`` additionally to ``grip_range``. The G1
values (``coupled_wm.world_model.models.G1_ACTION``) are ``max(|min|, |max|)`` of each command dim
over the 228 ``chunk_*.h5`` files of the G1 data, rounded to 2 decimals, with grips 20 and 21 in
``[0, 1]``; this script reproduces the envelope with ``--include 'chunk_*.h5'``. ``--rounding up``
instead rounds up, so the envelope (held in fp32 by the environment) contains every recorded
command. The per-dim minimum and maximum are printed too (the envelope is symmetric, so for an
asymmetric joint it also admits commands the data never reached).

``--grip-dims`` is required: list the dims to clamp to ``grip_range`` (G1: ``20 21``), or pass
it with no value for a robot without grip dims. ``grip_range`` is the data's range over those
dims, rounded outward (omitted without grip dims). Set the layout in one of two places:

* the ``env`` block of the RL config (``action_scale``, ``grip_dims``, ``grip_range``);
* the dynamics config, as a complete ``export.metadata.action`` block (``dim``, ``grip_dims``,
  ``grip_range``, ``scale``), so that it is written into the exported ``config.json`` (with
  ``export.preset: null``: the ``g1`` preset is for G1 data, and its other ``action`` keys would
  stay). On the command line the key needs the ``+`` prefix
  (``--set '+export.metadata.action={...}'``). For a trained model, ``python -m
  coupled_wm.world_model.export --metadata <yaml>`` merges the same block into its export.

Keys left ``null`` in the env block come from the dynamics ``config.json`` ``action`` block, and
keys missing there fall back to the G1 layout one by one: an unset ``grip_dims`` clamps dims 20
and 21 to ``[0, 1]`` (and raises for a dataset with fewer than 22 command dims).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

try:
    import coupled_wm  # noqa: F401
except ModuleNotFoundError:  # source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupled_wm.data.h5_dataset import list_h5_files


def command_range(files: Sequence[str], key: str = "commands") -> dict:
    """Per-dim ``min``, ``max`` and ``absmax`` of ``key`` over ``files``, plus the frame count."""
    lo = hi = None
    frames = 0
    for path in files:
        with h5py.File(path, "r") as f:
            cmd = np.asarray(f[key][()], dtype=np.float64)
        if cmd.ndim != 2:
            raise ValueError(f"{path}: '{key}' must be (T, A), got {cmd.shape}")
        if lo is not None and cmd.shape[1] != lo.shape[0]:
            raise ValueError(f"{path}: {cmd.shape[1]} command dims, earlier files have {lo.shape[0]}")
        if not np.isfinite(cmd).all():
            raise ValueError(f"{path}: '{key}' has non-finite values")
        f_lo, f_hi = cmd.min(axis=0), cmd.max(axis=0)
        lo = f_lo if lo is None else np.minimum(lo, f_lo)
        hi = f_hi if hi is None else np.maximum(hi, f_hi)
        frames += cmd.shape[0]
    return {"min": lo, "max": hi, "absmax": np.maximum(np.abs(lo), np.abs(hi)), "frames": frames}


def round_up(v: float, decimals: int) -> float:
    """The smallest multiple of ``10**-decimals`` that is ``>= v`` once both are cast to fp32."""
    step = 10.0 ** -decimals

    def at(k: int) -> np.float32:
        return np.float32(round(k * step, decimals))

    k = math.ceil(v / step)
    while at(k - 1) >= np.float32(v):
        k -= 1
    while at(k) < np.float32(v):
        k += 1
    return round(k * step, decimals) + 0.0


def envelope(absmax: np.ndarray, decimals: int, rounding: str = "nearest") -> list[float]:
    """``absmax`` rounded to ``decimals`` decimals (``"nearest"`` or ``"up"``); never below one step."""
    step = 10.0 ** -decimals
    if rounding == "nearest":
        values = [round(float(v), decimals) for v in absmax]
    elif rounding == "up":
        values = [round_up(float(v), decimals) for v in absmax]
    else:
        raise ValueError(f"rounding must be 'nearest' or 'up', got {rounding!r}")
    return [round(max(round(v / step), 1) * step, decimals) for v in values]


def grip_range(r: dict, grip_dims: Sequence[int], decimals: int) -> list[float]:
    """``[min, max]`` of the ``grip_dims`` commands, rounded outward to ``decimals`` decimals."""
    low = -round_up(-float(min(r["min"][d] for d in grip_dims)), decimals) + 0.0
    high = round_up(float(max(r["max"][d] for d in grip_dims)), decimals)
    if not low < high:
        raise ValueError(f"grip dims {list(grip_dims)} are constant ({low}) in the data")
    return [low, high]


def _fmt(values) -> str:
    return "[" + ", ".join(f"{v:.4g}" for v in values) + "]"


def _yaml_list(values, decimals: int | None = None) -> str:
    """A YAML flow list; floats in fixed point (``decimals``) without trailing zeros."""
    def fixed(v: float) -> str:
        s = f"{v:.{decimals}f}"
        return s.rstrip("0").rstrip(".") if "." in s else s

    return "[" + ", ".join(str(int(v)) if decimals is None else fixed(v) for v in values) + "]"


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description="Action layout (command envelope, grip clamp) of an HDF5 dataset.")
    parser.add_argument("data_dirs", nargs="+", help="dataset directories (*.h5 files)")
    parser.add_argument("--grip-dims", nargs="*", type=int, required=True,
                        help="command dims clamped to grip_range (G1: 20 21); no value = no grip dims")
    parser.add_argument("--include", nargs="+", default=["*.h5"], help="file-name globs to keep")
    parser.add_argument("--exclude", nargs="+", default=[], help="file-name globs to drop")
    parser.add_argument("--min-chunk-index", type=int, default=None, help="drop chunk_<N>.h5 with N below this")
    parser.add_argument("--key", default="commands", help="h5 dataset holding the commands")
    parser.add_argument("--decimals", type=int, default=2, help="round the envelope to this many decimals")
    parser.add_argument("--rounding", choices=("nearest", "up"), default="nearest",
                        help="nearest (the G1 values) or up (contains every recorded command)")
    args = parser.parse_args(argv)

    files = list_h5_files(args.data_dirs, include_globs=args.include, exclude_globs=args.exclude,
                          min_chunk_index=args.min_chunk_index)
    r = command_range(files, args.key)
    dim = len(r["absmax"])
    grips = list(args.grip_dims)
    if any(not 0 <= d < dim for d in grips) or len(set(grips)) != len(grips):
        parser.error(f"--grip-dims must be distinct indices in [0, {dim}), got {grips}")
    scale = _yaml_list(envelope(r["absmax"], args.decimals, args.rounding), args.decimals)
    layout = [("grip_dims", _yaml_list(grips))]
    if grips:
        layout.append(("grip_range", _yaml_list(grip_range(r, grips, args.decimals), args.decimals)))
    snippet = "\n".join([
        f"# max |{args.key}| per dim over {len(files)} files, {r['frames']} frames, rounded ({args.rounding}) "
        f"to {args.decimals} decimals",
        f"# min: {_fmt(r['min'])}",
        f"# max: {_fmt(r['max'])}",
        "# grip_range: min and max over the grip dims, rounded outward" if grips else "# no grip dims",
        "",
        "# RL config (configs/rl/*.yaml):",
        "env:",
        f"  action_scale: {scale}",
        *(f"  {k}: {v}" for k, v in layout),
        "",
        "# or the dynamics config (written into the exported config.json):",
        "export:",
        "  metadata:",
        "    action:",
        f"      dim: {dim}",
        *(f"      {k}: {v}" for k, v in layout),
        f"      scale: {scale}",
    ])
    print(snippet)
    return snippet


if __name__ == "__main__":
    main()
