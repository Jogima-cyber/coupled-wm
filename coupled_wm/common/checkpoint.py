# SPDX-License-Identifier: Apache-2.0
"""Checkpoint helpers: compile/DDP prefix stripping, atomic saves, pretrained dirs.

A *pretrained dir* holds ``model.safetensors`` (floating tensors stored as fp32) and
``config.json`` (``{"model_type": ..., "format_version": 1, ...}``): weights plus the
architecture/inference config, no optimizer state. Training checkpoints (``latest.pt``) are
plain ``torch.save`` dicts written with :func:`atomic_save`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

FORMAT_VERSION = 1
WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"

_WRAPPER_PREFIXES = ("module", "_orig_mod")


def strip_compile_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove ``torch.compile`` (``_orig_mod.``) and DDP (``module.``) wrapper prefixes.

    ``_orig_mod`` is removed wherever it appears (a compiled submodule puts it mid-key, e.g.
    ``encoder._orig_mod.conv.weight``); ``module`` only as a leading wrapper component, so a
    real submodule named ``module`` deeper in the tree is kept.
    """
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        parts = [p for p in key.split(".") if p != "_orig_mod"]
        while len(parts) > 1 and parts[0] in _WRAPPER_PREFIXES:
            parts.pop(0)
        new_key = ".".join(parts)
        if new_key in out:
            raise KeyError(f"stripping wrapper prefixes maps two keys onto {new_key!r}")
        out[new_key] = value
    return out


def _current_umask() -> int:
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def _tmp_path(path: Path) -> Path:
    """The fixed temporary name of ``path``: a process killed mid-write leaves at most this one
    file, which the next save of ``path`` truncates. Each target has a single writer."""
    return path.with_name(f".{path.name}.tmp")


def atomic_save(obj: Any, path: str | os.PathLike) -> None:
    """``torch.save`` to a temporary file in the target directory, fsync, then ``os.replace``.

    A crash mid-write leaves the previous file intact. The file gets the permissions of a
    regular ``open()`` under the current umask.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o666 & ~_current_umask())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_checkpoint(path: str | os.PathLike, map_location: Any = "cpu", mmap: bool = True) -> Any:
    """Load a training checkpoint (memory-mapped by default, so large weights are not copied)."""
    return torch.load(path, map_location=map_location, mmap=mmap, weights_only=False)


def _to_json(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if isinstance(obj, torch.Tensor) and obj.ndim == 0:
        return obj.item()
    if hasattr(obj, "item") and getattr(obj, "ndim", None) == 0:  # numpy scalar
        return obj.item()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(f"config value of type {type(obj).__name__} is not JSON-serialisable: {obj!r}")


def save_pretrained(save_dir: str | os.PathLike, state_dict: Mapping[str, torch.Tensor], config: Mapping) -> Path:
    """Write ``model.safetensors`` + ``config.json`` into ``save_dir``.

    Wrapper prefixes are stripped, floating tensors are cast to fp32, all tensors are moved
    to CPU and made contiguous. ``config`` must contain ``model_type``; ``format_version`` is
    added. Each file is written atomically.
    """
    from safetensors.torch import save_file

    if "model_type" not in config:
        raise KeyError("save_pretrained: config must contain 'model_type'")
    if config.get("format_version", FORMAT_VERSION) != FORMAT_VERSION:
        raise ValueError(f"save_pretrained writes format_version {FORMAT_VERSION}, got {config['format_version']}")

    tensors: dict[str, torch.Tensor] = {}
    seen_storage: set[int] = set()
    for key, value in strip_compile_prefix(state_dict).items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] is a {type(value).__name__}, not a tensor")
        t = value.detach().to("cpu")
        if t.is_floating_point():
            t = t.to(torch.float32)
        t = t.contiguous()
        ptr = t.untyped_storage().data_ptr()
        if t.numel() > 0 and ptr in seen_storage:
            t = t.clone()
        seen_storage.add(t.untyped_storage().data_ptr())
        tensors[key] = t

    cfg = _to_json(dict(config))
    cfg = {"model_type": cfg.pop("model_type"), "format_version": FORMAT_VERSION,
           **{k: v for k, v in cfg.items() if k != "format_version"}}

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tmp_w = _tmp_path(save_dir / WEIGHTS_NAME)
    tmp_c = _tmp_path(save_dir / CONFIG_NAME)
    try:
        save_file(tensors, str(tmp_w), metadata={"format": "pt"})
        tmp_c.write_text(json.dumps(cfg, indent=2) + "\n")
        os.replace(tmp_w, save_dir / WEIGHTS_NAME)
        os.replace(tmp_c, save_dir / CONFIG_NAME)
    finally:
        for tmp in (tmp_w, tmp_c):
            if tmp.exists():
                tmp.unlink()
    return save_dir


def load_pretrained_config(load_dir: str | os.PathLike) -> dict:
    """Read and check ``config.json`` of a pretrained dir."""
    path = Path(load_dir) / CONFIG_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (expected a pretrained dir with {CONFIG_NAME} + {WEIGHTS_NAME})")
    config = json.loads(path.read_text())
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported format_version {config.get('format_version')!r}")
    if "model_type" not in config:
        raise ValueError(f"{path}: missing 'model_type'")
    return config


def load_pretrained(load_dir: str | os.PathLike, device: str | torch.device = "cpu") -> tuple[dict[str, torch.Tensor], dict]:
    """Load ``(state_dict, config)`` from a directory written by :func:`save_pretrained`."""
    from safetensors.torch import load_file

    config = load_pretrained_config(load_dir)
    weights = Path(load_dir) / WEIGHTS_NAME
    if not weights.is_file():
        raise FileNotFoundError(f"{weights} not found")
    return load_file(str(weights), device=str(device)), config


def is_pretrained_dir(path: str | os.PathLike) -> bool:
    p = Path(path)
    return p.is_dir() and (p / CONFIG_NAME).is_file() and (p / WEIGHTS_NAME).is_file()
