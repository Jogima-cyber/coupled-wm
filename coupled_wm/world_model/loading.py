# SPDX-License-Identifier: Apache-2.0
"""Load DreamerV4 tokenizer / dynamics weights.

Accepted sources:

* a **pretrained dir** (``config.json`` + ``model.safetensors``, see
  :mod:`coupled_wm.world_model.export`); tokenizer keys are prefixed ``encoder.`` / ``decoder.``;
* a **training checkpoint** (``.pt``) holding the state dicts under ``enc`` / ``dec``
  (tokenizer) or ``dyn`` (dynamics) and the model config under ``model_config``;
* a checkpoint without ``model_config`` (the paper's) together with an explicit ``config=``.

``torch.compile`` / DDP wrapper prefixes are stripped. A *model config* is
``{"model_type": ..., "architecture": {...}, <metadata blocks>}``; ``architecture`` is the
dict :func:`~coupled_wm.world_model.models.build_tokenizer` / ``build_dynamics`` take, and it
must be complete when loading (no defaults are filled in).

The loaders build the modules on the meta device (no random init), recompute the
non-persistent buffers (RoPE tables, attention masks) on CPU in fp32 exactly as a regular
build does, strict-load the weights by assignment and then apply ``.to(device, dtype)``. The
result is identical to building on CPU, casting, and loading the state dict.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn

from coupled_wm.common.checkpoint import (
    WEIGHTS_NAME,
    load_checkpoint,
    load_pretrained,
    strip_compile_prefix,
)
from coupled_wm.common.system import warn_if_compile_unavailable
from coupled_wm.world_model.models import (
    DYNAMICS_MODEL_TYPE,
    DYNAMICS_RUNTIME_KEYS,
    TOKENIZER_MODEL_TYPE,
    TOKENIZER_RUNTIME_KEYS,
    DreamerV4Decoder,
    DreamerV4Dynamics,
    DreamerV4Encoder,
    build_dynamics,
    build_tokenizer,
    dynamics_arch,
    init_nonpersistent_buffers,
    tokenizer_arch,
)

log = logging.getLogger(__name__)

TOKENIZER_STATE_KEYS = ("enc", "dec")
DYNAMICS_STATE_KEY = "dyn"
MODEL_CONFIG_KEY = "model_config"
ENCODER_PREFIX = "encoder."
DECODER_PREFIX = "decoder."

_ARCH_FN = {TOKENIZER_MODEL_TYPE: tokenizer_arch, DYNAMICS_MODEL_TYPE: dynamics_arch}
_RESERVED = ("model_type", "format_version", "architecture")


# --------------------------------------------------------------------------------------
# Model configs
# --------------------------------------------------------------------------------------

def make_model_config(model_type: str, architecture: Mapping[str, Any], **metadata: Any) -> dict:
    """``{"model_type", "architecture": <complete>, **metadata}``; missing architecture keys take
    the paper defaults. This is what a training checkpoint stores under ``model_config``."""
    if model_type not in _ARCH_FN:
        raise ValueError(f"unknown model_type {model_type!r}; expected one of {sorted(_ARCH_FN)}")
    bad = sorted(set(metadata) & set(_RESERVED))
    if bad:
        raise KeyError(f"metadata may not set {bad}")
    return {"model_type": model_type, "architecture": _ARCH_FN[model_type](architecture),
            **copy.deepcopy(metadata)}


def tokenizer_model_config(architecture: Mapping[str, Any], **metadata: Any) -> dict:
    return make_model_config(TOKENIZER_MODEL_TYPE, architecture, **metadata)


def dynamics_model_config(architecture: Mapping[str, Any], **metadata: Any) -> dict:
    return make_model_config(DYNAMICS_MODEL_TYPE, architecture, **metadata)


def as_model_config(config: Mapping[str, Any], model_type: str) -> dict:
    """Normalise a model config (or a bare architecture dict) of ``model_type``.

    The architecture must be complete; ``format_version`` is dropped.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"model config must be a mapping, got {type(config).__name__}")
    if "architecture" in config or "model_type" in config:
        found = config.get("model_type", model_type)
        if found != model_type:
            raise ValueError(f"expected a {model_type!r} config, got model_type {found!r}")
        if "architecture" not in config:
            raise KeyError(f"{model_type} config has no 'architecture' block")
        arch = config["architecture"]
        metadata = {k: copy.deepcopy(v) for k, v in config.items() if k not in _RESERVED}
    else:
        arch, metadata = config, {}
    return {"model_type": model_type, "architecture": _ARCH_FN[model_type](arch, complete=True), **metadata}


def _arch_diff(a: Mapping, b: Mapping) -> list[str]:
    return [f"{k}: {a.get(k)!r} != {b.get(k)!r}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def checkpoint_model_config(ckpt: Mapping, config: Mapping | None, model_type: str, path: str | os.PathLike) -> dict:
    """The model config of a training checkpoint: the embedded ``model_config``, else ``config``.

    When both exist their architectures must agree, and ``config``'s metadata blocks are
    ignored (with a warning).
    """
    embedded = ckpt.get(MODEL_CONFIG_KEY)
    if embedded is None and config is None:
        raise ValueError(
            f"{path} has no embedded {MODEL_CONFIG_KEY!r}; pass config= (export CLI: --config) with its "
            "architecture, e.g. G1_TOKENIZER_ARCH or G1_DYNAMICS_ARCH of coupled_wm.world_model.models"
        )
    cfg = as_model_config(embedded if embedded is not None else config, model_type)
    if embedded is not None and config is not None:
        given = as_model_config(config, model_type)
        diff = _arch_diff(cfg["architecture"], given["architecture"])
        if diff:
            raise ValueError(f"config= disagrees with the architecture stored in {path}: " + "; ".join(diff))
        ignored = sorted(set(given) - set(_RESERVED))
        if ignored:
            log.warning("%s embeds its model config: the blocks %s of config= are ignored (the export CLI adds "
                        "blocks with --metadata)", path, ignored)
    return cfg


def _with_runtime(model_type: str, arch: Mapping, runtime: Mapping | None) -> dict:
    allowed = TOKENIZER_RUNTIME_KEYS if model_type == TOKENIZER_MODEL_TYPE else DYNAMICS_RUNTIME_KEYS
    runtime = dict(runtime or {})
    bad = sorted(set(runtime) - allowed)
    if bad:
        raise KeyError(f"{bad} cannot be overridden at load time (they must match the weights); "
                       f"runtime keys: {sorted(allowed)}")
    return _ARCH_FN[model_type]({**arch, **runtime}, complete=True)


# --------------------------------------------------------------------------------------
# Reading weights
# --------------------------------------------------------------------------------------

def _read(path: str | os.PathLike, model_type: str, config: Mapping | None) -> tuple[dict, dict]:
    """``(stripped state dict or dict of them, model config)`` from a pretrained dir or a .pt."""
    path = Path(path)
    if path.is_dir():
        state, stored = load_pretrained(path)
        cfg = as_model_config(stored, model_type)
        if config is not None:
            diff = _arch_diff(cfg["architecture"], as_model_config(config, model_type)["architecture"])
            if diff:
                raise ValueError(f"config= disagrees with {path}/config.json: " + "; ".join(diff))
        return {"pretrained": state}, cfg
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such checkpoint file or pretrained dir")
    ckpt = load_checkpoint(path, mmap=True)
    if not isinstance(ckpt, Mapping):
        raise TypeError(f"{path} does not hold a checkpoint dict")
    keys = TOKENIZER_STATE_KEYS if model_type == TOKENIZER_MODEL_TYPE else (DYNAMICS_STATE_KEY,)
    missing = [k for k in keys if k not in ckpt]
    if missing:
        kind = "tokenizer" if model_type == TOKENIZER_MODEL_TYPE else "dynamics"
        raise KeyError(f"{path} is not a {kind} checkpoint: missing {missing} (top-level keys: {sorted(ckpt)})")
    cfg = checkpoint_model_config(ckpt, config, model_type, path)
    return {k: strip_compile_prefix(ckpt[k]) for k in keys}, cfg


def read_tokenizer_weights(path: str | os.PathLike, config: Mapping | None = None) -> tuple[dict, dict, dict]:
    """``(encoder state, decoder state, model config)``, wrapper prefixes stripped."""
    states, cfg = _read(path, TOKENIZER_MODEL_TYPE, config)
    if "pretrained" not in states:
        return states["enc"], states["dec"], cfg
    enc, dec, other = {}, {}, []
    for key, value in states["pretrained"].items():
        if key.startswith(ENCODER_PREFIX):
            enc[key[len(ENCODER_PREFIX):]] = value
        elif key.startswith(DECODER_PREFIX):
            dec[key[len(DECODER_PREFIX):]] = value
        else:
            other.append(key)
    if other:
        raise KeyError(f"{path}: tensors outside 'encoder.'/'decoder.': {other[:5]}")
    return enc, dec, cfg


def read_dynamics_weights(path: str | os.PathLike, config: Mapping | None = None) -> tuple[dict, dict]:
    """``(dynamics state, model config)``, wrapper prefixes stripped."""
    states, cfg = _read(path, DYNAMICS_MODEL_TYPE, config)
    return states.get("pretrained", states.get(DYNAMICS_STATE_KEY)), cfg


# --------------------------------------------------------------------------------------
# Building modules from weights
# --------------------------------------------------------------------------------------

def load_state_dict_strict(module: nn.Module, state_dict: Mapping[str, torch.Tensor],
                           allow_missing: Iterable[str] = (), assign: bool = False, name: str = "module") -> None:
    """``load_state_dict`` after stripping wrapper prefixes; raises on any unexpected key and on
    missing keys that do not start with one of the ``allow_missing`` prefixes (e.g. a fresh
    ``reward_head.`` when initialising the reward stage from a backbone)."""
    state_dict = strip_compile_prefix(state_dict)
    allow_missing = tuple(allow_missing)
    expected = set(module.state_dict().keys())
    given = set(state_dict)
    unexpected = sorted(given - expected)
    missing = sorted(k for k in expected - given if not k.startswith(allow_missing))
    if unexpected or missing:
        raise RuntimeError(
            f"{name}: state dict does not match the architecture "
            f"(missing {len(missing)}: {missing[:8]}; unexpected {len(unexpected)}: {unexpected[:8]})"
        )
    module.load_state_dict(state_dict, strict=not allow_missing, assign=assign)


def _materialize(module: nn.Module, state_dict: Mapping[str, torch.Tensor], name: str) -> None:
    init_nonpersistent_buffers(module, "cpu")
    load_state_dict_strict(module, state_dict, assign=True, name=name)
    left = [n for n, t in itertools.chain(module.named_parameters(), module.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"{name}: tensors left uninitialised after loading: {left[:8]}")


def _finalize(module: nn.Module, device, dtype, compile):
    module = module.to(device=device) if dtype is None else module.to(device=device, dtype=dtype)
    module.eval().requires_grad_(False)
    if compile:
        warn_if_compile_unavailable(f"compile={compile}", "compile=False", device)
        module = torch.compile(module) if compile is True else torch.compile(module, mode=compile)
    return module


def load_tokenizer(path: str | os.PathLike, device: torch.device | str = "cpu", dtype: torch.dtype | None = None, *,
                   runtime: Mapping[str, Any] | None = None, config: Mapping | None = None,
                   compile: bool | str = False) -> tuple[DreamerV4Encoder, DreamerV4Decoder, dict]:
    """Load the tokenizer for inference: ``(encoder, decoder, model config)``.

    Modules are in eval mode with gradients disabled. ``dtype`` casts parameters and buffers
    (e.g. ``torch.bfloat16``); ``None`` keeps fp32. ``runtime`` overrides weight-independent
    keys (``seq_len``, ``context_len``, ``dropout``, ``mae_max_mask_prob``). ``compile`` is
    ``False``, ``True`` or a ``torch.compile`` mode applied to both modules.
    """
    enc_sd, dec_sd, cfg = read_tokenizer_weights(path, config)
    arch = _with_runtime(TOKENIZER_MODEL_TYPE, cfg["architecture"], runtime)
    with torch.device("meta"):
        enc, dec = build_tokenizer(arch)
    _materialize(enc, enc_sd, "encoder")
    _materialize(dec, dec_sd, "decoder")
    log.info("loaded tokenizer from %s", path)
    return _finalize(enc, device, dtype, compile), _finalize(dec, device, dtype, compile), cfg


def load_dynamics(path: str | os.PathLike, device: torch.device | str = "cpu", dtype: torch.dtype | None = None, *,
                  runtime: Mapping[str, Any] | None = None, config: Mapping | None = None,
                  compile: bool | str = False) -> tuple[DreamerV4Dynamics, dict]:
    """Load the dynamics model for inference: ``(dynamics, model config)``.

    Same conventions as :func:`load_tokenizer`; ``runtime`` may override ``seq_len``,
    ``context_len`` and ``dropout``.
    """
    dyn_sd, cfg = read_dynamics_weights(path, config)
    arch = _with_runtime(DYNAMICS_MODEL_TYPE, cfg["architecture"], runtime)
    with torch.device("meta"):
        dyn = build_dynamics(arch)
    _materialize(dyn, dyn_sd, "dynamics")
    log.info("loaded dynamics from %s", path)
    return _finalize(dyn, device, dtype, compile), cfg


def lineage_max_clip_len(dynamics_config: Mapping[str, Any]) -> int | None:
    """Longest clip (frames) a dynamics model and the models it was initialised from trained on.

    Read from ``training.lineage_max_clip_len`` (written by the dynamics trainer), else from
    ``training.clips`` of a stage trained without ``init_from``; ``None`` when unknown.
    """
    training = dynamics_config.get("training") or {}
    if training.get("lineage_max_clip_len") is not None:
        return int(training["lineage_max_clip_len"])
    clips = training.get("clips") or {}
    if training.get("init_from") or not clips:
        return None  # an initialised stage: its own clips do not bound what the lineage saw
    lengths = [int(clips[k]) for k in ("short_len", "long_len") if clips.get(k)]
    return max(lengths) if lengths else None


# --------------------------------------------------------------------------------------
# Tokenizer <-> dynamics pairing
# --------------------------------------------------------------------------------------

def file_sha256(path: str | os.PathLike, chunk_size: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def tokenizer_sha256(tokenizer_dir: str | os.PathLike) -> str:
    """Pairing fingerprint of a tokenizer pretrained dir: sha256 of its ``model.safetensors``."""
    return file_sha256(Path(tokenizer_dir) / WEIGHTS_NAME)


def check_pairing(tokenizer_config: Mapping, dynamics_config: Mapping,
                  tokenizer_dir: str | os.PathLike | None = None) -> None:
    """Raise unless the dynamics model can run on this tokenizer's latents.

    Checks the latent shape, and with ``tokenizer_dir`` (a pretrained dir) the sha256 recorded
    in the dynamics config's ``tokenizer`` block; when none is recorded, a warning says that
    only the shapes were checked.
    """
    tok = as_model_config(tokenizer_config, TOKENIZER_MODEL_TYPE)["architecture"]
    dyn_cfg = as_model_config(dynamics_config, DYNAMICS_MODEL_TYPE)
    dyn = dyn_cfg["architecture"]
    if (dyn["num_latents"], dyn["latent_dim"]) != (tok["n_latents"], tok["bottleneck_dim"]):
        raise ValueError(
            f"dynamics expects latents ({dyn['num_latents']}, {dyn['latent_dim']}) but the tokenizer "
            f"produces ({tok['n_latents']}, {tok['bottleneck_dim']})"
        )
    recorded = (dyn_cfg.get("tokenizer") or {}).get("sha256")
    if tokenizer_dir is not None and recorded is None:
        log.warning("the dynamics config records no tokenizer sha256: its pairing with %s is checked by latent "
                    "shape only", tokenizer_dir)
    if tokenizer_dir is not None and recorded is not None:
        actual = tokenizer_sha256(tokenizer_dir)
        if actual != recorded:
            raise ValueError(f"tokenizer {tokenizer_dir} (sha256 {actual[:12]}...) is not the one this dynamics "
                             f"model was trained with (sha256 {recorded[:12]}...)")


__all__ = [
    "DECODER_PREFIX", "DYNAMICS_STATE_KEY", "ENCODER_PREFIX", "MODEL_CONFIG_KEY", "TOKENIZER_STATE_KEYS",
    "as_model_config", "check_pairing", "checkpoint_model_config", "dynamics_model_config", "file_sha256",
    "lineage_max_clip_len", "load_dynamics", "load_state_dict_strict", "load_tokenizer", "make_model_config",
    "read_dynamics_weights", "read_tokenizer_weights", "tokenizer_model_config", "tokenizer_sha256",
]
