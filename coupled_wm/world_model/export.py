# SPDX-License-Identifier: Apache-2.0
"""Export world-model weights as pretrained dirs (``config.json`` + fp32 ``model.safetensors``).

Command line (a training checkpoint -> a pretrained dir)::

    python -m coupled_wm.world_model.export --ckpt runs/tokenizer/latest.pt --out weights/tokenizer
    python -m coupled_wm.world_model.export --ckpt runs/dynamics/latest.pt --out weights/dynamics \
        --tokenizer weights/tokenizer

The checkpoint kind is detected from its state dicts (``enc``/``dec`` = tokenizer, ``dyn`` =
dynamics). Its ``model_config`` is written to ``config.json``; ``--config`` supplies one
(YAML or JSON: a model config or a bare architecture dict) for checkpoints that have none.
``--metadata`` (YAML or JSON) adds or overrides ``config.json`` blocks, key by key within a
block, e.g. the ``action`` block (``scale``, ``grip_dims``, ``grip_range``) and the
``inference`` block the G1 environment reads for a dynamics model trained on new data::

    python -m coupled_wm.world_model.export --ckpt runs/dynamics/latest.pt --out weights/dynamics \
        --tokenizer weights/tokenizer --metadata my_robot_env.yaml

For a dynamics model, ``--tokenizer`` records the sha256 of that tokenizer's weights so
consumers can check the pairing (:func:`coupled_wm.world_model.loading.check_pairing`).

Before writing, the weights are strict-loaded into the architecture (on the meta device) and
checked to be finite. Optimizer state is never exported.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from coupled_wm.common.checkpoint import load_checkpoint, load_pretrained_config, save_pretrained, strip_compile_prefix
from coupled_wm.common.config import load_config
from coupled_wm.common.logging import setup_logging
from coupled_wm.world_model.loading import (
    DECODER_PREFIX,
    DYNAMICS_STATE_KEY,
    ENCODER_PREFIX,
    TOKENIZER_STATE_KEYS,
    checkpoint_model_config,
    as_model_config,
    check_pairing,
    load_state_dict_strict,
    tokenizer_sha256,
)
from coupled_wm.world_model.models import DYNAMICS_MODEL_TYPE, TOKENIZER_MODEL_TYPE, build_dynamics, build_tokenizer

log = logging.getLogger(__name__)


def _verify(module_states: Mapping[str, tuple[torch.nn.Module, Mapping[str, torch.Tensor]]]) -> None:
    for name, (module, state) in module_states.items():
        load_state_dict_strict(module, state, assign=True, name=name)
        bad = [k for k, t in state.items() if t.is_floating_point() and not bool(torch.isfinite(t).all())]
        if bad:
            raise ValueError(f"{name}: non-finite values in {bad[:8]}")


def save_tokenizer_pretrained(out_dir: str | os.PathLike, encoder_state: Mapping[str, torch.Tensor],
                              decoder_state: Mapping[str, torch.Tensor], model_config: Mapping[str, Any], *,
                              verify: bool = True) -> Path:
    """Write a tokenizer pretrained dir; keys are stored as ``encoder.*`` / ``decoder.*``."""
    cfg = as_model_config(model_config, TOKENIZER_MODEL_TYPE)
    enc_sd, dec_sd = strip_compile_prefix(encoder_state), strip_compile_prefix(decoder_state)
    if verify:
        with torch.device("meta"):
            enc, dec = build_tokenizer(cfg["architecture"])
        _verify({"encoder": (enc, enc_sd), "decoder": (dec, dec_sd)})
    state = {f"{ENCODER_PREFIX}{k}": v for k, v in enc_sd.items()}
    state.update({f"{DECODER_PREFIX}{k}": v for k, v in dec_sd.items()})
    out = save_pretrained(out_dir, state, cfg)
    log.info("wrote tokenizer pretrained dir %s (%d tensors)", out, len(state))
    return out


def save_dynamics_pretrained(out_dir: str | os.PathLike, dynamics_state: Mapping[str, torch.Tensor],
                             model_config: Mapping[str, Any], *, tokenizer_dir: str | os.PathLike | None = None,
                             verify: bool = True) -> Path:
    """Write a dynamics pretrained dir.

    With ``tokenizer_dir`` (a tokenizer pretrained dir) the latent shapes are checked and the
    tokenizer's sha256 is recorded under ``config["tokenizer"]["sha256"]``; a different sha256
    already recorded in ``model_config`` raises. A dynamics model trained from a tokenizer
    ``.pt`` records only its path, so only the latent shapes can be checked: a warning says so.
    """
    cfg = as_model_config(model_config, DYNAMICS_MODEL_TYPE)
    if tokenizer_dir is not None:
        tok_cfg = load_pretrained_config(tokenizer_dir)
        check_pairing(tok_cfg, cfg)
        sha = tokenizer_sha256(tokenizer_dir)
        tok_block = cfg.get("tokenizer") or {}
        recorded = tok_block.get("sha256")
        if recorded is not None and recorded != sha:
            raise ValueError(f"{tokenizer_dir} (sha256 {sha[:12]}...) is not the tokenizer recorded in the "
                             f"dynamics config (sha256 {recorded[:12]}...)")
        if recorded is None and tok_block:
            log.warning("the dynamics model was trained with the tokenizer %s, whose sha256 was not recorded: its "
                        "pairing with %s is checked by latent shape only; make sure these are the same weights",
                        tok_block.get("path"), tokenizer_dir)
        cfg["tokenizer"] = {**tok_block, "sha256": sha}
    dyn_sd = strip_compile_prefix(dynamics_state)
    if verify:
        with torch.device("meta"):
            dyn = build_dynamics(cfg["architecture"])
        _verify({"dynamics": (dyn, dyn_sd)})
    out = save_pretrained(out_dir, dyn_sd, cfg)
    log.info("wrote dynamics pretrained dir %s (%d tensors)", out, len(dyn_sd))
    return out


def checkpoint_kind(ckpt: Mapping) -> str:
    """``TOKENIZER_MODEL_TYPE`` or ``DYNAMICS_MODEL_TYPE`` from the state dicts a checkpoint holds."""
    has_tok = all(k in ckpt for k in TOKENIZER_STATE_KEYS)
    has_dyn = DYNAMICS_STATE_KEY in ckpt
    if has_tok == has_dyn:
        raise KeyError(f"cannot tell whether this is a tokenizer ('enc'+'dec') or dynamics ('dyn') checkpoint; "
                       f"top-level keys: {sorted(ckpt)}")
    return TOKENIZER_MODEL_TYPE if has_tok else DYNAMICS_MODEL_TYPE


def export_checkpoint(ckpt_path: str | os.PathLike, out_dir: str | os.PathLike, *, config: Mapping | None = None,
                      tokenizer_dir: str | os.PathLike | None = None, metadata: Mapping[str, Any] | None = None,
                      verify: bool = True) -> Path:
    """Convert a training checkpoint into a pretrained dir.

    ``metadata`` blocks are merged into ``config.json``: a mapping is merged key by key into
    the block of the same name, anything else replaces it (``model_type``, ``format_version``
    and ``architecture`` cannot be set). The checkpoint's ``global_update`` is recorded under
    ``provenance`` unless the config already states one.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = load_checkpoint(ckpt_path, mmap=True)
    kind = checkpoint_kind(ckpt)
    cfg = checkpoint_model_config(ckpt, config, kind, ckpt_path)
    for key, value in (metadata or {}).items():
        if key in ("model_type", "format_version", "architecture"):
            raise KeyError(f"metadata may not set {key!r}")
        old = cfg.get(key)
        both = isinstance(old, Mapping) and isinstance(value, Mapping)
        cfg[key] = {**old, **copy.deepcopy(value)} if both else copy.deepcopy(value)
    if "global_update" in ckpt:
        provenance = dict(cfg.get("provenance") or {})
        provenance.setdefault("global_update", int(ckpt["global_update"]))
        cfg["provenance"] = provenance
    if kind == TOKENIZER_MODEL_TYPE:
        if tokenizer_dir is not None:
            raise ValueError("--tokenizer only applies to a dynamics checkpoint")
        return save_tokenizer_pretrained(out_dir, ckpt["enc"], ckpt["dec"], cfg, verify=verify)
    return save_dynamics_pretrained(out_dir, ckpt[DYNAMICS_STATE_KEY], cfg, tokenizer_dir=tokenizer_dir,
                                    verify=verify)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a world-model training checkpoint as a pretrained dir.")
    parser.add_argument("--ckpt", required=True, help="training checkpoint (.pt)")
    parser.add_argument("--out", required=True, help="output pretrained dir")
    parser.add_argument("--config", help="YAML/JSON model config or architecture, for checkpoints without one")
    parser.add_argument("--metadata", help="YAML/JSON mapping of config.json blocks to add or override "
                                           "(e.g. action, inference), merged key by key within a block")
    parser.add_argument("--tokenizer", help="tokenizer pretrained dir to pair a dynamics model with")
    parser.add_argument("--no-verify", action="store_true", help="skip the strict-load and finiteness checks")
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config) if args.config else None
    metadata = load_config(args.metadata) if args.metadata else None
    export_checkpoint(args.ckpt, args.out, config=config, tokenizer_dir=args.tokenizer, metadata=metadata,
                      verify=not args.no_verify)


if __name__ == "__main__":
    main()
