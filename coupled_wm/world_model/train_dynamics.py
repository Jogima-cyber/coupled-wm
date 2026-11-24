# SPDX-License-Identifier: Apache-2.0
"""Train the DreamerV4 dynamics model on the latents of a frozen tokenizer (DDP).

    torchrun --nproc_per_node=4 -m coupled_wm.world_model.train_dynamics \\
        --config configs/world_model/g1_dynamics_backbone.yaml [--set key.path=value ...]

Also runs under ``srun`` (one task per GPU) or as a single process. Re-running the same
command resumes from ``<run_dir>/latest.pt``; ``run_dir`` is relative to the working directory
unless absolute, and the first log lines name the absolute path and whether the run resumes.

Stages (``stage`` in the config)
--------------------------------
``backbone``
    The dynamics transformer without reward head, from scratch (or warm-started with
    ``init_from``). Every update is made of ``train.short_len``-frame clips, except that every
    ``train.long_every``-th update (updates ``long_every``, ``2 * long_every``, ... counted from
    0; the logs count completed updates, so the first one is logged as update
    ``long_every + 1``) uses ``train.long_len``-frame clips for all of its micro-batches; the
    short-clip loader still advances on those updates.
``reward``
    Starts from a backbone (``init_from``: a pretrained dir or a training ``latest.pt`` without
    reward head) plus a freshly initialised agent token and multi-token reward head, with a
    fresh AdamW and a fresh schedule. The data must hold a per-frame reward
    (``data.reward_key``). The backbone must have reached its own ``schedule.max_updates``
    (``init_from_allow_unfinished: true`` waives this) and must have been trained with the same
    tokenizer (sha256 of the pretrained dir, when both sides record one).

Objective (DreamerV4 shortcut forcing, per micro-batch of clips)
----------------------------------------------------------------
* ``z = encoder(frames, proprio)`` from the frozen tokenizer (bf16, eval mode).
* Diffusion forcing: every frame gets its own noise level ``tau`` and step size
  ``d = 2**-s``; ``x = (1 - tau) * eps + tau * z`` (``tau = 1`` is clean). The model predicts the
  clean latents from ``[action[t], (tau, s), registers, x_t]`` and the causal history.
* A micro-batch is either *flow* (finest step ``s = log2(num_tau_levels)``, ``tau`` uniform on
  the fine grid, target ``z``) or *shortcut* (coarser ``s``; target = two half steps of the model
  itself at ``s + 1``, with a stop-gradient on the target; the loss backpropagates through the
  coarse-step prediction). Loss weight ``0.9 * tau + 0.1``.
* A fraction ``f = train.shortcut_fraction`` of the micro-batches is shortcut. The micro-batches
  are numbered over updates, ranks and micro-steps, ``j = (k * world_size + rank) * G + m`` for
  micro-step ``m`` of ``G`` accumulation steps, with ``k`` counting the updates of the same clip
  length; micro-batch ``j`` is shortcut iff ``floor((j + 1) f) > floor(j f)``. For ``f = 1/4``
  this is every 4th micro-batch: F F F S on every rank at ``G = 4`` (the paper's layout, for any
  world size), every 4th micro-step of each rank at larger ``G`` (F F F S F F F S at ``G = 8``),
  every 4th rank at ``G = 1``. An update of ``N = world_size * G`` micro-batches (``global_batch
  / micro_batch_per_gpu``) has exactly ``N f`` shortcut ones when that is an integer, otherwise
  ``floor`` or ``ceil`` of it; for ``f = 1/4`` it has at least one flow micro-batch whenever
  ``N > 1``. Counting ``k`` per clip length gives the long and the short clips the fraction
  ``f`` each.
* Each loss term is divided by its running RMS: an EMA of its rank-averaged square, updated every
  micro-step with decay ``loss.rms_decay ** (micro_batch_per_gpu * world_size /
  loss.rms_reference_windows)``. ``rms_decay`` is thus the decay over ``rms_reference_windows``
  clips (64: the paper's 64 GPUs x 1 clip, where it is the per-micro-step decay itself), and the
  normaliser averages over the same number of clips at any world size. Flow and shortcut
  micro-batches share one normaliser; every rank updates each normaliser once per micro-step,
  so its all-reduce stays aligned. With ``micro_batch_per_gpu > 1`` the EMA input is the square
  of a micro-batch mean (the paper used 1 clip per GPU). On CUDA the terms are fp32 (autocast
  runs ``pow`` and ``log_softmax`` in fp32), so the EMA state is fp32.
* Reward stage: head ``l`` of the reward head at frame ``t`` predicts ``reward[t + l]``
  (symlog two-hot cross-entropy); targets past the end of the clip are masked.

Action / frame alignment: ``action[t]`` (the data's ``commands[t]``, the command recorded at
frame ``t``) is consumed together with frame ``t``, with no shift. The world model therefore
learns whatever command-to-image latency the data has (about 2 frames for the G1 data); the G1
environment feeds the policy's command at the position of the frame it generates.

Precision: the encoder runs in bf16 outside autocast; the dynamics keeps fp32 master weights and
runs under bf16 autocast; actions and proprioception are cast to bf16.

Encoder context: the frozen encoder sees each clip whole, with temporal window
``tokenizer.context_len`` (null = the window the tokenizer was trained with, as in the G1
environment; 0 = none: the paper's backbone fed its 128-frame clips this way to a tokenizer
trained on 64-frame clips with no window). A value that differs from the tokenizer's is logged
as a warning. ``use_proprio`` / ``proprio_dim`` and the image size come from the tokenizer's
config and are checked against the data; frames stored at another size are resized to the
tokenizer's (bilinear, as the RL environment). ``data.color_order`` null takes
the tokenizer's recorded ``preprocessing.h5_color_order`` (else ``bgr``).

Global batch: ``train.global_batch`` clips per optimizer update = ``micro_batch_per_gpu`` x
world size x gradient-accumulation steps (which must be an integer). The schedule, the
shortcut fraction of an update and the loss normaliser's horizon do not depend on the world
size.

Budgets: ``schedule.warmup_updates`` / ``total_updates`` define the linear-warmup + cosine
learning-rate schedule; ``schedule.max_updates`` stops the stage. Updates are counted within the
stage; ``global_update`` adds the updates of the ``init_from`` checkpoint.

Checkpoints: ``<run_dir>/latest.pt`` (atomic) every ``checkpoint.save_every_updates`` and every
``checkpoint.save_every_minutes`` of wall time, on SIGTERM / SIGUSR1 (at the end of the current
update) and at the end. It holds the weights, AdamW, schedule, loss-scaler state, data position,
the file pool and per-rank RNG states, so a resumed run continues the uninterrupted one. Every
rank must see the same ``latest.pt`` (checked). A resume refuses a changed file pool
(``data.allow_pool_change: true`` continues on the new pool from the start of the current pass)
and an ``init_from`` source that has moved on since this run started from it. At the end,
``<run_dir>/pretrained/`` holds the weights + ``config.json``
(:mod:`coupled_wm.world_model.export` format); its ``training.lineage_max_clip_len`` is the
longest clip this stage and its ``init_from`` source trained on, which the G1 environment
compares with its context and KV-cache lengths.

Stopping: a run stopped by a signal before ``schedule.max_updates`` exits with code
:data:`EXIT_STOPPED` (75), so ``backbone && reward`` chains do not go on to the next stage.
torchrun forwards SIGTERM (not SIGUSR1) to the training processes and kills them 30 s later;
the current update and the save must fit in that window. SIGUSR1 sent to torchrun kills the
agent only: send it to the training processes (or launch them with ``srun``). DataLoader
workers ignore both signals.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import socket
import sys
import time
from fractions import Fraction
from itertools import islice
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from coupled_wm.common.checkpoint import (
    atomic_save,
    is_pretrained_dir,
    load_checkpoint,
    load_pretrained,
    load_pretrained_config,
    strip_compile_prefix,
)
from coupled_wm.common.config import load_config, save_config
from coupled_wm.common.distributed import DistInfo, barrier, cleanup, init_distributed
from coupled_wm.common.logging import ExperimentLogger, setup_logging
from coupled_wm.common.system import SignalSafeDataLoader, StopSignals, warn_if_compile_unavailable
from coupled_wm.data.h5_dataset import HDF5SequenceDataset, list_h5_files
from coupled_wm.world_model.export import save_dynamics_pretrained
from coupled_wm.world_model.loading import (
    MODEL_CONFIG_KEY,
    as_model_config,
    dynamics_model_config,
    lineage_max_clip_len,
    load_state_dict_strict,
    load_tokenizer,
    tokenizer_sha256,
)
from coupled_wm.world_model.models import (
    DYNAMICS_MODEL_TYPE,
    DYNAMICS_RUNTIME_KEYS,
    G1_ACTION,
    G1_CONTROL_HZ,
    G1_INFERENCE,
    DreamerV4Dynamics,
    build_dynamics,
    dynamics_arch,
)

log = logging.getLogger(__name__)

STAGES = ("backbone", "reward")
CHECKPOINT_FORMAT = "coupled_wm.dynamics_training"
CHECKPOINT_VERSION = 1
REWARD_PARAM_PREFIXES = ("agent_token", "reward_head.")
REWARD_ARCH_KEYS = frozenset({"use_reward_head", "mtp_length", "reward_hidden_dim", "reward_num_buckets",
                              "reward_symlog_range"})
EXPORT_PRESETS = {"g1": {"inference": G1_INFERENCE, "action": G1_ACTION, "control_hz": G1_CONTROL_HZ}}
ACTION_LAYOUT_KEYS = ("dim", "scale", "grip_dims", "grip_range")  # what the environment reads from ``action``
EXIT_STOPPED = 75
"""Exit code of a run stopped by a signal before ``schedule.max_updates`` (``EX_TEMPFAIL``:
re-run the same command to resume)."""

_REQUIRED = object()

DEFAULTS: dict[str, Any] = {
    "stage": _REQUIRED,
    "run_dir": _REQUIRED,
    "seed": 42,
    "init_from": None,
    "init_from_allow_unfinished": False,
    "tokenizer": {"path": _REQUIRED, "context_len": None, "compile": None},
    "data": {
        "dirs": None, "include_globs": ["*.h5"], "exclude_globs": [], "min_chunk_index": None,
        "file_list": None, "color_order": None, "reward_key": None, "reward_remap": None,
        "reward_override_dir": None, "num_workers": 4, "prefetch_factor": 2, "sampler_seed": 42,
        "start_epoch": 0, "allow_pool_change": False, "max_open_files": 256,
    },
    "model": {},
    "train": {
        "short_len": 32, "long_len": None, "long_every": 0, "global_batch": 256, "micro_batch_per_gpu": 1,
        "shortcut_fraction": 0.25,
    },
    "optim": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.1, "grad_clip": 1.0},
    "schedule": {"warmup_updates": _REQUIRED, "total_updates": _REQUIRED, "min_lr": 1e-8, "max_updates": _REQUIRED},
    "loss": {"rms_decay": 0.99, "rms_reference_windows": 64, "rms_eps": 1e-8},
    "compile": None,
    "checkpoint": {"save_every_updates": 1000, "save_every_minutes": 30, "archive_every_updates": None,
                   "export_on_save": False},
    "export": {"preset": None, "metadata": {}},
    "logging": {"print_every_updates": 100},
    "wandb": {"enabled": False, "mode": "offline", "project": "coupled-wm", "entity": None, "name": None,
              "group": None, "tags": [], "id": None},
    "dist": {"backend": "nccl", "timeout_minutes": 30},
}
# Blocks whose keys are not checked against DEFAULTS (validated elsewhere or free-form).
_FREE_FORM = {"model", "data.reward_remap", "export.metadata"}


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

def _merge_defaults(cfg: Any, defaults: Mapping[str, Any], path: str = "") -> dict:
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, Mapping):
        raise TypeError(f"config block {path or '<root>'!r} must be a mapping, got {cfg!r}")
    unknown = sorted(set(cfg) - set(defaults))
    if unknown:
        raise KeyError(f"unknown config key(s) {[f'{path}{k}' for k in unknown]}")
    out = {}
    for key, default in defaults.items():
        full = f"{path}{key}"
        value = cfg.get(key, default)
        if value is _REQUIRED:
            raise KeyError(f"config key {full!r} is required")
        if isinstance(default, dict):
            if full not in _FREE_FORM:
                value = _merge_defaults(value, default, f"{full}.")
            elif value is None:
                value = {}
            elif not isinstance(value, Mapping):
                raise TypeError(f"config block {full!r} must be a mapping, got {value!r}")
        out[key] = copy.deepcopy(value)
    return out


def resolve_config(raw: Mapping[str, Any]) -> dict:
    """Fill defaults, reject unknown keys and check the values of a dynamics-training config."""
    cfg = _merge_defaults(raw, DEFAULTS)
    if cfg["stage"] not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {cfg['stage']!r}")
    reward_stage = cfg["stage"] == "reward"

    data = cfg["data"]
    if data["dirs"] is None and data["file_list"] is None:
        raise KeyError("set data.dirs (or data.file_list)")
    if reward_stage and data["reward_key"] is None:
        raise ValueError("the reward stage trains the reward head: set data.reward_key (e.g. 'rewards')")
    if data["color_order"] not in (None, "bgr", "rgb"):
        raise ValueError(f"data.color_order must be null (the tokenizer's), 'bgr' or 'rgb', got {data['color_order']!r}")
    if not isinstance(data["start_epoch"], int) or data["start_epoch"] < 0:
        raise ValueError(f"data.start_epoch must be an integer >= 0, got {data['start_epoch']!r}")
    ctx = cfg["tokenizer"]["context_len"]
    if ctx is not None and (not isinstance(ctx, int) or ctx < 0):
        raise ValueError(f"tokenizer.context_len must be null (the tokenizer's own window) or an integer >= 0, "
                         f"got {ctx!r}")

    model = cfg["model"]
    if "use_reward_head" in model and bool(model["use_reward_head"]) != reward_stage:
        raise ValueError(f"model.use_reward_head is set by the stage ({cfg['stage']}: {reward_stage}); remove it")

    t = cfg["train"]
    for key in ("short_len", "global_batch", "micro_batch_per_gpu"):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer, got {t[key]!r}")
    if not isinstance(t["long_every"], int) or t["long_every"] < 0:
        raise ValueError(f"train.long_every must be an integer >= 0, got {t['long_every']!r}")
    if t["long_every"] and (not isinstance(t["long_len"], int) or t["long_len"] < 1):
        raise ValueError("train.long_every > 0 needs train.long_len")
    fraction = Fraction(str(t["shortcut_fraction"]))
    if not 0 <= fraction < 1:
        raise ValueError(f"train.shortcut_fraction must be in [0, 1), got {t['shortcut_fraction']}")
    _check_rms_config(cfg["loss"])

    s = cfg["schedule"]
    for key in ("warmup_updates", "total_updates", "max_updates"):
        if not isinstance(s[key], int) or s[key] < 0:
            raise ValueError(f"schedule.{key} must be a non-negative integer, got {s[key]!r}")
    if not s["warmup_updates"] < s["total_updates"]:
        raise ValueError("schedule.warmup_updates must be smaller than schedule.total_updates")
    if not 1 <= s["max_updates"] <= s["total_updates"]:
        raise ValueError("schedule.max_updates must be in [1, schedule.total_updates] (the cosine rises again "
                         "past total_updates)")

    if cfg["export"]["preset"] not in (None, *EXPORT_PRESETS):
        raise ValueError(f"export.preset must be null or one of {sorted(EXPORT_PRESETS)}")
    reserved = sorted(set(cfg["export"]["metadata"]) & {"model_type", "format_version", "architecture",
                                                          "tokenizer", "training"})
    if reserved:
        raise KeyError(f"export.metadata may not set {reserved} (written by the trainer)")
    if data["reward_remap"] is not None and not isinstance(data["reward_remap"], Mapping):
        raise TypeError("data.reward_remap must be a mapping {value: new_value} or null")
    if not cfg["checkpoint"]["save_every_updates"] >= 1:
        raise ValueError("checkpoint.save_every_updates must be >= 1")
    minutes = cfg["checkpoint"]["save_every_minutes"]
    if minutes is not None and (isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or minutes <= 0):
        raise ValueError(f"checkpoint.save_every_minutes must be null or a positive number, got {minutes!r}")
    if not isinstance(cfg["logging"]["print_every_updates"], int) or cfg["logging"]["print_every_updates"] < 0:
        raise ValueError("logging.print_every_updates must be an integer >= 0 (0: only the first update)")
    _check_metadata(cfg, model_arch(cfg))
    return cfg


def load_run_config(path: str, overrides: Sequence[str] = ()) -> dict:
    """Resolved config of a YAML file plus ``--set`` overrides (which may name any key of
    :data:`DEFAULTS`, also those the YAML omits)."""
    return resolve_config(load_config(path, overrides, DEFAULTS))


def model_arch(cfg: Mapping[str, Any]) -> dict:
    """Complete dynamics architecture: the ``model`` block plus ``use_reward_head`` from the stage."""
    arch = dynamics_arch({**cfg["model"], "use_reward_head": cfg["stage"] == "reward"})
    t = cfg["train"]
    max_len = max(t["short_len"], t["long_len"] if t["long_every"] else 0)
    if arch["seq_len"] < max_len:
        raise ValueError(f"model.seq_len {arch['seq_len']} is shorter than the longest clip ({max_len} frames)")
    return arch


def accumulation_steps(global_batch: int, micro_batch: int, world_size: int) -> int:
    per_step = micro_batch * world_size
    if global_batch % per_step:
        raise ValueError(f"train.global_batch {global_batch} is not a multiple of micro_batch_per_gpu x world size "
                         f"= {micro_batch} x {world_size}")
    return global_batch // per_step


def _check_rms_config(loss: Mapping[str, Any]) -> None:
    decay, reference = loss["rms_decay"], loss["rms_reference_windows"]
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 <= decay <= 1:
        raise ValueError(f"loss.rms_decay must be in [0, 1], got {decay!r}")
    if isinstance(reference, bool) or not isinstance(reference, int) or reference < 1:
        raise ValueError(f"loss.rms_reference_windows must be a positive integer, got {reference!r}")


def rms_decay_per_micro_step(rms_decay: float, windows_per_micro_step: int, reference_windows: int) -> float:
    """EMA decay of the loss normaliser for one micro-step of ``windows_per_micro_step`` clips
    (``micro_batch_per_gpu * world_size``): ``rms_decay ** (windows_per_micro_step / reference_windows)``.

    ``rms_decay`` is the decay over ``reference_windows`` clips, so the normaliser forgets at the
    same rate per clip (and per update) at any world size; it is ``rms_decay`` itself when
    ``windows_per_micro_step == reference_windows`` (the paper: 64 GPUs x 1 clip, 0.99).
    """
    return rms_decay ** (windows_per_micro_step / reference_windows)


# --------------------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------------------

def _all_reduce_mean_(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        if dist.get_backend() == "nccl":
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
        else:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t.div_(dist.get_world_size())
    return t


class RMSLossScaler:
    """Divides each named loss by the running RMS of its value.

    The squared value is averaged over ranks and tracked with an EMA (``decay``); the first value
    initialises it. The state keeps the dtype of the losses (fp32 on CUDA, see module docstring).
    """

    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = decay
        self.eps = eps
        self.ema_sq: dict[str, torch.Tensor] = {}

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sq = _all_reduce_mean_(value.detach().pow(2))
            prev = self.ema_sq.get(name)
            self.ema_sq[name] = sq if prev is None else self.decay * prev + (1.0 - self.decay) * sq
            rms = (self.ema_sq[name] + self.eps).sqrt()
        return value / rms

    def rms(self, name: str) -> float | None:
        ema = self.ema_sq.get(name)
        return None if ema is None else float((ema.float() + self.eps).sqrt())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.ema_sq.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor], device: torch.device | str) -> None:
        self.ema_sq = {k: torch.as_tensor(v).to(device) for k, v in state.items()}


def cosine_with_warmup(optimizer: torch.optim.Optimizer, warmup_updates: int, total_updates: int,
                       min_lr: float) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup from 0, then cosine decay to 0 at ``total_updates``, floored at ``min_lr``."""
    base_lr = optimizer.defaults["lr"]

    def lr_lambda(step: int) -> float:
        if step < warmup_updates:
            return float(step) / float(max(1, warmup_updates))
        progress = float(step - warmup_updates) / float(max(1, total_updates - warmup_updates))
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def shortcut_layout(update: int, rank: int, world_size: int, accum: int, fraction: Fraction) -> list[bool]:
    """Shortcut flag of each of the ``accum`` micro-steps of ``rank`` in update ``update``.

    The micro-batches of all updates and ranks form one sequence, ``j = (update * world_size +
    rank) * accum + m`` for micro-step ``m``; micro-batch ``j`` is shortcut iff
    ``floor((j + 1) * fraction) > floor(j * fraction)``. With ``fraction = 1/4`` that is every
    4th micro-batch: F F F S on every rank at ``accum = 4``, every 4th micro-step of each rank at
    larger ``accum``, every 4th rank at ``accum = 1``. An update has ``fraction * world_size *
    accum`` shortcut micro-batches over all ranks, rounded down or up when that is not an integer.
    """
    f = Fraction(fraction)
    first = (update * world_size + rank) * accum
    return [math.floor((j + 1) * f) > math.floor(j * f) for j in range(first, first + accum)]


def update_plan(stage_update: int, long_updates: int, long_every: int, rank: int, world_size: int, accum: int,
                fraction: Fraction) -> tuple[bool, list[bool]]:
    """``(long clips?, shortcut flag of each micro-step of rank)`` in stage update ``stage_update``.

    ``long_updates`` is the number of long-clip updates before it. :func:`shortcut_layout` is
    indexed by the updates of the same clip length, so each length gets ``fraction`` of its
    micro-batches as shortcut also when an update holds fewer than ``1 / fraction`` micro-batches
    (indexing by ``stage_update`` would then put every long update on shortcut-free slots).
    """
    long_update = bool(long_every) and stage_update > 0 and stage_update % long_every == 0
    index = long_updates if long_update else stage_update - long_updates
    return long_update, shortcut_layout(index, rank, world_size, accum, fraction)


def sample_noise_levels(B: int, T: int, device: torch.device, shortcut: bool, num_tau_levels: int):
    """Per-frame noise levels for one micro-batch.

    Returns ``(tau, d, tau_idx, step_idx)``: ``tau`` and step size ``d`` as fp32 ``(B, T)``,
    ``tau_idx`` the index of ``tau`` on the finest grid (``tau = 1 - (tau_idx + 1) / num_tau_levels``)
    and ``step_idx`` the step level ``s`` (``d = 2**-s``).
    """
    d_min_idx = int(math.log2(num_tau_levels))
    if not shortcut:
        step_idx = torch.full((B, T), d_min_idx, dtype=torch.long, device=device)
        rows = torch.randint(0, num_tau_levels, (B, T), device=device)
    else:
        step_idx = torch.randint(0, d_min_idx, (B, T), device=device)
        u = torch.rand(B, T, device=device)
        rows = (u * (2.0 ** step_idx)).floor().long()
    d = 0.5 ** step_idx.float()
    tau = 1 - (rows.float() + 1.0) * d
    tau_idx = ((rows + 1) * (2 ** (d_min_idx - step_idx))).long() - 1
    return tau, d, tau_idx, step_idx


def add_noise(z_clean: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """``(1 - tau) * eps + tau * z_clean`` with per-frame ``tau (B, T)``."""
    B, T = tau.shape
    eps = torch.randn_like(z_clean)
    tau = tau.view(B, T, 1, 1)
    return (1 - tau) * eps + tau * z_clean


def reward_mtp_loss(logits: torch.Tensor, rewards: torch.Tensor, head: nn.Module) -> torch.Tensor:
    """Two-hot cross-entropy of the MTP reward logits ``(B, T, L, bins)`` against ``reward[t + l]``.

    ``rewards`` is ``(B, T)``; slots past the end of the clip are masked out of the mean.
    """
    L = logits.shape[2]
    pad = L - 1
    targets = F.pad(rewards, (0, pad), value=0.0).unfold(dimension=1, size=L, step=1)
    valid = F.pad(torch.ones_like(rewards, dtype=torch.bool), (0, pad), value=False).unfold(dimension=1, size=L, step=1)
    low, low_w, high, high_w = head.get_targets(targets)
    log_probs = F.log_softmax(logits, dim=-1)
    lp_low = log_probs.gather(-1, low.unsqueeze(-1)).squeeze(-1)
    lp_high = log_probs.gather(-1, high.unsqueeze(-1)).squeeze(-1)
    per_step = -(low_w * lp_low + high_w * lp_high) * valid.float()
    return per_step.sum() / valid.sum().clamp(min=1.0)


def dynamics_loss(student: nn.Module, teacher: nn.Module, actions: torch.Tensor, z_clean: torch.Tensor,
                  rewards: torch.Tensor | None, levels, shortcut: bool, scaler: RMSLossScaler,
                  num_tau_levels: int, reward_head: nn.Module | None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Normalised loss of one micro-batch (call under bf16 autocast).

    ``student`` runs with gradient (the DDP wrapper); ``teacher`` is the same model without the
    DDP wrapper, used under ``no_grad`` for the shortcut targets. ``levels`` comes from
    :func:`sample_noise_levels`. Returns ``(loss, raw terms)``.
    """
    tau, d, tau_idx, step_idx = levels
    B, T = z_clean.shape[:2]
    tau = tau.to(dtype=z_clean.dtype)
    d = d.to(dtype=z_clean.dtype)
    d_min_idx = int(math.log2(num_tau_levels))

    z_input = add_noise(z_clean, tau)
    pred_z, pred_rewards = student(actions, z_input, tau_idx, step_idx)
    weights = (0.9 * tau + 0.1).view(B, T, 1, 1)
    raw: dict[str, torch.Tensor] = {}

    loss_reward = None
    if reward_head is not None:
        raw["reward"] = reward_mtp_loss(pred_rewards, rewards, reward_head)
        loss_reward = scaler("loss_reward", raw["reward"])

    if not shortcut:
        raw["flow"] = (weights * ((pred_z - z_clean) ** 2)).mean()
        loss = scaler("loss_dyn", raw["flow"])
    else:
        with torch.no_grad():
            tau_b = tau.view(B, T, 1, 1)
            d_half = d / 2.0
            step_half = step_idx + 1
            pred_1, _ = teacher(actions, z_input, tau_idx, step_half)
            b_1 = (pred_1 - z_input) / (1.0 - tau_b + 1e-6)
            z_mid = z_input + b_1 * d_half.view(B, T, 1, 1)
            tau_mid = tau + d_half
            tau_idx_mid = tau_idx - (2 ** (d_min_idx - step_half)).long()
            pred_2, _ = teacher(actions, z_mid, tau_idx_mid, step_half)
            b_2 = (pred_2 - z_mid) / (1.0 - tau_mid.view(B, T, 1, 1) + 1e-6)
            target = (1.0 - tau_b) * ((b_1 + b_2) / 2.0)
        raw["shortcut"] = (weights * (((pred_z - z_input) - target) ** 2)).mean()
        loss = scaler("loss_dyn", raw["shortcut"])

    if loss_reward is not None:
        loss = loss + loss_reward
    return loss, {k: v.detach() for k, v in raw.items()}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

class ResumableDistributedSampler(DistributedSampler):
    """``DistributedSampler`` whose epoch can start after its first ``start`` samples."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = 0

    def __iter__(self):
        return islice(super().__iter__(), self.start, None)

    def __len__(self) -> int:
        return self.num_samples - self.start


class BatchStream:
    """Endless micro-batches from a sampler-driven loader, reshuffled every pass, resumable.

    A pass is truncated to ``pass_len`` micro-batches (default: all of them). ``epoch`` and
    ``position`` (micro-batches drawn in the current pass) are the resume state.
    """

    def __init__(self, loader: DataLoader, sampler: ResumableDistributedSampler, pass_len: int | None = None,
                 epoch: int = 0, position: int = 0):
        self.loader = loader
        self.sampler = sampler
        full = sampler.num_samples // loader.batch_size
        self.pass_len = full if pass_len is None else pass_len
        if not 1 <= self.pass_len <= full:
            raise ValueError(f"a pass needs 1..{full} micro-batches, got {self.pass_len} (too little data for the "
                             "world size and batch?)")
        self.epoch = epoch
        self.position = position
        self._it = None

    def next(self) -> dict[str, torch.Tensor]:
        if self.position >= self.pass_len:
            self.epoch += 1
            self.position = 0
            self._it = None
        if self._it is None:
            self.sampler.set_epoch(self.epoch)
            self.sampler.start = self.position * self.loader.batch_size
            self._it = iter(self.loader)
        batch = next(self._it)
        self.position += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "position": self.position}


def _dataset_files(data: Mapping[str, Any]) -> list[str]:
    return list_h5_files(data["dirs"], include_globs=data["include_globs"], exclude_globs=data["exclude_globs"],
                         min_chunk_index=data["min_chunk_index"], file_list=data["file_list"])


def _make_loader(dataset: HDF5SequenceDataset, cfg: Mapping[str, Any], info: DistInfo):
    data, t = cfg["data"], cfg["train"]
    sampler = ResumableDistributedSampler(dataset, num_replicas=info.world_size, rank=info.rank, shuffle=True,
                                          seed=data["sampler_seed"], drop_last=True)
    workers = data["num_workers"]
    loader = SignalSafeDataLoader(
        dataset, batch_size=t["micro_batch_per_gpu"], sampler=sampler, num_workers=workers,
        pin_memory=info.device.type == "cuda", drop_last=True, persistent_workers=workers > 0,
        prefetch_factor=data["prefetch_factor"] if workers > 0 else None,
        generator=torch.Generator(),  # creating an iterator must not consume the global RNG
    )
    return loader, sampler


def data_pool(short_ds: HDF5SequenceDataset, long_ds: HDF5SequenceDataset | None) -> dict:
    """Fingerprint of the file pool a data position refers to: file basenames and frame counts
    in pool order, and the window counts of both streams (basenames, so a moved data directory
    still resumes)."""
    return {"files": [[os.path.basename(i.path), int(i.n_frames)] for i in short_ds.file_infos],
            "windows": {"short": len(short_ds), "long": None if long_ds is None else len(long_ds)}}


def _pool_change(saved: Mapping, current: Mapping) -> str:
    old = {name for name, _ in saved["files"]}
    new = {name for name, _ in current["files"]}
    added, removed = sorted(new - old), sorted(old - new)
    parts = [f"{len(saved['files'])} -> {len(current['files'])} files"]
    if added:
        parts.append(f"added {added[:5]}{' ...' if len(added) > 5 else ''}")
    if removed:
        parts.append(f"removed {removed[:5]}{' ...' if len(removed) > 5 else ''}")
    if not added and not removed:
        parts.append("same names, different order or frame counts")
    parts.append(f"windows {saved['windows']} -> {current['windows']}")
    return "; ".join(parts)


# --------------------------------------------------------------------------------------
# Models and checkpoints
# --------------------------------------------------------------------------------------

def _load_encoder(cfg: Mapping[str, Any], device: torch.device, max_len: int):
    """Frozen bf16 tokenizer encoder; returns ``(encoder, tokenizer model config, encoder context_len)``."""
    tok = cfg["tokenizer"]
    runtime = {"seq_len": max_len, "dropout": 0.0, "mae_max_mask_prob": 0.0}
    if tok["context_len"] is not None:
        runtime["context_len"] = tok["context_len"]
    encoder, decoder, tok_cfg = load_tokenizer(tok["path"], device="cpu", runtime=runtime)
    del decoder
    trained = tok_cfg["architecture"]["context_len"]
    context_len = trained if tok["context_len"] is None else tok["context_len"]
    if context_len != trained:
        log.warning("tokenizer.context_len=%d overrides the temporal window the tokenizer was trained with (%d); "
                    "the G1 environment encodes its context with the trained window, so the dynamics would train on "
                    "latents the environment does not produce (null keeps the trained window)", context_len, trained)
    encoder = encoder.to(device=device, dtype=torch.bfloat16)
    if tok["compile"]:
        encoder = torch.compile(encoder, mode=tok["compile"])
    return encoder, tok_cfg, context_len


def resolve_color_order(configured: str | None, tokenizer_config: Mapping[str, Any]) -> str:
    """``data.color_order``, with null meaning the tokenizer's recorded ``preprocessing.h5_color_order``
    (else ``bgr``). An explicit value that differs from the recorded one is allowed (the dynamics
    files may be stored differently from the tokenizer's) but logged as a warning."""
    recorded = (tokenizer_config.get("preprocessing") or {}).get("h5_color_order")
    if configured is None:
        return recorded or "bgr"
    if recorded is not None and configured != recorded:
        log.warning("data.color_order=%r but the tokenizer was trained on files stored as %r (its "
                    "preprocessing.h5_color_order): unless these files really are %r, the dynamics trains on "
                    "channel-swapped frames (data.color_order: null uses the tokenizer's)", configured, recorded,
                    configured)
    return configured


def _compare_arch(source: Mapping[str, Any], target: Mapping[str, Any], ignore: frozenset, what: str) -> list[str]:
    diff = [f"{k}: {source.get(k)!r} != {target.get(k)!r}" for k in sorted(set(source) | set(target))
            if k not in ignore and source.get(k) != target.get(k)]
    if diff:
        raise ValueError(f"{what} has a different architecture: " + "; ".join(diff))
    return [k for k in sorted(ignore & (set(source) | set(target))) if source.get(k) != target.get(k)]


def read_init_source(path: str | Path, weights: bool = True) -> tuple[dict | None, dict | None, dict]:
    """``(state dict, model config, progress)`` of an ``init_from`` source.

    ``progress`` holds ``global_update``, ``stage_update`` and ``max_updates`` (the source run's
    own budget; ``None`` when not recorded) and ``scaler`` (the loss-scaler state of a training
    checkpoint). ``weights=False`` skips the weights of a pretrained dir (``state`` is ``None``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"init_from: {path} does not exist")
    if is_pretrained_dir(path):
        state, src_cfg = load_pretrained(path) if weights else (None, load_pretrained_config(path))
        training = src_cfg.get("training") or {}
        meta = training or src_cfg.get("provenance") or {}
        progress = {"global_update": meta.get("global_update"), "stage_update": training.get("stage_update"),
                    "max_updates": (training.get("schedule") or {}).get("max_updates"), "scaler": None}
    elif path.is_file():
        ckpt = load_checkpoint(path, mmap=True)
        if "dyn" not in ckpt:
            raise KeyError(f"init_from: {path} is not a dynamics checkpoint (keys: {sorted(ckpt)})")
        state, src_cfg = ckpt["dyn"], ckpt.get(MODEL_CONFIG_KEY)
        progress = {"global_update": ckpt.get("global_update"),
                    "stage_update": (ckpt.get("counters") or {}).get("stage_update"),
                    "max_updates": ((ckpt.get("config") or {}).get("schedule") or {}).get("max_updates"),
                    "scaler": ckpt.get("loss_scaler", ckpt.get("scaler"))}
    else:
        raise FileNotFoundError(f"init_from: {path} is neither a pretrained dir nor a checkpoint file")
    return state, src_cfg, progress


def _check_source_finished(path: Path, progress: Mapping[str, Any], allow_unfinished: bool) -> None:
    done, budget = progress["stage_update"], progress["max_updates"]
    if done is None or budget is None:
        log.warning("init_from %s records no stage progress: cannot check that its stage reached its max_updates",
                    path)
    elif done < budget:
        msg = (f"init_from {path} stopped at stage update {done} of its schedule.max_updates {budget} (an interrupted "
               "run): re-run that stage to finish it")
        if not allow_unfinished:
            raise ValueError(msg + ", or set init_from_allow_unfinished: true to start from it anyway")
        log.warning("%s; starting from it anyway (init_from_allow_unfinished)", msg)


def _check_source_tokenizer(path: Path, src_cfg: Mapping[str, Any] | None, tok_info: Mapping[str, Any]) -> None:
    """The source must have been trained on the latents of the tokenizer this run uses."""
    src_tok = (src_cfg or {}).get("tokenizer") or {}
    src_sha, sha = src_tok.get("sha256"), tok_info.get("sha256")
    if src_sha and sha:
        if src_sha != sha:
            raise ValueError(f"init_from {path} was trained with the tokenizer {src_tok.get('path')} (sha256 "
                             f"{src_sha[:12]}...) but tokenizer.path is {tok_info['path']} (sha256 {sha[:12]}...): "
                             "set tokenizer.path to the tokenizer the source was trained with")
        return
    src_path = src_tok.get("path")
    if src_path is not None and os.path.normpath(src_path) != os.path.normpath(tok_info["path"]):
        log.warning("init_from %s was trained with the tokenizer %s and this run uses %s; their sha256 cannot be "
                    "compared (only tokenizer pretrained dirs record one): make sure these are the same weights", path,
                    src_path, tok_info["path"])
    else:
        log.warning("init_from %s: its pairing with the tokenizer %s cannot be checked (only tokenizer pretrained "
                    "dirs record a sha256)", path, tok_info["path"])


def init_from_checkpoint(model: DreamerV4Dynamics, path: str | Path, arch: Mapping[str, Any],
                         tok_info: Mapping[str, Any] | None = None,
                         allow_unfinished: bool = False) -> tuple[dict, dict | None]:
    """Load backbone weights (pretrained dir or training checkpoint) into ``model``.

    The source must not have a reward head; with ``model.use_reward_head`` the agent token and the
    reward head keep their fresh initialisation. It must have reached its own
    ``schedule.max_updates`` unless ``allow_unfinished``, and with ``tok_info`` (``path`` and,
    for a pretrained dir, ``sha256`` of this run's tokenizer) a different recorded tokenizer
    sha256 raises. Returns ``(provenance, loss-scaler state)``; the scaler state is that of a
    training checkpoint (``None`` for a pretrained dir).
    """
    path = Path(path)
    state, src_cfg, progress = read_init_source(path)
    state = strip_compile_prefix(state)
    if any(k.startswith(REWARD_PARAM_PREFIXES) for k in state):
        raise ValueError(f"init_from: {path} already has a reward head; start from a backbone checkpoint "
                         "(the reward stage trains a freshly initialised head)")
    if src_cfg is not None:
        src_arch = as_model_config(src_cfg, DYNAMICS_MODEL_TYPE)["architecture"]
        _compare_arch(src_arch, arch, DYNAMICS_RUNTIME_KEYS | REWARD_ARCH_KEYS, f"init_from {path}")
    else:
        log.warning("init_from %s stores no architecture; only parameter names and shapes are checked", path)
    _check_source_finished(path, progress, allow_unfinished)
    if tok_info is not None:
        _check_source_tokenizer(path, src_cfg, tok_info)
    allow = REWARD_PARAM_PREFIXES if model.use_reward_head else ()
    load_state_dict_strict(model, state, allow_missing=allow, name=f"init_from {path}")
    src_tok = (src_cfg or {}).get("tokenizer")
    provenance = {"path": str(path), "global_update": int(progress["global_update"] or 0),
                  "tokenizer": copy.deepcopy(src_tok) if src_tok else None,
                  "lineage_max_clip_len": lineage_max_clip_len(src_cfg or {})}
    scaler_state = progress["scaler"]
    log.info("initialised from %s (update %d)%s", path, provenance["global_update"],
             "; agent token and reward head are fresh" if model.use_reward_head else "")
    return provenance, scaler_state


def _rng_state(device: torch.device) -> dict[str, torch.Tensor]:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _gather_rng(device: torch.device) -> list[dict]:
    state = _rng_state(device)
    if not dist.is_initialized():
        return [state]
    states: list = [None] * dist.get_world_size()
    dist.all_gather_object(states, state)
    return states


def _restore_rng(states: Sequence[Mapping] | None, info: DistInfo, seed: int, global_update: int) -> None:
    if states is not None and len(states) == info.world_size:
        torch.set_rng_state(states[info.rank]["cpu"])
        if info.device.type == "cuda" and "cuda" in states[info.rank]:
            torch.cuda.set_rng_state(states[info.rank]["cuda"], info.device)
        return
    reseed = int(np.random.SeedSequence([seed, info.rank, global_update]).generate_state(1)[0])
    log.warning("checkpoint RNG states are for %s ranks, this run has %d: reseeding",
                None if states is None else len(states), info.world_size)
    torch.manual_seed(reseed)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

class _Run:
    """One stage of dynamics training on this rank (built by :func:`train`)."""

    def __init__(self, cfg: dict, info: DistInfo):
        self.cfg, self.info = cfg, info
        device, rank, world = info.device, info.rank, info.world_size
        t = cfg["train"]
        self.stage = cfg["stage"]
        self.run_dir = Path(cfg["run_dir"])
        self.latest = self.run_dir / "latest.pt"
        self.accum = accumulation_steps(t["global_batch"], t["micro_batch_per_gpu"], world)
        self.fraction = Fraction(str(t["shortcut_fraction"]))
        self.long_every = t["long_every"]
        max_len = max(t["short_len"], t["long_len"] if self.long_every else 0)
        self.arch = model_arch(cfg)

        torch.manual_seed(cfg["seed"] + rank)
        self.dyn = build_dynamics(self.arch)  # random init; checkpoint / backbone weights are loaded below

        if cfg["compile"] or cfg["tokenizer"]["compile"]:
            warn_if_compile_unavailable(f"compile={cfg['compile']} / tokenizer.compile={cfg['tokenizer']['compile']}",
                                        "--set compile=null --set tokenizer.compile=null", device)

        # Tokenizer encoder (frozen).
        self.encoder, tok_cfg, encoder_context_len = _load_encoder(cfg, device, max_len)
        tok_arch = tok_cfg["architecture"]
        if (self.arch["num_latents"], self.arch["latent_dim"]) != (tok_arch["n_latents"], tok_arch["bottleneck_dim"]):
            raise ValueError(f"model latents ({self.arch['num_latents']}, {self.arch['latent_dim']}) do not match the "
                             f"tokenizer ({tok_arch['n_latents']}, {tok_arch['bottleneck_dim']})")
        self.use_proprio = bool(tok_arch["use_proprio"])
        tok_info = {"path": str(cfg["tokenizer"]["path"]), "encoder_context_len": encoder_context_len}
        if is_pretrained_dir(cfg["tokenizer"]["path"]):
            sha = [tokenizer_sha256(cfg["tokenizer"]["path"]) if rank == 0 else None]
            if dist.is_initialized():
                dist.broadcast_object_list(sha, src=0)
            tok_info["sha256"] = sha[0]

        # Data.
        data = cfg["data"]
        self.files = _dataset_files(data)
        reward_stage = self.stage == "reward"
        self.color_order = resolve_color_order(data["color_order"], tok_cfg)
        tok_hw = list(tok_arch["image_size"])
        ds_kwargs = dict(stride=1, image_size=tok_hw, color_order=self.color_order,
                         load_proprio=self.use_proprio, reward_key=data["reward_key"] if reward_stage else None,
                         reward_remap=data["reward_remap"] if reward_stage else None,
                         reward_override_dir=data["reward_override_dir"] if reward_stage else None,
                         max_open_files=data["max_open_files"])
        self.short_ds = HDF5SequenceDataset(self.files, window=t["short_len"], **ds_kwargs)
        if self.short_ds.resized and rank == 0:
            log.info("resizing %s frames to the tokenizer's size %s (bilinear, as the RL environment)",
                     [list(s) for s in self.short_ds.native_image_sizes], tok_hw)
        if self.short_ds.action_dim != self.arch["action_dim"]:
            raise ValueError(f"the data has {self.short_ds.action_dim}-d commands but model.action_dim is "
                             f"{self.arch['action_dim']}")
        if self.use_proprio and self.short_ds.proprio_dim != tok_arch["proprio_dim"]:
            raise ValueError(f"the data has {self.short_ds.proprio_dim}-d proprio but the tokenizer expects "
                             f"{tok_arch['proprio_dim']}")
        short_loader, short_sampler = _make_loader(self.short_ds, cfg, info)
        # Epochs are truncated to whole updates.
        self.short_pass = (short_sampler.num_samples // t["micro_batch_per_gpu"] // self.accum) * self.accum
        if self.short_pass == 0:
            raise ValueError(f"{len(self.short_ds)} windows are too few for {world} ranks x {self.accum} "
                             "accumulation steps")
        self.long_ds = None
        if self.long_every:
            self.long_ds = HDF5SequenceDataset(self.files, window=t["long_len"], **ds_kwargs)
            long_loader, long_sampler = _make_loader(self.long_ds, cfg, info)

        # Weights, optimizer, schedule, loss scaler, counters.
        self.pool = data_pool(self.short_ds, self.long_ds)
        if rank == 0:
            _remove_partial_saves(self.latest)
        ckpt = load_checkpoint(self.latest, mmap=True) if self.latest.is_file() else None
        _check_ranks_agree(self.latest, None if ckpt is None else ckpt["counters"]["stage_update"])
        scaler_state = None
        self.init_info = None
        pool_changed = False
        if ckpt is not None:
            _check_resume(ckpt, cfg, self.arch, tok_info)
            pool_changed = _check_pool(ckpt, cfg, self.pool)
            if rank == 0:
                log.info("resuming %s at stage update %d%s", self.latest.resolve(), ckpt["counters"]["stage_update"],
                         " (init_from only initialises a new run)" if cfg["init_from"] else "")
            load_state_dict_strict(self.dyn, ckpt["dyn"], name=f"resume {self.latest}")
            self.init_info, scaler_state = ckpt["init"], ckpt["loss_scaler"]
        else:
            if self.stage == "reward" and not cfg["init_from"]:
                raise ValueError("the reward stage starts from a backbone: set init_from (pretrained dir or latest.pt)")
            if rank == 0:
                log.info("no %s: starting stage %s %s", self.latest.resolve(), self.stage,
                         f"from init_from {cfg['init_from']}" if cfg["init_from"] else "from scratch")
            if cfg["init_from"]:
                self.init_info, scaler_state = init_from_checkpoint(self.dyn, cfg["init_from"], self.arch, tok_info,
                                                                    cfg["init_from_allow_unfinished"])
        self.dyn.to(device)
        self.init_update = (self.init_info or {}).get("global_update", 0)

        o, s = cfg["optim"], cfg["schedule"]
        self.optimizer = torch.optim.AdamW(self.dyn.parameters(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                                           weight_decay=o["weight_decay"])
        self.scheduler = cosine_with_warmup(self.optimizer, s["warmup_updates"], s["total_updates"], s["min_lr"])
        loss = cfg["loss"]
        decay = rms_decay_per_micro_step(loss["rms_decay"], t["micro_batch_per_gpu"] * world,
                                         loss["rms_reference_windows"])
        self.scaler = RMSLossScaler(decay, loss["rms_eps"])
        if scaler_state:
            self.scaler.load_state_dict(scaler_state, device)
        self.counters = {"stage_update": 0, "long_updates": 0}
        short_state = {"epoch": data["start_epoch"], "position": 0}
        long_state = dict(short_state)
        wandb_id = None
        if ckpt is not None:
            self.optimizer.load_state_dict(ckpt["optim"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.counters = dict(ckpt["counters"])
            short_state, long_state = ckpt["data_state"]["short"], ckpt["data_state"]["long"]
            if pool_changed:  # the saved positions index another pool: restart the current passes
                short_state = {"epoch": short_state["epoch"], "position": 0}
                long_state = {"epoch": long_state["epoch"], "position": 0}
            _restore_rng(ckpt.get("rng"), info, cfg["seed"], self.init_update + self.counters["stage_update"])
            wandb_id = ckpt.get("wandb_run_id")
        del ckpt
        self.short = BatchStream(short_loader, short_sampler, self.short_pass, **short_state)
        self.long = BatchStream(long_loader, long_sampler, None, **long_state) if self.long_every else None
        self.long_state = long_state

        metadata = export_metadata(cfg)
        self.model_config = dynamics_model_config(
            self.arch, **metadata, tokenizer=tok_info,
            training=_training_metadata(cfg, self.short_ds, self.init_info, self.color_order),
        )
        if rank == 0 and reward_stage:
            _warn_action_layout(metadata)
        minutes = cfg["checkpoint"]["save_every_minutes"]
        self.save_every_s = None if minutes is None else 60.0 * minutes
        self.last_save = time.monotonic()

        self.compiled = torch.compile(self.dyn, mode=cfg["compile"]) if cfg["compile"] else self.dyn
        if dist.is_initialized():
            self.student = DDP(self.compiled, device_ids=[info.local_rank] if device.type == "cuda" else None,
                               find_unused_parameters=False)
        else:
            self.student = self.compiled
        self.reward_head = self.dyn.reward_head.heads[0] if self.dyn.use_reward_head else None

        wandb_cfg = dict(cfg["wandb"])
        wandb_cfg["name"] = wandb_cfg["name"] or self.run_dir.name
        wandb_cfg["id"] = wandb_cfg["id"] or wandb_id
        # Updates are logged at init_update + their 1-based stage index: a resumed launch drops the
        # events the previous one logged after its last checkpoint.
        purge_step = self.init_update + self.counters["stage_update"] + 1 if self.counters["stage_update"] else 0
        self.logger = ExperimentLogger(self.run_dir / "tb", rank, cfg, wandb_cfg, purge_step=purge_step)
        if rank == 0:
            save_config(cfg, self.run_dir / "config.yaml")
            self._log_setup()

    def _log_setup(self) -> None:
        cfg, t, s = self.cfg, self.cfg["train"], self.cfg["schedule"]
        n_params = sum(p.numel() for p in self.dyn.parameters())
        log.info("stage %s in %s | %d ranks x %d clips x %d accumulation steps = global batch %d | %s parameters",
                 self.stage, self.run_dir.resolve(), self.info.world_size, t["micro_batch_per_gpu"], self.accum,
                 t["global_batch"], f"{n_params:,}")
        log.info("data: %d files, %d windows of %d frames, %d micro-batches per epoch and rank", len(self.files),
                 len(self.short_ds), t["short_len"], self.short_pass)
        if self.long_ds is not None:
            log.info("every %dth update: %d windows of %d frames", self.long_every, len(self.long_ds), t["long_len"])
        log.info("schedule: lr %g, warmup %d, cosine to %d, stop at stage update %d", cfg["optim"]["lr"],
                 s["warmup_updates"], s["total_updates"], s["max_updates"])
        loss, world = cfg["loss"], self.info.world_size
        log.info("loss normaliser: EMA decay %.10g per micro-step of %d clips (loss.rms_decay %g per %d clips)",
                 self.scaler.decay, t["micro_batch_per_gpu"] * world, loss["rms_decay"], loss["rms_reference_windows"])
        n = world * self.accum
        layout = "".join("S" if flag else "F" for flag in shortcut_layout(0, 0, world, self.accum, self.fraction))
        log.info("shortcut fraction %s of the %d micro-batches of an update (all ranks); rank 0, first update: %s",
                 self.fraction, n, layout)
        per_update = n * self.fraction
        if self.fraction and n - math.ceil(per_update) < 1:
            log.warning("an update holds %d micro-batch(es) over all ranks: some updates have no flow micro-batch "
                        "(raise train.global_batch / train.micro_batch_per_gpu)", n)
        elif per_update.denominator != 1:
            log.info("%d micro-batches per update is not a multiple of 1 / shortcut_fraction: an update has %d or %d "
                     "shortcut micro-batches", n, math.floor(per_update), math.ceil(per_update))

    # ---------------------------------------------------------------- one update

    def _update(self) -> dict[str, float]:
        cfg, device, accum, world = self.cfg, self.info.device, self.accum, self.info.world_size
        u = self.counters["stage_update"]
        long_update, shortcut_steps = update_plan(u, self.counters["long_updates"], self.long_every, self.info.rank,
                                                  world, accum, self.fraction)
        num_tau_levels = self.arch["num_tau_levels"]
        lr = self.optimizer.param_groups[0]["lr"]
        t0 = time.perf_counter()

        self.optimizer.zero_grad(set_to_none=True)
        sums = {k: torch.zeros((), device=device) for k in ("flow", "shortcut", "reward", "total")}
        for micro in range(accum):
            batch = self.short.next()
            if long_update:  # the long clips replace this update's short clips
                batch = self.long.next()
            images = batch["image"].to(device, non_blocking=True).to(torch.bfloat16)
            actions = batch["action"].to(device, non_blocking=True).to(torch.bfloat16)
            proprio = batch["proprio"].to(device, non_blocking=True).to(torch.bfloat16) if self.use_proprio else None
            rewards = batch["reward"].to(device, non_blocking=True).float() if self.reward_head is not None else None
            with torch.no_grad():
                _, _, z_clean = self.encoder(images, proprio=proprio)
            z_clean = z_clean.clone()  # a CUDA-graph-compiled encoder reuses its output buffer
            B, T = z_clean.shape[:2]
            shortcut = shortcut_steps[micro]
            levels = sample_noise_levels(B, T, device, shortcut, num_tau_levels)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, raw = dynamics_loss(self.student, self.compiled, actions, z_clean, rewards, levels, shortcut,
                                          self.scaler, num_tau_levels, self.reward_head)
            (loss / accum).backward()
            sums["total"] += loss.detach().float()
            for k, v in raw.items():
                sums[k] += v.float()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.dyn.parameters(), max_norm=cfg["optim"]["grad_clip"])
        self.optimizer.step()
        self.scheduler.step()
        self.counters["stage_update"] += 1
        self.counters["long_updates"] += int(long_update)

        # Stop votes of all ranks and rank 0's wall-clock save decision ride on the stats reduction.
        save_due = (self.info.rank == 0 and self.save_every_s is not None
                    and time.monotonic() - self.last_save >= self.save_every_s)
        stats = torch.stack([sums["flow"], sums["shortcut"], sums["reward"], sums["total"],
                             torch.tensor(float(sum(shortcut_steps)), device=device),
                             torch.tensor(float(self.stop.poll() is not None), device=device),
                             torch.tensor(float(save_due), device=device)])
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        flow, short_loss, reward, total, n_shortcut, stop_votes, save_votes = stats.tolist()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        n_micro = accum * world  # micro-batches of this update over all ranks
        n_flow = n_micro - n_shortcut
        out = {"train/loss_total": total / n_micro, "train/lr": lr, "train/grad_norm": float(grad_norm),
               "train/clip_len": T, "train/shortcut_fraction": n_shortcut / n_micro, "perf/sec_per_update": dt,
               "perf/frames_per_sec": n_micro * cfg["train"]["micro_batch_per_gpu"] * T / dt,
               "stop": stop_votes, "save": save_votes}
        if n_flow:
            out["train/loss_flow"] = flow / n_flow
        if n_shortcut:
            out["train/loss_shortcut"] = short_loss / n_shortcut
        if self.reward_head is not None:
            out["train/loss_reward"] = reward / n_micro
        for name in ("loss_dyn", "loss_reward"):
            rms = self.scaler.rms(name)
            if rms is not None:
                out[f"train/rms_{name}"] = rms
        return out

    # ---------------------------------------------------------------- checkpoints

    def save(self, export: bool, archive: bool) -> None:
        u = self.counters["stage_update"]
        rng = _gather_rng(self.info.device)
        if self.info.rank == 0:
            self.model_config["training"].update(global_update=self.init_update + u, stage_update=u)
            atomic_save({
                "format": CHECKPOINT_FORMAT, "format_version": CHECKPOINT_VERSION, "stage": self.stage,
                MODEL_CONFIG_KEY: self.model_config, "dyn": self.dyn.state_dict(),
                "optim": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "loss_scaler": self.scaler.state_dict(), "global_update": self.init_update + u,
                "counters": dict(self.counters), "init": self.init_info,
                "data_state": {"short": self.short.state_dict(),
                               "long": self.long.state_dict() if self.long else self.long_state},
                "data_pool": self.pool, "rng": rng, "config": self.cfg,
                "wandb_run_id": self.logger.wandb_run.id if self.logger.wandb_run is not None else None,
            }, self.latest)
            log.info("saved %s (stage update %d)", self.latest, u)
            if archive:
                self.export(self.run_dir / "archive" / f"update_{u:07d}")
            if export:
                self.export(self.run_dir / "pretrained")
        barrier()
        self.last_save = time.monotonic()

    def export(self, out_dir: Path) -> None:
        u = self.counters["stage_update"]
        self.model_config["training"].update(global_update=self.init_update + u, stage_update=u)
        save_dynamics_pretrained(out_dir, self.dyn.state_dict(), self.model_config)

    # ---------------------------------------------------------------- loop

    def run(self) -> bool:
        """Train until ``schedule.max_updates`` (returns ``True``) or a stop signal (``False``)."""
        s, ck = self.cfg["schedule"], self.cfg["checkpoint"]
        print_every = self.cfg["logging"]["print_every_updates"]
        if self.counters["stage_update"] >= s["max_updates"]:
            log.info("stage update %d >= max_updates %d: nothing to train", self.counters["stage_update"],
                     s["max_updates"])
            if self.info.rank == 0 and not is_pretrained_dir(self.run_dir / "pretrained"):
                self.export(self.run_dir / "pretrained")
            barrier()
            return True
        self.dyn.train()
        with StopSignals() as self.stop:
            while self.counters["stage_update"] < s["max_updates"]:
                stats = self._update()
                u = self.counters["stage_update"]
                stop = stats.pop("stop") > 0
                save_due = stats.pop("save") > 0
                self.logger.scalars(stats, self.init_update + u)
                if self.info.rank == 0 and ((print_every and u % print_every == 0) or u == 1):
                    log.info("update %d/%d (global %d) | epoch %d | T=%d | loss %.4f | lr %.3g | grad norm %.3g | "
                             "%.2f s/update", u, s["max_updates"], self.init_update + u, self.short.epoch,
                             stats["train/clip_len"], stats["train/loss_total"], stats["train/lr"],
                             stats["train/grad_norm"], stats["perf/sec_per_update"])
                finished = u >= s["max_updates"]
                archive = bool(ck["archive_every_updates"]) and u % ck["archive_every_updates"] == 0
                if stop or finished or archive or save_due or u % ck["save_every_updates"] == 0:
                    self.save(export=finished or ck["export_on_save"], archive=archive)
                if stop and not finished:
                    log.warning("stop requested (signal %s): checkpointed at stage update %d of %d; re-run the same "
                                "command to resume (exit code %d)", self.stop.received, u, s["max_updates"],
                                EXIT_STOPPED)
                    return False
        return True

    def close(self) -> None:
        self.logger.close()


def _train(cfg: Mapping[str, Any], info: DistInfo | None = None) -> tuple[nn.Module, bool]:
    cfg = resolve_config(cfg)
    if info is None:
        info = init_distributed(backend=cfg["dist"]["backend"], timeout_minutes=cfg["dist"]["timeout_minutes"])
    run = _Run(cfg, info)
    try:
        finished = run.run()
    finally:
        run.close()
    return run.dyn, finished


def train(cfg: Mapping[str, Any], info: DistInfo | None = None) -> nn.Module:
    """Run (or resume) one stage; returns the trained dynamics module (unwrapped, on its device)."""
    return _train(cfg, info)[0]


def export_metadata(cfg: Mapping[str, Any]) -> dict:
    """Metadata blocks of the exported ``config.json``: ``export.preset`` overlaid with
    ``export.metadata`` block by block (``export.metadata.action.scale`` keeps the preset's
    other ``action`` keys)."""
    out = copy.deepcopy(EXPORT_PRESETS.get(cfg["export"]["preset"], {}))
    for key, value in copy.deepcopy(cfg["export"]["metadata"]).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _training_metadata(cfg: Mapping[str, Any], dataset: HDF5SequenceDataset, init_info: Mapping | None,
                       color_order: str) -> dict:
    t, data = cfg["train"], cfg["data"]
    return {
        "stage": cfg["stage"],
        "init_from": copy.deepcopy(init_info),
        "global_update": 0,
        "stage_update": 0,
        "clips": {"short_len": t["short_len"], "long_len": t["long_len"] if t["long_every"] else None,
                  "long_every": t["long_every"]},
        "lineage_max_clip_len": _lineage_clip_len(cfg, init_info),
        "global_batch": t["global_batch"],
        "shortcut_fraction": float(t["shortcut_fraction"]),
        "optimizer": {"name": "AdamW", **{k: cfg["optim"][k] for k in ("lr", "betas", "eps", "weight_decay",
                                                                          "grad_clip")}},
        "schedule": dict(cfg["schedule"]),
        "data": {"files": len(dataset.files), "frames": dataset.summary["frames"],
                 "include_globs": list(data["include_globs"]), "exclude_globs": list(data["exclude_globs"]),
                 "color_order": color_order, "image_resize": list(dataset.image_size) if dataset.resized else None, "start_epoch": data["start_epoch"],
                 "reward_key": dataset.reward_key,
                 "reward_remap": {str(k): v for k, v in (data["reward_remap"] or {}).items()} or None},
    }


def _lineage_clip_len(cfg: Mapping[str, Any], init_info: Mapping | None) -> int | None:
    """Longest clip this stage and the stages it was initialised from trained on (``None`` when the
    ``init_from`` source does not record its own)."""
    t = cfg["train"]
    own = max(t["short_len"], t["long_len"] if t["long_every"] else 0)
    if not init_info:
        return own
    source = init_info.get("lineage_max_clip_len")
    return None if source is None else max(own, int(source))


def _check_metadata(cfg: Mapping[str, Any], arch: Mapping[str, Any]) -> None:
    """The exported inference / action blocks must describe this model."""
    metadata = export_metadata(cfg)
    inference, action = metadata.get("inference") or {}, metadata.get("action") or {}
    d_min_idx = int(math.log2(arch["num_tau_levels"]))
    problems = []
    if inference.get("d_min_idx") not in (None, d_min_idx):
        problems.append(f"inference.d_min_idx {inference['d_min_idx']} != log2(model.num_tau_levels) = {d_min_idx}")
    if action.get("dim") not in (None, arch["action_dim"]):
        problems.append(f"action.dim {action['dim']} != model.action_dim {arch['action_dim']}")
    if problems:
        preset = cfg["export"]["preset"]
        hint = (f"; export.preset {preset!r} writes the {preset.upper()} constants and is for {preset.upper()} data "
                "only: for another layout set export.preset: null and give export.metadata.action (dim, scale, "
                "grip_dims, grip_range; scripts/compute_action_envelope.py) and export.metadata.inference"
                if preset else "")
        raise ValueError("export metadata does not match the model: " + "; ".join(problems) + hint)


def _warn_action_layout(metadata: Mapping[str, Any]) -> None:
    """Warn when the reward-stage export will not record a complete action layout: the G1
    environment then takes the missing keys from the G1 layout."""
    action = metadata.get("action") or {}
    needed = [k for k in ACTION_LAYOUT_KEYS if k != "grip_range" or action.get("grip_dims") != []]  # no grips: no range
    missing = [k for k in needed if k not in action]
    if missing:
        log.warning("the exported config.json will have %s: the G1 environment fills %s from the G1 layout. For other "
                    "data, give export.metadata.action (scripts/compute_action_envelope.py prints it).",
                    "no action block" if not action else f"no action {missing}", "it" if not action else "them")


def _check_resume(ckpt: Mapping, cfg: Mapping[str, Any], arch: Mapping[str, Any], tok_info: Mapping) -> None:
    where = f"{cfg['run_dir']}/latest.pt"
    if ckpt.get("format") != CHECKPOINT_FORMAT or ckpt.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError(f"{where} is not a dynamics training checkpoint of format {CHECKPOINT_VERSION}")
    if ckpt["stage"] != cfg["stage"]:
        raise ValueError(f"{where} is a {ckpt['stage']!r} run but this config is stage {cfg['stage']!r} "
                         "(use a new run_dir)")
    saved = as_model_config(ckpt[MODEL_CONFIG_KEY], DYNAMICS_MODEL_TYPE)
    changed = _compare_arch(saved["architecture"], arch, DYNAMICS_RUNTIME_KEYS, where)
    if changed:
        log.warning("resuming with changed runtime architecture keys %s", changed)
    saved_tok = saved.get("tokenizer") or {}
    recorded = saved_tok.get("sha256")
    if recorded and tok_info.get("sha256"):
        if recorded != tok_info["sha256"]:
            raise ValueError(f"{cfg['tokenizer']['path']} is not the tokenizer this run was trained with")
    elif saved_tok.get("path") and os.path.normpath(saved_tok["path"]) != os.path.normpath(tok_info["path"]):
        log.warning("%s was trained with the tokenizer %s and resumes with %s; without a sha256 on both sides (only "
                    "tokenizer pretrained dirs record one) the weights cannot be compared: make sure they are the "
                    "same", where, saved_tok["path"], tok_info["path"])
    _check_init_unchanged(ckpt, cfg)
    old = ckpt.get("config") or {}
    diff = [k for k in ("train", "optim", "schedule", "data", "loss") if old.get(k) != cfg.get(k)]
    if diff:
        log.warning("resuming with changed config blocks %s", diff)


def _check_init_unchanged(ckpt: Mapping, cfg: Mapping[str, Any]) -> None:
    """A run started from ``init_from`` must not resume after its source has moved on.

    The source run may have been resumed since (``latest.pt`` is overwritten in place, so the
    path alone says nothing): its current ``global_update`` must still be the one this run
    started from. ``init_from: null`` resumes without the check.
    """
    init, source = ckpt.get("init"), cfg["init_from"]
    if not init or not source:
        return
    where = f"{cfg['run_dir']}/latest.pt"
    if os.path.normpath(str(source)) != os.path.normpath(str(init["path"])):
        log.warning("%s was initialised from %s; init_from now names %s (only used by a new run)", where,
                    init["path"], source)
    if not Path(source).exists():
        return
    current = read_init_source(source, weights=False)[2]["global_update"]
    if current is not None and int(current) != int(init["global_update"]):
        raise ValueError(f"{where} was initialised from {init['path']} at update {init['global_update']}, but "
                         f"{source} is now at update {current}: use a new run_dir to start this stage from it, or set "
                         "init_from: null to resume this run as it is")


def _check_ranks_agree(latest: Path, stage_update: int | None) -> None:
    """Every rank must resume the same ``latest.pt`` (or none): ranks that resolve ``run_dir`` to
    different directories would otherwise train from different states."""
    if not dist.is_initialized():
        return
    views: list = [None] * dist.get_world_size()
    dist.all_gather_object(views, (socket.gethostname(), os.getcwd(), str(latest.resolve()), stage_update))
    if len({v[3] for v in views}) > 1:
        table = "; ".join(f"rank {r}: {host} cwd {cwd} -> {path} at stage update {u}"
                          for r, (host, cwd, path, u) in enumerate(views))
        raise RuntimeError(f"the ranks see different checkpoints ({table}): run_dir must name the same directory on "
                           "a filesystem every rank sees (use an absolute RUN_ROOT)")


def _check_pool(ckpt: Mapping, cfg: Mapping[str, Any], pool: Mapping) -> bool:
    """Compare the file pool with the one the saved data position refers to; ``True`` if it changed
    (allowed only with ``data.allow_pool_change``)."""
    saved = ckpt.get("data_pool")
    if saved is None or saved == pool:
        return False
    change = _pool_change(saved, pool)
    if not cfg["data"]["allow_pool_change"]:
        raise ValueError(f"the data pool changed since {cfg['run_dir']}/latest.pt was saved ({change}), so its data "
                         "position no longer applies; restore the files, or set data.allow_pool_change: true to "
                         "continue on the new pool from the start of the current pass")
    log.warning("the data pool changed (%s): continuing from the start of the current pass (data.allow_pool_change)",
                change)
    return True


def _remove_partial_saves(latest: Path) -> None:
    """Delete the temporary file(s) of :func:`atomic_save` left by a save that was killed."""
    for tmp in latest.parent.glob(f".{latest.name}*.tmp"):
        log.warning("removing %s, left by an interrupted save", tmp)
        tmp.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point; exits with :data:`EXIT_STOPPED` when stopped before ``max_updates``."""
    parser = argparse.ArgumentParser(description="Train the DreamerV4 dynamics model (stage backbone or reward).")
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/world_model/g1_dynamics_backbone.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value (repeatable), e.g. --set schedule.max_updates=100")
    args = parser.parse_args(argv)
    cfg = load_run_config(args.config, args.overrides)
    info = init_distributed(backend=cfg["dist"]["backend"], timeout_minutes=cfg["dist"]["timeout_minutes"])
    setup_logging(info.rank)
    try:
        _, finished = _train(cfg, info)
    finally:
        cleanup()
    if not finished:
        sys.exit(EXIT_STOPPED)


if __name__ == "__main__":
    main()
