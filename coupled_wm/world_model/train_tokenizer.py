# SPDX-License-Identifier: Apache-2.0
"""Train the DreamerV4 tokenizer (encoder + decoder) of the global world model.

    torchrun --nproc_per_node=4 -m coupled_wm.world_model.train_tokenizer \\
        --config configs/world_model/g1_tokenizer.yaml [--set train.global_batch=64 ...]

A plain ``python -m ...`` runs one process and ``srun`` job steps are detected too
(:func:`coupled_wm.common.distributed.init_distributed`).

Objective (the paper's). Each window of ``data.window`` frames is encoded (in train mode the
encoder drops patches MAE-style and masks each frame's proprio token with p = 0.5) and
decoded. The loss is ``mse_weight * MSE / rms_mse + lpips_weight * LPIPS / rms_lpips +
proprio_weight * MSE_proprio / rms_proprio``: MSE on [0, 1] pixels, LPIPS-VGG on frames scaled
to [-1, 1], MSE on the raw proprio vector, each divided by a running RMS. The running RMS is an
EMA of the rank-averaged squared loss (state kept in fp32), updated every micro-step with decay
``loss.rms_decay ** (micro_batch_per_gpu * world_size / loss.rms_reference_windows)``:
``rms_decay`` is the decay over ``rms_reference_windows`` windows (64: the paper's 64 GPUs x 1
window, where it is the per-micro-step decay itself), so the normaliser averages over the same
number of windows on any number of GPUs. With ``micro_batch_per_gpu > 1`` the EMA input is the
square of a micro-batch mean (the paper used 1 window per GPU). An update whose raw loss
(the same weighted sum without the RMS division) exceeds ``skip_threshold`` times its running
mean, after ``skip_warmup_updates`` updates, is not applied; the schedule still advances.
Neither is an update whose raw loss or either gradient norm is not finite, from the first update
on (clipping would spread one NaN gradient element over every gradient, and the step would make
every weight NaN); the loss normaliser and the running mean keep their state from before it, and
``max_nonfinite_updates`` such updates in a row stop the run without saving. A save also refuses
non-finite weights, so ``latest.pt`` always holds finite weights, and keeps the ``latest.pt``
it replaces as ``latest.prev.pt`` (a hard link: the copy costs nothing).
Encoder and decoder gradients are clipped separately.

Parallelism. Encoder and decoder are two FSDP roots (FULL_SHARD, one unit per transformer
layer) with bf16 parameters, gradient reduction and buffers over fp32 master weights; inputs
are cast to bf16. ``train.global_batch`` windows make one optimizer update: each rank runs
``global_batch / (micro_batch_per_gpu * world_size)`` micro-steps of ``micro_batch_per_gpu``
windows, so the updates, the schedule and the loss normaliser's horizon do not depend on the
number of GPUs.

Data order. Every epoch shuffles all windows with seed ``seed + epoch``
(``DistributedSampler``'s permutation); update ``k`` of the epoch consumes positions
``[k * global_batch, (k + 1) * global_batch)``, dealt to the ranks like
``DistributedSampler`` does. An epoch has ``num_windows // global_batch`` updates.

Schedule. Linear warmup over ``warmup_updates``, cosine decay to ``min_lr`` at
``total_updates`` (constant afterwards), stop at ``max_updates``. The learning rate of update
``u`` depends on ``u`` only.

Run directory (``run_dir``, relative to the working directory unless absolute)::

    latest.pt            resumable checkpoint written atomically by rank 0: weights, the full
                         (world-size independent) optimizer state, counters, data position,
                         loss scalers, RNG states, the model config and the run config
    archive/step_<N>/    weights-only pretrained dirs every ``archive_every_updates`` updates
    pretrained/          the final pretrained dir (config.json + model.safetensors)
    config.yaml, tb/     the resolved config of the last launch; TensorBoard (and wandb) logs

``latest.pt`` is written every ``checkpoint.save_every_updates`` updates, after
``checkpoint.save_every_minutes`` of training without a save, and at epoch ends.

Resuming. A launch continues from ``<run_dir>/latest.pt`` when it exists. Otherwise it starts
from ``resume`` (a ``latest.pt`` of another run) or, weights only, from ``init_from``,
else from scratch. Every rank must see the same checkpoint and data files (checked at start).
A run resumes on any number of GPUs for which ``micro_batch_per_gpu * world_size`` divides
``train.global_batch``: data order, learning rate, optimizer and loss scalers continue where
they stopped. Resuming on other data (the files, their frame counts and segments, their first
and last frames, or the window, stride or colour order) is refused unless
``data.allow_pool_change`` is set: train on new data in a new ``run_dir``.

Stopping. SIGUSR1 or SIGTERM received by the training processes makes every rank save the
last completed update and exit with status :data:`EXIT_STOPPED` (75); a run that reached
``max_updates`` exits 0, and relaunching it only re-exports ``pretrained/`` when that does not
hold the final update. With one ``srun`` task per GPU, ``sbatch --signal=USR1@600`` reaches the
trainer. Under ``torchrun``, signal the worker processes, early: torchrun does not forward
SIGUSR1, and on SIGTERM it kills the workers 30 s after forwarding it, which a full save may
not fit in (the G1 ``latest.pt`` is about 9 GB, gathered on rank 0). torchrun exits 1 for any
nonzero worker status. DataLoader workers ignore both signals.

LPIPS. Rank 0 builds LPIPS-VGG with torchvision's ImageNet VGG16 weights (downloaded into
``$TORCH_HOME`` on first use) and broadcasts it, so the other ranks never download. Where
compute nodes have no internet access, run ``python -m coupled_wm.world_model.train_tokenizer
--download-lpips`` once on a machine that has it (same ``$TORCH_HOME``).

torch.compile (``compile``) makes Triton build a small C module on first use: the machine
needs a C compiler (``$CC``, gcc or clang) and the Python headers, or ``--set compile=false``.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import functools
import hashlib
import logging
import math
import os
import shutil
import socket
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Sampler

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
from coupled_wm.world_model.export import save_tokenizer_pretrained
from coupled_wm.world_model.loading import (
    DECODER_PREFIX,
    ENCODER_PREFIX,
    MODEL_CONFIG_KEY,
    as_model_config,
    load_state_dict_strict,
    tokenizer_model_config,
)
from coupled_wm.world_model.models import (
    TOKENIZER_MODEL_TYPE,
    BlockCausalDecoderLayer,
    BlockCausalEncoderLayer,
    build_tokenizer,
    tokenizer_arch,
)

log = logging.getLogger("coupled_wm.world_model.train_tokenizer")  # also when run with -m

CHECKPOINT_FORMAT = "coupled_wm.train_tokenizer"
CHECKPOINT_VERSION = 1
EXIT_STOPPED = 75
"""Exit status of a run stopped by a signal before ``schedule.max_updates`` (``EX_TEMPFAIL``:
re-run the same command to resume); a finished run exits 0."""

_PRINT_FIRST_UPDATES = 5  # console lines for the first updates of every launch

DEFAULT_CONFIG: dict[str, Any] = {
    "run_dir": None,
    "seed": 42,
    "resume": "auto",
    "init_from": None,
    "init_allow_missing": [],
    "data": {
        "dirs": None,
        "include_globs": ["*.h5"],
        "exclude_globs": [],
        "min_chunk_index": None,
        "file_list": None,
        "window": 64,
        "stride": 1,
        "color_order": "bgr",
        "control_hz": None,
        "shuffle": True,
        "num_workers": 4,
        "prefetch_factor": 2,
        "allow_pool_change": False,
    },
    "model": {},
    "loss": {
        "mse_weight": 1.0,
        "lpips_weight": 0.2,
        "lpips_net": "vgg",
        "proprio_weight": 0.01,
        "rms_decay": 0.99,
        "rms_reference_windows": 64,
        "rms_eps": 1.0e-8,
    },
    "optim": {
        "lr": 1.0e-4,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
    },
    "schedule": {
        "warmup_updates": 1400,
        "total_updates": 345360,
        "min_lr": 1.0e-8,
        "max_updates": 45000,
    },
    "train": {
        "global_batch": 256,
        "micro_batch_per_gpu": 1,
        "mixed_precision": True,
        "sharding_strategy": "FULL_SHARD",
        "skip_threshold": 5.0,
        "skip_warmup_updates": 200,
        "skip_ema_decay": 0.99,
        "max_nonfinite_updates": 20,
    },
    "compile": "reduce-overhead",
    "checkpoint": {
        "save_every_updates": 1000,
        "save_every_minutes": 60,
        "at_epoch_end": True,
        "archive_every_updates": 2000,
        "keep_archives": 5,
    },
    "logging": {
        "print_every_updates": 200,
    },
    "wandb": {
        "enabled": False,
        "mode": "offline",
        "project": "coupled-wm",
        "entity": None,
        "name": None,
        "group": None,
        "tags": [],
    },
    "dist": {
        "backend": "nccl",
        "timeout_minutes": 30,
    },
}
"""Every key a tokenizer config may set. The model, loss, optim, schedule, train and compile
values are the paper's G1 recipe. The data block is generic (every ``*.h5`` file of
``data.dirs``; ``color_order: bgr`` is the G1 data's): the data pool and paths come from the
config file, and ``configs/world_model/g1_tokenizer.yaml`` selects the paper's 147 files. ``model`` is the
architecture dict of :func:`~coupled_wm.world_model.models.build_tokenizer` (missing keys take
the G1 values)."""

_FREE_BLOCKS = ("model",)
_SHARDING = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    "NO_SHARD": ShardingStrategy.NO_SHARD,
}
_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
# Architecture keys that change the network without changing any parameter shape: a
# weights-only initialisation must agree on them (shape changes fail the strict load anyway).
_SHAPE_FREE_KEYS = ("num_heads_q", "num_heads_kv_latent", "temporal_every")


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

def _merge(defaults: Mapping[str, Any], user: Mapping[str, Any], prefix: str = "") -> dict:
    if not isinstance(user, Mapping):
        raise TypeError(f"config block {prefix.rstrip('.') or '<root>'} must be a mapping, got {user!r}")
    unknown = sorted(set(user) - set(defaults))
    if unknown:
        raise KeyError(f"unknown config key(s) {[prefix + k for k in unknown]}")
    out = copy.deepcopy(dict(defaults))
    for key, value in user.items():
        if not prefix and isinstance(defaults[key], dict) and key not in _FREE_BLOCKS:
            out[key] = _merge(defaults[key], value or {}, f"{key}.")
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_config(cfg: Mapping[str, Any]) -> dict:
    """Fill defaults into ``cfg``, validate it and return the complete config.

    Unknown keys raise; ``model`` is completed by
    :func:`~coupled_wm.world_model.models.tokenizer_arch`.
    """
    c = _merge(DEFAULT_CONFIG, cfg)
    c["model"] = tokenizer_arch(c["model"])
    arch, data, sched, train = c["model"], c["data"], c["schedule"], c["train"]

    if not c["run_dir"]:
        raise ValueError("run_dir is required")
    if not data["dirs"] and not data["file_list"]:
        raise ValueError("data.dirs (or data.file_list) is required")
    if not 1 <= int(data["window"]) <= arch["seq_len"]:
        raise ValueError(f"data.window={data['window']} must be in [1, model.seq_len={arch['seq_len']}] "
                         "(the temporal RoPE table has seq_len positions)")
    for key in ("global_batch", "micro_batch_per_gpu"):
        if not isinstance(train[key], int) or train[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer, got {train[key]!r}")
    if train["sharding_strategy"] not in _SHARDING:
        raise ValueError(f"train.sharding_strategy must be one of {sorted(_SHARDING)}")
    if c["compile"] not in (False, True, None) and c["compile"] not in _COMPILE_MODES:
        raise ValueError(f"compile must be false, true or one of {_COMPILE_MODES}, got {c['compile']!r}")
    if sched["max_updates"] is None:
        sched["max_updates"] = sched["total_updates"]
    if not 0 <= sched["warmup_updates"] < sched["total_updates"]:
        raise ValueError("schedule: need 0 <= warmup_updates < total_updates")
    if sched["max_updates"] < 1:
        raise ValueError("schedule.max_updates must be >= 1")
    for key in ("mse_weight", "lpips_weight", "proprio_weight"):
        if c["loss"][key] < 0:
            raise ValueError(f"loss.{key} must be >= 0")
    decay, reference = c["loss"]["rms_decay"], c["loss"]["rms_reference_windows"]
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 <= decay <= 1:
        raise ValueError(f"loss.rms_decay must be in [0, 1], got {decay!r}")
    if isinstance(reference, bool) or not isinstance(reference, int) or reference < 1:
        raise ValueError(f"loss.rms_reference_windows must be a positive integer, got {reference!r}")
    streak = train["max_nonfinite_updates"]
    if isinstance(streak, bool) or not isinstance(streak, int) or streak < 1:
        raise ValueError(f"train.max_nonfinite_updates must be a positive integer, got {streak!r}")
    if c["checkpoint"]["save_every_updates"] < 1 or c["checkpoint"]["keep_archives"] < 1:
        raise ValueError("checkpoint.save_every_updates and checkpoint.keep_archives must be >= 1")
    minutes = c["checkpoint"]["save_every_minutes"]
    if minutes is not None and not minutes > 0:
        raise ValueError(f"checkpoint.save_every_minutes must be null or > 0, got {minutes!r}")
    if not isinstance(data["allow_pool_change"], bool):
        raise ValueError("data.allow_pool_change must be true or false")
    if c["resume"] != "auto" and c["init_from"]:
        raise ValueError("set resume (continue a run) or init_from (weights only), not both")
    return c


def grad_accum_steps(global_batch: int, micro_batch_per_gpu: int, world_size: int) -> int:
    """Micro-steps per optimizer update: ``global_batch / (micro_batch_per_gpu * world_size)``."""
    per_step = micro_batch_per_gpu * world_size
    if global_batch % per_step:
        raise ValueError(f"train.global_batch={global_batch} is not a multiple of micro_batch_per_gpu * world_size "
                         f"= {micro_batch_per_gpu} * {world_size}")
    return global_batch // per_step


# --------------------------------------------------------------------------------------
# Schedule, loss scaling, data order
# --------------------------------------------------------------------------------------

def lr_at(update: int, peak_lr: float, warmup_updates: int, total_updates: int, min_lr: float) -> float:
    """Learning rate of optimizer update ``update`` (0-based).

    Linear warmup from 0, then cosine decay floored at ``min_lr``: the paper's ``LambdaLR``
    with its progress capped at 1, so the rate stays at ``min_lr`` after ``total_updates``.
    """
    if update < warmup_updates:
        factor = float(update) / float(max(1, warmup_updates))
    else:
        progress = float(update - warmup_updates) / float(max(1, total_updates - warmup_updates))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        factor = max(min_lr / peak_lr, cosine)
    return peak_lr * factor


def _all_reduce_(t: torch.Tensor, op: str) -> torch.Tensor:
    """In-place all-reduce (``"mean"`` or ``"max"``) that also works on gloo (no AVG there)."""
    if dist.is_initialized() and dist.get_world_size() > 1:
        if op == "mean":
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t.div_(dist.get_world_size())
        elif op == "max":
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(op)
    return t


def rms_decay_per_micro_step(rms_decay: float, windows_per_micro_step: int, reference_windows: int) -> float:
    """EMA decay of the loss normaliser for one micro-step of ``windows_per_micro_step`` windows
    (``micro_batch_per_gpu * world_size``): ``rms_decay ** (windows_per_micro_step / reference_windows)``.

    ``rms_decay`` is the decay over ``reference_windows`` windows, so the normaliser forgets at
    the same rate per window (and per update) on any number of GPUs; it is ``rms_decay`` itself
    when ``windows_per_micro_step == reference_windows`` (the paper: 64 GPUs x 1 window, 0.99).
    """
    return rms_decay ** (windows_per_micro_step / reference_windows)


class RMSLossScaler:
    """Divides each named loss by the running RMS of its value.

    The squared loss is averaged over ranks and tracked by an EMA (``decay`` per call) in fp32;
    the division happens in the loss's own dtype.
    """

    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = decay
        self.eps = eps
        self.ema_sq: dict[str, torch.Tensor] = {}

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean_sq = _all_reduce_(value.detach().float().pow(2).mean(), "mean")
            prev = self.ema_sq.get(name)
            self.ema_sq[name] = mean_sq if prev is None else self.decay * prev + (1.0 - self.decay) * mean_sq
            rms = (self.ema_sq[name] + self.eps).sqrt()
        return value / rms.to(value.dtype)

    def rms(self) -> dict[str, float]:
        """Running RMS of each loss (without ``eps``)."""
        return {k: float(v.sqrt()) for k, v in self.ema_sq.items()}

    def state_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.ema_sq.items()}

    def load_state_dict(self, state: Mapping[str, float], device: torch.device | str = "cpu") -> None:
        self.ema_sq = {k: torch.tensor(float(v), dtype=torch.float32, device=device) for k, v in state.items()}

    def snapshot(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.ema_sq.items()}

    def restore(self, snapshot: Mapping[str, torch.Tensor]) -> None:
        self.ema_sq = {k: v.clone() for k, v in snapshot.items()}


class EpochWindowSampler(Sampler[int]):
    """This rank's window indices for one epoch, from a given update onwards.

    The epoch order is ``torch.randperm(num_windows)`` seeded with ``seed + epoch`` (the
    permutation of ``DistributedSampler``). Update ``k`` uses positions
    ``[k * global_batch, (k + 1) * global_batch)``; rank ``r`` reads positions
    ``r, r + world_size, ...``. From update 0 this is a prefix of ``DistributedSampler``'s
    sequence, and the windows of each update do not depend on the world size.
    """

    def __init__(self, num_windows: int, global_batch: int, rank: int, world_size: int, seed: int,
                 shuffle: bool = True):
        if global_batch % world_size:
            raise ValueError("global_batch must be a multiple of world_size")
        if num_windows < global_batch:
            raise ValueError(f"the dataset has {num_windows} windows, fewer than one update "
                             f"(train.global_batch={global_batch})")
        self.num_windows = num_windows
        self.global_batch = global_batch
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.updates_per_epoch = num_windows // global_batch
        self.epoch = 0
        self.start_update = 0

    def set_epoch(self, epoch: int, start_update: int = 0) -> None:
        if not 0 <= start_update <= self.updates_per_epoch:
            raise ValueError(f"start_update {start_update} outside [0, {self.updates_per_epoch}]")
        self.epoch = epoch
        self.start_update = start_update

    def order(self) -> list[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            return torch.randperm(self.num_windows, generator=g).tolist()
        return list(range(self.num_windows))

    def __iter__(self):
        end = self.updates_per_epoch * self.global_batch
        return iter(self.order()[self.start_update * self.global_batch + self.rank:end:self.world_size])

    def __len__(self) -> int:
        return (self.updates_per_epoch - self.start_update) * self.global_batch // self.world_size


def data_signature(dataset: HDF5SequenceDataset, data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """What the training windows are made of, to tell whether a resume sees the same data.

    ``pool_sha256`` hashes, file by file in pool order, the file name, frame count, segment
    boundaries and the bytes of the first and last frame (and proprio vector, when loaded).
    It does not depend on the directory, so a moved or copied pool keeps its signature.
    """
    h = hashlib.sha256()
    for path, info in zip(dataset.files, dataset.file_infos):
        segments = [(int(a), int(b)) for a, b in info.segments]
        h.update(repr((os.path.basename(path), info.n_frames, segments)).encode())
        with h5py.File(path, "r") as f:
            keys = ("images", "proprio") if dataset.load_proprio else ("images",)
            for key in keys:
                for t in (0, info.n_frames - 1):
                    h.update(np.ascontiguousarray(f[key][t]).tobytes())
    return {"pool_sha256": h.hexdigest(), "files": len(dataset.files), "window": data_cfg["window"],
            "stride": data_cfg["stride"], "color_order": data_cfg["color_order"]}


# --------------------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------------------

def _make_lpips(net: str, trunk_weights: bool, verbose: bool) -> nn.Module:
    import lpips

    return lpips.LPIPS(net=net, pnet_rand=not trunk_weights, verbose=verbose)


def build_lpips(net: str, device: torch.device, dtype: torch.dtype, rank: int) -> nn.Module:
    """Frozen LPIPS in eval mode. Only rank 0 loads the ImageNet trunk (and may download it);
    the other ranks build a randomly initialised copy and receive rank 0's tensors. If any
    rank fails to build it, every rank raises."""
    model, error = None, None
    try:
        model = _make_lpips(net, trunk_weights=rank == 0, verbose=rank == 0)
    except Exception as e:  # noqa: BLE001 - re-raised on every rank below
        error = e
    failed = _all_reduce_(torch.tensor([float(error is not None)], device=device), "max")
    if failed.item() > 0:
        cause = f"{type(error).__name__}: {error}" if error is not None else "see the log of the failing rank"
        raise RuntimeError(
            f"building LPIPS-{net} failed ({cause}). Rank 0 loads torchvision's ImageNet {net.upper()} weights "
            f"from {torch.hub.get_dir()}/checkpoints and downloads them there if missing; without internet "
            "access, run `python -m coupled_wm.world_model.train_tokenizer --download-lpips` once, with the "
            "same TORCH_HOME, on a machine that has it") from error
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if dist.is_initialized() and dist.get_world_size() > 1:
        for t in model.state_dict().values():
            dist.broadcast(t, src=0)
    return model


def wrap_fsdp(module: nn.Module, device: torch.device, mixed_precision: bool, sharding_strategy: str) -> FSDP:
    """FSDP1 with one unit per transformer layer (bf16 params / reduction / buffers when
    ``mixed_precision``), as the paper's tokenizer run."""
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    return FSDP(
        module,
        sharding_strategy=_SHARDING[sharding_strategy],
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={BlockCausalEncoderLayer, BlockCausalDecoderLayer}),
        mixed_precision=mp if mixed_precision else None,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,
    )


def read_init_weights(path: str | os.PathLike) -> tuple[dict, dict, dict | None]:
    """``(encoder state, decoder state, architecture or None)`` from a tokenizer pretrained dir
    or a training checkpoint with ``enc``/``dec`` state dicts (prefixes stripped)."""
    path = Path(path)
    if is_pretrained_dir(path):
        state, cfg = load_pretrained(path)
        enc = {k[len(ENCODER_PREFIX):]: v for k, v in state.items() if k.startswith(ENCODER_PREFIX)}
        dec = {k[len(DECODER_PREFIX):]: v for k, v in state.items() if k.startswith(DECODER_PREFIX)}
        if len(enc) + len(dec) != len(state):
            raise KeyError(f"{path}: tensors outside 'encoder.'/'decoder.'")
        return enc, dec, cfg.get("architecture")
    if not path.is_file():
        raise FileNotFoundError(f"init_from: {path} is neither a pretrained dir nor a checkpoint file")
    ckpt = load_checkpoint(path, mmap=True)
    if not isinstance(ckpt, Mapping) or "enc" not in ckpt or "dec" not in ckpt:
        raise KeyError(f"init_from: {path} holds no 'enc'/'dec' state dicts")
    arch = (ckpt.get(MODEL_CONFIG_KEY) or {}).get("architecture")
    return strip_compile_prefix(ckpt["enc"]), strip_compile_prefix(ckpt["dec"]), arch


def _expand_allow_missing(patterns: Sequence[str], module: nn.Module, prefix: str) -> list[str]:
    """State-dict keys of ``module`` matched by ``<prefix><key>`` glob patterns."""
    return [k for k in module.state_dict() if any(fnmatch.fnmatchcase(prefix + k, p) for p in patterns)]


def _as_tuple(x: Any) -> Any:
    return tuple(x) if isinstance(x, (list, tuple)) else x


def _keep_previous(path: Path, prev: Path) -> None:
    """Hard-link ``prev`` to ``path`` (when it exists), replacing an older ``prev``: the next atomic
    save renames a new file over ``path``, and the old file lives on as ``prev`` at no cost."""
    if not path.is_file():
        return
    tmp = prev.with_name(f".{prev.name}.tmp")
    try:
        tmp.unlink(missing_ok=True)
        os.link(path, tmp)
        os.replace(tmp, prev)
    except OSError as e:  # a filesystem without hard links: no previous copy
        log.warning("could not keep %s as %s: %s", path, prev, e)


def _rng_state(device: torch.device) -> dict[str, torch.Tensor]:
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _set_rng_state(state: Mapping[str, torch.Tensor], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def _exported_update(path: Path) -> int | None:
    """``provenance.global_update`` of the pretrained dir ``path``; None if absent or unreadable."""
    if not is_pretrained_dir(path):
        return None
    try:
        return (load_pretrained_config(path).get("provenance") or {}).get("global_update")
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------

class TokenizerTrainer:
    """One training launch: set up from ``cfg`` (see :data:`DEFAULT_CONFIG`), then :meth:`run`."""

    def __init__(self, cfg: Mapping[str, Any], info: DistInfo):
        self.cfg = resolve_config(cfg)
        self.device = info.device
        self.rank = info.rank
        self.world = info.world_size
        self.run_dir = Path(os.path.abspath(os.path.expanduser(self.cfg["run_dir"])))
        self.arch = self.cfg["model"]
        self.use_proprio = self.arch["use_proprio"]
        loss = self.cfg["loss"]
        self.w_mse, self.w_lpips = loss["mse_weight"], loss["lpips_weight"]
        self.w_prop = loss["proprio_weight"] if self.use_proprio else 0.0
        train = self.cfg["train"]
        self.global_batch = train["global_batch"]
        self.micro_batch = train["micro_batch_per_gpu"]
        self.accum = grad_accum_steps(self.global_batch, self.micro_batch, self.world)
        self.mixed_precision = bool(train["mixed_precision"])
        self.cdtype = torch.bfloat16 if self.mixed_precision else torch.float32
        sched = self.cfg["schedule"]
        self.max_updates = int(sched["max_updates"])
        self.offload = self.device.type != "cpu"

        self.global_update = 0
        self.epoch = 0
        self.epoch_update = 0
        self.ema_raw_loss: float | None = None
        self.skip_count = 0
        self.nonfinite_count = 0  # non-finite updates not applied, over the run
        self.nonfinite_streak = 0  # ... in a row, in this launch
        self.last_saved_update = -1
        self.last_save_time = time.monotonic()
        self.launch_first_update = 0
        self.wandb_run_id: str | None = None
        self.start_kind = "scratch"
        self.start_source: str | None = None
        self.start_path: Path | None = None
        self.init_source: str | None = None  # the init_from the run started from (file name)
        self.data_signature: dict[str, Any] | None = None
        self.logger: ExperimentLogger | None = None
        self.signals = StopSignals()

    # ------------------------------------------------------------------ setup

    def _start_checkpoint(self) -> Path | None:
        latest = self.run_dir / "latest.pt"
        resume = self.cfg["resume"]
        if resume != "auto" and not Path(resume).is_file():
            raise FileNotFoundError(f"resume: checkpoint {resume} does not exist")
        if latest.is_file():
            return latest
        return None if resume == "auto" else Path(resume)

    def _setup_data(self) -> None:
        d = self.cfg["data"]
        files = list_h5_files(d["dirs"], include_globs=d["include_globs"], exclude_globs=d["exclude_globs"],
                              min_chunk_index=d["min_chunk_index"], file_list=d["file_list"])
        self.dataset = HDF5SequenceDataset(
            files, window=d["window"], stride=d["stride"], image_size=self.arch["image_size"],
            color_order=d["color_order"], load_proprio=self.use_proprio, load_action=False, reward_key=None,
        )
        if self.dataset.resized and self.rank == 0:
            log.info("resizing %s frames to model.image_size %s (bilinear, as the RL environment)",
                     [list(s) for s in self.dataset.native_image_sizes], list(self.arch["image_size"]))
        if self.use_proprio and self.dataset.proprio_dim != self.arch["proprio_dim"]:
            raise ValueError(f"data proprio has {self.dataset.proprio_dim} dims but model.proprio_dim is "
                             f"{self.arch['proprio_dim']}")
        self.sampler = EpochWindowSampler(len(self.dataset), self.global_batch, self.rank, self.world,
                                          seed=self.cfg["seed"], shuffle=d["shuffle"])
        workers = int(d["num_workers"])
        self.loader = SignalSafeDataLoader(
            self.dataset, batch_size=self.micro_batch, sampler=self.sampler, num_workers=workers,
            pin_memory=self.device.type == "cuda", drop_last=True, persistent_workers=workers > 0,
            prefetch_factor=d["prefetch_factor"] if workers > 0 else None,
            generator=torch.Generator(),  # the global RNG drives the MAE masks; keep the loader off it
        )
        self.updates_per_epoch = self.sampler.updates_per_epoch
        self.data_signature = data_signature(self.dataset, d)

    def _remove_stale_temp_files(self) -> None:
        """Delete the temporary files of saves that were killed before they finished."""
        for pattern in (".latest.pt.tmp", "pretrained/.*.tmp*", "archive/step_*/.*.tmp*"):
            for p in self.run_dir.glob(pattern):
                size = p.stat().st_size
                p.unlink(missing_ok=True)
                log.warning("removed %s (%.1f MB), left by an interrupted save", p, size / 2**20)

    def _check_ranks_agree(self, view: Mapping[str, Any], start: Path | None, error: Exception | None) -> None:
        """Raise on every rank unless all ranks found the same start checkpoint and data pool.

        Ranks that disagree would enter different collectives and hang; relative paths resolve
        against each process's working directory, and a run dir may not be shared by all nodes.
        """
        if dist.is_initialized() and self.world > 1:
            where = (f"rank {self.rank} on {socket.gethostname()} (cwd {os.getcwd()})", str(start or "no checkpoint"),
                     str(self.run_dir))
            views: list = [None] * self.world
            dist.all_gather_object(views, (dict(view), where))
            if any(v != views[0][0] for v, _ in views):
                lines = []
                for v, (who, ckpt_path, _) in views:
                    data = (v["data"] or {}).get("pool_sha256", "none")[:12]
                    lines.append(f"  {who}: {ckpt_path} {v['checkpoint'] or ''}, data pool {data}"
                                 + (f", error {v['error']}" if v["error"] else ""))
                raise RuntimeError("the ranks do not see the same start checkpoint and data files "
                                   "((size, update) of the checkpoint, data pool signature):\n" + "\n".join(lines)
                                   + "\nrun_dir and data.dirs must name the same files on every node (relative "
                                   "paths resolve against each process's working directory)") from error
            run_dirs = sorted({rd for _, (_, _, rd) in views})
            if len(run_dirs) > 1 and self.rank == 0:
                log.warning("the ranks resolve run_dir to different directories %s: only rank 0 writes, so a "
                            "relaunch fails unless every rank sees rank 0's files", run_dirs)
        if error is not None:
            raise error

    def _check_data(self, ckpt: Mapping, path: Path) -> None:
        stored = ckpt.get("data_signature") or {}
        changed = [k for k, v in self.data_signature.items() if stored.get(k) != v]
        if not changed:
            return
        msg = f"{path} was trained on other data ({', '.join(changed)} changed)"
        if not self.cfg["data"]["allow_pool_change"]:
            raise ValueError(f"{msg}, and a launch with this run_dir continues that run. To train on this data from "
                             "scratch, use a new run_dir (init_from=<checkpoint> starts from its weights only); to "
                             "continue the run on the changed data, set data.allow_pool_change=true")
        log.warning("%s: continuing anyway (data.allow_pool_change)", msg)

    def _load_start_weights(self, enc: nn.Module, dec: nn.Module, ckpt: Mapping | None) -> None:
        init_from = self.cfg["init_from"]
        if ckpt is not None:
            if init_from and Path(init_from).name != self.init_source:
                log.warning("init_from=%s is not used: the run resumes from %s, which started from %s; use a new "
                            "run_dir to start from these weights", init_from, self.start_path,
                            self.init_source or "scratch")
            load_state_dict_strict(enc, ckpt["enc"], name="encoder")
            load_state_dict_strict(dec, ckpt["dec"], name="decoder")
            return
        if not init_from:
            return
        enc_sd, dec_sd, arch = read_init_weights(init_from)
        if arch is not None:
            src = tokenizer_arch(arch)
            diff = [f"{k}: {src[k]!r} != {self.arch[k]!r}" for k in _SHAPE_FREE_KEYS if src[k] != self.arch[k]]
            if diff:
                raise ValueError(f"init_from {init_from} has a different architecture: " + "; ".join(diff))
        patterns = list(self.cfg["init_allow_missing"])
        allow_enc = _expand_allow_missing(patterns, enc, ENCODER_PREFIX)
        allow_dec = _expand_allow_missing(patterns, dec, DECODER_PREFIX)
        missing = [ENCODER_PREFIX + k for k in allow_enc if k not in enc_sd]
        missing += [DECODER_PREFIX + k for k in allow_dec if k not in dec_sd]
        load_state_dict_strict(enc, enc_sd, allow_missing=allow_enc, name="encoder")
        load_state_dict_strict(dec, dec_sd, allow_missing=allow_dec, name="decoder")
        log.info("initialised weights from %s%s", init_from,
                 f"; kept fresh initialisation for {missing}" if missing else "")
        self.start_kind, self.start_source = "init_from", Path(init_from).name
        self.init_source = Path(init_from).name

    def _check_checkpoint(self, ckpt: Mapping, path: Path) -> None:
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint (use init_from to start from "
                             "the weights of another tokenizer checkpoint)")
        if ckpt.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError(f"{path}: unsupported checkpoint version {ckpt.get('format_version')!r}")
        stored = as_model_config(ckpt[MODEL_CONFIG_KEY], TOKENIZER_MODEL_TYPE)["architecture"]
        diff = [f"{k}: {stored[k]!r} != {self.arch[k]!r}" for k in stored if stored[k] != self.arch[k]]
        if diff:
            raise ValueError(f"model config differs from the architecture in {path}: " + "; ".join(diff))

    def _restore_counters(self, ckpt: Mapping, path: Path) -> None:
        self._check_data(ckpt, path)
        self.init_source = (ckpt[MODEL_CONFIG_KEY].get("provenance") or {}).get("init_from")
        self.global_update = int(ckpt["global_update"])
        self.epoch = int(ckpt["epoch"])
        self.epoch_update = int(ckpt["epoch_update"])
        if (ckpt["num_windows"], ckpt["global_batch"]) != (len(self.dataset), self.global_batch):
            log.warning("windows/global_batch changed since the checkpoint (%d/%d -> %d/%d): continuing with a "
                        "new epoch", ckpt["num_windows"], ckpt["global_batch"], len(self.dataset), self.global_batch)
            if self.epoch_update:
                self.epoch, self.epoch_update = self.epoch + 1, 0
        sched = {k: self.cfg["schedule"][k] for k in ("warmup_updates", "total_updates", "min_lr")}
        sched["lr"] = self.cfg["optim"]["lr"]
        if ckpt.get("schedule") != sched:
            log.warning("learning-rate schedule changed since the checkpoint: %s -> %s", ckpt.get("schedule"), sched)
        self.scaler.load_state_dict(ckpt["rms_scaler"], self.device)
        self.ema_raw_loss = ckpt["skip"]["ema_raw_loss"]
        self.skip_count = int(ckpt["skip"]["skip_count"])
        self.nonfinite_count = int(ckpt["skip"].get("nonfinite_count", 0))  # absent before 2026-10-10
        self.wandb_run_id = ckpt.get("wandb_run_id")
        self.last_saved_update = self.global_update
        self.start_kind, self.start_source = "resume", path.name

    def _load_optimizer(self, ckpt: Mapping) -> None:
        with self._full_state_dict_type():
            osd = FSDP.optim_state_dict_to_load(self.root, self.optim, ckpt["optim"] if self.rank == 0 else None)
        self.optim.load_state_dict(osd)
        o = self.cfg["optim"]
        wanted = {"betas": tuple(o["betas"]), "eps": o["eps"], "weight_decay": o["weight_decay"]}
        for group in self.optim.param_groups:
            changed = {k: (group[k], v) for k, v in wanted.items() if _as_tuple(group[k]) != _as_tuple(v)}
            if changed:
                log.warning("optimizer hyperparameters changed since the checkpoint: %s", changed)
            group.update(wanted)

    def setup(self) -> None:
        cfg = self.cfg
        start, ckpt, error = None, None, None
        view: dict[str, Any] = {"checkpoint": None}
        try:
            if self.rank == 0:
                self._remove_stale_temp_files()
            start = self._start_checkpoint()
            if start is not None:
                ckpt = load_checkpoint(start, mmap=True)
                self._check_checkpoint(ckpt, start)
                view["checkpoint"] = (start.stat().st_size, int(ckpt["global_update"]))
            self._setup_data()
        except Exception as e:  # noqa: BLE001 - raised on every rank once the ranks have compared notes
            error = e
        view.update(data=self.data_signature, error=None if error is None else f"{type(error).__name__}: {error}")
        self._check_ranks_agree(view, start, error)
        self.start_path = start
        loss = cfg["loss"]
        decay = rms_decay_per_micro_step(loss["rms_decay"], self.micro_batch * self.world,
                                         loss["rms_reference_windows"])
        self.scaler = RMSLossScaler(decay, loss["rms_eps"])
        if ckpt is not None:
            self._restore_counters(ckpt, start)
            log.info("resuming %s at update %d (epoch %d, update %d/%d of the epoch)", start, self.global_update,
                     self.epoch, self.epoch_update, self.updates_per_epoch)
        if self.global_update >= self.max_updates:
            self._ckpt = ckpt
            return

        self.lpips = (build_lpips(cfg["loss"]["lpips_net"], self.device, self.cdtype, self.rank)
                      if self.w_lpips > 0 else None)

        torch.manual_seed(cfg["seed"])  # identical initialisation on every rank
        enc, dec = build_tokenizer(self.arch)
        self._load_start_weights(enc, dec, ckpt)
        torch.manual_seed(cfg["seed"] + self.rank)  # per-rank masking randomness
        n_params = sum(p.numel() for m in (enc, dec) for p in m.parameters())

        self.encoder_module = enc
        train = cfg["train"]
        self.enc_fsdp = wrap_fsdp(enc.to(self.device), self.device, self.mixed_precision, train["sharding_strategy"])
        self.dec_fsdp = wrap_fsdp(dec.to(self.device), self.device, self.mixed_precision, train["sharding_strategy"])
        self.root = nn.ModuleDict({"encoder": self.enc_fsdp, "decoder": self.dec_fsdp})
        self.enc, self.dec = self.enc_fsdp, self.dec_fsdp
        if cfg["compile"]:
            warn_if_compile_unavailable(f"compile={cfg['compile']}", "--set compile=false", self.device)
            mode = None if cfg["compile"] is True else cfg["compile"]
            self.enc = torch.compile(self.enc_fsdp, mode=mode, fullgraph=False)
            self.dec = torch.compile(self.dec_fsdp, mode=mode, fullgraph=False)

        o = cfg["optim"]
        params = list(self.enc_fsdp.parameters()) + list(self.dec_fsdp.parameters())
        self.optim = torch.optim.AdamW(params, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                                       weight_decay=o["weight_decay"])
        if ckpt is not None:
            self._load_optimizer(ckpt)
            rng = ckpt["rng"]
            if len(rng) == self.world:
                _set_rng_state(rng[self.rank], self.device)
            else:
                torch.manual_seed(cfg["seed"] + self.rank + self.world * self.global_update)
        del ckpt

        log.info("tokenizer: %s parameters; world size %d, %d micro-step(s) of %d window(s) per rank per update "
                 "(global batch %d); %d updates per epoch; LR warmup %d, cosine to %d, stop at %d",
                 f"{n_params:,}", self.world, self.accum, self.micro_batch, self.global_batch,
                 self.updates_per_epoch, cfg["schedule"]["warmup_updates"], cfg["schedule"]["total_updates"],
                 self.max_updates)
        log.info("loss normaliser: EMA decay %.10g per micro-step of %d windows (loss.rms_decay %g per %d windows)",
                 self.scaler.decay, self.micro_batch * self.world, cfg["loss"]["rms_decay"],
                 cfg["loss"]["rms_reference_windows"])

    def _setup_logger(self) -> None:
        wandb_cfg = dict(self.cfg["wandb"])
        wandb_cfg["name"] = wandb_cfg.get("name") or self.run_dir.name
        if self.wandb_run_id:
            wandb_cfg["id"] = self.wandb_run_id
        error = None
        try:
            if self.rank == 0:
                self.run_dir.mkdir(parents=True, exist_ok=True)
                save_config(self.cfg, self.run_dir / "config.yaml")
            # Updates are logged at their 1-based index: a resumed launch drops the events the
            # previous one logged after its last checkpoint.
            purge_step = self.global_update + 1 if self.global_update else 0
            self.logger = ExperimentLogger(self.run_dir / "tb", self.rank, config=self.cfg, wandb_cfg=wandb_cfg,
                                           purge_step=purge_step)
        except Exception as e:  # noqa: BLE001 - re-raised on every rank below
            error = e
        failed = _all_reduce_(torch.tensor([float(error is not None)], device=self.device), "max")
        if failed.item() > 0:
            raise RuntimeError("experiment logger setup failed on rank 0 (see its traceback)") from error
        if self.logger.wandb_run is not None:
            self.wandb_run_id = self.logger.wandb_run.id

    # ------------------------------------------------------------------ checkpoints

    def _full_state_dict_type(self):
        return FSDP.state_dict_type(
            self.root, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=self.offload),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=self.offload),
        )

    def model_config(self) -> dict:
        d, s = self.cfg["data"], self.cfg["schedule"]
        preprocessing = {
            "image_size": list(self.arch["image_size"]),
            "image_format": "float RGB in [0, 1], (T, 3, H, W)",
            "h5_image_key": "images",
            "h5_image_format": "uint8 (T, H, W, 3)",
            "h5_color_order": d["color_order"],
            "proprio_dim": self.arch["proprio_dim"] if self.use_proprio else 0,
            "h5_proprio_key": "proprio" if self.use_proprio else None,
            "proprio_normalization": None,
            "h5_action_key": "commands",
        }
        if self.dataset.resized:
            preprocessing["h5_image_resize"] = "bilinear (F.interpolate, align_corners=False) on float [0, 1] frames"
        summary = self.dataset.summary
        provenance = {
            "trainer": "coupled_wm.world_model.train_tokenizer",
            "run": self.run_dir.name,
            "global_update": self.global_update,
            "epoch": self.epoch,
            "init_from": self.init_source,
            "data": {"files": summary["files"], "frames": summary["frames"], "windows": summary["windows"],
                     "window": summary["window"], "stride": summary["stride"],
                     "pool_sha256": self.data_signature["pool_sha256"]},
            "global_batch": self.global_batch,
            "schedule": {"lr": self.cfg["optim"]["lr"], "warmup_updates": s["warmup_updates"],
                         "total_updates": s["total_updates"], "min_lr": s["min_lr"], "max_updates": s["max_updates"]},
            "loss": dict(self.cfg["loss"]),
        }
        extra = {"control_hz": d["control_hz"]} if d["control_hz"] is not None else {}
        return tokenizer_model_config(self.arch, preprocessing=preprocessing, **extra, provenance=provenance)

    def save(self, archive: bool = False, export: bool = False) -> None:
        """Write ``latest.pt`` (and an archive / the final export). All ranks must call it."""
        with self._full_state_dict_type():
            enc_sd = self.enc_fsdp.state_dict()
            dec_sd = self.dec_fsdp.state_dict()
            osd = FSDP.optim_state_dict(self.root, self.optim)
        # The guard in _finish_update keeps non-finite updates out; this one keeps latest.pt valid
        # whatever got the weights there.
        bad = []
        if self.rank == 0:
            bad = [f"{part}.{k}" for part, sd in (("encoder", enc_sd), ("decoder", dec_sd))
                   for k, t in sd.items() if t.is_floating_point() and not bool(torch.isfinite(t).all())]
        if _all_reduce_(torch.tensor([float(len(bad))], device=self.device), "max").item() > 0:
            names = ", ".join(bad[:8]) + (f" and {len(bad) - 8} more" if len(bad) > 8 else "")
            raise RuntimeError(f"refusing to save update {self.global_update}: non-finite weights "
                               f"({names or 'see the log of rank 0'}); latest.pt keeps update "
                               f"{self.last_saved_update}")
        rng = [None] * self.world
        if dist.is_initialized() and self.world > 1:
            dist.all_gather_object(rng, _rng_state(self.device))
        else:
            rng = [_rng_state(self.device)]
        if self.rank == 0:
            t0 = time.perf_counter()
            model_config = self.model_config()
            s = self.cfg["schedule"]
            ckpt = {
                "format": CHECKPOINT_FORMAT,
                "format_version": CHECKPOINT_VERSION,
                MODEL_CONFIG_KEY: model_config,
                "enc": strip_compile_prefix(enc_sd),
                "dec": strip_compile_prefix(dec_sd),
                "optim": osd,
                "global_update": self.global_update,
                "epoch": self.epoch,
                "epoch_update": self.epoch_update,
                "num_windows": len(self.dataset),
                "data_signature": self.data_signature,
                "global_batch": self.global_batch,
                "schedule": {"warmup_updates": s["warmup_updates"], "total_updates": s["total_updates"],
                             "min_lr": s["min_lr"], "lr": self.cfg["optim"]["lr"]},
                "rms_scaler": self.scaler.state_dict(),
                "skip": {"ema_raw_loss": self.ema_raw_loss, "skip_count": self.skip_count,
                         "nonfinite_count": self.nonfinite_count},
                "rng": rng,
                "world_size": self.world,
                "wandb_run_id": self.wandb_run_id,
                "config": self.cfg,
            }
            _keep_previous(self.run_dir / "latest.pt", self.run_dir / "latest.prev.pt")
            atomic_save(ckpt, self.run_dir / "latest.pt")
            log.info("saved %s at update %d (%.1f s)", self.run_dir / "latest.pt", self.global_update,
                     time.perf_counter() - t0)
            if archive:
                self._archive(ckpt["enc"], ckpt["dec"], model_config)
            if export:
                out = save_tokenizer_pretrained(self.run_dir / "pretrained", ckpt["enc"], ckpt["dec"], model_config)
                log.info("exported the tokenizer to %s", out)
        barrier()
        self.last_saved_update = self.global_update
        self.last_save_time = time.monotonic()

    def _archive(self, enc_sd, dec_sd, model_config) -> None:
        root = self.run_dir / "archive"
        save_tokenizer_pretrained(root / f"step_{self.global_update:08d}", enc_sd, dec_sd, model_config)
        archives = sorted(p for p in root.glob("step_*") if p.is_dir())
        for old in archives[:-self.cfg["checkpoint"]["keep_archives"]]:
            shutil.rmtree(old, ignore_errors=True)

    def _export_from_checkpoint(self, ckpt: Mapping) -> None:
        """Export ``ckpt`` unless ``pretrained/`` already holds its update (a launch killed between
        writing ``latest.pt`` and the export leaves none, or that of an earlier ``max_updates``)."""
        out = self.run_dir / "pretrained"
        if self.rank == 0 and _exported_update(out) != ckpt["global_update"]:
            save_tokenizer_pretrained(out, ckpt["enc"], ckpt["dec"], ckpt[MODEL_CONFIG_KEY])
            log.info("exported update %d to %s", ckpt["global_update"], out)
        barrier()

    # ------------------------------------------------------------------ training

    def _stop_requested(self) -> bool:
        flag = torch.tensor([float(self.signals.poll() is not None)], device=self.device)
        return _all_reduce_(flag, "max").item() > 0

    def _save_due_by_time(self) -> bool:
        """Whether ``checkpoint.save_every_minutes`` have passed since the last save (rank 0's clock)."""
        minutes = self.cfg["checkpoint"]["save_every_minutes"]
        if not minutes:
            return False
        due = self.rank == 0 and time.monotonic() - self.last_save_time >= 60.0 * minutes
        return _all_reduce_(torch.tensor([float(due)], device=self.device), "max").item() > 0

    def _begin_update(self) -> None:
        self.optim.zero_grad(set_to_none=True)
        self._scaler_snapshot = self.scaler.snapshot()
        self._rng_snapshot = _rng_state(self.device)
        self._sums = torch.zeros(5, device=self.device)  # mse, lpips, raw, proprio, normalized loss
        self._xhat_max = torch.zeros((), device=self.device)

    def _discard_update(self) -> None:
        """Drop a partly accumulated update: state returns to the last completed update."""
        self.optim.zero_grad(set_to_none=True)
        self.scaler.restore(self._scaler_snapshot)
        _set_rng_state(self._rng_snapshot, self.device)

    def _micro_step(self, batch: Mapping[str, torch.Tensor]) -> None:
        images = batch["image"].to(self.device, non_blocking=True).to(self.cdtype)
        proprio = None
        if self.use_proprio:
            proprio = batch["proprio"].to(self.device, non_blocking=True).to(self.cdtype)

        _, _, z = self.enc(images, proprio=proprio)
        out = self.dec(z)
        x_hat = out[1]
        self._xhat_max = torch.maximum(self._xhat_max, x_hat.detach().abs().max().float())

        zero = torch.zeros((), device=self.device)
        mse = F.mse_loss(x_hat, images)
        lp = zero
        if self.lpips is not None:
            with torch.autocast(device_type=self.device.type, dtype=self.cdtype, enabled=self.mixed_precision):
                images_lp = images * 2.0 - 1.0
                x_hat_lp = x_hat * 2.0 - 1.0
                B, T, C, H, W = images_lp.shape
                lp = self.lpips(x_hat_lp.view(B * T, C, H, W), images_lp.view(B * T, C, H, W)).mean()
        prop = zero
        if self.w_prop > 0:
            prop = F.mse_loss(out[2], proprio)
            prop_norm = self.scaler("prop", prop)

        raw = self.w_mse * mse
        if self.lpips is not None:
            raw = raw + self.w_lpips * lp
        if self.w_prop > 0:
            raw = raw + self.w_prop * prop

        loss = self.w_mse * self.scaler("mse", mse)
        if self.lpips is not None:
            loss = loss + self.w_lpips * self.scaler("lpips", lp)
        if self.w_prop > 0:
            loss = loss + self.w_prop * prop_norm
        loss = loss / self.accum
        loss.backward()

        self._sums += torch.stack([t.detach().float() for t in (mse, lp, raw, prop, loss)])

    def _finish_update(self) -> dict[str, float]:
        """Clip, reduce the statistics, apply (or skip) the update and advance the counters."""
        tc = self.cfg["train"]
        clip = self.cfg["optim"]["grad_clip"]
        gn_enc = self.enc_fsdp.clip_grad_norm_(max_norm=clip)
        gn_dec = self.dec_fsdp.clip_grad_norm_(max_norm=clip)

        means = [self._sums]
        enc = self.encoder_module
        bottleneck = hasattr(enc, "_last_pretanh_abs_mean") and hasattr(enc, "_last_z_sat_frac")
        if bottleneck:
            means.append(torch.stack([enc._last_pretanh_abs_mean.detach().float(),
                                      enc._last_z_sat_frac.detach().float()]).to(self.device))
        mean_t = _all_reduce_(torch.cat(means), "mean")
        max_t = _all_reduce_(self._xhat_max.reshape(1), "max")
        values = torch.cat([mean_t, max_t, torch.stack([torch.as_tensor(gn_enc), torch.as_tensor(gn_dec)])
                            .float().to(self.device)]).tolist()
        mse_m, lpips_m, raw_m, prop_m = (v / self.accum for v in values[:4])
        stats = {"mse_mean": mse_m, "lpips_mean": lpips_m, "raw_loss_mean": raw_m, "prop_mean": prop_m,
                 "normalized_loss_mean": values[4]}
        if bottleneck:
            stats["pretanh_abs_mean"], stats["z_sat_frac"] = values[5], values[6]
        stats["xhat_abs_max"] = values[-3]
        stats["grad_norm_enc"], stats["grad_norm_dec"] = values[-2], values[-1]

        # Every rank holds the same reduced values, so every rank takes the same branch.
        nonfinite = not all(math.isfinite(v) for v in (raw_m, stats["grad_norm_enc"], stats["grad_norm_dec"]))
        if nonfinite:
            self.nonfinite_count += 1
            self.nonfinite_streak += 1
            log.warning("skipping update %d: non-finite raw loss or grad norm (raw loss %.4g, grad norms %.3g / %.3g; "
                        "%d in a row, %d in the run)", self.global_update, raw_m, stats["grad_norm_enc"],
                        stats["grad_norm_dec"], self.nonfinite_streak, self.nonfinite_count)
            if self.nonfinite_streak >= tc["max_nonfinite_updates"]:
                raise RuntimeError(f"{self.nonfinite_streak} non-finite updates in a row (train.max_nonfinite_updates):"
                                   f" stopping without saving; latest.pt keeps update {self.last_saved_update}")
            self.scaler.restore(self._scaler_snapshot)  # the normaliser may have taken a non-finite loss
        else:
            self.nonfinite_streak = 0

        threshold = tc["skip_threshold"]
        skip = nonfinite or (self.ema_raw_loss is not None and self.global_update >= tc["skip_warmup_updates"]
                             and raw_m > threshold * self.ema_raw_loss)
        if skip and not nonfinite:
            self.skip_count += 1
            log.warning("skipping update %d: raw loss %.4g is %.2fx its running mean %.4g (grad norms %.3g / %.3g)",
                        self.global_update, raw_m, raw_m / self.ema_raw_loss, self.ema_raw_loss,
                        stats["grad_norm_enc"], stats["grad_norm_dec"])
        if nonfinite:
            pass  # min() would carry a NaN into the running mean, and no later update could be skipped
        elif self.ema_raw_loss is None:
            self.ema_raw_loss = raw_m
        else:
            capped = min(raw_m, threshold * self.ema_raw_loss)
            decay = tc["skip_ema_decay"]
            self.ema_raw_loss = decay * self.ema_raw_loss + (1.0 - decay) * capped

        if skip:
            self.optim.zero_grad(set_to_none=True)
        else:
            lr = self._lr(self.global_update)
            for group in self.optim.param_groups:
                group["lr"] = lr
            self.optim.step()

        self.global_update += 1
        self.epoch_update += 1
        if self.epoch_update == self.updates_per_epoch:
            self.epoch, self.epoch_update = self.epoch + 1, 0
        stats["skipped"] = float(skip)
        return stats

    def _lr(self, update: int) -> float:
        s = self.cfg["schedule"]
        return lr_at(update, self.cfg["optim"]["lr"], s["warmup_updates"], s["total_updates"], s["min_lr"])

    def _log(self, stats: Mapping[str, float], update_time: float, data_time: float) -> None:
        u = self.global_update
        lg = self.logger
        for key in ("prop_mean", "mse_mean", "lpips_mean", "raw_loss_mean", "normalized_loss_mean",
                    "grad_norm_enc", "grad_norm_dec", "xhat_abs_max", "skipped"):
            lg.scalar(f"train/{key}", stats[key], u)
        lg.scalar("train/lr", self._lr(u), u)
        lg.scalar("train/skip_count", self.skip_count, u)
        lg.scalar("train/nonfinite_count", self.nonfinite_count, u)
        if self.ema_raw_loss is not None:  # None until the first finite update (rank 0 would raise, the others hang)
            lg.scalar("train/ema_raw_loss", self.ema_raw_loss, u)
        lg.scalar("train/epoch", self.epoch + self.epoch_update / self.updates_per_epoch, u)
        frames_per_s = self.global_batch * self.cfg["data"]["window"] / max(update_time, 1e-9)
        lg.scalar("perf/update_seconds", update_time, u)
        lg.scalar("perf/data_seconds", data_time, u)
        lg.scalar("perf/frames_per_second", frames_per_s, u)
        for name, value in self.scaler.rms().items():
            lg.scalar(f"rms_scaler/ema_rms_{name}", value, u)
        if "pretanh_abs_mean" in stats:
            lg.scalar("bottleneck/pretanh_abs_mean", stats["pretanh_abs_mean"], u)
            lg.scalar("bottleneck/z_sat_frac", stats["z_sat_frac"], u)
        every = self.cfg["logging"]["print_every_updates"]
        if every and (u % every == 0 or u - self.launch_first_update <= _PRINT_FIRST_UPDATES):
            log.info("update %d/%d | epoch %d (%d/%d) | loss %.5f raw %.5f mse %.3e lpips %.4f prop %.3e | "
                     "lr %.3e | grad %.2f/%.2f | %.2f s/update (data %.3f s) %.0f frames/s",
                     u, self.max_updates, self.epoch, self.epoch_update, self.updates_per_epoch,
                     stats["normalized_loss_mean"], stats["raw_loss_mean"], stats["mse_mean"], stats["lpips_mean"],
                     stats["prop_mean"], self._lr(u), stats["grad_norm_enc"], stats["grad_norm_dec"],
                     update_time, data_time, frames_per_s)

    def _after_update(self, epoch_ended: bool) -> str | None:
        """Checkpoint as configured; ``"done"`` once ``max_updates`` is reached."""
        ck = self.cfg["checkpoint"]
        u = self.global_update
        done = u >= self.max_updates
        archive = bool(ck["archive_every_updates"]) and u % ck["archive_every_updates"] == 0
        timed = self._save_due_by_time()
        if done or u % ck["save_every_updates"] == 0 or archive or (epoch_ended and ck["at_epoch_end"]) or timed:
            self.save(archive=archive, export=done)
        return "done" if done else None

    def _run_epoch(self) -> str | None:
        """Train until the epoch ends (``None``), ``max_updates`` (``"done"``) or a signal (``"signal"``)."""
        self.sampler.set_epoch(self.epoch, self.epoch_update)
        self.enc.train()
        self.dec.train()
        micro = 0
        data_time = 0.0
        t_update = time.perf_counter()
        t_data = time.perf_counter()
        for batch in self.loader:
            data_time += time.perf_counter() - t_data
            if self._stop_requested():
                if micro:
                    self._discard_update()
                return "signal"
            if micro == 0:
                self._begin_update()
            self._micro_step(batch)
            micro += 1
            if micro == self.accum:
                micro = 0
                epoch = self.epoch
                stats = self._finish_update()
                now = time.perf_counter()
                self._log(stats, now - t_update, data_time)
                status = self._after_update(epoch_ended=self.epoch != epoch)
                if status:
                    return status
                if self.epoch != epoch:
                    break
                data_time = 0.0
                t_update = time.perf_counter()
            t_data = time.perf_counter()
        return None

    def run(self) -> dict[str, Any]:
        """Train until ``schedule.max_updates`` or a stop signal; returns a short summary."""
        with self.signals:
            self.setup()
            if self.global_update >= self.max_updates:
                log.info("run %s is already at update %d (max_updates %d)", self.run_dir, self.global_update,
                         self.max_updates)
                self._export_from_checkpoint(self._ckpt)
                return {"global_update": self.global_update, "status": "done"}
            self._setup_logger()
            log.info("%s run in %s from update %d%s", self.start_kind, self.run_dir, self.global_update,
                     f" ({self.start_source})" if self.start_source else "")
            self.launch_first_update = self.global_update
            self.last_save_time = time.monotonic()
            try:
                status = None
                while status is None:
                    status = self._run_epoch()
                if status == "signal":
                    log.warning("stop signal received: saving update %d and exiting (relaunch to resume)",
                                self.global_update)
                    if self.last_saved_update != self.global_update:
                        self.save()
            finally:
                self.logger.close()
        return {"global_update": self.global_update, "status": status}


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    """Run the CLI; returns :meth:`TokenizerTrainer.run`'s summary (None for ``--download-lpips``)."""
    parser = argparse.ArgumentParser(description="Train the DreamerV4 tokenizer (see the module docstring).")
    parser.add_argument("--config", help="YAML config (e.g. configs/world_model/g1_tokenizer.yaml)")
    parser.add_argument("--set", dest="overrides", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. --set train.global_batch=64 schedule.max_updates=1000")
    parser.add_argument("--download-lpips", action="store_true",
                        help="download the LPIPS trunk weights into $TORCH_HOME and exit")
    args = parser.parse_args(argv)

    if args.download_lpips:
        setup_logging()
        net = DEFAULT_CONFIG["loss"]["lpips_net"]
        if args.config:
            net = resolve_config(load_config(args.config, args.overrides, DEFAULT_CONFIG))["loss"]["lpips_net"]
        _make_lpips(net, trunk_weights=True, verbose=True)
        log.info("LPIPS-%s weights are cached", net)
        return None
    if not args.config:
        parser.error("--config is required")

    # FSDP1's state_dict_type API, used for the full (world-size independent) checkpoints.
    warnings.filterwarnings("ignore", message=r"FSDP\.state_dict_type\(\)", category=FutureWarning)
    cfg = load_config(args.config, args.overrides, DEFAULT_CONFIG)  # --set reaches keys the YAML omits
    backend = cfg.get("dist", {}).get("backend", DEFAULT_CONFIG["dist"]["backend"])
    timeout = cfg.get("dist", {}).get("timeout_minutes", DEFAULT_CONFIG["dist"]["timeout_minutes"])
    info = init_distributed(require_process_group=True, backend=backend, timeout_minutes=timeout)
    setup_logging(info.rank)
    try:
        return TokenizerTrainer(cfg, info).run()
    finally:
        cleanup()


if __name__ == "__main__":
    summary = main()
    sys.exit(EXIT_STOPPED if summary is not None and summary["status"] == "signal" else 0)
