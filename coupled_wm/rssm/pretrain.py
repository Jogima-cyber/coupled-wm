# SPDX-License-Identifier: Apache-2.0
"""Offline pretraining of the RSSM (the local world model) on HDF5 episode data.

    python -m coupled_wm.rssm.pretrain --config configs/rssm/g1_rssm_pretrain.yaml [--set key=value ...]
    torchrun --nproc_per_node=N -m coupled_wm.rssm.pretrain --config ...   # optional data parallel

Objective: the DreamerV3 world-model loss of :func:`coupled_wm.rssm.rssm.reconstruction_loss`
(sum-of-squares image reconstruction, two-hot reward NLL with an fp32 log-softmax, balanced KL
with free nats), averaged over all rows of a batch of ``batch_size`` sequences; Adam, no gradient
clipping, bf16 autocast.

Rows. An item is a window of ``sequence_length + 1`` consecutive frames of one segment (it never
crosses a ``done``) turned into ``sequence_length`` rows: the window's first frame only supplies
the previous command, and every later frame is one row. For the row of episode frame ``t``::

    obs        = frame t                                (RGB, resized, minus 0.5)
    action     = commands[t] - commands[t-1]
    reward     = rewards[t]
    is_first   = 1 on the first row of the item, else 0

The world model consumes ``commands[t]`` with frame ``t``, so the action of a row is the command
change that produced its frame: the convention of the environment, the actor's rollout and online
RSSM finetuning. Every row's action is a real command difference, none is a padded zero (the
first row's action is ignored anyway: ``is_first`` resets the state and zeroes it).

Budget: ``epochs`` passes over the windows, ``len(windows) // batch_size`` optimizer steps each
(remainder dropped), windows reshuffled every epoch by a permutation that depends only on
``(seed, epoch)``. ``batch_size`` is global; with ``N`` processes each takes ``batch_size / N``
sequences (windows) of every global batch, so the data order does not depend on ``N``.

Frames are resized on the fly with the environment's op, optionally from a RAM cache
(``data.cache``). Every process builds its own cache, so under torchrun the host RAM it needs is
multiplied by the number of processes per node; the run stops with an error before building it
when a node does not have that memory. A float32 cache holds exactly the streamed frames; a uint8
cache re-quantises resized frames and a float16 cache rounds them (:func:`frame_values`).

Run directory (a relative path is relative to the working directory; the absolute path is
logged at startup)::

    config.yaml   the resolved config of the last launch
    latest.pt     resumable state (weights, optimizer, counters, RNG)
    rssm_e{N}/    pretrained dir after epoch N: model.safetensors + config.json (architecture,
                  preprocessing, action convention, data pool, provenance)
    tb/           TensorBoard events

Resuming. A launch continues from ``latest.pt`` when it exists; a finished run returns before
reading any data. The model, loss, optimizer, data pool and frame values must match the
checkpoint (:data:`RESUME_MUTABLE` lists the keys that may change). SIGTERM or SIGUSR1 makes
every rank save ``latest.pt`` within ``train.log_every`` steps and exit; relaunching the same
command resumes. torchrun forwards SIGTERM to its workers but not SIGUSR1.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset

import coupled_wm
from coupled_wm.common.checkpoint import atomic_save, load_checkpoint, strip_compile_prefix
from coupled_wm.common.config import load_config, save_config
from coupled_wm.common.distributed import DistInfo, barrier, cleanup, detect_launch, init_distributed
from coupled_wm.common.logging import ExperimentLogger, setup_logging
from coupled_wm.common.system import SignalSafeDataLoader, StopSignals, available_memory, warn_if_compile_unavailable
from coupled_wm.data.h5_dataset import FileInfo, HDF5SequenceDataset, list_h5_files, scan_file
from coupled_wm.rssm.rssm import (
    ARCH_KEYS,
    RELATIVE_ACTION_DEFINITION,
    RSSM,
    MSEDistribution,
    TwoHotEncodingDistribution,
    build_rssm,
    read_rssm_checkpoint,
    reconstruction_loss,
    rssm_config,
    save_rssm,
    unwrap_rssm,
)

log = logging.getLogger("coupled_wm.rssm.pretrain")

CHECKPOINT_FORMAT = "coupled_wm.rssm.pretrain/2"
LOSS_NAMES = ("rec_loss", "kl", "state_loss", "reward_loss", "observation_loss")
CACHE_BYTES = {"uint8": 1, "float16": 2, "float32": 4}
FRAME_VALUES = ("exact", "uint8-requantised", "float16-rounded")

DEFAULTS: dict[str, Any] = {
    "run_dir": None,
    "seed": 42,
    "init_from": None,
    "data": {
        "dirs": None,
        "include_globs": ["*.h5"],
        "exclude_globs": [],
        "min_chunk_index": None,
        "file_list": None,
        "color_order": "bgr",
        "reward_key": "rewards",
        "reward_remap": None,
        "reward_override_dir": None,
        "cache": False,
        "cache_dtype": "float32",
        "cache_workers": None,
        "num_workers": 4,
    },
    "model": {},
    "optim": {
        "lr": 1e-4,
        "eps": 1e-8,
    },
    "train": {
        "epochs": 10,
        "max_steps": None,
        "batch_size": 16,
        "sequence_length": 64,
        "kl_dynamic": 0.5,
        "kl_representation": 0.1,
        "kl_free_nats": 1.0,
        "kl_regularizer": 1.0,
        "precision": "bf16",
        "compile": False,
        "compile_mode": "reduce-overhead",
        "log_every": 10,
        "print_every": 500,
        "checkpoint_every_steps": 5000,
        "save_every_epochs": 1,
    },
    "validation": {
        "holdout_every": 0,
        "num_batches": 50,
        "every_steps": 5000,
    },
    "wandb": {"enabled": False},
}
_BLOCKS = ("data", "optim", "train", "validation")

# Keys a resumed run may change. Everything else (model, loss, optimizer, data selection and
# preprocessing, batch shape, seed) must match the checkpoint. The file pool is compared
# separately (so the data may move to another directory), and so are the frame values: the cache
# settings may only change to one that gives the same frames (see frame_values).
RESUME_MUTABLE = {
    "run_dir", "init_from",
    "data.dirs", "data.file_list", "data.cache", "data.cache_dtype", "data.cache_workers", "data.num_workers",
    "train.epochs", "train.max_steps", "train.compile", "train.compile_mode",
    "train.log_every", "train.print_every", "train.checkpoint_every_steps", "train.save_every_epochs",
    "validation.num_batches", "validation.every_steps",
    "wandb",
}


# --------------------------------------------------------------------------- config


def merge_defaults(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``cfg`` with every missing key of :data:`DEFAULTS` filled in; unknown keys raise."""
    cfg = copy.deepcopy(dict(cfg))
    unknown = sorted(set(cfg) - set(DEFAULTS))
    if unknown:
        raise KeyError(f"unknown top-level config keys {unknown}; expected {sorted(DEFAULTS)}")
    out: dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        value = cfg.get(key, copy.deepcopy(default))
        if key in _BLOCKS:
            value = dict(value or {})
            bad = sorted(set(value) - set(default))
            if bad:
                raise KeyError(f"unknown keys in '{key}': {bad}; expected {sorted(default)}")
            value = {**copy.deepcopy(default), **value}
        out[key] = value
    return out


def resolve_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fill defaults into ``cfg`` (:func:`merge_defaults`), validate it and resolve the architecture."""
    out = merge_defaults(cfg)
    if not out["run_dir"]:
        raise ValueError("run_dir is required")
    data, train, val = out["data"], out["train"], out["validation"]
    if data["dirs"] is None and data["file_list"] is None:
        raise ValueError("data.dirs or data.file_list is required")
    if data["reward_key"] is None:
        raise ValueError("data.reward_key is required: the RSSM reward head is trained on the per-frame reward")
    if data["cache_dtype"] not in CACHE_BYTES:
        raise ValueError(f"data.cache_dtype must be uint8, float16 or float32, got {data['cache_dtype']!r}")

    model = dict(out["model"] or {})
    bad = sorted(set(model) - set(ARCH_KEYS))
    if bad:
        raise KeyError(f"unknown keys in 'model': {bad}; architecture keys are {list(ARCH_KEYS)}")
    out["model"] = rssm_config(model)

    if train["precision"] not in ("bf16", "fp32"):
        raise ValueError(f"train.precision must be 'bf16' or 'fp32', got {train['precision']!r}")
    for key in ("epochs", "batch_size", "sequence_length", "log_every", "print_every",
                "checkpoint_every_steps", "save_every_epochs"):
        if int(train[key]) < 1:
            raise ValueError(f"train.{key} must be >= 1, got {train[key]}")
    if train["max_steps"] is not None and int(train["max_steps"]) < 1:
        raise ValueError(f"train.max_steps must be null or >= 1, got {train['max_steps']}")
    if int(val["holdout_every"]) < 0 or int(val["holdout_every"]) == 1:
        raise ValueError("validation.holdout_every must be 0 (off) or >= 2 (hold out every k-th file)")
    if int(val["num_batches"]) < 1 or int(val["every_steps"]) < 1:
        raise ValueError("validation.num_batches and validation.every_steps must be >= 1")
    if not isinstance(out["wandb"], Mapping):
        raise ValueError("wandb must be a mapping")
    return out


def _flatten(cfg: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in cfg.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and path != "model":
            flat.update(_flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


def _frozen_items(cfg: Mapping[str, Any]) -> dict[str, Any]:
    def mutable(path: str) -> bool:
        return any(path == m or path.startswith(m + ".") for m in RESUME_MUTABLE)

    return {k: v for k, v in _flatten(cfg).items() if not mutable(k)}


def frame_values(cache: bool, cache_dtype: str, resized: bool) -> str:
    """The values of the training frames for a cache setting, one of :data:`FRAME_VALUES`.

    * ``"exact"``: the environment's op in float32: streamed frames, a float32 cache, or a uint8
      cache of frames that need no resize.
    * ``"uint8-requantised"``: resized frames rounded to multiples of 1/255 (error <= 0.5/255).
    * ``"float16-rounded"``: frames rounded to float16.
    """
    if not cache or cache_dtype == "float32" or (cache_dtype == "uint8" and not resized):
        return "exact"
    return "uint8-requantised" if cache_dtype == "uint8" else "float16-rounded"


# --------------------------------------------------------------------------- rows


def window_to_rows(item: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Turn a window of ``T + 1`` frames into ``T`` RSSM training rows (see the module docstring).

    ``item`` has ``image`` (T+1, 3, H, W), ``action`` (T+1, A) (the file's commands) and
    ``reward`` (T+1,). Returns ``obs`` (T, 3, H, W) in [0, 1], ``action`` (T, A), ``reward`` (T,)
    and ``is_first`` (T,).
    """
    commands = item["action"]
    rows = commands.shape[0] - 1
    is_first = torch.zeros(rows, dtype=torch.float32)
    is_first[0] = 1.0
    return {"obs": item["image"][1:], "action": commands[1:] - commands[:-1], "reward": item["reward"][1:],
            "is_first": is_first}


class RSSMSequenceDataset(Dataset):
    """RSSM training rows over an :class:`HDF5SequenceDataset` with ``window = sequence_length + 1``."""

    def __init__(self, windows: HDF5SequenceDataset):
        if windows.window < 2:
            raise ValueError("the underlying windows need at least 2 frames (sequence_length + 1)")
        if windows.reward_key is None:
            raise ValueError("the underlying dataset must return rewards")
        self.windows = windows
        self.sequence_length = windows.window - 1

    @property
    def action_dim(self) -> int:
        return self.windows.action_dim

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return window_to_rows(self.windows[idx])


class EpochBatchSampler:
    """Batches of a seeded per-epoch permutation of ``num_items``; yields this rank's share.

    Global batch ``k`` of epoch ``e`` is ``perm_e[k * B:(k + 1) * B]`` with ``perm_e`` depending
    only on ``(seed, e)``; rank ``r`` of ``W`` gets its ``r``-th slice of ``B / W`` items. The
    last ``num_items % B`` items of each permutation are dropped. ``set_epoch(e, start_step)``
    selects the epoch and the first batch (for resuming mid-epoch).
    """

    def __init__(self, num_items: int, batch_size: int, seed: int, rank: int = 0, world_size: int = 1):
        if batch_size % world_size:
            raise ValueError(f"batch_size={batch_size} is not divisible by the world size {world_size}")
        self.num_items = int(num_items)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.steps_per_epoch = self.num_items // self.batch_size
        if self.steps_per_epoch == 0:
            raise ValueError(f"{num_items} training windows are fewer than one batch of {batch_size}")
        self.epoch = 0
        self.start_step = 0

    def set_epoch(self, epoch: int, start_step: int = 0) -> None:
        if not 0 <= start_step <= self.steps_per_epoch:
            raise ValueError(f"start_step={start_step} outside [0, {self.steps_per_epoch}]")
        self.epoch = int(epoch)
        self.start_step = int(start_step)

    def permutation(self, epoch: int) -> Tensor:
        epoch_seed = np.random.SeedSequence([self.seed, int(epoch)]).generate_state(1, dtype=np.uint64)[0]
        return torch.randperm(self.num_items, generator=torch.Generator().manual_seed(int(epoch_seed)))

    def __len__(self) -> int:
        return self.steps_per_epoch - self.start_step

    def __iter__(self):
        perm = self.permutation(self.epoch)
        share = self.batch_size // self.world_size
        lo = self.rank * share
        for step in range(self.start_step, self.steps_per_epoch):
            batch = perm[step * self.batch_size:(step + 1) * self.batch_size]
            yield batch[lo:lo + share].tolist()


# --------------------------------------------------------------------------- data


def _relative_paths(files: Sequence[str]) -> list[str]:
    root = os.path.commonpath([os.path.dirname(f) for f in files])
    return [os.path.relpath(f, root) for f in files]


def split_holdout(files: Sequence[str], holdout_every: int) -> tuple[list[str], list[str]]:
    """``(train, validation)`` files: every ``holdout_every``-th file of the pool is held out (0 = none)."""
    if not holdout_every:
        return list(files), []
    train = [f for i, f in enumerate(files) if (i + 1) % holdout_every]
    val = [f for i, f in enumerate(files) if not (i + 1) % holdout_every]
    if not train or not val:
        raise ValueError(f"validation.holdout_every={holdout_every} leaves an empty split of {len(files)} files")
    return train, val


def select_files(cfg: Mapping[str, Any]) -> list[str]:
    """The data pool of ``cfg`` (training and held-out files) in pool order."""
    data = cfg["data"]
    return list_h5_files(
        data["dirs"],
        include_globs=data["include_globs"],
        exclude_globs=data["exclude_globs"],
        min_chunk_index=data["min_chunk_index"],
        file_list=data["file_list"],
    )


def make_dataset(files: Sequence[str], cfg: Mapping[str, Any]) -> RSSMSequenceDataset:
    data, train = cfg["data"], cfg["train"]
    windows = HDF5SequenceDataset(
        files,
        window=int(train["sequence_length"]) + 1,
        stride=1,
        image_size=cfg["model"]["image_size"],
        color_order=data["color_order"],
        load_proprio=False,
        reward_key=data["reward_key"],
        reward_remap=data["reward_remap"],
        reward_override_dir=data["reward_override_dir"],
        cache=bool(data["cache"]),
        cache_dtype=data["cache_dtype"],
        cache_workers=data["cache_workers"],
        action_dim=cfg["model"]["action_size"],  # checked before the RAM cache is built
    )
    return RSSMSequenceDataset(windows)


def build_datasets(
    cfg: Mapping[str, Any], files: Sequence[str] | None = None, with_validation: bool = True
) -> tuple[RSSMSequenceDataset, RSSMSequenceDataset | None, dict]:
    """Train dataset, held-out dataset and the data-pool record for ``config.json``.

    ``files`` is the pool (default :func:`select_files`). The held-out dataset is ``None`` when
    there is no held-out file or ``with_validation`` is false (ranks that do not validate).
    """
    data = cfg["data"]
    files = select_files(cfg) if files is None else list(files)
    train_files, val_files = split_holdout(files, int(cfg["validation"]["holdout_every"]))
    train_ds = make_dataset(train_files, cfg)
    val_ds = make_dataset(val_files, cfg) if val_files and with_validation else None

    rel = _relative_paths(files)
    rel_of = dict(zip(files, rel))
    summary = train_ds.windows.summary
    source_sizes = sorted({tuple(info.image_hw) for info in train_ds.windows.file_infos})
    pool = {
        "num_files": len(train_files),
        "num_frames": summary["frames"],
        "num_segments": summary["segments"],
        "segments_shorter_than_window": summary["segments_shorter_than_window"],
        "window_frames": summary["window"],
        "num_windows": summary["windows"],
        "source_image_sizes": [list(hw) for hw in source_sizes],  # (H, W) of the stored training frames
        "selection": {
            "include_globs": list(data["include_globs"]),
            "exclude_globs": list(data["exclude_globs"]),
            "min_chunk_index": data["min_chunk_index"],
            "file_list": data["file_list"] is not None,
            "holdout_every": int(cfg["validation"]["holdout_every"]),
        },
        "files": [rel_of[f] for f in train_files],
        "validation_files": [rel_of[f] for f in val_files],
    }
    if val_ds is not None:
        pool["validation_windows"] = len(val_ds)
    return train_ds, val_ds, pool


# --------------------------------------------------------------------------- host memory


def scan_images(files: Sequence[str], color_order: str) -> list[FileInfo]:
    """Frame counts and stored image sizes of ``files`` from their metadata (no frame is read)."""
    return [scan_file(f, color_order=color_order, load_proprio=False, reward_key=None) for f in files]


def _all_gather(obj: Any, dist_info: DistInfo) -> list[Any]:
    if dist_info.world_size == 1:
        return [obj]
    out: list[Any] = [None] * dist_info.world_size
    dist.all_gather_object(out, obj)
    return out


def check_cache_memory(cfg: Mapping[str, Any], train_frames: int, val_frames: int, dist_info: DistInfo) -> None:
    """Raise ``MemoryError`` on every rank if the RAM frame caches would not fit on some node.

    Each process caches the ``train_frames`` training frames; rank 0 also caches the
    ``val_frames`` held-out frames. Ranks are grouped by host name, and a node's caches are
    compared with the smallest :func:`available_memory` its ranks report. Without ``data.cache``
    this does nothing.
    """
    data = cfg["data"]
    if not data["cache"]:
        return
    size = int(cfg["model"]["image_size"])
    per_frame = 3 * size * size * CACHE_BYTES[data["cache_dtype"]]
    host = socket.gethostname()
    entries = _all_gather((host, available_memory(), train_frames, val_frames), dist_info)
    host0, _, n_train, n_val = entries[0]  # every rank decides on the same numbers
    train_bytes, val_bytes = n_train * per_frame, n_val * per_frame
    short = None
    for node in dict.fromkeys(e[0] for e in entries):
        ranks = [e for e in entries if e[0] == node]
        need = train_bytes * len(ranks) + (val_bytes if node == host0 else 0)
        known = [e[1] for e in ranks if e[1] is not None]
        avail = min(known) if known else None
        if node == host:
            log.info("RAM frame cache: %.1f GB per process, %.1f GB on this node for %d process(es); %s",
                     train_bytes / 1e9, need / 1e9, len(ranks),
                     "available memory unknown" if avail is None else f"{avail / 1e9:.1f} GB available")
        if short is None and avail is not None and need > avail:
            short = (node, need, avail, len(ranks))
    if short is not None:
        node, need, avail, n = short
        raise MemoryError(
            f"the RAM frame caches need {need / 1e9:.1f} GB on node {node} ({n} process(es) x "
            f"{train_bytes / 1e9:.1f} GB of {data['cache_dtype']} {size}x{size} frames"
            f"{', plus the held-out frames on rank 0' if node == host0 and val_bytes else ''}) but only "
            f"{avail / 1e9:.1f} GB is available (MemAvailable and the cgroup memory limits). Every process "
            "holds its own cache: set data.cache=false to stream the same frames from disk, or run fewer "
            "processes per node."
        )


# --------------------------------------------------------------------------- loss


def rssm_loss(
    model: torch.nn.Module,
    obs: Tensor,
    actions: Tensor,
    rewards: Tensor,
    is_first: Tensor,
    kl_dynamic: float = 0.5,
    kl_representation: float = 0.1,
    kl_free_nats: float = 1.0,
    kl_regularizer: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Unroll ``model`` and return ``(loss, kl, kl_loss, reward_loss, observation_loss)``.

    Shapes: ``obs (T, B, C, H, W)`` in [-0.5, 0.5], ``actions (T, B, A)``, ``rewards (T, B, 1)``,
    ``is_first (T, B, 1)``. ``model`` may be wrapped in DDP and have compiled submodules.
    """
    core = unwrap_rssm(model)
    T, B = obs.shape[:2]
    rec, reward_logits, priors_logits, posteriors_logits = model(obs, actions, is_first)
    S, D = core.stochastic_size, core.discrete
    return reconstruction_loss(
        MSEDistribution(rec, dims=3),
        obs,
        TwoHotEncodingDistribution(reward_logits, dims=1, fp32_log_prob=True),
        rewards,
        priors_logits.view(T, B, S, D),
        posteriors_logits.view(T, B, S, D),
        kl_dynamic,
        kl_representation,
        kl_free_nats,
        kl_regularizer,
    )


def batch_to_device(batch: Mapping[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collated ``(B, T, ...)`` rows -> time-major ``(obs - 0.5, actions, rewards (T,B,1), is_first (T,B,1))``."""
    obs = batch["obs"].to(device, non_blocking=True).permute(1, 0, 2, 3, 4) - 0.5
    actions = batch["action"].to(device, non_blocking=True).transpose(0, 1)
    rewards = batch["reward"].to(device, non_blocking=True).transpose(0, 1).unsqueeze(-1)
    is_first = batch["is_first"].to(device, non_blocking=True).transpose(0, 1).unsqueeze(-1)
    return obs, actions, rewards, is_first


def _autocast(device: torch.device, precision: str):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def compile_rssm(model: RSSM, mode: str) -> None:
    """Compile the image encoder/decoder, reward head and the per-step ``dynamic`` in place (the
    paper's pretraining compiled these four, with ``mode="reduce-overhead"``)."""
    model.encoder = torch.compile(model.encoder, mode=mode)
    model.decoder = torch.compile(model.decoder, mode=mode)
    model.reward_model = torch.compile(model.reward_model, mode=mode)
    model.dynamic = torch.compile(model.dynamic, mode=mode)


def begin_compiled_step(compiled: bool) -> None:
    """Start a CUDA-graph generation; called before every training step and validation batch.

    A compiled call's output lives in CUDA-graph memory that the next generation reuses (modes
    with CUDA graphs, such as ``reduce-overhead``). A training step stays in one generation while
    its backward is pending, but under ``no_grad`` each compiled call would start a new one, so
    validation would read overwritten outputs (see :meth:`RSSM.forward`). Marking every step and
    every validation batch gives one rule for both: outputs are valid until the next mark.
    """
    if compiled:
        torch.compiler.cudagraph_mark_step_begin()


# --------------------------------------------------------------------------- checkpoints


def _rng_state(device: torch.device) -> dict[str, Tensor]:
    state = {"torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _set_rng_state(state: Mapping[str, Tensor], device: torch.device) -> None:
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


def _check_resume_config(ckpt: Mapping[str, Any], cfg: Mapping[str, Any], path: Path) -> None:
    """The checks of a resume that need no data: format, architecture and frozen config keys."""
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not an RSSM pretraining checkpoint of this version (format {ckpt.get('format')!r}, "
                         f"expected {CHECKPOINT_FORMAT!r})")
    arch = cfg["model"]
    if ckpt["rssm_config"] != arch:
        diff = {k: (ckpt["rssm_config"].get(k), v) for k, v in arch.items() if ckpt["rssm_config"].get(k) != v}
        raise ValueError(f"{path}: architecture differs from the config, (checkpoint, config): {diff}")
    old, new = _frozen_items(ckpt["config"]), _frozen_items(cfg)
    diff = {k: (old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}
    if diff:
        raise ValueError(
            f"{path}: the config differs from the checkpoint's in keys a resumed run may not change, "
            f"(checkpoint, config): {diff}. Use a new run_dir to start a different run."
        )


def _check_resume_frames(ckpt: Mapping[str, Any], cfg: Mapping[str, Any], frames: str, path: Path) -> None:
    """A resumed run must train on the frame values it started with (see :func:`frame_values`)."""
    if ckpt["frame_values"] != frames:
        data = cfg["data"]
        raise ValueError(
            f"{path}: the run trained on {ckpt['frame_values']!r} frames, but data.cache={data['cache']} with "
            f"data.cache_dtype={data['cache_dtype']!r} gives {frames!r} frames. Resume with a cache setting that "
            "gives the run's frames (exact frames: no cache, a float32 cache, or a uint8 cache of frames that "
            "need no resize) or start a new run_dir."
        )


def _check_resume_pool(ckpt: Mapping[str, Any], pool: Mapping[str, Any], path: Path) -> None:
    """A resumed run must train on the same files, frames and windows (the files may have moved)."""
    old_pool = ckpt["data"]
    for key in ("files", "validation_files", "num_frames", "num_windows"):
        if old_pool[key] != pool[key]:
            raise ValueError(f"{path}: the data pool differs from the checkpoint's ({key})")


def _metadata(cfg: Mapping[str, Any], pool: Mapping[str, Any], frames: str, prov: Mapping[str, Any]) -> dict[str, Any]:
    data, model = cfg["data"], cfg["model"]
    remap = data["reward_remap"]
    return {
        "preprocessing": {
            "image_size": [model["image_size"], model["image_size"]],
            "image_format": "float RGB (T, 3, H, W) = frame / 255 - 0.5, in [-0.5, 0.5]",
            "resize": "torch.nn.functional.interpolate(mode='bilinear', align_corners=False), no antialias, "
                      "on float [0, 1] frames",
            "frame_values": frames,
            "h5_image_key": "images",
            "h5_color_order": data["color_order"],
            "h5_action_key": "commands",
            "h5_reward_key": data["reward_key"],
            "reward_remap": [[float(k), float(v)] for k, v in remap.items()] if remap else None,
            "reward_override_dir": Path(data["reward_override_dir"]).name if data["reward_override_dir"] else None,
        },
        "action": {
            "mode": "relative",
            "dim": model["action_size"],
            "definition": RELATIVE_ACTION_DEFINITION,
            "units": "raw command units, no normalisation",
            "alignment": "row t = (frame t, action_t, reward[t]); commands[t] conditions frame t as in the world "
                         "model and the environment; a row with is_first = 1 resets the state and ignores its action",
        },
        "data": dict(pool),
        "provenance": dict(prov),
    }


# --------------------------------------------------------------------------- training


def validation_batch_size(n_val: int, batch_size: int, world_size: int) -> int:
    """The batch of the validation loader: the per-process training batch ``batch_size / world_size``.

    Only rank 0 validates (the other ranks wait at the barrier), so with ``N`` processes it reads each held-out
    batch of ``batch_size`` windows as ``N`` consecutive batches of ``batch_size / N``: the same windows, and the same
    mean losses up to the posterior's sampling noise (the losses are means over equal-size batches; the reward term,
    a mean over its labelled rows, becomes the mean of the smaller batches' means). The compiled model then sees the
    training batch shape, whose activations fit the process's GPU, instead of an ``N`` times larger batch whose
    CUDA graphs need the memory of a one-process run. ``n_val`` (a multiple of ``batch_size``, or fewer windows than
    one batch) below ``batch_size``: one batch of ``n_val``. One process: ``batch_size``, as before.
    """
    if n_val < batch_size:
        return n_val
    if batch_size % world_size:
        raise ValueError(f"batch_size={batch_size} is not divisible by the world size {world_size}")
    return batch_size // world_size


def _validate(model: RSSM, loader: DataLoader, device: torch.device, train_cfg: Mapping[str, Any], seed: int) -> dict:
    """Mean losses over the held-out batches, with a fixed sampling seed and the training RNG untouched.

    The RSSM has no train/eval-dependent layers, so the model stays in train mode.
    """
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    totals = torch.zeros(len(LOSS_NAMES), dtype=torch.float64)
    n = 0
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(seed)
        for batch in loader:
            obs, actions, rewards, is_first = batch_to_device(batch, device)
            begin_compiled_step(bool(train_cfg["compile"]))
            with _autocast(device, train_cfg["precision"]):
                out = rssm_loss(model, obs, actions, rewards, is_first, **_loss_kwargs(train_cfg))
            totals += torch.stack([x.detach().float() for x in out]).cpu().double()
            n += 1
    return dict(zip(LOSS_NAMES, (totals / max(n, 1)).tolist()))


def _loss_kwargs(train_cfg: Mapping[str, Any]) -> dict[str, float]:
    return {k: float(train_cfg[k]) for k in ("kl_dynamic", "kl_representation", "kl_free_nats", "kl_regularizer")}


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def train(cfg: Mapping[str, Any], dist_info: DistInfo | None = None) -> Path:
    """Run or resume RSSM pretraining as described in the module docstring; returns the run dir."""
    cfg = resolve_config(cfg)
    dist_info = dist_info or init_distributed()
    rank, world, device = dist_info.rank, dist_info.world_size, dist_info.device
    data_cfg, optim_cfg, train_cfg, val_cfg = cfg["data"], cfg["optim"], cfg["train"], cfg["validation"]
    run_dir = Path(cfg["run_dir"]).absolute()
    batch_size, seq_len = int(train_cfg["batch_size"]), int(train_cfg["sequence_length"])
    epochs = int(train_cfg["epochs"])
    max_steps = train_cfg["max_steps"]
    seed = int(cfg["seed"])

    log.info("rank %d of %d (local rank %d), device %s, process group %s",
             rank, world, dist_info.local_rank, device, dist_info.distributed)
    if device.type != "cuda":
        log.warning("no CUDA device is visible: training on the CPU (in a container, make the GPUs visible, "
                    "e.g. singularity exec --nv)")

    # An existing run: its state is read (memory-mapped) before any data, and every rank must see it.
    latest = run_dir / "latest.pt"
    ckpt = load_checkpoint(latest) if latest.is_file() else None
    found = None if ckpt is None else tuple((ckpt.get("state") or {}).get(k) for k in ("epoch", "step"))
    seen = _all_gather(found, dist_info)
    if any(s != seen[0] for s in seen):
        raise RuntimeError(
            f"the ranks see different runs in {run_dir} ((epoch, step) of latest.pt per rank, None = no "
            f"latest.pt: {seen}): run_dir must resolve to the same directory, on a filesystem shared by all ranks"
        )
    epoch, step, val_loss, started = 0, 0, None, _utc_now()
    if ckpt is not None:
        _check_resume_config(ckpt, cfg, latest)
        epoch, step = int(ckpt["state"]["epoch"]), int(ckpt["state"]["step"])
        spe = int(ckpt["state"]["steps_per_epoch"])
        if epoch >= epochs or (max_steps is not None and epoch * spe + step >= int(max_steps)):
            log.info("nothing to do: the run in %s is at epoch %d, step %d (epochs=%d, max_steps=%s)",
                     run_dir, epoch, step, epochs, max_steps)
            return run_dir
        log.info("resuming the run in %s", run_dir)
    else:
        log.info("new run in %s", run_dir)

    torch.manual_seed(seed + rank)
    files = select_files(cfg)
    train_files, val_files = split_holdout(files, int(val_cfg["holdout_every"]))
    # From the files' metadata, before any frame is read or cached.
    train_images = scan_images(train_files, data_cfg["color_order"])
    dims = sorted({info.action_dim for info in train_images})
    if dims != [int(cfg["model"]["action_size"])]:
        raise ValueError(f"the data's commands have {dims} dims but model.action_size is {cfg['model']['action_size']}")
    image_hw = (int(cfg["model"]["image_size"]),) * 2
    frames = frame_values(bool(data_cfg["cache"]), data_cfg["cache_dtype"],
                          resized=any(info.image_hw != image_hw for info in train_images))
    if ckpt is not None:
        _check_resume_frames(ckpt, cfg, frames, latest)
    check_cache_memory(cfg, sum(i.n_frames for i in train_images),
                       sum(i.n_frames for i in scan_images(val_files, data_cfg["color_order"])), dist_info)

    train_ds, val_ds, pool = build_datasets(cfg, files, with_validation=rank == 0)
    validate = bool(val_files)  # the same on every rank; only rank 0 holds the held-out data
    sampler = EpochBatchSampler(len(train_ds), batch_size, seed, rank, world)
    spe = sampler.steps_per_epoch
    log.info("%d training windows of %d rows from %d files (%s frames): %d steps per epoch of %d sequences "
             "(%d per rank)", len(train_ds), seq_len, pool["num_files"], frames, spe, batch_size, batch_size // world)

    model = build_rssm(cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(optim_cfg["lr"]), eps=float(optim_cfg["eps"]))
    log.info("RSSM: %s parameters, architecture %s", f"{sum(p.numel() for p in model.parameters()):,}", model.config)

    if ckpt is not None:
        _check_resume_pool(ckpt, pool, latest)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        val_loss = ckpt.get("val_loss")
        started = ckpt["run_started_utc"]
        if len(ckpt["rng"]) == world:
            _set_rng_state(ckpt["rng"][rank], device)
        else:
            log.warning("resuming with %d processes instead of %d: sampling RNG reseeded", world, len(ckpt["rng"]))
            torch.manual_seed(seed + rank + 1_000_003 * (epoch * spe + step))
        log.info("resumed from %s at epoch %d, step %d (global step %d)", latest, epoch, step, epoch * spe + step)
        del ckpt
    elif cfg["init_from"]:
        state_dict, _ = read_rssm_checkpoint(cfg["init_from"], model.config)
        model.load_state_dict(state_dict, strict=True)
        log.info("initialised the weights from %s", cfg["init_from"])

    if rank == 0:
        save_config(cfg, run_dir / "config.yaml")
    # Losses are logged at the 0-based index of their step and validation at the steps done, so a
    # resumed launch purges from the step after its checkpoint (the checkpoint step's validation
    # was logged before the save).
    purge_step = epoch * spe + step + 1 if (epoch or step) else 0
    logger = ExperimentLogger(run_dir / "tb", rank=rank, config=cfg, wandb_cfg=cfg["wandb"], purge_step=purge_step)

    compiled = bool(train_cfg["compile"])
    if compiled:
        warn_if_compile_unavailable("train.compile", "--set train.compile=false", device)
        compile_rssm(model, train_cfg["compile_mode"])
        if dist_info.distributed:
            # Dynamo's DDP optimizer would split the per-step graph at DDP's bucket boundaries: the unroll calls
            # it once per time step, so CUDA-graph trees then record T x (buckets) graphs, and their alias check
            # over every live recorded output made the first step of a 4-GPU run take over 50 min (torch 2.9.1).
            # The parameters are reused at every step, so their gradients are ready only at the end of the
            # backward and the split bought no overlap anyway.
            torch._dynamo.config.optimize_ddp = False
            log.info("compile under DDP: torch._dynamo.config.optimize_ddp = False (one graph per call, as on 1 GPU)")
    net: torch.nn.Module = model
    if dist_info.distributed:
        net = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    num_workers = int(data_cfg["num_workers"])
    loader = SignalSafeDataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = None
    if val_ds is not None:
        # whole batches only, so a compiled model sees one batch shape
        n_val = min(len(val_ds), int(val_cfg["num_batches"]) * batch_size) // batch_size * batch_size or len(val_ds)
        order = torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(seed))[:n_val].tolist()
        val_batch = validation_batch_size(n_val, batch_size, world)
        log.info("validation: %d held-out windows in batches of %d on rank 0", n_val, val_batch)
        val_loader = SignalSafeDataLoader(Subset(val_ds, order), batch_size=val_batch, shuffle=False,
                                          num_workers=min(num_workers, 2), pin_memory=device.type == "cuda")

    loss_kw = _loss_kwargs(train_cfg)

    def provenance(done_epochs: int, global_step: int) -> dict[str, Any]:
        group = optimizer.param_groups[0]
        return {
            "trainer": "coupled_wm.rssm.pretrain",
            "coupled_wm_version": coupled_wm.__version__,
            "torch_version": torch.__version__,
            "run_started_utc": started,
            "written_utc": _utc_now(),
            "epoch": done_epochs,
            "epochs": epochs,
            "global_step": global_step,
            "steps_per_epoch": spe,
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "world_size": world,
            "seed": seed,
            "optimizer": {"name": "Adam", "lr": group["lr"], "betas": list(group["betas"]), "eps": group["eps"],
                          "weight_decay": group["weight_decay"], "grad_clip": None},
            "loss": {**loss_kw, "observation": "sum of squared errors over (C, H, W)",
                     "reward": "two-hot NLL over symlog bins on [-20, 20], fp32 log-softmax",
                     "reduction": "mean over all (T, B) rows"},
            "precision": "bf16 autocast" if train_cfg["precision"] == "bf16" else "fp32",
            "compiled": compiled,
            "init_from": cfg["init_from"],
            "validation_loss": val_loss,
        }

    def save_latest(done_epochs: int, step_in_epoch: int) -> None:
        rng = _all_gather(_rng_state(device), dist_info)
        if rank == 0:
            atomic_save(
                {
                    "format": CHECKPOINT_FORMAT,
                    "model": strip_compile_prefix(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "state": {"epoch": done_epochs, "step": step_in_epoch,
                              "global_step": done_epochs * spe + step_in_epoch, "steps_per_epoch": spe},
                    "rng": rng,
                    "config": cfg,
                    "rssm_config": model.config,
                    "data": pool,
                    "frame_values": frames,
                    "run_started_utc": started,
                    "val_loss": val_loss,
                },
                latest,
            )
            log.info("saved %s (epoch %d, step %d)", latest, done_epochs, step_in_epoch)
        barrier()

    def run_validation(global_step: int) -> None:
        nonlocal val_loss
        if val_loader is not None:
            values = _validate(model, val_loader, device, train_cfg, seed)
            val_loss = values["rec_loss"]
            logger.scalars(values, global_step, prefix="val/")
            log.info("validation at step %d: %s", global_step, ", ".join(f"{k} {v:.4f}" for k, v in values.items()))
        barrier()

    signals = StopSignals()

    def stop_requested() -> bool:
        """Whether any rank received a stop signal (a collective: every rank calls it at the same step)."""
        if world == 1:
            return signals.poll() is not None
        flag = torch.tensor([float(signals.poll() is not None)], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item() > 0)

    def stopped(done_epochs: int, step_in_epoch: int) -> Path:
        log.warning("stop signal received: saved epoch %d, step %d to %s; relaunch the same command to resume",
                    done_epochs, step_in_epoch, latest)
        return run_dir

    frames_per_step = batch_size * seq_len
    log_every, print_every = int(train_cfg["log_every"]), int(train_cfg["print_every"])
    last_log: tuple[int, float] | None = None
    last_print = (epoch * spe + step, time.time())
    try:
        with signals:
            while epoch < epochs:
                sampler.set_epoch(epoch, step)
                for batch in loader:
                    global_step = epoch * spe + step  # index of this optimizer step
                    obs, actions, rewards, is_first = batch_to_device(batch, device)
                    optimizer.zero_grad()
                    begin_compiled_step(compiled)
                    with _autocast(device, train_cfg["precision"]):
                        losses = rssm_loss(net, obs, actions, rewards, is_first, **loss_kw)
                    losses[0].backward()
                    optimizer.step()
                    step += 1

                    if global_step % log_every == 0:
                        values = torch.stack([x.detach().float() for x in losses])
                        if dist_info.distributed:
                            dist.all_reduce(values)
                            values /= world
                        values = dict(zip(LOSS_NAMES, values.tolist()))
                        if not all(np.isfinite(v) for v in values.values()):
                            raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}, step {global_step}: {values}")
                        logger.scalars(values, global_step, prefix="losses/")
                        now = time.time()
                        if last_log is not None:
                            logger.scalar("training/fps", frames_per_step * (global_step - last_log[0]) / (now - last_log[1]),
                                          global_step)
                        last_log = (global_step, now)
                        if global_step - last_print[0] >= print_every:
                            rate = (global_step - last_print[0]) / (now - last_print[1])
                            remaining = epochs * spe - global_step
                            if max_steps is not None:
                                remaining = min(remaining, int(max_steps) - global_step)
                            log.info("epoch %d step %d/%d (global %d): %s | %.2f steps/s, %.0f frames/s, eta %.1f h",
                                     epoch + 1, step, spe, global_step,
                                     ", ".join(f"{k} {v:.4f}" for k, v in values.items()),
                                     rate, rate * frames_per_step, remaining / max(rate, 1e-9) / 3600)
                            last_print = (global_step, now)

                    done = global_step + 1  # optimizer steps completed
                    if step < spe:  # the end of an epoch is handled below
                        if validate and done % int(val_cfg["every_steps"]) == 0:
                            run_validation(done)
                        stop = done % log_every == 0 and stop_requested()
                        if max_steps is not None and done >= int(max_steps):
                            save_latest(epoch, step)
                            log.info("stopped at max_steps=%d; rerun with a larger max_steps (or null) to continue", done)
                            return run_dir
                        if stop or done % int(train_cfg["checkpoint_every_steps"]) == 0:
                            save_latest(epoch, step)
                        if stop:
                            return stopped(epoch, step)

                if step != spe:
                    raise RuntimeError(f"epoch {epoch + 1} ended after {step} of {spe} steps")
                epoch, step = epoch + 1, 0
                if validate:
                    run_validation(epoch * spe)
                if epoch % int(train_cfg["save_every_epochs"]) == 0 or epoch == epochs:
                    if rank == 0:
                        out = save_rssm(run_dir / f"rssm_e{epoch}", model,
                                        _metadata(cfg, pool, frames, provenance(epoch, epoch * spe)))
                        log.info("epoch %d done: wrote %s", epoch, out)
                    barrier()
                save_latest(epoch, 0)
                if max_steps is not None and epoch * spe >= int(max_steps):
                    log.info("stopped at max_steps=%d; rerun with a larger max_steps (or null) to continue", epoch * spe)
                    return run_dir
                if epoch < epochs and stop_requested():
                    return stopped(epoch, 0)
            log.info("training complete: %d epochs, %d steps", epochs, epochs * spe)
            return run_dir
    finally:
        logger.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pretrain the RSSM (local world model) on HDF5 episode data.")
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/rssm/g1_rssm_pretrain.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.epochs=20 (repeatable)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides, DEFAULTS)  # --set reaches keys left at their default
    setup_logging(detect_launch()[1])  # before init_distributed, so that its launch summary is printed
    dist_info = init_distributed()
    try:
        train(cfg, dist_info)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
