# SPDX-License-Identifier: Apache-2.0
"""The G1 world-model environment: the frozen DreamerV4 tokenizer + dynamics as a batched env.

The global model of the paper, used forward-only. Each episode is seeded from a window of
``init_context_len`` real frames; the dynamics then generates one latent frame per step,
conditioned on an absolute joint command, and the decoder renders it.

Actions are relative
--------------------
``step(a)`` takes a per-step command delta ``a`` ``[B, A]`` (env units: rad, grip fraction).
The env integrates it onto the previous absolute command, clamps the result to the symmetric
envelope ``[-action_scale, action_scale]`` and the grip dims to ``grip_range``, and feeds the
clamped absolute command ``c_t`` to the dynamics::

    c_t = clamp(c_{t-1} + a_t)        c_0 = commands[init_context_len - 1] of the seed window

The integration runs in fp32; ``c_t`` is cast to bf16 only at the dynamics input. The clamped
command becomes the next anchor, so the integrator never winds up. ``info["applied_action"]``
(``c_t - c_{t-1}``) and ``info["clamped"]`` report where a clamp changed the action; they are
diagnostics, the caller's action is what a learner should record.

Action/frame alignment: ``c_t`` is consumed together with the frame it generates (the
dynamics' convention: ``action[t]`` conditions frame ``t``), and the step reward is the reward
head's prediction for that same frame.

``drop_action_indices`` names dims the policy does not control (the G1 base velocity
commands 0-2): the caller sends zeros there. ``dropped_dims_reset`` sets what they hold:
``hold`` keeps the seed window's last command for the whole episode (paper behaviour), ``zero``
sets them to 0 in the anchor at reset (the context frames keep their recorded commands; the
first generated frame sees the stop), ``skip_nonzero`` never seeds from a window whose anchor
has a nonzero dropped dim (the pool's windows are enumerated as in the other modes, then
filtered; construction logs how many are excluded).

One generation step
-------------------
``num_inference_steps`` (power of two) shortcut/Euler passes from pure noise with
``update_cache=False``, then one cache-writing pass: the result, re-noised to noise index
``seq_cor_tau_idx``, goes through the model once with ``update_cache=True`` so that the KV
cache holds the new frame; that pass's prediction is discarded and the decoder renders the
Euler result. The context is prefilled the same way (latents noised to ``seq_cor_tau_idx``). Noise index ``k``
means signal level ``tau = 1 - (k + 1) / num_tau_levels``. The KV cache keeps the last
``kv_cache_len`` frames, so a read-only pass attends to ``kv_cache_len + 1`` frames.

Constants and where they come from
----------------------------------
The command clamp (``action_scale``, ``grip_dims``, ``grip_range``) and the generation
constants (``init_context_len``, ``kv_cache_len``, ``num_inference_steps``,
``seq_cor_tau_idx``, ``reward_mtp_index``) come from the env config, else from the dynamics
``config.json`` (its ``action`` / ``inference`` blocks), else from the paper's G1 values. The
seed files' channel order comes from ``data.color_order``, else from the tokenizer's
``preprocessing.h5_color_order``. Construction logs every value with its source and warns when
a G1 default fills a value the model's config does not record (models trained on other data
should record their own). ``d_min_idx`` is always ``log2(num_tau_levels)``;
``seq_cor_tau_idx`` indexes the noise grid, so its G1 default is refused for a model whose
grid is not the G1 model's 128 levels. Seed windows whose anchor command lies more than
:data:`ANCHOR_TOLERANCE` outside the clamp are counted in a warning (an envelope that does not
describe the seed data). An explicit ``data.color_order`` that differs from the
tokenizer's recorded order is allowed (seed files may be stored differently from the
tokenizer's training files) but warned about. When the dynamics config records the longest
clip of its training lineage (``training.lineage_max_clip_len``, or the clips of a stage
trained from scratch), a context prefill or read-only pass longer than that warns: it would
attend over temporal distances (RoPE positions) the model never trained on. So does a context
encoded with another temporal window than the one the dynamics' training latents were encoded
with (``tokenizer.encoder_context_len`` in its config).

Episodes, resets, outputs
-------------------------
All envs share one clock: every ``episode_length`` steps the whole batch truncates (``done``
is a time limit, never a termination), ``step`` resets every env internally and returns the
new first frames; the last generated frames are in ``info["obs_before_reset"]``. ``reset``
also resets all envs. Observations are decoder frames bilinearly resized to ``img_size``
(``align_corners=False``, no antialias), RGB, float32. The decoder's head is linear, so frames
are not clamped to [0, 1] (values slightly outside occur). Observations and rewards are fp32:
on CUDA, bf16 autocast already computes the resize and the reward softmax in fp32; on CPU the
bf16 results are cast.

Replaying recorded commands
---------------------------
:meth:`step` is :meth:`step_command` (one generation step conditioned on an absolute command,
taken as given) after the integration and the clamp, plus the reset at the time limit;
:meth:`reset_all` is :meth:`reset_from_context` (the context prefill) on a seed-pool window.
Calling the two directly replays a recorded episode: prefill its first ``init_context_len``
frames, then generate each further frame from its recorded command, by exactly the computation
the RL rollouts use (``scripts/compare_rollouts.py``).

Seeding
-------
Seed windows come from :class:`~coupled_wm.data.h5_dataset.HDF5SequenceDataset` (the shared
loader, same window enumeration). They are drawn in passes over the pool. Each pass makes two
int64 draws from the global torch RNG (or from a private generator when ``loader_seed`` is
set), whatever ``data.num_workers`` is: the two draws the paper's DataLoader (``shuffle=True``,
``drop_last=True``, 4 workers) made when its iterator was created. The pass visits a random
permutation in batches of ``num_envs`` (the remainder is dropped). The first pass is drawn at
construction and its first batch seeds the first episode. With ``dropped_dims_reset`` ``hold``
or ``zero`` and ``loader_seed`` null, a given torch seed therefore selects the same windows as
in the paper's environment. ``skip_nonzero`` makes the same draws but permutes only the eligible
windows, so it selects other windows.

The global CPU RNG state after construction is not the paper's: the paper env random-initialised
its world models on CPU before loading the weights, this one builds them on the meta device. So
for the same seed, modules a caller initialises after the env (the trainer's actor and critics)
and later CPU-RNG draws differ from the paper run; the CUDA generation noise does not, nor do
the seed windows under ``hold`` or ``zero``.

:meth:`state_dict` / :meth:`load_state_dict` carry the position in the current pass (and the
private generator's state), so a resumed run continues the pass instead of replaying it.

Seed frames are moved to the device as uint8 and mapped to the exact bf16 values of
``float32(x) / 255``. Files not stored at the tokenizer's ``image_size`` are resized like the
world-model trainers resize them (:func:`~coupled_wm.data.h5_dataset.frames_to_tensor`:
bilinear, ``align_corners=False``, on float [0, 1]) and cast to bf16 on the device.

Memory (per process, paper G1 models)
-------------------------------------
About 2.9 GB of bf16 weights, plus KV caches of about 0.55 GB per env (32-frame cache:
dynamics 0.31 GB, decoder 0.24 GB), checked against the free device memory before they are
allocated. Every reset also encodes ``num_envs x init_context_len`` frames in one encoder call.
With ``compile_encoder: reduce-overhead`` that call is recorded as a CUDA graph, and PyTorch keeps
the memory the graph works in reserved for the life of the process (its CUDA-graph memory pool,
19.73 GiB at 64 envs on an H100), so that later resets rerun the graph at the same addresses;
with ``compile_encoder: false`` the call runs eagerly and its memory is freed after each reset. Seed windows travel from the loader workers as uint8, about
6.3 MB per G1 env per reset (float32, four times more, when they are resized). The trainer
adds its replay buffer on the same device unless ``train.replay_device: cpu``.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from coupled_wm.common.checkpoint import is_pretrained_dir, load_checkpoint, load_pretrained_config
from coupled_wm.common.system import SignalSafeDataLoader, stop_loader_workers, warn_if_compile_unavailable
from coupled_wm.data.h5_dataset import HDF5SequenceDataset, frames_to_tensor, list_h5_files
from coupled_wm.world_model.loading import (
    as_model_config,
    check_pairing,
    checkpoint_model_config,
    lineage_max_clip_len,
    load_dynamics,
    load_tokenizer,
)
from coupled_wm.world_model.models import (
    DYNAMICS_MODEL_TYPE,
    G1_ACTION,
    G1_DYNAMICS_ARCH,
    G1_INFERENCE,
    G1_PREPROCESSING,
    TOKENIZER_MODEL_TYPE,
)

log = logging.getLogger(__name__)

DROPPED_DIMS_RESET = ("hold", "zero", "skip_nonzero")

SEED_POOL_DEFAULTS: dict[str, Any] = {
    "dirs": None,
    "file_list": None,
    "include_globs": ["*.h5"],
    "exclude_globs": [],
    "min_chunk_index": None,
    "color_order": None,
    "num_workers": 4,
    "prefetch_factor": 2,
    "pin_memory": True,
}
"""Keys of the env's ``data`` block (the seed pool), named as in the world-model and RSSM
``data`` blocks; ``None`` color order = the tokenizer's recorded order."""


class SeedWindowDataset(HDF5SequenceDataset):
    """Seed pool items: ``index`` (window id), ``image`` (T,3,H,W) RGB, ``action`` (T,A) and,
    with ``load_proprio``, ``proprio`` (T,P) float32.

    ``image`` is uint8 when every file is stored at ``image_size`` (4x less shared and pinned
    memory than float32; :meth:`DreamerV4VecEnvG1Relative._frames_to_device` maps it to the
    exact bf16 values of ``float32(x) / 255``). Otherwise (``resize``) it is float32 in [0, 1],
    resized by :func:`~coupled_wm.data.h5_dataset.frames_to_tensor`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_sizes = sorted({tuple(int(s) for s in info.image_hw) for info in self.file_infos})
        self.resize = self.image_size is not None and self.native_sizes != [tuple(self.image_size)]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raw = self.raw_window(idx)
        frames = np.ascontiguousarray(raw["images"])
        if self.resize:
            image = frames_to_tensor(frames, "rgb", self.image_size)
        else:
            image = torch.from_numpy(frames).permute(0, 3, 1, 2)
        item = {
            "index": torch.tensor(idx, dtype=torch.int64),
            "image": image,
            "action": torch.from_numpy(raw["commands"]).float(),
        }
        if "proprio" in raw:
            item["proprio"] = torch.from_numpy(raw["proprio"]).float()
        return item

    def frame_commands(self, offset: int) -> np.ndarray:
        """``commands[start + offset]`` of every window as float32, shape ``(len(self), A)``."""
        return self.window_frame_values("commands", offset).astype(np.float32)


ANCHOR_TOLERANCE = 0.01
"""Seed-window anchors may lie this far outside the command clamp before the env warns (the G1
envelope is the data's per-dim max |command| rounded to 0.01: the paper pool exceeds it by up to
0.005)."""


def _draw_pass_seed(generator: torch.Generator | None = None) -> int:
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    return int(torch.empty((), dtype=torch.int64).random_(generator=generator).item())


def _pass_batches(n: int, batch_size: int, pass_seed: int) -> list[list[int]]:
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(pass_seed)).tolist()
    return [perm[i:i + batch_size] for i in range(0, n - batch_size + 1, batch_size)]


def draw_seed_pass(n: int, batch_size: int, generator: torch.Generator | None = None) -> list[list[int]]:
    """Batches of one random pass over ``n`` items, the last partial batch dropped.

    Consumes ``generator`` (``None`` = the global torch RNG) with the two int64 draws the
    paper's multi-worker ``DataLoader(dataset, batch_size, shuffle=True, drop_last=True,
    num_workers=4)`` made when its iterator was created: the iterator's base seed, then the
    ``RandomSampler`` seed of its permutation (a single-process DataLoader defers the second
    draw to the first ``next()``). The batches are the ones that iterator yields.
    """
    return _pass_batches(n, batch_size, _draw_pass_seed(generator))


def _int_list(name: str, values: Sequence[Any] | None, upper: int) -> tuple[int, ...]:
    values = tuple(int(v) for v in (values or ()))
    bad = [v for v in values if not 0 <= v < upper]
    if bad or len(set(values)) != len(values):
        raise ValueError(f"{name} must be distinct indices in [0, {upper}), got {list(values)}")
    return values


def _seed_pool_config(data: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"data must be a mapping (the env config's data block), got {type(data).__name__}")
    unknown = sorted(set(data) - set(SEED_POOL_DEFAULTS))
    if unknown:
        raise KeyError(f"unknown env data key(s) {unknown}; valid keys: {sorted(SEED_POOL_DEFAULTS)} (the window "
                       "is init_context_len, frames are resized to the tokenizer's image_size)")
    return {**SEED_POOL_DEFAULTS, **data}


def _check_kv_cache_fits(device: torch.device, kv_bytes: int, num_envs: int, kv_cache_len: int) -> None:
    """Raise before allocating KV caches that cannot fit the free memory of a CUDA device."""
    if device.type != "cuda":
        return
    free, _ = torch.cuda.mem_get_info(device)
    if kv_bytes > free:
        raise torch.OutOfMemoryError(
            f"the world-model KV caches of {num_envs} envs ({kv_bytes / 1e9:.2f} GB for {kv_cache_len} frames) do not "
            f"fit the {free / 1e9:.2f} GB free on {device}. The trainer runs num_actors / world_size envs per "
            "process: run on more GPUs (a smaller num_actors changes the experiment).")


class DreamerV4VecEnvG1Relative:
    """Batched relative-action G1 environment on the DreamerV4 world model (see module doc).

    Args:
        num_envs: envs in this process (the trainer passes ``num_actors // world_size``).
        data: the seed pool, with the keys of the world-model ``data`` blocks
            (:data:`SEED_POOL_DEFAULTS`): ``dirs`` / ``file_list`` / ``include_globs`` /
            ``exclude_globs`` / ``min_chunk_index`` as in
            :func:`~coupled_wm.data.h5_dataset.list_h5_files`; ``color_order`` of the h5 frames
            (``None`` = the tokenizer's ``preprocessing.h5_color_order``, ``"bgr"`` for G1);
            ``num_workers`` (0 = load in the main process; batches pass through ``/dev/shm``
            otherwise), ``prefetch_factor`` and ``pin_memory`` of the seed-window loader.
        tokenizer_ckpt / dynamics_ckpt: pretrained dirs, or training checkpoints that embed
            their ``model_config`` (see :mod:`coupled_wm.world_model.loading`).
        device: torch device for the world model.
        episode_length: steps per episode (time limit).
        img_size: side of the square observations returned to the agent.
        drop_action_indices: action dims the policy does not control.
        dropped_dims_reset: ``"hold"`` (paper), ``"zero"`` or ``"skip_nonzero"``.
        action_scale, grip_dims, grip_range: the absolute-command clamp; ``None`` = the
            dynamics ``config.json`` ``action`` block, else the paper's G1 values.
        init_context_len, kv_cache_len, num_inference_steps, seq_cor_tau_idx,
            reward_mtp_index: generation constants; ``None`` = the dynamics ``config.json``
            ``inference`` block, else the paper's G1 values 32, 32, 4, 12, 0.
        compile_encoder: ``torch.compile`` the encoder: ``False``, ``True`` (default mode) or a
            mode name; the paper used ``"reduce-overhead"`` (CUDA graphs, whose memory stays
            reserved for the whole run; numerics do not depend on the mode). Compiling needs a C compiler
            (gcc or clang, or ``CC``) and the Python headers at run time; without them use
            ``False`` (eager). The decoder and dynamics are only called through
            ``forward_step`` and run eagerly.
        loader_seed: draw seed windows from a private generator with this seed instead of the
            global torch RNG (reproducible evaluation; ``None`` = paper behaviour).
        verify_pairing: check the tokenizer sha256 recorded in the dynamics config (pretrained
            dirs only); the latent shapes are always checked.
    """

    def __init__(
        self,
        num_envs: int,
        data: Mapping[str, Any],
        tokenizer_ckpt: str | os.PathLike,
        dynamics_ckpt: str | os.PathLike,
        device: torch.device | str,
        episode_length: int,
        img_size: int = 64,
        drop_action_indices: Sequence[int] = (),
        dropped_dims_reset: str = "hold",
        action_scale: Sequence[float] | None = None,
        grip_dims: Sequence[int] | None = None,
        grip_range: Sequence[float] | None = None,
        init_context_len: int | None = None,
        kv_cache_len: int | None = None,
        num_inference_steps: int | None = None,
        seq_cor_tau_idx: int | None = None,
        reward_mtp_index: int | None = None,
        compile_encoder: bool | str = "reduce-overhead",
        loader_seed: int | None = None,
        verify_pairing: bool = True,
    ):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.episode_length = int(episode_length)
        self.img_size = int(img_size)
        if self.num_envs < 1 or self.episode_length < 1 or self.img_size < 1:
            raise ValueError("num_envs, episode_length and img_size must be >= 1")
        if dropped_dims_reset not in DROPPED_DIMS_RESET:
            raise ValueError(f"dropped_dims_reset must be one of {DROPPED_DIMS_RESET}, got {dropped_dims_reset!r}")
        self.dropped_dims_reset = dropped_dims_reset
        data = _seed_pool_config(data)

        tokenizer_cfg = self._read_model_config(tokenizer_ckpt, TOKENIZER_MODEL_TYPE)
        dynamics_cfg = self._read_model_config(dynamics_ckpt, DYNAMICS_MODEL_TYPE)
        self.tokenizer_config, self.dynamics_config = tokenizer_cfg, dynamics_cfg
        check_pairing(tokenizer_cfg, dynamics_cfg,
                      tokenizer_ckpt if verify_pairing and is_pretrained_dir(tokenizer_ckpt) else None)
        tok_arch, dyn_arch = tokenizer_cfg["architecture"], dynamics_cfg["architecture"]
        self.action_dim = self.num_actions = int(dyn_arch["action_dim"])
        self.num_tau_levels = int(dyn_arch["num_tau_levels"])
        self.d_min_idx = int(math.log2(self.num_tau_levels))
        scale, grips, grip_lim = self._resolve_constants(
            tokenizer_cfg, dynamics_cfg, data["color_order"], action_scale, grip_dims, grip_range, init_context_len,
            kv_cache_len, num_inference_steps, seq_cor_tau_idx, reward_mtp_index)
        self._check_generation_constants(dyn_arch)
        self._check_trained_window(dynamics_cfg)
        self._check_encoder_window(tok_arch, dynamics_cfg)

        scale = [float(s) for s in scale]
        if len(scale) != self.action_dim or min(scale) <= 0:
            raise ValueError(f"action_scale ({self.constant_sources['action_scale']}) needs {self.action_dim} positive "
                             f"entries (the dynamics action_dim), got {len(scale)}: {scale}")
        self.grip_dims = _int_list(f"grip_dims ({self.constant_sources['grip_dims']})", grips, self.action_dim)
        self.grip_range = tuple(float(v) for v in grip_lim)
        if len(self.grip_range) != 2 or not self.grip_range[0] < self.grip_range[1]:
            raise ValueError(f"grip_range must be [low, high] with low < high, got {list(self.grip_range)}")
        self.drop_action_indices = _int_list("drop_action_indices", drop_action_indices, self.action_dim)
        if dropped_dims_reset != "hold" and not self.drop_action_indices:
            raise ValueError(f"dropped_dims_reset={dropped_dims_reset!r} needs drop_action_indices")
        self.action_scale = torch.tensor(scale, dtype=torch.float32, device=self.device)
        low = [self.grip_range[0] if d in self.grip_dims else -math.inf for d in range(self.action_dim)]
        high = [self.grip_range[1] if d in self.grip_dims else math.inf for d in range(self.action_dim)]
        self._grip_low = torch.tensor(low, dtype=torch.float32, device=self.device)
        self._grip_high = torch.tensor(high, dtype=torch.float32, device=self.device)

        self._build_seed_pool(data, tok_arch, loader_seed)
        self._loader_iter = self._new_pass()

        # The decoder and dynamics RoPE tables cover every frame position of an episode (the
        # paper's length). The encoder only sees the context; a longer table leaves its outputs
        # unchanged.
        seq_len = self.init_context_len + self.episode_length + 10
        self.enc, self.dec, _ = load_tokenizer(tokenizer_ckpt, self.device, torch.bfloat16,
                                               runtime={"seq_len": seq_len, "mae_max_mask_prob": 0.0})
        self.dyn, _ = load_dynamics(dynamics_ckpt, self.device, torch.bfloat16, runtime={"seq_len": seq_len})
        if compile_encoder:
            warn_if_compile_unavailable(f"env.compile_encoder={compile_encoder}", "env.compile_encoder: false",
                                        self.device)
            self.enc = torch.compile(self.enc) if compile_encoder is True else torch.compile(self.enc,
                                                                                          mode=compile_encoder)
        self.kv_cache_bytes = (self.dyn.kv_cache_bytes(self.num_envs, self.kv_cache_len)
                               + self.dec.kv_cache_bytes(self.num_envs, self.kv_cache_len))
        _check_kv_cache_fits(self.device, self.kv_cache_bytes, self.num_envs, self.kv_cache_len)
        self.dyn.init_cache(self.num_envs, self.device, max_seq_len=self.kv_cache_len)
        self.dec.init_cache(self.num_envs, self.device, max_seq_len=self.kv_cache_len)
        self._noise_shape = (self.num_envs, 1, int(dyn_arch["num_latents"]), int(dyn_arch["latent_dim"]))
        self._u8_to_unit = (torch.arange(256, dtype=torch.float32) / 255.0).to(torch.bfloat16).to(self.device)

        log.info(
            "G1 env: %d envs, episode %d, KV cache %.2f GB, drop %s (%s), obs %dpx",
            self.num_envs, self.episode_length, self.kv_cache_bytes / 1e9, list(self.drop_action_indices),
            self.dropped_dims_reset, self.img_size,
        )

        self.t = 0
        self.traj_steps = 0
        self.img0: torch.Tensor | None = None
        self.last_abs_action: torch.Tensor | None = None
        self.seed_window_ids: torch.Tensor | None = None
        try:
            self.reset_all()
        except torch.OutOfMemoryError as err:
            pool = (" With mode 'reduce-overhead' the compiled encoder call is a CUDA graph whose memory stays "
                    "reserved for the whole run; compile_encoder: false runs it eagerly and frees that memory after "
                    "each reset." if compile_encoder else "")
            raise torch.OutOfMemoryError(
                f"out of device memory in the first reset: after {self.kv_cache_bytes / 1e9:.2f} GB of KV caches the "
                f"encoder encodes {self.num_envs} x {self.init_context_len} frames in one batch.{pool} The trainer "
                f"runs num_actors / world_size envs per process: more GPUs lower it. ({err})") from err

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], *, num_envs: int, device: torch.device | str,
                    episode_length: int) -> "DreamerV4VecEnvG1Relative":
        """Build from an ``env`` config block: its keys are this class's keyword arguments
        (plus an optional ``name``; ``data`` is the nested seed-pool block); the caller supplies
        ``num_envs``, ``device`` and ``episode_length``."""
        cfg = dict(cfg)
        name = cfg.pop("name", cls.__name__)
        if name != cls.__name__:
            raise ValueError(f"env config is for {name!r}, not {cls.__name__}")
        reserved = {"self", "num_envs", "device", "episode_length"}
        allowed = set(inspect.signature(cls.__init__).parameters) - reserved
        unknown = sorted(set(cfg) - allowed)
        if unknown:
            raise KeyError(f"unknown env config key(s) {unknown}; valid keys: {sorted(allowed)}")
        return cls(num_envs=num_envs, device=device, episode_length=episode_length, **cfg)

    @staticmethod
    def _read_model_config(path: str | os.PathLike, model_type: str) -> dict:
        if is_pretrained_dir(path):
            return as_model_config(load_pretrained_config(path), model_type)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path}: neither a pretrained dir (config.json + model.safetensors) "
                                    "nor a checkpoint file")
        return checkpoint_model_config(load_checkpoint(path, mmap=True), None, model_type, path)

    def _resolve_constants(self, tokenizer_cfg, dynamics_cfg, color_order, action_scale, grip_dims, grip_range,
                           init_context_len, kv_cache_len, num_inference_steps, seq_cor_tau_idx,
                           reward_mtp_index) -> tuple[Any, Any, Any]:
        """Pick every constant from the env config, the model configs or the G1 defaults, and
        record its source in :attr:`constant_sources`. Sets the generation constants and
        ``color_order``; returns the unchecked ``(action_scale, grip_dims, grip_range)``."""
        inference = dynamics_cfg.get("inference") or {}
        action = dynamics_cfg.get("action") or {}
        preprocessing = tokenizer_cfg.get("preprocessing") or {}
        recorded_d_min = inference.get("d_min_idx")
        if recorded_d_min is not None and int(recorded_d_min) != self.d_min_idx:
            raise ValueError(f"the dynamics config.json has inference.d_min_idx={recorded_d_min} but num_tau_levels="
                             f"{self.num_tau_levels} gives log2 = {self.d_min_idx}")
        g1_levels = int(G1_DYNAMICS_ARCH["num_tau_levels"])
        if seq_cor_tau_idx is None and inference.get("seq_cor_tau_idx") is None and self.num_tau_levels != g1_levels:
            raise ValueError(f"seq_cor_tau_idx indexes the noise grid: its G1 default "
                             f"({G1_INFERENCE['seq_cor_tau_idx']}) is for {g1_levels} levels and this dynamics has "
                             f"{self.num_tau_levels}; set seq_cor_tau_idx in the env config or "
                             "inference.seq_cor_tau_idx in the dynamics config.json")

        dyn_json, tok_json = "dynamics config.json", "tokenizer config.json"
        self.constant_sources: dict[str, str] = {}
        values: dict[str, Any] = {}

        def pick(name, explicit, recorded, default, where, key=None):
            key = key or name
            if explicit is not None:
                value, source = explicit, "env config"
            elif recorded.get(key) is not None:
                value, source = recorded[key], where
            else:
                value, source = default[key], "G1 default"
            self.constant_sources[name] = source
            values[name] = value
            return value

        self.init_context_len = int(pick("init_context_len", init_context_len, inference, G1_INFERENCE, dyn_json))
        self.kv_cache_len = int(pick("kv_cache_len", kv_cache_len, inference, G1_INFERENCE, dyn_json))
        self.num_inference_steps = int(pick("num_inference_steps", num_inference_steps, inference, G1_INFERENCE,
                                            dyn_json))
        self.seq_cor_tau_idx = int(pick("seq_cor_tau_idx", seq_cor_tau_idx, inference, G1_INFERENCE, dyn_json))
        self.reward_mtp_index = int(pick("reward_mtp_index", reward_mtp_index, inference, G1_INFERENCE, dyn_json))
        clamp = (pick("action_scale", action_scale, action, G1_ACTION, dyn_json, key="scale"),
                 pick("grip_dims", grip_dims, action, G1_ACTION, dyn_json),
                 pick("grip_range", grip_range, action, G1_ACTION, dyn_json))
        self.color_order = pick("color_order", color_order, preprocessing, G1_PREPROCESSING, tok_json,
                                key="h5_color_order")

        log.info("G1 env constants: %s", ", ".join(f"{k}={v} ({self.constant_sources[k]})" for k, v in values.items()))
        defaults = [k for k, s in self.constant_sources.items()
                    if s == "G1 default" and not (k == "grip_range" and not values["grip_dims"])]  # no grips: unused
        if defaults:
            log.warning("%s: not recorded in the model configs, the paper's G1 values are used (%s). A model "
                        "trained on other data should record its own (the dynamics config.json action / inference "
                        "blocks: python -m coupled_wm.world_model.export --metadata <yaml>, with the action block "
                        "printed by scripts/compute_action_envelope.py; the tokenizer's "
                        "preprocessing.h5_color_order) or the env config should set them.",
                        ", ".join(defaults), "; ".join(f"{k}={values[k]}" for k in defaults))
        recorded_color = preprocessing.get("h5_color_order")
        if color_order is not None and recorded_color is not None and color_order != recorded_color:
            log.warning("data.color_order=%r but the tokenizer recorded h5_color_order=%r for its training files: the "
                        "seed frames are read as %r. Use data.color_order: null unless the seed files are stored in "
                        "another channel order than the tokenizer's training files.",
                        color_order, recorded_color, color_order)
        return clamp

    def _check_generation_constants(self, dyn_arch: Mapping[str, Any]) -> None:
        n = self.num_inference_steps
        if n < 1 or n & (n - 1) or n > self.num_tau_levels:
            raise ValueError(f"num_inference_steps must be a power of two <= num_tau_levels "
                             f"({self.num_tau_levels}), got {n}")
        if not 1 <= self.init_context_len <= self.kv_cache_len:
            raise ValueError(f"init_context_len ({self.init_context_len}) must be in [1, kv_cache_len="
                             f"{self.kv_cache_len}]: the context is prefilled into the KV cache")
        if not 0 <= self.seq_cor_tau_idx < self.num_tau_levels:
            raise ValueError(f"seq_cor_tau_idx must be in [0, {self.num_tau_levels}), got {self.seq_cor_tau_idx}")
        if not dyn_arch["use_reward_head"]:
            raise ValueError("the dynamics model has no reward head; the environment needs one for its reward")
        if not 0 <= self.reward_mtp_index < int(dyn_arch["mtp_length"]):
            raise ValueError(f"reward_mtp_index must be in [0, mtp_length={dyn_arch['mtp_length']}), "
                             f"got {self.reward_mtp_index}")

    def _check_encoder_window(self, tok_arch: Mapping[str, Any], dynamics_cfg: Mapping[str, Any]) -> None:
        """Warn when the context is encoded with another temporal window than the latents the
        dynamics trained on (``tokenizer.encoder_context_len``, recorded by the dynamics trainer)."""
        trained = (dynamics_cfg.get("tokenizer") or {}).get("encoder_context_len")
        if trained is None:
            return
        used = int(tok_arch.get("context_len") or 0)

        def seen(window: int) -> int:  # frames an encoder position attends to within the context (0 = no window)
            return self.init_context_len if window <= 0 else min(window, self.init_context_len)

        if seen(int(trained)) != seen(used):
            log.warning("the dynamics trained on latents encoded with a temporal window of %d frames (0 = none), but "
                        "the env encodes its %d-frame context with the tokenizer's window of %d: the context latents "
                        "are not the ones the dynamics saw (train the dynamics with tokenizer.context_len: null)",
                        int(trained), self.init_context_len, used)

    def _check_trained_window(self, dynamics_cfg: Mapping[str, Any]) -> None:
        """Warn when generation attends over more frames than the dynamics trained on."""
        longest = lineage_max_clip_len(dynamics_cfg)
        if longest is None:
            log.info("the dynamics config records no training clip length: init_context_len and kv_cache_len are not "
                     "checked against it")
            return
        window = int(dynamics_cfg["architecture"].get("context_len") or 0)

        def seen(frames: int) -> int:  # frames one attention pass covers, within the model's sliding window
            return min(frames, window) if window > 0 else frames

        trained, step, prefill = seen(longest), seen(self.kv_cache_len + 1), seen(self.init_context_len)
        if max(step, prefill) > trained:
            log.warning("the dynamics trained on clips of at most %d frames, but a generation step attends to %d "
                        "frames (kv_cache_len + 1) and the context prefill to %d: temporal distances (RoPE positions) "
                        "it never trained on. Lower kv_cache_len / init_context_len, or train the dynamics on longer "
                        "clips.",
                        trained, step, prefill)

    def _build_seed_pool(self, data: Mapping[str, Any], tok_arch: Mapping[str, Any], loader_seed) -> None:
        files = list_h5_files(data["dirs"], include_globs=data["include_globs"], exclude_globs=data["exclude_globs"],
                              min_chunk_index=data["min_chunk_index"], file_list=data["file_list"])
        self.dataset = SeedWindowDataset(files, window=self.init_context_len, stride=1,
                                         image_size=tuple(tok_arch["image_size"]), color_order=self.color_order,
                                         load_proprio=bool(tok_arch["use_proprio"]), reward_key=None)
        if self.dataset.resize:
            log.info("seed frames of size %s are resized to the tokenizer's %s (bilinear, align_corners=False)",
                     [list(s) for s in self.dataset.native_sizes], list(tok_arch["image_size"]))
        if self.dataset.action_dim != self.action_dim:
            raise ValueError(f"seed pool commands have {self.dataset.action_dim} dims, the dynamics takes "
                             f"{self.action_dim}")
        if tok_arch["use_proprio"] and self.dataset.proprio_dim != int(tok_arch["proprio_dim"]):
            raise ValueError(f"seed pool proprio has {self.dataset.proprio_dim} dims, the tokenizer takes "
                             f"{tok_arch['proprio_dim']}")

        # skip_nonzero filters the enumerated windows: window ids stay those of the whole pool.
        anchors = self.dataset.frame_commands(self.init_context_len - 1)
        eligible = np.arange(len(self.dataset))
        if self.dropped_dims_reset == "skip_nonzero":
            eligible = np.flatnonzero((anchors[:, list(self.drop_action_indices)] == 0).all(axis=1))
            excluded = len(self.dataset) - len(eligible)
            log.info("dropped_dims_reset=skip_nonzero: %d of %d seed windows (%.1f%%) have a nonzero command on dims "
                     "%s at the anchor frame and are excluded; episodes are seeded from the other %d",
                     excluded, len(self.dataset), 100.0 * excluded / len(self.dataset), list(self.drop_action_indices),
                     len(eligible))
        if len(eligible) < self.num_envs:
            raise ValueError(f"the seed pool has {len(eligible)} usable windows, fewer than num_envs={self.num_envs}")
        self._eligible = eligible
        self._check_anchor_envelope(anchors[eligible])

        self._seed_generator = None if loader_seed is None else torch.Generator().manual_seed(int(loader_seed))
        workers = int(data["num_workers"])
        self._loader_kwargs = dict(
            num_workers=workers,
            pin_memory=bool(data["pin_memory"]) and self.device.type == "cuda",
            prefetch_factor=int(data["prefetch_factor"]) if workers > 0 else None,
            generator=torch.Generator(),  # worker seeding; keeps the global RNG to the two draws per pass
        )

    def _check_anchor_envelope(self, anchors: np.ndarray) -> None:
        """Warn when seed windows start from a command the clamp moves by more than
        :data:`ANCHOR_TOLERANCE`: their first generated frame jumps to the envelope (a sign that
        the envelope describes other data)."""
        dims = [d for d in range(self.action_dim)
                if self.dropped_dims_reset == "hold" or d not in self.drop_action_indices]
        scale = self.action_scale.cpu().numpy()
        low = np.maximum(-scale, self._grip_low.cpu().numpy())
        high = np.minimum(scale, self._grip_high.cpu().numpy())
        excess = np.maximum(low - anchors, anchors - high)[:, dims]
        outside = excess.max(axis=1) > ANCHOR_TOLERANCE
        if outside.any():
            worst = excess.max(axis=0)
            top = [dims[i] for i in np.argsort(-worst)[:5] if worst[i] > ANCHOR_TOLERANCE]
            log.warning("%d of %d seed windows (%.1f%%) start from a command more than %g outside the clamp (by up "
                        "to %.3g, on dims %s; action_scale from the %s): their first generated frame moves to the "
                        "envelope. Set env.action_scale / grip_dims / grip_range, or record the data's envelope in "
                        "the dynamics config.json (scripts/compute_action_envelope.py).", int(outside.sum()),
                        len(anchors), 100.0 * outside.mean(), ANCHOR_TOLERANCE, float(worst.max()), top,
                        self.constant_sources["action_scale"])

    def _new_pass(self, pass_seed: int | None = None, start: int = 0):
        """Iterator over one random pass through the seed pool in batches of ``num_envs``, from
        batch ``start``; ``pass_seed=None`` draws a new pass."""
        if pass_seed is None:
            pass_seed = _draw_pass_seed(self._seed_generator)
        self._pass_seed, self._pass_pos = int(pass_seed), int(start)
        batches = [self._eligible[b].tolist() for b in _pass_batches(len(self._eligible), self.num_envs, pass_seed)]
        if start >= len(batches):
            return iter(())
        return iter(SignalSafeDataLoader(self.dataset, batch_sampler=batches[start:], **self._loader_kwargs))

    def _next_batch(self) -> dict[str, torch.Tensor]:
        try:
            batch = next(self._loader_iter)
        except StopIteration:
            self._loader_iter = self._new_pass()
            batch = next(self._loader_iter)
        self._pass_pos += 1
        return batch

    def _frames_to_device(self, frames: torch.Tensor) -> torch.Tensor:
        """Seed frames ``(B, T, 3, H, W)`` -> bf16 on the device: uint8 through the exact
        ``x / 255`` table (one env at a time), resized float32 frames by a cast."""
        frames = frames.to(self.device)
        if frames.dtype != torch.uint8:
            return frames.to(torch.bfloat16)
        out = torch.empty(frames.shape, dtype=torch.bfloat16, device=self.device)
        for b in range(frames.shape[0]):
            out[b] = self._u8_to_unit[frames[b].long()]
        return out

    # ------------------------------------------------------------------ public API

    def seed_windows(self) -> list[tuple[str, int]]:
        """``(file, start frame)`` of the window that seeded each env's current episode."""
        if self.seed_window_ids is None:
            raise RuntimeError("the current episodes were seeded by reset_from_context, not from the seed pool")
        return [self.dataset.window_info(int(i)) for i in self.seed_window_ids]

    def state_dict(self) -> dict[str, Any]:
        """The seed-window pass position (and the private generator's state, with
        ``loader_seed``), for resuming a run where it stopped. The global RNG is the caller's."""
        return {
            "pass_seed": self._pass_seed,
            "batches_drawn": self._pass_pos,
            "generator": None if self._seed_generator is None else self._seed_generator.get_state(),
            "pool_windows": len(self._eligible),
            "num_envs": self.num_envs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Continue the seed-window pass of :meth:`state_dict`: the next reset draws the batch the
        saved run would have drawn next. The pool and ``num_envs`` must be the saved ones."""
        saved = (int(state["pool_windows"]), int(state["num_envs"]))
        if saved != (len(self._eligible), self.num_envs):
            raise ValueError(f"the env state is for {saved[0]} seed windows in batches of {saved[1]}, this env has "
                             f"{len(self._eligible)} in batches of {self.num_envs}")
        if (state["generator"] is None) != (self._seed_generator is None):
            raise ValueError("the env state and this env disagree on loader_seed (private generator or global RNG)")
        if self._seed_generator is not None:
            self._seed_generator.set_state(state["generator"])
        self._stop_loader()
        self._loader_iter = self._new_pass(int(state["pass_seed"]), int(state["batches_drawn"]))

    def reset(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> torch.Tensor:
        """Reset every env (envs share one clock, a subset cannot be reset); returns ``img0``."""
        if env_ids is not None:
            ids = torch.as_tensor(env_ids).reshape(-1).tolist()
            if sorted(ids) != list(range(self.num_envs)):
                raise ValueError(f"{type(self).__name__} resets all {self.num_envs} envs together; got env_ids {ids}")
        return self.reset_all()

    @torch.no_grad()
    def reset_all(self) -> torch.Tensor:
        """Seed every env from a fresh data window; returns (and stores) ``img0`` ``[B, 3, S, S]``."""
        batch = self._next_batch()
        img0 = self.reset_from_context(batch["image"], batch["action"], batch.get("proprio"))
        self.seed_window_ids = batch["index"]
        return img0

    @torch.no_grad()
    def reset_from_context(self, images: torch.Tensor, commands: torch.Tensor,
                           proprio: torch.Tensor | None = None) -> torch.Tensor:
        """Seed every env from the given context frames instead of a seed-pool window.

        ``images`` ``(B, init_context_len, 3, H, W)``: RGB at the tokenizer's ``image_size``,
        uint8 or float in ``[0, 1]`` (as :class:`SeedWindowDataset` yields them); ``commands``
        ``(B, init_context_len, A)``: the absolute commands of those frames; ``proprio``
        ``(B, init_context_len, P)``, required when the tokenizer uses proprio. The context is
        prefilled exactly as :meth:`reset_all` prefills a seed window, and the last context
        command becomes the integration anchor. Returns (and stores) ``img0``. Until the next
        :meth:`reset_all`, :meth:`seed_windows` is undefined.
        """
        B, ctx = self.num_envs, self.init_context_len
        image_hw = tuple(int(s) for s in self.tokenizer_config["architecture"]["image_size"])
        if images.dim() != 5 or tuple(images.shape[:3]) != (B, ctx, 3) or tuple(images.shape[3:]) != image_hw:
            raise ValueError(f"images must be ({B}, {ctx}, 3, {image_hw[0]}, {image_hw[1]}) (num_envs, "
                             f"init_context_len, RGB, the tokenizer's image_size), got {tuple(images.shape)}")
        if tuple(commands.shape) != (B, ctx, self.action_dim):
            raise ValueError(f"commands must be ({B}, {ctx}, {self.action_dim}), got {tuple(commands.shape)}")
        if (proprio is not None) != bool(self.tokenizer_config["architecture"]["use_proprio"]):
            raise ValueError("proprio must be given exactly when the tokenizer uses proprio")
        self.traj_steps = 0
        self.seed_window_ids = None
        for cache in (*self.dyn.caches, *self.dec.caches_l, *self.dec.caches_r):
            if cache is not None:
                cache.reset()

        images = self._frames_to_device(images)
        actions = commands.to(self.device, dtype=torch.bfloat16)
        proprio = proprio.to(self.device, dtype=torch.bfloat16) if proprio is not None else None

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            _, _, z_gt = self.enc(images, proprio=proprio)

            # The context enters the cache noised to seq_cor_tau_idx, the level of the cache-writing
            # pass after every generated frame.
            seq_cor_tau = 1.0 - (self.seq_cor_tau_idx + 1) / self.num_tau_levels
            tau = torch.full((B, ctx, 1, 1), seq_cor_tau, dtype=z_gt.dtype, device=self.device)
            eps = torch.randn_like(z_gt)
            z_cor = (1.0 - tau) * eps + tau * z_gt
            sigma_idx = torch.full((B, ctx), self.seq_cor_tau_idx, dtype=torch.long, device=self.device)
            step_idx = torch.full((B, ctx), self.d_min_idx, dtype=torch.long, device=self.device)
            self.dyn.forward_step(action=actions, noisy_z=z_cor, sigma_idx=sigma_idx, step_idx=step_idx,
                                  start_step_idx=0, update_cache=True)

            # The decoder has no noise conditioning: it is prefilled with the clean latents.
            frames = self.dec.forward_step(z_gt, start_step_idx=0, update_cache=True)[1]
            self.img0 = F.interpolate(frames[:, -1], size=(self.img_size, self.img_size), mode="bilinear",
                                      align_corners=False).float()
            self.t = ctx

        # Integration anchor (fp32): the last context command, as the dynamics consumed it (bf16).
        anchor = actions[:, ctx - 1].to(torch.float32)
        if self.dropped_dims_reset == "zero":
            anchor[:, list(self.drop_action_indices)] = 0.0
        self.last_abs_action = anchor
        return self.img0

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        """Advance every env by one frame.

        ``actions``: relative commands ``[B, A]`` (or ``[B, 1, A]``). Returns
        ``(obs [B, 3, S, S], reward [B], done [B] bool, info)`` with ``info`` keys
        ``obs_before_reset`` (the generated frames, also when ``done`` replaced ``obs`` by the
        next episode's first frames), ``dones`` (= ``done``), ``applied_action`` ``[B, A]``
        (executed command minus the previous one) and ``clamped`` ``[B, A]`` bool.
        """
        a_rel = actions.detach().to(self.device, dtype=torch.float32)
        if a_rel.dim() == 2:
            a_rel = a_rel.unsqueeze(1)
        if tuple(a_rel.shape) != (self.num_envs, 1, self.action_dim):
            raise ValueError(f"actions must be [{self.num_envs}, {self.action_dim}], got {tuple(actions.shape)}")

        anchor = self.last_abs_action.unsqueeze(1)
        integrated = anchor + a_rel
        scale = self.action_scale.view(1, 1, -1)
        command = torch.clamp(integrated, -scale, scale)
        command = torch.minimum(torch.maximum(command, self._grip_low), self._grip_high)
        applied = (command - anchor).squeeze(1)
        clamped = (command != integrated).squeeze(1)
        obs, reward = self.step_command(command.squeeze(1))

        done = torch.full((self.num_envs,), self.traj_steps >= self.episode_length, dtype=torch.bool,
                          device=self.device)
        obs_before_reset = obs.clone()
        if self.traj_steps >= self.episode_length:
            obs = self.reset_all().clone()
        info = {"obs_before_reset": obs_before_reset, "dones": done, "applied_action": applied, "clamped": clamped}
        return obs, reward, done, info

    @torch.no_grad()
    def step_command(self, command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate the next frame of every env, conditioned on the absolute ``command``.

        The generation half of :meth:`step`, without the integration, the clamp and the reset at
        the time limit. ``command`` ``[B, A]`` (or ``[B, 1, A]``) is taken as given (cast to fp32,
        then to bf16 at the dynamics input), conditions the frame being generated and becomes
        the anchor of the next :meth:`step`. Replaying recorded commands calls it with
        ``commands[t]`` for the frame ``t`` being generated. Returns ``(obs [B, 3, S, S],
        reward [B])``, fp32, and advances the episode clock; after ``episode_length`` frames the
        envs must be reset before the next call.
        """
        command = command.detach().to(self.device, dtype=torch.float32)
        if command.dim() == 3:
            command = command.squeeze(1)
        if tuple(command.shape) != (self.num_envs, self.action_dim):
            raise ValueError(f"command must be [{self.num_envs}, {self.action_dim}], got {tuple(command.shape)}")
        if self.traj_steps >= self.episode_length:
            raise RuntimeError(f"the episode already has episode_length={self.episode_length} generated frames: "
                               "reset the envs first")
        self.last_abs_action = command.clone()
        cmd = command.unsqueeze(1).to(torch.bfloat16)

        B, n = self.num_envs, self.num_inference_steps
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            # Shortcut/Euler passes from pure noise, read-only on the cache.
            z_t = torch.randn(self._noise_shape, dtype=torch.bfloat16, device=self.device)
            step_val = 1.0 / n
            step_level = int(math.log2(n))
            step_idx = torch.full((B, 1), step_level, dtype=torch.long, device=self.device)
            reward_logits = None
            for i in range(n):
                tau_curr = i / n
                tau_idx = (n - i) * 2 ** (self.d_min_idx - step_level) - 1
                sigma_idx = torch.full((B, 1), tau_idx, dtype=torch.long, device=self.device)
                pred_z, reward_logits = self.dyn.forward_step(action=cmd, noisy_z=z_t, sigma_idx=sigma_idx,
                                                              step_idx=step_idx, start_step_idx=self.t,
                                                              update_cache=False)
                z_t = z_t + (pred_z - z_t) / max(1.0 - tau_curr, 1e-5) * step_val

            # Cache-writing pass: re-noise the result to seq_cor_tau_idx and write it to the cache
            # (the prediction is discarded).
            seq_cor_tau = 1.0 - (self.seq_cor_tau_idx + 1) / self.num_tau_levels
            tau = torch.full((B, 1, 1, 1), seq_cor_tau, dtype=torch.bfloat16, device=self.device)
            eps = torch.randn(self._noise_shape, dtype=torch.bfloat16, device=self.device)
            sigma_idx = torch.full((B, 1), self.seq_cor_tau_idx, dtype=torch.long, device=self.device)
            fine_idx = torch.full((B, 1), self.d_min_idx, dtype=torch.long, device=self.device)
            self.dyn.forward_step(action=cmd, noisy_z=(1.0 - tau) * eps + tau * z_t, sigma_idx=sigma_idx,
                                  step_idx=fine_idx, start_step_idx=self.t, update_cache=True)

            frames = self.dec.forward_step(z_t, start_step_idx=self.t, update_cache=True)[1]
            obs = F.interpolate(frames[:, 0], size=(self.img_size, self.img_size), mode="bilinear",
                                align_corners=False).float()

            # Reward: expectation of the symlog two-hot head of the last Euler pass, then symexp.
            k = self.reward_mtp_index
            probs = torch.softmax(reward_logits, dim=-1)
            buckets = self.dyn.reward_head.heads[k].buckets.view(1, 1, 1, -1).to(probs.dtype)
            expected_symlog = (probs * buckets).sum(dim=-1)
            reward = (torch.sign(expected_symlog) * (torch.exp(torch.abs(expected_symlog)) - 1.0))[:, 0, k].float()

        self.t += 1
        self.traj_steps += 1
        return obs, reward

    def _stop_loader(self, _stop=stop_loader_workers) -> None:
        # _stop is bound here: __del__ may run after the interpreter has cleared this module.
        iterator, self._loader_iter = getattr(self, "_loader_iter", None), None
        _stop(iterator)

    def close(self) -> None:
        """Stop the loader workers and close the seed pool's files."""
        self._stop_loader()
        self.dataset.close()

    def __del__(self) -> None:
        self._stop_loader()


__all__ = ["DROPPED_DIMS_RESET", "SEED_POOL_DEFAULTS", "DreamerV4VecEnvG1Relative", "SeedWindowDataset",
           "draw_seed_pass"]
