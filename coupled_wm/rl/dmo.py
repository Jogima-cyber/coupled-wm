# SPDX-License-Identifier: Apache-2.0
"""DMO: first-order actor-critic through coupled global and local world models.

The actor is trained by backpropagating through ``steps_num``-step unrolls (SHAC). The forward
rollout is produced by the *global* model (the diffusion world-model environment, no gradient);
the gradient is routed through the *local* model (the RSSM). At every step the latent fed to the
actor is the RSSM posterior of the frame the global model produced, and
:class:`~coupled_wm.rssm.GradientSwapingFunction` gives it the gradient of the RSSM's imagined
prior for that step (forward on-manifold, backward through cheap differentiable dynamics). The
reward differentiated by the actor is the RSSM's two-hot reward head on that latent, plus a
normalised entropy bonus ``alpha * (H + |T|) / (2 |T|)`` with a learned temperature ``alpha``
servoing the policy entropy ``H`` to ``target_entropy = T``. An ensemble of MSE critics trained
with TD(lambda) bootstraps the unroll; the RSSM is finetuned online on a replay buffer of the rollouts.

Replay buffer rows and the RSSM sequence convention
    The replay buffer holds one row per env step and env: ``(o_t, a_t, r_t, is_first_t)``, where
    ``a_t`` is the (env-width, pre-clamp) action taken at ``o_t`` and ``r_t`` the reward the env
    returned with ``o_{t+1}``. For RSSM training a window is presented as
    ``(o_t, a_{t-1}, r_{t-1})``: the frame, the action that produced it and the reward that
    arrived with it, which is how the actor uses the model (``latent_projector(z_t, h_t, a_t,
    o_{t+1})`` then ``reward_model(latent_{t+1})``). Row 0 of a window and every row with
    ``is_first = 1`` (an episode's seed frame) carry no reward label: their reward term is
    masked.

Episode boundaries
    When an episode ends (batch-synchronous dones) an extra *terminal row* holds the terminal
    frame ``obs_before_reset`` with zero action and reward, so the last frame and the last reward
    of every episode are training data. Sequences are cut by ``is_first`` only; the validity mask
    removes the rows from the replay buffer's write head on (the head row included). In the unroll, an env
    that resets at step ``i`` has its latent re-projected from the new episode's first frame
    (zero state and action, ``is_first = 1``) right after step ``i``'s value bootstrap
    ``V(terminal)``; a closure that starts on a reset (after the replay buffer fill, an evaluation or a
    resume) does the same before step 0. Both ablations do the same.

Global vs per-rank quantities
    ``train.num_actors`` (``eval.player.num_actors`` under ``--play``), ``train.replay_size``,
    ``train.min_replay``, ``rssm.batch_size`` and ``eval.num_episodes`` are global and divided by
    the world size: ``num_actors``, ``replay_size`` and ``batch_size`` must be multiples of it,
    ``min_replay`` is floored and ``num_episodes`` rounded up per rank. ``rssm.udt`` (updates per
    epoch) and the critic minibatches are per rank (DDP averages gradients across ranks).

Checkpoints and stopping
    ``latest.pt`` (every ``train.save_interval`` iterations, at the end, and when SIGTERM or
    SIGUSR1 arrives) holds everything needed to resume, including each rank's RNG state and, when
    the env has ``state_dict()`` / ``load_state_dict()``, its position in its sequence of seed
    windows. A signal is acted on at the end of the current iteration.

Resume and the replay buffer
    The replay buffer is not in ``latest.pt``. With ``train.checkpoint_replay`` each rank also
    saves its replay buffer and a resumed run reloads it. Otherwise (the default) a resumed run
    starts from an empty replay buffer, refills it with ``train.min_replay`` rows of the current
    (stochastic) policy, and RSSM finetuning goes on with its usual ``rssm.udt`` updates per
    iteration on that replay buffer:
    the ``rssm.init_udt`` warm-up runs once per run, on the first fill, and not again on resume.
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.clip_grad import clip_grad_norm_

from coupled_wm.common.checkpoint import atomic_save, load_checkpoint, strip_compile_prefix
from coupled_wm.common.config import save_config
from coupled_wm.common.distributed import DistInfo, barrier
from coupled_wm.common.logging import ExperimentLogger
from coupled_wm.common.system import StopSignals, warn_if_compile_unavailable
from coupled_wm.rl.actor import build_actor, config_max_entropy, log_std_bounds
from coupled_wm.rl.critic import build_critic
from coupled_wm.rl.utils import torch_utils as tu
from coupled_wm.rl.utils.average_meter import AverageMeter
from coupled_wm.rl.utils.common import seeding
from coupled_wm.rl.utils.dataset import CriticDataset
from coupled_wm.rl.utils.time_report import TimeReport
from coupled_wm.rssm.rssm import (
    RELATIVE_ACTION_DEFINITION,
    GradientSwapingFunction,
    MSEDistribution,
    TwoHotEncodingDistribution,
    build_rssm,
    read_rssm_checkpoint,
    reconstruction_loss,
    rssm_config,
)

log = logging.getLogger(__name__)

POLICY_FORMAT = "coupled_wm.rl.policy"
TRAINING_STATE_FORMAT = "coupled_wm.rl.training_state"
FORMAT_VERSION = 1


# ---------------------------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------------------------


class ReplaySample(NamedTuple):
    """Windows of ``seq_len`` consecutive rows, batch-major: ``(B, T, ...)``."""

    observations: torch.Tensor  # (B, T, C, H, W) float32 in [0, 1]
    actions: torch.Tensor       # (B, T, A)
    rewards: torch.Tensor       # (B, T)
    is_firsts: torch.Tensor     # (B, T) bool
    mask: torch.Tensor          # (B, T) float, 0 from the replay buffer's write head on


class SequenceReplay:
    """Per-rank ring buffer of env steps, written one full-width block (``num_envs`` rows) at a time.

    Row ``k * num_envs + e`` is env ``e``'s ``k``-th written step, so a window of env ``e`` is
    the rows ``start + j * num_envs``. The most recently written block is the *write head*: a
    window that reaches it is masked from that row on (the rows behind it belong to an older lap
    of the ring, or are empty).
    """

    def __init__(self, capacity, obs_shape, action_dim, seq_len, num_envs,
                 storing_device="cpu", training_device="cpu"):
        if capacity % num_envs:
            raise ValueError(f"replay buffer capacity {capacity} must be a multiple of num_envs={num_envs}")
        if capacity < seq_len * num_envs:
            raise ValueError(f"replay buffer capacity {capacity} holds fewer than seq_len={seq_len} steps per env")
        self.capacity = int(capacity)
        self.seq_len = int(seq_len)
        self.num_envs = int(num_envs)
        self.storing_device = torch.device(storing_device)
        self.training_device = torch.device(training_device)
        self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32, device=self.storing_device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.storing_device)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=self.storing_device)
        self.is_firsts = torch.zeros((capacity,), dtype=torch.bool, device=self.storing_device)
        self.markers = torch.zeros((capacity,), dtype=torch.bool, device=self.storing_device)
        self.pos = 0
        self.full = False
        self._head = None

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.observations, self.actions, self.rewards, self.is_firsts, self.markers))

    @staticmethod
    def row_bytes(obs_shape, action_dim) -> int:
        """Bytes per row: fp32 frame, action and reward, bool ``is_first`` and write-head marker."""
        return 4 * (math.prod(obs_shape) + action_dim + 1) + 2

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs, action, reward, is_first):
        """Write one row per env: ``obs (N, C, H, W)``, ``action (N, A)``, ``reward (N,)``, ``is_first (N,)``."""
        n = self.num_envs
        if obs.shape[0] != n:
            raise ValueError(f"replay buffer add: got {obs.shape[0]} rows, expected num_envs={n}")
        start, stop = self.pos, self.pos + n
        dev = self.storing_device
        self.observations[start:stop] = obs.detach().to(dev)
        self.actions[start:stop] = action.detach().to(dev)
        self.rewards[start:stop] = reward.detach().reshape(n).to(dev)
        self.is_firsts[start:stop] = is_first.detach().reshape(n).to(dev)
        if self._head is not None:
            self.markers[self._head[0]:self._head[1]] = False
        self.markers[start:stop] = True
        self._head = (start, stop)
        self.pos = stop % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size) -> ReplaySample:
        starts = torch.randint(0, len(self), size=(batch_size,), device=self.storing_device)
        return self.get_windows(starts)

    def get_windows(self, starts) -> ReplaySample:
        steps = torch.arange(self.seq_len, device=self.storing_device)
        idx = (starts.to(self.storing_device)[:, None] + steps[None, :] * self.num_envs) % self.capacity
        markers = self.markers[idx].float()
        mask = torch.cat([torch.ones_like(markers[:, :1]), (1 - markers).cumprod(dim=1)[:, 1:]], dim=1)
        dev = self.training_device
        return ReplaySample(
            observations=self.observations[idx].to(dev),
            actions=self.actions[idx].to(dev),
            rewards=self.rewards[idx].to(dev),
            is_firsts=self.is_firsts[idx].to(dev),
            mask=mask.to(dev),
        )

    def state_dict(self) -> dict:
        return {"observations": self.observations.cpu(), "actions": self.actions.cpu(),
                "rewards": self.rewards.cpu(), "is_firsts": self.is_firsts.cpu(),
                "markers": self.markers.cpu(), "pos": self.pos, "full": self.full, "head": self._head}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key in ("observations", "actions", "rewards", "is_firsts", "markers"):
            target = getattr(self, key)
            if tuple(state[key].shape) != tuple(target.shape):
                raise ValueError(f"replay buffer {key}: saved shape {tuple(state[key].shape)} != {tuple(target.shape)}")
            target.copy_(state[key])
        self.pos, self.full, self._head = int(state["pos"]), bool(state["full"]), state["head"]


class RSSMBatch(NamedTuple):
    """Time-major RSSM training batch."""

    observations: torch.Tensor  # (T, B, C, H, W) in [-0.5, 0.5]
    actions: torch.Tensor       # (T, B, A): a_{t-1}, the action that produced o_t
    rewards: torch.Tensor       # (T, B, 1): r_{t-1}, the reward returned with o_t
    is_firsts: torch.Tensor     # (T, B, 1) float, row 0 always 1
    mask: torch.Tensor          # (T, B) valid rows
    reward_mask: torch.Tensor   # (T, B) rows with a reward label


def rssm_training_batch(sample: ReplaySample) -> RSSMBatch:
    """Present replay buffer windows to the RSSM as ``(o_t, a_{t-1}, r_{t-1}, is_first_t)``.

    Row 0 is treated as the start of a sequence (``is_first = 1``); its action and reward come
    from outside the window and are zero. The reward term is masked on row 0 and on every
    episode seed row (``is_first = 1``), whose shifted label is not a reward of that episode.
    """
    observations = sample.observations.permute(1, 0, 2, 3, 4)  # (T, B, C, H, W)
    actions = sample.actions.permute(1, 0, 2)                  # (T, B, A)
    rewards = sample.rewards.permute(1, 0)                     # (T, B)
    is_firsts = sample.is_firsts.permute(1, 0)                 # (T, B)
    mask = sample.mask.permute(1, 0)                           # (T, B)

    is_firsts = is_firsts.unsqueeze(-1).float()
    is_firsts[0, :] = torch.ones_like(is_firsts[0, :])
    observations = observations - 0.5

    actions = torch.cat((torch.zeros_like(actions[:1]), actions[:-1]), dim=0)
    rewards = torch.cat((torch.zeros_like(rewards[:1]), rewards[:-1]), dim=0)
    reward_mask = mask * (1.0 - is_firsts.squeeze(-1))
    return RSSMBatch(observations, actions, rewards.unsqueeze(-1), is_firsts, mask, reward_mask)


def reset_alignment(fill_steps: int, episode_length: int, steps_num: int) -> tuple[bool, str]:
    """Where episode resets fall in the ``steps_num``-step unrolls that follow the replay buffer fill.

    Dones are batch-synchronous and every episode lasts ``episode_length`` steps, so after
    ``fill_steps`` fill steps the resets land at a fixed unroll step when ``episode_length`` is a
    multiple of ``steps_num``, and drift otherwise. Returns ``(aligned, message)`` where
    ``aligned`` means every reset is on an unroll's last step (between unrolls).
    """
    head = (f"replay buffer fill = {fill_steps} env steps ({fill_steps / episode_length:g} episodes of "
            f"{episode_length}), unrolls of {steps_num} steps")
    if episode_length % steps_num:
        return False, (f"{head}: episode_length is not a multiple of steps_num, so resets drift "
                       f"through the unroll (handled by the mid-unroll latent refresh)")
    reset_step = (-fill_steps - 1) % steps_num
    if reset_step == steps_num - 1:
        return True, f"{head}: episode resets land between unrolls (on unroll step {reset_step})"
    return False, (f"{head}: episode resets land inside unrolls, on unroll step {reset_step} "
                   f"(handled by the mid-unroll latent refresh)")


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


@dataclass
class RunOptions:
    """Command-line options of a run (not part of the YAML config)."""

    seed: int = 0
    logdir: str | None = None
    exp_name: str | None = None
    no_eval: bool = False
    no_diffusion: bool = False
    no_rssm_finetuning: bool = False
    play: bool = False
    resume: str | None = None  # training checkpoint (latest.pt) the trainer resumes from


def make_env(env_cfg: Mapping[str, Any], num_envs: int, device: torch.device, episode_length: int):
    """Build the environment of an ``env`` config block.

    ``name`` is an environment of :mod:`coupled_wm.envs` or a ``"module:Class"`` path; the other
    keys are constructor keyword arguments, to which ``num_envs``, ``device`` and
    ``episode_length`` are added.
    """
    name = env_cfg["name"]
    if ":" not in name:
        from coupled_wm.envs import make_env as make_registered_env

        return make_registered_env(env_cfg, num_envs=num_envs, device=device, episode_length=episode_length)
    import importlib

    module_name, cls_name = name.split(":", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    kwargs = {k: v for k, v in env_cfg.items() if k != "name"}
    return cls(num_envs=num_envs, device=device, episode_length=episode_length, **kwargs)


def check_warmstart_action(action_meta: Mapping[str, Any] | None, path, finetuned: bool) -> None:
    """Check the action convention recorded with an RSSM warm start against the trainer's.

    The actor emits relative commands and the replay buffer pairs frame ``t`` with
    ``commands[t] - commands[t-1]``. An absolute-action RSSM has the same shapes and would load
    silently, so it is rejected. A relative RSSM with another row alignment only warns: online
    finetuning (``finetuned``) re-fits it, while under ``--no-diffusion`` and
    ``--no-rssm-finetuning`` the mismatch stays for the whole run. A bare state dict carries no
    metadata and is not checked.
    """
    if not action_meta:
        return
    mode = action_meta.get("mode")
    if mode is not None and mode != "relative":
        raise ValueError(f"RSSM warm start {path} was pretrained on {mode!r} actions; the DMO actor's actions are "
                         "relative command changes (coupled_wm.rssm.pretrain writes relative-action RSSMs)")
    definition = action_meta.get("definition")
    if definition is not None and definition != RELATIVE_ACTION_DEFINITION:
        outcome = ("online RSSM finetuning re-fits it" if finetuned else
                   "this run does not finetune the RSSM, so the mismatch stays for the whole run")
        log.warning("RSSM warm start %s pairs frame t with %r; the trainer's rows use %r (%s)", path, definition,
                    RELATIVE_ACTION_DEFINITION, outcome)


def _remap_pairs(remap) -> list[tuple[float, float]] | None:
    """A recorded ``reward_remap`` (``{value: new}`` or ``[[value, new], ...]``) as sorted float pairs."""
    if not remap:
        return None
    return sorted((float(k), float(v)) for k, v in (remap.items() if isinstance(remap, Mapping) else remap))


def load_stripped_state_dict(module: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Strict-load a wrapper-free state dict into ``module``, which may hold compiled submodules."""
    target = {new: old for new, old in strip_compile_prefix({k: k for k in module.state_dict()}).items()}
    state_dict = strip_compile_prefix(state_dict)
    missing = sorted(set(target) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(target))
    if missing or unexpected:
        raise KeyError(f"state dict mismatch for {type(module).__name__}: missing {missing[:8]}, "
                       f"unexpected {unexpected[:8]}")
    module.load_state_dict({target[k]: v for k, v in state_dict.items()}, strict=True)


def _cpu_state(state_dict: Mapping[str, torch.Tensor]) -> dict:
    return {k: v.detach().cpu() for k, v in strip_compile_prefix(state_dict).items()}


def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


class RSSMPolicy:
    """Deterministic policy acting from env frames: RSSM posterior filter + the actor's mean action.

    Each :meth:`act` projects the frame with the previous latent and action (resetting envs whose
    previous step ended an episode), then returns ``tanh(mean) * action_scale`` expanded to the
    env's action width (dropped dims are 0).
    """

    def __init__(self, actor, rssm, num_actions, drop_action_indices, num_envs, device):
        self.actor = actor
        self.rssm = rssm
        self.num_actions = num_actions
        self.drop_action_indices = tuple(drop_action_indices)
        self.kept_action_indices = [i for i in range(num_actions) if i not in self.drop_action_indices]
        self.num_envs = num_envs
        self.device = device
        self.stoch_size = rssm.stochastic_size * rssm.discrete
        self.recurrent_size = rssm.recurrent_state_size
        self.reset()

    def reset(self):
        n, dev = self.num_envs, self.device
        self.recurrent_state = torch.zeros(1, n, self.recurrent_size, device=dev)
        self.posterior = torch.zeros(1, n, self.stoch_size, device=dev)
        self.action = torch.zeros(1, n, self.num_actions, device=dev)
        self.is_first = torch.ones(1, n, device=dev)

    def expand_action(self, action):
        if not self.drop_action_indices:
            return action
        out = action.new_zeros(*action.shape[:-1], self.num_actions)
        out[..., self.kept_action_indices] = action
        return out

    @torch.no_grad()
    def act(self, obs):
        batch_obs = obs - 0.5
        self.recurrent_state, self.posterior = self.rssm.latent_projector(
            self.posterior, self.recurrent_state, self.action, batch_obs, self.is_first)
        actor_input = torch.cat([self.posterior, self.recurrent_state], dim=-1)
        action, _ = self.actor(actor_input, deterministic=True)
        action = torch.tanh(action) * self.actor.action_scale
        self.action = self.expand_action(action)
        return self.action.squeeze(0)

    def observe_dones(self, done):
        self.is_first = done.float().unsqueeze(0)


# ---------------------------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------------------------


class DMOTrainer:
    """DMO actor-critic trainer (see the module docstring).

    Args:
        cfg: the run config (blocks ``env``, ``actor``, ``critic``, ``rssm``, ``train``, ``eval``,
            ``wandb``).
        run: command-line options (seed, log dir, ablation flags).
        dist_info: from :func:`coupled_wm.common.distributed.init_distributed`.
        env: an already-built environment (tests); built from ``cfg["env"]`` if ``None``.
    """

    def __init__(self, cfg: Mapping[str, Any], run: RunOptions, dist_info: DistInfo, env=None):
        self.cfg = copy.deepcopy(dict(cfg))
        self.run_opts = run
        self.rank = dist_info.rank
        self.world_size = dist_info.world_size
        self.local_rank = dist_info.local_rank
        self.is_distributed = dist_info.distributed and self.world_size > 1
        self.device = dist_info.device

        train_cfg, eval_cfg = self.cfg["train"], self.cfg["eval"]
        rssm_cfg, critic_cfg = self.cfg["rssm"], self.cfg["critic"]

        # Different seed per rank: different env rollouts.
        seeding(run.seed + self.rank)

        self.no_eval = bool(run.no_eval)
        self.no_diffusion = bool(run.no_diffusion)
        self.no_rssm_finetuning = bool(run.no_rssm_finetuning)
        self.finetune_rssm = not (self.no_diffusion or self.no_rssm_finetuning)

        # Config checks that need no environment come first: building the env loads the world model.
        actors_key = "eval.player.num_actors" if run.play else "train.num_actors"
        total_actors = int(eval_cfg["player"]["num_actors"] if run.play else train_cfg["num_actors"])
        if total_actors % self.world_size:
            raise ValueError(f"{actors_key}={total_actors} (global) must be divisible by the world size "
                             f"{self.world_size}")
        num_envs = int(env.num_envs) if env is not None else total_actors // self.world_size
        global_rssm_batch = int(rssm_cfg["batch_size"])
        if global_rssm_batch % self.world_size:
            raise ValueError(f"rssm.batch_size={global_rssm_batch} (global) must be divisible by the world "
                             f"size {self.world_size}")
        self.rssm_batch_size = global_rssm_batch // self.world_size
        self.target_entropy = float(train_cfg["target_entropy"])
        if self.target_entropy == 0:
            raise ValueError("train.target_entropy must be non-zero (the entropy bonus is normalised by "
                             "|target_entropy|)")
        config_entropy_bound = config_max_entropy(self.cfg["actor"])
        if config_entropy_bound is not None:
            self._check_target_entropy(config_entropy_bound)
        if run.play:
            self.replay_size = self.replay_device = self.min_replay = None  # play mode has no replay buffer
        else:
            self._plan_replay(train_cfg, num_envs, env.img_size if env is not None else self.cfg["env"].get("img_size"))
        warmstart = None if (run.play or run.resume) else rssm_cfg.get("pretrained")
        if warmstart and not os.path.exists(warmstart):
            raise FileNotFoundError(f"rssm.pretrained: {warmstart} does not exist")

        if env is None:
            env = self._make_env(num_envs, actors_key, total_actors)
        self.env = env
        self.num_envs = self.num_eval_envs = int(self.env.num_envs)
        self.num_actions = int(self.env.num_actions)
        self.img_size = int(self.env.img_size)
        self.max_episode_length = int(self.env.episode_length)
        log.info("num_envs (per rank) = %d, num_actions = %d, img_size = %d, episode_length = %d",
                 self.num_envs, self.num_actions, self.img_size, self.max_episode_length)

        # The actor controls the kept dims only; dropped dims enter the env and the RSSM as 0.
        self.drop_action_indices = tuple(int(i) for i in self.cfg["env"].get("drop_action_indices", ()) or ())
        if len(set(self.drop_action_indices)) != len(self.drop_action_indices) or not all(
                0 <= i < self.num_actions for i in self.drop_action_indices):
            raise ValueError(f"drop_action_indices {self.drop_action_indices} must be distinct indices "
                             f"below num_actions={self.num_actions}")
        self.kept_action_indices = [i for i in range(self.num_actions) if i not in self.drop_action_indices]
        self.policy_num_actions = len(self.kept_action_indices)
        if self.drop_action_indices:
            log.info("drop_action_indices = %s: actor outputs %d dims, env sees %d",
                     self.drop_action_indices, self.policy_num_actions, self.num_actions)

        self.steps_num = int(train_cfg["steps_num"])
        self.dyn_seq_len = int(rssm_cfg["sequence_length"])
        self.num_eval_episodes = int(eval_cfg["num_episodes"])
        self.eval_every = int(eval_cfg["every"])
        self.gamma = float(train_cfg.get("gamma", 0.99))
        self.critic_method = train_cfg.get("critic_method", "td-lambda")
        if self.critic_method not in ("one-step", "td-lambda"):
            raise ValueError(f"critic_method must be 'one-step' or 'td-lambda', got {self.critic_method!r}")
        self.lam = float(train_cfg.get("lambda", 0.95))
        self.max_epochs = int(train_cfg["max_epochs"])
        self.actor_lr = float(train_cfg["actor_learning_rate"])
        self.critic_lr = float(train_cfg["critic_learning_rate"])
        self.dyn_model_lr = float(rssm_cfg["learning_rate"])
        self.lr_schedule = train_cfg.get("lr_schedule", "linear")
        if self.lr_schedule not in ("linear", "constant"):
            raise ValueError(f"lr_schedule must be 'linear' or 'constant', got {self.lr_schedule!r}")
        self.actor_lr_schedule_min = float(train_cfg.get("actor_lr_schedule_min", 1e-5))
        self.critic_lr_schedule_min = float(train_cfg.get("critic_lr_schedule_min", 1e-5))
        self.critic_iterations = int(train_cfg.get("critic_iterations", 16))
        self.num_batch = int(train_cfg.get("num_batch", 4))
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.truncate_grad = bool(train_cfg["truncate_grads"])
        self.grad_norm = float(train_cfg["grad_norm"])
        self.save_interval = int(train_cfg.get("save_interval", 500))
        self.checkpoint_replay = bool(train_cfg.get("checkpoint_replay", False))
        self.use_compile = bool(train_cfg.get("compile", True))

        # RSSM (local world model)
        self.rssm_arch = rssm_config({**rssm_cfg, "action_size": self.num_actions, "image_size": self.img_size})
        self.stochastic_size = self.rssm_arch["stochastic_size"]
        self.discrete_size = self.rssm_arch["discrete_size"]
        self.recurrent_state_size = self.rssm_arch["recurrent_state_size"]
        self.kl_dynamic = float(rssm_cfg["kl_dynamic"])
        self.kl_representation = float(rssm_cfg["kl_representation"])
        self.kl_free_nats = float(rssm_cfg["kl_free_nats"])
        self.kl_regularizer = float(rssm_cfg["kl_regularizer"])
        self.straight_through = bool(rssm_cfg.get("straight_through_stochastic_grad", False))
        self.num_act_obs = self.stochastic_size * self.discrete_size + self.recurrent_state_size

        # Actor and critics
        self.actor = build_actor(self.cfg["actor"], self.num_act_obs, self.policy_num_actions, self.device)
        if self.is_distributed:
            self.actor = DDP(self.actor, device_ids=self._ddp_device_ids())

        self.clipped_double_critic = bool(critic_cfg.get("clipped_double_critic", True))
        if self.clipped_double_critic:
            num_critics = int(critic_cfg.get("num_critics", 10))
            self.critic = [build_critic(critic_cfg, self.num_act_obs, self.device) for _ in range(num_critics)]
        else:
            self.critic = [build_critic(critic_cfg, self.num_act_obs, self.device)]
        if self.is_distributed:
            self.critic = [DDP(c, device_ids=self._ddp_device_ids()) for c in self.critic]
        self.target_critic_alpha = float(critic_cfg.get("target_critic_alpha", 0.4))
        if not self.clipped_double_critic:
            # Polyak-averaged target: a plain module whose parameters take no gradient (the
            # actor's bootstrap still differentiates through it with respect to its input).
            self.target_critic_net = copy.deepcopy(self.critic_cores[0]).requires_grad_(False)

        self.rssm = build_rssm(self.rssm_arch)
        self.rssm_warmstart_info = None
        if run.play or run.resume:
            pass  # the weights come from the evaluated or resumed checkpoint
        elif rssm_cfg.get("pretrained"):
            self._load_rssm_warmstart(rssm_cfg["pretrained"])
        else:
            log.warning("rssm.pretrained is not set: the RSSM starts from random weights")
        self.rssm = self.rssm.to(self.device)
        if self.use_compile:
            warn_if_compile_unavailable("train.compile", "--set train.compile=false", self.device)
            self._compile_rssm()
        if self.is_distributed:
            self.rssm = DDP(self.rssm, device_ids=self._ddp_device_ids())

        betas = tuple(float(b) for b in train_cfg["betas"])
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), betas=betas, lr=self.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(
            itertools.chain(*[c.parameters() for c in self.critic]), betas=betas, lr=self.critic_lr)
        self.dyn_model_optimizer = torch.optim.Adam(self.rssm.parameters(), lr=self.dyn_model_lr)

        # Critic data of one unroll
        n, s = self.num_envs, self.steps_num
        self.obs_buf = torch.zeros((s, n, self.num_act_obs), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((s, n), dtype=torch.float32, device=self.device)
        self.ent_buf = torch.zeros((s, n), dtype=torch.float32, device=self.device)
        self.done_mask = torch.zeros((s, n), dtype=torch.float32, device=self.device)
        self.next_values = torch.zeros((s, n), dtype=torch.float32, device=self.device)
        self.target_values = torch.zeros((s, n), dtype=torch.float32, device=self.device)

        # Replay buffer (sized in _plan_replay, allocated in train())
        self.replay = None
        self.dyn_udt = int(rssm_cfg.get("udt", 256))
        self.init_dyn_udt = int(rssm_cfg.get("init_udt", 1000))

        # Entropy temperature
        if config_entropy_bound is None:
            self.max_entropy = self.actor_core.max_entropy()
            self._check_target_entropy(self.max_entropy)
        else:
            self.max_entropy = config_entropy_bound
        log.info("target_entropy = %g, maximum attainable entropy = %.2f (margin %.2f nats)",
                 self.target_entropy, self.max_entropy, self.max_entropy - self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=float(train_cfg.get("alpha_learning_rate", 5e-3)))

        self.pretrained = False

        # Counters and episode statistics
        self.iter_count = 0
        self.step_count = 0
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.value_loss = np.inf
        self.dyn_loss = 0.0
        self.grad_norm_before_clip = torch.tensor(0.0)
        self.grad_norm_after_clip = torch.tensor(0.0)
        self._reset_episode_stats()
        self._clamp_hits = torch.zeros((), dtype=torch.float32, device=self.device)
        self._clamp_total = 0
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.local_episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        self.logger = None
        self.time_report = TimeReport()
        for name in ("algorithm", "compute actor loss", "forward simulation", "backward simulation",
                     "prepare critic dataset", "actor training", "critic training", "dyn model training",
                     "env step", "eval env step", "collecting episode loss", "recomputing reward"):
            self.time_report.add_timer(name)
        self._resumed = False
        if run.resume:
            self.load_training_state(run.resume)

    # ---- construction helpers ---------------------------------------------------------------

    def _check_target_entropy(self, max_entropy):
        if not self.target_entropy < max_entropy:
            actor_cfg = self.cfg["actor"]
            raise ValueError(
                f"train.target_entropy={self.target_entropy} is not below the policy's maximum attainable entropy "
                f"{max_entropy:.2f} (actor.action_scale {actor_cfg.get('action_scale', 1.0)}, log_std in "
                f"{list(log_std_bounds(actor_cfg))}): alpha would diverge")

    def _plan_replay(self, train_cfg, num_envs, img_size):
        """Per-rank replay buffer size, device and fill; a GPU replay buffer larger than the GPU fails here."""
        global_replay = int(float(train_cfg["replay_size"]))
        if global_replay % self.world_size:
            raise ValueError(f"train.replay_size={global_replay} (global) must be divisible by the world size "
                             f"{self.world_size}")
        self.replay_size = global_replay // self.world_size
        if self.replay_size % num_envs:
            raise ValueError(
                f"per-rank replay buffer size {self.replay_size} (= replay_size / world_size) must be a multiple of "
                f"the per-rank num_envs={num_envs}; e.g. replay_size="
                f"{self.replay_size // num_envs * num_envs * self.world_size}")
        replay_device = str(train_cfg.get("replay_device", "cuda")).lower()
        if replay_device not in ("cuda", "cpu"):
            raise ValueError(f"train.replay_device must be 'cuda' or 'cpu', got {replay_device!r}")
        self.replay_device = self.device if replay_device == "cuda" else torch.device("cpu")
        self.min_replay = int(train_cfg["min_replay"]) // self.world_size
        if self.replay_device.type == "cuda" and img_size is not None:
            nbytes = self.replay_size * SequenceReplay.row_bytes((3, int(img_size), int(img_size)), 0)
            total = torch.cuda.get_device_properties(self.replay_device).total_memory
            log.info("replay buffer: %d rows per rank, at least %.1f GiB on %s", self.replay_size, nbytes / 2**30,
                     self.replay_device)
            if nbytes > total:
                raise ValueError(f"the replay buffer needs at least {nbytes / 2**30:.1f} GiB per rank, more than the "
                                 f"{total / 2**30:.1f} GiB of {self.replay_device}; {self._replay_memory_hint()}")

    def _replay_memory_hint(self):
        return (f"train.replay_size={self.replay_size * self.world_size} is global and split over the world size "
                f"{self.world_size}: more GPUs, or train.replay_device: cpu (host RAM), lower the per-rank GPU memory")

    def _make_env(self, num_envs, actors_key, total_actors):
        try:
            return make_env(self.cfg["env"], num_envs, self.device, int(self.cfg["train"]["max_episode_length"]))
        except torch.OutOfMemoryError as e:
            raise torch.OutOfMemoryError(
                f"out of GPU memory while building the environment with {num_envs} envs per rank ({actors_key}="
                f"{total_actors} over the world size {self.world_size}): the world model's KV caches and reset "
                f"prefill grow with the envs per rank, so more GPUs lower the per-rank memory. {e}") from e

    def autocast(self):
        """bf16 autocast on CUDA (the paper's mixed precision); CPU runs (tests) stay in fp32,
        because CPU autocast keeps ops such as softmax in bf16."""
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda")

    def _ddp_device_ids(self):
        return [self.local_rank] if self.device.type == "cuda" else None

    def _load_rssm_warmstart(self, path):
        state_dict, stored = read_rssm_checkpoint(path, self.rssm_arch)
        self.rssm.load_state_dict(state_dict, strict=True)
        meta = {k: v for k, v in stored.items() if k not in self.rssm_arch and k not in ("model_type", "format_version")}
        check_warmstart_action(meta.get("action"), path, self.finetune_rssm)
        preprocessing = meta.get("preprocessing") or {}
        recorded = preprocessing.get("h5_color_order")
        env_order = getattr(self.env, "color_order", None)
        if recorded and env_order and recorded != env_order:
            log.warning("RSSM warm start %s was pretrained on h5 frames stored as %s, the env reads its seed pool as "
                        "%s: check that both describe how the files are stored (env.data.color_order; null = the "
                        "tokenizer's recorded order)", path, recorded, env_order)
        dyn_data = ((getattr(self.env, "dynamics_config", None) or {}).get("training") or {}).get("data") or {}
        if "reward_remap" in preprocessing and "reward_remap" in dyn_data:
            rssm_remap, dyn_remap = _remap_pairs(preprocessing["reward_remap"]), _remap_pairs(dyn_data["reward_remap"])
            if rssm_remap != dyn_remap:
                log.warning("RSSM warm start %s was pretrained with reward_remap %s, the dynamics reward head with %s: "
                            "online finetuning fits the RSSM's reward head to the world model's rewards", path,
                            rssm_remap, dyn_remap)
        self.rssm_warmstart_info = {"path": str(path), **meta}
        log.info("RSSM warm start: %s%s", path, f" (metadata: {meta})" if meta else " (bare state dict)")

    def _compile_rssm(self):
        rssm = self.rssm
        rssm.encoder = torch.compile(rssm.encoder, mode="reduce-overhead")
        rssm.decoder = torch.compile(rssm.decoder, mode="reduce-overhead")
        if self.finetune_rssm:
            rssm.reward_model = torch.compile(rssm.reward_model, mode="reduce-overhead")
        else:
            rssm.reward_model = torch.compile(rssm.reward_model)
        rssm.dynamic = torch.compile(rssm.dynamic)
        rssm.latent_projector = torch.compile(rssm.latent_projector)
        if self.no_diffusion:
            rssm.imagination = torch.compile(rssm.imagination)
        else:
            rssm._transition = torch.compile(rssm._transition)

    def _build_replay(self):
        if self.replay is not None:
            return
        obs_shape = (3, self.img_size, self.img_size)
        try:
            self.replay = SequenceReplay(
                self.replay_size, obs_shape, self.num_actions, self.dyn_seq_len,
                self.num_envs, storing_device=self.replay_device, training_device=self.device)
        except torch.OutOfMemoryError as e:
            nbytes = self.replay_size * SequenceReplay.row_bytes(obs_shape, self.num_actions)
            raise torch.OutOfMemoryError(
                f"out of GPU memory allocating the replay buffer ({self.replay_size} rows per rank, "
                f"{nbytes / 2**30:.1f} GiB on {self.replay_device}); {self._replay_memory_hint()}") from e
        log.info("replay buffer: %d rows per rank (global %d) on %s, %.2f GiB per rank",
                 self.replay_size, self.replay_size * self.world_size, self.replay_device,
                 self.replay.nbytes / 2**30)

    def _reset_episode_stats(self):
        n = self.num_envs
        self.episode_loss = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.local_episode_loss = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.episode_discounted_loss = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.episode_gamma = torch.ones(n, dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros(n, dtype=int, device=self.device)

    @property
    def rssm_core(self):
        return self.rssm.module if self.is_distributed else self.rssm

    @property
    def actor_core(self):
        return self.actor.module if self.is_distributed else self.actor

    @property
    def critic_cores(self):
        return [c.module if self.is_distributed else c for c in self.critic]

    # ---- rollout pieces ----------------------------------------------------------------------

    def _expand_action(self, action):
        """Insert zero columns at ``drop_action_indices``: policy width -> env width."""
        if not self.drop_action_indices:
            return action
        out = action.new_zeros(*action.shape[:-1], self.num_actions)
        out[..., self.kept_action_indices] = action
        return out

    def target_critic(self, obs):
        """``(V for the actor's bootstrap, V for the critic's TD target)``.

        Clipped double critic: the ensemble mean, and the min of two random members. Otherwise the
        Polyak target network for both.
        """
        if self.clipped_double_critic:
            preds = torch.cat([critic(obs) for critic in self.critic_cores], -1)
            targets_selected = torch.randperm(len(self.critic))[:2]
            target_preds = preds[..., targets_selected]
            return preds.mean(-1), target_preds.min(-1).values
        value = self.target_critic_net(obs).squeeze(-1)
        return value, value

    def _env_step(self, action, record_clamps=False):
        next_obs, rew, done, extra_info = self.env.step(action)
        done = extra_info["dones"]
        if record_clamps and "clamped" in extra_info:
            # Diagnostics only: how often the env's command clamp changed a policy-controlled dim.
            clamped = extra_info["clamped"][:, self.kept_action_indices]
            self._clamp_hits += clamped.sum()
            self._clamp_total += clamped.numel()
        real_next_obs = next_obs.clone()
        if done.any():
            done_idx = done.nonzero().squeeze(-1)
            real_next_obs[done_idx] = extra_info["obs_before_reset"][done_idx]
        return next_obs, rew, done, real_next_obs

    def _add_terminal_row(self, real_next_obs, done):
        """After a done: one extra row with the terminal frame, zero action and zero reward.

        Under the RSSM convention the terminal row presents ``(o_T, a_{T-1}, r_{T-1})`` and the
        next episode's seed row ``(o'_0, 0, 0, is_first=1)`` (its reward term is masked).
        """
        if not bool(done.any()):
            return
        if not bool(done.all()):
            raise RuntimeError(
                f"the terminal replay buffer row is written for all envs at once, so episode ends must be "
                f"batch-synchronous; got {int(done.sum())}/{done.numel()} envs done")
        n = self.num_envs
        self.replay.add(real_next_obs,
                        torch.zeros(n, self.num_actions, device=self.device),
                        torch.zeros(n, device=self.device),
                        torch.zeros(n, dtype=torch.bool, device=self.device))

    def _refresh_reset_latents(self, obs, recurrent_state, posterior, is_first):
        """Re-project the envs with ``is_first = 1`` from ``obs`` (their new episode's first frame)
        with zero state and action, as the RSSM starts a sequence."""
        n = self.num_envs
        with torch.no_grad():
            r_reset = torch.zeros(1, n, self.recurrent_state_size, device=self.device)
            p_reset = torch.zeros(1, n, self.stochastic_size * self.discrete_size, device=self.device)
            a_reset = torch.zeros(1, n, self.num_actions, device=self.device)
            r_fresh, p_fresh = self.rssm_core.latent_projector(
                p_reset, r_reset, a_reset, obs - 0.5, torch.ones(1, n, device=self.device))
            mask = is_first.view(1, n, 1)
            recurrent_state = torch.where(mask > 0.5, r_fresh, recurrent_state)
            posterior = torch.where(mask > 0.5, p_fresh, posterior)
        return recurrent_state, posterior

    def initialize_env(self):
        self.env.reset(torch.arange(self.num_envs))

    def _start_rollout_from_reset(self):
        """Carry state for the first unroll after an env reset (evaluation, resume)."""
        n = self.num_envs
        self.last_obs = self.env.img0.clone()
        self.recurrent_state = torch.zeros(1, n, self.recurrent_state_size, device=self.device)
        self.posterior = torch.zeros(1, n, self.stochastic_size * self.discrete_size, device=self.device)
        self.action = torch.zeros(1, n, self.num_actions, device=self.device)
        self.is_first = torch.ones(1, n, device=self.device)
        self.is_first_true = torch.ones(1, n, device=self.device)

    def fill_replay_buffer(self):
        """Run the current policy until each rank's replay buffer holds ``min_replay`` steps (terminal
        rows are extra), then project the last frame to start the first unroll from it."""
        filled_nb = 0
        n = self.num_envs
        report_every = max(1, math.ceil(self.min_replay / n) // 10)
        with torch.no_grad():
            obs = self.env.img0.clone()

            recurrent_state = torch.zeros(1, n, self.recurrent_state_size, device=self.device)
            posterior = torch.zeros(1, n, self.stochastic_size * self.discrete_size, device=self.device)
            action = torch.zeros(1, n, self.num_actions, device=self.device)
            is_first = torch.ones(1, n, device=self.device)

            step = 0
            while filled_nb < self.min_replay:
                batch_obs = obs - 0.5
                recurrent_state, posterior = self.rssm_core.latent_projector(
                    posterior, recurrent_state, action, batch_obs, is_first)

                actor_input = torch.cat([posterior, recurrent_state], dim=-1)
                action, _ = self.actor_core(actor_input, deterministic=False)
                action = torch.tanh(action) * self.actor_core.action_scale
                action = self._expand_action(action)

                self.time_report.start_timer("env step")
                next_obs, rew, done, real_next_obs = self._env_step(action.squeeze(0))
                self.time_report.end_timer("env step")

                self.replay.add(obs, action.squeeze(0), rew, is_first.squeeze(0) > 0.5)
                self._add_terminal_row(real_next_obs, done)
                is_first = done.float().unsqueeze(0)

                filled_nb += n
                obs = next_obs.clone()
                step += 1
                if step % report_every == 0:
                    log.info("replay fill: %d / %d steps per rank", filled_nb, self.min_replay)

            batch_obs = obs - 0.5
            recurrent_state, posterior = self.rssm_core.latent_projector(
                posterior, recurrent_state, action, batch_obs, is_first)

        self.last_obs = obs.clone()
        self.recurrent_state = recurrent_state.clone()
        self.posterior = posterior.clone()
        self.action = action.clone()
        self.is_first = is_first.clone()
        self.is_first_true = is_first.clone()
        log.info("replay fill done (%d rows per rank)", len(self.replay))

    # ---- actor loss --------------------------------------------------------------------------

    def compute_actor_loss(self):
        rew_acc = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)
        gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        next_values = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)
        next_values_target = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)

        actor_loss = torch.tensor(0., dtype = torch.float32, device = self.device)

        obs = self.last_obs.clone()
        recurrent_state = self.recurrent_state.clone()
        posterior = self.posterior.clone()
        actions = self.action.clone()
        is_first = self.is_first.clone()
        # The replay buffer's reset flag is carried separately: `is_first` is cleared once the latent of
        # a reset env has been re-projected, the replay buffer row of the new episode still needs it.
        is_first_true = self.is_first_true.clone()

        horizon = self.steps_num

        with self.autocast():
            if bool((is_first > 0.5).any()):
                # The unroll starts on a reset: the carried latent belongs to the previous episode
                # (or is zero). Start the reset envs from their new first frame.
                recurrent_state, posterior = self._refresh_reset_latents(obs, recurrent_state, posterior, is_first)
                is_first = torch.zeros_like(is_first)

            full_latent = torch.cat([posterior, recurrent_state], dim=-1)

            for i in range(horizon):
                # collect data for critic training
                with torch.no_grad():
                    self.obs_buf[i] = full_latent

                actions, ent = self.actor(full_latent)
                actions = torch.tanh(actions) * self.actor_core.action_scale
                actions = self._expand_action(actions)

                with torch.no_grad():
                    next_obs, rew, done, real_next_obs = self._env_step(actions.squeeze(0), record_clamps=True)

                if not self.no_diffusion:
                    # DMO step: the forward latent is the RSSM posterior of the frame the global
                    # model produced; its gradient is that of the RSSM's imagined prior.
                    batch_obs = real_next_obs - 0.5
                    recurrent_state, posterior = self.rssm_core.latent_projector(
                        posterior.clone(), recurrent_state.clone(), actions.clone(), batch_obs, is_first)
                    imagined_prior = self.rssm_core.imagined_prior(recurrent_state, straight_through=self.straight_through)
                    posterior = GradientSwapingFunction.apply(imagined_prior, posterior.clone())
                else:
                    # No-diffusion ablation: the actor unrolls in the RSSM's own imagination.
                    posterior, recurrent_state = self.rssm_core.imagination(
                        posterior.clone(), recurrent_state.clone(), actions.clone(), sample_state=False,
                        straight_through=self.straight_through)
                    posterior = posterior.view(*posterior.shape[:-2], -1)

                full_latent = torch.cat([posterior, recurrent_state], dim=-1)

                self.replay.add(obs, actions.squeeze(0), rew, is_first_true.squeeze(0) > 0.5)
                self._add_terminal_row(real_next_obs, done)
                is_first = done.float().unsqueeze(0)
                is_first_true = is_first.clone()
                obs = next_obs.clone()

                with torch.no_grad():
                    raw_rew = rew.clone()
                    true_rew = rew.clone()

                # Differentiable reward: the RSSM reward head on the latent of the new frame.
                self.time_report.start_timer("recomputing reward")
                pr = TwoHotEncodingDistribution(self.rssm_core.reward_model(full_latent), dims=1)
                rew = pr.mode[0, :, 0].clone()
                local_rew = rew.detach().clone()

                self.ent_buf[i] = ent[0].detach().clone()
                norm_ent = (ent[0] + abs(self.target_entropy)) / (2*abs(self.target_entropy))
                rew += self.alpha * norm_ent
                with torch.no_grad():
                    true_rew += self.alpha * norm_ent
                self.time_report.end_timer("recomputing reward")

                self.episode_length += 1

                done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

                # At a done, full_latent is the latent of obs_before_reset: next_values is the
                # truncation bootstrap V(terminal).
                value_obs = full_latent[0]
                next_values[i + 1], next_values_target[i + 1] = self.target_critic(value_obs)
                if (next_values[i + 1].abs() > 1e6).any():
                    raise FloatingPointError(f"critic value out of range at unroll step {i}")

                rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

                if i < horizon - 1:
                    actor_loss = actor_loss + (- rew_acc[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]).sum()
                else:
                    # terminate all envs at the end of optimization iteration
                    actor_loss = actor_loss + (- rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]).sum()

                # compute gamma for next step
                gamma = gamma * self.gamma

                # clear up gamma and rew_acc for done envs
                gamma[done_env_ids] = 1.
                rew_acc[i + 1, done_env_ids] = 0.

                if bool((is_first > 0.5).any()):
                    # The env has reset and sits at the new episode's first frame (obs) while
                    # full_latent still describes the terminal frame. Start the next step from the
                    # new frame; is_first is cleared so the next projector neither re-resets the
                    # state nor zeroes the next action.
                    recurrent_state, posterior = self._refresh_reset_latents(obs, recurrent_state, posterior, is_first)
                    is_first = torch.zeros_like(is_first)
                    full_latent = torch.cat([posterior, recurrent_state], dim=-1)

                # collect data for critic training
                with torch.no_grad():
                    self.rew_buf[i] = true_rew.clone()
                    if i < horizon - 1:
                        self.done_mask[i] = done.clone().to(torch.float32)
                    else:
                        self.done_mask[i, :] = 1.
                    self.next_values[i] = next_values_target[i + 1].clone()

                # collect episode loss
                self.time_report.start_timer("collecting episode loss")
                with torch.no_grad():
                    self.episode_loss -= raw_rew
                    self.local_episode_loss -= local_rew
                    self.episode_discounted_loss -= self.episode_gamma * raw_rew
                    self.episode_gamma *= self.gamma
                    if len(done_env_ids) > 0:
                        self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                        self.local_episode_loss_meter.update(self.local_episode_loss[done_env_ids])
                        self.episode_discounted_loss_meter.update(self.episode_discounted_loss[done_env_ids])
                        self.episode_length_meter.update(self.episode_length[done_env_ids])
                        for done_env_id in done_env_ids:
                            if (self.episode_loss[done_env_id] > 1e6 or self.episode_loss[done_env_id] < -1e6):
                                raise FloatingPointError("episode return out of range")

                            self.episode_loss_his.append(self.episode_loss[done_env_id].item())
                            self.episode_discounted_loss_his.append(self.episode_discounted_loss[done_env_id].item())
                            self.episode_length_his.append(self.episode_length[done_env_id].item())
                            self.episode_loss[done_env_id] = 0.
                            self.local_episode_loss[done_env_id] = 0.
                            self.episode_discounted_loss[done_env_id] = 0.
                            self.episode_length[done_env_id] = 0
                            self.episode_gamma[done_env_id] = 1.
                self.time_report.end_timer("collecting episode loss")

        actor_loss /= horizon * self.num_envs

        self.actor_loss = actor_loss.detach().cpu().item()

        self.step_count += horizon * self.num_envs * self.world_size

        self.last_obs = obs.clone()
        self.recurrent_state = recurrent_state.clone().detach()
        self.posterior = posterior.clone().detach()
        self.action = actions.clone().detach()
        self.is_first = is_first.clone()
        self.is_first_true = is_first_true.clone()

        return actor_loss

    # ---- evaluation --------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_policy(self, num_games, restore_training_rollout=True):
        """Deterministic evaluation on all ranks (each runs ``ceil(num_games / world_size)``
        episodes); returns the mean (episode loss, discounted loss, length) over all ranks.

        Episode loss is the negated episode return. With ``restore_training_rollout`` the env is
        reset afterwards and the next unroll starts from the new first frames.
        """
        games_per_rank = math.ceil(num_games / self.world_size)
        n = self.num_eval_envs
        episode_loss = torch.zeros(n, dtype = torch.float32, device = self.device)
        episode_length = torch.zeros(n, dtype = torch.float32, device = self.device)
        episode_gamma = torch.ones(n, dtype = torch.float32, device = self.device)
        episode_discounted_loss = torch.zeros(n, dtype = torch.float32, device = self.device)
        totals = torch.zeros(4, dtype=torch.float64, device=self.device)  # loss, discounted loss, length, count

        self.env.reset(torch.arange(n))
        obs = self.env.img0.clone()
        policy = RSSMPolicy(self.actor_core, self.rssm_core, self.num_actions, self.drop_action_indices, n, self.device)

        games_cnt = 0
        while games_cnt < games_per_rank:
            action = policy.act(obs)

            self.time_report.start_timer("eval env step")
            next_obs, rew, done, _ = self._env_step(action)
            self.time_report.end_timer("eval env step")

            policy.observe_dones(done)
            obs = next_obs.clone()

            episode_length += 1
            episode_loss -= rew
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma
            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)
            if len(done_env_ids) > 0:
                totals += torch.stack([episode_loss[done_env_ids].sum(), episode_discounted_loss[done_env_ids].sum(),
                                       episode_length[done_env_ids].sum(),
                                       torch.tensor(float(len(done_env_ids)), device=self.device)]).double()
                episode_loss[done_env_ids] = 0.
                episode_discounted_loss[done_env_ids] = 0.
                episode_length[done_env_ids] = 0
                episode_gamma[done_env_ids] = 1.
                games_cnt += len(done_env_ids)

        if self.is_distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        count = totals[3].item()
        mean_policy_loss = totals[0].item() / count
        mean_policy_discounted_loss = totals[1].item() / count
        mean_episode_length = totals[2].item() / count
        log.info("evaluation over %d episodes: return %.3f, discounted return %.3f, length %.1f",
                 int(count), -mean_policy_loss, -mean_policy_discounted_loss, mean_episode_length)

        if restore_training_rollout:
            self.env.reset(torch.arange(self.num_envs))
            self._start_rollout_from_reset()
            self._reset_episode_stats()

        if self.logger is not None:
            self.logger.scalar('eval_rewards/step', -mean_policy_loss, self.step_count)
            self.logger.scalar('eval_rewards/iter', -mean_policy_loss, self.iter_count)

        return mean_policy_loss, mean_policy_discounted_loss, mean_episode_length

    # ---- critic ------------------------------------------------------------------------------

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            value_num_envs = self.num_envs
            Ai = torch.zeros(value_num_envs, dtype = torch.float32, device = self.device)
            Bi = torch.zeros(value_num_envs, dtype = torch.float32, device = self.device)
            lam = torch.ones(value_num_envs, dtype = torch.float32, device = self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1. - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (self.lam * self.gamma * Ai + self.gamma * self.next_values[i] + (1. - lam) / (1. - self.lam) * self.rew_buf[i])
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError

    def compute_critic_loss(self, batch_sample):
        predicted_values = torch.cat([critic(batch_sample['obs']) for critic in self.critic], -1)
        target_values = batch_sample['target_values'].unsqueeze(-1)
        critic_loss = ((predicted_values - target_values) ** 2).mean()
        return critic_loss

    # ---- RSSM finetuning ---------------------------------------------------------------------

    def train_dyn_model(self):
        train_steps = self.dyn_udt
        if not self.pretrained:
            train_steps = self.init_dyn_udt
            log.info("RSSM finetuning on the filled replay buffer: %d updates", train_steps)

        totals = torch.zeros(5, dtype=torch.float64, device=self.device)
        for ts in range(train_steps):
            if not self.pretrained and (ts + 1) % 500 == 0:
                log.info("RSSM finetuning %d / %d", ts + 1, train_steps)
            batch = rssm_training_batch(self.replay.sample(self.rssm_batch_size))

            with self.autocast():
                reconstructed_obs, reward_logits, priors_logits, posteriors_logits = self.rssm(
                    batch.observations, batch.actions, batch.is_firsts, segment_len=self.dyn_seq_len)

                po = MSEDistribution(reconstructed_obs, dims=len(reconstructed_obs.shape[2:]))
                pr = TwoHotEncodingDistribution(reward_logits, dims=1)
                priors_logits = priors_logits.view(*priors_logits.shape[:-1], self.stochastic_size, self.discrete_size)
                posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], self.stochastic_size, self.discrete_size)
                rec_loss, kl, state_loss, reward_loss, observation_loss = reconstruction_loss(
                    po,
                    batch.observations,
                    pr,
                    batch.rewards,
                    priors_logits,
                    posteriors_logits,
                    self.kl_dynamic,
                    self.kl_representation,
                    self.kl_free_nats,
                    self.kl_regularizer,
                    mask=batch.mask,
                    reward_mask=batch.reward_mask,
                )

            self.dyn_model_optimizer.zero_grad()
            rec_loss.backward()
            self.dyn_model_optimizer.step()

            totals += torch.stack([x.detach().double() for x in (rec_loss, reward_loss, kl, state_loss, observation_loss)])

        totals = (self.all_reduce_mean(totals) / train_steps).tolist()

        if self.logger is not None:
            for name, value in zip(("rec_loss", "reward_loss", "kl", "state_loss", "observation_loss"), totals):
                self.logger.scalar(f"scalars/{name}", value, self.iter_count)

        self.dyn_loss = totals[0]
        self.pretrained = True

    def all_reduce_mean(self, tensor):
        if self.is_distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size
        return tensor

    # ---- training loop -----------------------------------------------------------------------

    def _setup_run_dir(self):
        logdir = self.run_opts.logdir
        if not logdir:
            raise ValueError("training needs a log directory")
        self.log_dir = Path(logdir)
        if self.rank == 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            save_config({**self.cfg, "run": asdict(self.run_opts)}, self.log_dir / "config.yaml")
        wandb_cfg = dict(self.cfg.get("wandb") or {})
        if wandb_cfg.get("enabled") and not wandb_cfg.get("name"):
            wandb_cfg["name"] = self.run_opts.exp_name or self.log_dir.name
        # A resumed launch purges the events logged after the checkpoint on the env-step axis; the
        # iteration-axis tags (*/iter, scalars/*) hold smaller step values and are never purged.
        purge_step = self.step_count + 1 if self._resumed else 0
        self.logger = ExperimentLogger(self.log_dir / "tb", rank=self.rank,
                                       config={**self.cfg, "run": asdict(self.run_opts)}, wandb_cfg=wandb_cfg,
                                       purge_step=purge_step)

    def train(self) -> bool:
        """Train to ``train.max_epochs`` (from the resumed iteration, if any).

        Returns ``False`` when SIGTERM or SIGUSR1 stopped the run early: ``latest.pt`` then holds
        the last finished iteration. A run that is already at ``max_epochs`` only rewrites its
        final policy.
        """
        self.start_time = time.time()
        self._setup_run_dir()
        if self.iter_count >= self.max_epochs:
            log.info("the run is already at iteration %d (max_epochs=%d): writing final_policy.pt, nothing to train",
                     self.iter_count, self.max_epochs)
            self._save_final_outputs()
            self.close()
            return True
        with StopSignals() as stop:
            return self._train(stop)

    def _train(self, stop: StopSignals) -> bool:
        self.time_report.start_timer("algorithm")

        self._build_replay()
        if self.replay.pos == 0 and not self.replay.full:
            fill_steps = math.ceil(self.min_replay / self.num_envs)
            aligned, message = reset_alignment(fill_steps, self.max_episode_length, self.steps_num)
            (log.info if aligned else log.warning)(message)

        self.initialize_env()
        self._reset_episode_stats()
        if self._resumed and len(self.replay) > 0:
            log.info("resuming with the checkpointed replay buffer (%d rows per rank)", len(self.replay))
            self._start_rollout_from_reset()
        else:
            if self._resumed:
                updates = (f", then {self.dyn_udt} RSSM updates per iteration (the {self.init_dyn_udt}-update warm-up "
                           "is not rerun)" if self.finetune_rssm and self.pretrained else "")
                log.info("resuming without a replay buffer checkpoint: refilling %d rows per rank with the current "
                         "policy%s", self.min_replay, updates)
            self.fill_replay_buffer()

        def actor_closure():
            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")
            actor_loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.time_report.end_timer("backward simulation")

            with torch.no_grad():
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters())

                if torch.isnan(self.grad_norm_before_clip):
                    raise FloatingPointError(f"[rank {self.rank}] NaN actor gradient")

            self.time_report.end_timer("compute actor loss")

            return actor_loss

        for epoch in range(self.iter_count, self.max_epochs):
            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == 'linear':
                actor_lr = (self.actor_lr_schedule_min - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
                critic_lr = (self.critic_lr_schedule_min - self.critic_lr) * float(epoch / self.max_epochs) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = critic_lr
            else:
                lr = self.actor_lr

            # RSSM finetuning (DDP)
            self.time_report.start_timer("dyn model training")
            if self.finetune_rssm:
                self.train_dyn_model()
            else:
                self.dyn_loss = 0.
            self.time_report.end_timer("dyn model training")

            # train actor (DDP syncs gradients in backward)
            self.time_report.start_timer("actor training")
            self.actor_optimizer.step(actor_closure)
            self.time_report.end_timer("actor training")
            # The actor loss backpropagates through the RSSM and the critics; those parameter
            # gradients are not used.
            for p in self.rssm.parameters():
                p.grad = None

            # train alpha
            alpha_loss = (self.log_alpha.exp() * (self.ent_buf.view(-1) - self.target_entropy)).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            if self.is_distributed and self.log_alpha.grad is not None:
                dist.all_reduce(self.log_alpha.grad, op=dist.ReduceOp.SUM)
                self.log_alpha.grad /= self.world_size
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

            # train critic
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(self.batch_size, self.obs_buf, self.target_values)
            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.
            nonfinite_grads = torch.zeros((), dtype=torch.float32, device=self.device)
            for j in range(self.critic_iterations):
                total_critic_loss = 0.
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad()
                    with self.autocast():
                        training_critic_loss = self.compute_critic_loss(batch_sample)
                    training_critic_loss.backward()

                    # Non-finite critic gradients are zeroed (SHAC), and counted for the log.
                    for critic in self.critic:
                        for params in critic.parameters():
                            if params.grad is not None:
                                nonfinite_grads += (~torch.isfinite(params.grad)).sum()
                                params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    if self.truncate_grad:
                        for critic in self.critic:
                            clip_grad_norm_(critic.parameters(), self.grad_norm)

                    self.critic_optimizer.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1

                current_loss = (total_critic_loss / batch_cnt).detach()
                if self.is_distributed:
                    current_loss = self.all_reduce_mean(current_loss)
                self.value_loss = current_loss.cpu().item()
            self.time_report.end_timer("critic training")

            self.iter_count += 1

            time_end_epoch = time.time()

            # logging (metrics are averaged over ranks)
            actor_loss_t = self.all_reduce_mean(torch.tensor(self.actor_loss, device=self.device))
            self.actor_loss = actor_loss_t.item()
            ent_mean_val = self.all_reduce_mean(self.ent_buf.mean()).cpu().item()
            nonfinite_t = nonfinite_grads
            clamp_stats = torch.stack([self._clamp_hits, torch.tensor(float(self._clamp_total), device=self.device)])
            self._clamp_hits.zero_()
            self._clamp_total = 0
            if self.is_distributed:
                dist.all_reduce(nonfinite_t, op=dist.ReduceOp.SUM)
                dist.all_reduce(clamp_stats, op=dist.ReduceOp.SUM)

            if self.logger is not None:
                time_elapse = time.time() - self.start_time
                lg = self.logger
                lg.scalar('lr/iter', lr, self.iter_count)
                lg.scalar('actor_loss/step', self.actor_loss, self.step_count)
                lg.scalar('actor_loss/iter', self.actor_loss, self.iter_count)
                lg.scalar('value_loss/step', self.value_loss, self.step_count)
                lg.scalar('value_loss/iter', self.value_loss, self.iter_count)
                lg.scalar('value_loss/nonfinite_grads', nonfinite_t.item(), self.iter_count)
                lg.scalar('entropy/ent_mean/step', ent_mean_val, self.step_count)
                lg.scalar('entropy/ent_mean/iter', ent_mean_val, self.iter_count)
                lg.scalar('entropy/alpha/step', self.alpha, self.step_count)
                lg.scalar('entropy/alpha/iter', self.alpha, self.iter_count)
                lg.scalar('entropy/target_ent/step', self.target_entropy, self.step_count)
                lg.scalar('entropy/target_ent/iter', self.target_entropy, self.iter_count)
                if clamp_stats[1] > 0:
                    lg.scalar('env/clamp_rate/iter', (clamp_stats[0] / clamp_stats[1]).item(), self.iter_count)

            # Episode metrics, valid once every rank has finished an episode.
            have_episodes = torch.tensor(float(len(self.episode_loss_meter)), device=self.device)
            if self.is_distributed:
                dist.all_reduce(have_episodes, op=dist.ReduceOp.MIN)
            metrics = torch.tensor(np.array([
                self.episode_length_meter.get_mean(),
                self.episode_loss_meter.get_mean(),
                self.local_episode_loss_meter.get_mean(),
                self.episode_discounted_loss_meter.get_mean(),
            ], dtype=np.float32), device=self.device)
            metrics = self.all_reduce_mean(metrics)
            mean_episode_length, mean_policy_loss, mean_local_policy_loss, mean_policy_discounted_loss = metrics.tolist()
            if have_episodes.item() == 0:
                mean_policy_loss = mean_policy_discounted_loss = np.inf
                mean_episode_length = 0

            if have_episodes.item() > 0:
                if self.rank == 0:
                    if mean_policy_loss < self.best_policy_loss:
                        log.info("save best policy with loss %.2f", mean_policy_loss)
                        self.save_policy("best_policy")
                        self.best_policy_loss = mean_policy_loss

                if self.logger is not None:
                    lg = self.logger
                    lg.scalar('policy_loss/step', mean_policy_loss, self.step_count)
                    lg.scalar('policy_loss/time', mean_policy_loss, time_elapse)
                    lg.scalar('policy_loss/iter', mean_policy_loss, self.iter_count)
                    lg.scalar('rewards/step', -mean_policy_loss, self.step_count)
                    lg.scalar('rewards/time', -mean_policy_loss, time_elapse)
                    lg.scalar('rewards/iter', -mean_policy_loss, self.iter_count)
                    lg.scalar('local_rewards/step', -mean_local_policy_loss, self.step_count)
                    lg.scalar('local_rewards/time', -mean_local_policy_loss, time_elapse)
                    lg.scalar('local_rewards/iter', -mean_local_policy_loss, self.iter_count)
                    lg.scalar('policy_discounted_loss/step', mean_policy_discounted_loss, self.step_count)
                    lg.scalar('policy_discounted_loss/iter', mean_policy_discounted_loss, self.iter_count)
                    lg.scalar('best_policy_loss/step', self.best_policy_loss, self.step_count)
                    lg.scalar('best_policy_loss/iter', self.best_policy_loss, self.iter_count)
                    lg.scalar('episode_lengths/iter', mean_episode_length, self.iter_count)
                    lg.scalar('episode_lengths/step', mean_episode_length, self.step_count)
                    lg.scalar('episode_lengths/time', mean_episode_length, time_elapse)

            if self.rank == 0:
                memory = (f", peak GPU memory {torch.cuda.max_memory_allocated(self.device) / 2**30:.1f} GiB"
                          if self.device.type == "cuda" else "")
                log.info(
                    'iter {}: ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, fps total {:.2f}, '
                    'value loss {:.2f}, grad norm before clip {:.4f}, grad norm after clip {:.2f}, dyn loss {:.7f}, '
                    'alpha {:.4g}, entropy {:.2f}{}'.format(
                        self.iter_count, mean_policy_loss, mean_policy_discounted_loss, mean_episode_length,
                        self.steps_num * self.num_envs * self.world_size / (time_end_epoch - time_start_epoch),
                        self.value_loss, float(self.grad_norm_before_clip), float(self.grad_norm_after_clip),
                        self.dyn_loss, self.alpha, ent_mean_val, memory))
                self.logger.flush()

            # update the Polyak target critic (before the save: latest.pt holds this iteration's target)
            if not self.clipped_double_critic:
                with torch.no_grad():
                    alpha = self.target_critic_alpha
                    for param, param_targ in zip(self.critic_cores[0].parameters(), self.target_critic_net.parameters()):
                        param_targ.data.mul_(alpha)
                        param_targ.data.add_((1. - alpha) * param.data)

            saved = self.save_interval > 0 and self.iter_count % self.save_interval == 0
            if saved:
                self.save_training_state()
            if self._stop_requested(stop):
                if not saved:
                    self.save_training_state()
                log.warning("stop signal received: iteration %d is saved in %s; relaunch the same command to resume",
                            self.iter_count, self.log_dir / "latest.pt")
                self.close()
                return False

            if (self.iter_count % self.eval_every == 0) and not self.no_eval:
                self.evaluate_policy(self.num_eval_episodes)

        self.time_report.end_timer("algorithm")

        if self.rank == 0:
            self.time_report.report()
        self._save_final_outputs()
        if not (self.save_interval > 0 and self.iter_count % self.save_interval == 0):
            self.save_training_state()

        if not self.no_eval:
            self.evaluate_policy(self.num_eval_episodes)

        self.close()
        return True

    def _stop_requested(self, stop: StopSignals) -> bool:
        """Whether a stop signal reached any rank (a collective call)."""
        flag = torch.tensor(float(stop.poll() is not None), device=self.device)
        if self.is_distributed:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def _save_final_outputs(self):
        """``final_policy.pt`` and the episode histories (rank 0)."""
        if self.rank != 0:
            return
        self.save_policy('final_policy')
        np.save(self.log_dir / 'episode_loss_his.npy', np.array(self.episode_loss_his))
        np.save(self.log_dir / 'episode_discounted_loss_his.npy', np.array(self.episode_discounted_loss_his))
        np.save(self.log_dir / 'episode_length_his.npy', np.array(self.episode_length_his))

    # ---- play --------------------------------------------------------------------------------

    def play(self, checkpoint, num_games=None):
        """Evaluate the actor and RSSM of a policy or training checkpoint (deterministic)."""
        self.load_policy(checkpoint)
        num_games = int(num_games if num_games is not None else self.cfg["eval"]["player"]["games_num"])
        return self.evaluate_policy(num_games, restore_training_rollout=False)

    # ---- checkpoints -------------------------------------------------------------------------

    def policy_state(self) -> dict:
        """Weights and config needed to act and evaluate (actor, RSSM, critics)."""
        return {
            "format": POLICY_FORMAT,
            "format_version": FORMAT_VERSION,
            "config": self.cfg,
            "run": asdict(self.run_opts),
            "iter": self.iter_count,
            "step": self.step_count,
            "num_actions": self.num_actions,
            "drop_action_indices": list(self.drop_action_indices),
            "actor": _cpu_state(self.actor_core.state_dict()),
            "critics": [_cpu_state(c.state_dict()) for c in self.critic_cores],
            "target_critic": (None if self.clipped_double_critic else _cpu_state(self.target_critic_net.state_dict())),
            "rssm": _cpu_state(self.rssm_core.state_dict()),
            "rssm_config": dict(self.rssm_core.config),
            "rssm_warmstart": self.rssm_warmstart_info,
        }

    def save_policy(self, name):
        """Write ``<log_dir>/<name>.pt`` (call on rank 0)."""
        path = self.log_dir / f"{name}.pt"
        atomic_save(self.policy_state(), path)
        return path

    def save_training_state(self):
        """Write ``<log_dir>/latest.pt`` with everything needed to resume (collective call).

        With ``train.checkpoint_replay`` every rank also writes its replay buffer to
        ``<log_dir>/replay/rank<r>.pt``.
        """
        rank_state = {"rng": _rng_state(),
                      "env": self.env.state_dict() if hasattr(self.env, "state_dict") else None}
        if self.is_distributed:
            rank_states = [None] * self.world_size
            dist.all_gather_object(rank_states, rank_state)
        else:
            rank_states = [rank_state]
        if self.checkpoint_replay and self.replay is not None:
            atomic_save({"iter": self.iter_count, **self.replay.state_dict()},
                        self.log_dir / "replay" / f"rank{self.rank}.pt")
        if self.rank == 0:
            meters = {name: {"mean": m.mean.detach().cpu(), "current_size": m.current_size}
                      for name, m in self._meters().items()}
            state = {
                **self.policy_state(),
                "format": TRAINING_STATE_FORMAT,
                "world_size": self.world_size,
                "optimizers": {
                    "actor": self.actor_optimizer.state_dict(),
                    "critic": self.critic_optimizer.state_dict(),
                    "rssm": self.dyn_model_optimizer.state_dict(),
                    "alpha": self.a_optimizer.state_dict(),
                },
                "log_alpha": self.log_alpha.detach().cpu(),
                "counters": {"iter": self.iter_count, "step": self.step_count,
                             "best_policy_loss": float(self.best_policy_loss), "rssm_pretrained": self.pretrained},
                "meters": meters,
                "episode_history": {"loss": self.episode_loss_his, "discounted_loss": self.episode_discounted_loss_his,
                                    "length": self.episode_length_his},
                "rng": [r["rng"] for r in rank_states],
                "env": [r["env"] for r in rank_states],
                "replay_checkpointed": bool(self.checkpoint_replay and self.replay is not None),
            }
            atomic_save(state, self.log_dir / "latest.pt")
            log.info("saved training state at iteration %d", self.iter_count)
        barrier()

    def _meters(self):
        return {"episode_loss": self.episode_loss_meter, "local_episode_loss": self.local_episode_loss_meter,
                "episode_discounted_loss": self.episode_discounted_loss_meter,
                "episode_length": self.episode_length_meter}

    def load_policy(self, path):
        """Load actor and RSSM weights (and critics when present) from a policy or training checkpoint."""
        payload = load_checkpoint(path)
        if payload.get("format") not in (POLICY_FORMAT, TRAINING_STATE_FORMAT):
            raise ValueError(f"{path}: not a coupled_wm.rl checkpoint (format={payload.get('format')!r})")
        load_stripped_state_dict(self.actor_core, payload["actor"])
        load_stripped_state_dict(self.rssm_core, payload["rssm"])
        if len(payload["critics"]) != len(self.critic_cores):
            raise ValueError(f"{path}: {len(payload['critics'])} critics saved, {len(self.critic_cores)} configured")
        for critic, sd in zip(self.critic_cores, payload["critics"]):
            load_stripped_state_dict(critic, sd)
        if payload.get("target_critic") is not None and not self.clipped_double_critic:
            load_stripped_state_dict(self.target_critic_net, payload["target_critic"])
        log.info("loaded policy from %s (iteration %s)", path, payload.get("iter"))
        return payload

    def load_training_state(self, path):
        """Resume: networks, optimizers, temperature, counters, meters, RNG and the env's position
        in its seed-window sequence (and the replay buffer when it was checkpointed). Called by the
        constructor for ``RunOptions.resume``, or before :meth:`train`."""
        payload = self.load_policy(path)
        if payload["format"] != TRAINING_STATE_FORMAT:
            raise ValueError(f"{path}: a policy checkpoint cannot be resumed from (no optimizer state)")
        self._warn_config_changes(payload.get("config", {}))
        self.rssm_warmstart_info = payload.get("rssm_warmstart")
        self.actor_optimizer.load_state_dict(payload["optimizers"]["actor"])
        self.critic_optimizer.load_state_dict(payload["optimizers"]["critic"])
        self.dyn_model_optimizer.load_state_dict(payload["optimizers"]["rssm"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        self.a_optimizer.load_state_dict(payload["optimizers"]["alpha"])
        self.alpha = self.log_alpha.exp().item()
        counters = payload["counters"]
        self.iter_count = int(counters["iter"])
        self.step_count = int(counters["step"])
        self.best_policy_loss = float(counters["best_policy_loss"])
        self.pretrained = bool(counters["rssm_pretrained"])
        for name, meter in self._meters().items():
            meter.mean = payload["meters"][name]["mean"].to(self.device)
            meter.current_size = int(payload["meters"][name]["current_size"])
        history = payload["episode_history"]
        self.episode_loss_his = list(history["loss"])
        self.episode_discounted_loss_his = list(history["discounted_loss"])
        self.episode_length_his = list(history["length"])

        same_world = int(payload["world_size"]) == self.world_size
        replay_file = Path(path).parent / "replay" / f"rank{self.rank}.pt"
        if payload.get("replay_checkpointed") and self.checkpoint_replay and not same_world:
            log.warning("the replay buffer was checkpointed with world size %d, now %d: it is refilled",
                        payload["world_size"], self.world_size)
        elif payload.get("replay_checkpointed") and self.checkpoint_replay:
            replay_state = load_checkpoint(replay_file) if replay_file.is_file() else None
            if replay_state is not None and int(replay_state["iter"]) == self.iter_count:
                self._build_replay()
                self.replay.load_state_dict(replay_state)
            else:
                log.warning("replay buffer checkpoint %s missing or from another iteration: the replay buffer is "
                            "refilled", replay_file)

        if same_world:
            _set_rng_state(payload["rng"][self.rank])
        else:
            log.warning("checkpoint was written with world size %d, now %d: RNG state not restored",
                        payload["world_size"], self.world_size)
        self._restore_env_state(payload.get("env"), same_world)
        self._resumed = True
        log.info("resumed from %s at iteration %d", path, self.iter_count)

    def _restore_env_state(self, env_states, same_world):
        """Continue the env's seed-window sequence; otherwise the resumed run would redraw it."""
        reason = None
        if not same_world:
            reason = "the world size changed"
        elif not env_states or env_states[self.rank] is None or not hasattr(self.env, "load_state_dict"):
            reason = "the checkpoint or the env has no env state"
        else:
            try:
                self.env.load_state_dict(env_states[self.rank])
            except ValueError as e:
                reason = str(e)
        if reason is None:
            log.info("the environment continues its seed-window sequence from the checkpoint")
        else:
            log.warning("the environment's seed-window sequence is not restored (%s): the resumed run may seed its "
                        "episodes from windows the original run already used", reason)

    def _warn_config_changes(self, saved_cfg):
        def flat(d, prefix=""):
            out = {}
            for k, v in (d or {}).items():
                if isinstance(v, Mapping):
                    out.update(flat(v, f"{prefix}{k}."))
                else:
                    out[f"{prefix}{k}"] = v
            return out

        old, new = flat(saved_cfg), flat(self.cfg)
        changed = sorted(k for k in set(old) | set(new) if old.get(k, "<absent>") != new.get(k, "<absent>"))
        if changed:
            log.warning("config differs from the checkpoint's in: %s", ", ".join(changed))

    def close(self):
        if self.logger is not None:
            self.logger.close()
        if hasattr(self.env, "close"):
            self.env.close()
