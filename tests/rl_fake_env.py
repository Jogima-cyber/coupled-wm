# SPDX-License-Identifier: Apache-2.0
"""A tiny CPU stand-in for the G1 world-model environment (tests only) and a matching RL config.

``FakeWorldModelEnv`` has the API the DMO trainer uses: ``num_envs``, ``num_actions``,
``img_size``, ``episode_length``, ``img0``, ``reset(env_ids)`` and ``step(actions) -> (obs,
reward, done, info)`` with ``info["obs_before_reset"]``, ``info["dones"]`` and
``info["clamped"]``, plus the optional ``state_dict()`` / ``load_state_dict()`` (the position in
the sequence of episode seeds) and ``color_order``. All envs share one clock (batch-synchronous
time-limit dones). The state is a clamped integral of the actions, rendered into a small image;
the reward depends on the state. Each reset draws new states from the env's generator (its "seed
windows"). It records every reset and step in ``history`` so tests can rebuild what the trainer
should have stored.
"""

from __future__ import annotations

import copy

import torch

try:
    import pytest
except ImportError:  # the env is also used by command-line runs outside pytest
    pytest = None


def _single_thread():
    """Tiny CPU models run fastest on one thread (and do not fight other processes for cores)."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


single_thread = pytest.fixture(autouse=True, scope="module")(_single_thread) if pytest else None


class FakeWorldModelEnv:
    def __init__(self, num_envs, device, episode_length, img_size=16, num_actions=5, drop_action_indices=(),
                 seed=0, color_order=None, **unused):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.episode_length = int(episode_length)
        self.img_size = int(img_size)
        self.num_actions = int(num_actions)
        self.drop_action_indices = tuple(drop_action_indices)
        self.color_order = color_order
        self.gen = torch.Generator().manual_seed(seed)
        pix = 3 * self.img_size * self.img_size
        self.render_w = torch.randn(self.num_actions, pix, generator=self.gen) * 1.5
        self.render_b = torch.randn(pix, generator=self.gen) * 0.3
        self.target = torch.linspace(-0.5, 0.5, self.num_actions)
        self.history = []
        self.closed = False
        self.reset()

    def _render(self, state):
        img = torch.sigmoid(state @ self.render_w + self.render_b)
        return img.view(-1, 3, self.img_size, self.img_size).to(self.device)

    def reset(self, env_ids=None):
        self.traj_steps = 0
        self.state = torch.rand(self.num_envs, self.num_actions, generator=self.gen) - 0.5
        self.img0 = self._render(self.state)
        self.history.append({"kind": "reset", "obs": self.img0.clone()})
        return self.img0

    @torch.no_grad()
    def step(self, actions):
        actions = actions.detach().to(torch.float32).cpu().reshape(self.num_envs, self.num_actions)
        integrated = self.state + actions
        self.state = integrated.clamp(-1.0, 1.0)
        clamped = self.state != integrated
        obs = self._render(self.state)
        reward = (1.0 - ((self.state - self.target) ** 2).sum(-1)).to(self.device)
        self.traj_steps += 1
        done_flag = self.traj_steps >= self.episode_length
        done = torch.full((self.num_envs,), done_flag, dtype=torch.bool, device=self.device)
        obs_before_reset = obs.clone()
        self.history.append({"kind": "step", "action": actions.clone(), "reward": reward.clone(),
                             "obs": obs_before_reset.clone(), "done": done_flag})
        if done_flag:
            obs = self.reset().clone()
        info = {"obs_before_reset": obs_before_reset, "dones": done, "clamped": clamped.to(self.device)}
        return obs, reward, done, info

    def state_dict(self):
        """Where the env is in its sequence of episode seeds."""
        return {"generator": self.gen.get_state(), "num_envs": self.num_envs}

    def load_state_dict(self, state):
        if state["num_envs"] != self.num_envs:
            raise ValueError(f"the env state is for {state['num_envs']} envs, this env has {self.num_envs}")
        self.gen.set_state(state["generator"])

    def close(self):
        self.closed = True


def tiny_config(**overrides):
    """A small RL config for the fake env (4 actors, 16x16 frames, tiny RSSM)."""
    cfg = {
        "env": {
            "name": "rl_fake_env:FakeWorldModelEnv",
            "img_size": 16,
            "num_actions": 5,
            "drop_action_indices": [0],
            "seed": 0,
        },
        "actor": {
            "name": "SAPOActorStochasticMLP",
            "units": [16, 16],
            "activation": "silu",
            "action_scale": [0.2, 0.2, 0.2, 1.0],
            "log_std_min": -5.0,
            "log_std_max": 2.0,
        },
        "critic": {
            "name": "CriticMLP",
            "units": [16, 16],
            "activation": "silu",
            "clipped_double_critic": True,
            "num_critics": 3,
            "target_critic_alpha": 0.2,
        },
        "rssm": {
            "pretrained": None,
            "straight_through_stochastic_grad": False,
            "sequence_length": 8,
            "batch_size": 4,
            "stochastic_size": 4,
            "discrete_size": 4,
            "recurrent_state_size": 16,
            "dense_units": 16,
            "channels_multiplier": 2,
            "mlp_layers": 2,
            "reward_model_bins": 31,
            "kl_dynamic": 0.5,
            "kl_representation": 0.1,
            "kl_free_nats": 1.0,
            "kl_regularizer": 1.0,
            "learning_rate": 1e-3,
            "init_udt": 3,
            "udt": 2,
        },
        "train": {
            "max_epochs": 3,
            "num_actors": 4,
            "max_episode_length": 8,
            "steps_num": 4,
            "gamma": 0.99,
            "critic_method": "td-lambda",
            "lambda": 0.95,
            "lr_schedule": "linear",
            "actor_learning_rate": 2e-3,
            "actor_lr_schedule_min": 1e-5,
            "critic_learning_rate": 5e-4,
            "critic_lr_schedule_min": 1e-5,
            "betas": [0.7, 0.95],
            "critic_iterations": 2,
            "num_batch": 2,
            "grad_norm": 1.0,
            "truncate_grads": True,
            "target_entropy": -8.0,
            "alpha_learning_rate": 5e-3,
            "replay_size": 400,
            "min_replay": 32,
            "replay_device": "cuda",
            "checkpoint_replay": False,
            "save_interval": 2,
            "compile": False,
        },
        "eval": {"num_episodes": 4, "every": 2, "player": {"num_actors": 4, "games_num": 4}},
        "wandb": {"enabled": False},
    }
    cfg = copy.deepcopy(cfg)
    for path, value in overrides.items():
        node = cfg
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = value
    return cfg
