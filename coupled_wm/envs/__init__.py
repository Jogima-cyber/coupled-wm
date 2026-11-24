# SPDX-License-Identifier: Apache-2.0
"""World-model environments: the G1 DreamerV4 environment the DMO trainer rolls out in."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from coupled_wm.envs.g1_env import DROPPED_DIMS_RESET, DreamerV4VecEnvG1Relative, SeedWindowDataset, draw_seed_pass

ENVS = {"DreamerV4VecEnvG1Relative": DreamerV4VecEnvG1Relative}


def make_env(cfg: Mapping[str, Any], *, num_envs: int, device: torch.device | str, episode_length: int):
    """Build the environment named by ``cfg["name"]`` from its ``env`` config block."""
    name = cfg.get("name")
    if name not in ENVS:
        raise KeyError(f"unknown env name {name!r}; available: {sorted(ENVS)}")
    return ENVS[name].from_config(cfg, num_envs=num_envs, device=device, episode_length=episode_length)


__all__ = ["DROPPED_DIMS_RESET", "ENVS", "DreamerV4VecEnvG1Relative", "SeedWindowDataset", "draw_seed_pass", "make_env"]
