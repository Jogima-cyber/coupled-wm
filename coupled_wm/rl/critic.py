# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Scalar MLP value function (MSE-trained)."""

import numpy as np
import torch
import torch.nn as nn

from coupled_wm.rl import model_utils


class CriticMLP(nn.Module):
    """``V(latent)``: Linear -> activation -> LayerNorm blocks, orthogonal init, scalar output.

    ``cfg_critic`` keys: ``units`` (hidden sizes), ``activation``.
    """

    def __init__(self, obs_dim, cfg_critic, device='cuda:0'):
        super(CriticMLP, self).__init__()

        self.device = device
        self.layer_dims = [obs_dim] + list(cfg_critic['units']) + [1]

        init_ = lambda m: model_utils.init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(init_(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1])))
            if i < len(self.layer_dims) - 2:
                modules.append(model_utils.get_activation_func(cfg_critic['activation']))
                modules.append(torch.nn.LayerNorm(self.layer_dims[i + 1]))

        self.critic = nn.Sequential(*modules).to(device)
        self.obs_dim = obs_dim

    def forward(self, observations):
        return self.critic(observations)


CRITICS = {"CriticMLP": CriticMLP}


def build_critic(cfg_critic, obs_dim, device):
    name = cfg_critic.get("name", "CriticMLP")
    if name not in CRITICS:
        raise ValueError(f"unknown critic {name!r}; available: {sorted(CRITICS)}")
    return CRITICS[name](obs_dim, cfg_critic, device=device)
