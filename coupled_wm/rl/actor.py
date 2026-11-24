# SPDX-License-Identifier: Apache-2.0
"""Stochastic tanh-squashed Gaussian actor (SAPO-style) and its entropy bound.

The actor outputs a pre-tanh sample ``u``; the trainer applies ``tanh(u) * action_scale``. The
second output is a single-sample estimate of the entropy of the squashed, scaled action,
``-log N(u; mu, std) + sum_d log(action_scale_d * (1 - tanh(u_d)^2) + 1e-6)``.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from coupled_wm.rl import model_utils


class SAPOActorStochasticMLP(nn.Module):
    """MLP actor with a state-dependent, tanh-bounded log-std.

    ``cfg_actor`` keys: ``units`` (hidden sizes), ``activation``, ``action_scale`` (scalar or a
    list of ``action_dim`` per-dim scales), ``log_std_min``, ``log_std_max``.
    """

    def __init__(self, obs_dim, action_dim, cfg_actor, device='cuda:0'):
        super(SAPOActorStochasticMLP, self).__init__()

        self.device = device
        self.layer_dims = [obs_dim] + list(cfg_actor['units'])

        modules = []
        for i in range(len(self.layer_dims) - 1):
            modules.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            modules.append(model_utils.get_activation_func(cfg_actor['activation']))
            modules.append(torch.nn.LayerNorm(self.layer_dims[i+1]))

        self.mlp = nn.Sequential(*modules).to(device)
        self.fc_mean = nn.Linear(self.layer_dims[-1], action_dim).to(device)
        self.fc_logstd = nn.Linear(self.layer_dims[-1], action_dim).to(device)

        self.action_dim = action_dim
        self.obs_dim = obs_dim

        # action_scale is a config value, not learned state: a plain attribute rather than a
        # buffer, because DDP's buffer broadcast would copy_() into it every forward and break
        # backward through the per-dim multiplier. _apply keeps it on the parameters' device.
        asc = cfg_actor.get("action_scale", 1.0)
        if isinstance(asc, (list, tuple)):
            if len(asc) != action_dim:
                raise ValueError(
                    f"actor.action_scale has {len(asc)} entries but the policy controls {action_dim} dims (the env's "
                    f"actions minus env.drop_action_indices); env.action_scale is the env's absolute command clamp")
            self.action_scale = torch.tensor(asc, dtype=torch.float32, device=device)
        else:
            self.action_scale = float(asc)
        self.LOG_STD_MIN, self.LOG_STD_MAX = log_std_bounds(cfg_actor)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        if isinstance(self.action_scale, torch.Tensor):
            self.action_scale = self.action_scale.to(self.fc_mean.weight.device)
        return self

    def action_scale_list(self):
        """Per-dim action scales as a list of ``action_dim`` floats."""
        if isinstance(self.action_scale, torch.Tensor):
            return [float(s) for s in self.action_scale.cpu()]
        return [float(self.action_scale)] * self.action_dim

    def max_entropy(self):
        """Maximum attainable value of the expected entropy estimate (see :func:`max_attainable_entropy`)."""
        return max_attainable_entropy(self.action_scale_list(), self.LOG_STD_MIN, self.LOG_STD_MAX)

    def forward(self, obs, deterministic = False):
        """Return ``(u, entropy_estimate)``; deterministic mode returns ``(mean, None)``."""
        h = self.mlp(obs)
        mean = self.fc_mean(h)

        if deterministic:
            return mean, None

        log_std = self.fc_logstd(h)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        std = log_std.exp()
        dist = Normal(mean, std)
        sample = dist.rsample()  # reparameterization trick (mean + std * N(0,1))

        log_prob = dist.log_prob(sample)
        log_prob -= torch.log(self.action_scale * (1 - torch.tanh(sample).pow(2)) + 1e-6)
        entropy = -log_prob.sum(-1)
        return sample, entropy


def log_std_bounds(cfg_actor):
    """``(log_std_min, log_std_max)`` of an actor config."""
    return float(cfg_actor.get("log_std_min", -5.0)), float(cfg_actor.get("log_std_max", 2.0))


def config_max_entropy(cfg_actor):
    """:func:`max_attainable_entropy` of the actor ``cfg_actor`` describes, or ``None`` when its
    ``action_scale`` is a scalar (the bound then depends on the env's action count)."""
    scale = cfg_actor.get("action_scale", 1.0)
    if not isinstance(scale, (list, tuple)):
        return None
    return max_attainable_entropy(scale, *log_std_bounds(cfg_actor))


ACTORS = {"SAPOActorStochasticMLP": SAPOActorStochasticMLP}


def build_actor(cfg_actor, obs_dim, action_dim, device):
    name = cfg_actor.get("name", "SAPOActorStochasticMLP")
    if name not in ACTORS:
        raise ValueError(f"unknown actor {name!r}; available: {sorted(ACTORS)}")
    return ACTORS[name](obs_dim, action_dim, cfg_actor, device=device)


_HALF_LOG_2PIE = 0.5 * math.log(2.0 * math.pi * math.e)


def _expected_log_jacobian(scale, mu, sigma, n_nodes=160):
    """``E_{u ~ N(mu, sigma)}[log(scale * (1 - tanh(u)^2) + 1e-6)]`` by Gauss-Hermite quadrature.

    ``mu`` and ``sigma`` broadcast; computed in float64 with a stable ``log sech^2``.
    """
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / math.sqrt(2.0 * math.pi)
    u = np.asarray(mu)[..., None] + np.asarray(sigma)[..., None] * x
    a = np.abs(u)
    log_sech2 = 2.0 * (math.log(2.0) - a - np.log1p(np.exp(-2.0 * a)))
    vals = np.logaddexp(math.log(scale) + log_sech2, math.log(1e-6))
    return (vals * w).sum(-1)


def max_attainable_entropy(action_scale, log_std_min, log_std_max):
    """Upper bound of the expected entropy estimate the actor can reach.

    Per dimension, the expectation of ``-log N(u; mu, std) + log(s (1 - tanh(u)^2) + 1e-6)`` is
    maximised over ``mu`` and ``log std`` in ``[log_std_min, log_std_max]`` (the actor's bounds);
    the total is the sum over dimensions. A ``target_entropy`` at or above this value cannot be
    reached and makes the temperature diverge. For bounds containing ``log std = -0.13`` the
    per-dimension maximum is ``log(s) + 0.6836`` at ``mu = 0``.
    """
    log_stds = np.linspace(log_std_min, log_std_max, int(round((log_std_max - log_std_min) / 0.005)) + 1)
    mus = np.linspace(-1.5, 1.5, 13)
    total = 0.0
    cache = {}
    for s in action_scale:
        s = float(s)
        if s <= 0:
            raise ValueError(f"action_scale entries must be positive, got {s}")
        if s not in cache:
            mu_g, ls_g = np.meshgrid(mus, log_stds, indexing="ij")
            ent = _HALF_LOG_2PIE + ls_g + _expected_log_jacobian(s, mu_g, np.exp(ls_g))
            cache[s] = float(ent.max())
        total += cache[s]
    return total
