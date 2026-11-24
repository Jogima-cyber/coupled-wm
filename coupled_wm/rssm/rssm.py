# SPDX-License-Identifier: Apache-2.0
#
# Adapted from sheeprl's DreamerV3 implementation (https://github.com/Eclectic-Sheep/sheeprl,
# Apache-2.0), itself a port of DreamerV3 (https://github.com/danijar/dreamerv3, MIT);
# LayerNormGRUCell follows DreamerV2 (https://github.com/danijar/dreamerv2, MIT).
"""DreamerV3-style RSSM: the local, low-dimensional world model.

The RSSM supplies the backward pass of DMO. In each actor step the forward latent is the true
posterior of the frame produced by the global world model, and the gradient is routed through
the RSSM's imagined prior instead (:class:`GradientSwapingFunction`).

Conventions:

* Observations enter as RGB float images in ``[-0.5, 0.5]`` (``frame / 255 - 0.5``), shape
  ``(..., C, H, W)`` with ``H = W = image_size``.
* A latent is ``cat(posterior, h)``. The posterior is ``stochastic_size`` one-hot categoricals of
  ``discrete_size`` classes, flattened. ``h`` is the deterministic GRU state of size
  ``recurrent_state_size``.
* Row ``t`` of a training sequence is ``(o_t, a_t, r_t, is_first_t)``. ``a_t`` is the action that
  led *into* ``o_t``. ``is_first_t = 1`` resets the state to the learned initial state and zeroes
  ``a_t``.
* The reward head is a two-hot classifier over ``reward_model_bins`` bins, uniform in symlog
  space on ``[-20, 20]`` (:class:`TwoHotEncodingDistribution`).

The submodule names reproduce the paper checkpoint's state-dict keys after
:func:`coupled_wm.common.strip_compile_prefix`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.distributions import Independent, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence
from torch.distributions.utils import probs_to_logits

from coupled_wm.common.checkpoint import (
    is_pretrained_dir,
    load_pretrained,
    save_pretrained,
    strip_compile_prefix,
)

MODEL_TYPE = "rssm"

RELATIVE_ACTION_DEFINITION = "commands[t] - commands[t-1]"
"""Action of the row of frame ``t`` for relative-action RSSMs (pretraining and online finetuning),
recorded as ``action.definition`` in a pretrained dir's ``config.json``."""

# Architecture keys of a config dict (see :func:`rssm_config`).
REQUIRED_KEYS = (
    "action_size",
    "image_size",
    "stochastic_size",
    "discrete_size",
    "recurrent_state_size",
    "dense_units",
    "channels_multiplier",
    "mlp_layers",
    "reward_model_bins",
)
OPTIONAL_DEFAULTS = {"image_channels": 3, "unimix": 0.01, "learnable_initial_recurrent_state": True}
DERIVED_KEYS = ("bottleneck_spatial", "cnn_output_dim")
ARCH_KEYS = REQUIRED_KEYS + tuple(OPTIONAL_DEFAULTS) + DERIVED_KEYS


# ---------------------------------------------------------------------------------------------
# Image encoder / decoder
# ---------------------------------------------------------------------------------------------


class LayerNormChannelLast(nn.LayerNorm):
    """LayerNorm over the channel dimension of an NCHW tensor."""

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(f"Input tensor must be 4D (NCHW), received {x.dim()}D instead: {x.shape}")
        input_dtype = x.dtype
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.to(input_dtype)


def _conv_block(conv: nn.Module, channels: int) -> list[nn.Module]:
    return [conv, LayerNormChannelLast((channels,), eps=0.001, elementwise_affine=True), nn.SiLU()]


class CNNEncoder(nn.Module):
    """Four stride-2 convolutions: ``(N, C, S, S) -> (N, 8m, S/16, S/16)`` with ``m = channels_multiplier``."""

    def __init__(self, channels_multiplier: int = 32, image_channels: int = 3):
        super().__init__()
        m = channels_multiplier
        widths = [image_channels, m, 2 * m, 4 * m, 8 * m]
        layers: list[nn.Module] = []
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            conv = nn.Conv2d(c_in, c_out, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
            layers += _conv_block(conv, c_out)
        self._model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)


class DeCNN(nn.Module):
    """Four stride-2 transposed convolutions, the mirror of :class:`CNNEncoder`."""

    def __init__(self, channels_multiplier: int = 32, image_channels: int = 3) -> None:
        super().__init__()
        m = channels_multiplier
        widths = [8 * m, 4 * m, 2 * m, m]
        layers: list[nn.Module] = []
        for c_in, c_out in zip(widths[:-1], widths[1:]):
            conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
            layers += _conv_block(conv, c_out)
        layers.append(nn.ConvTranspose2d(m, image_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
        self._model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)


class CNNDecoder(nn.Module):
    """Latent ``(N, S*D + R)`` -> image ``(N, C, 16b, 16b)``, ``b = bottleneck_spatial``.

    ``cnn_output_dim`` must equal ``8 * channels_multiplier * bottleneck_spatial**2``.
    """

    def __init__(
        self,
        stochastic_size: int,
        discrete: int,
        recurrent_state_size: int,
        cnn_output_dim: int,
        channels_multiplier: int = 32,
        bottleneck_spatial: int = 4,
        image_channels: int = 3,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(stochastic_size * discrete + recurrent_state_size, cnn_output_dim, bias=True),
            nn.Unflatten(dim=1, unflattened_size=(-1, bottleneck_spatial, bottleneck_spatial)),
            DeCNN(channels_multiplier=channels_multiplier, image_channels=image_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------------------------
# Recurrent core
# ---------------------------------------------------------------------------------------------


class LayerNormGRUCell(nn.Module):
    """GRU cell with a LayerNorm after the joint state/input projection, no bias (DreamerV2).

    ``input (B, input_size)`` with ``hx (B, hidden_size)``, or a sequence of length 1:
    ``input (1, B, input_size)`` with ``hx (1, B, hidden_size)`` or ``(B, hidden_size)``. The new
    state has the leading dimension of ``input``.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(3 * hidden_size, eps=0.001, elementwise_affine=True)

    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        is_3d = input.dim() == 3
        if is_3d:
            if input.shape[0] != 1:
                raise ValueError(f"LayerNormGRUCell: a 3-D input must be a sequence of length 1, got {input.shape[0]}")
            input = input.squeeze(0)
        if hx.dim() == 3:
            hx = hx.squeeze(0)
        if input.dim() != 2 or hx.dim() != 2:
            raise ValueError(f"LayerNormGRUCell: expected (B, X) inputs, got input {tuple(input.shape)}, hx {tuple(hx.shape)}")

        x = self.linear(torch.cat((hx, input), -1))
        x = self.layer_norm(x)
        reset, cand, update = torch.chunk(x, 3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        hx = update * cand + (1 - update) * hx
        return hx.unsqueeze(0) if is_3d else hx


class RecurrentModel(nn.Module):
    """``h_t = GRU(MLP(cat(z_{t-1}, a_t)), h_{t-1})``."""

    def __init__(self, action_size: int, stochastic_size: int, recurrent_state_size: int, dense_units: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_size + stochastic_size, dense_units, bias=False),
            nn.LayerNorm((dense_units,), eps=0.001, elementwise_affine=True),
            nn.SiLU(),
        )
        self.rnn = LayerNormGRUCell(input_size=dense_units, hidden_size=recurrent_state_size)

    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        return self.rnn(self.mlp(input), hx)


def compute_stochastic_state(
    logits: Tensor, discrete: int = 32, sample: bool = True, straight_through: bool = False
) -> Tensor:
    """Stochastic state ``(..., S, D)`` from flat logits ``(..., S*D)``.

    * ``sample=True``: a one-hot sample with the straight-through gradient of the probabilities.
    * ``sample=False``: the mode (argmax one-hot). It carries no gradient unless
      ``straight_through=True``, in which case it is ``mode + (probs - probs.detach())``: the
      forward value is bit-identical to the mode and the gradient is that of ``probs``.
    """
    logits = logits.view(*logits.shape[:-1], -1, discrete)
    dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
    if sample:
        return dist.rsample()
    mode = dist.mode
    if straight_through:
        probs = dist.base_dist.probs
        mode = mode + (probs - probs.detach())
    return mode


# ---------------------------------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------------------------------


class RSSM(nn.Module):
    """Recurrent state-space model with image encoder/decoder and a two-hot reward head.

    The constructor records its architecture in :attr:`config` (the keys of :data:`ARCH_KEYS`),
    which is what :func:`save_rssm` writes to ``config.json``. Use :func:`build_rssm` to build
    one from a config dict.
    """

    def __init__(
        self,
        action_size: int,
        stochastic_size: int,
        discrete: int,
        recurrent_state_size: int,
        dense_units: int,
        cnn_output_dim: int,
        channels_multiplier: int,
        mlp_layers: int,
        reward_model_bins: int,
        learnable_initial_recurrent_state: bool = True,
        bottleneck_spatial: int = 4,
        image_channels: int = 3,
        unimix: float = 0.01,
    ):
        super().__init__()
        if cnn_output_dim != 8 * channels_multiplier * bottleneck_spatial**2:
            raise ValueError(
                f"cnn_output_dim={cnn_output_dim} must equal 8 * channels_multiplier * bottleneck_spatial**2 = "
                f"{8 * channels_multiplier * bottleneck_spatial ** 2}"
            )
        self.config = {
            "action_size": int(action_size),
            "image_size": 16 * int(bottleneck_spatial),
            "stochastic_size": int(stochastic_size),
            "discrete_size": int(discrete),
            "recurrent_state_size": int(recurrent_state_size),
            "dense_units": int(dense_units),
            "channels_multiplier": int(channels_multiplier),
            "mlp_layers": int(mlp_layers),
            "reward_model_bins": int(reward_model_bins),
            "image_channels": int(image_channels),
            "unimix": float(unimix),
            "learnable_initial_recurrent_state": bool(learnable_initial_recurrent_state),
            "bottleneck_spatial": int(bottleneck_spatial),
            "cnn_output_dim": int(cnn_output_dim),
        }
        self.unimix = float(unimix)
        self.discrete = discrete
        self.recurrent_state_size = recurrent_state_size
        self.stochastic_size = stochastic_size

        self.encoder = CNNEncoder(channels_multiplier=channels_multiplier, image_channels=image_channels)
        self.decoder = CNNDecoder(
            stochastic_size=stochastic_size,
            discrete=discrete,
            recurrent_state_size=recurrent_state_size,
            cnn_output_dim=cnn_output_dim,
            channels_multiplier=channels_multiplier,
            bottleneck_spatial=bottleneck_spatial,
            image_channels=image_channels,
        )
        self.recurrent_model = RecurrentModel(
            action_size=action_size,
            stochastic_size=stochastic_size * discrete,
            recurrent_state_size=recurrent_state_size,
            dense_units=dense_units,
        )
        self.representation_model = nn.Sequential(
            nn.Linear(cnn_output_dim + recurrent_state_size, dense_units, bias=False),
            nn.LayerNorm((dense_units,), eps=0.001, elementwise_affine=True),
            nn.SiLU(),
            nn.Linear(dense_units, stochastic_size * discrete, bias=True),
        )
        self.transition_model = nn.Sequential(
            nn.Linear(recurrent_state_size, dense_units, bias=False),
            nn.LayerNorm((dense_units,), eps=0.001, elementwise_affine=True),
            nn.SiLU(),
            nn.Linear(dense_units, stochastic_size * discrete, bias=True),
        )

        layers: list[nn.Module] = [
            nn.Linear(stochastic_size * discrete + recurrent_state_size, dense_units, bias=False),
            nn.LayerNorm((dense_units,), eps=0.001, elementwise_affine=True),
            nn.SiLU(),
        ]
        for _ in range(mlp_layers - 1):
            layers += [
                nn.Linear(dense_units, dense_units, bias=False),
                nn.LayerNorm((dense_units,), eps=0.001, elementwise_affine=True),
                nn.SiLU(),
            ]
        layers.append(nn.Linear(dense_units, reward_model_bins, bias=True))
        self.reward_model = nn.Sequential(*layers)

        if learnable_initial_recurrent_state:
            self.initial_recurrent_state = nn.Parameter(torch.zeros(recurrent_state_size, dtype=torch.float32))
        else:
            self.register_buffer("initial_recurrent_state", torch.zeros(recurrent_state_size, dtype=torch.float32))

    def get_initial_states(self, batch_shape: Sequence[int] | torch.Size) -> tuple[Tensor, Tensor]:
        """Learned initial ``h`` and the mode of its prior, broadcast to ``batch_shape``."""
        initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)
        initial_posterior = self._transition(initial_recurrent_state, sample_state=False)[1]
        return initial_recurrent_state, initial_posterior

    def dynamic(
        self, posterior: Tensor, recurrent_state: Tensor, action: Tensor, embedded_obs: Tensor, is_first: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One training step with a sampled posterior.

        Shapes: ``posterior (1, B, S, D)``, ``recurrent_state (1, B, R)``, ``action (1, B, A)``,
        ``embedded_obs (1, B, E)``, ``is_first (1, B, 1)``. Returns ``(h, posterior, prior,
        posterior_logits, prior_logits)``.
        """
        action = (1 - is_first) * action

        initial_recurrent_state, initial_posterior = self.get_initial_states(recurrent_state.shape[:2])
        recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
        posterior = posterior.view(*posterior.shape[:-2], -1)
        posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)

        recurrent_state = self.recurrent_model(torch.cat((posterior, action), -1), recurrent_state)
        prior_logits, prior = self._transition(recurrent_state)
        posterior_logits, posterior = self._representation(recurrent_state, embedded_obs)
        return recurrent_state, posterior, prior, posterior_logits, prior_logits

    def latent_projector(
        self, posterior: Tensor, recurrent_state: Tensor, action: Tensor, batch_obs: Tensor, is_first: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Filter one observed frame: ``(h_t, z_t)`` with ``z_t`` the posterior mode.

        Shapes: ``posterior (1, B, S*D)``, ``recurrent_state (1, B, R)``, ``action (1, B, A)``,
        ``batch_obs (B, C, H, W)``, ``is_first (1, B)``. ``h_t`` is differentiable in the action
        and the previous state; the encoder and the posterior run under ``no_grad``, so ``z_t``
        is a constant (the "true posterior" of DMO).
        """
        action = (1 - is_first).unsqueeze(-1) * action

        initial_recurrent_state, initial_posterior = self.get_initial_states(recurrent_state.shape[:2])
        keep, reset = (1 - is_first).unsqueeze(-1), is_first.unsqueeze(-1)
        recurrent_state = keep * recurrent_state + reset * initial_recurrent_state
        posterior = keep * posterior + reset * initial_posterior.view_as(posterior)

        bs = batch_obs.size(0)
        with torch.no_grad():
            embedded_obs = self.encoder(batch_obs)
        embedded_obs = embedded_obs.reshape(1, bs, -1)

        recurrent_state = self.recurrent_model(torch.cat((posterior, action), -1), recurrent_state)
        with torch.no_grad():
            _, posterior = self._representation(recurrent_state, embedded_obs, sample_state=False)
        posterior = posterior.view(*posterior.shape[:-2], -1)
        return recurrent_state, posterior

    def imagined_prior(self, recurrent_state: Tensor, straight_through: bool = False) -> Tensor:
        """Prior mode ``(..., S*D)`` given ``h``: the backward path of the DMO gradient swap.

        ``straight_through=False`` (default, the paper's setting) returns the prior mode; the swap
        then routes the actor gradient through ``h``. ``straight_through=True`` adds the
        straight-through gradient of the prior probabilities; the forward value is unchanged.
        """
        _, prior = self._transition(recurrent_state, sample_state=False, straight_through=straight_through)
        return prior.view(*prior.shape[:-2], -1)

    def imagination(
        self,
        prior: Tensor,
        recurrent_state: Tensor,
        actions: Tensor,
        sample_state: bool = True,
        *,
        straight_through: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """One step of latent imagination: ``(z_{t+1}, h_{t+1})`` from ``(z_t, h_t, a)``, no observation.

        ``z_{t+1}`` is ``(..., S, D)``. With ``sample_state=False`` it is the prior mode, and
        ``straight_through`` acts as in :meth:`imagined_prior`: ``False`` (default) gives no
        gradient through ``z_{t+1}``, ``True`` adds the straight-through gradient of the prior
        probabilities with the same forward value. A sample (``sample_state=True``) always carries
        that gradient.
        """
        recurrent_state = self.recurrent_model(torch.cat((prior, actions), -1), recurrent_state)
        _, imagined_prior = self._transition(
            recurrent_state, sample_state=sample_state, straight_through=straight_through
        )
        return imagined_prior, recurrent_state

    def _uniform_mix(self, logits: Tensor) -> Tensor:
        dim = logits.dim()
        if dim == 3:
            logits = logits.view(*logits.shape[:-1], -1, self.discrete)
        elif dim != 4:
            raise RuntimeError(f"The logits expected shape is 3 or 4: received a {dim}D tensor")
        if self.unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = torch.ones_like(probs) / self.discrete
            probs = (1 - self.unimix) * probs + self.unimix * uniform
            logits = probs_to_logits(probs)
        return logits.view(*logits.shape[:-2], -1)

    def _representation(
        self, recurrent_state: Tensor, embedded_obs: Tensor, sample_state: bool = True
    ) -> tuple[Tensor, Tensor]:
        """Posterior ``q(z | h, embed)``: returns ``(logits (..., S*D), state (..., S, D))``."""
        logits = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(logits, discrete=self.discrete, sample=sample_state)

    def _transition(
        self, recurrent_out: Tensor, sample_state: bool = True, straight_through: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Prior ``p(z | h)``: returns ``(logits (..., S*D), state (..., S, D))``.

        ``straight_through`` only applies to the mode (``sample_state=False``), see
        :func:`compute_stochastic_state`.
        """
        logits = self.transition_model(recurrent_out)
        logits = self._uniform_mix(logits)
        return logits, compute_stochastic_state(
            logits, discrete=self.discrete, sample=sample_state, straight_through=straight_through
        )

    def forward(
        self, observations: Tensor, actions: Tensor, is_firsts: Tensor, segment_len: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Unroll a batch of sequences for the reconstruction loss.

        Shapes: ``observations (T, B, C, H, W)`` in ``[-0.5, 0.5]``, ``actions (T, B, A)``,
        ``is_firsts (T, B, 1)``; ``segment_len``, if given, must equal ``T``. Returns
        ``(reconstructed_obs (T, B, C, H, W), reward_logits (T, B, bins), priors_logits
        (T, B, S*D), posteriors_logits (T, B, S*D))``.

        With submodules compiled with CUDA graphs (``mode="reduce-overhead"``), the output of a
        compiled call stays valid only within one CUDA-graph generation. Under autograd the pending
        backward keeps a whole call in one generation. Under ``torch.no_grad()`` every compiled call
        would start a new generation and overwrite the encoder output before the unroll reads it, so
        call ``torch.compiler.cudagraph_mark_step_begin()`` before each such call. Once a process
        has called it, generations only change at a mark, so a training loop then marks the start
        of every step too (never between a forward and its backward).
        """
        seq_len, batch = observations.shape[:2]
        if segment_len is not None and segment_len != seq_len:
            raise ValueError(f"segment_len={segment_len} must equal the sequence length {seq_len}")

        obs_flat = observations.reshape(-1, *observations.shape[-3:])
        embedded_obs = self.encoder(obs_flat)
        embedded_obs = embedded_obs.reshape(*observations.shape[:-3], -1)

        recurrent_state = torch.zeros(1, batch, self.recurrent_state_size, device=observations.device)
        posterior = torch.zeros(1, batch, self.stochastic_size, self.discrete, device=observations.device)

        recurrent_states_list = []
        priors_logits_list = []
        posteriors_list = []
        posteriors_logits_list = []
        for i in range(seq_len):
            recurrent_state, posterior, _, posterior_logits, prior_logits = self.dynamic(
                posterior,
                recurrent_state,
                actions[i : i + 1],
                embedded_obs[i : i + 1],
                is_firsts[i : i + 1],
            )
            # Copies out of the graph memory of a CUDA-graph compiled `dynamic` (the per-step buffers
            # of the paper's pretraining unroll).
            recurrent_state = recurrent_state.clone()
            posterior = posterior.clone()
            posterior_logits = posterior_logits.clone()
            prior_logits = prior_logits.clone()

            recurrent_states_list.append(recurrent_state)
            priors_logits_list.append(prior_logits)
            posteriors_list.append(posterior)
            posteriors_logits_list.append(posterior_logits)

        recurrent_states = torch.cat(recurrent_states_list, dim=0)
        priors_logits = torch.cat(priors_logits_list, dim=0)
        posteriors = torch.cat(posteriors_list, dim=0)
        posteriors_logits = torch.cat(posteriors_logits_list, dim=0)

        latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)
        reward_logits = self.reward_model(latent_states)

        latent_states_flat = latent_states.reshape(-1, *latent_states.shape[2:])
        reconstructed_obs = self.decoder(latent_states_flat)
        reconstructed_obs = reconstructed_obs.reshape(observations.shape)
        return reconstructed_obs, reward_logits, priors_logits, posteriors_logits


class GradientSwapingFunction(Function):
    """DMO gradient swap: the forward value is ``true_state``, the gradient goes to ``img_state``.

    ``true_state`` is the posterior of the frame produced by the global model (on-manifold);
    ``img_state`` is the RSSM's imagined prior for the same step. If ``img_state`` has no
    gradient, the output is a constant.
    """

    @staticmethod
    def forward(ctx, img_state, true_state):
        return true_state.clone()

    @staticmethod
    def backward(ctx, grad_true_next_state):
        return grad_true_next_state, None


# ---------------------------------------------------------------------------------------------
# Distributions and loss (DreamerV3, Eq. 5 of https://arxiv.org/abs/2301.04104)
# ---------------------------------------------------------------------------------------------


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class TwoHotEncodingDistribution:
    """Categorical over ``logits.shape[-1]`` bins uniform on ``[low, high]`` in symlog space.

    ``log_prob`` computes the log-softmax in fp32. With ``fp32_log_prob=False`` (the paper's RL
    finetuning) the result is cast back to the logits dtype (bf16 under autocast) before the
    two-hot product; with ``True`` it stays fp32, as in the paper's offline RSSM pretraining.
    The two only differ for low-precision logits.
    """

    def __init__(
        self,
        logits: Tensor,
        dims: int = 0,
        low: float = -20,
        high: float = 20,
        transfwd: Callable[[Tensor], Tensor] = symlog,
        transbwd: Callable[[Tensor], Tensor] = symexp,
        fp32_log_prob: bool = False,
    ):
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
        self.transfwd = transfwd
        self.transbwd = transbwd
        self.fp32_log_prob = fp32_log_prob

    @property
    def mode(self) -> Tensor:
        """The prediction: ``transbwd`` of the probability-weighted mean of the bins."""
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    def log_prob(self, x: Tensor) -> Tensor:
        x = self.transfwd(x)
        # below in [-1, len(bins) - 1], above in [0, len(bins)], both then clamped into range.
        below = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        above = below + 1
        above = torch.minimum(above, torch.full_like(above, len(self.bins) - 1))
        below = torch.maximum(below, torch.zeros_like(below))

        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, len(self.bins)) * weight_below[..., None]
            + F.one_hot(above, len(self.bins)) * weight_above[..., None]
        ).squeeze(-2)

        logits_fp32 = self.logits.float()
        log_pred = logits_fp32 - torch.logsumexp(logits_fp32, dim=-1, keepdims=True)
        if not self.fp32_log_prob:
            log_pred = log_pred.to(self.logits.dtype)
        return (target * log_pred).sum(dim=self.dims)


class MSEDistribution:
    """DreamerV3's MSE decoder head: ``log_prob`` is the negative sum of squared errors over the
    last ``dims`` dimensions."""

    def __init__(self, mode: Tensor, dims: int):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])

    def log_prob(self, value: Tensor) -> Tensor:
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        return -distance.sum(self._dims)


def reconstruction_loss(
    po: MSEDistribution,
    observations: Tensor,
    pr: TwoHotEncodingDistribution,
    rewards: Tensor,
    priors_logits: Tensor,
    posteriors_logits: Tensor,
    kl_dynamic: float = 0.5,
    kl_representation: float = 0.1,
    kl_free_nats: float = 1.0,
    kl_regularizer: float = 1.0,
    mask: Tensor | None = None,
    reward_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """DreamerV3 world-model loss: observation + reward + balanced KL with free nats.

    Per row ``(t, b)``: ``kl_regularizer * (kl_dynamic * max(KL[sg(q) || p], free) +
    kl_representation * max(KL[q || sg(p)], free)) - log po(o) - log pr(r)``.

    Reduction over the ``(T, B)`` rows:

    * no masks: plain mean (the paper's offline pretraining);
    * ``mask (T, B)``: weighted mean ``sum(x * m) / sum(m)``, so rows with ``m = 0`` (invalid
      replay buffer rows) contribute nothing;
    * ``reward_mask (T, B)``: the reward term gets its own weighted mean over ``reward_mask``
      (it replaces ``mask`` for that term, it is not multiplied by it); the other terms use
      ``mask`` (all ones if not given). Use it for rows whose reward label is not a real label.

    ``priors_logits``/``posteriors_logits`` are ``(T, B, S, D)``. Returns ``(loss, kl, kl_loss,
    reward_loss, observation_loss)``, all scalars reduced the same way as their term.
    """
    observation_loss = -po.log_prob(observations)
    reward_loss = -pr.log_prob(rewards)
    dyn_loss = kl = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
    )
    free_nats = torch.full_like(dyn_loss, kl_free_nats)
    dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)
    repr_loss = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
    )
    repr_loss = kl_representation * torch.maximum(repr_loss, free_nats)
    kl_loss = dyn_loss + repr_loss

    if mask is None and reward_mask is None:
        loss = (kl_regularizer * kl_loss + observation_loss + reward_loss).mean()
        return loss, kl.mean(), kl_loss.mean(), reward_loss.mean(), observation_loss.mean()

    rows = observation_loss.shape
    for name, m_ in (("mask", mask), ("reward_mask", reward_mask)):
        if m_ is not None and m_.shape != rows:
            raise ValueError(f"{name} has shape {tuple(m_.shape)}, expected the per-row loss shape {tuple(rows)}")
    if mask is None:
        mask = torch.ones(rows, device=observation_loss.device)
    m = mask.to(observation_loss.dtype)
    denom = m.sum().clamp_min(1.0)

    def wmean(x: Tensor) -> Tensor:
        return (x * m).sum() / denom

    if reward_mask is None:
        loss = wmean(kl_regularizer * kl_loss + observation_loss + reward_loss)
        reward_report = wmean(reward_loss)
    else:
        rm = reward_mask.to(observation_loss.dtype)
        reward_report = (reward_loss * rm).sum() / rm.sum().clamp_min(1.0)
        loss = wmean(kl_regularizer * kl_loss + observation_loss) + reward_report
    return loss, wmean(kl), wmean(kl_loss), reward_report, wmean(observation_loss)


# ---------------------------------------------------------------------------------------------
# Config, build, save/load
# ---------------------------------------------------------------------------------------------


def rssm_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Architecture config from a config dict, with defaults and derived keys filled in.

    ``cfg`` may be the RL config's ``rssm`` block plus ``action_size`` and ``image_size``, a
    pretraining config's model block, or a ``config.json``; keys outside :data:`ARCH_KEYS` are
    ignored. ``bottleneck_spatial = image_size // 16`` and ``cnn_output_dim = 8 *
    channels_multiplier * bottleneck_spatial**2`` are derived; if given they must agree.
    """
    missing = [k for k in REQUIRED_KEYS if cfg.get(k) is None]
    if missing:
        raise KeyError(f"RSSM config is missing {missing}")
    out: dict[str, Any] = {k: int(cfg[k]) for k in REQUIRED_KEYS}
    out["image_channels"] = int(cfg.get("image_channels", OPTIONAL_DEFAULTS["image_channels"]))
    out["unimix"] = float(cfg.get("unimix", OPTIONAL_DEFAULTS["unimix"]))
    out["learnable_initial_recurrent_state"] = bool(
        cfg.get("learnable_initial_recurrent_state", OPTIONAL_DEFAULTS["learnable_initial_recurrent_state"])
    )
    if out["image_size"] % 16:
        raise ValueError(f"image_size={out['image_size']} must be a multiple of 16 (the CNN downsamples 16x)")
    out["bottleneck_spatial"] = out["image_size"] // 16
    out["cnn_output_dim"] = 8 * out["channels_multiplier"] * out["bottleneck_spatial"] ** 2
    for key in DERIVED_KEYS:
        if cfg.get(key) is not None and int(cfg[key]) != out[key]:
            raise ValueError(
                f"RSSM config: {key}={cfg[key]} disagrees with image_size={out['image_size']} and "
                f"channels_multiplier={out['channels_multiplier']} (expected {out[key]})"
            )
    return out


def build_rssm(cfg: Mapping[str, Any]) -> RSSM:
    """Build an (uncompiled, randomly initialised) :class:`RSSM` from a config dict (see :func:`rssm_config`)."""
    c = rssm_config(cfg)
    return RSSM(
        action_size=c["action_size"],
        stochastic_size=c["stochastic_size"],
        discrete=c["discrete_size"],
        recurrent_state_size=c["recurrent_state_size"],
        dense_units=c["dense_units"],
        cnn_output_dim=c["cnn_output_dim"],
        channels_multiplier=c["channels_multiplier"],
        mlp_layers=c["mlp_layers"],
        reward_model_bins=c["reward_model_bins"],
        learnable_initial_recurrent_state=c["learnable_initial_recurrent_state"],
        bottleneck_spatial=c["bottleneck_spatial"],
        image_channels=c["image_channels"],
        unimix=c["unimix"],
    )


def unwrap_rssm(model: nn.Module) -> RSSM:
    """The :class:`RSSM` inside DDP (``.module``) and/or ``torch.compile`` (``._orig_mod``) wrappers."""
    while not isinstance(model, RSSM):
        inner = getattr(model, "module", None)
        if not isinstance(inner, nn.Module):
            inner = getattr(model, "_orig_mod", None)
        if not isinstance(inner, nn.Module):
            raise TypeError(f"{type(model).__name__} does not wrap an RSSM")
        model = inner
    return model


def save_rssm(save_dir: str | os.PathLike, model: nn.Module, metadata: Mapping[str, Any] | None = None) -> Path:
    """Write a pretrained dir: weights + ``config.json`` = architecture config + ``metadata``.

    ``metadata`` holds what the weights do not encode (action convention, colour order, data
    pool, epoch, ...); it may not redefine an architecture key. Compile/DDP prefixes are stripped.
    """
    rssm = unwrap_rssm(model)
    metadata = dict(metadata or {})
    clash = sorted(set(metadata) & (set(ARCH_KEYS) | {"model_type", "format_version"}))
    if clash:
        raise KeyError(f"save_rssm: metadata may not redefine {clash}")
    return save_pretrained(save_dir, rssm.state_dict(), {"model_type": MODEL_TYPE, **rssm.config, **metadata})


def read_rssm_checkpoint(
    path: str | os.PathLike, config: Mapping[str, Any] | None = None
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Read RSSM weights as ``(state_dict, config)`` with wrapper prefixes stripped (on CPU).

    ``path`` is a pretrained dir written by :func:`save_rssm`, or a bare ``state_dict`` ``.pt``
    file (the paper's ``rssm_e10.pt`` format), which needs ``config``. When both a stored config
    and ``config`` exist, their architectures must agree. The returned config is the stored one
    (architecture + metadata), or the architecture of ``config`` for a bare file.
    """
    path = Path(path)
    if is_pretrained_dir(path):
        state_dict, stored = load_pretrained(path)
        if stored.get("model_type") != MODEL_TYPE:
            raise ValueError(f"{path}: model_type is {stored.get('model_type')!r}, expected {MODEL_TYPE!r}")
        arch = rssm_config(stored)
        if config is not None:
            expected = rssm_config(config)
            diff = {k: (arch[k], expected[k]) for k in ARCH_KEYS if arch[k] != expected[k]}
            if diff:
                raise ValueError(f"{path}: stored RSSM architecture differs from the config, (stored, config): {diff}")
        return state_dict, {**stored, **arch}
    if path.is_file():
        if config is None:
            raise ValueError(f"{path} is a bare state_dict without a config; pass the RSSM config")
        state_dict = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if not isinstance(state_dict, Mapping) or not all(isinstance(v, Tensor) for v in state_dict.values()):
            raise ValueError(f"{path}: expected a bare state_dict of tensors")
        return strip_compile_prefix(state_dict), rssm_config(config)
    raise FileNotFoundError(f"{path}: neither a pretrained RSSM dir nor a checkpoint file")


def load_rssm(
    path: str | os.PathLike, config: Mapping[str, Any] | None = None, device: str | torch.device = "cpu"
) -> tuple[RSSM, dict[str, Any]]:
    """Build an uncompiled :class:`RSSM` and strict-load ``path`` into it (see :func:`read_rssm_checkpoint`)."""
    state_dict, stored = read_rssm_checkpoint(path, config)
    rssm = build_rssm(stored)
    rssm.load_state_dict(state_dict, strict=True)
    return rssm.to(device), stored
