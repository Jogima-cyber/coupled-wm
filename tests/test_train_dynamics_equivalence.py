# SPDX-License-Identifier: Apache-2.0
"""The dynamics objective of ``train_dynamics`` against the paper training script.

The reference functions below are copied verbatim from the paper training code (the
behaviour-cloning branch, off in every paper run, is removed). Both sides run the release
``DreamerV4Dynamics`` on identical inputs and RNG states; losses, loss-scaler states and
parameter gradients must be bit-identical.
"""

from __future__ import annotations

import copy
from fractions import Fraction

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from coupled_wm.world_model.models import build_dynamics
from coupled_wm.world_model.train_dynamics import (
    RMSLossScaler,
    dynamics_loss,
    sample_noise_levels,
    shortcut_layout,
    update_plan,
)

# ----------------------------------------------------------------------------- paper code


class PaperRMSLossScaler:
    """Tracks running RMS for named losses and returns normalized losses."""
    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = decay
        self.eps = eps
        self.ema_sq = {}  # name -> scalar tensor

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sq = value.detach().pow(2)
            if dist.is_initialized():
                dist.all_reduce(sq, op=dist.ReduceOp.AVG)
            if name not in self.ema_sq:
                self.ema_sq[name] = sq
            else:
                self.ema_sq[name] = (
                    self.decay * self.ema_sq[name] + (1.0 - self.decay) * sq
                )
            rms = (self.ema_sq[name] + self.eps).sqrt()
        return value / rms


def paper_sample_shortcut_batch(B, T, device, micro_idx, GRAD_ACCUM_STEPS, num_tau_levels):
    """Sample (tau, d) for Diffusion Forcing with Shortcut split. See parent script."""
    cutoff = int(GRAD_ACCUM_STEPS * 0.75)
    is_flow_worker = (micro_idx < cutoff)
    d_min_idx = int(np.log2(num_tau_levels))

    tau_idxs = torch.zeros(B, T, dtype=torch.long, device=device)
    step_idxs = torch.zeros(B, T, dtype=torch.long, device=device)

    if is_flow_worker:
        step_idxs.fill_(d_min_idx)
        tau_idxs = torch.randint(0, num_tau_levels, (B, T), device=device)
    else:
        s_cols = torch.randint(0, d_min_idx, (B, T), device=device)
        step_idxs = s_cols
        max_rows = (2.0 ** s_cols)
        u = torch.rand(B, T, device=device)
        s_rows = (u * max_rows).floor().long()
        tau_idxs = s_rows

    d_vals = (0.5 ** step_idxs.float())
    tau_vals = 1 - (tau_idxs.float() + 1.0) * d_vals
    real_tau_idxs = ((tau_idxs + 1) * (2 ** (d_min_idx - step_idxs))).long() - 1
    return tau_vals, d_vals, real_tau_idxs, step_idxs


def paper_add_noise_flow_matching(z_clean, tau):
    """x_tau = (1 - tau) * eps + tau * z_clean. tau=1 -> clean, tau=0 -> noise."""
    B, T, N, D = z_clean.shape
    eps = torch.randn_like(z_clean)
    tau_ex = tau.view(B, T, 1, 1)
    z_noisy = (1 - tau_ex) * eps + tau_ex * z_clean
    return z_noisy


def paper_compute_dynamics_loss(dyn_model, actions, z_clean, reward_targets, is_demo,
                                tau_vals, d_vals, tau_idxs, step_idxs, micro_idx,
                                GRAD_ACCUM_STEPS, num_tau_levels, loss_scaler):
    B, T, N, D = z_clean.shape

    tau_vals = tau_vals.to(dtype=z_clean.dtype)
    d_vals = d_vals.to(dtype=z_clean.dtype)

    cutoff = int(GRAD_ACCUM_STEPS * 0.75)
    is_flow_worker = (micro_idx < cutoff)
    d_min_idx = int(np.log2(num_tau_levels))

    z_input = paper_add_noise_flow_matching(z_clean, tau_vals)

    pred_z1_student, pred_rewards = dyn_model(actions, z_input, tau_idxs, step_idxs)

    weights = 0.9 * tau_vals + 0.1
    weights = weights.view(*tau_vals.shape, 1, 1)

    loss_reward = 0.0
    loss_reward_raw = 0.0
    loss_policy = 0.0

    if dyn_model.module.train_reward_model and pred_rewards is not None:
        B, T, L, Buckets = pred_rewards.shape

        if reward_targets.dim() == 3 and reward_targets.shape[-1] == 1:
            reward_targets = reward_targets.squeeze(-1)

        pad_len = L - 1
        rewards_padded = F.pad(reward_targets, (0, pad_len), value=0.0)

        valid_mask = torch.ones_like(reward_targets, dtype=torch.bool)
        valid_mask_padded = F.pad(valid_mask, (0, pad_len), value=False)
        mask_windowed = valid_mask_padded.unfold(dimension=1, size=L, step=1)

        targets_windowed = rewards_padded.unfold(dimension=1, size=L, step=1)

        head_module = dyn_model.module.reward_head.heads[0]
        low, low_w, high, high_w = head_module.get_targets(targets_windowed)

        log_probs = F.log_softmax(pred_rewards, dim=-1)
        lp_low = log_probs.gather(-1, low.unsqueeze(-1)).squeeze(-1)
        lp_high = log_probs.gather(-1, high.unsqueeze(-1)).squeeze(-1)

        loss_per_step = -(low_w * lp_low + high_w * lp_high)
        loss_per_step = loss_per_step * mask_windowed.float()

        raw_r_loss = loss_per_step.sum() / mask_windowed.sum().clamp(min=1.0)

        loss_reward = loss_scaler("loss_reward", raw_r_loss)
        loss_reward_raw = raw_r_loss.detach().item()

    if is_flow_worker:
        diff = pred_z1_student - z_clean
        raw_flow_loss = (weights * (diff ** 2)).mean()
        loss_flow_norm = loss_scaler("loss_dyn", raw_flow_loss)

        total_loss = loss_flow_norm + loss_reward + loss_policy
        return total_loss, raw_flow_loss.detach().item(), 0.0, loss_reward_raw
    else:
        with torch.no_grad():
            inner_model = dyn_model.module if hasattr(dyn_model, "module") else dyn_model
            d_half = d_vals / 2.0
            step_idxs_half = step_idxs + 1
            tau_idxs_1 = tau_idxs
            z_in_short = z_input

            pred_z1_t1, _ = inner_model(actions, z_in_short, tau_idxs_1, step_idxs_half)
            denom_t1 = (1.0 - tau_vals.view(B, T, 1, 1) + 1e-6)
            b_prime = (pred_z1_t1 - z_in_short) / denom_t1
            z_prime = z_in_short + b_prime * d_half.view(B, T, 1, 1)

            tau_new_vals = tau_vals + d_half
            tau_idxs_2 = tau_idxs_1 - (2 ** (d_min_idx - step_idxs_half)).long()

            pred_z1_t2, _ = inner_model(actions, z_prime, tau_idxs_2, step_idxs_half)
            denom_t2 = (1.0 - tau_new_vals.view(B, T, 1, 1) + 1e-6)
            b_double_prime = (pred_z1_t2 - z_prime) / denom_t2

            v_target = (b_prime + b_double_prime) / 2.0
            target_update = (1.0 - tau_vals.view(B, T, 1, 1)) * v_target

        student_update = pred_z1_student - z_input
        diff = student_update - target_update
        raw_short_loss = (weights * (diff ** 2)).mean()
        loss_short_norm = loss_scaler("loss_dyn", raw_short_loss)

        total_loss = loss_short_norm + loss_reward + loss_policy
        return total_loss, 0.0, raw_short_loss.detach().item(), loss_reward_raw


class PaperDDPView:
    """Stands in for the paper's ``DDP(torch.compile(dyn))``: callable, with ``.module``."""

    def __init__(self, model):
        self.module = model
        model.train_reward_model = model.use_reward_head

    def __call__(self, *args):
        return self.module(*args)


# ----------------------------------------------------------------------------- tests

TINY_DYN = dict(action_dim=6, num_latents=8, latent_dim=4, d_model=32, num_layers=4, num_heads=2, num_registers=2,
                num_tau_levels=16, temporal_every=2, mlp_ratio=4.0, seq_len=16, context_len=6, dropout=0.0,
                mtp_length=3, reward_hidden_dim=16, reward_num_buckets=31, reward_symlog_range=[-20.0, 20.0])


@pytest.mark.parametrize("micro_idx,shortcut", [(0, False), (2, False), (3, True)])
def test_noise_levels_match_paper(micro_idx, shortcut):
    torch.manual_seed(5)
    ref = paper_sample_shortcut_batch(3, 7, torch.device("cpu"), micro_idx, 4, 128)
    torch.manual_seed(5)
    out = sample_noise_levels(3, 7, torch.device("cpu"), shortcut, 128)
    for a, b in zip(ref, out, strict=True):
        assert a.dtype == b.dtype and torch.equal(a, b)
    tau, d, tau_idx, step = out
    assert torch.equal(tau, 1 - (tau_idx.float() + 1) / 128)  # fine-grid index of tau
    assert bool(((step == 7) if not shortcut else (step < 7)).all())


@pytest.mark.parametrize("world", [1, 2, 64])
def test_shortcut_split_matches_paper_at_four_accumulation_steps(world):
    """Every rank and update, long-clip ones included (every 8th, as the G1 backbone), splits as
    the paper's at its 4 accumulation steps."""
    accum = 4
    paper = [not (m < int(accum * 0.75)) for m in range(accum)]
    long_updates = 0
    for update in range(40):
        plans = [update_plan(update, long_updates, 8, rank, world, accum, Fraction(1, 4)) for rank in range(world)]
        long_updates += plans[0][0]
        assert all(flags == paper for _, flags in plans)
    assert long_updates == 4


def _inputs(seed, with_reward, dtype):
    g = torch.Generator().manual_seed(seed)
    B, T = 2, 9
    z = (torch.rand(B, T, 8, 4, generator=g) * 2 - 1).to(dtype)
    actions = torch.randn(B, T, 6, generator=g).to(dtype)
    rewards = (torch.rand(B, T, generator=g) < 0.3).float() if with_reward else None
    return B, T, z, actions, rewards


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("with_reward", [False, True])
def test_dynamics_loss_matches_paper(autocast, with_reward):
    torch.manual_seed(0)
    base = build_dynamics({**TINY_DYN, "use_reward_head": with_reward})
    ours_model, paper_model = copy.deepcopy(base), copy.deepcopy(base)
    paper_view = PaperDDPView(paper_model)
    ours_scaler, paper_scaler = RMSLossScaler(), PaperRMSLossScaler()
    dtype = torch.bfloat16 if autocast else torch.float32
    reward_head = ours_model.reward_head.heads[0] if with_reward else None

    # Two updates of 4 micro-steps (3 flow + 1 shortcut), so the scaler EMA is exercised.
    for update in range(2):
        for micro in range(4):
            B, T, z, actions, rewards = _inputs(100 * update + micro, with_reward, dtype)
            shortcut = shortcut_layout(update, 0, 64, 4, Fraction(1, 4))[micro]  # the paper's 64 ranks

            torch.manual_seed(1000 + micro)
            levels = paper_sample_shortcut_batch(B, T, torch.device("cpu"), micro, 4, 16)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                ref, ref_flow, ref_short, ref_rew = paper_compute_dynamics_loss(
                    paper_view, actions, z, rewards, None, *levels, micro, 4, 16, paper_scaler)
            (ref / 4).backward()

            torch.manual_seed(1000 + micro)
            levels = sample_noise_levels(B, T, torch.device("cpu"), shortcut, 16)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                out, raw = dynamics_loss(ours_model, ours_model, actions, z, rewards, levels, shortcut, ours_scaler,
                                         16, reward_head)
            (out / 4).backward()

            assert out.dtype == ref.dtype and torch.equal(out, ref)
            assert raw["shortcut" if shortcut else "flow"].item() == (ref_short if shortcut else ref_flow)
            if with_reward:
                assert raw["reward"].item() == ref_rew
            else:
                assert "reward" not in raw

    assert ours_scaler.ema_sq.keys() == paper_scaler.ema_sq.keys()
    for k in ours_scaler.ema_sq:
        assert torch.equal(ours_scaler.ema_sq[k], paper_scaler.ema_sq[k])
    for (name, p), (_, q) in zip(ours_model.named_parameters(), paper_model.named_parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), name


def test_scaler_state_round_trip():
    s = RMSLossScaler()
    for v in (0.5, 0.25, 2.0):
        s("loss_dyn", torch.tensor(v))
    t = RMSLossScaler()
    t.load_state_dict(s.state_dict(), "cpu")
    assert torch.equal(t("loss_dyn", torch.tensor(1.0)), s("loss_dyn", torch.tensor(1.0)))
    assert t.rms("loss_dyn") == s.rms("loss_dyn") and t.rms("loss_reward") is None
