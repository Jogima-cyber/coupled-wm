# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the RSSM model: build/config, forward/backward, the straight-through hook (DMO
swap and --no-diffusion imagination), the masked loss, save/load, and the paper checkpoint's key
layout."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from coupled_wm.common.config import load_config
from coupled_wm.rssm import (
    ARCH_KEYS,
    GradientSwapingFunction,
    MSEDistribution,
    TwoHotEncodingDistribution,
    build_rssm,
    compute_stochastic_state,
    load_rssm,
    read_rssm_checkpoint,
    reconstruction_loss,
    rssm_config,
    save_rssm,
    symlog,
    unwrap_rssm,
)
from coupled_wm.rssm.rssm import LayerNormGRUCell

REPO = Path(__file__).resolve().parents[1]

TINY = dict(action_size=5, image_size=64, stochastic_size=4, discrete_size=4, recurrent_state_size=32,
            dense_units=16, channels_multiplier=4, mlp_layers=2, reward_model_bins=15)

# The rssm block of the release G1 RL config (configs/rl/g1_dmo.yaml): architecture keys plus
# training keys (pretrained, straight_through_stochastic_grad, learning_rate, udt, ...) that
# rssm_config ignores. G1 adds the sizes the trainer takes from the env, and the derived CNN
# sizes explicitly, so that their consistency checks run.
G1_RL_BLOCK = load_config(REPO / "configs/rl/g1_dmo.yaml")["rssm"]
G1 = {**G1_RL_BLOCK, "action_size": 22, "image_size": 64, "bottleneck_spatial": 4, "cnn_output_dim": 12288}

# Key -> shape of the paper warm start rssm_e10.pt after strip_compile_prefix.
E10_SHAPES = {
    "initial_recurrent_state": (4096,),
    **{f"encoder._model.{i}.weight": s for i, s in
       ((0, (96, 3, 4, 4)), (3, (192, 96, 4, 4)), (6, (384, 192, 4, 4)), (9, (768, 384, 4, 4)))},
    **{f"encoder._model.{i}.{p}": (c,)
       for i, c in ((1, 96), (4, 192), (7, 384), (10, 768)) for p in ("weight", "bias")},
    "decoder.model.0.weight": (12288, 5120), "decoder.model.0.bias": (12288,),
    **{f"decoder.model.2._model.{i}.weight": s for i, s in
       ((0, (768, 384, 4, 4)), (3, (384, 192, 4, 4)), (6, (192, 96, 4, 4)), (9, (96, 3, 4, 4)))},
    **{f"decoder.model.2._model.{i}.{p}": (c,) for i, c in ((1, 384), (4, 192), (7, 96)) for p in ("weight", "bias")},
    "decoder.model.2._model.9.bias": (3,),
    "recurrent_model.mlp.0.weight": (1024, 1046),
    "recurrent_model.mlp.1.weight": (1024,), "recurrent_model.mlp.1.bias": (1024,),
    "recurrent_model.rnn.linear.weight": (12288, 5120),
    "recurrent_model.rnn.layer_norm.weight": (12288,), "recurrent_model.rnn.layer_norm.bias": (12288,),
    "representation_model.0.weight": (1024, 16384),
    "representation_model.1.weight": (1024,), "representation_model.1.bias": (1024,),
    "representation_model.3.weight": (1024, 1024), "representation_model.3.bias": (1024,),
    "transition_model.0.weight": (1024, 4096),
    "transition_model.1.weight": (1024,), "transition_model.1.bias": (1024,),
    "transition_model.3.weight": (1024, 1024), "transition_model.3.bias": (1024,),
    "reward_model.0.weight": (1024, 5120),
    **{f"reward_model.{i}.weight": (1024, 1024) for i in (3, 6, 9, 12)},
    **{f"reward_model.{i}.{p}": (1024,) for i in (1, 4, 7, 10, 13) for p in ("weight", "bias")},
    "reward_model.15.weight": (255, 1024), "reward_model.15.bias": (255,),
}


def tiny_rssm(seed=0):
    torch.manual_seed(seed)
    return build_rssm(TINY)


def tiny_batch(T=5, B=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    obs = torch.rand(T, B, 3, 64, 64, generator=g) - 0.5
    act = torch.randn(T, B, TINY["action_size"], generator=g)
    is_first = torch.zeros(T, B, 1)
    is_first[0] = 1.0
    is_first[2, 1] = 1.0  # an episode start inside the window
    rew = torch.randint(0, 2, (T, B, 1), generator=g).float()
    return obs, act, is_first, rew


def model_loss(rssm, obs, act, is_first, rew, **loss_kw):
    T, B = obs.shape[:2]
    S, D = TINY["stochastic_size"], TINY["discrete_size"]
    rec, reward_logits, priors_logits, posteriors_logits = rssm(obs, act, is_first)
    return reconstruction_loss(
        MSEDistribution(rec, dims=3), obs, TwoHotEncodingDistribution(reward_logits, dims=1), rew,
        priors_logits.view(T, B, S, D), posteriors_logits.view(T, B, S, D), **loss_kw)


# --- config / build ------------------------------------------------------------------------------


def test_rssm_config_derives_and_validates():
    assert {"pretrained", "straight_through_stochastic_grad", "learning_rate", "udt"} <= set(G1_RL_BLOCK)
    c = rssm_config(G1)
    assert set(c) == set(ARCH_KEYS)  # the RL block's training keys are ignored
    assert c == rssm_config({**G1_RL_BLOCK, "action_size": 22, "image_size": 64})
    assert c["bottleneck_spatial"] == 4 and c["cnn_output_dim"] == 12288
    assert c["image_channels"] == 3 and c["unimix"] == 0.01 and c["learnable_initial_recurrent_state"] is True
    underived = {k: v for k, v in G1.items() if k not in ("cnn_output_dim", "bottleneck_spatial")}
    c128 = rssm_config({**underived, "image_size": 128})
    assert c128["bottleneck_spatial"] == 8 and c128["cnn_output_dim"] == 8 * 96 * 64
    with pytest.raises(ValueError, match="cnn_output_dim"):
        rssm_config({**G1, "cnn_output_dim": 49152})
    with pytest.raises(ValueError, match="bottleneck_spatial"):
        rssm_config({**G1, "image_size": 128})
    with pytest.raises(ValueError, match="multiple of 16"):
        rssm_config({**underived, "image_size": 72})
    with pytest.raises(KeyError, match="action_size"):
        rssm_config(G1_RL_BLOCK)


def test_build_records_config():
    rssm = build_rssm(G1_RL_BLOCK | {"action_size": 7, "image_size": 32, "channels_multiplier": 4,
                                     "recurrent_state_size": 8, "dense_units": 8})
    assert rssm.config == rssm_config(rssm.config)
    assert rssm.config["image_size"] == 32 and rssm.config["cnn_output_dim"] == 8 * 4 * 4
    assert rssm.recurrent_model.mlp[0].in_features == 32 * 32 + 7


def test_g1_architecture_matches_paper_checkpoint_layout():
    with torch.device("meta"):
        rssm = build_rssm(G1)
    shapes = {k: tuple(v.shape) for k, v in rssm.state_dict().items()}
    assert shapes == E10_SHAPES
    assert sum(p.numel() for p in rssm.parameters()) == 172_126_594


# --- forward / backward ------------------------------------------------------------------------------


def test_forward_backward_tiny():
    rssm = tiny_rssm()
    obs, act, is_first, rew = tiny_batch()
    T, B = obs.shape[:2]
    rec, reward_logits, priors_logits, posteriors_logits = rssm(obs, act, is_first)
    assert rec.shape == obs.shape
    assert reward_logits.shape == (T, B, TINY["reward_model_bins"])
    assert priors_logits.shape == posteriors_logits.shape == (T, B, 16)
    # Without free nats every parameter gets a gradient (at init the KL is below 1 nat, where the
    # free-nats clamp stops the gradient to the prior).
    loss, kl, kl_loss, reward_loss, obs_loss = model_loss(rssm, obs, act, is_first, rew, kl_free_nats=0.0)
    assert all(torch.isfinite(x) and x.dim() == 0 for x in (loss, kl, kl_loss, reward_loss, obs_loss))
    loss.backward()
    for name, p in rssm.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    with pytest.raises(ValueError, match="segment_len"):
        rssm(obs, act, is_first, segment_len=T - 1)


def test_is_first_resets_state():
    """A row with is_first=1 ignores the carried state and its action."""
    rssm = tiny_rssm()
    B = 2
    obs = torch.rand(B, 3, 64, 64) - 0.5
    ones = torch.ones(1, B)
    h0 = torch.randn(1, B, TINY["recurrent_state_size"])
    z0 = torch.randn(1, B, 16)
    a0 = torch.randn(1, B, TINY["action_size"])
    h_a, z_a = rssm.latent_projector(z0, h0, a0, obs, ones)
    h_b, z_b = rssm.latent_projector(torch.zeros_like(z0), torch.zeros_like(h0), torch.zeros_like(a0), obs, ones)
    assert torch.allclose(h_a, h_b) and torch.equal(z_a, z_b)


def test_imagination_shapes():
    rssm = tiny_rssm()
    z = torch.zeros(1, 3, 16)
    h = torch.zeros(1, 3, TINY["recurrent_state_size"])
    prior, h1 = rssm.imagination(z, h, torch.randn(1, 3, TINY["action_size"]), sample_state=False)
    assert prior.shape == (1, 3, 4, 4) and h1.shape == h.shape
    assert torch.equal(prior.sum(-1), torch.ones(1, 3, 4))


def test_layer_norm_gru_cell():
    """DreamerV2's cell: no bias, LayerNorm(eps 1e-3) on W [h, x], update gate biased by -1."""
    torch.manual_seed(0)
    cell = LayerNormGRUCell(input_size=6, hidden_size=4)
    assert set(cell.state_dict()) == {"linear.weight", "layer_norm.weight", "layer_norm.bias"}
    with torch.no_grad():
        cell.layer_norm.weight.uniform_(0.5, 1.5)
        cell.layer_norm.bias.uniform_(-0.5, 0.5)
    x, h = torch.randn(3, 6), torch.randn(3, 4)
    z = torch.nn.functional.layer_norm(torch.cat([h, x], -1) @ cell.linear.weight.T, (12,),
                                       cell.layer_norm.weight, cell.layer_norm.bias, eps=1e-3)
    reset, cand, update = z.split(4, -1)
    cand = torch.tanh(torch.sigmoid(reset) * cand)
    update = torch.sigmoid(update - 1)
    expected = update * cand + (1 - update) * h
    out = cell(x, h)
    assert out.shape == (3, 4) and torch.allclose(out, expected, atol=1e-6)
    assert torch.equal(cell(x[None], h[None]), out[None])  # a sequence of length 1, as the RSSM calls it
    assert torch.equal(cell(x[None], h), out[None])
    with pytest.raises(ValueError, match="length 1"):
        cell(torch.randn(2, 3, 6), h)


def test_mse_head_is_the_negative_sum_of_squared_errors():
    rec, obs = torch.randn(4, 2, 3, 5, 5), torch.randn(4, 2, 3, 5, 5)
    assert torch.equal(MSEDistribution(rec, dims=3).log_prob(obs), -((rec - obs) ** 2).sum((-3, -2, -1)))


# --- DMO swap and the straight-through hook ------------------------------------------------------


def test_compute_stochastic_state_modes():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 12, requires_grad=True)
    mode = compute_stochastic_state(logits, discrete=4, sample=False)
    st = compute_stochastic_state(logits, discrete=4, sample=False, straight_through=True)
    assert mode.shape == (2, 3, 3, 4) and not mode.requires_grad
    assert torch.equal(st, mode) and st.requires_grad
    assert torch.equal(mode.argmax(-1), logits.view(2, 3, 3, 4).argmax(-1))
    w = torch.randn_like(st)
    (g_st,) = torch.autograd.grad((st * w).sum(), logits)
    (g_probs,) = torch.autograd.grad((torch.softmax(logits.view(2, 3, 3, 4), -1) * w).sum(), logits)
    assert torch.allclose(g_st, g_probs)
    sample = compute_stochastic_state(logits, discrete=4, sample=True)
    assert sample.requires_grad and torch.equal(sample.sum(-1), torch.ones(2, 3, 3))


def _dmo_step(rssm, straight_through):
    """One DMO actor step as the trainer runs it: filter the real frame, swap in the prior's gradient."""
    B = 3
    g = torch.Generator().manual_seed(3)
    z = torch.zeros(1, B, 16)
    h = torch.zeros(1, B, TINY["recurrent_state_size"])
    action = torch.randn(1, B, TINY["action_size"], generator=g).requires_grad_(True)
    next_obs = torch.rand(B, 3, 64, 64, generator=g) - 0.5
    h1, true_post = rssm.latent_projector(z, h, action, next_obs, torch.zeros(1, B))
    prior = rssm.imagined_prior(h1, straight_through=straight_through)
    swapped = GradientSwapingFunction.apply(prior, true_post.clone())
    return action, h1, true_post, prior, swapped


def test_straight_through_flag_off_is_paper_behaviour():
    rssm = tiny_rssm()
    action, h1, true_post, prior, swapped = _dmo_step(rssm, straight_through=False)
    assert torch.equal(swapped, true_post)
    assert not true_post.requires_grad and not prior.requires_grad
    assert not swapped.requires_grad and swapped.grad_fn is None  # no gradient through the stochastic half
    assert h1.requires_grad
    w = torch.randn(1, 3, 16 + TINY["recurrent_state_size"])
    full = torch.cat([swapped, h1], -1)
    (g_full,) = torch.autograd.grad((full * w).sum(), action, retain_graph=True)
    (g_h,) = torch.autograd.grad((h1 * w[..., 16:]).sum(), action)
    assert torch.equal(g_full, g_h)


def test_straight_through_flag_on_adds_stochastic_gradient():
    rssm = tiny_rssm()
    _, _, _, prior_off, _ = _dmo_step(rssm, straight_through=False)
    action, h1, true_post, prior, swapped = _dmo_step(rssm, straight_through=True)
    assert torch.equal(prior, prior_off)  # forward value unchanged
    assert torch.equal(swapped, true_post)
    assert swapped.requires_grad
    w = torch.randn(1, 3, 16)
    (g_stoch,) = torch.autograd.grad((swapped * w).sum(), action)
    assert g_stoch.abs().sum() > 0
    # The swap hands the posterior's gradient to the prior unchanged: equal to d(prior . w)/da.
    action2, h1b, _, prior2, _ = _dmo_step(rssm, straight_through=True)
    (g_prior,) = torch.autograd.grad((prior2 * w).sum(), action2)
    assert torch.allclose(g_stoch, g_prior)


# --- imagination (the --no-diffusion ablation) and the straight-through hook -------------------


def _imagination_unroll(rssm, straight_through, steps=4, detach_z=False):
    """The --no-diffusion actor unroll: ``a_t = policy(z_t, h_t)``, ``(z, h) <- imagination(z, h, a_t)``
    with the prior mode, reward head on each new latent. Returns the latents, return and policy weight."""
    B, Z = 3, 16
    torch.manual_seed(5)
    policy = torch.nn.Linear(Z + TINY["recurrent_state_size"], TINY["action_size"])
    h = torch.randn(1, B, TINY["recurrent_state_size"])
    z = torch.nn.functional.one_hot(torch.randint(0, 4, (1, B, 4)), 4).float().view(1, B, Z)
    latents, ret = [], 0.0
    for _ in range(steps):
        a = torch.tanh(policy(torch.cat([z, h], -1)))
        z, h = rssm.imagination(z, h, a, sample_state=False, straight_through=straight_through)
        z = z.view(1, B, Z)
        if detach_z:
            z = z.detach()
        latents.append(torch.cat([z, h], -1))
        ret = ret + TwoHotEncodingDistribution(rssm.reward_model(latents[-1]), dims=1).mode.sum()
    return latents, ret, policy.weight


def test_imagination_straight_through_off_is_paper_behaviour():
    rssm = tiny_rssm()
    z = torch.zeros(1, 3, 16)
    h = torch.randn(1, 3, TINY["recurrent_state_size"])
    a = torch.randn(1, 3, TINY["action_size"], requires_grad=True)
    default, h1 = rssm.imagination(z, h, a, sample_state=False)
    off, h_off = rssm.imagination(z, h, a, sample_state=False, straight_through=False)
    # The paper's step: GRU update, then the plain prior mode.
    h_ref = rssm.recurrent_model(torch.cat((z, a), -1), h)
    _, z_ref = rssm._transition(h_ref, sample_state=False)
    assert torch.equal(default, z_ref) and torch.equal(off, z_ref)
    assert torch.equal(h1, h_ref) and torch.equal(h_off, h_ref)
    assert not off.requires_grad and h_off.requires_grad

    # Over an unroll the gradient reaches the actor through h only: identical to detaching z.
    latents, ret, w = _imagination_unroll(rssm, straight_through=False)
    latents_d, ret_d, w_d = _imagination_unroll(rssm, straight_through=False, detach_z=True)
    assert all(torch.equal(x, y) for x, y in zip(latents, latents_d))
    (g,) = torch.autograd.grad(ret, w)
    (g_d,) = torch.autograd.grad(ret_d, w_d)
    assert torch.equal(g, g_d) and g.abs().sum() > 0


def test_imagination_straight_through_on_adds_stochastic_gradient():
    rssm = tiny_rssm()
    z = torch.zeros(1, 3, 16)
    h = torch.randn(1, 3, TINY["recurrent_state_size"])
    a = torch.randn(1, 3, TINY["action_size"], requires_grad=True)
    off, h_off = rssm.imagination(z, h, a, sample_state=False)
    on, h_on = rssm.imagination(z, h, a, sample_state=False, straight_through=True)
    assert torch.equal(on, off) and torch.equal(h_on, h_off)  # forward value unchanged
    assert on.requires_grad
    # Its gradient is that of the prior probabilities, as in the DMO swap's imagined_prior.
    w = torch.randn(1, 3, 4, 4)
    (g_on,) = torch.autograd.grad((on * w).sum(), a, retain_graph=True)
    (g_prior,) = torch.autograd.grad((rssm.imagined_prior(h_on, straight_through=True) * w.view(1, 3, 16)).sum(), a)
    assert torch.allclose(g_on, g_prior) and g_on.abs().sum() > 0

    latents_off, ret_off, w_off = _imagination_unroll(rssm, straight_through=False)
    latents_on, ret_on, w_on = _imagination_unroll(rssm, straight_through=True)
    assert all(torch.equal(x, y) for x, y in zip(latents_on, latents_off))  # same trajectory
    assert torch.equal(ret_on, ret_off)
    (g_off,) = torch.autograd.grad(ret_off, w_off)
    (g_on,) = torch.autograd.grad(ret_on, w_on)
    assert not torch.allclose(g_on, g_off)  # the stochastic half adds a gradient path


def test_imagination_straight_through_is_keyword_only_and_moot_for_samples():
    rssm = tiny_rssm()
    z = torch.zeros(1, 3, 16)
    h = torch.randn(1, 3, TINY["recurrent_state_size"])
    a = torch.randn(1, 3, TINY["action_size"])
    with pytest.raises(TypeError):
        rssm.imagination(z, h, a, False, True)
    torch.manual_seed(2)
    sample, _ = rssm.imagination(z, h, a)
    torch.manual_seed(2)
    sample_st, _ = rssm.imagination(z, h, a, straight_through=True)
    assert torch.equal(sample, sample_st) and sample.requires_grad


@pytest.mark.parametrize("straight_through", [False, True])
def test_imagination_traces_under_torch_compile(straight_through):
    """The trainer compiles ``imagination`` under --no-diffusion; the flag must survive tracing
    (graph-only backend: no C compiler needed)."""
    torch._dynamo.reset()
    eager = tiny_rssm()
    compiled = tiny_rssm()
    compiled.imagination = torch.compile(compiled.imagination, backend="aot_eager")
    grads = []
    for model in (eager, compiled):
        latents, ret, w = _imagination_unroll(model, straight_through=straight_through)
        (g,) = torch.autograd.grad(ret, w)
        grads.append((latents, g))
    (lat_e, g_e), (lat_c, g_c) = grads
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(lat_e, lat_c))
    assert torch.allclose(g_e, g_c, atol=1e-6, rtol=1e-5)


# --- loss ----------------------------------------------------------------------------------------------


def _loss_inputs(T=4, B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    rec = torch.randn(T, B, 3, 2, 2, generator=g, requires_grad=True)
    obs = torch.randn(T, B, 3, 2, 2, generator=g)
    rlogits = torch.randn(T, B, 15, generator=g, requires_grad=True)
    pl = torch.randn(T, B, 4, 4, generator=g, requires_grad=True)
    ql = torch.randn(T, B, 4, 4, generator=g, requires_grad=True)
    rew = torch.randint(0, 3, (T, B, 1), generator=g).float()
    return rec, obs, rlogits, pl, ql, rew


def _per_row(rec, obs, rlogits, pl, ql, rew):
    """Per-row (obs, reward, kl, kl_loss) terms through the unmasked path with B=1 slices."""
    po, pr = MSEDistribution(rec, dims=3), TwoHotEncodingDistribution(rlogits, dims=1)
    obs_rows = -po.log_prob(obs)
    rew_rows = -pr.log_prob(rew)
    T, B = obs_rows.shape
    kl_rows, kll_rows = torch.empty(T, B), torch.empty(T, B)
    for t in range(T):
        for b in range(B):
            sl = (slice(t, t + 1), slice(b, b + 1))
            out = reconstruction_loss(MSEDistribution(rec[sl], dims=3), obs[sl],
                                      TwoHotEncodingDistribution(rlogits[sl], dims=1), rew[sl], pl[sl], ql[sl])
            kl_rows[t, b], kll_rows[t, b] = out[1], out[2]
    return obs_rows, rew_rows, kl_rows, kll_rows


def test_loss_unmasked_is_plain_mean():
    rec, obs, rlogits, pl, ql, rew = _loss_inputs()
    obs_rows, rew_rows, kl_rows, kll_rows = _per_row(rec, obs, rlogits, pl, ql, rew)
    loss, kl, kl_loss, reward_loss, obs_loss = reconstruction_loss(
        MSEDistribution(rec, dims=3), obs, TwoHotEncodingDistribution(rlogits, dims=1), rew, pl, ql)
    assert torch.allclose(loss, (kll_rows + obs_rows + rew_rows).mean())
    assert torch.allclose(kl, kl_rows.mean()) and torch.allclose(kl_loss, kll_rows.mean())
    assert torch.allclose(reward_loss, rew_rows.mean()) and torch.allclose(obs_loss, obs_rows.mean())
    assert (kll_rows >= 0.5 * 1.0 + 0.1 * 1.0 - 1e-6).all()  # free nats bound each KL term


def test_loss_mask_weighting():
    rec, obs, rlogits, pl, ql, rew = _loss_inputs()
    obs_rows, rew_rows, _, kll_rows = _per_row(rec, obs, rlogits, pl, ql, rew)
    mask = torch.tensor([[1.0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 1]])
    args = (MSEDistribution(rec, dims=3), obs, TwoHotEncodingDistribution(rlogits, dims=1), rew, pl, ql)

    ones = reconstruction_loss(*args, mask=torch.ones(4, 3))
    plain = reconstruction_loss(*args)
    assert all(torch.allclose(a, b) for a, b in zip(ones, plain))

    loss, _, _, reward_loss, obs_loss = reconstruction_loss(*args, mask=mask)
    keep = mask.bool()
    assert torch.allclose(loss, (kll_rows + obs_rows + rew_rows)[keep].mean())
    assert torch.allclose(reward_loss, rew_rows[keep].mean()) and torch.allclose(obs_loss, obs_rows[keep].mean())

    grads = torch.autograd.grad(loss, (rec, rlogits, pl, ql))
    for g_ in grads:
        assert torch.all(g_[~keep] == 0) and g_[keep].abs().sum() > 0

    rec2 = rec.detach().clone()
    rec2[~keep] += 100.0  # masked rows cannot move the loss
    loss2 = reconstruction_loss(MSEDistribution(rec2, dims=3), *args[1:], mask=mask)[0]
    assert torch.allclose(loss, loss2)


def test_loss_reward_mask_has_its_own_denominator():
    rec, obs, rlogits, pl, ql, rew = _loss_inputs()
    obs_rows, rew_rows, _, kll_rows = _per_row(rec, obs, rlogits, pl, ql, rew)
    args = (MSEDistribution(rec, dims=3), obs, TwoHotEncodingDistribution(rlogits, dims=1), rew, pl, ql)
    mask = torch.ones(4, 3)
    mask[3, 0] = 0
    reward_mask = mask.clone()
    reward_mask[0] = 0  # e.g. the seed rows of a window: no real reward label
    loss, _, _, reward_loss, obs_loss = reconstruction_loss(*args, mask=mask, reward_mask=reward_mask)
    m, rm = mask.bool(), reward_mask.bool()
    assert torch.allclose(reward_loss, rew_rows[rm].mean())
    assert torch.allclose(obs_loss, obs_rows[m].mean())
    assert torch.allclose(loss, (kll_rows + obs_rows)[m].mean() + rew_rows[rm].mean())
    (g_r,) = torch.autograd.grad(loss, rlogits)
    assert torch.all(g_r[0] == 0) and g_r[1:].abs().sum() > 0

    only_rm = reconstruction_loss(*args, reward_mask=reward_mask)
    assert torch.allclose(only_rm[0], (kll_rows + obs_rows).mean() + rew_rows[rm].mean())

    with pytest.raises(ValueError, match="mask"):
        reconstruction_loss(*args, mask=mask.unsqueeze(-1))


def test_loss_on_rssm_outputs_with_masks():
    rssm = tiny_rssm()
    obs, act, is_first, rew = tiny_batch()
    mask = torch.ones(obs.shape[:2])
    reward_mask = mask.clone()
    reward_mask[0] = 0
    loss = model_loss(rssm, obs, act, is_first, rew, mask=mask, reward_mask=reward_mask)[0]
    loss.backward()
    assert torch.isfinite(loss)


# --- reward head ---------------------------------------------------------------------------------------


def test_two_hot_roundtrip_and_precision_flag():
    bins = torch.linspace(-20, 20, 255)
    for value in (0.0, 1.0, 2.0, -3.5):
        target = symlog(torch.tensor(value))
        below = int((bins <= target).sum()) - 1
        w_above = (target - bins[below]) / (bins[below + 1] - bins[below])
        probs = torch.zeros(255)
        probs[below], probs[below + 1] = 1 - w_above, w_above
        dist = TwoHotEncodingDistribution(torch.log(probs.clamp_min(1e-30)), dims=1)
        assert torch.allclose(dist.mode, torch.tensor([value]), atol=1e-4)
        entropy = -(probs * torch.log(probs.clamp_min(1e-30))).sum()  # cross-entropy of the target with itself
        assert torch.allclose(-dist.log_prob(torch.tensor([value])), entropy, atol=1e-4)

    logits = torch.randn(6, 255)
    r = torch.randn(6, 1)
    a = TwoHotEncodingDistribution(logits, dims=1).log_prob(r)
    b = TwoHotEncodingDistribution(logits, dims=1, fp32_log_prob=True).log_prob(r)
    assert torch.equal(a, b)  # identical for fp32 logits
    lb = logits.bfloat16()
    a = TwoHotEncodingDistribution(lb, dims=1).log_prob(r)
    b = TwoHotEncodingDistribution(lb, dims=1, fp32_log_prob=True).log_prob(r)
    assert a.dtype == b.dtype == torch.float32 and torch.allclose(a, b, rtol=1e-2) and not torch.equal(a, b)


# --- save / load ---------------------------------------------------------------------------------------


def _outputs(rssm):
    obs, act, is_first, _ = tiny_batch(T=3, B=2)
    torch.manual_seed(11)  # the posterior is sampled in forward()
    with torch.no_grad():
        return rssm(obs, act, is_first)


def test_save_load_pretrained_roundtrip(tmp_path):
    rssm = tiny_rssm()
    out = save_rssm(tmp_path / "rssm_e1", rssm, metadata={"relative_actions": True, "color_order": "rgb"})
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "rssm" and cfg["format_version"] == 1
    assert {k: cfg[k] for k in ARCH_KEYS} == rssm.config and cfg["relative_actions"] is True

    loaded, stored = load_rssm(out, config={**TINY, "batch_size": 7})
    assert stored["color_order"] == "rgb"
    assert all(torch.equal(a, b) for a, b in zip(_outputs(rssm), _outputs(loaded)))

    with pytest.raises(ValueError, match="dense_units"):
        load_rssm(out, config={**TINY, "dense_units": 8})
    with pytest.raises(KeyError, match="redefine"):
        save_rssm(tmp_path / "bad", rssm, metadata={"action_size": 3})


def test_load_bare_compiled_state_dict(tmp_path):
    """The paper's format: a bare state_dict whose compiled submodules carry ``_orig_mod.``."""
    rssm = tiny_rssm()
    compiled = ("encoder.", "decoder.", "reward_model.")
    sd = {}
    for k, v in rssm.state_dict().items():
        head = next((p for p in compiled if k.startswith(p)), None)
        sd[f"{head}_orig_mod.{k[len(head):]}" if head else k] = v
    path = tmp_path / "rssm_e10.pt"
    torch.save(sd, path)
    with pytest.raises(ValueError, match="config"):
        read_rssm_checkpoint(path)
    loaded, stored = load_rssm(path, config=TINY)
    assert stored == rssm.config
    assert all(torch.equal(a, b) for a, b in zip(_outputs(rssm), _outputs(loaded)))
    with pytest.raises(RuntimeError):
        load_rssm(path, config={**TINY, "action_size": 6})


def test_save_unwraps_ddp_like_wrappers(tmp_path):
    class Wrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    rssm = tiny_rssm()
    assert unwrap_rssm(Wrapper(rssm)) is rssm
    save_rssm(tmp_path / "w", Wrapper(rssm))
    loaded, _ = load_rssm(tmp_path / "w")
    assert all(torch.equal(a, b) for a, b in zip(_outputs(rssm), _outputs(loaded)))
    with pytest.raises(TypeError):
        unwrap_rssm(nn.Linear(2, 2))
