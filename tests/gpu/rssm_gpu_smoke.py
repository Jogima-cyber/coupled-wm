# SPDX-License-Identifier: Apache-2.0
"""GPU smoke test of the RSSM under CUDA, bf16 autocast and ``torch.compile``, compiled exactly as
the trainers compile it.

    python tests/gpu/rssm_gpu_smoke.py [--rssm <pretrained dir>] [--data <chunk.h5>]

Compile sets, each checked on its own copy of the model, compiled from a fresh state
(``torch._dynamo.reset()``) as in a training process of its own. ``torch.compile`` reuses a graph
compiled for another model copy when the backends compare equal: without the reset, the second
set's encoder, decoder and reward head would run the first set's graphs and CUDA-graph trees.

* ``rl``: ``DMOTrainer._compile_rssm`` with RSSM finetuning: encoder, decoder and reward head with
  ``mode="reduce-overhead"`` (CUDA graphs); ``dynamic``, ``latent_projector`` and ``_transition``
  (``imagination`` under --no-diffusion) with the default mode. The paper's RL compiled the same.
* ``pretrain``: :func:`coupled_wm.rssm.pretrain.compile_rssm` with the shipped ``reduce-overhead``:
  encoder, decoder, reward head and ``dynamic``, as the paper's pretraining script. It runs last and
  marks every step (:func:`~coupled_wm.rssm.pretrain.begin_compiled_step`) as pretraining does:
  once a process marks steps, CUDA-graph generations only change at marks.

Inputs: with ``--data``, ``B`` windows of real G1 frames turned into rows as pretraining does
(64 px, the env's resize); otherwise uniform noise frames. Check a trained RSSM on real frames: on
noise its posteriors are diffuse and its reward head confidently wrong, where sampling noise is
largest.

1. With ``--rssm``: strict load of a real checkpoint at the G1 size, on CPU first.
2. CUDA-graph replays, the first calls of the fresh compile: three calls of the training-step loss
   with the same seed, which are the warmup, the recording and the first replay of its CUDA graphs,
   as the first three steps of a training run. Their losses agree within one bf16 rounding (bitwise
   unless a cuDNN algorithm is nondeterministic; a stale graph output is off by far more). Each
   call's gradients, copied right after its backward, are finite for the same parameters, with
   cosine at least ``1 - 1e-4`` to call 1's. For ``pretrain`` also the validation forward under
   ``no_grad``.
3. Per module, compiled vs eager on identical inputs (encoder, decoder, reward head, one ``dynamic``
   step), forward and backward. bf16 keeps 8 significant bits, so every bf16 op rounds with a
   relative error up to ``2**-8``; a compiled kernel fuses ops and changes reduction orders, which
   moves some of these roundings. The criterion is relative to the module's own bf16 error: the
   compiled output may differ from eager by at most twice eager's distance to the same module in
   fp32 (relative L2 errors, TF32 off). The reward head is compared where the loss uses it: logits
   and the per-row two-hot NLL (and its gradient). The ``dynamic`` step is compared on its
   deterministic outputs (``h`` and both logits); how often its sampled posterior differs from
   eager's (same RNG state) is printed next to the rate of an independent redraw.
4. The full unroll with sampled posteriors. A one-ulp difference upstream of a sample can flip a
   categorical; that sequence then follows another latent path and a confident reward head changes
   its NLL by nats, so one seed proves nothing. Compiled and eager are compared as random variables
   over ``SEEDS`` posterior seeds: per loss term ``|mean_c - mean_e| <= 4 SE + 4 * 2**-8 * |mean_e|``.
   The flipped samples per seed are printed.
5. ``rl``: the DMO swap and the --no-diffusion ``imagination`` step with ``straight_through`` off
   (default: no gradient through the stochastic half) and on (that gradient, same forward value).
6. The pretraining loss equals the paper pretraining script's step bitwise in eager CUDA bf16, loss
   and gradients: the script's unroll (per-step outputs written into fp32 buffers) and its two-hot
   log-softmax ``logits - logsumexp(logits)`` (fp32 under CUDA autocast). Both steps run with
   deterministic kernels (``torch.use_deterministic_algorithms``, cuDNN deterministic, the script
   sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` unless it is set): cuDNN's default convolution backward
   algorithms do not reproduce gradients bitwise from run to run. The RL finetuning's precision
   (that log-softmax cast back to bf16) must give a different reward loss.

Every check prints its numbers; the script exits non-zero if any failed.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
try:
    import coupled_wm  # noqa: F401
except ModuleNotFoundError:  # source checkout without `pip install -e .`
    sys.path.insert(0, str(REPO))

from coupled_wm.data.h5_dataset import HDF5SequenceDataset  # noqa: E402
from coupled_wm.rl.dmo import DMOTrainer  # noqa: E402
from coupled_wm.rssm import (  # noqa: E402
    GradientSwapingFunction,
    MSEDistribution,
    TwoHotEncodingDistribution,
    build_rssm,
    load_rssm,
    reconstruction_loss,
)
from coupled_wm.rssm.pretrain import begin_compiled_step, compile_rssm, rssm_loss, window_to_rows  # noqa: E402

G1 = dict(action_size=22, image_size=64, stochastic_size=32, discrete_size=32, recurrent_state_size=4096,
          dense_units=1024, channels_multiplier=96, mlp_layers=5, reward_model_bins=255)
U = 2.0 ** -8  # bf16 unit roundoff
T, B = 16, 4
SEEDS = 8
LOSS_NAMES = ("loss", "kl", "kl_loss", "reward", "obs")
FAILURES: list[str] = []


def check(ok: bool, what: str) -> None:
    print(f"    {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        FAILURES.append(what)


def bf16():
    return torch.autocast("cuda", dtype=torch.bfloat16)


@contextlib.contextmanager
def fp32():
    """Full fp32 (no autocast, no TF32): the reference for the bf16 rounding error."""
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


@contextlib.contextmanager
def deterministic():
    """Deterministic CUDA kernels (cuBLAS also needs ``CUBLAS_WORKSPACE_CONFIG``, set in :func:`main`)."""
    flags = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
             torch.backends.cudnn.benchmark)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(flags[0])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags[1:]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double(), b.detach().double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def within_bf16_error(name: str, compiled, eager, ref) -> None:
    """``|compiled - eager| <= 2 |eager - fp32|`` in relative L2 (plus an fp32-level floor)."""
    e_compile, e_bf16 = rel(compiled, eager), rel(eager, ref)
    check(e_compile <= 2 * e_bf16 + 1e-6,
          f"{name:<26} compiled vs eager {e_compile:.2e}, eager bf16 vs fp32 {e_bf16:.2e}")


# --------------------------------------------------------------------------- models and data


def compile_like(eager, compile_set: str, no_diffusion: bool = False):
    model = copy.deepcopy(eager)
    if compile_set == "pretrain":
        compile_rssm(model, "reduce-overhead")  # train.compile_mode of configs/rssm/g1_rssm_pretrain.yaml
    else:
        DMOTrainer._compile_rssm(SimpleNamespace(rssm=model, finetune_rssm=True, no_diffusion=no_diffusion))
    return model


def make_batch(data: str | None, action_size: int, device):
    """Time-major ``(obs - 0.5, actions, is_first, rewards, mask, reward_mask)``."""
    if data:
        windows = HDF5SequenceDataset([data], window=T + 1, image_size=64, color_order="bgr",
                                      load_proprio=False, action_dim=action_size)
        rows = [window_to_rows(windows[i]) for i in torch.linspace(0, len(windows) - 1, B).long().tolist()]
        obs = torch.stack([r["obs"] for r in rows], 1).to(device) - 0.5
        act = torch.stack([r["action"] for r in rows], 1).to(device)
        rew = torch.stack([r["reward"] for r in rows], 1).to(device).unsqueeze(-1)
        is_first = torch.stack([r["is_first"] for r in rows], 1).to(device).unsqueeze(-1)
    else:
        g = torch.Generator(device=device).manual_seed(1)
        obs = torch.rand(T, B, 3, 64, 64, generator=g, device=device) - 0.5
        act = 0.02 * torch.randn(T, B, action_size, generator=g, device=device)
        rew = torch.randint(0, 2, (T, B, 1), generator=g, device=device).float()
        is_first = torch.zeros(T, B, 1, device=device)
        is_first[0] = 1
    is_first[T // 2, B // 2] = 1  # an episode start inside a sequence, as replay buffer rows have
    mask = torch.ones(T, B, device=device)
    mask[-3:, 1] = 0
    reward_mask = mask * (1 - is_first.squeeze(-1))
    return obs, act, is_first, rew, mask, reward_mask


def unroll(model, obs, act, is_first):
    """``RSSM.forward`` call by call, also returning the sampled posteriors, ``h`` and embeddings."""
    n_t, n_b = obs.shape[:2]
    emb = model.encoder(obs.reshape(-1, *obs.shape[-3:])).reshape(n_t, n_b, -1)
    h = torch.zeros(1, n_b, model.recurrent_state_size, device=obs.device)
    z = torch.zeros(1, n_b, model.stochastic_size, model.discrete, device=obs.device)
    hs, zs, priors, posts = [], [], [], []
    for t in range(n_t):
        h, z, _, post_logits, prior_logits = model.dynamic(z, h, act[t:t + 1], emb[t:t + 1], is_first[t:t + 1])
        h, z, post_logits, prior_logits = h.clone(), z.clone(), post_logits.clone(), prior_logits.clone()
        hs.append(h), zs.append(z), posts.append(post_logits), priors.append(prior_logits)
    hs, zs = torch.cat(hs), torch.cat(zs)
    latent = torch.cat((zs.view(n_t, n_b, -1), hs), -1)
    reward_logits = model.reward_model(latent)
    rec = model.decoder(latent.reshape(n_t * n_b, -1)).reshape(obs.shape)
    return rec, reward_logits, torch.cat(priors), torch.cat(posts), zs, hs, emb


def loss_terms(model, outputs, batch, pretrain: bool):
    """The loss each trainer computes: pretraining's (unmasked, fp32 log-softmax) or RL finetuning's."""
    rec, reward_logits, priors, posts = outputs[:4]
    obs, _, _, rew, mask, reward_mask = batch
    shape = (*obs.shape[:2], model.stochastic_size, model.discrete)
    masks = {} if pretrain else dict(mask=mask, reward_mask=reward_mask)
    return reconstruction_loss(
        MSEDistribution(rec, dims=3), obs, TwoHotEncodingDistribution(reward_logits, dims=1, fp32_log_prob=pretrain),
        rew, priors.view(shape), posts.view(shape), **masks)


def training_loss(model, batch, pretrain: bool):
    obs, act, is_first, rew = batch[:4]
    if pretrain:
        return rssm_loss(model, obs, act, rew, is_first)
    return loss_terms(model, model(obs, act, is_first), batch, pretrain)


def parameter_grads(model) -> dict[str, torch.Tensor]:
    """Copies of the parameter gradients by name: with CUDA graphs, ``.grad`` may hold graph memory
    that the next call overwrites."""
    return {name.replace("_orig_mod.", ""): p.grad.detach().clone()
            for name, p in model.named_parameters() if p.grad is not None}


# --------------------------------------------------------------------------- 2. CUDA-graph replays


def replay_checks(compiled, batch, pretrain: bool, mark) -> None:
    def three_calls(grad: bool):
        losses, grads = [], []
        for _ in range(3):
            compiled.zero_grad(set_to_none=True)
            mark()
            torch.manual_seed(5)
            with torch.set_grad_enabled(grad), bf16():
                out = training_loss(compiled, batch, pretrain)
            if grad:
                out[0].backward()
                grads.append(parameter_grads(compiled))
            losses.append(torch.stack([x.detach().double() for x in out]))
        return losses, grads

    def relative(a, b):
        return ((a - b).abs() / b.abs().clamp_min(1e-12)).max().item()

    def flat(g, names):
        return torch.cat([g[n].flatten().double() for n in names])

    losses, grads = three_calls(grad=True)
    for i, g in enumerate(grads):
        nonfinite = [n for n, x in g.items() if not torch.isfinite(x).all()]
        unmatched = sorted(g.keys() ^ grads[0].keys())
        check(not nonfinite and not unmatched,
              f"training-step call {i + 1}: gradients of {len(g)} parameters, finite"
              + (f"; non-finite: {', '.join(nonfinite)}" if nonfinite else "")
              + (f"; a gradient in only one of calls 1 and {i + 1}: {', '.join(unmatched)}" if unmatched else ""))
    for i in (1, 2):
        d = relative(losses[i], losses[0])
        check(bool(torch.isfinite(losses[i]).all()) and d <= U,
              f"training-step call {i + 1}: max relative loss difference to call 1 {d:.1e} "
              f"(bitwise {torch.equal(losses[i], losses[0])})")
        names = [n for n in grads[0] if n in grads[i]]
        cos = F.cosine_similarity(flat(grads[i], names), flat(grads[0], names), dim=0).item()
        check(cos >= 1 - 1e-4, f"training-step call {i + 1}: gradient cosine to call 1 {cos:.8f}")
    if pretrain:
        val, _ = three_calls(grad=False)
        for i in (0, 1, 2):
            d = relative(val[i], val[0])
            check(bool(torch.isfinite(val[i]).all()) and d <= U,
                  f"validation forward (no_grad) call {i + 1}: finite, max relative difference to call 1 {d:.1e}")
        print("    info validation / training-step losses, same seed: "
              + ", ".join(f"{n} {a:.4f} / {b:.4f}" for n, a, b in zip(LOSS_NAMES, val[0].tolist(), losses[0].tolist())))


# --------------------------------------------------------------------------- 3. per module


def module_checks(eager, compiled, states, batch, pretrain: bool, mark) -> None:
    obs, act, is_first, rew = batch[:4]
    zs, hs, emb = states
    latent = torch.cat((zs.view(T, B, -1), hs), -1)

    def run(module_of, fn, inputs, grad_inputs, use_compiled, low_precision):
        """``(outputs, gradients)`` of ``fn`` on fresh leaf copies of ``inputs``."""
        model = compiled if use_compiled else eager
        inputs = [x.detach().clone().requires_grad_(i in grad_inputs) for i, x in enumerate(inputs)]
        params = list(module_of(model).parameters())
        if use_compiled:
            mark()
        torch.manual_seed(11)
        with bf16() if low_precision else fp32():
            outputs, objective = fn(model, *inputs)
        grads = torch.autograd.grad(objective, params + [inputs[i] for i in grad_inputs], allow_unused=True)
        return outputs, torch.cat([g.flatten().double() for g in grads if g is not None])

    def projection(*outputs):
        g = torch.Generator(device=outputs[0].device).manual_seed(5)
        return sum((torch.randn(o.shape, generator=g, device=o.device) * o.float()).sum() for o in outputs)

    def encoder(m, x):
        e = m.encoder(x)
        return (e,), projection(e)

    def decoder(m, x):
        y = m.decoder(x)
        return (y,), projection(y)

    def reward_head(m, x):
        logits = m.reward_model(x)
        nll = -TwoHotEncodingDistribution(logits, dims=1, fp32_log_prob=pretrain).log_prob(rew)
        return (logits, nll), nll.mean()

    def dynamic(m, z, h, a, e, first):
        h1, z1, _, post_logits, prior_logits = m.dynamic(z, h, a, e, first)
        return (h1, post_logits, prior_logits, z1.detach()), projection(h1, post_logits, prior_logits)

    t = T // 2  # a step where one sequence restarts (is_first = 1)
    cases = (
        ("encoder", lambda m: m.encoder, encoder, [obs.reshape(-1, *obs.shape[-3:])], [], ["embedding"]),
        ("decoder", lambda m: m.decoder, decoder, [latent.reshape(T * B, -1)], [0], ["image"]),
        ("reward head", lambda m: m.reward_model, reward_head, [latent], [0], ["logits", "two-hot NLL"]),
        ("dynamic", lambda m: m, dynamic, [zs[t - 1:t], hs[t - 1:t], act[t:t + 1], emb[t:t + 1], is_first[t:t + 1]],
         [1, 2, 3], ["h", "posterior logits", "prior logits"]),
    )
    for name, module_of, fn, inputs, grad_inputs, output_names in cases:
        out_c, g_c = run(module_of, fn, inputs, grad_inputs, True, True)
        out_e, g_e = run(module_of, fn, inputs, grad_inputs, False, True)
        out_r, g_r = run(module_of, fn, inputs, grad_inputs, False, False)
        for label, c, e, r in zip(output_names, out_c, out_e, out_r):
            within_bf16_error(f"{name}: {label}", c, e, r)
        within_bf16_error(f"{name}: gradients", g_c, g_e, g_r)
        if name == "reward head":
            print(f"    info max |d logit| {(out_c[0] - out_e[0]).abs().max().item():.3e}, "
                  f"max |d NLL| {(out_c[1] - out_e[1]).abs().max().item():.3e} nats")
        if name == "dynamic":
            sample_c, sample_e = out_c[3], out_e[3]
            differ = (sample_c.argmax(-1) != sample_e.argmax(-1)).float().mean().item()
            probs = out_e[1].float().view_as(sample_e).softmax(-1)
            redraw = (1 - (probs ** 2).sum(-1)).mean().item()  # P(an independent redraw differs)
            print(f"    info dynamic: the posterior sample differs from eager's (same RNG state) in {differ:.2%} "
                  f"of the categoricals; an independent redraw would change {redraw:.2%} (near 0%: same random "
                  "stream; near the redraw rate: another stream, e.g. CUDA-graph replays; both are valid samples)")


# --------------------------------------------------------------------------- 4. sampled unroll


def sampling_checks(eager, compiled, batch, pretrain: bool, mark) -> None:
    obs, act, is_first = batch[:3]
    values = {"eager": [], "compiled": []}
    for k in range(SEEDS):
        samples = {}
        for tag, model in (("eager", eager), ("compiled", compiled)):
            model.zero_grad(set_to_none=True)
            if tag == "compiled":
                mark()
            torch.manual_seed(100 + k)
            with bf16():
                outputs = unroll(model, obs, act, is_first)
                terms = loss_terms(model, outputs, batch, pretrain)
            terms[0].backward()  # completes the step, as training does
            values[tag].append(torch.stack([x.detach().double() for x in terms]).cpu())
            samples[tag] = outputs[4].argmax(-1)
        flipped = samples["eager"] != samples["compiled"]
        steps = flipped.flatten(1).any(1).nonzero()
        first = steps[0].item() if len(steps) else None
        d = values["compiled"][-1] - values["eager"][-1]
        print(f"    seed {k}: {int(flipped.sum())} of {flipped.numel()} posterior samples differ (first at step "
              f"{first}); compiled - eager: {', '.join(f'{n} {x:+.4f}' for n, x in zip(LOSS_NAMES, d.tolist()))}")
    c, e = torch.stack(values["compiled"]), torch.stack(values["eager"])
    se = ((c.var(0) + e.var(0)) / SEEDS).sqrt()
    for i, name in enumerate(LOSS_NAMES):
        diff = (c[:, i].mean() - e[:, i].mean()).abs().item()
        bound = 4 * se[i].item() + 4 * U * e[:, i].mean().abs().item()
        check(math.isfinite(c[:, i].mean().item()) and diff <= bound,
              f"{name:<8} over {SEEDS} seeds: compiled {c[:, i].mean():.4f}, eager {e[:, i].mean():.4f} "
              f"(sd {e[:, i].std():.4f}); |difference| {diff:.4f} <= {bound:.4f}")


# --------------------------------------------------------------------------- 5. straight-through


def dmo_step(model, straight_through: bool, z, h, next_obs):
    action = (0.02 * torch.randn(1, z.shape[1], model.config["action_size"], device=z.device)).requires_grad_(True)
    with bf16():
        h1, post = model.latent_projector(z, h, action, next_obs, torch.zeros(1, z.shape[1], device=z.device))
        prior = model.imagined_prior(h1, straight_through=straight_through)
        swapped = GradientSwapingFunction.apply(prior, post.clone())
        reward = TwoHotEncodingDistribution(model.reward_model(torch.cat([swapped, h1], -1)), dims=1).mode.sum()
    return action, post, prior, swapped, reward


def imagination_step(model, straight_through: bool, z, h):
    action = (0.02 * torch.randn(1, z.shape[1], model.config["action_size"], device=z.device)).requires_grad_(True)
    with bf16():
        z1, h1 = model.imagination(z, h, action, sample_state=False, straight_through=straight_through)
        z1 = z1.view(*z1.shape[:-2], -1)
        reward = TwoHotEncodingDistribution(model.reward_model(torch.cat([z1, h1], -1)), dims=1).mode.sum()
    return action, z1, reward


def extra_gradient(step, model, *inputs):
    """``step`` with the flag off and on from one seed, and the gradient the flag adds (relative)."""
    torch.manual_seed(3)
    off = step(model, False, *inputs)
    (g_off,) = torch.autograd.grad(off[-1], off[0])
    torch.manual_seed(3)
    on = step(model, True, *inputs)
    (g_on,) = torch.autograd.grad(on[-1], on[0])
    return off, on, ((g_on - g_off).norm() / g_off.norm().clamp_min(1e-12)).item()


def straight_through_checks(eager, compiled, compiled_no_diffusion, states, batch) -> None:
    zs, hs, _ = states
    t = 3
    z, h, next_obs = zs[t:t + 1].flatten(-2), hs[t:t + 1], batch[0][t + 1]
    for model, tag in ((eager, "eager"), (compiled, "compiled")):
        off, on, extra = extra_gradient(dmo_step, model, z, h, next_obs)
        _, post_off, prior_off, sw_off, _ = off
        _, post_on, prior_on, sw_on, _ = on
        check(torch.equal(sw_off, post_off) and not sw_off.requires_grad,
              f"{tag} DMO swap, straight_through off: forward = posterior, no gradient through it")
        check(torch.equal(prior_on, prior_off) and torch.equal(sw_on, post_on) and sw_on.requires_grad and extra > 0,
              f"{tag} DMO swap, straight_through on: same forward, |extra gradient| / |h-only| = {extra:.3f}")
    for model, tag in ((eager, "eager"), (compiled_no_diffusion, "compiled (--no-diffusion set)")):
        off, on, extra = extra_gradient(imagination_step, model, z, h)
        check(not off[1].requires_grad, f"{tag} imagination, straight_through off: no gradient through z")
        check(torch.equal(on[1], off[1]) and on[1].requires_grad and extra > 0,
              f"{tag} imagination, straight_through on: same z, |extra gradient| / |h-only| = {extra:.3f}")


# --------------------------------------------------------------------------- 6. paper pretraining step


def script_unroll(rssm, obs, act, is_first):
    """The unroll of the paper's RSSM pretraining script."""
    n_t, n_b = obs.shape[:2]
    c = rssm.config
    sd = c["stochastic_size"] * c["discrete_size"]
    emb = rssm.encoder(obs.reshape(-1, *obs.shape[-3:])).reshape(n_t, n_b, -1)
    h = torch.zeros(1, n_b, c["recurrent_state_size"], device=obs.device)
    post = torch.zeros(1, n_b, c["stochastic_size"], c["discrete_size"], device=obs.device)
    hs = torch.empty(n_t, n_b, c["recurrent_state_size"], device=obs.device)
    priors = torch.empty(n_t, n_b, sd, device=obs.device)
    posts = torch.empty(n_t, n_b, c["stochastic_size"], c["discrete_size"], device=obs.device)
    posts_logits = torch.empty(n_t, n_b, sd, device=obs.device)
    for i in range(n_t):
        h, post, _, post_logits, prior_logits = rssm.dynamic(post, h, act[i:i + 1], emb[i:i + 1], is_first[i:i + 1])
        hs[i], priors[i], posts[i], posts_logits[i] = h, prior_logits, post, post_logits
    latent = torch.cat((posts.view(n_t, n_b, -1), hs), -1)
    rec = rssm.decoder(latent.reshape(-1, latent.shape[-1])).reshape(obs.shape)
    return rec, rssm.reward_model(latent), priors, posts_logits


class ScriptTwoHot:
    """``TwoHotEncodingDistribution.log_prob`` of the paper's pretraining script (sheeprl's, Apache-2.0)."""

    def __init__(self, logits, low=-20, high=20):
        self.logits = logits
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)

    def log_prob(self, x):
        x = torch.sign(x) * torch.log(1 + torch.abs(x))
        below = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        above = torch.minimum(below + 1, torch.full_like(below, len(self.bins) - 1))
        below = torch.maximum(below, torch.zeros_like(below))
        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        target = (F.one_hot(below, len(self.bins)) * (dist_to_above / total)[..., None]
                  + F.one_hot(above, len(self.bins)) * (dist_to_below / total)[..., None]).squeeze(-2)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdims=True)
        return (target * log_pred).sum(dim=(-1,))


def check_pretraining_loss_matches_script(device) -> None:
    cfg = dict(G1, recurrent_state_size=256, dense_units=128, channels_multiplier=16, mlp_layers=2)
    torch.manual_seed(0)
    rssm = build_rssm(cfg).to(device)
    n_t, n_b = 8, 3
    g = torch.Generator(device=device).manual_seed(2)
    obs = torch.rand(n_t, n_b, 3, 64, 64, generator=g, device=device) - 0.5
    act = 0.02 * torch.randn(n_t, n_b, 22, generator=g, device=device)
    is_first = torch.zeros(n_t, n_b, 1, device=device)
    is_first[0] = 1
    rew = torch.randint(0, 3, (n_t, n_b, 1), generator=g, device=device).float()

    def script_step(m):
        rec, reward_logits, priors, posts = script_unroll(m, obs, act, is_first)
        out = reconstruction_loss(MSEDistribution(rec, dims=3), obs, ScriptTwoHot(reward_logits), rew,
                                  priors.view(n_t, n_b, 32, 32), posts.view(n_t, n_b, 32, 32))
        rl_reward = (-TwoHotEncodingDistribution(reward_logits, dims=1).log_prob(rew)).mean()
        return out, reward_logits.dtype, rl_reward

    def run(step):
        rssm.zero_grad(set_to_none=True)
        torch.manual_seed(9)
        with bf16():
            out, logits_dtype, rl_reward = step(rssm)
        out[0].backward()
        return out, [p.grad.clone() for p in rssm.parameters()], logits_dtype, rl_reward

    def bitwise(a, b):
        """Whether two runs agree bitwise: (losses, gradients)."""
        return tuple(all(torch.equal(x, y) for x, y in zip(a[i], b[i])) for i in (0, 1))

    with deterministic():
        script, repeat = run(script_step), run(script_step)
        release = run(lambda m: (rssm_loss(m, obs, act, rew, is_first), None, None))
    _, _, logits_dtype, rl_reward = script
    o2 = release[0]
    same_loss, same_grad = bitwise(script, release)
    reproducible = all(bitwise(script, repeat))
    check(logits_dtype == torch.bfloat16 and same_loss and same_grad,
          f"pretraining loss vs paper script ({logits_dtype} reward logits, deterministic kernels): loss bitwise "
          f"{same_loss}, grads bitwise {same_grad} (the script's step run twice: bitwise {reproducible})")
    check(not torch.equal(rl_reward, o2[3]),
          f"reward loss {o2[3].item():.6f} differs at the RL finetuning precision ({rl_reward.item():.6f})")


# --------------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description="GPU smoke test of the RSSM under bf16 autocast and torch.compile.")
    parser.add_argument("--rssm", default=None, help="pretrained RSSM dir, e.g. $WEIGHTS_DIR/rssm (optional)")
    parser.add_argument("--data", default=None, help="one G1 HDF5 file, for real input frames (optional)")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    # Deterministic cuBLAS for check 6. cuBLAS reads this once per process, at its first call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cuda")
    inductor_config.fallback_random = True  # compiled sampling consumes the eager RNG stream

    if args.rssm:
        eager, stored = load_rssm(args.rssm, config=G1)  # raises unless every key and shape matches
        print(f"1. strict load\n    ok   {args.rssm} ({stored.get('model_type', 'bare state dict')}) at the G1 size")
    else:
        torch.manual_seed(0)
        eager = build_rssm(G1)
    eager = eager.to(device)
    batch = make_batch(args.data, eager.config["action_size"], device)
    print(f"inputs: {'frames of ' + args.data if args.data else 'uniform noise frames'}, T={T}, B={B}; "
          f"RSSM: {args.rssm or 'random init'}")

    with torch.no_grad(), bf16():
        torch.manual_seed(0)
        mirrored = unroll(eager, *batch[:3])
        torch.manual_seed(0)
        forward = eager(*batch[:3])
    check(all(torch.equal(a, b) for a, b in zip(mirrored[1:4], forward[1:4])) and rel(mirrored[0], forward[0]) <= U,
          "the unroll of this script reproduces RSSM.forward")
    states = mirrored[4:]  # sampled posteriors, h and embeddings (in their training dtypes) of the rows

    for compile_set in ("rl", "pretrain"):  # pretrain last: it marks CUDA-graph steps
        pretrain = compile_set == "pretrain"
        mark = (lambda: begin_compiled_step(True)) if pretrain else (lambda: None)
        torch._dynamo.reset()  # this set's own graphs and CUDA-graph trees, as in its own training process
        compiled = compile_like(eager, compile_set)
        print(f"== compile set '{compile_set}'\n2. CUDA-graph replays: the training step's warmup, recording, replay")
        replay_checks(compiled, batch, pretrain, mark)
        print("3. per module: compiled vs eager (bf16) vs eager fp32")
        module_checks(eager, compiled, states, batch, pretrain, mark)
        print(f"4. the unroll with sampled posteriors, over {SEEDS} seeds")
        sampling_checks(eager, compiled, batch, pretrain, mark)
        if not pretrain:
            print("5. straight_through on the DMO swap and on --no-diffusion imagination")
            straight_through_checks(eager, compiled, compile_like(eager, "rl", no_diffusion=True), states, batch)
        del compiled

    print("6. pretraining step vs the paper script")
    check_pretraining_loss_matches_script(device)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):\n  " + "\n  ".join(FAILURES))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
