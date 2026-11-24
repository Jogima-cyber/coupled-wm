# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest
import torch
from torch.nn.attention import SDPBackend

from coupled_wm.world_model import models as M
from coupled_wm.world_model.models import (
    G1_DYNAMICS_ARCH,
    G1_TOKENIZER_ARCH,
    MHA_GQA,
    build_dynamics,
    build_tokenizer,
    dynamics_arch,
    init_nonpersistent_buffers,
    tokenizer_arch,
)

TINY_TOK = dict(image_size=[32, 32], patch_size=8, in_channels=3, d_model=64, n_layers=8, num_heads_q=4,
                num_heads_kv_latent=4, temporal_every=4, n_latents=8, bottleneck_dim=4, use_proprio=True,
                proprio_dim=5, num_proprio=1, seq_len=16, context_len=0, dropout=0.0, mae_max_mask_prob=0.9)
TINY_DYN = dict(action_dim=6, num_latents=8, latent_dim=4, d_model=64, num_layers=8, num_heads=4, num_registers=2,
                num_tau_levels=16, temporal_every=4, mlp_ratio=4.0, seq_len=40, context_len=12, dropout=0.0,
                use_reward_head=True, mtp_length=3, reward_hidden_dim=32, reward_num_buckets=31,
                reward_symlog_range=[-20.0, 20.0])


def _inputs(B=2, T=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        video=torch.rand(B, T, 3, 32, 32, generator=g),
        proprio=torch.randn(B, T, 5, generator=g),
        action=torch.randn(B, T, 6, generator=g),
        z=torch.rand(B, T, 8, 4, generator=g) * 2 - 1,
        sigma=torch.randint(0, 16, (B, T), generator=g),
        step=torch.randint(0, 5, (B, T), generator=g),
    )


def _tiny_models(seed=0):
    torch.manual_seed(seed)
    enc, dec = build_tokenizer(TINY_TOK)
    dyn = build_dynamics(TINY_DYN)
    return enc.eval(), dec.eval(), dyn.eval()


# ---------------------------------------------------------------- paper G1 layout

# sha256 of the sorted "key shape" lines of the paper models' state dicts (the paper's models.py
# built with the G1 constructor arguments on the meta device); the paper checkpoints strict-load
# into exactly these layouts.
PAPER_LAYOUT_SHA256 = {
    "enc": "85c17910b0e1ee6a75de8f54c65e7901454662ba0658efe00d2533a3712324bc",
    "dec": "aeca4543e1b3f01de17a1ed373c626941b862efae6d3f1c46e19e819fbb8fe7c",
    "dyn": "cc8b50eed5060dccfc600d30de8757500f95edc4d2c0c50b4499ca0ae10d6307",
}


def _layout_sha256(module):
    lines = "".join(f"{k} {tuple(v.shape)}\n" for k, v in sorted(module.state_dict().items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def test_g1_state_dict_keys_and_shapes_are_the_paper_checkpoints():
    with torch.device("meta"):
        enc, dec = build_tokenizer()
        dyn = build_dynamics()
    assert {name: _layout_sha256(m) for name, m in (("enc", enc), ("dec", dec), ("dyn", dyn))} == PAPER_LAYOUT_SHA256


def test_g1_architecture_matches_paper_checkpoint_layout():
    """Key sets / sizes of the paper checkpoints (enc 309, dec 324, dyn 252 keys)."""
    with torch.device("meta"):
        enc, dec = build_tokenizer()
        dyn = build_dynamics()
    expected = {"enc": (enc, 309, 378_654_752, 0), "dec": (dec, 324, 378_653_490, 16 * 257 * 513),
                "dyn": (dyn, 252, 681_631_768, 8 * 255)}
    for name, (m, n_keys, n_params, n_buf) in expected.items():
        sd = m.state_dict()
        assert len(sd) == n_keys, name
        assert sum(p.numel() for p in m.parameters()) == n_params, name
        assert sum(b.numel() for k, b in m.named_buffers() if k in sd) == n_buf, name
    esd, dsd = enc.state_dict(), dyn.state_dict()
    assert esd["patch_embed.weight"].shape == (1024, 3, 16, 16)
    assert esd["layers.0.ffn_L.up.weight"].shape == (2730, 1024)
    assert dsd["act_proj.weight"].shape == (1536, 22)
    assert dsd["sigma_embed.weight"].shape == (128, 768) and dsd["step_embed.weight"].shape == (8, 768)
    assert dsd["layers.0.ffn.up.weight"].shape == (4096, 1536)
    assert [i for i, layer in enumerate(enc.layers) if layer.temporal] == [3, 7, 11, 15]
    assert [i for i, layer in enumerate(dyn.layers) if layer.temporal] == list(range(3, 24, 4))
    assert [i for i, layer in enumerate(enc.layers) if layer.is_last] == [14, 15]
    assert not hasattr(dyn, "policy_head")


def test_g1_defaults_are_the_paper_values():
    assert tokenizer_arch() == tokenizer_arch(G1_TOKENIZER_ARCH, complete=True) == G1_TOKENIZER_ARCH
    assert dynamics_arch() == dynamics_arch(G1_DYNAMICS_ARCH, complete=True) == G1_DYNAMICS_ARCH
    assert len(M.G1_ACTION["scale"]) == M.G1_ACTION["dim"] == G1_DYNAMICS_ARCH["action_dim"]
    assert M.G1_INFERENCE["d_min_idx"] == 7 == int(torch.log2(torch.tensor(128.0)))
    rep = M.G1_ACTION["representation"]
    assert all(group in rep for group in ("base velocity", "(0-2)", "joint positions (3-19)", "grips", "(20-21)"))


# ---------------------------------------------------------------- arch validation


def test_arch_validation():
    with pytest.raises(KeyError, match="mlp_ratio"):
        tokenizer_arch({"mlp_ratio": 4.0})  # unused by the tokenizer: not a key
    with pytest.raises(KeyError, match="missing"):
        tokenizer_arch({"d_model": 64}, complete=True)
    with pytest.raises(ValueError, match="divisible"):
        tokenizer_arch({**TINY_TOK, "image_size": [30, 32]})
    with pytest.raises(ValueError, match="num_heads_kv_latent"):
        tokenizer_arch({**TINY_TOK, "num_heads_kv_latent": 3})
    with pytest.raises(TypeError):
        tokenizer_arch({**TINY_TOK, "use_proprio": 1})
    with pytest.raises(TypeError):
        dynamics_arch({**TINY_DYN, "num_layers": 2.5})
    with pytest.raises(ValueError, match="power of two"):
        dynamics_arch({**TINY_DYN, "num_tau_levels": 100})
    assert tokenizer_arch({**TINY_TOK, "image_size": 32})["image_size"] == [32, 32]
    assert dynamics_arch({**TINY_DYN, "num_layers": 8.0})["num_layers"] == 8
    a = dynamics_arch(TINY_DYN)
    a["reward_symlog_range"].append(1.0)
    assert TINY_DYN["reward_symlog_range"] == [-20.0, 20.0]  # normalisation copies
    with pytest.raises(ValueError, match="num_proprio == 1"):
        tokenizer_arch({**TINY_TOK, "num_proprio": 2})
    assert tokenizer_arch({**TINY_TOK, "use_proprio": False, "num_proprio": 2})["num_proprio"] == 2  # unused


@pytest.mark.parametrize("n_layers, temporal_every", [(3, 1), (5, 1)])
def test_tokenizer_rejects_layer_schedules_that_drop_a_needed_stream(n_layers, temporal_every):
    with pytest.raises(ValueError, match=f"n_layers={n_layers} with temporal_every={temporal_every}"):
        tokenizer_arch({**TINY_TOK, "n_layers": n_layers, "temporal_every": temporal_every})


@pytest.mark.parametrize("n_layers, temporal_every", [(1, 1), (2, 1), (4, 1), (5, 2), (7, 3), (6, 4), (3, 4)])
def test_every_accepted_layer_schedule_runs(n_layers, temporal_every):
    torch.manual_seed(0)
    enc, dec = build_tokenizer({**TINY_TOK, "n_layers": n_layers, "temporal_every": temporal_every})
    x = _inputs(T=3)
    with torch.no_grad():
        _, _, Z = enc.eval()(x["video"], proprio=x["proprio"])
        _, x_hat, prop_hat = dec.eval()(Z)
    assert x_hat.shape == (2, 3, 3, 32, 32) and prop_hat.shape == (2, 3, 5)


# ---------------------------------------------------------------- forward behaviour


def test_tokenizer_shapes_and_bottleneck_range():
    enc, dec, _ = _tiny_models()
    x = _inputs()
    with torch.no_grad():
        P, L, Z = enc(x["video"], proprio=x["proprio"])
        R, x_hat, prop_hat = dec(Z)
    assert P is None  # the last layer has no patch stream
    assert L.shape == (2, 10, 8, 64) and Z.shape == (2, 10, 8, 4)
    assert Z.abs().max() <= 1.0
    assert x_hat.shape == (2, 10, 3, 32, 32) and prop_hat.shape == (2, 10, 5)


def test_encoder_is_causal_in_time():
    enc, _, _ = _tiny_models()
    x = _inputs()
    with torch.no_grad():
        _, _, Z = enc(x["video"], proprio=x["proprio"])
        video = x["video"].clone()
        video[:, 6:] = torch.rand_like(video[:, 6:])
        _, _, Z2 = enc(video, proprio=x["proprio"])
    torch.testing.assert_close(Z[:, :6], Z2[:, :6], atol=1e-6, rtol=0)
    assert not torch.allclose(Z[:, 6:], Z2[:, 6:])


def test_encoder_context_window():
    enc, _, _ = _tiny_models()
    x = _inputs(T=10)
    with torch.no_grad():
        _, _, Z = enc(x["video"], proprio=x["proprio"])
        for layer in enc.layers:
            if layer.temporal:
                layer.context_len = 10
        _, _, Z_same = enc(x["video"], proprio=x["proprio"])  # T <= window: plain causal
        for layer in enc.layers:
            if layer.temporal:
                layer.context_len = 4
        _, _, Z_win = enc(x["video"], proprio=x["proprio"])
    assert torch.equal(Z, Z_same)
    assert torch.allclose(Z[:, :4], Z_win[:, :4], atol=1e-6)
    assert not torch.allclose(Z[:, 4:], Z_win[:, 4:])


def test_decoder_cached_matches_full_forward():
    enc, dec, _ = _tiny_models()
    x = _inputs(T=10)
    with torch.no_grad():
        _, _, Z = enc(x["video"], proprio=x["proprio"])
        _, x_full, p_full = dec(Z)
        dec.init_cache(2, "cpu", max_seq_len=16)
        _, x0, p0 = dec.forward_step(Z[:, :6], start_step_idx=0)
        steps = [dec.forward_step(Z[:, t:t + 1], start_step_idx=t) for t in range(6, 10)]
    torch.testing.assert_close(x0, x_full[:, :6], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(p0, p_full[:, :6], atol=1e-5, rtol=1e-5)
    for t, (_, xt, pt) in zip(range(6, 10), steps):
        torch.testing.assert_close(xt[:, 0], x_full[:, t], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pt[:, 0], p_full[:, t], atol=1e-5, rtol=1e-5)


def test_gqa_decoder_cached_matches_full_forward():
    """``num_heads_kv_latent < num_heads_q``: the latent-stream caches hold the KV heads."""
    torch.manual_seed(0)
    _, dec = build_tokenizer({**TINY_TOK, "num_heads_kv_latent": 2})
    dec.eval()
    Z = _inputs(T=10)["z"]
    with torch.no_grad():
        _, x_full, p_full = dec(Z)
        dec.init_cache(2, "cpu", max_seq_len=16)
        _, x0, p0 = dec.forward_step(Z[:, :6], start_step_idx=0)
        torch.testing.assert_close(x0, x_full[:, :6], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(p0, p_full[:, :6], atol=1e-5, rtol=1e-5)
        for t in range(6, 10):
            for update in (False, True):
                _, xt, pt = dec.forward_step(Z[:, t:t + 1], start_step_idx=t, update_cache=update)
                torch.testing.assert_close(xt[:, 0], x_full[:, t], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(pt[:, 0], p_full[:, t], atol=1e-5, rtol=1e-5)
    latent_heads = [c.k_cache.shape[1] for c in dec.caches_l if c is not None]
    assert latent_heads and set(latent_heads) == {2}


def test_decoder_rejects_multiframe_step_onto_nonempty_cache():
    _, dec, _ = _tiny_models()
    Z = _inputs()["z"]
    dec.init_cache(2, "cpu", max_seq_len=16)
    with torch.no_grad():
        dec.forward_step(Z[:, :4], start_step_idx=0)
        dec.forward_step(Z[:, 4:5], start_step_idx=4)
        with pytest.raises(RuntimeError, match="empty KV cache"):
            dec.forward_step(Z[:, 5:7], start_step_idx=5)


def test_dynamics_cached_matches_full_forward():
    _, _, dyn = _tiny_models()
    x = _inputs(T=10)
    args = (x["action"], x["z"], x["sigma"], x["step"])
    with torch.no_grad():
        z_full, r_full = dyn(*args)
        dyn.init_cache(2, "cpu", max_seq_len=16)
        z0, r0 = dyn.forward_step(*(a[:, :6] for a in args), start_step_idx=0, update_cache=True)
        torch.testing.assert_close(z0, z_full[:, :6], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(r0, r_full[:, :6], atol=1e-5, rtol=1e-5)
        for t in range(6, 10):
            frame = tuple(a[:, t:t + 1] for a in args)
            z_read, r_read = dyn.forward_step(*frame, start_step_idx=t, update_cache=False)
            z_upd, r_upd = dyn.forward_step(*frame, start_step_idx=t, update_cache=True)
            for out in (z_read, z_upd):
                torch.testing.assert_close(out[:, 0], z_full[:, t], atol=1e-5, rtol=1e-5)
            for out in (r_read, r_upd):
                torch.testing.assert_close(out[:, 0], r_full[:, t], atol=1e-5, rtol=1e-5)
    assert r_full.shape == (2, 10, 3, 31)


def test_dynamics_cached_generation_keeps_the_sliding_window():
    """A KV cache longer than ``context_len`` (12) gives the windowed whole-clip forward."""
    _, _, dyn = _tiny_models()
    x = _inputs(T=24)
    args = (x["action"], x["z"], x["sigma"], x["step"])
    with torch.no_grad():
        z_full, r_full = dyn(*args)  # T > context_len: sliding-window mask
        dyn.init_cache(2, "cpu", max_seq_len=16)
        z0, r0 = dyn.forward_step(*(a[:, :14] for a in args), start_step_idx=0, update_cache=True)
        torch.testing.assert_close(z0, z_full[:, :14], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(r0, r_full[:, :14], atol=1e-5, rtol=1e-5)
        for t in range(14, 24):  # the cache fills up (16) and rolls
            frame = tuple(a[:, t:t + 1] for a in args)
            for update in (False, True):
                z_t, r_t = dyn.forward_step(*frame, start_step_idx=t, update_cache=update)
                torch.testing.assert_close(z_t[:, 0], z_full[:, t], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(r_t[:, 0], r_full[:, t], atol=1e-5, rtol=1e-5)


def test_cache_overflow_errors_are_readable():
    _, dec, dyn = _tiny_models()
    x = _inputs(T=10)
    with pytest.raises(ValueError, match="cannot write 5 frames into a KV cache of 4"):
        M.KVCache(4, 1, 1, 2, "cpu", torch.float32).update(torch.zeros(1, 1, 5, 2), torch.zeros(1, 1, 5, 2))
    dyn.init_cache(2, "cpu", max_seq_len=4)
    with torch.no_grad(), pytest.raises(ValueError, match="does not fit the KV cache"):
        dyn.forward_step(*(x[k][:, :6] for k in ("action", "z", "sigma", "step")), start_step_idx=0)
    dec.init_cache(2, "cpu", max_seq_len=4)
    with torch.no_grad(), pytest.raises(ValueError, match="KV cache of 4 frames"):
        dec.forward_step(x["z"][:, :6], start_step_idx=0)


def test_kv_cache_bytes_match_the_allocation():
    torch.manual_seed(0)
    _, dec = build_tokenizer({**TINY_TOK, "num_heads_kv_latent": 2})
    dyn = build_dynamics(TINY_DYN)
    for m, caches in ((dec, lambda: [*dec.caches_l, *dec.caches_r]), (dyn, lambda: dyn.caches)):
        m.init_cache(3, "cpu", max_seq_len=7)
        allocated = sum(t.numel() * t.element_size() for c in caches() if c is not None for t in (c.k_cache, c.v_cache))
        assert m.kv_cache_bytes(3, 7) == allocated
        assert m.kv_cache_bytes(3, 7, torch.bfloat16) == allocated // 2
    with torch.device("meta"):  # paper G1, bf16, 32-frame cache: per env
        _, g1_dec = build_tokenizer()
        g1_dyn = build_dynamics()
    assert g1_dyn.kv_cache_bytes(1, 32, torch.bfloat16) == 310_247_424
    assert g1_dec.kv_cache_bytes(1, 32, torch.bfloat16) == 235_405_312


def test_dynamics_agent_token_is_hidden_from_world_tokens():
    _, _, dyn = _tiny_models()
    x = _inputs()
    with torch.no_grad():
        z1, r1 = dyn(x["action"], x["z"], x["sigma"], x["step"])
        dyn.agent_token.add_(1.0)
        z2, r2 = dyn(x["action"], x["z"], x["sigma"], x["step"])
    assert torch.equal(z1, z2)
    assert not torch.allclose(r1, r2)


def test_dynamics_without_reward_head():
    torch.manual_seed(0)
    dyn = build_dynamics({**TINY_DYN, "use_reward_head": False}).eval()
    assert dyn.agent_token is None and dyn.reward_head is None
    assert not any(k.startswith(("agent_token", "reward_head")) for k in dyn.state_dict())
    x = _inputs()
    with torch.no_grad():
        z, r = dyn(x["action"], x["z"], x["sigma"], x["step"])
        dyn.init_cache(2, "cpu", max_seq_len=16)
        zs, rs = dyn.forward_step(x["action"][:, :3], x["z"][:, :3], x["sigma"][:, :3], x["step"][:, :3], 0)
    assert r is None and rs is None and z.shape == (2, 10, 8, 4)
    torch.testing.assert_close(zs, z[:, :3], atol=1e-5, rtol=1e-5)


def test_kv_cache_rolls_when_full():
    cache = M.KVCache(4, 1, 1, 2, "cpu", torch.float32)
    for t in range(6):
        k = torch.full((1, 1, 1, 2), float(t))
        keys, _ = cache.update(k, k)
    assert cache.curr_len == 4
    assert keys[0, 0, :, 0].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_symlog_two_hot_targets_reconstruct_value():
    head = M.SymlogTwoHotHead(4, num_buckets=255)
    r = torch.tensor([0.0, 1.0, -3.5, 2.0, 1e10, -1e10])  # symlog(1e10) = 23 > 20
    low, lw, high, hw = head.get_targets(r)
    rec = lw * head.buckets[low] + hw * head.buckets[high]
    torch.testing.assert_close(rec[:4], head.to_symlog(r[:4]))
    assert torch.allclose(lw + hw, torch.ones_like(lw))
    assert rec[4] == 20.0 and rec[5] == -20.0  # clamped to the bucket range
    assert lw[0] == 1.0 and hw[0] == 0.0  # 0 lies exactly on the centre bucket


# ---------------------------------------------------------------- buffers / init


def test_rope_tables_do_not_depend_on_their_length():
    short = MHA_GQA(64 * 16, 16, 16, 64, max_seq_len=32)
    long = MHA_GQA(64 * 16, 16, 16, 64, max_seq_len=106)
    assert torch.equal(short.cos_cache, long.cos_cache[:32])
    assert torch.equal(short.sin_cache, long.sin_cache[:32])


def test_meta_build_plus_buffer_init_equals_cpu_build():
    enc, dec, dyn = _tiny_models()
    with torch.device("meta"):
        menc, mdec = build_tokenizer(TINY_TOK)
        mdyn = build_dynamics(TINY_DYN)
    for cpu_m, meta_m in ((enc, menc), (dec, mdec), (dyn, mdyn)):
        persistent = set(cpu_m.state_dict())
        init_nonpersistent_buffers(meta_m, "cpu")
        cpu_np = {k: b for k, b in cpu_m.named_buffers() if k not in persistent}
        meta_np = {k: b for k, b in meta_m.named_buffers() if k not in persistent}
        assert cpu_np.keys() == meta_np.keys() and cpu_np
        assert all(torch.equal(cpu_np[k], meta_np[k]) for k in cpu_np)
        assert all(b.is_meta for k, b in meta_m.named_buffers() if k in persistent)
        assert all(p.is_meta for p in meta_m.parameters())


def test_seeded_build_is_reproducible():
    a, b = _tiny_models(seed=3), _tiny_models(seed=3)
    for ma, mb in zip(a, b):
        sa, sb = ma.state_dict(), mb.state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_sdpa_backends_end_with_the_math_fallback():
    assert M._SDPA_BACKENDS == [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


def test_sliding_window_attention_runs_on_cpu_in_bf16():
    """Masked (sliding-window and agent-token) attention in bf16 on CPU."""
    _, _, dyn = _tiny_models()
    dyn = dyn.to(torch.bfloat16)
    x = _inputs(T=14)  # > context_len: sliding-window mask path
    with torch.no_grad():
        z, r = dyn(x["action"].bfloat16(), x["z"].bfloat16(), x["sigma"], x["step"])
    assert z.dtype == torch.bfloat16 and torch.isfinite(z).all() and torch.isfinite(r).all()


def test_training_mode_masks_and_diagnostics():
    enc, _, _ = _tiny_models()
    enc.train()
    x = _inputs()
    torch.manual_seed(0)
    _, _, Z1 = enc(x["video"], proprio=x["proprio"])
    torch.manual_seed(1)
    _, _, Z2 = enc(x["video"], proprio=x["proprio"])
    assert not torch.allclose(Z1, Z2)  # MAE + proprio masking are random in training
    assert 0.0 <= float(enc._last_z_sat_frac) <= 1.0 and float(enc._last_pretanh_abs_mean) > 0


@pytest.mark.parametrize("case", ["merged_causal", "merged_broadcast_mask", "per_sequence_mask"])
def test_sdpa_split_over_the_batch_matches_one_call(monkeypatch, case):
    """A batch of more than _SDPA_MAX_BATCH sequences is attended in several calls: same outputs and gradients."""
    torch.manual_seed(0)
    attn = MHA_GQA(32, 4, 2, 8, max_seq_len=16)
    if case == "per_sequence_mask":  # [10, S, D]: 10 sequences, one mask each
        x = torch.randn(10, 6, 32, requires_grad=True)
        mask = torch.randn(10, 1, 6, 6)
    else:  # [2, 5, S, D]: leading dims merged to 10 sequences
        x = torch.randn(2, 5, 6, 32, requires_grad=True)
        mask = None
        if case == "merged_broadcast_mask":
            mask = torch.zeros(6, 6)
            mask[:, 4:] = -1e4

    def run():
        x.grad = None
        attn.zero_grad()
        out = attn(x, x, attn_mask=mask, is_causal=case == "merged_causal")
        (out.square().sum() + out.sum()).backward()
        return [out.detach(), x.grad.clone(), *(p.grad.clone() for p in attn.parameters())]

    one = run()
    monkeypatch.setattr(M, "_SDPA_MAX_BATCH", 3)  # 10 sequences: calls of 3, 3, 3 and 1
    split = run()
    for a, b in zip(one, split):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
