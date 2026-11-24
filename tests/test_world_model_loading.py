# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch

from coupled_wm.common.checkpoint import atomic_save
from coupled_wm.world_model.export import save_dynamics_pretrained, save_tokenizer_pretrained
from coupled_wm.world_model.loading import (
    check_pairing,
    dynamics_model_config,
    load_dynamics,
    load_state_dict_strict,
    load_tokenizer,
    read_dynamics_weights,
    read_tokenizer_weights,
    tokenizer_model_config,
    tokenizer_sha256,
)
from coupled_wm.world_model.models import build_dynamics, build_tokenizer

TINY_TOK = dict(image_size=[32, 32], patch_size=8, in_channels=3, d_model=64, n_layers=8, num_heads_q=4,
                num_heads_kv_latent=4, temporal_every=4, n_latents=8, bottleneck_dim=4, use_proprio=True,
                proprio_dim=5, num_proprio=1, seq_len=16, context_len=0, dropout=0.0, mae_max_mask_prob=0.9)
TINY_DYN = dict(action_dim=6, num_latents=8, latent_dim=4, d_model=64, num_layers=8, num_heads=4, num_registers=2,
                num_tau_levels=16, temporal_every=4, mlp_ratio=4.0, seq_len=40, context_len=12, dropout=0.0,
                use_reward_head=True, mtp_length=3, reward_hidden_dim=32, reward_num_buckets=31,
                reward_symlog_range=[-20.0, 20.0])


def _compiled_keys(sd):
    return {f"_orig_mod.{k}": v for k, v in sd.items()}


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    enc, dec = build_tokenizer(TINY_TOK)
    dyn = build_dynamics(TINY_DYN)
    g = torch.Generator().manual_seed(1)
    x = dict(video=torch.rand(2, 10, 3, 32, 32, generator=g), proprio=torch.randn(2, 10, 5, generator=g),
             action=torch.randn(2, 10, 6, generator=g), z=torch.rand(2, 10, 8, 4, generator=g) * 2 - 1,
             sigma=torch.randint(0, 16, (2, 10), generator=g), step=torch.randint(0, 5, (2, 10), generator=g))
    return enc.eval(), dec.eval(), dyn.eval(), x


def _outputs(enc, dec, dyn, x):
    with torch.no_grad():
        _, L, Z = enc(x["video"], proprio=x["proprio"])
        _, img, prop = dec(Z)
        z, r = dyn(x["action"], x["z"], x["sigma"], x["step"])
    return [L, Z, img, prop, z, r]


def _assert_same(a, b):
    for u, v in zip(a, b, strict=True):
        assert torch.equal(u, v)


@pytest.fixture
def train_ckpts(tiny, tmp_path):
    enc, dec, dyn, _ = tiny
    tok = tmp_path / "tok" / "latest.pt"
    dyn_p = tmp_path / "dyn" / "latest.pt"
    atomic_save({"enc": _compiled_keys(enc.state_dict()), "dec": _compiled_keys(dec.state_dict()),
                 "model_config": tokenizer_model_config(TINY_TOK), "global_update": 7}, tok)
    atomic_save({"dyn": _compiled_keys(dyn.state_dict()), "model_config": dynamics_model_config(TINY_DYN),
                 "global_update": 9, "optim": {"state": {}}}, dyn_p)
    return tok, dyn_p


def test_load_training_checkpoints(tiny, train_ckpts):
    enc, dec, dyn, x = tiny
    tok, dyn_p = train_ckpts
    enc2, dec2, tcfg = load_tokenizer(tok)
    dyn2, dcfg = load_dynamics(dyn_p)
    assert tcfg["model_type"] == "dreamerv4_tokenizer" and tcfg["architecture"]["d_model"] == 64
    assert dcfg["architecture"]["num_tau_levels"] == 16
    for m in (enc2, dec2, dyn2):
        assert not m.training and not any(p.requires_grad for p in m.parameters())
        assert not any(t.is_meta for t in list(m.parameters()) + list(m.buffers()))
    _assert_same(_outputs(enc, dec, dyn, x), _outputs(enc2, dec2, dyn2, x))


def test_pretrained_dir_roundtrip(tiny, tmp_path):
    enc, dec, dyn, x = tiny
    tok_dir = save_tokenizer_pretrained(tmp_path / "tokenizer", enc.state_dict(), dec.state_dict(),
                                        tokenizer_model_config(TINY_TOK, control_hz=10))
    dyn_dir = save_dynamics_pretrained(tmp_path / "dynamics", dyn.state_dict(), dynamics_model_config(TINY_DYN),
                                       tokenizer_dir=tok_dir)
    enc2, dec2, tcfg = load_tokenizer(tok_dir)
    dyn2, dcfg = load_dynamics(dyn_dir)
    assert tcfg["control_hz"] == 10 and "format_version" not in tcfg
    assert dcfg["tokenizer"]["sha256"] == tokenizer_sha256(tok_dir)
    _assert_same(_outputs(enc, dec, dyn, x), _outputs(enc2, dec2, dyn2, x))
    check_pairing(tcfg, dcfg, tok_dir)


def test_bf16_load_equals_env_style_construction(tiny, tmp_path):
    """Meta build + assign + cast == build on CPU, cast to bf16, load_state_dict (the paper env)."""
    enc, dec, dyn, x = tiny
    tok_dir = save_tokenizer_pretrained(tmp_path / "tokenizer", enc.state_dict(), dec.state_dict(),
                                        tokenizer_model_config(TINY_TOK))
    dyn_dir = save_dynamics_pretrained(tmp_path / "dynamics", dyn.state_dict(), dynamics_model_config(TINY_DYN))
    runtime_tok = {"seq_len": 20, "mae_max_mask_prob": 0.0}
    enc_ref, dec_ref = build_tokenizer({**TINY_TOK, **runtime_tok})
    dyn_ref = build_dynamics({**TINY_DYN, "seq_len": 30})
    for ref, src in ((enc_ref, enc), (dec_ref, dec), (dyn_ref, dyn)):
        ref.to(torch.bfloat16).eval()
        ref.load_state_dict(src.state_dict())
    enc2, dec2, _ = load_tokenizer(tok_dir, dtype=torch.bfloat16, runtime=runtime_tok)
    dyn2, _ = load_dynamics(dyn_dir, dtype=torch.bfloat16, runtime={"seq_len": 30})
    for ref, new in ((enc_ref, enc2), (dec_ref, dec2), (dyn_ref, dyn2)):
        ref_t = dict(ref.state_dict(keep_vars=True), **dict(ref.named_buffers()))
        new_t = dict(new.state_dict(keep_vars=True), **dict(new.named_buffers()))
        assert ref_t.keys() == new_t.keys()
        assert all(torch.equal(ref_t[k], new_t[k]) and ref_t[k].dtype == new_t[k].dtype for k in ref_t)
    xb = {k: (v.bfloat16() if v.is_floating_point() else v) for k, v in x.items()}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _assert_same(_outputs(enc_ref, dec_ref, dyn_ref, xb), _outputs(enc2, dec2, dyn2, xb))
        for m in (dyn_ref, dyn2):
            m.init_cache(2, "cpu", max_seq_len=4)
        outs = []
        for m in (dyn_ref, dyn2):
            with torch.no_grad():
                seq = [m.forward_step(xb["action"][:, :4], xb["z"][:, :4], xb["sigma"][:, :4], xb["step"][:, :4], 0)]
                for t in range(4, 10):
                    frame = (xb["action"][:, t:t + 1], xb["z"][:, t:t + 1], xb["sigma"][:, t:t + 1],
                             xb["step"][:, t:t + 1])
                    seq.append(m.forward_step(*frame, start_step_idx=t, update_cache=False))
                    seq.append(m.forward_step(*frame, start_step_idx=t, update_cache=True))
            outs.append([t for pair in seq for t in pair])
        _assert_same(*outs)


def test_checkpoint_without_config_needs_one(tiny, tmp_path):
    enc, dec, dyn, x = tiny
    path = tmp_path / "paper_like.pt"
    atomic_save({"enc": _compiled_keys(enc.state_dict()), "dec": _compiled_keys(dec.state_dict())}, path)
    with pytest.raises(ValueError, match="config="):
        load_tokenizer(path)
    with pytest.raises(KeyError, match="missing"):
        load_tokenizer(path, config={k: v for k, v in TINY_TOK.items() if k != "seq_len"})
    enc2, dec2, cfg = load_tokenizer(path, config=TINY_TOK)
    assert cfg["architecture"]["n_latents"] == 8
    with torch.no_grad():
        assert torch.equal(enc(x["video"], proprio=x["proprio"])[2], enc2(x["video"], proprio=x["proprio"])[2])


def test_config_conflicts_and_wrong_kind_raise(train_ckpts, tmp_path):
    tok, dyn_p = train_ckpts
    with pytest.raises(ValueError, match="temporal_every"):
        load_tokenizer(tok, config={**TINY_TOK, "temporal_every": 2})
    with pytest.raises(KeyError, match="not a tokenizer"):
        load_tokenizer(dyn_p)
    with pytest.raises(KeyError, match="not a dynamics"):
        load_dynamics(tok)
    enc, dec, cfg = read_tokenizer_weights(tok)
    tok_dir = save_tokenizer_pretrained(tmp_path / "t", enc, dec, cfg)
    with pytest.raises(ValueError, match="dreamerv4_dynamics"):
        load_dynamics(tok_dir)
    with pytest.raises(FileNotFoundError):
        load_tokenizer(tmp_path / "nope.pt")


def test_runtime_overrides(train_ckpts, tiny):
    tok, dyn_p = train_ckpts
    _, _, _, x = tiny
    enc_a, _, _ = load_tokenizer(tok)
    enc_b, _, _ = load_tokenizer(tok, runtime={"seq_len": 64, "context_len": 0})
    with torch.no_grad():
        assert torch.equal(enc_a(x["video"], proprio=x["proprio"])[2], enc_b(x["video"], proprio=x["proprio"])[2])
    with pytest.raises(KeyError, match="cannot be overridden"):
        load_tokenizer(tok, runtime={"temporal_every": 2})
    with pytest.raises(KeyError, match="cannot be overridden"):
        load_dynamics(dyn_p, runtime={"num_heads": 8})
    with pytest.raises(KeyError, match="cannot be overridden"):
        load_dynamics(dyn_p, runtime={"mae_max_mask_prob": 0.0})


def test_incomplete_or_tampered_pretrained_config_raises(tiny, tmp_path):
    enc, dec, _, _ = tiny
    d = save_tokenizer_pretrained(tmp_path / "t", enc.state_dict(), dec.state_dict(), tokenizer_model_config(TINY_TOK))
    cfg = json.loads((d / "config.json").read_text())
    del cfg["architecture"]["num_heads_q"]
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(KeyError, match="num_heads_q"):
        load_tokenizer(d)
    cfg["architecture"]["num_heads_q"] = 4
    cfg["architecture"]["d_model"] = 32
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_tokenizer(d)


def test_strict_key_checks(tiny, tmp_path):
    _, _, dyn, _ = tiny
    sd = dict(dyn.state_dict())
    extra = dict(sd, **{"policy_head.mean_heads.0.weight": torch.zeros(1)})
    path = tmp_path / "x.pt"
    atomic_save({"dyn": extra, "model_config": dynamics_model_config(TINY_DYN)}, path)
    with pytest.raises(RuntimeError, match="unexpected 1"):
        load_dynamics(path)
    missing = {k: v for k, v in sd.items() if not k.startswith("reward_head.heads.2")}
    atomic_save({"dyn": missing, "model_config": dynamics_model_config(TINY_DYN)}, path)
    with pytest.raises(RuntimeError, match="missing 3"):
        load_dynamics(path)


def test_backbone_to_reward_stage_init(tiny):
    """A backbone (no reward head) initialises a reward-head model with an explicit allow-list."""
    torch.manual_seed(0)
    backbone = build_dynamics({**TINY_DYN, "use_reward_head": False})
    reward = build_dynamics(TINY_DYN)
    with pytest.raises(RuntimeError, match="missing"):
        load_state_dict_strict(reward, _compiled_keys(backbone.state_dict()))
    load_state_dict_strict(reward, _compiled_keys(backbone.state_dict()), allow_missing=("agent_token", "reward_head."))
    assert torch.equal(reward.out_proj.weight, backbone.out_proj.weight)
    with pytest.raises(RuntimeError, match="unexpected"):
        load_state_dict_strict(backbone, reward.state_dict())


def test_check_pairing(tiny, tmp_path, caplog):
    enc, dec, dyn, _ = tiny
    tok_dir = save_tokenizer_pretrained(tmp_path / "t", enc.state_dict(), dec.state_dict(),
                                        tokenizer_model_config(TINY_TOK))
    dcfg = dynamics_model_config(TINY_DYN, tokenizer={"sha256": "0" * 64})
    check_pairing(tokenizer_model_config(TINY_TOK), dcfg)  # shapes only
    with caplog.at_level("WARNING", logger="coupled_wm.world_model.loading"):
        check_pairing(tokenizer_model_config(TINY_TOK), dynamics_model_config(TINY_DYN), tok_dir)
    assert "records no tokenizer sha256" in caplog.text  # nothing to compare with tok_dir: said so
    with pytest.raises(ValueError, match="not the one"):
        check_pairing(tokenizer_model_config(TINY_TOK), dcfg, tok_dir)
    with pytest.raises(ValueError, match="latents"):
        check_pairing(tokenizer_model_config({**TINY_TOK, "bottleneck_dim": 8}), dcfg)
    _, cfg = read_dynamics_weights(save_dynamics_pretrained(tmp_path / "d", dyn.state_dict(),
                                                            dynamics_model_config(TINY_DYN), tokenizer_dir=tok_dir))
    check_pairing(tokenizer_model_config(TINY_TOK), cfg, tok_dir)
