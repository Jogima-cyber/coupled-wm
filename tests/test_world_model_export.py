# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
import yaml
from safetensors import safe_open

from coupled_wm.common.checkpoint import atomic_save
from coupled_wm.world_model import export as E
from coupled_wm.world_model.loading import (
    dynamics_model_config,
    file_sha256,
    load_dynamics,
    load_tokenizer,
    tokenizer_model_config,
)
from coupled_wm.world_model.models import build_dynamics, build_tokenizer

TINY_TOK = dict(image_size=[32, 32], patch_size=8, in_channels=3, d_model=64, n_layers=8, num_heads_q=4,
                num_heads_kv_latent=4, temporal_every=4, n_latents=8, bottleneck_dim=4, use_proprio=True,
                proprio_dim=5, num_proprio=1, seq_len=16, context_len=0, dropout=0.0, mae_max_mask_prob=0.9)
TINY_DYN = dict(action_dim=6, num_latents=8, latent_dim=4, d_model=64, num_layers=8, num_heads=4, num_registers=2,
                num_tau_levels=16, temporal_every=4, mlp_ratio=4.0, seq_len=40, context_len=12, dropout=0.0,
                use_reward_head=True, mtp_length=3, reward_hidden_dim=32, reward_num_buckets=31,
                reward_symlog_range=[-20.0, 20.0])


def _compiled(sd):
    return {f"_orig_mod.{k}": v for k, v in sd.items()}


@pytest.fixture(scope="module")
def states():
    torch.manual_seed(0)
    enc, dec = build_tokenizer(TINY_TOK)
    dyn = build_dynamics(TINY_DYN)
    return enc.state_dict(), dec.state_dict(), dyn.state_dict()


def _safetensors_keys(path):
    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k).dtype for k in f.keys()}


def test_cli_exports_training_checkpoints(states, tmp_path):
    enc, dec, dyn = states
    tok_ckpt, dyn_ckpt = tmp_path / "tok.pt", tmp_path / "dyn.pt"
    atomic_save({"enc": _compiled(enc), "dec": _compiled(dec), "global_update": 12, "optim": {"state": {}},
                 "model_config": tokenizer_model_config(TINY_TOK)}, tok_ckpt)
    atomic_save({"dyn": _compiled(dyn), "global_update": 34, "model_config": dynamics_model_config(TINY_DYN)},
                dyn_ckpt)
    E.main(["--ckpt", str(tok_ckpt), "--out", str(tmp_path / "tokenizer")])
    E.main(["--ckpt", str(dyn_ckpt), "--out", str(tmp_path / "dynamics"), "--tokenizer", str(tmp_path / "tokenizer")])

    tcfg = json.loads((tmp_path / "tokenizer" / "config.json").read_text())
    dcfg = json.loads((tmp_path / "dynamics" / "config.json").read_text())
    assert list(tcfg)[:3] == ["model_type", "format_version", "architecture"]
    assert tcfg["architecture"] == TINY_TOK and tcfg["provenance"]["global_update"] == 12
    assert dcfg["provenance"]["global_update"] == 34
    assert dcfg["tokenizer"]["sha256"] == file_sha256(tmp_path / "tokenizer" / "model.safetensors")
    keys = _safetensors_keys(tmp_path / "tokenizer" / "model.safetensors")
    assert len(keys) == len(enc) + len(dec)
    assert all(k.startswith(("encoder.", "decoder.")) and "_orig_mod" not in k for k in keys)
    assert set(keys.values()) == {torch.float32}
    assert set(_safetensors_keys(tmp_path / "dynamics" / "model.safetensors")) == set(dyn)
    load_tokenizer(tmp_path / "tokenizer")
    load_dynamics(tmp_path / "dynamics")


def test_cli_config_file_for_checkpoint_without_config(states, tmp_path):
    _, _, dyn = states
    ckpt = tmp_path / "dyn.pt"
    atomic_save({"dyn": _compiled(dyn)}, ckpt)
    with pytest.raises(ValueError, match="config="):
        E.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "out")])
    cfg_file = tmp_path / "model.yaml"
    cfg_file.write_text(yaml.safe_dump(TINY_DYN))
    E.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "out"), "--config", str(cfg_file)])
    assert json.loads((tmp_path / "out" / "config.json").read_text())["architecture"] == TINY_DYN


def test_export_rejects_bad_weights(states, tmp_path):
    enc, dec, dyn = states
    bad = dict(dyn)
    bad["out_proj.weight"] = torch.full_like(dyn["out_proj.weight"], float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        E.save_dynamics_pretrained(tmp_path / "d", bad, dynamics_model_config(TINY_DYN))
    with pytest.raises(RuntimeError, match="size mismatch"):
        E.save_tokenizer_pretrained(tmp_path / "t", enc, dec, tokenizer_model_config({**TINY_TOK, "proprio_dim": 6}))
    partial = {k: v for k, v in dyn.items() if k not in ("out_proj.weight", "out_proj.bias")}
    with pytest.raises(RuntimeError, match=r"missing 2: \['out_proj.bias', 'out_proj.weight'\]; unexpected 0"):
        E.save_dynamics_pretrained(tmp_path / "d", partial, dynamics_model_config(TINY_DYN))
    with pytest.raises(RuntimeError, match="missing 0: .*unexpected [1-9]"):
        E.save_dynamics_pretrained(tmp_path / "d", dyn, dynamics_model_config({**TINY_DYN, "use_reward_head": False}))
    assert not (tmp_path / "d").exists() and not (tmp_path / "t").exists()


def test_cli_metadata_merges_config_blocks(states, tmp_path, caplog, monkeypatch):
    """--metadata adds config.json blocks to a checkpoint that embeds its model config (where
    --config can only check the architecture), key by key within a block."""
    monkeypatch.setattr(E, "setup_logging", lambda: None)  # keep caplog's handler on the root logger
    _, _, dyn = states
    ckpt = tmp_path / "dyn.pt"
    atomic_save({"dyn": dyn, "global_update": 5,
                 "model_config": dynamics_model_config(TINY_DYN, action={"dim": 6, "scale": [1.0] * 6})}, ckpt)
    meta = tmp_path / "env.yaml"
    meta.write_text(yaml.safe_dump({"action": {"scale": [2.0] * 6, "grip_dims": [5], "grip_range": [0.0, 1.0]},
                                    "inference": {"kv_cache_len": 8}}))
    E.main(["--ckpt", str(ckpt), "--out", str(tmp_path / "d"), "--metadata", str(meta)])
    cfg = json.loads((tmp_path / "d" / "config.json").read_text())
    assert cfg["action"] == {"dim": 6, "scale": [2.0] * 6, "grip_dims": [5], "grip_range": [0.0, 1.0]}
    assert cfg["inference"] == {"kv_cache_len": 8} and cfg["provenance"]["global_update"] == 5
    with pytest.raises(KeyError, match="architecture"):
        E.export_checkpoint(ckpt, tmp_path / "x", metadata={"architecture": TINY_DYN})
    with caplog.at_level("WARNING", logger="coupled_wm.world_model.loading"):
        E.export_checkpoint(ckpt, tmp_path / "d2", config=dynamics_model_config(TINY_DYN, action={"dim": 6}))
    assert "blocks ['action'] of config= are ignored" in caplog.text
    assert json.loads((tmp_path / "d2" / "config.json").read_text())["action"] == {"dim": 6, "scale": [1.0] * 6}


def test_export_pairing_conflict_and_ambiguous_kind(states, tmp_path, caplog):
    enc, dec, dyn = states
    tok_dir = E.save_tokenizer_pretrained(tmp_path / "t", enc, dec, tokenizer_model_config(TINY_TOK))
    with pytest.raises(ValueError, match="not the tokenizer recorded"):
        E.save_dynamics_pretrained(tmp_path / "d", dyn, dynamics_model_config(TINY_DYN, tokenizer={"sha256": "ab"}),
                                   tokenizer_dir=tok_dir)
    # Trained from a tokenizer .pt: no sha256 to compare, so the pairing is by shape only (warned).
    with caplog.at_level("WARNING", logger="coupled_wm.world_model.export"):
        E.save_dynamics_pretrained(tmp_path / "d", dyn, dynamics_model_config(TINY_DYN, tokenizer={"path": "tok.pt"}),
                                   tokenizer_dir=tok_dir)
    assert "trained with the tokenizer tok.pt, whose sha256 was not recorded" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="coupled_wm.world_model.export"):
        E.save_dynamics_pretrained(tmp_path / "d", dyn, json.loads((tmp_path / "d" / "config.json").read_text()),
                                   tokenizer_dir=tok_dir)
    assert "sha256 was not recorded" not in caplog.text
    with pytest.raises(KeyError, match="cannot tell"):
        E.checkpoint_kind({"enc": {}, "dec": {}, "dyn": {}})
    with pytest.raises(KeyError, match="cannot tell"):
        E.checkpoint_kind({"model": {}})
