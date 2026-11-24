# SPDX-License-Identifier: Apache-2.0
"""scripts/download_weights.py, scripts/compute_action_envelope.py, and the standalone scripts' imports."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from coupled_wm.rssm.rssm import build_rssm, save_rssm
from coupled_wm.world_model.export import save_dynamics_pretrained, save_tokenizer_pretrained
from coupled_wm.world_model.loading import dynamics_model_config, tokenizer_model_config
from coupled_wm.world_model.models import build_dynamics, build_tokenizer

REPO = Path(__file__).resolve().parents[1]
TOK = dict(image_size=[16, 16], patch_size=8, d_model=32, n_layers=4, num_heads_q=2, num_heads_kv_latent=2,
           temporal_every=2, n_latents=4, bottleneck_dim=4, use_proprio=True, proprio_dim=3, num_proprio=1,
           seq_len=8, context_len=0, dropout=0.0, mae_max_mask_prob=0.0)
DYN = dict(action_dim=4, num_latents=4, latent_dim=4, d_model=32, num_layers=4, num_heads=2, num_registers=2,
           num_tau_levels=16, temporal_every=2, mlp_ratio=2.0, seq_len=16, context_len=6, dropout=0.0,
           use_reward_head=True, mtp_length=2, reward_hidden_dim=16, reward_num_buckets=31,
           reward_symlog_range=[-20.0, 20.0])
RSSM = dict(action_size=4, image_size=16, stochastic_size=4, discrete_size=4, recurrent_state_size=16,
            dense_units=16, channels_multiplier=2, mlp_layers=1, reward_model_bins=15)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_action_envelope(make_h5, tmp_path):
    a = make_h5(tmp_path / "data" / "chunk_0000.h5", 10, action_dim=3, file_id=0)
    make_h5(tmp_path / "data" / "chunk_0001.h5", 6, action_dim=3, file_id=0)
    with h5py.File(a, "a") as f:
        cmd = f["commands"][()]
        cmd[:, 0] = np.linspace(-1.234, 0.5, 10)
        cmd[:, 2] = 0.0
        f["commands"][...] = cmd
    script = _script("compute_action_envelope")
    data = str(tmp_path / "data")
    snippet = yaml.safe_load(script.main([data, "--grip-dims", "1", "--decimals", "2", "--rounding", "up"]))
    with h5py.File(a, "r") as f, h5py.File(tmp_path / "data" / "chunk_0001.h5", "r") as g:
        absmax = np.abs(np.concatenate([f["commands"][()], g["commands"][()]])).max(0)
    up = snippet["env"]["action_scale"]
    # the env holds the envelope in fp32: it contains every recorded (fp32) command, by less than a step
    assert all(np.float32(e) >= v and e - 0.01 < v for e, v in zip(up, absmax) if v > 0)
    # grip dim 1 holds t + 0.01 for t = 0..9: its range, rounded outward
    assert snippet["env"]["grip_dims"] == [1] and snippet["env"]["grip_range"] == [0.01, 9.01]
    assert snippet["export"]["metadata"]["action"] == {"dim": 3, "grip_dims": [1], "grip_range": [0.01, 9.01],
                                                       "scale": up}

    no_grips = yaml.safe_load(script.main([data, "--grip-dims"]))
    nearest = no_grips["env"]["action_scale"]
    assert nearest == pytest.approx([max(round(v, 2), 0.01) for v in absmax])
    assert no_grips["env"]["grip_dims"] == [] and "grip_range" not in no_grips["env"]
    assert no_grips["export"]["metadata"]["action"] == {"dim": 3, "grip_dims": [], "scale": nearest}
    with pytest.raises(SystemExit):  # the grip layout is an explicit choice
        script.main([data])
    with pytest.raises(SystemExit):
        script.main([data, "--grip-dims", "3"])


def test_action_envelope_rounding():
    script = _script("compute_action_envelope")
    f32 = np.float32
    assert script.envelope(np.array([0.0, 0.3, 0.301]), 2, "up") == [0.01, 0.3, 0.31]
    assert script.envelope(np.array([0.0, f32(0.3), 0.301, 1.796]), 2) == [0.01, 0.3, 0.3, 1.8]
    # fp32 commands on a grid point stay there; one ulp above it, "up" goes to the next step
    on = [f32(0.3), f32(1.97), f32(0.07), f32(0.5), f32(2.5)]
    above = [np.nextafter(v, f32(10)) for v in on]
    assert script.envelope(np.array(on + above, dtype=np.float64), 2, "up") == [
        0.3, 1.97, 0.07, 0.5, 2.5, 0.31, 1.98, 0.08, 0.51, 2.51]
    grid = np.round(np.arange(1, 300) * 0.01, 2).astype(np.float32)
    values = np.concatenate([grid, np.nextafter(grid, f32(10)), np.nextafter(grid, f32(-10))])
    up = script.envelope(values.astype(np.float64), 2, "up")
    assert all(f32(e) >= v and e - 0.01 < v for e, v in zip(up, values))
    assert script.envelope(np.array([2.0]), 0, "up") == [2.0]
    assert script._yaml_list([20.0, 0.3], 0) == "[20, 0]"
    assert script._yaml_list([1.0, 0.25, 12345.67], 2) == "[1, 0.25, 12345.67]"


def _publish(root: Path, tokenizer_sha_from: Path | None = None) -> None:
    torch.manual_seed(0)
    enc, dec = build_tokenizer(TOK)
    tok = save_tokenizer_pretrained(root / "tokenizer", enc.state_dict(), dec.state_dict(), tokenizer_model_config(TOK))
    save_dynamics_pretrained(root / "dynamics", build_dynamics(DYN).state_dict(), dynamics_model_config(DYN),
                             tokenizer_dir=tokenizer_sha_from or tok)
    save_rssm(root / "rssm", build_rssm(RSSM), {"action": {"mode": "relative"}})


def test_download_weights(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake_snapshot_download(repo_id, revision, local_dir, token, allow_patterns):
        calls.append((repo_id, revision, allow_patterns))
        _publish(Path(local_dir))
        return local_dir

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    script = _script("download_weights")
    script.main(["--repo-id", "org/repo", "--out", str(tmp_path / "w")])
    assert calls == [("org/repo", None, ["tokenizer/*", "dynamics/*", "rssm/*"])]
    assert all((tmp_path / "w" / c / "model.safetensors").is_file() for c in ("tokenizer", "dynamics", "rssm"))

    # a dynamics model paired with another tokenizer is rejected
    other = tmp_path / "other"
    torch.manual_seed(1)
    enc, dec = build_tokenizer(TOK)
    save_tokenizer_pretrained(other, enc.state_dict(), dec.state_dict(), tokenizer_model_config(TOK))
    _publish(tmp_path / "bad", tokenizer_sha_from=other)
    with pytest.raises(ValueError, match="not the one this dynamics"):
        script.verify(tmp_path / "bad")
    script.verify(tmp_path / "bad", check_sha=False)
    with pytest.raises(FileNotFoundError, match="not a pretrained dir"):
        script.verify(tmp_path / "missing", ["rssm"])


STANDALONE = sorted((REPO / "tests" / "gpu").glob("*.py")) + sorted((REPO / "scripts").glob("*.py"))


@pytest.mark.parametrize("path", STANDALONE, ids=lambda p: str(p.relative_to(REPO)))
def test_standalone_scripts_run_from_a_source_checkout(path, tmp_path):
    """Each script finds ``coupled_wm`` without ``pip install -e .`` or the repo on PYTHONPATH."""
    paths = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and Path(p).resolve() != REPO]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths), "OMP_NUM_THREADS": "1"}
    out = subprocess.run([sys.executable, str(path), "--help"], cwd=tmp_path, env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
