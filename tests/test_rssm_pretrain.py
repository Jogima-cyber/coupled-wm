# SPDX-License-Identifier: Apache-2.0
"""CPU tests of RSSM pretraining: row alignment, sampler, paper loss, training, CUDA-graph step
marks, saving, resuming, frame values, host-memory check, stop signals and startup logging."""

from __future__ import annotations

import json
import logging
import os
import signal
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from coupled_wm.common import system
from coupled_wm.common.checkpoint import load_pretrained
from coupled_wm.common.system import STOP_SIGNALS
from coupled_wm.data import HDF5SequenceDataset, list_h5_files
from coupled_wm.rssm import MSEDistribution, TwoHotEncodingDistribution, build_rssm, load_rssm, reconstruction_loss
from coupled_wm.rssm import pretrain
from coupled_wm.rssm.pretrain import (
    EpochBatchSampler,
    RSSMSequenceDataset,
    batch_to_device,
    build_datasets,
    frame_values,
    resolve_config,
    rssm_loss,
    split_holdout,
    train,
    validation_batch_size,
    window_to_rows,
)

TINY_MODEL = dict(action_size=3, image_size=16, stochastic_size=4, discrete_size=4, recurrent_state_size=16,
                  dense_units=8, channels_multiplier=2, mlp_layers=1, reward_model_bins=15)


def frame_code(t: int) -> int:
    return (7 * t) % 256


def write_episode_file(path: Path, n_frames: int, dones_at: list[int], file_id: int,
                       image_hw: tuple[int, int] = (32, 32), action_dim: int = 3) -> Path:
    """Frame t: every pixel equals ``frame_code(t)``; commands grow quadratically in t, so every
    command difference is non-zero and identifies its frame; ``rewards[t]`` is unique per frame."""
    t = np.arange(n_frames)
    images = np.broadcast_to(np.array([frame_code(i) for i in t], dtype=np.uint8)[:, None, None, None],
                             (n_frames, *image_hw, 3)).copy()
    commands = (1e-3 * (t[:, None] ** 2) * (np.arange(action_dim)[None] + 1) + 10.0 * file_id).astype(np.float32)
    rewards = (0.5 * t + 100.0 * file_id).astype(np.float32)
    dones = np.zeros(n_frames, np.float32)
    dones[dones_at] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=images)
        f.create_dataset("commands", data=commands)
        f.create_dataset("rewards", data=rewards)
        f.create_dataset("dones", data=dones)
    return path


def read_file(path: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in ("commands", "rewards", "dones")}


@pytest.fixture
def pool(tmp_path) -> Path:
    d = tmp_path / "pool"
    write_episode_file(d / "chunk_0000.h5", 40, dones_at=[17, 39], file_id=0)  # segments of 18 and 22 frames
    write_episode_file(d / "chunk_0001.h5", 30, dones_at=[5, 29], file_id=1)   # 6 (shorter than a window) and 24
    write_episode_file(d / "chunk_0002.h5", 26, dones_at=[], file_id=2)        # one segment, no done at all
    return d


def tiny_cfg(run_dir: Path, data_dir: Path, **train_overrides) -> dict:
    return {
        "run_dir": str(run_dir),
        "seed": 3,
        "data": {"dirs": [str(data_dir)], "num_workers": 0, "cache_workers": 0},
        "model": dict(TINY_MODEL),
        "train": {"epochs": 2, "batch_size": 4, "sequence_length": 8, "precision": "fp32", "compile": False,
                  "log_every": 1, "print_every": 5, "checkpoint_every_steps": 1000, **train_overrides},
    }


# ----------------------------------------------------------------------------- rows


def test_window_to_rows():
    commands = torch.tensor([[1.0, 0.0], [1.5, -1.0], [3.0, 2.0], [3.5, 2.0]])
    item = {"image": torch.arange(4.0).view(4, 1, 1, 1).expand(4, 3, 2, 2), "action": commands,
            "reward": torch.tensor([10.0, 11.0, 12.0, 13.0])}
    rows = window_to_rows(item)
    assert torch.equal(rows["obs"][:, 0, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(rows["action"], torch.tensor([[0.5, -1.0], [1.5, 3.0], [0.5, 0.0]]))
    assert torch.equal(rows["reward"], torch.tensor([11.0, 12.0, 13.0]))
    assert torch.equal(rows["is_first"], torch.tensor([1.0, 0.0, 0.0]))


@pytest.mark.parametrize("seq_len", [1, 5, 8])
def test_rows_pair_frame_t_with_its_command_change_and_reward(pool, seq_len):
    files = list_h5_files(pool)
    windows = HDF5SequenceDataset(files, window=seq_len + 1, image_size=16, load_proprio=False)
    ds = RSSMSequenceDataset(windows)
    raw = {f: read_file(f) for f in files}

    expected = 0
    for f, r in raw.items():
        ends = list(np.flatnonzero(r["dones"] > 0.5))
        if not ends or ends[-1] != len(r["dones"]) - 1:
            ends.append(len(r["dones"]) - 1)
        start = 0
        for end in ends:
            expected += max(0, (end - start + 1) - seq_len)  # windows of seq_len + 1 frames
            start = end + 1
    assert len(ds) == expected

    for idx in range(len(ds)):
        item = ds[idx]
        path, start = windows.window_info(idx)
        r = raw[path]
        frames = start + 1 + np.arange(seq_len)  # episode frame index of every row
        # the window (start .. start + seq_len) lies in one segment: no done before its last frame
        assert r["dones"][start:start + seq_len].sum() == 0
        codes = torch.round(item["obs"] * 255).to(torch.int64)
        for k, t in enumerate(frames):
            assert torch.all(codes[k] == frame_code(int(t))), (idx, k)
        cmd_change = torch.from_numpy(r["commands"][frames] - r["commands"][frames - 1])
        assert torch.equal(item["action"], cmd_change)
        assert torch.equal(item["reward"], torch.from_numpy(r["rewards"][frames]))
        assert torch.equal(item["is_first"], torch.eye(1, seq_len)[0])
        assert bool((item["action"] != 0).any(dim=-1).all()), "a row carries a zero (padded) action"


def test_batches_are_time_major(pool):
    windows = HDF5SequenceDataset(list_h5_files(pool), window=9, image_size=16, load_proprio=False)
    batch = next(iter(DataLoader(RSSMSequenceDataset(windows), batch_size=4)))
    obs, actions, rewards, is_first = batch_to_device(batch, torch.device("cpu"))
    assert obs.shape == (8, 4, 3, 16, 16) and actions.shape == (8, 4, 3)
    assert rewards.shape == (8, 4, 1) and is_first.shape == (8, 4, 1)
    assert torch.equal(obs, batch["obs"].transpose(0, 1) - 0.5)
    assert torch.equal(rewards[:, :, 0], batch["reward"].T)
    assert torch.equal(is_first[:, :, 0], batch["is_first"].T)
    assert is_first[0].eq(1).all() and is_first[1:].eq(0).all()


# ----------------------------------------------------------------------------- sampler


def test_validation_batch_is_the_per_process_training_batch():
    """Rank 0 validates alone: with N processes it reads each held-out batch as N batches of the per-process
    training batch (the CUDA graphs of a batch N times larger may not fit beside the training's); one process and a
    held-out set smaller than one batch are unchanged."""
    assert validation_batch_size(3072, 256, 1) == 256
    assert validation_batch_size(3072, 256, 2) == 128
    assert validation_batch_size(3072, 256, 4) == 64
    assert validation_batch_size(8, 4, 2) == 2
    assert validation_batch_size(22, 32, 2) == 22  # fewer windows than one batch: one batch of them
    with pytest.raises(ValueError, match="not divisible"):
        validation_batch_size(3072, 256, 3)


def test_epoch_sampler_order_resume_and_sharding():
    s = EpochBatchSampler(103, 8, seed=5)
    assert s.steps_per_epoch == 12 and len(s) == 12
    s.set_epoch(0)
    e0 = list(s)
    s.set_epoch(1)
    e1 = list(s)
    assert e0 != e1
    flat = [i for b in e0 for i in b]
    assert len(flat) == len(set(flat)) == 96 and all(0 <= i < 103 for i in flat)
    s.set_epoch(0)
    assert list(s) == e0  # depends only on (seed, epoch)
    s.set_epoch(1, start_step=5)
    assert len(s) == 7 and list(s) == e1[5:]

    shards = []
    for rank in range(4):
        r = EpochBatchSampler(103, 8, seed=5, rank=rank, world_size=4)
        r.set_epoch(1)
        shards.append(list(r))
    assert [sum((sh[k] for sh in shards), []) for k in range(12)] == e1
    with pytest.raises(ValueError, match="divisible"):
        EpochBatchSampler(103, 8, seed=5, world_size=3)
    with pytest.raises(ValueError, match="fewer than one batch"):
        EpochBatchSampler(7, 8, seed=5)


# ----------------------------------------------------------------------------- loss


# Adapted from sheeprl, sheeprl/utils/distribution.py.
class PaperTwoHot:
    """``TwoHotEncodingDistribution.log_prob`` of the paper's pretraining script (log-softmax as computed there)."""

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
        target = (torch.nn.functional.one_hot(below, len(self.bins)) * (dist_to_above / total)[..., None]
                  + torch.nn.functional.one_hot(above, len(self.bins)) * (dist_to_below / total)[..., None]).squeeze(-2)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdims=True)
        return (target * log_pred).sum(dim=(-1,))


def paper_script_loss(rssm, batch_obs, batch_actions, batch_rews, batch_is_firsts):
    """The training step body of the paper's RSSM pretraining script (hand-written unroll)."""
    T, B = batch_obs.shape[:2]
    S, D, R = rssm.stochastic_size, rssm.discrete, rssm.recurrent_state_size
    recurrent_state = torch.zeros(1, B, R)
    recurrent_states = torch.empty(T, B, R)
    priors_logits = torch.empty(T, B, S * D)
    posterior = torch.zeros(1, B, S, D)
    posteriors = torch.empty(T, B, S, D)
    posteriors_logits = torch.empty(T, B, S * D)
    embedded_obs = rssm.encoder(batch_obs.reshape(-1, *batch_obs.shape[-3:]))
    embedded_obs = embedded_obs.reshape(*batch_obs.shape[:-3], -1)
    for i in range(T):
        recurrent_state, posterior, _, posterior_logits, prior_logits = rssm.dynamic(
            posterior, recurrent_state, batch_actions[i:i + 1], embedded_obs[i:i + 1], batch_is_firsts[i:i + 1])
        recurrent_states[i] = recurrent_state
        priors_logits[i] = prior_logits
        posteriors[i] = posterior
        posteriors_logits[i] = posterior_logits
    latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)
    pr = PaperTwoHot(rssm.reward_model(latent_states))
    latent_states = latent_states.reshape(-1, *latent_states.shape[2:])
    reconstructed_obs = rssm.decoder(latent_states).reshape(batch_obs.shape)
    po = MSEDistribution(reconstructed_obs, dims=3)
    return reconstruction_loss(po, batch_obs, pr, batch_rews, priors_logits.view(T, B, S, D),
                               posteriors_logits.view(T, B, S, D), 0.5, 0.1, 1.0, 1.0)


def test_loss_is_the_paper_scripts_bitwise():
    torch.manual_seed(0)
    model = build_rssm(dict(TINY_MODEL, image_size=32, recurrent_state_size=24, mlp_layers=2))
    T, B = 6, 3
    g = torch.Generator().manual_seed(1)
    obs = torch.rand(T, B, 3, 32, 32, generator=g) - 0.5
    actions = 0.05 * torch.randn(T, B, 3, generator=g)
    rewards = torch.randint(0, 2, (T, B, 1), generator=g).float()
    is_first = torch.zeros(T, B, 1)
    is_first[0] = 1
    results = []
    for fn in (paper_script_loss, lambda m, *x: rssm_loss(m, *x)):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(7)
        out = fn(model, obs, actions, rewards, is_first)
        out[0].backward()
        results.append(([o.detach() for o in out], [p.grad.clone() for p in model.parameters()]))
    (o1, g1), (o2, g2) = results
    assert all(torch.equal(a, b) for a, b in zip(o1, o2))
    assert all(torch.equal(a, b) for a, b in zip(g1, g2))


def test_reward_term_keeps_the_paper_scripts_fp32_log_softmax_under_bf16(monkeypatch):
    """Under CUDA bf16 autocast the paper script's ``logits - logsumexp(logits)`` comes out fp32
    (``logsumexp`` autocasts to fp32). Pretraining must keep the script's precision. CPU autocast keeps ``logsumexp`` in bf16, so the
    script's formula is applied to the fp32-cast logits here; only the reward term is compared
    (on CPU, softmax and with it the KL terms differ from the script's fp32 buffers)."""
    # CPU autocast also leaves the categorical probabilities in bf16, which can fail the
    # distributions' simplex check at this tiny width; the check is not what is tested here.
    monkeypatch.setattr(torch.distributions.Distribution, "_validate_args", False)
    torch.manual_seed(0)
    model = build_rssm(dict(TINY_MODEL, image_size=32, recurrent_state_size=24, mlp_layers=2))
    T, B = 6, 3
    g = torch.Generator().manual_seed(1)
    obs = torch.rand(T, B, 3, 32, 32, generator=g) - 0.5
    actions = 0.05 * torch.randn(T, B, 3, generator=g)
    rewards = torch.randint(0, 3, (T, B, 1), generator=g).float()
    is_first = torch.zeros(T, B, 1)
    is_first[0] = 1
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        torch.manual_seed(7)
        reward_loss = rssm_loss(model, obs, actions, rewards, is_first)[3]
        torch.manual_seed(7)
        reward_logits = model(obs, actions, is_first)[1]
    assert reward_logits.dtype == torch.bfloat16
    script = (-PaperTwoHot(reward_logits.float()).log_prob(rewards)).mean()
    rl_finetuning = (-TwoHotEncodingDistribution(reward_logits, dims=1).log_prob(rewards)).mean()
    assert torch.equal(reward_loss, script)
    assert not torch.equal(reward_loss, rl_finetuning)  # the comparison tells the two precisions apart


# ----------------------------------------------------------------------------- config


def test_resolve_config_validates(tmp_path, pool):
    cfg = tiny_cfg(tmp_path / "run", pool)
    out = resolve_config(cfg)
    assert out["optim"] == {"lr": 1e-4, "eps": 1e-8} and out["init_from"] is None
    assert out["model"]["cnn_output_dim"] == 16 and out["model"]["bottleneck_spatial"] == 1
    for bad, match in [
        ({**cfg, "extra": 1}, "unknown top-level"),
        ({**cfg, "train": {**cfg["train"], "lr": 1e-3}}, "unknown keys in 'train'"),
        ({**cfg, "train": {**cfg["train"], "action_mode": "relative"}}, "unknown keys in 'train'"),
        ({**cfg, "optim": {"learning_rate": 1e-3}}, "unknown keys in 'optim'"),
        ({**cfg, "model": {**cfg["model"], "num_bins": 101}}, "unknown keys in 'model'"),
        ({**cfg, "run_dir": None}, "run_dir"),
        ({**cfg, "data": {**cfg["data"], "reward_key": None}}, "reward_key"),
        ({**cfg, "data": {**cfg["data"], "cache_dtype": "bfloat16"}}, "cache_dtype"),
        ({**cfg, "validation": {"holdout_every": 1}}, "holdout_every"),
    ]:
        with pytest.raises((KeyError, ValueError), match=match):
            resolve_config(bad)


def test_frame_values():
    """Only settings that reproduce the environment's float32 frames are 'exact'."""
    for resized in (False, True):
        assert frame_values(False, "uint8", resized) == "exact"  # streamed: the cache dtype is unused
        assert frame_values(True, "float32", resized) == "exact"
        assert frame_values(True, "float16", resized) == "float16-rounded"
    assert frame_values(True, "uint8", resized=False) == "exact"  # stored uint8 frames, cached as they are
    assert frame_values(True, "uint8", resized=True) == "uint8-requantised"


def test_holdout_split():
    files = [f"f{i}" for i in range(7)]
    assert split_holdout(files, 0) == (files, [])
    train_files, val_files = split_holdout(files, 3)
    assert val_files == ["f2", "f5"] and train_files == ["f0", "f1", "f3", "f4", "f6"]
    with pytest.raises(ValueError, match="empty split"):
        split_holdout(files[:1], 2)


def test_action_size_must_match_the_data(tmp_path, pool):
    cfg = resolve_config({**tiny_cfg(tmp_path / "run", pool), "model": dict(TINY_MODEL, action_size=4)})
    with pytest.raises(ValueError, match="model.action_size is 4"):
        train(cfg)  # from the file metadata, before any frame is read or cached
    with pytest.raises(ValueError, match="expected action_dim=4"):
        build_datasets(cfg)  # the dataset checks it before building its cache


def test_held_out_data_is_built_only_where_it_is_used(tmp_path, pool):
    cfg = resolve_config({**tiny_cfg(tmp_path / "run", pool), "validation": {"holdout_every": 3}})
    train_ds, val_ds, record = build_datasets(cfg)
    assert val_ds is not None and record["validation_windows"] == len(val_ds)
    train_ds2, val_ds2, record2 = build_datasets(cfg, with_validation=False)  # ranks other than 0
    assert val_ds2 is None and "validation_windows" not in record2
    assert record2["validation_files"] == record["validation_files"] == ["chunk_0002.h5"]
    assert len(train_ds2) == len(train_ds) and record2["source_image_sizes"] == [[32, 32]]


# ----------------------------------------------------------------------------- training


def test_tiny_training_run_writes_loadable_epoch_dirs(tmp_path, pool):
    run = tmp_path / "run"
    train(tiny_cfg(run, pool))
    ckpt = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    spe = ckpt["state"]["steps_per_epoch"]
    # windows of 9 frames: segments of 18, 22, 6 (too short), 24 and 26 frames
    assert ckpt["data"]["num_windows"] == 10 + 14 + 16 + 18 and spe == 58 // 4
    assert ckpt["state"] == {"epoch": 2, "step": 0, "global_step": 2 * spe, "steps_per_epoch": spe}
    assert all(not k.startswith("_orig_mod") and "._orig_mod." not in k for k in ckpt["model"])

    for epoch in (1, 2):
        rssm, stored = load_rssm(run / f"rssm_e{epoch}", config=TINY_MODEL)
        assert stored["provenance"]["epoch"] == epoch and stored["provenance"]["global_step"] == epoch * spe
    state_dict, stored = load_pretrained(run / "rssm_e2")
    strict = build_rssm(stored)
    strict.load_state_dict(state_dict, strict=True)
    for k, v in ckpt["model"].items():
        assert torch.equal(state_dict[k], v), k
    e1, _ = load_pretrained(run / "rssm_e1")
    assert not all(torch.equal(e1[k], state_dict[k]) for k in e1)  # epoch 2 trained further

    cfg_json = json.loads((run / "rssm_e2" / "config.json").read_text())
    assert cfg_json["model_type"] == "rssm" and cfg_json["action_size"] == 3 and cfg_json["image_size"] == 16
    assert cfg_json["action"]["mode"] == "relative"
    assert cfg_json["action"]["definition"] == "commands[t] - commands[t-1]"
    assert cfg_json["preprocessing"]["image_size"] == [16, 16]
    assert cfg_json["preprocessing"]["h5_color_order"] == "bgr"
    assert cfg_json["preprocessing"]["frame_values"] == "exact"
    assert cfg_json["data"]["files"] == ["chunk_0000.h5", "chunk_0001.h5", "chunk_0002.h5"]
    assert cfg_json["data"]["num_windows"] == 58 and cfg_json["data"]["window_frames"] == 9
    assert cfg_json["data"]["source_image_sizes"] == [[32, 32]]
    prov = cfg_json["provenance"]
    assert prov["optimizer"]["lr"] == 1e-4 and prov["optimizer"]["eps"] == 1e-8 and prov["batch_size"] == 4
    assert prov["loss"]["kl_dynamic"] == 0.5 and prov["validation_loss"] is None
    assert not any(os.path.isabs(f) for f in cfg_json["data"]["files"])  # no absolute paths in the export

    assert (run / "config.yaml").is_file()
    assert any(p.name.startswith("events.out.tfevents") for p in (run / "tb").iterdir())


def test_a_finished_run_returns_before_reading_data(tmp_path, pool, monkeypatch):
    run = tmp_path / "run"
    train(tiny_cfg(run, pool, epochs=1))
    mtime = (run / "latest.pt").stat().st_mtime_ns

    def no_data(*args, **kwargs):
        raise AssertionError("a finished run read its data")

    monkeypatch.setattr(pretrain, "select_files", no_data)
    monkeypatch.setattr(pretrain, "build_datasets", no_data)
    train(tiny_cfg(run, pool, epochs=1))  # complete: nothing to do
    assert (run / "latest.pt").stat().st_mtime_ns == mtime
    cfg = tiny_cfg(run, pool, epochs=1)
    cfg["optim"] = {"lr": 3e-4}
    with pytest.raises(ValueError, match="optim.lr"):  # the data-free resume checks still run
        train(cfg)


def test_resume_is_bitwise_identical_to_an_uninterrupted_run(tmp_path, pool):
    full = tmp_path / "full"
    train(tiny_cfg(full, pool))
    spe = torch.load(full / "latest.pt", map_location="cpu", weights_only=False)["state"]["steps_per_epoch"]

    run = tmp_path / "pieces"
    cfg = tiny_cfg(run, pool, max_steps=4)
    train(cfg)
    state = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["state"]
    assert state == {"epoch": 0, "step": 4, "global_step": 4, "steps_per_epoch": spe}
    assert not (run / "rssm_e1").exists()

    cfg = tiny_cfg(run, pool, max_steps=spe)  # stops exactly at the end of epoch 1
    cfg["data"]["cache"] = True  # float32 RAM cache: the same frames as streaming
    train(cfg)
    assert torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["state"]["epoch"] == 1
    assert (run / "rssm_e1").is_dir() and not (run / "rssm_e2").exists()

    train(tiny_cfg(run, pool, max_steps=None))
    a = torch.load(full / "latest.pt", map_location="cpu", weights_only=False)
    b = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert a["state"] == b["state"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    for epoch in (1, 2):
        wa = load_file(full / f"rssm_e{epoch}" / "model.safetensors")
        wb = load_file(run / f"rssm_e{epoch}" / "model.safetensors")
        assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_resume_rejects_a_different_run(tmp_path, pool):
    run = tmp_path / "run"
    train(tiny_cfg(run, pool, max_steps=3))
    cfg = tiny_cfg(run, pool, max_steps=6)
    cfg["optim"] = {"lr": 3e-4}
    with pytest.raises(ValueError, match="optim.lr"):
        train(cfg)
    cfg = tiny_cfg(run, pool, max_steps=6)
    cfg["model"]["dense_units"] = 12
    with pytest.raises(ValueError, match="architecture"):
        train(cfg)
    write_episode_file(pool / "chunk_0003.h5", 20, dones_at=[], file_id=3)
    with pytest.raises(ValueError, match="data pool"):
        train(tiny_cfg(run, pool, max_steps=6))


def _cache(cfg: dict, dtype: str | None) -> dict:
    cfg["data"].update({"cache": dtype is not None, "cache_dtype": dtype or "float32"})
    return cfg


def _no_cache(*args, **kwargs):
    raise AssertionError("the cache was built")


def test_resume_keeps_the_frame_values_and_the_start_time(tmp_path, pool, monkeypatch):
    """The pool's 32x32 frames are resized to 16x16: a uint8 cache re-quantises them and a float16
    cache rounds them, so a run started on exact frames refuses both; no cache and a float32 cache
    give the same frames and may alternate."""
    run = tmp_path / "run"
    train(tiny_cfg(run, pool, max_steps=2))  # streamed frames
    latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["frame_values"] == "exact"
    latest["run_started_utc"] = "2001-02-03T04:05:06+00:00"  # tell the first launch's start time apart
    torch.save(latest, run / "latest.pt")
    with monkeypatch.context() as m:
        m.setattr(HDF5SequenceDataset, "_build_cache", _no_cache)  # refused before any frame is cached
        for dtype, values in (("uint8", "uint8-requantised"), ("float16", "float16-rounded")):
            with pytest.raises(ValueError, match=f"'exact' frames.*gives '{values}' frames"):
                train(_cache(tiny_cfg(run, pool, max_steps=4), dtype))
    train(_cache(tiny_cfg(run, pool, max_steps=4), "float32"))
    train(_cache(tiny_cfg(run, pool, epochs=1), None))
    stored = json.loads((run / "rssm_e1" / "config.json").read_text())
    assert stored["preprocessing"]["frame_values"] == "exact"
    assert stored["provenance"]["run_started_utc"] == "2001-02-03T04:05:06+00:00"

    native = tmp_path / "native"  # frames stored at the model's 16x16: a uint8 cache holds them exactly
    for i in range(3):
        write_episode_file(native / f"chunk_{i:04d}.h5", 30, dones_at=[], file_id=i, image_hw=(16, 16))
    run2 = tmp_path / "run2"
    train(tiny_cfg(run2, native, max_steps=2))
    train(_cache(tiny_cfg(run2, native, max_steps=4), "uint8"))
    assert torch.load(run2 / "latest.pt", map_location="cpu", weights_only=False)["state"]["global_step"] == 4
    with pytest.raises(ValueError, match="'float16-rounded' frames"):
        train(_cache(tiny_cfg(run2, native, max_steps=6), "float16"))


@pytest.mark.parametrize("compile_", [True, False])
def test_compiled_runs_mark_a_cuda_graph_step_before_every_forward(tmp_path, pool, monkeypatch, compile_):
    """With ``train.compile`` every training step and every validation batch starts its own
    CUDA-graph generation right before its forward; without it nothing is marked."""
    events = []
    loss = pretrain.rssm_loss

    def recorded_loss(*args, **kwargs):
        events.append("train" if torch.is_grad_enabled() else "validation")
        return loss(*args, **kwargs)

    monkeypatch.setattr(pretrain, "rssm_loss", recorded_loss)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: events.append("mark"))
    monkeypatch.setattr(pretrain, "compile_rssm", lambda model, mode: None)  # no compiler on the CPU
    cfg = tiny_cfg(tmp_path / "run", pool, epochs=1, compile=compile_)
    cfg["validation"] = {"holdout_every": 3, "num_batches": 2, "every_steps": 4}
    train(cfg)
    forwards = [e for e in events if e != "mark"]
    # 40 training windows: 10 steps; validation (2 batches) after steps 4 and 8 and at the epoch end.
    assert forwards.count("train") == 10 and forwards.count("validation") == 6
    if compile_:
        four_steps_then_validation = ["mark", "train"] * 4 + ["mark", "validation"] * 2
        assert events == four_steps_then_validation * 2 + ["mark", "train"] * 2 + ["mark", "validation"] * 2
    else:
        assert events == forwards


def test_validation_holdout_and_init_from(tmp_path, pool):
    run = tmp_path / "run"
    cfg = tiny_cfg(run, pool, epochs=1)
    cfg["validation"] = {"holdout_every": 3, "num_batches": 2, "every_steps": 4}
    train(cfg)
    stored = json.loads((run / "rssm_e1" / "config.json").read_text())
    assert stored["data"]["files"] == ["chunk_0000.h5", "chunk_0001.h5"]
    assert stored["data"]["validation_files"] == ["chunk_0002.h5"]
    assert np.isfinite(stored["provenance"]["validation_loss"])

    run2 = tmp_path / "run2"
    cfg2 = tiny_cfg(run2, pool, epochs=1, max_steps=1)
    cfg2["init_from"] = str(run / "rssm_e1")
    cfg2["optim"] = {"lr": 0.0}
    train(cfg2)
    a, _ = load_pretrained(run / "rssm_e1")
    b = torch.load(run2 / "latest.pt", map_location="cpu", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)  # lr 0: the weights stay those of init_from


# ----------------------------------------------------------------------------- host memory


def test_a_cache_larger_than_the_memory_stops_before_it_is_built(tmp_path, pool, monkeypatch):
    monkeypatch.setattr(HDF5SequenceDataset, "_build_cache", _no_cache)
    frames = 40 + 30 + 26
    monkeypatch.setattr(pretrain, "available_memory", lambda: frames * 3 * 16 * 16 * 4 - 1)
    with pytest.raises(MemoryError, match="data.cache=false"):
        train(_cache(tiny_cfg(tmp_path / "run", pool), "float32"))
    with pytest.raises(AssertionError, match="cache was built"):  # a uint8 cache needs a quarter: it fits
        train(_cache(tiny_cfg(tmp_path / "run", pool), "uint8"))
    train(_cache(tiny_cfg(tmp_path / "run", pool, max_steps=1), None))  # streaming needs no RAM cache


# ----------------------------------------------------------------------------- stop signals


def _signal_at_step(monkeypatch, step: int, signum: int) -> None:
    """Make the training loop's ``step``-th loss computation (1-based) send ``signum`` to this process."""
    calls = {"n": 0}
    loss = pretrain.rssm_loss

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == step:
            os.kill(os.getpid(), signum)
        return loss(*args, **kwargs)

    monkeypatch.setattr(pretrain, "rssm_loss", wrapped)


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGUSR1"])
def test_a_stop_signal_saves_and_the_run_resumes_bitwise(tmp_path, pool, monkeypatch, signame):
    signum = getattr(signal, signame)
    full = tmp_path / "full"
    train(tiny_cfg(full, pool))

    run = tmp_path / "run"
    handler = signal.getsignal(signum)
    with monkeypatch.context() as m:
        _signal_at_step(m, 5, signum)
        train(tiny_cfg(run, pool))
    assert signal.getsignal(signum) is handler  # the previous handler is restored
    state = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["state"]
    assert (state["epoch"], state["step"]) == (0, 5)  # saved after the step that was running
    assert not (run / "rssm_e1").exists()

    train(tiny_cfg(run, pool))
    a = torch.load(full / "latest.pt", map_location="cpu", weights_only=False)
    b = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert a["state"] == b["state"]
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])


def test_a_stop_signal_at_the_end_of_an_epoch(tmp_path, pool, monkeypatch):
    run = tmp_path / "run"
    spe = 58 // 4
    with monkeypatch.context() as m:
        _signal_at_step(m, spe, signal.SIGTERM)
        train(tiny_cfg(run, pool))
    state = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)["state"]
    assert (state["epoch"], state["step"]) == (1, 0)
    assert (run / "rssm_e1").is_dir() and not (run / "rssm_e2").exists()


def test_data_loader_workers_ignore_the_stop_signals(tmp_path, pool, monkeypatch):
    """Batch schedulers signal every process of a job: the workers must outlive the main process's save."""
    out = tmp_path / "workers"
    out.mkdir()
    ignore = system.ignore_stop_signals

    def init(worker_id: int) -> None:  # runs in the forked worker
        ignore(worker_id)
        names = [signal.getsignal(getattr(signal, s)).name for s in STOP_SIGNALS]
        (out / str(os.getpid())).write_text(" ".join(names))

    monkeypatch.setattr(system, "ignore_stop_signals", init)
    cfg = tiny_cfg(tmp_path / "run", pool, max_steps=2)
    cfg["data"]["num_workers"] = 1
    train(cfg)
    assert [p.read_text() for p in out.iterdir()] == ["SIG_IGN SIG_IGN"]


# ----------------------------------------------------------------------------- startup


def test_startup_logs_the_absolute_run_dir_and_the_device(tmp_path, pool, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.INFO, logger="coupled_wm.rssm.pretrain")
    train(tiny_cfg(Path("runs/rel"), pool, max_steps=1))
    run = Path.cwd() / "runs" / "rel"
    messages = [r.getMessage() for r in caplog.records]
    assert f"new run in {run}" in messages
    assert any(m.startswith("rank 0 of 1 (local rank 0), device cpu") for m in messages)
    assert any("training on the CPU" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert f"saved {run / 'latest.pt'} (epoch 0, step 1)" in messages

    caplog.clear()
    train(tiny_cfg(Path("runs/rel"), pool, max_steps=2))
    messages = [r.getMessage() for r in caplog.records]
    assert f"resuming the run in {run}" in messages
    assert f"resumed from {run / 'latest.pt'} at epoch 0, step 1 (global step 1)" in messages


# ----------------------------------------------------------------------------- the G1 config

REPO = Path(__file__).resolve().parents[1]


def test_g1_config_is_the_paper_recipe(monkeypatch):
    from coupled_wm.common.config import load_config
    from coupled_wm.rssm import rssm_config

    monkeypatch.setenv("G1_DATA_ROOT", "/data/g1")
    monkeypatch.setenv("RUN_ROOT", "/scratch/runs")
    cfg = resolve_config(load_config(REPO / "configs/rssm/g1_rssm_pretrain.yaml"))
    assert cfg["run_dir"] == "/scratch/runs/g1_rssm" and cfg["init_from"] is None and cfg["seed"] == 42
    assert cfg["model"] == rssm_config(dict(
        action_size=22, image_size=64, stochastic_size=32, discrete_size=32, recurrent_state_size=4096,
        dense_units=1024, channels_multiplier=96, mlp_layers=5, reward_model_bins=255))
    assert cfg["optim"] == {"lr": 1e-4, "eps": 1e-8}
    t = cfg["train"]
    assert (t["batch_size"], t["sequence_length"], t["epochs"]) == (16, 64, 10)
    assert (t["kl_dynamic"], t["kl_representation"], t["kl_free_nats"], t["kl_regularizer"]) == (0.5, 0.1, 1.0, 1.0)
    assert t["precision"] == "bf16" and t["compile"]
    assert t["compile_mode"] == "reduce-overhead" and t["save_every_epochs"] == 1
    d = cfg["data"]
    assert d["dirs"] == ["/data/g1/new_wm_h5_segmented_small/world_model",
                         "/data/g1/new_wm_h5_demos_segmented_small/world_model"]
    assert d["include_globs"] == ["chunk_*.h5"] and d["color_order"] == "bgr" and d["reward_key"] == "rewards"
    assert d["reward_override_dir"] is None and cfg["validation"]["holdout_every"] == 0
    assert frame_values(d["cache"], d["cache_dtype"], resized=True) == "exact"  # 256-px data, 64-px model

    # The RL warm start must strict-load: same architecture as the RL rssm block.
    rl = load_config(REPO / "configs/rl/g1_dmo.yaml")
    rl_arch = rssm_config({**rl["rssm"], "action_size": 22, "image_size": rl["env"]["img_size"]})
    assert rl_arch == cfg["model"]


def test_resumed_launch_purges_events_after_the_checkpoint(tmp_path, pool, monkeypatch):
    purge_steps = []

    class Recording(pretrain.ExperimentLogger):
        def __init__(self, *args, purge_step=None, **kwargs):
            purge_steps.append(purge_step)
            super().__init__(*args, purge_step=purge_step, **kwargs)

    monkeypatch.setattr(pretrain, "ExperimentLogger", Recording)
    run = tmp_path / "run"
    train(tiny_cfg(run, pool, max_steps=2))
    train(tiny_cfg(run, pool, max_steps=3))
    assert purge_steps == [0, 3]  # step 2 (0-based) is the resumed launch's first; its loss was logged before
