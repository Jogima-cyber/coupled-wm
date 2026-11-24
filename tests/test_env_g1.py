# SPDX-License-Identifier: Apache-2.0
"""The G1 world-model environment on tiny random world models (CPU)."""

from __future__ import annotations

import gc
import importlib.util
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from coupled_wm.common.config import load_config
from coupled_wm.common.system import ignore_stop_signals
from coupled_wm.data.h5_dataset import frames_to_tensor
from coupled_wm.envs import DreamerV4VecEnvG1Relative, draw_seed_pass, make_env
from coupled_wm.envs.g1_env import _check_kv_cache_fits
from coupled_wm.world_model.export import save_dynamics_pretrained, save_tokenizer_pretrained
from coupled_wm.world_model.loading import dynamics_model_config, tokenizer_model_config
from coupled_wm.world_model.models import G1_ACTION, G1_INFERENCE, build_dynamics, build_tokenizer

REPO = Path(__file__).resolve().parents[1]

TINY_TOK = dict(image_size=[32, 32], patch_size=8, in_channels=3, d_model=32, n_layers=4, num_heads_q=2,
                num_heads_kv_latent=2, temporal_every=2, n_latents=8, bottleneck_dim=4, use_proprio=True,
                proprio_dim=5, num_proprio=1, seq_len=64, context_len=0, dropout=0.0, mae_max_mask_prob=0.0)
TINY_DYN = dict(action_dim=6, num_latents=8, latent_dim=4, d_model=32, num_layers=4, num_heads=2, num_registers=2,
                num_tau_levels=16, temporal_every=2, mlp_ratio=2.0, seq_len=64, context_len=96, dropout=0.0,
                use_reward_head=True, mtp_length=2, reward_hidden_dim=16, reward_num_buckets=33,
                reward_symlog_range=[-20.0, 20.0])
TINY_INFERENCE = dict(num_inference_steps=2, d_min_idx=4, seq_cor_tau_idx=2, init_context_len=4, kv_cache_len=6,
                      reward_mtp_index=1)
TINY_ACTION = dict(dim=6, base_velocity_dims=[0, 1], grip_dims=[4, 5], grip_range=[0.0, 1.0],
                   scale=[0.3, 0.4, 0.5, 0.6, 1.0, 1.0])
BASE, GRIPS = (0, 1), (4, 5)
ENV_LOG = "coupled_wm.envs.g1_env"


# ----------------------------------------------------------------------------- fixtures

def write_episode_file(path: Path, n_frames: int, scale, *, base_dims=BASE, grip_dims=GRIPS, dones_at=None,
                       base_nonzero=(), proprio_dim=5, image_hw=(32, 32), seed=0) -> Path:
    """An h5 file in the release format with G1-like commands: joints random-walk inside 0.9x
    the envelope, grips in {0, 1}, base dims 0 except on the ``base_nonzero`` frames."""
    rng = np.random.default_rng(seed)
    scale = np.asarray(scale, dtype=np.float32)
    commands = np.clip(np.cumsum(rng.normal(0.0, 0.05, (n_frames, len(scale))), axis=0), -0.9 * scale, 0.9 * scale)
    commands[:, list(grip_dims)] = rng.random((n_frames, len(grip_dims))) > 0.5
    commands[:, list(base_dims)] = 0.0
    for t in base_nonzero:
        commands[t, base_dims[t % len(base_dims)]] = 0.25
    dones = np.zeros(n_frames, dtype=np.float32)
    dones[[n_frames - 1] if dones_at is None else dones_at] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=rng.integers(0, 256, (n_frames, *image_hw, 3), dtype=np.uint8))
        f.create_dataset("commands", data=commands.astype(np.float32))
        f.create_dataset("dones", data=dones)
        f.create_dataset("proprio", data=rng.normal(size=(n_frames, proprio_dim)).astype(np.float32))
        f.create_dataset("rewards", data=(rng.random(n_frames) > 0.8).astype(np.float32))
    return path


def save_weights(out: Path, *, preprocessing=None, dyn_arch=TINY_DYN, **dyn_metadata) -> Path:
    """Tiny random tokenizer + dynamics pretrained dirs under ``out``; ``dyn_metadata`` are the
    dynamics config.json blocks (inference, action, training, ...)."""
    torch.manual_seed(0)
    enc, dec = build_tokenizer(TINY_TOK)
    dyn = build_dynamics(dyn_arch)
    tok_meta = {} if preprocessing is None else {"preprocessing": preprocessing}
    tok_dir = save_tokenizer_pretrained(out / "tokenizer", enc.state_dict(), dec.state_dict(),
                                        tokenizer_model_config(TINY_TOK, **tok_meta))
    save_dynamics_pretrained(out / "dynamics", dyn.state_dict(), dynamics_model_config(dyn_arch, **dyn_metadata),
                             tokenizer_dir=tok_dir)
    return out


@pytest.fixture(scope="module")
def tiny_weights(tmp_path_factory):
    return save_weights(tmp_path_factory.mktemp("weights"), preprocessing={"h5_color_order": "bgr"},
                        inference=TINY_INFERENCE, action=TINY_ACTION)


@pytest.fixture(scope="module")
def tiny_pool(tmp_path_factory):
    d = tmp_path_factory.mktemp("pool")
    scale = TINY_ACTION["scale"]
    write_episode_file(d / "chunk_0000.h5", 30, scale, dones_at=[11, 29], base_nonzero=[3, 9, 20], seed=1)
    write_episode_file(d / "chunk_0001.h5", 26, scale, base_nonzero=[5, 6, 7], seed=2)
    write_episode_file(d / "chunk_0002.h5", 20, scale, dones_at=[2, 19], seed=3)
    return d


def new_env(weights, pool, data=None, **kw) -> DreamerV4VecEnvG1Relative:
    args = dict(num_envs=3, data={"dirs": [str(pool)], "num_workers": 0, **(data or {})},
                tokenizer_ckpt=weights / "tokenizer", dynamics_ckpt=weights / "dynamics", device="cpu",
                episode_length=5, img_size=16, compile_encoder=False, drop_action_indices=list(BASE))
    args.update(kw)
    return DreamerV4VecEnvG1Relative(**args)


def env_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == ENV_LOG and r.levelno == logging.WARNING]


def rollout(env, actions):
    out = [env.img0.clone()]
    for a in actions:
        obs, reward, done, info = env.step(a)
        out += [obs, reward, done, info["obs_before_reset"], info["applied_action"], env.last_abs_action.clone()]
    return out


def anchor_command(env, i):
    path, start = env.dataset.window_info(i)
    with h5py.File(path, "r") as f:
        return torch.from_numpy(f["commands"][start + env.init_context_len - 1])


# ----------------------------------------------------------------------------- seeding

class _Indices(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


@pytest.mark.parametrize("workers", [0, 2])
def test_seed_pass_matches_dataloader(workers):
    n, batch = 23, 4
    torch.manual_seed(5)
    expected = []
    for _ in range(2):
        loader = DataLoader(_Indices(n), batch_size=batch, shuffle=True, drop_last=True, num_workers=workers)
        expected.append([b.tolist() for b in loader])
    rng_after = torch.rand(3)

    torch.manual_seed(5)
    got = [draw_seed_pass(n, batch) for _ in range(2)]
    assert got == expected
    assert torch.equal(torch.rand(3), rng_after)


def test_seed_windows_follow_the_global_rng(tiny_weights, tiny_pool):
    torch.manual_seed(3)
    env = new_env(tiny_weights, tiny_pool)
    torch.manual_seed(3)
    batches = draw_seed_pass(len(env.dataset), env.num_envs)
    assert env.seed_window_ids.tolist() == batches[0]
    env.reset()
    assert env.seed_window_ids.tolist() == batches[1]
    assert [env.dataset.window_info(i) for i in batches[1]] == env.seed_windows()


def test_seed_pass_rollover(tiny_weights, tiny_pool):
    # A private generator: on CPU the resets' generation noise also draws from the global RNG.
    env = new_env(tiny_weights, tiny_pool, num_envs=8, loader_seed=5)
    n = len(env._eligible)
    n_batches = n // 8
    seen = [env.seed_window_ids.tolist()]
    for _ in range(n_batches):
        env.reset()
        seen.append(env.seed_window_ids.tolist())
    first_pass = sorted(i for b in seen[:n_batches] for i in b)
    assert len(set(first_pass)) == len(first_pass) == n_batches * 8
    # The second pass is a fresh one, drawn from the same generator with the paper's two draws.
    g = torch.Generator().manual_seed(5)
    passes = [draw_seed_pass(n, 8, g) for _ in range(2)]
    expected = [env._eligible[b].tolist() for b in passes[0] + passes[1][:1]]
    assert seen == expected


@pytest.mark.parametrize("loader_seed", [None, 7])
def test_resume_continues_the_seed_pass(tiny_weights, tiny_pool, loader_seed):
    def windows(env, n):
        out = []
        for _ in range(n):
            env.reset()
            out.append(env.seed_window_ids.tolist())
        return out

    def checkpoint(env):
        state, rng = env.state_dict(), torch.get_rng_state()
        return state, rng, windows(env, n_batches + 1)  # the uninterrupted run, through a rollover

    torch.manual_seed(1)
    ref = new_env(tiny_weights, tiny_pool, num_envs=8, loader_seed=loader_seed)
    n_batches = len(ref._eligible) // 8
    windows(ref, 2)
    checkpoints = [checkpoint(ref)]  # mid-pass
    windows(ref, n_batches - ref.state_dict()["batches_drawn"])
    checkpoints.append(checkpoint(ref))  # right after a pass's last batch
    assert checkpoints[1][0]["batches_drawn"] == n_batches

    for state, rng, expected in checkpoints:
        torch.manual_seed(2)  # a resumed run builds its env, then restores its RNG and the env state
        resumed = new_env(tiny_weights, tiny_pool, num_envs=8, loader_seed=loader_seed)
        torch.set_rng_state(rng)
        resumed.load_state_dict(state)
        assert windows(resumed, n_batches + 1) == expected

    with pytest.raises(ValueError, match="batches of 8"):
        new_env(tiny_weights, tiny_pool, num_envs=3, loader_seed=loader_seed).load_state_dict(state)


def test_loader_seed_pins_the_windows(tiny_weights, tiny_pool):
    ids = []
    for global_seed in (1, 2):
        torch.manual_seed(global_seed)
        ids.append(new_env(tiny_weights, tiny_pool, loader_seed=11).seed_window_ids.tolist())
    assert ids[0] == ids[1]
    torch.manual_seed(1)
    assert new_env(tiny_weights, tiny_pool, loader_seed=12).seed_window_ids.tolist() != ids[0]


def test_workers_do_not_change_anything(tiny_weights, tiny_pool):
    actions = torch.randn(7, 3, 6, generator=torch.Generator().manual_seed(0)) * 0.05
    outs = []
    for workers in (0, 2):
        torch.manual_seed(4)
        env = new_env(tiny_weights, tiny_pool, data={"num_workers": workers})
        if workers:  # the trainer, not the loader workers, handles SIGTERM / SIGUSR1
            assert env._loader_iter._worker_init_fn is ignore_stop_signals
        outs.append(rollout(env, actions))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


def test_close_and_del_stop_the_loader_workers(tiny_weights, tiny_pool):
    env = new_env(tiny_weights, tiny_pool, data={"num_workers": 2})
    workers = list(env._loader_iter._workers)
    assert all(w.is_alive() for w in workers)
    env.close()
    assert env._loader_iter is None and not any(w.is_alive() for w in workers)

    env = new_env(tiny_weights, tiny_pool, data={"num_workers": 2})
    iterator = env._loader_iter  # outlives the env
    del env
    gc.collect()
    assert not any(w.is_alive() for w in iterator._workers)


_UNCLOSED_ENV_SCRIPT = """
import sys
from pathlib import Path
from coupled_wm.envs import DreamerV4VecEnvG1Relative

weights, pool = map(Path, sys.argv[1:])
env = DreamerV4VecEnvG1Relative(num_envs=3, data={"dirs": [str(pool)], "num_workers": 2},
                                tokenizer_ckpt=weights / "tokenizer", dynamics_ckpt=weights / "dynamics",
                                device="cpu", episode_length=5, img_size=16, compile_encoder=False,
                                drop_action_indices=[0, 1])
print(" ".join(str(w.pid) for w in env._loader_iter._workers), flush=True)
"""


def test_a_script_that_does_not_close_the_env_exits(tiny_weights, tiny_pool):
    """The env holds a worker iterator; its workers ignore SIGTERM, which multiprocessing uses to
    stop daemonic children at exit."""
    run_env = {**os.environ, "OMP_NUM_THREADS": "1"}
    run_env["PYTHONPATH"] = os.pathsep.join([str(REPO), run_env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, "-c", _UNCLOSED_ENV_SCRIPT, str(tiny_weights), str(tiny_pool)],
                          env=run_env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    pids = [int(pid) for pid in proc.stdout.split()]
    assert len(pids) == 2
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_same_seed_same_rollout(tiny_weights, tiny_pool):
    actions = torch.randn(6, 3, 6, generator=torch.Generator().manual_seed(1)) * 0.05
    outs = []
    for _ in range(2):
        torch.manual_seed(9)
        outs.append(rollout(new_env(tiny_weights, tiny_pool), actions))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


# ----------------------------------------------------------------------------- contract

def test_reset_and_step_contract(tiny_weights, tiny_pool):
    torch.manual_seed(0)
    env = new_env(tiny_weights, tiny_pool, episode_length=3)
    assert (env.num_envs, env.num_actions, env.action_dim, env.img_size) == (3, 6, 6, 16)
    assert (env.init_context_len, env.kv_cache_len, env.num_inference_steps) == (4, 6, 2)
    assert env.img0.shape == (3, 3, 16, 16) and env.t == 4 and env.traj_steps == 0
    first_windows = env.seed_windows()

    for k in range(3):
        obs, reward, done, info = env.step(torch.zeros(3, 6))
        # fp32 outputs on every device (the RSSM and the replay buffer take them outside autocast)
        assert obs.dtype == reward.dtype == env.img0.dtype == info["obs_before_reset"].dtype == torch.float32
        assert obs.shape == (3, 3, 16, 16) and reward.shape == (3,)
        assert done.shape == (3,) and done.dtype == torch.bool
        assert set(info) == {"obs_before_reset", "dones", "applied_action", "clamped"}
        assert info["applied_action"].shape == (3, 6) and info["clamped"].dtype == torch.bool
        assert torch.equal(info["dones"], done)
        if k < 2:
            assert not done.any() and torch.equal(obs, info["obs_before_reset"]) and env.t == 5 + k
    assert done.all()
    assert torch.equal(obs, env.img0) and not torch.equal(obs, info["obs_before_reset"])
    assert env.t == 4 and env.traj_steps == 0 and env.seed_windows() != first_windows


def test_reset_returns_img0_and_rejects_subsets(tiny_weights, tiny_pool):
    env = new_env(tiny_weights, tiny_pool)
    assert env.reset(torch.arange(3)) is env.img0
    with pytest.raises(ValueError, match="resets all"):
        env.reset([0, 2])
    with pytest.raises(ValueError, match="actions must be"):
        env.step(torch.zeros(3, 5))


def test_u8_frames_are_exact(tiny_weights, tiny_pool):
    env = new_env(tiny_weights, tiny_pool)
    x = torch.arange(256, dtype=torch.uint8).view(1, 1, 1, 16, 16).expand(1, 1, 3, 16, 16)
    x = torch.cat([x, 255 - x])  # two envs with different frames
    out = env._frames_to_device(x)
    assert out.dtype == torch.bfloat16 and torch.equal(out, (x.float() / 255.0).to(torch.bfloat16))


def test_reset_prefills_the_seed_window(tiny_weights, tiny_pool):
    torch.manual_seed(0)
    env = new_env(tiny_weights, tiny_pool)
    for b, i in enumerate(env.seed_window_ids.tolist()):
        anchor = anchor_command(env, i).to(torch.bfloat16).float()
        assert torch.equal(env.last_abs_action[b], anchor)
    item = env.dataset[int(env.seed_window_ids[0])]
    path, start = env.dataset.window_info(int(env.seed_window_ids[0]))
    with h5py.File(path, "r") as f:
        raw = f["images"][start:start + 4]
    assert not env.dataset.resize and item["image"].dtype == torch.uint8
    assert torch.equal(item["image"], torch.from_numpy(raw[..., ::-1].copy()).permute(0, 3, 1, 2))


def test_seed_frames_are_resized_to_the_tokenizer(tiny_weights, tmp_path):
    """Frames stored at another size reach the encoder as the world-model trainers load them."""
    pool = tmp_path / "pool"
    write_episode_file(pool / "chunk_0000.h5", 14, TINY_ACTION["scale"], image_hw=(48, 48), seed=4)
    write_episode_file(pool / "chunk_0001.h5", 10, TINY_ACTION["scale"], image_hw=(32, 32), seed=5)
    torch.manual_seed(0)
    env = new_env(tiny_weights, pool, num_envs=4)
    assert env.dataset.resize and env.dataset.native_sizes == [(32, 32), (48, 48)]
    seen = []
    encoder = env.enc
    env.enc = lambda images, proprio=None: seen.append(images) or encoder(images, proprio=proprio)
    for _ in range(3):
        env.reset()
        expected = []
        for i in env.seed_window_ids.tolist():
            path, start = env.dataset.window_info(i)
            with h5py.File(path, "r") as f:
                frames = f["images"][start:start + env.init_context_len]
            expected.append(frames_to_tensor(frames, "bgr", (32, 32)).to(torch.bfloat16))
        assert seen[-1].dtype == torch.bfloat16 and torch.equal(seen[-1], torch.stack(expected))
    obs, *_ = env.step(torch.zeros(4, 6))
    assert obs.shape == (4, 3, 16, 16)


# ----------------------------------------------------------------------------- actions

def test_fp32_integration_and_bf16_dynamics_input(tiny_weights, tiny_pool):
    torch.manual_seed(0)
    env = new_env(tiny_weights, tiny_pool, episode_length=10)
    seen = []
    forward_step = env.dyn.forward_step

    def spy(**kw):
        seen.append(kw["action"])
        return forward_step(**kw)

    env.dyn.forward_step = spy
    delta = torch.zeros(3, 6)
    delta[:, 2] = 1e-4
    expected = env.last_abs_action.clone()
    for _ in range(4):
        expected = torch.clamp(expected + delta, -env.action_scale, env.action_scale)
        seen.clear()
        env.step(delta)
        assert env.last_abs_action.dtype == torch.float32
        assert torch.equal(env.last_abs_action, expected)
        assert all(a.dtype == torch.bfloat16 and torch.equal(a, expected.unsqueeze(1).to(torch.bfloat16))
                   for a in seen)
    assert len(seen) == env.num_inference_steps + 1


def test_clamps_and_diagnostics(tiny_weights, tiny_pool):
    torch.manual_seed(0)
    env = new_env(tiny_weights, tiny_pool, episode_length=10)
    anchor = env.last_abs_action.clone()
    a = torch.zeros(3, 6)
    a[:, 2] = 10.0   # beyond the joint envelope (0.5)
    a[:, 3] = 0.01   # inside
    a[:, 4] = -5.0   # grip: envelope -1, then grip range 0
    a[:, 5] = 0.4
    _, _, _, info = env.step(a)
    cmd = env.last_abs_action
    assert torch.all(cmd[:, 2] == 0.5) and torch.all(cmd[:, 4] == 0.0)
    assert torch.equal(cmd[:, 3], anchor[:, 3] + 0.01)
    assert torch.equal(cmd[:, 5], torch.clamp(anchor[:, 5] + 0.4, 0.0, 1.0))
    assert torch.equal(info["applied_action"], cmd - anchor)
    assert info["clamped"][:, 2].all() and info["clamped"][:, 4].all() and not info["clamped"][:, 3].any()
    assert torch.equal(info["clamped"][:, 5], anchor[:, 5] + 0.4 > 1.0)


@pytest.mark.parametrize("mode", ["hold", "zero"])
def test_dropped_dims_reset_hold_and_zero(tiny_weights, tiny_pool, mode):
    torch.manual_seed(2)
    env = new_env(tiny_weights, tiny_pool, num_envs=4, dropped_dims_reset=mode)
    seen_nonzero = False
    for _ in range(len(env.dataset) // 4):
        for b, i in enumerate(env.seed_window_ids.tolist()):
            anchor = anchor_command(env, i).to(torch.bfloat16).float()
            seen_nonzero |= bool((anchor[list(BASE)] != 0).any())
            if mode == "zero":
                anchor[list(BASE)] = 0.0
            assert torch.equal(env.last_abs_action[b], anchor)
        env.reset()
    assert seen_nonzero


def test_hold_and_zero_draw_the_same_windows(tiny_weights, tiny_pool):
    ids = []
    for mode in ("hold", "zero"):
        torch.manual_seed(6)
        env = new_env(tiny_weights, tiny_pool, dropped_dims_reset=mode)
        drawn = [env.seed_window_ids.tolist()]
        for _ in range(3):
            env.reset()
            drawn.append(env.seed_window_ids.tolist())
        ids.append(drawn)
    assert ids[0] == ids[1]


def test_dropped_dims_reset_skip_nonzero(tiny_weights, tiny_pool, caplog):
    """skip_nonzero filters the enumerated pool: window ids are those of the whole pool, a pass
    makes the same RNG draws as under hold and permutes the eligible windows only."""
    torch.manual_seed(0)
    with caplog.at_level(logging.INFO, logger=ENV_LOG):
        env = new_env(tiny_weights, tiny_pool, num_envs=2, dropped_dims_reset="skip_nonzero")
    torch.manual_seed(0)
    hold = new_env(tiny_weights, tiny_pool, num_envs=2, dropped_dims_reset="hold")
    assert len(env.dataset) == len(hold.dataset) and env._pass_seed == hold._pass_seed
    anchors = torch.stack([anchor_command(env, i) for i in range(len(env.dataset))])
    eligible = (anchors[:, list(BASE)] == 0).all(dim=1).nonzero().flatten().tolist()
    excluded = len(env.dataset) - len(eligible)
    assert 0 < excluded < len(env.dataset) and env._eligible.tolist() == eligible
    messages = [r.getMessage() for r in caplog.records if r.name == ENV_LOG]
    assert any(f"{excluded} of {len(env.dataset)} seed windows" in m and "excluded" in m for m in messages)
    first = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(env._pass_seed))[:2].tolist()
    assert env.seed_window_ids.tolist() == [eligible[i] for i in first]
    drawn = []
    for _ in range(len(eligible) // 2):  # one pass
        drawn += env.seed_window_ids.tolist()
        assert torch.all(env.last_abs_action[:, list(BASE)] == 0)
        env.reset()
    assert set(drawn) <= set(eligible) and len(set(drawn)) == len(drawn) == 2 * (len(eligible) // 2)
    # the seed position of a run does not carry over to a run with another mode (another pool)
    assert env.state_dict()["pool_windows"] == len(eligible)
    with pytest.raises(ValueError, match="seed windows"):
        hold.load_state_dict(env.state_dict())


# ----------------------------------------------------------------------------- validation

@pytest.mark.parametrize("kw, match", [
    (dict(init_context_len=7), "init_context_len"),                 # > kv_cache_len 6
    (dict(num_inference_steps=3), "power of two"),
    (dict(num_inference_steps=32), "power of two"),                 # > num_tau_levels 16
    (dict(seq_cor_tau_idx=16), "seq_cor_tau_idx"),
    (dict(reward_mtp_index=2), "reward_mtp_index"),
    (dict(action_scale=[0.1] * 5), "action_scale"),
    (dict(grip_dims=[6]), "grip_dims"),
    (dict(grip_range=[1.0, 0.0]), "grip_range"),
    (dict(drop_action_indices=[0, 0]), "drop_action_indices"),
    (dict(dropped_dims_reset="keep"), "dropped_dims_reset"),
    (dict(dropped_dims_reset="zero", drop_action_indices=[]), "needs drop_action_indices"),
    (dict(num_envs=100), "fewer than num_envs"),
])
def test_invalid_settings_raise(tiny_weights, tiny_pool, kw, match):
    with pytest.raises(ValueError, match=match):
        new_env(tiny_weights, tiny_pool, **kw)


def test_data_must_match_the_models(tiny_weights, tmp_path):
    write_episode_file(tmp_path / "a" / "chunk_0000.h5", 12, [0.3] * 5, base_dims=(0,), grip_dims=(4,))
    with pytest.raises(ValueError, match="commands have 5 dims"):
        new_env(tiny_weights, tmp_path / "a", drop_action_indices=[0])
    write_episode_file(tmp_path / "c" / "chunk_0000.h5", 12, TINY_ACTION["scale"], proprio_dim=4)
    with pytest.raises(ValueError, match="proprio has 4 dims"):
        new_env(tiny_weights, tmp_path / "c")


def test_checkpoint_without_model_config_raises(tiny_weights, tiny_pool, tmp_path):
    dyn = build_dynamics(TINY_DYN)
    torch.save({"dyn": dyn.state_dict()}, tmp_path / "dyn.pt")
    with pytest.raises(ValueError, match="model_config"):
        new_env(tiny_weights, tiny_pool, dynamics_ckpt=tmp_path / "dyn.pt")
    with pytest.raises(FileNotFoundError):
        new_env(tiny_weights, tiny_pool, dynamics_ckpt=tmp_path / "missing")


def test_training_checkpoint_with_model_config_loads(tiny_weights, tiny_pool, tmp_path):
    torch.manual_seed(0)
    dyn = build_dynamics(TINY_DYN)
    cfg = dynamics_model_config(TINY_DYN, inference=TINY_INFERENCE, action=TINY_ACTION)
    sd = {f"_orig_mod.{k}": v for k, v in dyn.state_dict().items()}
    torch.save({"dyn": sd, "model_config": cfg, "optim": {}}, tmp_path / "latest.pt")
    env = new_env(tiny_weights, tiny_pool, dynamics_ckpt=tmp_path / "latest.pt")
    assert env.num_inference_steps == 2


# ----------------------------------------------------------------------------- config

def test_g1_rl_config_env_block(tiny_weights, tiny_pool):
    cfg = load_config(REPO / "configs/rl/g1_dmo.yaml", [])["env"]
    assert cfg["name"] == "DreamerV4VecEnvG1Relative"
    data = cfg["data"]
    assert (data["exclude_globs"], data["min_chunk_index"]) == (["patch*"], 21)  # the paper pool
    assert data["color_order"] is None  # inherited from the tokenizer (bgr for the G1 weights)
    # The seed pool takes the keys of the world-model / RSSM data blocks.
    rssm_data = load_config(REPO / "configs/rssm/g1_rssm_pretrain.yaml", [])["data"]
    assert {"dirs", "include_globs", "exclude_globs", "min_chunk_index", "file_list"} <= set(data) & set(rssm_data)
    assert cfg["drop_action_indices"] == [0, 1, 2] and cfg["compile_encoder"] == "reduce-overhead"
    mode = cfg["dropped_dims_reset"]
    assert mode == "skip_nonzero"  # never seed from a window with a nonzero base command
    cfg.update(tokenizer_ckpt=str(tiny_weights / "tokenizer"), dynamics_ckpt=str(tiny_weights / "dynamics"),
               data={**data, "dirs": [str(tiny_pool)], "min_chunk_index": None, "num_workers": 0},
               drop_action_indices=list(BASE), compile_encoder=False)
    env = make_env(cfg, num_envs=2, device="cpu", episode_length=4)
    assert env.dropped_dims_reset == mode and env.img_size == 64 and env.dataset.color_order == "bgr"
    with pytest.raises(KeyError, match="unknown env config key"):
        make_env({**cfg, "dataset_path": "x"}, num_envs=2, device="cpu", episode_length=4)
    with pytest.raises(KeyError, match="unknown env data key"):
        make_env({**cfg, "data": {**cfg["data"], "window": 32}}, num_envs=2, device="cpu", episode_length=4)
    with pytest.raises(KeyError, match="unknown env name"):
        make_env({**cfg, "name": "Other"}, num_envs=2, device="cpu", episode_length=4)


def test_defaults_come_from_the_dynamics_config(tiny_weights, tiny_pool, caplog):
    with caplog.at_level(logging.INFO, logger=ENV_LOG):
        env = new_env(tiny_weights, tiny_pool, kv_cache_len=5)
    assert env.action_scale.tolist() == pytest.approx(TINY_ACTION["scale"])
    assert env.grip_dims == (4, 5) and env.grip_range == (0.0, 1.0)
    assert (env.seq_cor_tau_idx, env.reward_mtp_index, env.d_min_idx) == (2, 1, int(math.log2(16)))
    assert G1_INFERENCE["kv_cache_len"] == 32 and len(G1_ACTION["scale"]) == 22
    sources = {k: "dynamics config.json" for k in ("init_context_len", "num_inference_steps", "seq_cor_tau_idx",
                                                   "reward_mtp_index", "action_scale", "grip_dims", "grip_range")}
    assert env.constant_sources == {**sources, "kv_cache_len": "env config", "color_order": "tokenizer config.json"}
    assert not env_warnings(caplog)
    assert any("grip_dims=[4, 5] (dynamics config.json)" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("action, from_g1", [
    (None, ["action_scale", "grip_dims", "grip_range"]),     # no action block (export.preset: null)
    ({"scale": [0.1] * 22}, ["grip_dims", "grip_range"]),    # a partial block
])
def test_g1_defaults_are_reported(tmp_path, action, from_g1, caplog):
    """A 22-dim model that does not record its action layout gets the G1 one, with a warning."""
    weights = save_weights(tmp_path / "w", preprocessing={"h5_color_order": "bgr"},
                           dyn_arch=dict(TINY_DYN, action_dim=22), inference=TINY_INFERENCE,
                           **({} if action is None else {"action": action}))
    pool = tmp_path / "pool"
    write_episode_file(pool / "chunk_0000.h5", 16, G1_ACTION["scale"], base_dims=(0, 1, 2), grip_dims=(20, 21))
    with caplog.at_level(logging.WARNING, logger=ENV_LOG):
        env = new_env(weights, pool, drop_action_indices=[0, 1, 2])
    assert [k for k, s in env.constant_sources.items() if s == "G1 default"] == from_g1
    warnings = env_warnings(caplog)
    (warning,) = [w for w in warnings if "G1 values" in w]
    assert all(f"{k}=" in warning for k in from_g1)
    # The commands lie within 0.9x the G1 envelope: only the 0.1 envelope leaves seed anchors outside the clamp.
    assert any("outside the clamp" in w for w in warnings) == (action is not None)
    assert env.grip_dims == (20, 21) and env.grip_range == (0.0, 1.0)
    assert env.action_scale.tolist() == pytest.approx(G1_ACTION["scale"] if action is None else action["scale"])


def test_color_order_is_inherited_from_the_tokenizer(tmp_path, tiny_pool, caplog):
    weights = save_weights(tmp_path / "rgb", preprocessing={"h5_color_order": "rgb"}, inference=TINY_INFERENCE,
                           action=TINY_ACTION)
    env = new_env(weights, tiny_pool)
    assert env.color_order == env.dataset.color_order == "rgb"
    assert env.constant_sources["color_order"] == "tokenizer config.json"
    path, start = env.dataset.window_info(0)
    with h5py.File(path, "r") as f:
        raw = f["images"][start:start + 4]
    assert torch.equal(env.dataset[0]["image"], torch.from_numpy(raw).permute(0, 3, 1, 2))  # stored order kept

    with caplog.at_level(logging.WARNING, logger=ENV_LOG):
        env = new_env(weights, tiny_pool, data={"color_order": "bgr"})  # an explicit order still wins
    assert env.dataset.color_order == "bgr"
    assert any("h5_color_order='rgb'" in w for w in env_warnings(caplog))

    caplog.clear()
    bare = save_weights(tmp_path / "bare", inference=TINY_INFERENCE, action=TINY_ACTION)  # no preprocessing block
    with caplog.at_level(logging.WARNING, logger=ENV_LOG):
        env = new_env(bare, tiny_pool)
    assert env.color_order == "bgr" and env.constant_sources["color_order"] == "G1 default"
    assert any("color_order=bgr" in w for w in env_warnings(caplog))


def test_noise_grid_constants(tmp_path, tiny_pool, caplog):
    """d_min_idx is log2(num_tau_levels); the G1 seq_cor_tau_idx is refused on another grid."""
    weights = save_weights(tmp_path / "w", preprocessing={"h5_color_order": "bgr"}, action=TINY_ACTION)
    with pytest.raises(ValueError, match="seq_cor_tau_idx indexes the noise grid"):
        new_env(weights, tiny_pool, init_context_len=4, kv_cache_len=6)
    with caplog.at_level(logging.WARNING, logger=ENV_LOG):
        env = new_env(weights, tiny_pool, init_context_len=4, kv_cache_len=6, seq_cor_tau_idx=2)
    assert env.d_min_idx == 4 and env.num_inference_steps == G1_INFERENCE["num_inference_steps"]
    assert "num_inference_steps, reward_mtp_index" in env_warnings(caplog)[0]

    bad = save_weights(tmp_path / "bad", preprocessing={"h5_color_order": "bgr"},
                       inference=dict(TINY_INFERENCE, d_min_idx=7), action=TINY_ACTION)
    with pytest.raises(ValueError, match="d_min_idx=7"):
        new_env(bad, tiny_pool)


ONE_STAGE = {"init_from": None, "clips": {"short_len": 3, "long_len": None, "long_every": 0}}
FROM_BACKBONE = {"init_from": {"path": "backbone/latest.pt", "global_update": 10},
                 "clips": {"short_len": 3, "long_len": None, "long_every": 0}}


@pytest.mark.parametrize("training, context_len, warns", [
    (ONE_STAGE, 96, True),                                     # 3-frame clips, 7-frame attention
    ({**ONE_STAGE, "clips": {"short_len": 3, "long_len": 7, "long_every": 8}}, 96, False),
    (FROM_BACKBONE, 96, False),                                # lineage unknown: not checked
    ({**FROM_BACKBONE, "lineage_max_clip_len": 6}, 96, True),
    ({**FROM_BACKBONE, "lineage_max_clip_len": 7}, 96, False),
    ({**ONE_STAGE, "clips": {"short_len": 5, "long_len": None, "long_every": 0}}, 5, False),  # sliding window
])
def test_attention_beyond_the_trained_clips_warns(tmp_path, tiny_pool, caplog, training, context_len, warns):
    weights = save_weights(tmp_path / "w", preprocessing={"h5_color_order": "bgr"},
                           dyn_arch=dict(TINY_DYN, context_len=context_len), inference=TINY_INFERENCE,
                           action=TINY_ACTION, training=training)
    with caplog.at_level(logging.WARNING, logger=ENV_LOG):
        new_env(weights, tiny_pool)  # init_context_len 4, kv_cache_len 6: a step attends to 7 frames
    assert any("never trained on" in w for w in env_warnings(caplog)) == warns


def test_kv_caches_are_checked_before_allocation(tiny_weights, tiny_pool, monkeypatch):
    env = new_env(tiny_weights, tiny_pool)
    caches = (*env.dyn.caches, *env.dec.caches_l, *env.dec.caches_r)
    assert env.kv_cache_bytes == sum(t.numel() * t.element_size() for c in caches if c is not None
                                     for t in (c.k_cache, c.v_cache))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (10**9, 8 * 10**10))
    cuda = torch.device("cuda", 0)
    _check_kv_cache_fits(cuda, 10**9, 16, 32)
    with pytest.raises(torch.OutOfMemoryError, match="16 envs .*num_actors / world_size"):
        _check_kv_cache_fits(cuda, 10**9 + 1, 16, 32)
    _check_kv_cache_fits(torch.device("cpu"), 10**15, 16, 32)


def test_out_of_memory_in_the_first_reset_names_the_encoder_batch(tiny_weights, tiny_pool, monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda module, *args, **kwargs: module)

    def oom(self):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(DreamerV4VecEnvG1Relative, "reset_all", oom)
    with pytest.raises(torch.OutOfMemoryError, match=r"encodes 3 x 4 frames.*compile_encoder: false"):
        new_env(tiny_weights, tiny_pool, compile_encoder="reduce-overhead")


def test_action_envelope_snippets_set_the_whole_layout(tiny_weights, tiny_pool, tmp_path):
    """Either snippet of scripts/compute_action_envelope.py replaces the G1 grip fallback."""
    path = REPO / "scripts/compute_action_envelope.py"
    spec = importlib.util.spec_from_file_location("compute_action_envelope", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    snippet = yaml.safe_load(script.main([str(tiny_pool), "--grip-dims", *map(str, GRIPS), "--rounding", "up"]))

    def dynamics_dir(name, **metadata):
        torch.manual_seed(0)
        return save_dynamics_pretrained(tmp_path / name, build_dynamics(TINY_DYN).state_dict(),
                                        dynamics_model_config(TINY_DYN, inference=TINY_INFERENCE, **metadata),
                                        tokenizer_dir=tiny_weights / "tokenizer")

    bare = dynamics_dir("no_action_block")  # e.g. exported with export.preset: null
    with pytest.raises(ValueError, match=r"grip_dims .*got \[20, 21\]"):  # the G1 fallback
        new_env(tiny_weights, tiny_pool, dynamics_ckpt=bare, action_scale=snippet["env"]["action_scale"])
    envs = [new_env(tiny_weights, tiny_pool, dynamics_ckpt=bare, **snippet["env"]),
            new_env(tiny_weights, tiny_pool, dynamics_ckpt=dynamics_dir("exported", **snippet["export"]["metadata"]))]
    for env in envs:
        assert env.grip_dims == GRIPS and env.grip_range == (0.0, 1.0)
        assert env.action_scale.tolist() == pytest.approx(snippet["env"]["action_scale"])
    no_grips = yaml.safe_load(script.main([str(tiny_pool), "--grip-dims"]))
    assert new_env(tiny_weights, tiny_pool, dynamics_ckpt=bare, **no_grips["env"]).grip_dims == ()


@pytest.mark.parametrize("encoder_context_len, warned", [(2, True), (0, False), (8, False), (None, False)])
def test_encoder_window_of_the_dynamics_training(tmp_path, tiny_pool, caplog, encoder_context_len, warned):
    """The tiny tokenizer has no window; a dynamics trained on 2-frame-window latents sees other context
    latents in the env (8 frames covers the whole 4-frame context: the same latents)."""
    tok_block = {} if encoder_context_len is None else {"tokenizer": {"encoder_context_len": encoder_context_len}}
    weights = save_weights(tmp_path / "w", preprocessing={"h5_color_order": "bgr"}, inference=TINY_INFERENCE,
                           action=TINY_ACTION, **tok_block)
    with caplog.at_level(logging.WARNING, logger=ENV_LOG):
        new_env(weights, tiny_pool)
    assert any("temporal window of" in w for w in env_warnings(caplog)) == warned
