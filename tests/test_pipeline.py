# SPDX-License-Identifier: Apache-2.0
"""The components wired together on tiny models (CPU): tokenizer -> dynamics (both stages) ->
G1 environment; RSSM pretraining -> RL warm start -> DMO training on that environment; and the
observation and action conventions the environment, the RL replay buffer and RSSM pretraining share.

The data mimics the G1 layout at a smaller size: 32x32 BGR frames (the world model's size),
16x16 observations (the RSSM's size), 6-d commands with base dims 0-1 and grips 4-5.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from coupled_wm.common.checkpoint import load_checkpoint, load_pretrained_config
from coupled_wm.common.distributed import DistInfo
from coupled_wm.data.h5_dataset import HDF5SequenceDataset, frames_to_tensor, list_h5_files
from coupled_wm.envs import g1_env, make_env
from coupled_wm.rl import evaluate as rl_evaluate
from coupled_wm.rl.dmo import DMOTrainer, RunOptions, SequenceReplay, rssm_training_batch
from coupled_wm.rssm import pretrain as rssm_pretrain
from coupled_wm.rssm.rssm import RELATIVE_ACTION_DEFINITION, rssm_config
from coupled_wm.world_model import train_dynamics as TD
from coupled_wm.world_model import train_tokenizer as TT
from rl_fake_env import single_thread, tiny_config  # noqa: F401 (fixture)

CPU = DistInfo(0, 0, 1, torch.device("cpu"), False)
SCALE = [0.3, 0.4, 0.8, 0.8, 1.0, 1.0]
TOKENIZER = dict(image_size=[32, 32], patch_size=8, in_channels=3, d_model=32, n_layers=4, num_heads_q=2,
                 num_heads_kv_latent=2, temporal_every=2, n_latents=4, bottleneck_dim=4, use_proprio=True,
                 proprio_dim=3, num_proprio=1, seq_len=8, context_len=0, dropout=0.0, mae_max_mask_prob=0.5)
DYNAMICS = dict(action_dim=6, num_latents=4, latent_dim=4, d_model=32, num_layers=4, num_heads=2, num_registers=2,
                num_tau_levels=16, temporal_every=2, mlp_ratio=2.0, seq_len=16, context_len=6, dropout=0.0,
                mtp_length=2, reward_hidden_dim=16, reward_num_buckets=31, reward_symlog_range=[-20.0, 20.0])
INFERENCE = dict(num_inference_steps=2, d_min_idx=4, seq_cor_tau_idx=2, init_context_len=4, kv_cache_len=6,
                 reward_mtp_index=0)
ACTION = dict(dim=6, base_velocity_dims=[0, 1], grip_dims=[4, 5], grip_range=[0.0, 1.0], scale=SCALE)
RSSM = dict(action_size=6, image_size=16, stochastic_size=4, discrete_size=4, recurrent_state_size=16,
            dense_units=16, channels_multiplier=2, mlp_layers=2, reward_model_bins=31)


def write_episodes(path: Path, n_frames: int, dones_at: list[int], seed: int) -> None:
    """G1-like file: joints random-walk inside the envelope, grips in {0, 1}, base dims 0."""
    rng = np.random.default_rng(seed)
    scale = np.asarray(SCALE, dtype=np.float32)
    commands = np.clip(np.cumsum(rng.normal(0.0, 0.05, (n_frames, 6)), axis=0), -0.9 * scale, 0.9 * scale)
    commands[:, [4, 5]] = rng.random((n_frames, 2)) > 0.5
    commands[:, [0, 1]] = 0.0
    dones = np.zeros(n_frames, dtype=np.float32)
    dones[dones_at] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=rng.integers(0, 256, (n_frames, 32, 32, 3), dtype=np.uint8))
        f.create_dataset("commands", data=commands.astype(np.float32))
        f.create_dataset("dones", data=dones)
        f.create_dataset("proprio", data=rng.normal(size=(n_frames, 3)).astype(np.float32))
        f.create_dataset("rewards", data=(rng.random(n_frames) > 0.7).astype(np.float32))


def env_block(tokenizer, dynamics, data_dir, **over):
    return {"name": "DreamerV4VecEnvG1Relative", "tokenizer_ckpt": str(tokenizer), "dynamics_ckpt": str(dynamics),
            "verify_pairing": True, "data": {"dirs": [str(data_dir)], "color_order": "bgr", "num_workers": 0},
            "img_size": 16, "drop_action_indices": [0, 1], "dropped_dims_reset": "hold", "compile_encoder": False,
            **over}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Train every component on the same synthetic data; returns the run dirs."""
    root = tmp_path_factory.mktemp("pipeline")
    data = root / "data"
    for i in range(3):
        write_episodes(data / f"chunk_{i:04d}.h5", 40, dones_at=[19, 39], seed=i)

    tok_cfg = {
        "run_dir": str(root / "tokenizer"), "seed": 1, "compile": False,
        "data": {"dirs": [str(data)], "window": 4, "num_workers": 0},
        "model": TOKENIZER,
        "loss": {"lpips_weight": 0.0},
        "schedule": {"warmup_updates": 1, "total_updates": 20, "max_updates": 2},
        "train": {"global_batch": 4, "micro_batch_per_gpu": 2},
        "checkpoint": {"save_every_updates": 100, "archive_every_updates": 0},
        "logging": {"print_every_updates": 0},
    }
    (root / "tokenizer.yaml").write_text(json.dumps(tok_cfg))
    assert TT.main(["--config", str(root / "tokenizer.yaml")])["status"] == "done"

    common = {
        "seed": 2, "tokenizer": {"path": str(root / "tokenizer" / "pretrained")},
        "model": DYNAMICS, "optim": {"lr": 1e-3},
        "checkpoint": {"save_every_updates": 100}, "logging": {"print_every_updates": 0},
        "export": {"metadata": {"inference": INFERENCE, "action": ACTION, "control_hz": 10}},
    }
    TD.train({**common, "stage": "backbone", "run_dir": str(root / "backbone"),
              "data": {"dirs": [str(data)], "num_workers": 0},
              "train": {"short_len": 4, "long_len": 8, "long_every": 2, "global_batch": 2},
              "schedule": {"warmup_updates": 1, "total_updates": 20, "max_updates": 2}}, CPU)
    TD.train({**common, "stage": "reward", "run_dir": str(root / "reward"),
              "init_from": str(root / "backbone" / "latest.pt"),
              "data": {"dirs": [str(data)], "num_workers": 0, "reward_key": "rewards"},
              "train": {"short_len": 4, "global_batch": 2},
              "schedule": {"warmup_updates": 1, "total_updates": 20, "max_updates": 2}}, CPU)

    rssm_pretrain.train({
        "run_dir": str(root / "rssm"), "seed": 3,
        "data": {"dirs": [str(data)], "num_workers": 0, "cache": True, "cache_workers": 0},
        "model": RSSM,
        "train": {"epochs": 1, "batch_size": 8, "sequence_length": 4, "precision": "fp32", "compile": False},
    }, CPU)
    return {"root": root, "data": data, "tokenizer": root / "tokenizer", "reward": root / "reward",
            "rssm": root / "rssm" / "rssm_e1"}


def test_world_model_outputs_drive_the_environment(pipeline):
    tok_dir, dyn_dir = pipeline["tokenizer"] / "pretrained", pipeline["reward"] / "pretrained"
    dcfg = load_pretrained_config(dyn_dir)
    assert dcfg["inference"] == INFERENCE and dcfg["action"] == ACTION and dcfg["tokenizer"]["sha256"]
    # the reward stage trained on 4-frame clips from a backbone that also saw 8-frame clips
    assert dcfg["training"]["lineage_max_clip_len"] == 8 and dcfg["tokenizer"]["encoder_context_len"] == 0
    sources = [(tok_dir, dyn_dir), (pipeline["tokenizer"] / "latest.pt", pipeline["reward"] / "latest.pt")]
    for tok, dyn in sources:
        torch.manual_seed(0)
        env = make_env(env_block(tok, dyn, pipeline["data"]), num_envs=2, device="cpu", episode_length=3)
        assert (env.init_context_len, env.kv_cache_len, env.num_inference_steps) == (4, 6, 2)
        assert env.action_scale.tolist() == pytest.approx(SCALE)
        for t in range(3):
            obs, reward, done, info = env.step(torch.zeros(2, 6))
            assert obs.shape == (2, 3, 16, 16) and torch.isfinite(obs.float()).all()
            assert torch.isfinite(reward.float()).all() and bool(done.all()) == (t == 2)
        env.close()


def test_rssm_pretrain_output_is_the_rl_warm_start(pipeline, tmp_path):
    stored = load_pretrained_config(pipeline["rssm"])
    assert stored["action"]["mode"] == "relative"
    assert stored["action"]["definition"] == RELATIVE_ACTION_DEFINITION
    assert stored["preprocessing"]["image_size"] == [16, 16] and stored["preprocessing"]["h5_color_order"] == "bgr"

    cfg = tiny_config(**{"train.max_epochs": 2, "train.num_actors": 2, "train.max_episode_length": 8,
                         "train.min_replay": 32, "train.replay_size": 400, "eval.num_episodes": 2,
                         "eval.every": 2, "eval.player.num_actors": 2, "eval.player.games_num": 2,
                         "actor.action_scale": [0.2, 0.2, 1.0, 1.0], "rssm.pretrained": str(pipeline["rssm"])})
    cfg["env"] = env_block(pipeline["tokenizer"] / "pretrained", pipeline["reward"] / "pretrained", pipeline["data"])
    cfg["rssm"].update({k: v for k, v in RSSM.items() if k not in ("action_size", "image_size")})
    tr = DMOTrainer(cfg, RunOptions(seed=0, logdir=str(tmp_path / "run")), CPU)
    assert tr.rssm_warmstart_info["action"]["definition"] == RELATIVE_ACTION_DEFINITION
    assert rssm_config(stored) == tr.rssm_arch
    tr.train()
    assert tr.iter_count == 2 and math.isfinite(tr.actor_loss) and math.isfinite(tr.dyn_loss)

    summary = rl_evaluate.main(["--checkpoint", str(tmp_path / "run" / "final_policy.pt"), "--num-envs", "2",
                                "--device", "cpu", "--output-dir", str(tmp_path / "eval"), "--set",
                                "+env.loader_seed=0"])
    assert summary["episodes"] == 2 and math.isfinite(summary["mean_return"])


def test_observations_match_the_rssm_preprocessing(pipeline):
    """With an identity decoder the env returns its seed frame: same RGB order, [0, 1] range and
    resize as the frames RSSM pretraining reads from the h5 files."""
    torch.manual_seed(0)
    env = make_env(env_block(pipeline["tokenizer"] / "pretrained", pipeline["reward"] / "pretrained",
                             pipeline["data"]), num_envs=2, device="cpu", episode_length=4)
    captured = {}
    to_device = env._frames_to_device
    env._frames_to_device = lambda frames: captured.setdefault("x", to_device(frames))
    env.dec.forward_step = lambda z, **kw: (None, captured["x"])
    obs = env.reset_all().float()
    for b, (path, start) in enumerate(env.seed_windows()):
        with h5py.File(path, "r") as f:
            raw = f["images"][start + env.init_context_len - 1: start + env.init_context_len]
        expected = frames_to_tensor(raw, "bgr", (16, 16))[0]
        assert (obs[b] - expected).abs().max() < 1e-2
        assert (obs[b] - expected.flip(0)).abs().max() > 0.1  # the channel order is not symmetric
    window = HDF5SequenceDataset(list_h5_files(pipeline["data"]), window=env.init_context_len, image_size=16,
                                 load_proprio=False, reward_key=None)[env.seed_window_ids[0].item()]["image"][-1]
    assert (obs[0] - window).abs().max() < 1e-2
    env.close()


def test_action_convention_env_rl_rows_and_pretraining_rows(pipeline):
    """Frame t of the env is generated from c_t = c_{t-1} + a_{t-1}. The RL replay buffer presents it to
    the RSSM with a_{t-1}; RSSM pretraining on a recording of the same episode (commands[t] = c_t)
    presents it with commands[t] - commands[t-1]. Both are the same action."""
    torch.manual_seed(0)
    env = make_env(env_block(pipeline["tokenizer"] / "pretrained", pipeline["reward"] / "pretrained",
                             pipeline["data"]), num_envs=2, device="cpu", episode_length=8)
    commands = [env.last_abs_action.clone()]               # c_0: the seed window's last command
    replay = SequenceReplay(6 * 2, (3, 16, 16), 6, seq_len=6, num_envs=2)
    obs, is_first = env.img0.float(), torch.ones(2, dtype=torch.bool)
    gen = torch.Generator().manual_seed(1)
    actions = []
    for _ in range(6):
        a = torch.zeros(2, 6)
        a[:, 2:4] = (torch.rand(2, 2, generator=gen) - 0.5) * 0.02  # small joint moves: no clamp
        next_obs, reward, done, info = env.step(a)
        assert not info["clamped"].any() and not done.any()
        replay.add(obs, a, reward, is_first)
        actions.append(a)
        commands.append(env.last_abs_action.clone())
        obs, is_first = next_obs.float(), done
    rl_rows = rssm_training_batch(replay.get_windows(torch.tensor([0, 1])))

    commands = torch.stack(commands)                       # (7, B, A): c_0 .. c_6
    pretrain_actions = commands[1:] - commands[:-1]        # the pretraining rows of frames 1..6
    for t in range(1, 6):                                  # RL row t holds frame t and a_{t-1}
        assert torch.equal(rl_rows.actions[t], actions[t - 1])
        torch.testing.assert_close(pretrain_actions[t - 1], rl_rows.actions[t], atol=1e-6, rtol=0)
    assert torch.equal(rl_rows.actions[0], torch.zeros(2, 6)) and bool((rl_rows.is_firsts[0] == 1).all())
    env.close()


def test_resumed_rl_run_continues_the_env_seed_pass(pipeline, tmp_path):
    """latest.pt carries the G1 environment's position in its seed-window pass."""
    cfg = tiny_config(**{"train.max_epochs": 2, "train.num_actors": 2, "train.max_episode_length": 8,
                         "train.min_replay": 32, "train.replay_size": 400, "train.save_interval": 2, "eval.every": 100,
                         "actor.action_scale": [0.2, 0.2, 1.0, 1.0], "rssm.pretrained": str(pipeline["rssm"])})
    cfg["env"] = env_block(pipeline["tokenizer"] / "pretrained", pipeline["reward"] / "pretrained", pipeline["data"])
    cfg["rssm"].update({k: v for k, v in RSSM.items() if k not in ("action_size", "image_size")})
    run = tmp_path / "run"
    torch.manual_seed(0)
    DMOTrainer(cfg, RunOptions(seed=0, logdir=str(run), no_eval=True), CPU).train()
    saved = load_checkpoint(run / "latest.pt")["env"][0]

    resumed = DMOTrainer(cfg, RunOptions(seed=0, logdir=str(run), no_eval=True, resume=str(run / "latest.pt")), CPU)
    env = resumed.env
    batches = g1_env._pass_batches(len(env._eligible), env.num_envs, int(saved["pass_seed"]))
    assert int(saved["batches_drawn"]) < len(batches)  # the saved pass has batches left
    env.reset()
    assert env.seed_window_ids.tolist() == [int(env._eligible[i]) for i in batches[int(saved["batches_drawn"])]]
    resumed.close()
