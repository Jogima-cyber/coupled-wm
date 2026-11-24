# SPDX-License-Identifier: Apache-2.0
"""scripts/compare_rollouts.py and the environment's API for replaying recorded commands
(reset_from_context, step_command), on tiny random world models and synthetic data (CPU)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from coupled_wm.envs import DreamerV4VecEnvG1Relative
from coupled_wm.world_model.export import save_dynamics_pretrained, save_tokenizer_pretrained
from coupled_wm.world_model.loading import dynamics_model_config, tokenizer_model_config
from coupled_wm.world_model.models import build_dynamics, build_tokenizer

REPO = Path(__file__).resolve().parents[1]
TOK = dict(image_size=[16, 16], patch_size=8, in_channels=3, d_model=32, n_layers=4, num_heads_q=2,
           num_heads_kv_latent=2, temporal_every=2, n_latents=4, bottleneck_dim=4, use_proprio=True, proprio_dim=3,
           num_proprio=1, seq_len=32, context_len=0, dropout=0.0, mae_max_mask_prob=0.0)
DYN = dict(action_dim=4, num_latents=4, latent_dim=4, d_model=32, num_layers=4, num_heads=2, num_registers=2,
           num_tau_levels=16, temporal_every=2, mlp_ratio=2.0, seq_len=32, context_len=0, dropout=0.0,
           use_reward_head=True, mtp_length=2, reward_hidden_dim=16, reward_num_buckets=31,
           reward_symlog_range=[-20.0, 20.0])
INFERENCE = dict(num_inference_steps=2, d_min_idx=4, seq_cor_tau_idx=2, init_context_len=4, kv_cache_len=6,
                 reward_mtp_index=0)
# A clamp wider than the synthetic commands (1000 * file_id + t + d / 100): no command is clamped.
ACTION = dict(dim=4, grip_dims=[], grip_range=[0.0, 1.0], scale=[2000.0] * 4)
HORIZON = 6


def _script():
    spec = importlib.util.spec_from_file_location("compare_rollouts", REPO / "scripts" / "compare_rollouts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _script()


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    """Tokenizer, dynamics ``a`` (4 context frames), ``b`` (other weights, 3 context frames)."""
    root = tmp_path_factory.mktemp("weights")
    torch.manual_seed(0)
    enc, dec = build_tokenizer(TOK)
    tok = save_tokenizer_pretrained(root / "tokenizer", enc.state_dict(), dec.state_dict(),
                                    tokenizer_model_config(TOK, preprocessing={"h5_color_order": "bgr"}))
    for name, seed, context in (("a", 1, 4), ("b", 2, 3)):
        torch.manual_seed(seed)
        config = dynamics_model_config(DYN, inference={**INFERENCE, "init_context_len": context}, action=ACTION,
                                       control_hz=5)
        save_dynamics_pretrained(root / name / "dynamics", build_dynamics(DYN).state_dict(), config,
                                 tokenizer_dir=tok)
    return root


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    from conftest import write_h5

    d = tmp_path_factory.mktemp("pool")
    write_h5(d / "chunk_0000.h5", 40, dones_at=[19, 39], image_hw=(16, 16), file_id=0)   # two segments of 20
    write_h5(d / "chunk_0001.h5", 30, dones_at=[], image_hw=(16, 16), file_id=1)         # one segment of 30
    write_h5(d / "patch_0000.h5", 30, image_hw=(16, 16), file_id=2)
    return d


def run(script, weights, pool, out, *extra):
    argv = [str(pool), "--tokenizer", str(weights / "tokenizer"), "--dynamics", str(weights / "a" / "dynamics"),
            "--exclude", "patch*", "--horizon", str(HORIZON), "--device", "cpu", "--out", str(out), *extra]
    return script.main(argv)


def recorded(path, key, start, stop):
    with h5py.File(path, "r") as f:
        return np.asarray(f[key][start:stop])


# ------------------------------------------------------- the environment's API for replaying recorded commands

def new_env(weights, pool, **kw):
    return DreamerV4VecEnvG1Relative(
        num_envs=2, data={"dirs": [str(pool)], "num_workers": 0, "exclude_globs": ["patch*"]},
        tokenizer_ckpt=weights / "tokenizer", dynamics_ckpt=weights / "a" / "dynamics", device="cpu",
        episode_length=3, img_size=16, compile_encoder=False, loader_seed=0, **kw)


def test_replay_api_is_the_env_computation(weights, pool):
    """reset_from_context + step_command are reset_all + step without the pool draw and the integration."""
    env, replay = new_env(weights, pool), new_env(weights, pool)
    torch.manual_seed(5)
    img0 = env.reset_all().clone()
    ids = env.seed_window_ids.tolist()
    steps = [env.step(torch.zeros(2, 4)) for _ in range(3)]

    items = [replay.dataset[i] for i in ids]
    batch = {k: torch.stack([it[k] for it in items]) for k in ("image", "action", "proprio")}
    torch.manual_seed(5)
    assert torch.equal(replay.reset_from_context(batch["image"], batch["action"], batch["proprio"]), img0)
    with pytest.raises(RuntimeError, match="reset_from_context"):
        replay.seed_windows()
    for k, (obs, reward, done, info) in enumerate(steps):
        obs_r, reward_r = replay.step_command(replay.last_abs_action)
        assert torch.equal(obs_r, info["obs_before_reset"]) and torch.equal(reward_r, reward)
    assert done.all()
    with pytest.raises(RuntimeError, match="reset the envs first"):
        replay.step_command(replay.last_abs_action)
    with pytest.raises(ValueError, match="images must be"):
        replay.reset_from_context(batch["image"][:, 1:], batch["action"], batch["proprio"])
    with pytest.raises(ValueError, match="proprio"):
        replay.reset_from_context(batch["image"], batch["action"])
    replay.reset_from_context(batch["image"], batch["action"], batch["proprio"])
    with pytest.raises(ValueError, match="command must be"):
        replay.step_command(torch.zeros(2, 3))


# ----------------------------------------------------------------------------- the script

def test_replays_the_recorded_commands(script, weights, pool, tmp_path, monkeypatch):
    """Each model is primed with the frames just before the generated ones and generates frame t
    from commands[t]; outputs and labels line up with the file."""
    calls = []
    reset, step = DreamerV4VecEnvG1Relative.reset_from_context, DreamerV4VecEnvG1Relative.step_command

    def spy_reset(self, images, commands, proprio=None):
        if self.img0 is not None:  # not the constructor's reset from the env's own seed pool
            calls.append(("reset", self.init_context_len, images.clone(), commands.clone(), proprio.clone()))
        return reset(self, images, commands, proprio)

    def spy_step(self, command):
        calls.append(("step", self.init_context_len, command.clone()))
        return step(self, command)

    monkeypatch.setattr(DreamerV4VecEnvG1Relative, "reset_from_context", spy_reset)
    monkeypatch.setattr(DreamerV4VecEnvG1Relative, "step_command", spy_step)
    index = run(script, weights, pool, tmp_path / "out", "--num", "3", "--batch-size", "2", "--seed", "1",
                "--dynamics-b", str(weights / "b" / "dynamics"), "--format", "gif", "--context-frames", "2",
                "--sheet-every", "4")
    windows = index["windows"]
    assert index["settings"]["window_context"] == 4 and index["settings"]["fps"] == 5.0
    assert {k: m["init_context_len"] for k, m in index["models"].items()} == {"a": 4, "b": 3}
    assert all("patch" not in w["file"] for w in windows) and len(windows) == 3

    # model a: batches [w0, w1], [w2, w2 (padding)]; then model b
    resets = [c for c in calls if c[0] == "reset"]
    steps = [c for c in calls if c[0] == "step"]
    assert [c[1] for c in resets] == [4, 4, 3, 3] and len(steps) == 4 * HORIZON
    for r, (_, context, images, commands, proprio) in enumerate(resets):
        batch = [windows[i] for i in ([0, 1] if r % 2 == 0 else [2, 2])]
        for b, w in enumerate(batch):
            lo, hi = w["start"] + 4 - context, w["start"] + 4
            assert (lo, hi) == tuple(w["context_frames"]["a" if r < 2 else "b"])
            rgb = recorded(w["file"], "images", lo, hi)[..., ::-1]
            assert torch.equal(images[b], torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2))
            assert torch.equal(commands[b], torch.from_numpy(recorded(w["file"], "commands", lo, hi)))
            assert torch.equal(proprio[b], torch.from_numpy(recorded(w["file"], "proprio", lo, hi)))
            for k in range(HORIZON):
                command = steps[r * HORIZON + k][2][b]
                t = w["start"] + 4 + k
                assert torch.equal(command, torch.from_numpy(recorded(w["file"], "commands", t, t + 1)[0]))

    out = tmp_path / "out"
    for w in windows:
        record = json.loads((out / (w["contact_sheet"][:-len("_sheet.png")] + ".json")).read_text())
        assert record == w
        frames = list(range(w["start"] + 4, w["start"] + 4 + HORIZON))
        assert w["frame_index"] == frames and w["first_generated_frame"] == frames[0]
        labels = recorded(w["file"], "rewards", frames[0], frames[-1] + 1)
        assert w["reward_label"] == labels.tolist() and w["return"]["label"] == pytest.approx(labels.sum())
        assert {k: len(v) for k, v in w["reward_pred"].items()} == {"a": HORIZON, "b": HORIZON}
        assert all(np.isfinite(v).all() for v in w["reward_pred"].values())
        with Image.open(out / w["video"]) as gif:
            assert gif.n_frames == 2 + HORIZON  # context frames, then the generated ones
            assert gif.size[0] % 16 == 0 and gif.size[1] % 16 == 0 and gif.size[0] >= 3 * 16
        with Image.open(out / w["contact_sheet"]) as sheet:  # GT, a, b rows; steps 1, 4, 6
            assert sheet.size[0] > 3 * 16 and sheet.size[1] > 3 * 16
    assert json.loads((out / "index.json").read_text())["windows"] == windows


def test_seeds_and_identical_noise(script, weights, pool, tmp_path):
    """Same seed: same windows and rewards. The same model as column b reproduces column a exactly
    (identical generation noise), across a padded batch too."""
    same = run(script, weights, pool, tmp_path / "s1", "--num", "3", "--batch-size", "2", "--format", "gif",
               "--dynamics-b", str(weights / "a" / "dynamics"))
    again = run(script, weights, pool, tmp_path / "s1_again", "--num", "3", "--batch-size", "2", "--format", "gif")
    other = run(script, weights, pool, tmp_path / "s2", "--num", "3", "--seed", "7", "--format", "gif")
    fewer = run(script, weights, pool, tmp_path / "s1_fewer", "--num", "2", "--batch-size", "2", "--format", "gif")
    for w, v in zip(same["windows"], again["windows"]):
        assert w["reward_pred"]["a"] == w["reward_pred"]["b"] == v["reward_pred"]["a"]
        assert (w["file"], w["start"]) == (v["file"], v["start"])
    key = [(w["file"], w["start"]) for w in same["windows"]]
    assert [(w["file"], w["start"]) for w in other["windows"]] != key
    assert [(w["file"], w["start"]) for w in fewer["windows"]] == key[:2]


def test_named_windows(script, weights, pool, tmp_path):
    index = run(script, weights, pool, tmp_path / "named", "--windows", "chunk_0001.h5:5", "chunk_0000.h5:10",
                "--format", "gif")
    assert [(Path(w["file"]).name, w["start"]) for w in index["windows"]] == [("chunk_0001.h5", 5),
                                                                              ("chunk_0000.h5", 10)]
    assert index["windows"][1]["frame_index"][-1] == 19  # the last frame of the segment
    with pytest.raises(ValueError, match="no window"):  # would cross the done at frame 19
        run(script, weights, pool, tmp_path / "bad", "--windows", "chunk_0000.h5:11", "--format", "gif")
    with pytest.raises(ValueError, match="names 0 files"):
        run(script, weights, pool, tmp_path / "bad", "--windows", "patch_0000.h5:0", "--format", "gif")


def test_invalid_arguments(script, weights, pool, tmp_path):
    with pytest.raises(ValueError, match="env-arg"):
        run(script, weights, pool, tmp_path / "x", "--env-arg", "action_scale=1")
    with pytest.raises(ValueError, match="only"):
        run(script, weights, pool, tmp_path / "x", "--num", "1000", "--format", "gif")
    with pytest.raises(SystemExit):
        run(script, weights, pool, tmp_path / "x", "--horizon", "0")
    # a generation constant from --env-arg reaches the environment and the window length
    index = run(script, weights, pool, tmp_path / "ctx", "--env-arg", "init_context_len=2", "--num", "1",
                "--format", "gif")
    assert index["models"]["a"]["init_context_len"] == 2 and index["settings"]["window_frames"] == 2 + HORIZON
    assert index["models"]["a"]["sources"]["init_context_len"] == "env config"


def test_video_backends(script, tmp_path, monkeypatch):
    frames = [np.full((32, 48, 3), 16 * i, dtype=np.uint8) for i in range(5)]
    assert script.sheet_steps(64, 8) == [1, 8, 16, 24, 32, 40, 48, 56, 64] and script.sheet_steps(6, 8) == [1, 6]
    for backend, module in (("imageio-ffmpeg", "imageio_ffmpeg"), ("pyav", "av")):
        try:
            __import__(module)
        except ImportError:
            continue
        path = script.write_video(tmp_path / backend, frames, 10.0, backend)
        assert path.suffix == ".mp4" and path.stat().st_size > 0
        if module == "imageio_ffmpeg":
            import imageio_ffmpeg

            assert imageio_ffmpeg.count_frames_and_secs(str(path))[0] == 5
    gif = script.write_video(tmp_path / "clip", frames, 10.0, "gif")
    with Image.open(gif) as im:
        assert im.n_frames == 5 and im.size == (48, 32)

    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setitem(sys.modules, "av", None)
    assert script.video_backend("auto") == "gif"
    with pytest.raises(RuntimeError, match="imageio-ffmpeg"):
        script.video_backend("mp4")
