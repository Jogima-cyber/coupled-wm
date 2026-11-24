# SPDX-License-Identifier: Apache-2.0
"""The G1 RL config and entry point: the values the release sets, checked without the paper artifacts."""

import math
from pathlib import Path

from coupled_wm.common.config import load_config
from coupled_wm.rl import train as rl_train
from coupled_wm.rl.dmo import reset_alignment

REPO = Path(__file__).resolve().parents[1]


def test_g1_rl_config():
    cfg = load_config(REPO / "configs/rl/g1_dmo.yaml")
    train, rssm = cfg["train"], cfg["rssm"]
    # 19-dim policy: 17 joints at 0.02 rad and 2 grips at 1.0; base velocity dims 0-2 dropped
    assert cfg["actor"]["action_scale"] == [0.02] * 17 + [1.0, 1.0]
    assert cfg["env"]["drop_action_indices"] == [0, 1, 2] and train["target_entropy"] == -58
    # replay buffer fill 40960 / 128 actors = 320 env steps = 5 episodes of 64: resets land between 16-step unrolls
    assert (train["min_replay"], train["num_actors"], train["max_episode_length"], train["steps_num"]) == (
        40960, 128, 64, 16)
    fill_steps = math.ceil(train["min_replay"] / train["num_actors"])
    assert reset_alignment(fill_steps, train["max_episode_length"], train["steps_num"])[0]
    # global RSSM finetuning batch (32 per rank on the paper's 8 GPUs)
    assert rssm["batch_size"] == 256 and rssm["batch_size"] % 8 == 0
    assert rssm["straight_through_stochastic_grad"] is False


def test_no_diffusion_keeps_the_old_flag_name():
    assert rl_train.parse_args(["--no-diffusion"]).no_diffusion
    assert rl_train.parse_args(["--dreamer-dreamer-dreamer"]).no_diffusion
    assert not rl_train.parse_args([]).no_diffusion
