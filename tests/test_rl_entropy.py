# SPDX-License-Identifier: Apache-2.0
"""The actor's action scale, its maximum attainable entropy, and the trainer's target-entropy check."""

import math
from pathlib import Path

import pytest
import torch

from coupled_wm.common.distributed import DistInfo
from coupled_wm.rl.actor import SAPOActorStochasticMLP, max_attainable_entropy
from coupled_wm.rl.dmo import DMOTrainer, RunOptions
from rl_fake_env import FakeWorldModelEnv, single_thread, tiny_config  # noqa: F401 (fixture)

G1_SCALE = [0.02] * 17 + [1.0, 1.0]
ACTOR_CFG = {"units": [8], "activation": "silu", "action_scale": [0.02, 1.0], "log_std_min": -5.0,
             "log_std_max": 2.0}


def test_g1_bound_and_target():
    h_max = max_attainable_entropy(G1_SCALE, -5.0, 2.0)
    assert h_max == pytest.approx(-53.51, abs=0.01)
    assert -58 < h_max
    # per dim: log(s) + 0.6836 at mu = 0, log std = -0.13
    assert h_max == pytest.approx(sum(math.log(s) for s in G1_SCALE) + 0.6836 * 19, abs=0.01)


def test_bounds_of_other_layouts():
    # joints at 0.01 rad: the bound is -65.29, so a target of -58 is unreachable
    assert max_attainable_entropy([0.01] * 17 + [1.0, 1.0], -5.0, 2.0) == pytest.approx(-65.29, abs=0.01)
    # 22-dim layouts (no dropped action dims)
    assert max_attainable_entropy([0.02] * 20 + [1.0, 1.0], -5.0, 2.0) == pytest.approx(-63.20, abs=0.01)
    assert max_attainable_entropy([0.01] * 20 + [1.0, 1.0], -5.0, 2.0) == pytest.approx(-77.06, abs=0.01)
    # a log-std range that excludes the optimum lowers the bound
    assert max_attainable_entropy([1.0], -5.0, -2.0) < max_attainable_entropy([1.0], -5.0, 2.0)


def test_bound_matches_monte_carlo_of_the_actor_estimate():
    """The bound is the expectation of the actor's own entropy estimate at its maximiser."""
    torch.manual_seed(0)
    cfg = {"units": [8], "activation": "silu", "action_scale": [0.02, 1.0], "log_std_min": -5.0, "log_std_max": 2.0}
    actor = SAPOActorStochasticMLP(4, 2, cfg, device="cpu")
    with torch.no_grad():
        for p in actor.parameters():
            p.zero_()
        # log_std = -5 + 3.5 * (tanh(b) + 1) = -0.13  ->  tanh(b) = 0.3914...
        actor.fc_logstd.bias.fill_(math.atanh((-0.13 + 5.0) / 3.5 - 1.0))
        _, ent = actor(torch.zeros(400_000, 4, dtype=torch.float32))
    assert ent.double().mean().item() == pytest.approx(max_attainable_entropy([0.02, 1.0], -5.0, 2.0), abs=0.01)


def _trainer(tmp_path, cfg):
    env = FakeWorldModelEnv(num_envs=4, device="cpu", episode_length=8, img_size=16, num_actions=5)
    info = DistInfo(0, 0, 1, torch.device("cpu"), False)
    return DMOTrainer(cfg, RunOptions(logdir=str(tmp_path)), info, env=env)


def test_trainer_rejects_an_unreachable_target(tmp_path):
    # 4 policy dims (3 @ 0.2, 1 @ 1.0): maximum -2.09
    with pytest.raises(ValueError, match="maximum attainable entropy"):
        _trainer(tmp_path, tiny_config(**{"train.target_entropy": -2.0}))
    with pytest.raises(ValueError, match="non-zero"):
        _trainer(tmp_path, tiny_config(**{"train.target_entropy": 0.0}))
    tr = _trainer(tmp_path, tiny_config(**{"train.target_entropy": -2.2}))
    assert tr.max_entropy == pytest.approx(-2.09, abs=0.01)


def test_g1_config_target_is_reachable():
    from coupled_wm.common.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/rl/g1_dmo.yaml")
    actor = cfg["actor"]
    assert len(actor["action_scale"]) == 22 - len(cfg["env"]["drop_action_indices"]) == 19
    h_max = max_attainable_entropy(actor["action_scale"], actor["log_std_min"], actor["log_std_max"])
    assert cfg["train"]["target_entropy"] == -58
    assert cfg["train"]["target_entropy"] < h_max


def test_trainer_checks_a_scalar_action_scale_against_the_env_dims(tmp_path):
    # 4 policy dims at 0.2: maximum 4 * (log 0.2 + 0.6836) = -3.70
    with pytest.raises(ValueError, match="maximum attainable entropy"):
        _trainer(tmp_path, tiny_config(**{"actor.action_scale": 0.2, "train.target_entropy": -3.6}))
    tr = _trainer(tmp_path, tiny_config(**{"actor.action_scale": 0.2, "train.target_entropy": -3.8}))
    assert tr.max_entropy == pytest.approx(4 * (math.log(0.2) + 0.6836), abs=0.01)


def test_action_scale_follows_the_module_device():
    """``action_scale`` is a plain attribute (not a buffer), yet ``.to()`` moves it with the weights."""
    actor = SAPOActorStochasticMLP(4, 2, ACTOR_CFG, device="cpu").to("meta")
    assert actor.fc_mean.weight.device.type == "meta" and actor.action_scale.device.type == "meta"
    mean, _ = actor(torch.zeros(1, 3, 4, device="meta"), deterministic=True)
    assert (torch.tanh(mean) * actor.action_scale).device.type == "meta"  # the product of RSSMPolicy.act


def test_action_scale_length_names_both_keys():
    with pytest.raises(ValueError, match="env.action_scale"):
        SAPOActorStochasticMLP(4, 3, ACTOR_CFG, device="cpu")
