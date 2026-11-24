# SPDX-License-Identifier: Apache-2.0
"""Policy checkpoints, resumable training state, stopping, and the RSSM warm start of the DMO trainer."""

import os
import shutil
import signal
import threading

import pytest
import torch

from coupled_wm.common.checkpoint import load_checkpoint
from coupled_wm.common.distributed import DistInfo
from coupled_wm.rl import dmo
from coupled_wm.rl.dmo import DMOTrainer, RunOptions, load_stripped_state_dict
from coupled_wm.rssm.rssm import RELATIVE_ACTION_DEFINITION, build_rssm, save_rssm
from rl_fake_env import FakeWorldModelEnv, single_thread, tiny_config  # noqa: F401 (fixture)

CPU = DistInfo(0, 0, 1, torch.device("cpu"), False)


def make_trainer(logdir, cfg, seed=0, **run):
    env = FakeWorldModelEnv(num_envs=cfg["train"]["num_actors"], device="cpu",
                            episode_length=cfg["train"]["max_episode_length"], img_size=cfg["env"]["img_size"],
                            num_actions=cfg["env"]["num_actions"], seed=seed)
    return DMOTrainer(cfg, RunOptions(seed=seed, logdir=str(logdir), **run), CPU, env=env)


def _assert_state_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k].cpu(), b[k].cpu()), k


def _assert_optimizer_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    assert sa["state"].keys() == sb["state"].keys()
    for k in sa["state"]:
        for name, value in sa["state"][k].items():
            other = sb["state"][k][name]
            assert torch.equal(torch.as_tensor(value).cpu(), torch.as_tensor(other).cpu()), (k, name)


def test_training_state_roundtrip_and_resume(tmp_path):
    cfg = tiny_config(**{"train.max_epochs": 2, "train.save_interval": 2, "eval.every": 100})
    tr = make_trainer(tmp_path, cfg)
    tr.train()
    state = load_checkpoint(tmp_path / "latest.pt")
    assert state["format"] == "coupled_wm.rl.training_state"
    assert state["counters"]["iter"] == 2 and state["counters"]["rssm_pretrained"]
    for key in list(state["actor"]) + list(state["rssm"]):
        assert "_orig_mod" not in key and not key.startswith("module.")

    cfg2 = tiny_config(**{"train.max_epochs": 4, "train.save_interval": 2, "eval.every": 100})
    resumed = make_trainer(tmp_path, cfg2, seed=7)
    resumed.load_training_state(tmp_path / "latest.pt")
    _assert_state_equal(tr.actor_core.state_dict(), resumed.actor_core.state_dict())
    _assert_state_equal(tr.rssm_core.state_dict(), resumed.rssm_core.state_dict())
    for c1, c2 in zip(tr.critic_cores, resumed.critic_cores):
        _assert_state_equal(c1.state_dict(), c2.state_dict())
    for o1, o2 in ((tr.actor_optimizer, resumed.actor_optimizer), (tr.critic_optimizer, resumed.critic_optimizer),
                   (tr.dyn_model_optimizer, resumed.dyn_model_optimizer), (tr.a_optimizer, resumed.a_optimizer)):
        _assert_optimizer_equal(o1, o2)
    assert torch.equal(tr.log_alpha, resumed.log_alpha) and tr.alpha == resumed.alpha
    assert (resumed.iter_count, resumed.step_count, resumed.pretrained) == (tr.iter_count, tr.step_count, True)
    assert resumed.best_policy_loss == tr.best_policy_loss
    assert len(resumed.episode_loss_meter) == len(tr.episode_loss_meter)
    assert torch.equal(resumed.episode_loss_meter.mean, tr.episode_loss_meter.mean)
    assert torch.equal(torch.get_rng_state(), state["rng"][0]["torch"])

    resumed.train()
    assert resumed.iter_count == 4
    assert load_checkpoint(tmp_path / "latest.pt")["counters"]["iter"] == 4


def test_resume_with_checkpointed_replay(tmp_path):
    cfg = tiny_config(**{"train.max_epochs": 2, "train.save_interval": 2, "eval.every": 100,
                         "train.checkpoint_replay": True})
    tr = make_trainer(tmp_path, cfg)
    tr.train()
    assert (tmp_path / "replay" / "rank0.pt").is_file()
    resumed = make_trainer(tmp_path, cfg)
    resumed.load_training_state(tmp_path / "latest.pt")
    assert resumed.replay is not None and len(resumed.replay) == len(tr.replay)
    assert torch.equal(resumed.replay.observations, tr.replay.observations)
    assert torch.equal(resumed.replay.markers, tr.replay.markers)


def test_resume_without_replay_checkpoint_refills_and_keeps_the_update_schedule(tmp_path, monkeypatch, caplog):
    """Without train.checkpoint_replay a resumed run refills min_replay rows with the current
    policy, then does the usual rssm.udt updates per iteration: init_udt is not rerun."""
    settings = {"train.save_interval": 2, "eval.every": 100, "rssm.init_udt": 5, "rssm.udt": 2}
    cfg = tiny_config(**settings, **{"train.max_epochs": 2})
    make_trainer(tmp_path, cfg).train()
    sampled_from, filled = [], []
    real_sample, real_fill = dmo.SequenceReplay.sample, DMOTrainer.fill_replay_buffer

    def sample(replay, batch_size):
        sampled_from.append(len(replay))
        return real_sample(replay, batch_size)

    def fill(trainer):
        real_fill(trainer)
        filled.append(len(trainer.replay))

    monkeypatch.setattr(dmo.SequenceReplay, "sample", sample)
    monkeypatch.setattr(DMOTrainer, "fill_replay_buffer", fill)
    resumed = make_trainer(tmp_path, tiny_config(**settings, **{"train.max_epochs": 3}),
                           resume=str(tmp_path / "latest.pt"))
    with caplog.at_level("INFO", logger="coupled_wm.rl.dmo"):
        assert resumed.train() and resumed.iter_count == 3
    n, train = cfg["train"]["num_actors"], cfg["train"]
    fill_steps = train["min_replay"] // n
    # min_replay rows plus one terminal row per env at each episode end during the fill
    assert filled == [train["min_replay"] + n * (fill_steps // train["max_episode_length"])]
    assert sampled_from == filled * 2  # rssm.udt updates on the refilled replay buffer, not rssm.init_udt
    assert any("warm-up is not rerun" in r.getMessage() for r in caplog.records)


def test_policy_checkpoint_play(tmp_path):
    cfg = tiny_config(**{"train.max_epochs": 2, "eval.every": 100})
    tr = make_trainer(tmp_path, cfg)
    tr.train()
    payload = load_checkpoint(tmp_path / "final_policy.pt")
    assert payload["format"] == "coupled_wm.rl.policy" and "optimizers" not in payload
    assert payload["drop_action_indices"] == [0] and payload["num_actions"] == 5
    assert payload["config"]["actor"] == cfg["actor"]

    player = make_trainer(tmp_path / "play", cfg, seed=1, play=True)
    loss, _, length = player.play(tmp_path / "final_policy.pt", num_games=4)
    _assert_state_equal(tr.actor_core.state_dict(), player.actor_core.state_dict())
    _assert_state_equal(tr.rssm_core.state_dict(), player.rssm_core.state_dict())
    assert length == cfg["train"]["max_episode_length"] and loss == loss


def test_load_into_compiled_submodules():
    rssm = build_rssm({"action_size": 3, "image_size": 16, "stochastic_size": 2, "discrete_size": 2,
                       "recurrent_state_size": 8, "dense_units": 8, "channels_multiplier": 1, "mlp_layers": 1,
                       "reward_model_bins": 7})
    source = {k: torch.randn_like(v) for k, v in rssm.state_dict().items()}
    rssm.encoder = torch.compile(rssm.encoder)  # wrapper only: nothing is compiled until called
    assert any("_orig_mod" in k for k in rssm.state_dict())
    load_stripped_state_dict(rssm, source)
    assert torch.equal(rssm.encoder._orig_mod._model[0].weight, source["encoder._model.0.weight"])
    with pytest.raises(KeyError, match="missing"):
        load_stripped_state_dict(rssm, {k: v for k, v in source.items() if not k.startswith("decoder")})


def _rssm_block_arch(cfg):
    return {**cfg["rssm"], "action_size": cfg["env"]["num_actions"], "image_size": cfg["env"]["img_size"]}


@pytest.mark.parametrize("kind", ["pretrained_dir", "bare_state_dict"])
def test_rssm_warm_start(tmp_path, kind):
    cfg = tiny_config()
    torch.manual_seed(123)
    source = build_rssm(_rssm_block_arch(cfg))
    if kind == "pretrained_dir":
        path = save_rssm(tmp_path / "rssm", source, {"action_convention": "relative"})
    else:
        # the paper's format: a bare state dict with torch.compile prefixes on three submodules
        sd = {k.replace("encoder.", "encoder._orig_mod.", 1).replace("reward_model.", "reward_model._orig_mod.", 1): v
              for k, v in source.state_dict().items()}
        path = tmp_path / "rssm_e10.pt"
        torch.save(sd, path)
    tr = make_trainer(tmp_path, tiny_config(**{"rssm.pretrained": str(path)}))
    _assert_state_equal(source.state_dict(), tr.rssm_core.state_dict())
    assert tr.rssm_warmstart_info["path"] == str(path)


def test_rssm_warm_start_architecture_mismatch(tmp_path):
    cfg = tiny_config()
    source = build_rssm({**_rssm_block_arch(cfg), "recurrent_state_size": 32})
    path = save_rssm(tmp_path / "rssm", source)
    with pytest.raises(ValueError, match="recurrent_state_size"):
        make_trainer(tmp_path, tiny_config(**{"rssm.pretrained": str(path)}))


def test_rssm_warm_start_action_convention(tmp_path, caplog):
    cfg = tiny_config()
    source = build_rssm(_rssm_block_arch(cfg))
    absolute = save_rssm(tmp_path / "abs", source, {"action": {"mode": "absolute", "definition": "commands[t]"}})
    with pytest.raises(ValueError, match="'absolute' actions"):
        make_trainer(tmp_path, tiny_config(**{"rssm.pretrained": str(absolute)}))
    shifted = save_rssm(tmp_path / "shifted", source,
                        {"action": {"mode": "relative", "definition": "commands[t-1] - commands[t-2]"}})
    with caplog.at_level("WARNING", logger="coupled_wm.rl.dmo"):
        make_trainer(tmp_path, tiny_config(**{"rssm.pretrained": str(shifted)}))
    assert "commands[t-1] - commands[t-2]" in caplog.text and "re-fits it" in caplog.text
    for ablation in ("no_diffusion", "no_rssm_finetuning"):
        caplog.clear()
        with caplog.at_level("WARNING", logger="coupled_wm.rl.dmo"):
            make_trainer(tmp_path, tiny_config(**{"rssm.pretrained": str(shifted)}), **{ablation: True})
        assert "the mismatch stays for the whole run" in caplog.text, ablation


def test_rssm_warm_start_color_order(tmp_path, caplog):
    cfg = tiny_config()
    source = build_rssm(_rssm_block_arch(cfg))
    rgb = save_rssm(tmp_path / "rgb", source, {"preprocessing": {"h5_color_order": "rgb"}})
    for env_order, warned in (("bgr", True), ("rgb", False)):
        caplog.clear()
        env = FakeWorldModelEnv(4, "cpu", 8, num_actions=5, color_order=env_order)
        with caplog.at_level("WARNING", logger="coupled_wm.rl.dmo"):
            DMOTrainer(tiny_config(**{"rssm.pretrained": str(rgb)}), RunOptions(logdir=str(tmp_path)), CPU, env=env)
        assert ("stored as rgb" in caplog.text) == warned, env_order


def test_latest_holds_the_target_critic_of_its_iteration(tmp_path):
    """The Polyak update of iteration k comes before the save of iteration k, so a resumed
    single-critic run has the target the uninterrupted run has."""
    cfg = tiny_config(**{"train.max_epochs": 2, "train.save_interval": 2, "eval.every": 100,
                         "critic.clipped_double_critic": False, "train.critic_method": "one-step"})
    tr = make_trainer(tmp_path, cfg)
    tr.train()
    state = load_checkpoint(tmp_path / "latest.pt")
    assert state["counters"]["iter"] == 2
    _assert_state_equal(state["target_critic"], tr.target_critic_net.state_dict())


def test_resume_continues_the_env_seed_sequence(tmp_path, caplog):
    cfg = tiny_config(**{"train.max_epochs": 2, "train.save_interval": 2, "eval.every": 100})
    make_trainer(tmp_path, cfg).train()
    saved = load_checkpoint(tmp_path / "latest.pt")["env"][0]
    generator = torch.Generator()
    generator.set_state(saved["generator"])
    next_states = torch.rand(4, 5, generator=generator) - 0.5  # what the original run would seed next

    resumed = make_trainer(tmp_path, cfg, resume=str(tmp_path / "latest.pt"))
    first_states = resumed.env.history[0]["obs"]  # the constructor's reset, before the resume
    resumed.env.reset()
    assert torch.equal(resumed.env.state, next_states)
    assert not torch.equal(resumed.env.history[-1]["obs"], first_states)

    # an env state that does not fit this env is reported, not applied
    with caplog.at_level("WARNING", logger="coupled_wm.rl.dmo"):
        DMOTrainer(cfg, RunOptions(logdir=str(tmp_path), resume=str(tmp_path / "latest.pt")), CPU,
                   env=FakeWorldModelEnv(2, "cpu", 8, num_actions=5))
    assert "seed-window sequence is not restored (the env state is for 4 envs" in caplog.text


def test_resume_does_not_need_the_warm_start(tmp_path):
    source = build_rssm(_rssm_block_arch(tiny_config()))
    warm = save_rssm(tmp_path / "rssm", source, {"action": {"mode": "relative",
                                                            "definition": RELATIVE_ACTION_DEFINITION}})
    cfg = tiny_config(**{"rssm.pretrained": str(warm), "train.max_epochs": 2, "train.save_interval": 2,
                         "eval.every": 100})
    run_dir = tmp_path / "run"
    tr = make_trainer(run_dir, cfg)
    tr.train()
    shutil.rmtree(warm)
    with pytest.raises(FileNotFoundError, match="rssm.pretrained"):
        make_trainer(run_dir, cfg)
    resumed = make_trainer(run_dir, cfg, resume=str(run_dir / "latest.pt"))
    assert resumed.rssm_warmstart_info["path"] == str(warm)
    _assert_state_equal(tr.rssm_core.state_dict(), resumed.rssm_core.state_dict())


def test_stop_signal_saves_and_exits(tmp_path):
    assert threading.current_thread() is threading.main_thread()  # the handler is installed in the main thread
    cfg = tiny_config(**{"train.max_epochs": 4, "train.save_interval": 100, "eval.every": 100})
    tr = make_trainer(tmp_path, cfg)
    real_update = tr.train_dyn_model

    def update_then_signal():
        real_update()
        if tr.iter_count == 1:  # during the second iteration
            os.kill(os.getpid(), signal.SIGUSR1)

    tr.train_dyn_model = update_then_signal
    previous = signal.getsignal(signal.SIGUSR1)
    assert tr.train() is False
    assert signal.getsignal(signal.SIGUSR1) is previous
    assert load_checkpoint(tmp_path / "latest.pt")["counters"]["iter"] == 2
    assert not (tmp_path / "final_policy.pt").exists() and tr.env.closed

    resumed = make_trainer(tmp_path, cfg, resume=str(tmp_path / "latest.pt"))
    assert resumed.train() is True and resumed.iter_count == 4
    assert (tmp_path / "final_policy.pt").is_file()


def test_relaunching_a_finished_run_only_writes_the_final_policy(tmp_path, monkeypatch):
    cfg = tiny_config(**{"train.max_epochs": 2, "train.save_interval": 2, "eval.every": 100})
    make_trainer(tmp_path, cfg).train()
    (tmp_path / "final_policy.pt").unlink()
    again = make_trainer(tmp_path, cfg, resume=str(tmp_path / "latest.pt"))
    monkeypatch.setattr(again, "fill_replay_buffer", lambda: pytest.fail("refilled the replay buffer"))
    monkeypatch.setattr(again, "evaluate_policy", lambda *a, **k: pytest.fail("evaluated again"))
    assert again.train() is True
    assert (tmp_path / "final_policy.pt").is_file() and again.replay is None


def test_rssm_warm_start_reward_remap(tmp_path, caplog):
    """Warns when the RSSM and the world model's reward head were trained on remapped labels differently."""
    source = build_rssm(_rssm_block_arch(tiny_config()))
    for rssm_remap, warned in ((None, True), ([[10.0, 2.0]], False)):
        caplog.clear()
        warm = save_rssm(tmp_path / str(warned), source, {"preprocessing": {"reward_remap": rssm_remap}})
        env = FakeWorldModelEnv(4, "cpu", 8, num_actions=5)
        env.dynamics_config = {"training": {"data": {"reward_remap": {"10.0": 2.0}}}}
        with caplog.at_level("WARNING", logger="coupled_wm.rl.dmo"):
            DMOTrainer(tiny_config(**{"rssm.pretrained": str(warm)}), RunOptions(logdir=str(tmp_path)), CPU, env=env)
        assert ("reward_remap" in caplog.text) == warned, rssm_remap


def test_resumed_launch_purges_events_after_the_checkpoint(tmp_path, monkeypatch):
    import coupled_wm.rl.dmo as dmo

    purge_steps = []

    class Recording(dmo.ExperimentLogger):
        def __init__(self, *args, purge_step=None, **kwargs):
            purge_steps.append(purge_step)
            super().__init__(*args, purge_step=purge_step, **kwargs)

    monkeypatch.setattr(dmo, "ExperimentLogger", Recording)
    cfg = tiny_config(**{"train.max_epochs": 2, "train.save_interval": 2, "eval.every": 100})
    make_trainer(tmp_path, cfg).train()
    saved_step = load_checkpoint(tmp_path / "latest.pt")["counters"]["step"]
    cfg = tiny_config(**{"train.max_epochs": 3, "train.save_interval": 2, "eval.every": 100})
    make_trainer(tmp_path, cfg, resume=str(tmp_path / "latest.pt")).train()
    assert purge_steps == [0, saved_step + 1]  # on the env-step axis
