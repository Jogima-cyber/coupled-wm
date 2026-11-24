# SPDX-License-Identifier: Apache-2.0
"""A few iterations of the full DMO loop on the fake environment, and the command-line tools."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from coupled_wm.common.checkpoint import load_checkpoint
from coupled_wm.common.config import load_config, save_config
from coupled_wm.common.distributed import DistInfo
from coupled_wm.rl import dmo
from coupled_wm.rl import evaluate as rl_evaluate
from coupled_wm.rl import train as rl_train
from coupled_wm.rl.dmo import DMOTrainer, RSSMPolicy, RunOptions
from rl_fake_env import FakeWorldModelEnv, single_thread, tiny_config  # noqa: F401 (fixture)

CPU = DistInfo(0, 0, 1, torch.device("cpu"), False)


def run_training(logdir, cfg, **run):
    tr = DMOTrainer(cfg, RunOptions(seed=0, logdir=str(logdir), **run), CPU)
    tr.train()
    return tr


@pytest.mark.parametrize("variant", [
    {},                                                         # DMO, eval on
    {"run": {"no_diffusion": True}},                            # ablation: imagination unroll
    {"run": {"no_rssm_finetuning": True}},                      # ablation: frozen RSSM
    {"cfg": {"rssm.straight_through_stochastic_grad": True}},
    {"run": {"no_diffusion": True}, "cfg": {"rssm.straight_through_stochastic_grad": True}},
    {"cfg": {"critic.clipped_double_critic": False, "train.critic_method": "one-step"}},
    {"cfg": {"train.replay_device": "cpu", "train.max_episode_length": 6, "train.min_replay": 20,
             "train.lr_schedule": "constant"}},                 # resets inside unrolls
])
def test_dmo_loop_runs(tmp_path, variant):
    cfg = tiny_config(**{"train.max_epochs": 4, **variant.get("cfg", {})})
    tr = run_training(tmp_path, cfg, **variant.get("run", {}))
    assert tr.iter_count == 4
    assert math.isfinite(tr.actor_loss) and math.isfinite(tr.value_loss) and math.isfinite(tr.alpha)
    finetuned = not variant.get("run")
    assert tr.pretrained == finetuned
    if finetuned:
        assert math.isfinite(tr.dyn_loss) and tr.dyn_loss > 0
    for name in ("final_policy.pt", "latest.pt", "config.yaml", "episode_loss_his.npy"):
        assert (tmp_path / name).is_file(), name
    assert (tmp_path / "tb").is_dir()
    saved = load_config(tmp_path / "config.yaml")
    assert saved["run"]["no_diffusion"] == bool(variant.get("run", {}).get("no_diffusion", False))
    assert tr.step_count == 4 * cfg["train"]["steps_num"] * cfg["train"]["num_actors"]
    assert tr.env.closed
    if not cfg["critic"]["clipped_double_critic"]:
        assert not any(p.requires_grad for p in tr.target_critic_net.parameters())
        # Polyak: the target moved towards the critic but is not equal to it
        live = dict(tr.critic_cores[0].named_parameters())
        assert any(not torch.equal(p, live[n]) for n, p in tr.target_critic_net.named_parameters())


def test_rssm_batch_is_global(tmp_path):
    cfg = tiny_config(**{"rssm.batch_size": 6})
    with pytest.raises(ValueError, match="divisible"):
        DMOTrainer(cfg, RunOptions(logdir=str(tmp_path)), DistInfo(0, 0, 4, torch.device("cpu"), False),
                   env=FakeWorldModelEnv(1, "cpu", 8, num_actions=5))
    tr = DMOTrainer(tiny_config(**{"rssm.batch_size": 8}), RunOptions(logdir=str(tmp_path)),
                    DistInfo(0, 0, 4, torch.device("cpu"), False), env=FakeWorldModelEnv(1, "cpu", 8, num_actions=5))
    assert tr.rssm_batch_size == 2 and tr.min_replay == 8 and tr.replay_size == 100


def test_replay_size_must_split_into_env_blocks(tmp_path):
    with pytest.raises(ValueError, match="multiple of the per-rank num_envs"):
        DMOTrainer(tiny_config(**{"train.replay_size": 402}), RunOptions(logdir=str(tmp_path)), CPU,
                   env=FakeWorldModelEnv(4, "cpu", 8, num_actions=5))


def test_play_has_no_replay_constraints(tmp_path):
    """--play builds no replay buffer: player.num_actors need not divide train.replay_size."""
    tr = DMOTrainer(tiny_config(**{"eval.player.num_actors": 3}), RunOptions(play=True), CPU,
                    env=FakeWorldModelEnv(3, "cpu", 8, num_actions=5))
    assert tr.replay_size is None and tr.num_envs == 3


@pytest.mark.parametrize("overrides,world,play,error,match", [
    ({"train.num_actors": 6}, 4, False, ValueError, "train.num_actors=6"),
    ({"eval.player.num_actors": 6}, 4, True, ValueError, "eval.player.num_actors=6"),
    ({"eval.player.num_actors": 1}, 4, True, ValueError, "eval.player.num_actors=1"),
    ({"rssm.batch_size": 6}, 4, False, ValueError, "rssm.batch_size=6"),
    ({"train.replay_size": 402}, 1, False, ValueError, "multiple of the per-rank num_envs"),
    ({"train.replay_size": 402}, 4, False, ValueError, "train.replay_size=402"),
    ({"train.target_entropy": -2.0}, 1, False, ValueError, "maximum attainable entropy"),
    ({"rssm.pretrained": "/nonexistent/rssm"}, 1, False, FileNotFoundError, "rssm.pretrained"),
])
def test_config_errors_come_before_the_world_model_is_loaded(tmp_path, monkeypatch, overrides, world, play,
                                                              error, match):
    def make_env(*args, **kwargs):
        raise AssertionError("the env was built before the config was checked")

    monkeypatch.setattr(dmo, "make_env", make_env)
    with pytest.raises(error, match=match):
        DMOTrainer(tiny_config(**overrides), RunOptions(logdir=str(tmp_path), play=play),
                   DistInfo(0, 0, world, torch.device("cpu"), False))


def test_out_of_memory_names_the_knobs(tmp_path, monkeypatch):
    def make_env(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(dmo, "make_env", make_env)
    with pytest.raises(torch.OutOfMemoryError, match=r"4 envs per rank \(train.num_actors=4 over the world size 1\)"):
        DMOTrainer(tiny_config(), RunOptions(logdir=str(tmp_path)), CPU)

    tr = DMOTrainer(tiny_config(), RunOptions(logdir=str(tmp_path)), CPU, env=FakeWorldModelEnv(4, "cpu", 8))

    def replay_init(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(dmo.SequenceReplay, "__init__", replay_init)
    with pytest.raises(torch.OutOfMemoryError, match="train.replay_device: cpu"):
        tr._build_replay()


def test_gpu_replay_larger_than_the_gpu_fails_before_the_env(tmp_path, monkeypatch):
    monkeypatch.setattr(dmo, "make_env", lambda *a, **k: pytest.fail("the env was built"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=2**20))
    cuda = DistInfo(0, 0, 1, torch.device("cuda", 0), False)
    cfg = tiny_config(**{"train.replay_device": "cuda"})
    # 400 rows of 16x16 fp32 frames: 1.2 MB > 1 MiB
    with pytest.raises(ValueError, match="the replay buffer needs at least .* train.replay_device: cpu"):
        DMOTrainer(cfg, RunOptions(logdir=str(tmp_path)), cuda)


def test_evaluation_strips_end_on_the_terminal_frame():
    env = FakeWorldModelEnv(num_envs=2, device="cpu", episode_length=5, num_actions=5)
    torch.manual_seed(0)
    cfg = tiny_config()
    tr = DMOTrainer(cfg, RunOptions(play=True), CPU, env=env)
    policy = RSSMPolicy(tr.actor_core, tr.rssm_core, 5, (0,), 2, torch.device("cpu"))
    episodes, trajectories = rl_evaluate.evaluate(env, policy, num_episodes=4, gamma=0.99, save_trajectories=2)
    assert len(episodes) == 4 and all(e["length"] == 5 for e in episodes)
    frames, rewards = trajectories[0]
    assert frames.shape[0] == 6 and rewards.shape == (5,)
    steps = [h for h in env.history if h["kind"] == "step"]
    assert torch.equal(frames[-1], steps[4]["obs"][0])                 # terminal frame, not the next reset
    assert rewards.tolist() == pytest.approx([s["reward"][0].item() for s in steps[:5]])
    assert episodes[0]["return"] == pytest.approx(sum(s["reward"][0].item() for s in steps[:5]))


def test_command_line_train_play_resume_evaluate(tmp_path):
    cfg = tiny_config(**{"train.max_epochs": 2, "eval.every": 1})
    cfg_path = tmp_path / "cfg.yaml"
    save_config(cfg, cfg_path)
    run_dir = tmp_path / "run"

    rl_train.main(["--config", str(cfg_path), "--seed", "3", "--logdir", str(run_dir), "--no-eval"])
    assert load_checkpoint(run_dir / "latest.pt")["counters"]["iter"] == 2

    loss, _, length = rl_train.main(["--play", "--checkpoint", str(run_dir / "final_policy.pt")])
    assert length == cfg["train"]["max_episode_length"] and np.isfinite(loss)

    # the run directory holds latest.pt: training continues from iteration 2
    rl_train.main(["--config", str(cfg_path), "--seed", "3", "--logdir", str(run_dir), "--no-eval",
                   "--set", "train.max_epochs=3"])
    assert load_checkpoint(run_dir / "latest.pt")["counters"]["iter"] == 3

    with pytest.raises(SystemExit, match="no_rssm_finetuning"):
        rl_train.main(["--config", str(cfg_path), "--logdir", str(run_dir), "--no-rssm-finetuning"])

    summary = rl_evaluate.main(["--checkpoint", str(run_dir / "best_policy.pt"), "--num-envs", "2",
                                "--episodes", "4", "--output-dir", str(tmp_path / "eval"), "--device", "cpu"])
    assert summary["episodes"] == 4
    written = json.loads((tmp_path / "eval" / "summary.json").read_text())
    assert written["mean_return"] == pytest.approx(summary["mean_return"])
    assert (tmp_path / "eval" / "traj_0.png").is_file() and (tmp_path / "eval" / "episodes.csv").is_file()


# The meta device stands in for cuda on CPU-only machines: like cuda, an op that mixes it with a CPU
# tensor of one or more dims raises. Meta tensors hold no values, so torch.distributions cannot
# validate its arguments on them, and the episode bookkeeping (.item(), .cpu()) cannot run.
META = torch.device("meta")
on_meta = pytest.mark.filterwarnings("ignore:.*copying from a non-meta parameter")


def act_steps(policy, env, steps=3):
    """RSSMPolicy.act from the env's frames, with episode starts in between (what evaluate() does)."""
    obs = env.reset(torch.arange(env.num_envs))
    policy.reset()
    for t in range(steps):
        action = policy.act(obs)
        assert action.device == env.device and tuple(action.shape) == (env.num_envs, env.num_actions)
        policy.observe_dones(torch.full((env.num_envs,), t == 1, dtype=torch.bool, device=env.device))


def plain_tensor_attributes(module):
    """Tensors held as plain attributes (not parameters or buffers): nn.Module.to() skips them."""
    return {f"{name}.{key}": value for name, m in module.named_modules() for key, value in vars(m).items()
            if isinstance(value, torch.Tensor)}


@on_meta
@pytest.mark.parametrize("checkpoint_format", ["release", "paper"])
def test_evaluate_acts_on_the_env_device(tmp_path, monkeypatch, checkpoint_format):
    """The standalone evaluation builds the policy on --device: every tensor act() uses is there."""
    monkeypatch.setattr(torch.distributions.Distribution, "_validate_args", False)
    cfg = tiny_config()
    tr = DMOTrainer(cfg, RunOptions(play=True), CPU, env=FakeWorldModelEnv(4, "cpu", 8, num_actions=5))
    args = ["--checkpoint", str(tmp_path / "policy.pt"), "--num-envs", "2", "--device", "meta",
            "--output-dir", str(tmp_path / "eval")]
    if checkpoint_format == "release":
        torch.save(tr.policy_state(), tmp_path / "policy.pt")
    else:  # the paper code saved wrapper-prefixed state dicts and no config
        torch.save({"actor": {f"module.{k}": v for k, v in tr.actor_core.state_dict().items()},
                    "rssm": {f"module.{k}": v for k, v in tr.rssm_core.state_dict().items()}}, tmp_path / "policy.pt")
        save_config(cfg, tmp_path / "cfg.yaml")
        args += ["--config", str(tmp_path / "cfg.yaml")]
    acted = []

    def evaluate_on_meta(env, policy, num_episodes, gamma, save_trajectories=0):
        assert env.device == META and policy.device == META
        devices = {t.device for module in (policy.actor, policy.rssm)
                   for t in (*module.parameters(), *module.buffers(), *plain_tensor_attributes(module).values())}
        assert devices == {META}
        act_steps(policy, env)
        acted.append(True)
        return [{"env": e, "return": 0.0, "discounted_return": 0.0, "length": 8} for e in range(num_episodes)], []

    monkeypatch.setattr(rl_evaluate, "evaluate", evaluate_on_meta)
    assert rl_evaluate.main(args)["episodes"] == 2 and acted


@on_meta
def test_trainer_policy_modules_follow_the_device(monkeypatch):
    """--play and the in-training evaluation act with the trainer's actor and RSSM. The RSSM is
    built on the CPU and moved; the actor's per-dim action_scale is a plain attribute: no tensor
    may stay behind when they move."""
    monkeypatch.setattr(torch.distributions.Distribution, "_validate_args", False)
    env = FakeWorldModelEnv(2, "cpu", 8, num_actions=5)
    tr = DMOTrainer(tiny_config(), RunOptions(play=True), CPU, env=env)
    assert isinstance(tr.actor_core.action_scale, torch.Tensor)  # the per-dim scale of the G1 config's form
    actor, rssm = tr.actor_core.to(META), tr.rssm_core.to(META)
    assert {t.device for m in (actor, rssm) for t in plain_tensor_attributes(m).values()} == {META}
    meta_env = FakeWorldModelEnv(2, META, 8, num_actions=5)
    act_steps(RSSMPolicy(actor, rssm, tr.num_actions, tr.drop_action_indices, 2, META), meta_env)


def test_evaluate_a_paper_format_checkpoint(tmp_path):
    """The paper code saved DDP/compile-prefixed state dicts and no config."""
    cfg = tiny_config()
    tr = DMOTrainer(cfg, RunOptions(play=True), CPU, env=FakeWorldModelEnv(4, "cpu", 8, num_actions=5))
    paper = {
        "actor": {f"module.{k}": v for k, v in tr.actor_core.state_dict().items()},
        "critic": [{f"module.{k}": v for k, v in c.state_dict().items()} for c in tr.critic_cores],
        "target_critic": None,
        "rssm": {f"module.{k}".replace("module.encoder.", "module.encoder._orig_mod."): v
                 for k, v in tr.rssm_core.state_dict().items()},
        "act_rms": None, "ret_rms": None, "cfg": None,
    }
    torch.save(paper, tmp_path / "best_policy.pt")
    with pytest.raises(SystemExit, match="--config"):
        rl_evaluate.main(["--checkpoint", str(tmp_path / "best_policy.pt"), "--device", "cpu"])
    save_config(cfg, tmp_path / "cfg.yaml")
    summary = rl_evaluate.main(["--checkpoint", str(tmp_path / "best_policy.pt"), "--config", str(tmp_path / "cfg.yaml"),
                                "--num-envs", "2", "--save-trajectories", "0", "--device", "cpu"])
    assert summary["episodes"] == 2 and summary["iteration"] is None
