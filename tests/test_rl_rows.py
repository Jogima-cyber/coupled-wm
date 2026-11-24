# SPDX-License-Identifier: Apache-2.0
"""What the DMO trainer writes to its replay buffer and feeds its actor, checked against a fake env's history.

Covers the replay buffer row convention, the terminal row after every episode end, the masked
reward of episode seed rows (also in what RSSM finetuning optimises), sequence cutting by
``is_first`` only, the latent refresh at resets inside and at the start of an unroll (DMO and
both ablations), and the straight-through flag (DMO gradient swap and --no-diffusion unroll).
"""

import pytest
import torch

from coupled_wm.common.distributed import DistInfo
from coupled_wm.rl import dmo
from coupled_wm.rl.dmo import DMOTrainer, RunOptions, rssm_training_batch
from rl_fake_env import FakeWorldModelEnv, single_thread, tiny_config  # noqa: F401 (fixture)


def make_trainer(tmp_path, cfg=None, seed=0, **run):
    cfg = cfg or tiny_config()
    env = FakeWorldModelEnv(num_envs=cfg["train"]["num_actors"], device="cpu",
                            episode_length=cfg["train"]["max_episode_length"], img_size=cfg["env"]["img_size"],
                            num_actions=cfg["env"]["num_actions"], seed=cfg["env"]["seed"])
    info = DistInfo(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"), distributed=False)
    return DMOTrainer(cfg, RunOptions(seed=seed, logdir=str(tmp_path), **run), info, env=env)


def start(trainer):
    trainer._build_replay()
    trainer.initialize_env()
    trainer._reset_episode_stats()
    history_start = len(trainer.env.history) - 1  # the reset of initialize_env
    trainer.fill_replay_buffer()
    return history_start


def expected_rows(history):
    """Replay buffer rows implied by an env history starting at a reset: one row per step, plus a
    terminal row (terminal frame, zero action and reward) after each episode end."""
    rows = []
    cur_obs, cur_first = None, True
    for h in history:
        if h["kind"] == "reset":
            cur_obs, cur_first = h["obs"], True
            continue
        rows.append({"obs": cur_obs, "action": h["action"], "reward": h["reward"], "is_first": cur_first,
                     "terminal": False})
        cur_obs, cur_first = h["obs"], False
        if h["done"]:
            rows.append({"obs": h["obs"], "action": torch.zeros_like(h["action"]),
                         "reward": torch.zeros_like(h["reward"]), "is_first": False, "terminal": True})
    return rows


def replay_rows(rb, env_id):
    n = rb.num_envs
    idx = torch.arange(env_id, len(rb), n)
    return rb.observations[idx], rb.actions[idx], rb.rewards[idx], rb.is_firsts[idx]


def test_fill_counts_env_steps_and_writes_terminal_rows(tmp_path):
    tr = make_trainer(tmp_path, tiny_config(**{"train.min_replay": 64}))  # 16 env steps = 2 episodes of 8
    h0 = start(tr)
    steps = [h for h in tr.env.history[h0:] if h["kind"] == "step"]
    assert len(steps) == 16
    assert len(tr.replay) == (16 + 2) * tr.num_envs
    # the fill ended on an episode end: the next unroll starts on a reset
    assert bool((tr.is_first > 0.5).all()) and bool((tr.is_first_true > 0.5).all())


@pytest.mark.parametrize("episode_length,min_replay", [(8, 32), (6, 20)])
def test_replay_rows_match_env_history(tmp_path, episode_length, min_replay):
    cfg = tiny_config(**{"train.max_episode_length": episode_length, "train.min_replay": min_replay})
    tr = make_trainer(tmp_path, cfg)
    h0 = start(tr)
    for _ in range(5):
        tr.compute_actor_loss()
    rows = expected_rows(tr.env.history[h0:])
    for e in range(tr.num_envs):
        obs, act, rew, first = replay_rows(tr.replay, e)
        assert len(obs) == len(rows)
        for k, row in enumerate(rows):
            assert torch.equal(obs[k], row["obs"][e]), k
            assert torch.equal(act[k], row["action"][e]), k
            assert rew[k].item() == pytest.approx(row["reward"][e].item(), abs=0, rel=0), k
            assert bool(first[k]) == row["is_first"], k
    # dropped action dims are zero in every stored action
    assert torch.all(tr.replay.actions[:len(tr.replay), 0] == 0)


def test_rssm_rows_pair_each_frame_with_the_action_and_reward_that_produced_it(tmp_path):
    cfg = tiny_config(**{"train.max_episode_length": 6, "train.min_replay": 20, "rssm.sequence_length": 12})
    tr = make_trainer(tmp_path, cfg)
    h0 = start(tr)
    for _ in range(4):
        tr.compute_actor_loss()
    rows = expected_rows(tr.env.history[h0:])
    n = tr.num_envs
    starts = torch.arange(0, (len(rows) - 12) * n, n)  # windows of env 0 that stay behind the write head
    batch = rssm_training_batch(tr.replay.get_windows(starts))
    for b, s in enumerate((starts // n).tolist()):
        assert batch.mask[:, b].tolist() == [1.0] * 12  # episode ends inside windows do not cut them
        for t in range(1, 12):
            row, prev = rows[s + t], rows[s + t - 1]
            assert torch.equal(batch.observations[t, b], row["obs"][0] - 0.5)
            if row["is_first"]:
                # seed row of an episode: a reset, no reward label
                assert batch.is_firsts[t, b].item() == 1.0
                assert batch.reward_mask[t, b].item() == 0.0
                assert prev["terminal"]
                continue
            # (o_t, a_{t-1}, r_{t-1}): the action that produced o_t, the reward returned with it
            assert torch.equal(batch.actions[t, b], prev["action"][0])
            assert batch.rewards[t, b, 0].item() == prev["reward"][0].item()
            assert batch.reward_mask[t, b].item() == 1.0
            if row["terminal"]:
                assert batch.rewards[t, b, 0].item() != 0.0  # the episode's last reward is a label
        assert batch.reward_mask[0, b].item() == 0.0 and batch.is_firsts[0, b].item() == 1.0


def _fresh_projection(tr, frame):
    n = tr.num_envs
    with torch.no_grad(), tr.autocast():
        h, z = tr.rssm_core.latent_projector(
            torch.zeros(1, n, tr.stochastic_size * tr.discrete_size), torch.zeros(1, n, tr.recurrent_state_size),
            torch.zeros(1, n, tr.num_actions), frame - 0.5, torch.ones(1, n))
    return torch.cat([z, h], dim=-1)[0]


ABLATIONS = pytest.mark.parametrize("run", [{}, {"no_diffusion": True}, {"no_rssm_finetuning": True}],
                                    ids=["dmo", "no_diffusion", "no_rssm_finetuning"])


@ABLATIONS
def test_latent_refresh_at_a_reset_inside_the_unroll(tmp_path, run):
    # episodes of 6 steps, fill of 4 steps, unrolls of 4: the first unroll's env steps are the
    # episode's steps 5-8, so the episode ends at unroll step 1
    cfg = tiny_config(**{"train.max_episode_length": 6, "train.min_replay": 16})
    tr = make_trainer(tmp_path, cfg, **run)
    h0 = start(tr)
    tr.compute_actor_loss()
    history = tr.env.history[h0:]
    done_steps = [i for i, h in enumerate(history) if h["kind"] == "step" and h["done"]]
    assert len(done_steps) == 1
    new_first_frame = history[done_steps[0] + 1]["obs"]
    assert history[done_steps[0] + 1]["kind"] == "reset"
    # unroll step 2 acts from the new episode's first frame, projected from a fresh state
    assert torch.equal(tr.obs_buf[2], _fresh_projection(tr, new_first_frame))
    # and the state is not reset again at that step: step 3 is not a fresh projection of step 2's frame
    assert not torch.equal(tr.obs_buf[3], _fresh_projection(tr, history[done_steps[0] + 2]["obs"]))


@ABLATIONS
def test_latent_refresh_at_unroll_start_after_evaluation(tmp_path, run):
    tr = make_trainer(tmp_path, **run)
    start(tr)
    tr.compute_actor_loss()
    tr.evaluate_policy(4)
    img0 = tr.env.img0.clone()
    assert bool((tr.is_first > 0.5).all())
    tr.compute_actor_loss()
    assert torch.equal(tr.obs_buf[0], _fresh_projection(tr, img0))


def test_seed_row_after_evaluation_has_no_reward_label(tmp_path):
    tr = make_trainer(tmp_path)
    start(tr)
    tr.compute_actor_loss()                      # interrupted episode: 4 of 8 steps
    rows_before = len(tr.replay) // tr.num_envs
    last_reward = tr.replay.rewards[(rows_before - 1) * tr.num_envs].item()
    tr.evaluate_policy(4)
    tr.compute_actor_loss()
    seed = rows_before * tr.num_envs              # env 0's first row after the evaluation
    assert bool(tr.replay.is_firsts[seed])
    batch = rssm_training_batch(tr.replay.get_windows(torch.tensor([seed - tr.num_envs])))
    assert batch.rewards[1, 0, 0].item() == last_reward  # label shifted in from the interrupted episode...
    assert batch.reward_mask[1, 0].item() == 0.0          # ...and masked
    assert batch.is_firsts[1, 0, 0].item() == 1.0


def test_dones_must_be_batch_synchronous(tmp_path):
    tr = make_trainer(tmp_path)
    tr._build_replay()
    done = torch.tensor([True, False, False, False])
    with pytest.raises(RuntimeError, match="batch-synchronous"):
        tr._add_terminal_row(tr.env.img0, done)


def test_rssm_finetuning_masks_invalid_rows_and_unlabelled_rewards(tmp_path, monkeypatch):
    """train_dyn_model optimises the loss with the batch's validity and reward masks: the seed
    row of an episode (and row 0 of every window) carries no reward label."""
    cfg = tiny_config(**{"train.max_episode_length": 6, "train.min_replay": 16, "rssm.sequence_length": 10,
                         "rssm.init_udt": 1})
    tr = make_trainer(tmp_path, cfg)
    start(tr)  # 4 fill steps: rows 0-3 of each env, the write head at row 3
    for _ in range(2):
        tr.compute_actor_loss()  # 8 more steps; the episode ends at step 6 (terminal row 6, seed row 7)
    # windows of env 0 from row 0: a seed row inside (row 7) and the write head (row 12) not reached
    sample = tr.replay.get_windows(torch.zeros(tr.rssm_batch_size, dtype=torch.long))
    monkeypatch.setattr(tr.replay, "sample", lambda batch_size: sample)
    seen = []
    real_loss = dmo.reconstruction_loss

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(dmo, "reconstruction_loss", spy)
    tr.train_dyn_model()
    expected = rssm_training_batch(sample)
    assert len(seen) == 1
    assert torch.equal(seen[0]["mask"], expected.mask) and torch.equal(seen[0]["reward_mask"], expected.reward_mask)
    assert expected.reward_mask[:, 0].tolist() == [0.0] + [1.0] * 6 + [0.0] + [1.0] * 2  # row 0 and the seed row
    assert expected.is_firsts[7, 0, 0].item() == 1.0


@pytest.mark.parametrize("run", [{}, {"no_diffusion": True}], ids=["dmo", "no_diffusion"])
def test_straight_through_flag_changes_the_gradient_not_the_rollout(tmp_path, run):
    """The flag acts on the actor's backward pass in the DMO swap and in the --no-diffusion
    imagination unroll; the forward rollout is unchanged."""
    results = {}
    for flag in (False, True):
        cfg = tiny_config(**{"rssm.straight_through_stochastic_grad": flag})
        tr = make_trainer(tmp_path / str(flag), cfg, seed=3, **run)
        start(tr)
        loss = tr.compute_actor_loss()
        loss.backward()
        transition_grad = sum(float(p.grad.abs().sum()) for p in tr.rssm_core.transition_model.parameters()
                              if p.grad is not None)
        results[flag] = (loss.item(), transition_grad, tr.obs_buf.clone())
    assert results[False][0] == results[True][0]
    assert torch.equal(results[False][2], results[True][2])
    assert results[False][1] == 0.0      # default: no gradient through the stochastic latent
    assert results[True][1] > 0.0
