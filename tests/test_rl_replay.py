# SPDX-License-Identifier: Apache-2.0
"""Replay buffer, RSSM row convention and reset-alignment check of the DMO trainer."""

import pytest
import torch

from coupled_wm.rl.dmo import ReplaySample, SequenceReplay, reset_alignment, rssm_training_batch


def _replay(capacity=24, seq_len=4, num_envs=2):
    return SequenceReplay(capacity, (1, 2, 2), action_dim=1, seq_len=seq_len, num_envs=num_envs)


def _add_step(rb, k, is_first=False):
    """Row block k: obs value k (+0.5 for env 1), action 10k, reward 100k."""
    n = rb.num_envs
    obs = torch.tensor([k + 0.5 * e for e in range(n)]).view(n, 1, 1, 1).expand(n, 1, 2, 2)
    rb.add(obs, torch.full((n, 1), 10.0 * k), torch.full((n,), 100.0 * k), torch.full((n,), is_first))


def test_capacity_must_be_a_multiple_of_num_envs():
    with pytest.raises(ValueError, match="multiple"):
        SequenceReplay(25, (1, 2, 2), 1, seq_len=4, num_envs=2)
    with pytest.raises(ValueError, match="seq_len"):
        SequenceReplay(6, (1, 2, 2), 1, seq_len=4, num_envs=2)


def test_windows_follow_one_env_and_stop_at_the_write_head():
    rb = _replay()
    for k in range(6):
        _add_step(rb, k, is_first=(k == 0))
    assert len(rb) == 12 and not rb.full
    # env 1, window starting at block 1: blocks 1..4, all behind the head (block 5)
    s = rb.get_windows(torch.tensor([3]))
    assert s.observations[0, :, 0, 0, 0].tolist() == [1.5, 2.5, 3.5, 4.5]
    assert s.actions[0, :, 0].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert s.mask[0].tolist() == [1.0, 1.0, 1.0, 1.0]
    # window starting at block 3 reaches the head (block 5) at its row 2: masked from there on
    s = rb.get_windows(torch.tensor([6]))
    assert s.observations[0, :2, 0, 0, 0].tolist() == [3.0, 4.0]
    assert s.mask[0].tolist() == [1.0, 1.0, 0.0, 0.0]
    # a window starting on the head keeps its first row only
    s = rb.get_windows(torch.tensor([10]))
    assert s.mask[0].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_wraparound_and_full():
    rb = _replay(capacity=8, seq_len=2)
    for k in range(5):
        _add_step(rb, k)
    assert rb.full and rb.pos == 2
    # rows 0-1 hold block 4 (the head), rows 2-7 blocks 1-3
    assert rb.observations[0, 0, 0, 0].item() == 4.0
    s = rb.get_windows(torch.tensor([6]))  # block 3 then block 4 (wrapped), block 4 is the head
    assert s.observations[0, :, 0, 0, 0].tolist() == [3.0, 4.0]
    assert s.mask[0].tolist() == [1.0, 0.0]
    torch.manual_seed(0)
    sample = rb.sample(16)
    assert sample.observations.shape == (16, 2, 1, 2, 2)


def test_dones_do_not_cut_windows_only_is_first():
    """Episode boundaries are carried by is_first; the validity mask ignores them."""
    rb = _replay()
    for k in range(8):
        _add_step(rb, k, is_first=(k in (0, 4)))
    s = rb.get_windows(torch.tensor([4]))  # blocks 2..5 of env 0, an episode starts at block 4
    assert s.mask[0].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert s.is_firsts[0].tolist() == [False, False, True, False]


def test_rssm_batch_convention_and_reward_mask():
    # one window: rows (o_t, a_t, r_t); an episode starts at row 3
    T, B = 6, 1
    obs = torch.arange(T, dtype=torch.float32).view(B, T, 1, 1, 1).expand(B, T, 3, 2, 2).clone()
    sample = ReplaySample(
        observations=obs,
        actions=torch.arange(T, dtype=torch.float32).view(B, T, 1) * 10,
        rewards=torch.arange(T, dtype=torch.float32).view(B, T) * 100,
        is_firsts=torch.tensor([[False, False, False, True, False, False]]),
        mask=torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]),
    )
    batch = rssm_training_batch(sample)
    assert batch.observations.shape == (T, B, 3, 2, 2)
    assert torch.equal(batch.observations[:, 0, 0, 0, 0], torch.arange(T, dtype=torch.float32) - 0.5)
    # row t carries the previous row's action and reward
    assert batch.actions[:, 0, 0].tolist() == [0.0, 0.0, 10.0, 20.0, 30.0, 40.0]
    assert batch.rewards[:, 0, 0].tolist() == [0.0, 0.0, 100.0, 200.0, 300.0, 400.0]
    assert batch.is_firsts[:, 0, 0].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert batch.mask[:, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    # reward term masked on row 0, on the seed row (row 3) and on invalid rows
    assert batch.reward_mask[:, 0].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]


def test_replay_state_roundtrip():
    rb = _replay()
    for k in range(7):
        _add_step(rb, k, is_first=(k == 0))
    other = _replay()
    other.load_state_dict(rb.state_dict())
    starts = torch.arange(len(rb))
    a, b = rb.get_windows(starts), other.get_windows(starts)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    _add_step(rb, 7)
    _add_step(other, 7)
    assert torch.equal(rb.markers, other.markers)


def test_reset_alignment():
    # paper: min_replay 40000 on 128 actors -> 313 fill steps -> resets at unroll step 6
    aligned, msg = reset_alignment(313, 64, 16)
    assert not aligned and "unroll step 6" in msg
    # release: min_replay 40960 -> 320 fill steps = 5 episodes -> resets between unrolls
    aligned, msg = reset_alignment(320, 64, 16)
    assert aligned and "between unrolls" in msg
    aligned, msg = reset_alignment(320, 60, 16)
    assert not aligned and "drift" in msg
