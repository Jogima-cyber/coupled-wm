# SPDX-License-Identifier: Apache-2.0
"""The DMO loop under DDP: two CPU processes (gloo), evaluation on, checkpoint and resume; and the
ranks' agreement on resuming."""

import os
import socket

import torch
import torch.multiprocessing as mp

from coupled_wm.common.checkpoint import load_checkpoint


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, logdir, max_epochs):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    from coupled_wm.common.distributed import cleanup, init_distributed
    from coupled_wm.rl.dmo import DMOTrainer, RunOptions
    from rl_fake_env import tiny_config

    info = init_distributed(timeout_minutes=5)
    assert info.distributed and info.world_size == world
    cfg = tiny_config(**{"train.max_epochs": max_epochs, "train.save_interval": 2, "eval.every": 1,
                         "eval.num_episodes": 6, "rssm.batch_size": 4, "train.checkpoint_replay": True})
    trainer = DMOTrainer(cfg, RunOptions(seed=5, logdir=logdir), info)
    assert trainer.is_distributed and trainer.num_envs == 2 and trainer.rssm_batch_size == 2
    if os.path.exists(os.path.join(logdir, "latest.pt")):
        trainer.load_training_state(os.path.join(logdir, "latest.pt"))
    trainer.train()
    # DDP keeps every rank's networks identical
    flat = torch.cat([p.detach().flatten() for p in trainer.actor.parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(world)]
    torch.distributed.all_gather(gathered, flat)
    assert all(torch.equal(gathered[0], g) for g in gathered)
    cleanup()


def test_two_rank_training_with_evaluation_and_resume(tmp_path):
    logdir = str(tmp_path / "run")
    mp.spawn(_worker, args=(2, _free_port(), logdir, 2), nprocs=2, join=True)
    state = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert state["world_size"] == 2 and len(state["rng"]) == 2
    assert state["counters"]["iter"] == 2
    assert (tmp_path / "run" / "replay" / "rank0.pt").is_file() and (tmp_path / "run" / "replay" / "rank1.pt").is_file()
    # two global steps per iteration: 4 steps x 4 envs (2 per rank)
    assert state["step"] == 2 * 4 * 4

    mp.spawn(_worker, args=(2, _free_port(), logdir, 3), nprocs=2, join=True)
    assert load_checkpoint(tmp_path / "run" / "latest.pt")["counters"]["iter"] == 3


def _resume_agreement_worker(rank, world, port, checkpoint):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    from coupled_wm.common.distributed import cleanup, init_distributed
    from coupled_wm.rl.train import _check_resume_agreement

    info = init_distributed(timeout_minutes=5)
    # node-local log dirs: only rank 0 sees latest.pt; every rank must stop, none may hang
    try:
        _check_resume_agreement(checkpoint if rank == 0 else None, info)
    except SystemExit as e:
        assert "storage that every node sees" in str(e), e
    else:
        raise AssertionError(f"rank {rank} went on although the ranks disagree")
    _check_resume_agreement(checkpoint, info)
    _check_resume_agreement(None, info)
    cleanup()


def test_ranks_must_agree_on_resuming(tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"")
    mp.spawn(_resume_agreement_worker, args=(2, _free_port(), str(checkpoint)), nprocs=2, join=True)
