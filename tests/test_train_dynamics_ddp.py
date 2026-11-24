# SPDX-License-Identifier: Apache-2.0
"""Two-process (gloo, CPU) run of the dynamics trainer through the torchrun code path."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from coupled_wm.common.checkpoint import load_checkpoint
from coupled_wm.common.distributed import cleanup
from test_train_dynamics import assets, tiny_cfg  # noqa: F401  (module-scoped fixture)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int, cfg: dict, out_dir: str) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port))
    import coupled_wm.world_model.train_dynamics as td

    micro_batches = []  # (clip length, shortcut?) of every micro-step of this rank
    sample = td.sample_noise_levels

    def recording(B, T, device, shortcut, num_tau_levels):
        micro_batches.append((T, shortcut))
        return sample(B, T, device, shortcut, num_tau_levels)

    td.sample_noise_levels = recording
    try:
        dyn = td.train(cfg)
        torch.save({k: v.detach().clone() for k, v in dyn.state_dict().items()}, os.path.join(out_dir, f"rank{rank}.pt"))
        torch.save(micro_batches, os.path.join(out_dir, f"micro_batches{rank}.pt"))
    finally:
        cleanup()


# global batch -> (shortcut micro-steps of each rank per update, data position after 6 updates).
# Updates 2 and 4 use long clips. 16 short windows / 2 ranks = 8 micro-batches per rank and
# epoch (truncated to whole updates); 8 long windows -> 4.
_TWO_RANK_RUNS = {
    # 1 micro-step: the 2nd and 4th short-clip updates (1, 5) and the 2nd long one (4) put rank 1
    # on the shortcut objective.
    2: ([[0], [0], [0], [0], [0], [0]], [[0], [1], [0], [0], [1], [1]],
        {"short": {"epoch": 0, "position": 6}, "long": {"epoch": 0, "position": 2}}),
    # 2 micro-steps: F F on rank 0 and F S on rank 1 in every update, long ones included.
    4: ([[0, 0]] * 6, [[0, 1]] * 6, {"short": {"epoch": 1, "position": 4}, "long": {"epoch": 0, "position": 4}}),
}


@pytest.mark.parametrize("global_batch", sorted(_TWO_RANK_RUNS))
def test_two_rank_run(assets, tmp_path, global_batch):  # noqa: F811
    """Ranks on different objectives within an update stay in step (gloo would hang otherwise)."""
    run_dir = tmp_path / "run"
    cfg = tiny_cfg(assets, run_dir, schedule={"max_updates": 6}, train={"global_batch": global_batch})
    mp.spawn(_worker, args=(2, _free_port(), cfg, str(tmp_path)), nprocs=2, join=True)

    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt") for r in range(2))
    for k in r0:
        assert torch.equal(r0[k], r1[k]), k  # DDP kept the replicas identical
    *layouts, data_state = _TWO_RANK_RUNS[global_batch]
    for rank, layout in enumerate(layouts):
        seen = torch.load(tmp_path / f"micro_batches{rank}.pt")
        lengths = [8 if u in (2, 4) else 4 for u in range(6) for _ in layout[u]]
        assert seen == [(T, bool(s)) for T, s in zip(lengths, (s for update in layout for s in update))], rank
    ckpt = load_checkpoint(run_dir / "latest.pt", mmap=False)
    assert ckpt["counters"] == {"stage_update": 6, "long_updates": 2}
    assert set(ckpt["loss_scaler"]) == {"loss_dyn"} and len(ckpt["data_pool"]["files"]) == 2
    assert len(ckpt["rng"]) == 2
    assert ckpt["config"]["train"]["global_batch"] == global_batch
    for k in r0:
        assert torch.equal(ckpt["dyn"][k], r0[k]), k
    assert ckpt["data_state"] == data_state


def _rank_dir_worker(rank: int, world: int, port: int, cfg: dict, base: str) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port))
    from coupled_wm.world_model.train_dynamics import train

    try:
        train({**cfg, "run_dir": os.path.join(base, f"r{rank}")})  # a run_dir that differs between ranks
    finally:
        cleanup()


def test_ranks_that_see_different_checkpoints_stop(assets, tmp_path):  # noqa: F811
    from coupled_wm.world_model.train_dynamics import train

    cfg = tiny_cfg(assets, tmp_path / "r0", schedule={"max_updates": 1})
    train(cfg)  # only rank 0's directory holds a latest.pt
    with pytest.raises(Exception, match="the ranks see different checkpoints"):
        mp.spawn(_rank_dir_worker, args=(2, _free_port(), {**cfg, "schedule": {**cfg["schedule"], "max_updates": 2}},
                                         str(tmp_path)), nprocs=2, join=True)
    assert not (tmp_path / "r1" / "latest.pt").exists()
