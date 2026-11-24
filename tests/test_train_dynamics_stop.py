# SPDX-License-Identifier: Apache-2.0
"""Dynamics trainer stop and resume guards: SIGTERM under torchrun with DataLoader workers, the
exit code of a stopped run, unfinished or moved-on init sources, changed file pools."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

import coupled_wm.world_model.train_dynamics as td
from conftest import write_h5
from coupled_wm.common.checkpoint import load_checkpoint
from coupled_wm.common.config import save_config
from coupled_wm.world_model.train_dynamics import train
from test_train_dynamics import DATA_KW, _deep_update, assets, tiny_cfg, write_pool  # noqa: F401  (fixture)


def _main(cfg: dict, path) -> None:
    save_config(cfg, path)
    td.main(["--config", str(path)])


def _ckpt(run_dir) -> dict:
    return load_checkpoint(run_dir / "latest.pt", mmap=False)


def _wait_for(path, text: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and text in path.read_text():
            return
        if proc.poll() is not None:
            raise AssertionError(f"trainer exited early ({proc.returncode}):\n{path.read_text()[-3000:]}")
        time.sleep(0.2)
    proc.kill()
    raise AssertionError(f"{text!r} not seen in {timeout} s:\n{path.read_text()[-3000:]}")


def test_torchrun_sigterm_checkpoints_with_loader_workers(assets, tmp_path):  # noqa: F811
    """torchrun forwards SIGTERM to the whole process group, DataLoader workers included: the
    workers must survive it so that the trainer finishes the update and writes latest.pt."""
    run_dir = tmp_path / "run"
    cfg = tiny_cfg(assets, run_dir, data={"num_workers": 2},
                   schedule={"total_updates": 100000, "max_updates": 100000},
                   checkpoint={"save_every_updates": 100000}, logging={"print_every_updates": 1},
                   dist={"backend": "gloo"})
    cfg_path = tmp_path / "cfg.yaml"
    save_config(cfg, cfg_path)
    log_path = tmp_path / "train.log"
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
             "-m", "coupled_wm.world_model.train_dynamics", "--config", str(cfg_path)],
            stdout=log_file, stderr=subprocess.STDOUT, env=env)
        try:
            _wait_for(log_path, "update 3/", proc, timeout=180)
            proc.send_signal(signal.SIGTERM)  # to the torchrun agent, as a scheduler would
            proc.wait(timeout=120)
        finally:
            if proc.poll() is None:
                proc.kill()
    out = log_path.read_text()
    assert "stop requested" in out, out[-3000:]
    ckpt = load_checkpoint(run_dir / "latest.pt", mmap=False)
    assert ckpt["counters"]["stage_update"] >= 3
    assert not list(run_dir.glob(".latest.pt*.tmp"))


def test_stopped_stage_exits_nonzero_and_is_not_a_valid_init(assets, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(td, "setup_logging", lambda rank: None)
    real_update = td._Run._update

    def update_then_signal(self):
        if self.counters["stage_update"] == 1:
            os.kill(os.getpid(), signal.SIGUSR1)  # as a scheduler's early-warning signal
        return real_update(self)

    bb = tmp_path / "bb"
    backbone = tiny_cfg(assets, bb, schedule={"max_updates": 4})
    monkeypatch.setattr(td._Run, "_update", update_then_signal)
    with pytest.raises(SystemExit) as stopped:
        _main(backbone, tmp_path / "bb.yaml")
    assert stopped.value.code == td.EXIT_STOPPED
    assert _ckpt(bb)["counters"]["stage_update"] == 2
    monkeypatch.setattr(td._Run, "_update", real_update)

    reward = tiny_cfg(assets, tmp_path / "rw", stage="reward", init_from=str(bb / "latest.pt"),
                      schedule={"max_updates": 1})
    with pytest.raises(ValueError, match="stopped at stage update 2 of its schedule.max_updates 4"):
        train(reward)
    assert not (tmp_path / "rw" / "latest.pt").exists()
    train(_deep_update(reward, {"run_dir": str(tmp_path / "rw_anyway"), "init_from_allow_unfinished": True}))

    _main(backbone, tmp_path / "bb.yaml")  # finishes: no SystemExit
    assert _ckpt(bb)["counters"]["stage_update"] == 4
    train(reward)
    train(_deep_update(reward, {"run_dir": str(tmp_path / "rw_dir"), "init_from": str(bb / "pretrained")}))


def test_resume_refuses_an_init_source_that_moved_on(assets, tmp_path):  # noqa: F811
    bb, rw = tmp_path / "bb", tmp_path / "rw"
    train(tiny_cfg(assets, bb, schedule={"max_updates": 2}))
    reward = tiny_cfg(assets, rw, stage="reward", init_from=str(bb / "latest.pt"), schedule={"max_updates": 2})
    train(_deep_update(reward, {"schedule": {"max_updates": 1}}))
    train(tiny_cfg(assets, bb, schedule={"max_updates": 3}))  # the backbone is trained further in place
    with pytest.raises(ValueError, match="is now at update 3"):
        train(reward)
    train(_deep_update(reward, {"init_from": None}))  # resume the reward run as it is
    ckpt = _ckpt(rw)
    assert ckpt["counters"]["stage_update"] == 2 and ckpt["init"]["global_update"] == 2


def test_resume_refuses_a_changed_file_pool(assets, tmp_path):  # noqa: F811
    data = write_pool(tmp_path / "data")
    run_dir = tmp_path / "bb"
    cfg = tiny_cfg(assets, run_dir, data={"dirs": [str(data)]}, schedule={"max_updates": 3})
    train(cfg)
    assert _ckpt(run_dir)["data_state"] == {"short": {"epoch": 0, "position": 6}, "long": {"epoch": 0, "position": 2}}

    moved = shutil.copytree(data, tmp_path / "moved")  # same files elsewhere: still the same pool
    cfg = _deep_update(cfg, {"data": {"dirs": [str(moved)]}, "schedule": {"max_updates": 4}})
    train(cfg)
    assert _ckpt(run_dir)["data_state"]["short"] == {"epoch": 0, "position": 8}

    write_h5(moved / "chunk_0002.h5", 12, file_id=2, **DATA_KW)
    cfg = _deep_update(cfg, {"schedule": {"max_updates": 5}})
    with pytest.raises(ValueError, match=r"data pool changed .*2 -> 3 files; added \['chunk_0002.h5'\]"):
        train(cfg)
    train(_deep_update(cfg, {"data": {"allow_pool_change": True}}))
    ckpt = _ckpt(run_dir)  # update 4 (long clips) restarted both passes of epoch 0 on the new pool
    assert ckpt["data_state"] == {"short": {"epoch": 0, "position": 2}, "long": {"epoch": 0, "position": 2}}
    assert [name for name, _ in ckpt["data_pool"]["files"]] == ["chunk_0000.h5", "chunk_0001.h5", "chunk_0002.h5"]
    train(_deep_update(cfg, {"schedule": {"max_updates": 6}}))  # the new pool is now the saved one
