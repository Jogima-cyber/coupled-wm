# SPDX-License-Identifier: Apache-2.0
import logging
import sys
import time
import types

import pytest

from coupled_wm.common import distributed as cd
from coupled_wm.common.logging import ExperimentLogger, setup_logging


def test_rank0_writes_tensorboard(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # importing wandb would fail
    lg = ExperimentLogger(tmp_path / "tb", rank=0, config={"a": 1, "b": {"c": (1, 2)}}, wandb_cfg={"enabled": False})
    lg.scalar("loss", 0.5, 1)
    lg.scalars({"x": 1, "y": 2.0}, step=2, prefix="train/")
    lg.close()
    assert any(p.name.startswith("events.out.tfevents") for p in (tmp_path / "tb").iterdir())


def test_other_ranks_are_silent(tmp_path):
    lg = ExperimentLogger(tmp_path / "tb", rank=1, config={"a": 1})
    lg.scalar("loss", 0.5, 1)
    lg.close()
    assert not lg.active
    assert not (tmp_path / "tb").exists()


def test_wandb_enabled_but_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(ImportError, match="wandb.enabled"):
        ExperimentLogger(tmp_path / "tb", rank=0, wandb_cfg={"enabled": True})


@pytest.fixture
def fake_wandb(monkeypatch):
    calls = []

    class Run:
        id = "abc"

        def finish(self):
            pass

    def init(**kwargs):
        calls.append(kwargs)
        return Run()

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))
    return calls


def test_wandb_writes_next_to_the_tensorboard_dir(tmp_path, fake_wandb):
    """wandb's files must not sit inside the dir sync_tensorboard watches (it would re-sync its own copies)."""
    run_dir = tmp_path / "rssm_g1"
    ExperimentLogger(run_dir / "tb", rank=0, wandb_cfg={"enabled": True}).close()
    ExperimentLogger(run_dir / "tb", rank=0, wandb_cfg={"enabled": True, "name": "cfg-name"}).close()
    ExperimentLogger(run_dir / "tb", rank=0, wandb_cfg={"enabled": True}, run_name="seed20").close()
    assert all(c["dir"] == str(run_dir) and c["sync_tensorboard"] for c in fake_wandb)
    assert [c["name"] for c in fake_wandb] == ["rssm_g1", "cfg-name", "seed20"]


def test_purge_step_drops_steps_logged_after_the_last_checkpoint(tmp_path):
    from tensorboard.backend.event_processing import plugin_event_accumulator

    lg = ExperimentLogger(tmp_path / "tb", rank=0, purge_step=0)
    for step in range(10):
        lg.scalar("loss", step, step)
    lg.close()
    time.sleep(1.1)  # event file names carry a timestamp in seconds
    lg = ExperimentLogger(tmp_path / "tb", rank=0, purge_step=5)  # resumed from a checkpoint at step 5
    for step in range(5, 8):
        lg.scalar("loss", 100 + step, step)
    lg.close()
    acc = plugin_event_accumulator.EventAccumulator(str(tmp_path / "tb"))
    acc.Reload()
    events = acc.Tensors("loss")
    assert [e.step for e in events] == list(range(8))
    assert float(events[5].tensor_proto.float_val[0]) == 105.0


def test_launch_summary_survives_late_logging_setup(monkeypatch, capsys):
    """Entry points call init_distributed before setup_logging; the launch line must still appear."""
    for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_STEP_ID", "SLURM_PROCID"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    cd.init_distributed()
    try:
        setup_logging(rank=0)
        out = capsys.readouterr().out
        assert "launcher=single rank=0 local_rank=0 world_size=1" in out
        setup_logging(rank=0)
        assert "launcher=" not in capsys.readouterr().out  # reported once
    finally:
        cd.cleanup()


def test_setup_logging_levels():
    setup_logging(rank=0)
    assert logging.getLogger().level == logging.INFO
    setup_logging(rank=3)
    assert logging.getLogger().level == logging.WARNING
    setup_logging(rank=0)
