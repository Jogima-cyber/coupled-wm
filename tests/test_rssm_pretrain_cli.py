# SPDX-License-Identifier: Apache-2.0
"""RSSM pretraining through its command line: one process, and two processes under torchrun (gloo, CPU)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from coupled_wm.rssm import load_rssm
from coupled_wm.rssm.pretrain import train
from test_rssm_pretrain import TINY_MODEL, tiny_cfg, write_episode_file

REPO = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in [k for k in env if k.startswith("SLURM_")] + ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        env.pop(key, None)
    return env


def _write_config(tmp_path: Path, run_dir: Path | str, **extra) -> tuple[Path, Path]:
    data = tmp_path / "pool"
    for i in range(3):
        write_episode_file(data / f"chunk_{i:04d}.h5", 30, dones_at=[], file_id=i)
    cfg = {**tiny_cfg(run_dir, data, epochs=2), **extra}
    path = tmp_path / "rssm.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path, data


def _run(args: list[str], ok: bool = True, timeout: int = 600) -> str:
    proc = subprocess.run(args, env=_env(), cwd=REPO, capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    assert (proc.returncode == 0) == ok, out[-6000:]
    return out


def _torchrun(*args: str) -> list[str]:
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
            "-m", "coupled_wm.rssm.pretrain", *args]


def test_module_cli_with_overrides(tmp_path):
    run = tmp_path / "run"
    cfg_path, _ = _write_config(tmp_path, run)
    out = _run([sys.executable, "-m", "coupled_wm.rssm.pretrain", "--config", str(cfg_path),
                "--set", "train.epochs=1", "--set", "train.kl_free_nats=0.5"])
    assert "launcher=single rank=0 local_rank=0 world_size=1 device=cpu" in out  # logged before setup
    assert f"new run in {run}" in out and "training on the CPU" in out
    stored = json.loads((run / "rssm_e1" / "config.json").read_text())
    assert stored["provenance"]["loss"]["kl_free_nats"] == 0.5 and stored["provenance"]["world_size"] == 1
    assert not (run / "rssm_e2").exists()
    load_rssm(run / "rssm_e1", config=TINY_MODEL)


def test_torchrun_two_processes_then_resume_on_one(tmp_path):
    run = tmp_path / "run"
    cfg_path, data = _write_config(tmp_path, run)
    _run(_torchrun("--config", str(cfg_path), "--set", "train.max_steps=20"))
    ckpt = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    spe = ckpt["state"]["steps_per_epoch"]
    assert spe == 3 * (30 - 8) // 4 and ckpt["state"]["global_step"] == 20
    assert len(ckpt["rng"]) == 2
    assert not torch.equal(ckpt["rng"][0]["torch"], ckpt["rng"][1]["torch"])  # per-rank sampling streams
    stored = json.loads((run / "rssm_e1" / "config.json").read_text())
    assert stored["provenance"]["world_size"] == 2 and stored["provenance"]["batch_size"] == 4
    load_rssm(run / "rssm_e1", config=TINY_MODEL)

    train(tiny_cfg(run, data, epochs=2))  # same global batches on one process; RNG reseeded
    ckpt = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert ckpt["state"] == {"epoch": 2, "step": 0, "global_step": 2 * spe, "steps_per_epoch": spe}
    assert len(ckpt["rng"]) == 1
    load_rssm(run / "rssm_e2", config=TINY_MODEL)


def test_torchrun_validation_and_cache(tmp_path):
    """Held-out data is built and evaluated on rank 0 only, in batches of the per-process training batch; every
    rank still takes part in the validation barriers and in the cache memory check."""
    run = tmp_path / "run"
    cfg_path, _ = _write_config(tmp_path, run, validation={"holdout_every": 3, "num_batches": 2, "every_steps": 3})
    out = _run(_torchrun("--config", str(cfg_path), "--set", "train.epochs=1", "--set", "data.cache=true"))
    assert "RAM frame cache:" in out and "for 2 process(es)" in out
    assert "validation: 8 held-out windows in batches of 2 on rank 0" in out  # the per-process batch (4 / 2)
    assert "validation at step 3:" in out
    stored = json.loads((run / "rssm_e1" / "config.json").read_text())
    assert stored["data"]["validation_files"] == ["chunk_0002.h5"] and stored["data"]["validation_windows"] == 22
    assert math.isfinite(stored["provenance"]["validation_loss"]) and stored["provenance"]["world_size"] == 2


def test_torchrun_ranks_that_see_different_run_dirs_stop_with_an_error(tmp_path):
    """A run dir that is not shared by all ranks (here one per rank) is refused on relaunch,
    instead of ranks resuming at different steps and blocking in a collective."""
    cfg_path, _ = _write_config(tmp_path, str(tmp_path / "run_${RANK}"))
    _run(_torchrun("--config", str(cfg_path), "--set", "train.max_steps=2"))
    assert (tmp_path / "run_0" / "latest.pt").is_file() and not (tmp_path / "run_1" / "latest.pt").exists()
    out = _run(_torchrun("--config", str(cfg_path), "--set", "train.max_steps=4"), ok=False, timeout=300)
    assert "the ranks see different runs" in out and "shared by all ranks" in out
