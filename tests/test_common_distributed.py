# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from coupled_wm.common import distributed as cd

_LAUNCH_VARS = [
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "SLURM_PROCID",
    "SLURM_LOCALID", "SLURM_NTASKS", "SLURM_STEP_ID", "SLURM_STEP_NUM_TASKS", "SLURM_STEP_NODELIST",
    "SLURM_JOB_NODELIST", "SLURM_JOB_ID", "SLURM_JOBID", "SLURM_STEP_TASKS_PER_NODE", "SLURM_NODEID",
]


@pytest.fixture
def clean_env(monkeypatch):
    for v in _LAUNCH_VARS:
        monkeypatch.setenv(v, "unset")  # record the original value so teardown restores it
        monkeypatch.delenv(v)
    return monkeypatch


def test_single_process(clean_env):
    assert cd.detect_launch() == ("single", 0, 0, 1)
    info = cd.init_distributed()
    assert (info.rank, info.local_rank, info.world_size, info.distributed) == (0, 0, 1, False)
    assert info.is_main and cd.is_main()
    if not torch.cuda.is_available():
        assert info.device == torch.device("cpu")
    cd.barrier()
    cd.cleanup()


def test_torchrun_vars(clean_env):
    clean_env.setenv("RANK", "5")
    clean_env.setenv("LOCAL_RANK", "1")
    clean_env.setenv("WORLD_SIZE", "8")
    clean_env.setenv("SLURM_PROCID", "0")  # torchrun inside an allocation: torchrun wins
    clean_env.setenv("SLURM_STEP_ID", "0")
    assert cd.detect_launch() == ("torchrun", 5, 1, 8)


def test_launcher_without_local_rank(clean_env):
    clean_env.setenv("RANK", "6")
    clean_env.setenv("WORLD_SIZE", "8")
    assert cd.detect_launch() == ("torchrun", 6, None, 8)


def test_cuda_device_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert cd.cuda_device_index("slurm", 6, 2) == 0  # srun --gpus-per-task=1: each task sees one GPU
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert cd.cuda_device_index("torchrun", 5, 1) == 1
    assert cd.cuda_device_index("torchrun", 6, None) == 2  # no LOCAL_RANK: not the global rank
    with pytest.raises(ValueError, match="SLURM_LOCALID.*4 CUDA devices"):
        cd.cuda_device_index("slurm", 5, 5)


def test_per_task_gpu_binding_uses_device_0(clean_env, monkeypatch):
    clean_env.setenv("SLURM_STEP_ID", "0")
    clean_env.setenv("SLURM_PROCID", "0")
    clean_env.setenv("SLURM_LOCALID", "3")
    clean_env.setenv("SLURM_NTASKS", "1")
    used = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", used.append)
    info = cd.init_distributed()
    try:
        assert used == [0] and info.local_rank == 0 and info.device == torch.device("cuda", 0)
    finally:
        cd.cleanup()


def test_local_world_size(clean_env):
    assert cd.local_world_size() == 1
    clean_env.setenv("SLURM_STEP_TASKS_PER_NODE", "4(x2),3")
    assert cd.local_world_size() == 1  # not inside a job step
    clean_env.setenv("SLURM_STEP_ID", "0")
    assert cd.local_world_size() == 4
    clean_env.setenv("SLURM_NODEID", "2")
    assert cd.local_world_size() == 3
    clean_env.setenv("LOCAL_WORLD_SIZE", "8")
    assert cd.local_world_size() == 8


def test_slurm_step(clean_env, monkeypatch):
    clean_env.setenv("SLURM_STEP_ID", "0")
    clean_env.setenv("SLURM_PROCID", "6")
    clean_env.setenv("SLURM_LOCALID", "2")
    clean_env.setenv("SLURM_NTASKS", "16")
    clean_env.setenv("SLURM_STEP_NUM_TASKS", "8")
    clean_env.setenv("SLURM_STEP_NODELIST", "gpu[012-013]")
    clean_env.setenv("SLURM_JOB_NODELIST", "gpu[001-013]")
    clean_env.setenv("SLURM_JOB_ID", "123456")

    def no_scontrol(*args, **kwargs):
        raise FileNotFoundError("scontrol")

    monkeypatch.setattr(subprocess, "run", no_scontrol)
    assert cd.detect_launch() == ("slurm", 6, 2, 8)
    assert os.environ["MASTER_ADDR"] == "gpu012"
    assert os.environ["MASTER_PORT"] == str(10000 + 123456 % 20000)


def test_slurm_uses_scontrol_and_keeps_exported_master(clean_env, monkeypatch):
    clean_env.setenv("SLURM_STEP_ID", "3")
    clean_env.setenv("SLURM_PROCID", "0")
    clean_env.setenv("SLURM_NTASKS", "4")
    clean_env.setenv("SLURM_STEP_NODELIST", "gpu[1-2]")
    clean_env.setenv("MASTER_PORT", "29511")

    class Out:
        stdout = "gpu1\ngpu2\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Out())
    assert cd.detect_launch() == ("slurm", 0, None, 4)  # no SLURM_LOCALID exported
    assert os.environ["MASTER_ADDR"] == "gpu1" and os.environ["MASTER_PORT"] == "29511"


def test_sbatch_script_without_srun_is_single(clean_env):
    clean_env.setenv("SLURM_PROCID", "0")
    clean_env.setenv("SLURM_LOCALID", "0")
    clean_env.setenv("SLURM_NTASKS", "4")
    clean_env.setenv("SLURM_JOB_ID", "1")
    assert cd.detect_launch() == ("single", 0, 0, 1)


@pytest.mark.parametrize("hostlist,expected", [
    ("login06", ["login06"]),
    ("node[001-003,010]", ["node001", "node002", "node003", "node010"]),
    ("a[1-2],b7,c[08-09]", ["a1", "a2", "b7", "c08", "c09"]),
    ("r[1-2]n[3-4]", ["r1n3", "r1n4", "r2n3", "r2n4"]),
])
def test_expand_hostlist(hostlist, expected):
    assert cd.expand_hostlist(hostlist) == expected


def test_one_rank_group_when_required(clean_env):
    info = cd.init_distributed(require_process_group=True, backend="gloo", timeout_minutes=1)
    try:
        assert info.distributed and dist.is_initialized() and dist.get_world_size() == 1
        t = torch.ones(2)
        dist.all_reduce(t)
        assert torch.equal(t, torch.ones(2))
        cd.barrier()
    finally:
        cd.cleanup()
    assert not dist.is_initialized()


def test_one_rank_group_under_torchrun(tmp_path):
    """torchrun with one process: the group must use torchrun's rendezvous store (a private
    127.0.0.1 store would wait forever for torchrun's agent store)."""
    script = tmp_path / "one_rank.py"
    script.write_text(textwrap.dedent("""
        import torch, torch.distributed as dist
        from coupled_wm.common import distributed as cd
        info = cd.init_distributed(require_process_group=True, backend="gloo", timeout_minutes=1)
        t = torch.ones(1)
        dist.all_reduce(t)
        print("OK", info.world_size, dist.get_world_size(), float(t))
        cd.cleanup()
    """))
    env = {k: v for k, v in os.environ.items() if k not in _LAUNCH_VARS}
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")])
    out = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", str(script)],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK 1 1 1.0" in out.stdout
