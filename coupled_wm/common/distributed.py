# SPDX-License-Identifier: Apache-2.0
"""Process-group setup for torchrun, SLURM ``srun`` and single-process launches.

Launcher detection order:

1. torchrun (or any launcher exporting ``RANK``/``WORLD_SIZE``/``LOCAL_RANK`` and
   ``MASTER_ADDR``/``MASTER_PORT``).
2. A SLURM job step started by ``srun`` (``SLURM_STEP_ID`` is set): rank, local rank and
   world size come from ``SLURM_PROCID``/``SLURM_LOCALID``/``SLURM_STEP_NUM_TASKS``
   (``SLURM_NTASKS`` as a fallback). ``MASTER_ADDR`` defaults to the first host of the
   step's node list and ``MASTER_PORT`` to ``10000 + job_id % 20000``.
   A plain ``python`` inside an ``sbatch`` script or ``salloc`` shell is NOT a job step,
   so it runs as a single process even if the allocation has several tasks.
3. Otherwise a single process (world size 1).

CUDA device: the local rank, except that a process which sees exactly one GPU (per-task GPU
binding, e.g. ``srun --gpus-per-task=1``) uses device 0, and a launcher that exports no local
rank gets ``rank % device_count``. A local rank beyond the visible devices raises.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from itertools import product

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)

_DIST_INFO: "DistInfo | None" = None
_PENDING_REPORT: str | None = None


@dataclass
class DistInfo:
    rank: int
    local_rank: int  # index of this process's CUDA device (the launcher's local rank on CPU)
    world_size: int
    device: torch.device
    distributed: bool  # True when a torch.distributed process group is initialised

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def expand_hostlist(hostlist: str) -> list[str]:
    """Expand a SLURM host list such as ``"node[001-003,010],gpu7"`` into host names."""
    hosts: list[str] = []
    for item in _split_top_level(hostlist):
        parts = re.split(r"(\[[^\]]*\])", item)
        choices = []
        for part in parts:
            if part.startswith("[") and part.endswith("]"):
                choices.append(_expand_range(part[1:-1]))
            elif part:
                choices.append([part])
        hosts.extend("".join(p) for p in product(*choices))
    return hosts


def _split_top_level(s: str) -> list[str]:
    items, depth, cur = [], 0, []
    for ch in s.strip():
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        items.append("".join(cur))
    return [i for i in (x.strip() for x in items) if i]


def _expand_range(spec: str) -> list[str]:
    out = []
    for token in spec.split(","):
        token = token.strip()
        if "-" in token:
            lo, hi = token.split("-", 1)
            width = len(lo)
            out.extend(str(i).zfill(width) for i in range(int(lo), int(hi) + 1))
        else:
            out.append(token)
    return out


def _first_slurm_host() -> str:
    nodelist = os.environ.get("SLURM_STEP_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return "127.0.0.1"
    try:
        out = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.split()
        if out:
            return out[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return expand_hostlist(nodelist)[0]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def detect_launch() -> tuple[str, int, int | None, int]:
    """Return ``(launcher, rank, local_rank, world_size)`` from the environment.

    ``launcher`` is ``"torchrun"``, ``"slurm"`` or ``"single"``; ``local_rank`` is ``None``
    when the launcher does not export one. For SLURM this also sets
    ``MASTER_ADDR``/``MASTER_PORT`` when they are not already exported.
    """
    env = os.environ
    if "RANK" in env and "WORLD_SIZE" in env:
        rank = int(env["RANK"])
        world = int(env["WORLD_SIZE"])
        local_rank = int(env["LOCAL_RANK"]) if "LOCAL_RANK" in env else None
        return "torchrun", rank, local_rank, world
    if "SLURM_STEP_ID" in env and "SLURM_PROCID" in env:
        rank = int(env["SLURM_PROCID"])
        local_rank = int(env["SLURM_LOCALID"]) if "SLURM_LOCALID" in env else None
        world = int(env.get("SLURM_STEP_NUM_TASKS") or env.get("SLURM_NTASKS") or 1)
        if world > 1:
            env.setdefault("MASTER_ADDR", _first_slurm_host())
            job_id = int(env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID") or 0)
            env.setdefault("MASTER_PORT", str(10000 + job_id % 20000))
        return "slurm", rank, local_rank, world
    return "single", 0, 0, 1


def local_world_size() -> int:
    """Processes of this launch on this node: ``LOCAL_WORLD_SIZE`` (torchrun), else the tasks of
    this node in the SLURM job step (``SLURM_STEP_TASKS_PER_NODE``), else 1."""
    env = os.environ
    if "LOCAL_WORLD_SIZE" in env:
        return int(env["LOCAL_WORLD_SIZE"])
    if "SLURM_STEP_ID" in env and env.get("SLURM_STEP_TASKS_PER_NODE"):
        per_node = []
        for item in env["SLURM_STEP_TASKS_PER_NODE"].split(","):  # e.g. "4(x2),3"
            m = re.fullmatch(r"(\d+)(?:\(x(\d+)\))?", item.strip())
            if m is None:
                return 1
            per_node.extend([int(m.group(1))] * int(m.group(2) or 1))
        node = int(env.get("SLURM_NODEID", 0))
        return per_node[node] if node < len(per_node) else per_node[0]
    return 1


def cuda_device_index(launcher: str, rank: int, local_rank: int | None) -> int:
    """The CUDA device this process uses (see the module docstring)."""
    count = torch.cuda.device_count()
    if count == 1:
        return 0
    if local_rank is None:
        return rank % count
    if local_rank >= count:
        source = "LOCAL_RANK" if launcher == "torchrun" else "SLURM_LOCALID"
        raise ValueError(
            f"local rank {local_rank} (from {source}) but this process sees {count} CUDA devices "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}); start at most {count} "
            "processes per node, or bind one GPU per process"
        )
    return local_rank


def init_distributed(
    require_process_group: bool = False,
    backend: str = "nccl",
    timeout_minutes: int = 30,
) -> DistInfo:
    """Detect the launcher, set the CUDA device and initialise the process group.

    A process group is created whenever ``world_size > 1``. With ``world_size == 1`` it is
    created only if ``require_process_group`` (e.g. FSDP): through torchrun's rendezvous when
    launched by torchrun, otherwise as a 1-rank group on 127.0.0.1.
    The ``nccl`` backend falls back to ``gloo`` when CUDA is unavailable.

    The launch summary is logged here if logging is configured, otherwise by the next
    :func:`coupled_wm.common.logging.setup_logging` call.
    """
    global _DIST_INFO, _PENDING_REPORT
    launcher, rank, local_rank, world = detect_launch()

    if torch.cuda.is_available():
        local_rank = cuda_device_index(launcher, rank, local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0 if local_rank is None else local_rank
        device = torch.device("cpu")
        if backend == "nccl":
            backend = "gloo"

    timeout = timedelta(minutes=timeout_minutes)
    if not dist.is_initialized():
        # torchrun serves its own rendezvous store at MASTER_ADDR/PORT, also for one process.
        torchrun_store = launcher == "torchrun" and "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ
        if world > 1 or (require_process_group and torchrun_store):
            dist.init_process_group(
                backend=backend, init_method="env://", rank=rank, world_size=world, timeout=timeout,
            )
        elif require_process_group:
            dist.init_process_group(
                backend=backend, init_method=f"tcp://127.0.0.1:{_free_port()}",
                rank=0, world_size=1, timeout=timeout,
            )

    _DIST_INFO = DistInfo(
        rank=rank, local_rank=local_rank, world_size=world, device=device,
        distributed=dist.is_initialized(),
    )
    _PENDING_REPORT = (
        f"launcher={launcher} rank={rank} local_rank={local_rank} world_size={world} device={device} "
        f"process_group={_DIST_INFO.distributed} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )
    if logging.getLogger().hasHandlers():
        report_launch()
    return _DIST_INFO


def report_launch() -> None:
    """Log the summary of the last :func:`init_distributed` call, once."""
    global _PENDING_REPORT
    if _PENDING_REPORT is not None:
        log.info("%s", _PENDING_REPORT)
        _PENDING_REPORT = None


def get_dist_info() -> DistInfo:
    """The ``DistInfo`` of the last ``init_distributed`` call (single CPU process if none)."""
    if _DIST_INFO is None:
        return DistInfo(0, 0, 1, torch.device("cpu"), False)
    return _DIST_INFO


def is_main() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return get_dist_info().rank == 0


def barrier() -> None:
    if not dist.is_initialized():
        return
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def cleanup() -> None:
    global _DIST_INFO
    if dist.is_initialized():
        dist.destroy_process_group()
    _DIST_INFO = None
