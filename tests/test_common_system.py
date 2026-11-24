# SPDX-License-Identifier: Apache-2.0
"""Stop signals and the DataLoader workers that ignore them, host memory and the torch.compile
toolchain check."""

import ctypes
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from coupled_wm.common import system

REPO = Path(__file__).resolve().parents[1]


def test_stop_signals_are_recorded_and_the_handlers_restored():
    before = {name: signal.getsignal(getattr(signal, name)) for name in system.STOP_SIGNALS}
    with system.StopSignals() as stop:
        assert stop.received is None
        os.kill(os.getpid(), signal.SIGUSR1)
        assert stop.received == signal.SIGUSR1
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop.received == signal.SIGTERM
    assert {name: signal.getsignal(getattr(signal, name)) for name in system.STOP_SIGNALS} == before


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="needs SIGUSR1")
def test_poll_takes_the_handlers_back_from_a_c_level_handler():
    """A library that installs its own C handler (as a CUDA kernel build under torch.compile does
    for SIGUSR1) hides the signal from Python, and signal.getsignal does not show it; poll()
    installs the Python handlers again."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
    libc.signal.restype = ctypes.c_void_p
    with system.StopSignals() as stop:
        libc.signal(signal.SIGUSR1, 1)  # SIG_IGN, behind Python's back
        assert signal.getsignal(signal.SIGUSR1) == stop._handle
        os.kill(os.getpid(), signal.SIGUSR1)
        assert stop.poll() is None  # lost: the foreign handler held the signal
        os.kill(os.getpid(), signal.SIGUSR1)
        assert stop.poll() == signal.SIGUSR1


def test_dataloader_workers_ignore_the_stop_signals():
    before = {name: signal.getsignal(getattr(signal, name)) for name in system.STOP_SIGNALS}
    try:
        system.ignore_stop_signals(0)
        assert all(signal.getsignal(getattr(signal, name)) is signal.SIG_IGN for name in system.STOP_SIGNALS)
    finally:
        for name, handler in before.items():
            signal.signal(getattr(signal, name), handler)


_WORKER_INITS = []


def _record_worker_init(worker_id: int) -> None:
    _WORKER_INITS.append(worker_id)


class _WorkerState(Dataset):
    """Item: whether the worker ignores each stop signal, and how many times ``_record_worker_init`` ran in it."""

    def __len__(self):
        return 64

    def __getitem__(self, i):
        ignored = [signal.getsignal(getattr(signal, name)) is signal.SIG_IGN for name in system.STOP_SIGNALS]
        return torch.tensor([*ignored, len(_WORKER_INITS)])


@pytest.mark.parametrize("worker_init_fn, inits", [(None, 0), (_record_worker_init, 1)])
def test_signal_safe_loader_workers_survive_the_stop_signals(worker_init_fn, inits):
    loader = system.SignalSafeDataLoader(_WorkerState(), batch_size=None, num_workers=2,
                                         worker_init_fn=worker_init_fn)
    it = iter(loader)
    rows = [next(it), next(it)]  # one from each worker: both have run their init
    workers = list(it._workers)
    for w in workers:
        for name in system.STOP_SIGNALS:
            os.kill(w.pid, getattr(signal, name))
    rows += list(it)  # a worker killed by the signal would fail this with "DataLoader worker ... killed by signal"
    assert len(rows) == len(loader.dataset)
    assert all(row.tolist() == [1] * len(system.STOP_SIGNALS) + [inits] for row in rows)
    assert not any(w.is_alive() for w in workers)


def test_signal_safe_loader_without_workers_is_a_plain_loader():
    loader = system.SignalSafeDataLoader(range(4), batch_size=2)
    assert loader.worker_init_fn is None
    assert [b.tolist() for b in loader] == [[0, 1], [2, 3]]


def test_stop_loader_workers():
    it = iter(system.SignalSafeDataLoader(range(64), batch_size=2, num_workers=2, persistent_workers=True))
    next(it)
    workers = list(it._workers)
    assert all(w.is_alive() for w in workers)
    system.stop_loader_workers(it)
    assert not any(w.is_alive() for w in workers)
    system.stop_loader_workers(it)
    system.stop_loader_workers(iter(range(3)))


_EXIT_SCRIPT = """
import sys
from coupled_wm.common.system import SignalSafeDataLoader

def main():
    persistent = sys.argv[1] == "persistent"
    it = iter(SignalSafeDataLoader(range(64), batch_size=2, num_workers=2, persistent_workers=persistent))
    if persistent:
        list(it)  # a persistent loader keeps its workers after the pass
    else:
        next(it)
    print(" ".join(str(w.pid) for w in it._workers), flush=True)
    if sys.argv[1] == "raise":
        raise RuntimeError("uncaught")
    return it

it = main()
"""


@pytest.mark.parametrize("mode, returncode", [("held", 0), ("persistent", 0), ("raise", 1)])
def test_a_process_holding_a_worker_iterator_exits(mode, returncode):
    """Workers that ignore SIGTERM must not keep the process from exiting: at exit, multiprocessing
    terminates daemonic children with SIGTERM and joins them. The iterator outlives the script in
    a global, in a persistent loader after its pass, or in the frames of an uncaught exception."""
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, "-c", _EXIT_SCRIPT, mode], env=env, capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == returncode, proc.stderr
    assert "Error in atexit" not in proc.stderr and "Exception ignored" not in proc.stderr, proc.stderr
    for pid in map(int, proc.stdout.split()):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


_PINNED_EXIT_SCRIPT = """
import atexit
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as It
from coupled_wm.common.system import SignalSafeDataLoader

def torch_handler(w):
    print(int(w.is_alive()), flush=True)
    It._clean_up_worker(w)

def pinned_init(self, loader, _init=It.__init__):  # torch's branch for persistent workers and pinned memory
    _init(self, loader)
    for w in self._workers:
        atexit.register(torch_handler, w)

first = iter(SignalSafeDataLoader(range(8), batch_size=2, num_workers=1))
next(first)
It.__init__ = pinned_init
second = iter(SignalSafeDataLoader(range(8), batch_size=2, num_workers=2, persistent_workers=True))
next(second)
"""


def test_workers_are_stopped_before_torchs_exit_handlers():
    """With pinned memory, torch registers an exit handler for each worker of a persistent iterator
    when it creates the iterator: it waits 5 s for the worker, then sends it SIGTERM. The workers of
    an iterator created after the first must already be stopped when that handler runs."""
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, "-c", _PINNED_EXIT_SCRIPT], env=env, capture_output=True, text=True,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["0", "0"], proc.stdout  # 1: a live worker, which torch's handler waits for


def test_available_memory_reads_this_host():
    avail = system.available_memory()
    assert avail is None or avail > 0


def test_compile_toolchain_problem(tmp_path, monkeypatch):
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(system.shutil, "which", lambda name: None)
    assert "no C compiler" in system.compile_toolchain_problem()
    monkeypatch.setenv("CC", "cc")
    monkeypatch.setattr(system.sysconfig, "get_paths", lambda scheme=None: {"include": str(tmp_path)})
    assert "Python.h" in system.compile_toolchain_problem()
    (tmp_path / "Python.h").touch()
    assert system.compile_toolchain_problem() is None


@pytest.mark.parametrize("device, main, warns", [("cuda", True, True), ("cpu", True, False), ("cuda", False, False)])
def test_compile_warning_on_the_main_cuda_process_only(monkeypatch, caplog, device, main, warns):
    monkeypatch.setattr(system, "compile_toolchain_problem", lambda: "no C compiler ($CC, gcc or clang)")
    monkeypatch.setattr(system, "is_main", lambda: main)
    with caplog.at_level(logging.WARNING, logger="coupled_wm.common.system"):
        system.warn_if_compile_unavailable("train.compile", "--set train.compile=false", device)
    messages = [r.getMessage() for r in caplog.records]
    assert bool(messages) == warns
    if warns:
        assert "train.compile" in messages[0] and "--set train.compile=false" in messages[0]
