# SPDX-License-Identifier: Apache-2.0
"""Process-level helpers shared by the trainers and the environment: stop signals and the
DataLoader workers that ignore them, host memory and the run-time requirements of
``torch.compile``."""

from __future__ import annotations

import atexit
import functools
import logging
import os
import shutil
import signal
import sysconfig
import threading
import weakref
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import DataLoader

from coupled_wm.common.distributed import is_main

log = logging.getLogger(__name__)

STOP_SIGNALS = ("SIGTERM", "SIGUSR1")
"""Signals that ask a trainer to checkpoint and exit at its next safe point."""


class StopSignals:
    """Records :data:`STOP_SIGNALS` while active (main thread only); restores the previous handlers.

    A trainer calls :meth:`poll` between updates, so a signal never interrupts one.
    """

    def __init__(self):
        self.received: int | None = None
        self._previous: dict[int, Any] = {}

    def _handle(self, signum, frame):
        self.received = signum

    def __enter__(self) -> "StopSignals":
        if threading.current_thread() is threading.main_thread():
            for name in STOP_SIGNALS:
                sig = getattr(signal, name, None)
                if sig is not None:
                    self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def poll(self) -> int | None:
        """The signal received so far (``None``: none), after installing the handlers again.

        A library can replace a Python signal handler at the C level, and ``signal.getsignal`` then
        still reports the Python one. Under ``torch.compile`` on CUDA, the first kernel build installs
        a SIGUSR1 handler of its own that does not pass the signal on, so a compiled trainer never
        saw SIGUSR1 (SIGTERM still arrived). Installing the handlers again at every poll takes them
        back (one ``sigaction`` per signal, a few microseconds). A signal that arrives before the
        next poll, while the foreign handler holds it, is still lost; send SIGTERM where that matters.
        """
        if self._previous and threading.current_thread() is threading.main_thread():
            for sig in self._previous:
                signal.signal(sig, self._handle)
        return self.received

    def __exit__(self, *exc) -> bool:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous.clear()
        return False


def ignore_stop_signals(worker_id: int = 0) -> None:
    """DataLoader ``worker_init_fn``: the stop signals are for the main process only.

    Batch schedulers and torchrun signal the whole process group; a DataLoader worker killed by
    the signal makes the main process fail (``DataLoader worker ... killed by signal``) before
    it has saved. Use it through :class:`SignalSafeDataLoader`, which also stops such workers
    before the interpreter exits.
    """
    for name in STOP_SIGNALS:
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, signal.SIG_IGN)


def _ignore_stop_signals_then(init: Callable[[int], None], worker_id: int) -> None:
    ignore_stop_signals(worker_id)
    init(worker_id)


def stop_loader_workers(iterator: Any) -> None:
    """Stop the worker processes of a DataLoader iterator now, through torch's own shutdown (the
    done event and a sentinel on each index queue, not a signal). A no-op for a single-process
    iterator, an exhausted one and one already stopped."""
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


_worker_iterators: weakref.WeakSet = weakref.WeakSet()


def _stop_live_workers() -> None:
    for iterator in list(_worker_iterators):
        try:
            stop_loader_workers(iterator)
        except Exception:
            log.exception("could not stop the DataLoader workers of %r", iterator)


def _stop_at_exit(iterator: Any) -> None:
    # stays the last registered exit handler (see SignalSafeDataLoader)
    atexit.unregister(_stop_live_workers)
    atexit.register(_stop_live_workers)
    _worker_iterators.add(iterator)


class SignalSafeDataLoader(DataLoader):
    """A ``DataLoader`` whose worker processes ignore :data:`STOP_SIGNALS` (they run
    :func:`ignore_stop_signals` before any ``worker_init_fn``) and are stopped before the
    interpreter exits.

    At exit, ``multiprocessing`` stops daemonic children with SIGTERM and then joins them: a
    worker that ignores SIGTERM would keep a process that still holds one of its iterators
    waiting forever. The worker iterators of these loaders are stopped through torch's shutdown
    by an exit handler that runs before those of ``multiprocessing`` and of torch's DataLoader
    (after which that shutdown is a no-op): ``atexit`` runs the last registered handler first,
    so this one is registered again with each worker iterator. torch registers a handler of its
    own for each worker of a persistent iterator with pinned memory, which waits 5 s for the
    worker and then sends it the SIGTERM it ignores.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_workers > 0:
            init = self.worker_init_fn
            self.worker_init_fn = (ignore_stop_signals if init is None
                                   else functools.partial(_ignore_stop_signals_then, init))

    def __iter__(self):
        iterator = super().__iter__()
        if self.num_workers > 0:
            _stop_at_exit(iterator)
        return iterator


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _stat_sum(path: Path, keys: Sequence[str]) -> int:
    try:
        values = dict(line.split()[:2] for line in path.read_text().splitlines() if len(line.split()) >= 2)
        return sum(int(values.get(k, 0)) for k in keys)
    except (OSError, ValueError):
        return 0


def available_memory() -> int | None:
    """Bytes of RAM this process can still use, or ``None`` if unknown.

    The smaller of the host's ``MemAvailable`` and, for every memory-limited cgroup this process
    is in (v1 or v2, up to the root), its limit minus its usage plus its reclaimable page cache.
    Batch schedulers and container runtimes enforce their memory limits through these cgroups.
    """
    candidates = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
    except (OSError, ValueError, IndexError):
        pass
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        entries = []
    for entry in entries:
        parts = entry.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, rel = parts
        if hierarchy == "0" and controllers == "":
            root, limit_file, usage_file = Path("/sys/fs/cgroup"), "memory.max", "memory.current"
            cache_keys = ("active_file", "inactive_file")
        elif "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory")
            limit_file, usage_file = "memory.limit_in_bytes", "memory.usage_in_bytes"
            cache_keys = ("total_active_file", "total_inactive_file")
        else:
            continue
        d = root / rel.lstrip("/")
        while True:
            limit, usage = _read_int(d / limit_file), _read_int(d / usage_file)
            if limit is not None and usage is not None and limit < 1 << 60:
                candidates.append(limit - usage + _stat_sum(d / "memory.stat", cache_keys))
            if d == root or root not in d.parents:
                break
            d = d.parent
    return min(candidates) if candidates else None


def compile_toolchain_problem() -> str | None:
    """What Triton lacks here to build its C launcher (a C compiler, ``Python.h``), or None.

    Mirrors ``triton.runtime.build``: ``$CC``, else gcc or clang, and the include dir of the
    default sysconfig scheme (``posix_local`` read as ``posix_prefix``).
    """
    if not (os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")):
        return "no C compiler ($CC, gcc or clang)"
    scheme = sysconfig.get_default_scheme()
    include = sysconfig.get_paths(scheme="posix_prefix" if scheme == "posix_local" else scheme)["include"]
    if not os.path.isfile(os.path.join(include, "Python.h")):
        return f"no Python.h in {include} (the Python development headers)"
    return None


def warn_if_compile_unavailable(setting: str, fallback: str, device: torch.device | str) -> None:
    """Warn (main process only) when ``setting`` compiles for CUDA on a machine where Triton
    cannot build its C launcher.

    A warning, not an error: Triton reuses a launcher already in its cache, and its build step
    can be configured to use another compiler.
    """
    if torch.device(device).type != "cuda" or not is_main():
        return
    problem = compile_toolchain_problem()
    if problem:
        log.warning("%s uses torch.compile, but this machine has %s: Triton builds a C module on first use and "
                    "fails unless its cache already holds one. Install gcc (or clang, or set CC) and the Python "
                    "headers (python3-dev), or run eagerly: %s", setting, problem, fallback)
