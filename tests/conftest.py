# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures: tiny synthetic HDF5 episode files in the release data format."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest


def frame_value(t: int, c: int) -> int:
    """Pixel value written for frame ``t``, stored channel ``c`` (unique for t < 64)."""
    return (t % 64) * 4 + c


def write_h5(
    path: str | Path,
    n_frames: int,
    *,
    dones_at: list[int] | None = None,
    image_hw: tuple[int, int] = (8, 8),
    action_dim: int = 4,
    proprio_dim: int | None = 3,
    rewards: np.ndarray | None = None,
    reward_key: str | None = "rewards",
    file_id: int = 0,
    extra: dict[str, np.ndarray] | None = None,
    images_attrs: dict[str, str] | None = None,
) -> Path:
    """Write one episode file.

    Frame ``t`` is recognisable in every dataset: pixel value ``frame_value(t, c)`` in stored
    channel ``c``, ``commands[t, d] = 1000 * file_id + t + d / 100``,
    ``proprio[t, d] = -(1000 * file_id + t) - d / 100``. ``dones_at=None`` puts a single done
    on the last frame.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_frames)
    h, w = image_hw
    images = np.empty((n_frames, h, w, 3), dtype=np.uint8)
    for c in range(3):
        images[..., c] = frame_value(t, c)[:, None, None]
    commands = (1000 * file_id + t)[:, None] + np.arange(action_dim)[None] / 100.0
    dones = np.zeros(n_frames, dtype=np.float32)
    dones[[n_frames - 1] if dones_at is None else dones_at] = 1.0
    if rewards is None:
        rewards = (t % 3 == 0).astype(np.float32)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("images", data=images, chunks=(1, h, w, 3))
        for k, v in (images_attrs or {}).items():
            ds.attrs[k] = v
        f.create_dataset("commands", data=commands.astype(np.float32))
        f.create_dataset("dones", data=dones)
        if proprio_dim is not None:
            proprio = -(1000 * file_id + t)[:, None] - np.arange(proprio_dim)[None] / 100.0
            f.create_dataset("proprio", data=proprio.astype(np.float32))
        if reward_key is not None:
            f.create_dataset(reward_key, data=np.asarray(rewards, dtype=np.float32))
        for k, v in (extra or {}).items():
            f.create_dataset(k, data=v)
    return path


@pytest.fixture
def make_h5():
    """Factory fixture: ``make_h5(path, n_frames, **kwargs)`` (see :func:`write_h5`)."""
    return write_h5


@pytest.fixture
def h5_pool(tmp_path):
    """A 3-file pool with mid-file dones and one segment shorter than 8 frames."""
    d = tmp_path / "pool"
    write_h5(d / "chunk_0000.h5", 30, dones_at=[9, 29], file_id=0)        # segments 10, 20
    write_h5(d / "chunk_0001.h5", 25, dones_at=[4, 24], file_id=1)        # segments 5 (short), 20
    write_h5(d / "chunk_0002.h5", 16, dones_at=[], file_id=2)             # one segment, no done at all
    return d
