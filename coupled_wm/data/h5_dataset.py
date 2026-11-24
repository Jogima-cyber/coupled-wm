# SPDX-License-Identifier: Apache-2.0
"""Sliding-window sequence dataset over a pool of HDF5 episode files.

One loader serves the tokenizer, the dynamics model, RSSM pretraining and the seeding of
the G1 world-model environment. The on-disk format (see ``docs/data_format.md``) is one
``.h5`` file per chunk with datasets at the file root:

==============  ===============  ==================================================
key             shape / dtype    meaning
==============  ===============  ==================================================
``images``      (T,H,W,3) uint8  camera frames, channel order given by ``color_order``
``commands``    (T,A) float      the command that conditions frame t (only if ``load_action``)
``dones``       (T,)             1 on the LAST frame of each contiguous segment
``proprio``     (T,P) float      raw proprioception (only if ``load_proprio``)
``rewards``     (T,) float       per-frame reward label of frame t (``reward_key``)
==============  ===============  ==================================================

Alignment: ``commands[t]`` is the command whose effect frame ``t`` shows; the world model
consumes it together with frame ``t``. A logger that records ``(obs_t, action_t)`` with
``action_t`` producing ``obs_{t+1}`` therefore stores ``commands[t] = action_{t-1}``.
``dones`` marks the last frame of an episode, not the first (an ``is_first`` flag shifted
by one frame is not a valid ``dones``). An ``images`` (or file) attribute ``channel_order``
(``"bgr"``/``"rgb"``) makes every loader check its ``color_order`` against the file.

Windows never cross a ``done`` boundary or a file boundary. Windows are enumerated file by
file (in the given file order), segment by segment, start by start; dataset index ``i`` is
the ``i``-th window in that order. This is the exact enumeration of the paper's loaders, so
a seeded sampler maps the same seed to the same window.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from coupled_wm.common.distributed import local_world_size
from coupled_wm.common.system import available_memory

log = logging.getLogger(__name__)

_CHUNK_RE = re.compile(r"chunk_(\d+)\.h5")
_CACHE_DTYPES = {"uint8": torch.uint8, "float16": torch.float16, "float32": torch.float32}
_COLOR_ATTRS = ("channel_order", "color_order")


# --------------------------------------------------------------------------- file pools


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (str, os.PathLike)):
        return [x]
    return list(x)


def _read_file_list(file_list: Any) -> tuple[list[str], Path | None]:
    if isinstance(file_list, (str, os.PathLike)):
        path = Path(file_list)
        entries = []
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                entries.append(line)
        return entries, path.parent
    return [os.fspath(e) for e in file_list], None


def _resolve_listed(entry: str, data_dirs: list[Path], base: Path | None) -> str:
    p = Path(entry).expanduser()
    if p.is_absolute():
        candidates = [p]
    elif data_dirs:
        candidates = [d / p for d in data_dirs]
    else:
        candidates = [(base or Path.cwd()) / p]
    found = [c for c in candidates if c.is_file()]
    if not found:
        raise FileNotFoundError(f"file_list entry {entry!r} not found (looked in: {[str(c) for c in candidates]})")
    if len(found) > 1:
        raise ValueError(f"file_list entry {entry!r} is ambiguous: {[str(c) for c in found]}")
    return os.path.abspath(found[0])


def _scan_dir(data_dir: Path) -> list[str]:
    """Files and dangling symlinks directly inside ``data_dir``, sorted by name."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir {data_dir} does not exist or is not a directory")
    names = []
    with os.scandir(data_dir) as it:
        for entry in it:
            if entry.is_file() or (entry.is_symlink() and not os.path.exists(entry.path)):
                names.append(entry.name)
    return [os.path.abspath(data_dir / n) for n in sorted(names)]


def list_h5_files(
    data_dirs: str | os.PathLike | Sequence[str | os.PathLike] | None,
    include_globs: Sequence[str] = ("*.h5",),
    exclude_globs: Sequence[str] = (),
    min_chunk_index: int | None = None,
    file_list: str | os.PathLike | Sequence[str | os.PathLike] | None = None,
) -> list[str]:
    """Return the file pool: absolute paths in a deterministic order.

    * Directory scan (default): each dir in the given order, file names sorted within a dir.
      Symlinks are followed (the link name is what the filters see). A dangling link raises
      if it survives the filters; one the filters drop is ignored.
    * ``file_list`` (a sequence, or a text file with one path per line and ``#`` comments):
      the listed files in the listed order. Relative entries resolve against ``data_dirs``
      (exactly one must contain the entry) or, without ``data_dirs``, against the list
      file's directory (text file) or the working directory. The filters below must keep
      every listed entry: a listed file that a filter drops raises.
    * Filters on the file name, in order: keep names matching any ``include_globs``, drop
      names matching any ``exclude_globs``, drop files named exactly ``chunk_<N>.h5`` with
      ``N < min_chunk_index``; any other name is kept (``patch_2_chunk_0004.h5`` is kept).

    Raises if the result is empty or a file appears twice (same real path).
    """
    dirs = [Path(d).expanduser() for d in _as_list(data_dirs)]
    include_globs = tuple(_as_list(include_globs)) or ("*.h5",)
    exclude_globs = tuple(_as_list(exclude_globs))

    if file_list is not None:
        entries, base = _read_file_list(file_list)
        files = [_resolve_listed(e, dirs, base) for e in entries]
        source = f"file_list ({len(files)} entries)"
    else:
        if not dirs:
            raise ValueError("list_h5_files: give data_dirs or file_list")
        files = []
        for d in dirs:
            files.extend(_scan_dir(d))
        source = ", ".join(str(d) for d in dirs)

    candidates = files
    n0 = len(files)
    files = [f for f in files if any(fnmatch.fnmatchcase(os.path.basename(f), g) for g in include_globs)]
    n_inc = len(files)
    files = [f for f in files if not any(fnmatch.fnmatchcase(os.path.basename(f), g) for g in exclude_globs)]
    n_exc = len(files)
    if min_chunk_index:
        def below(f: str) -> bool:
            m = _CHUNK_RE.fullmatch(os.path.basename(f))
            return m is not None and int(m.group(1)) < min_chunk_index
        files = [f for f in files if not below(f)]

    if file_list is not None and len(files) != n0:
        kept = set(files)
        dropped = [os.path.basename(f) for f in candidates if f not in kept]
        raise ValueError(
            f"{len(dropped)} file_list entries are dropped by the name filters (include_globs="
            f"{list(include_globs)}, exclude_globs={list(exclude_globs)}, min_chunk_index={min_chunk_index}): "
            f"{dropped[:5]}; a file_list is used as given, so relax the filters or remove the entries"
        )
    dangling = [f for f in files if not os.path.exists(f)]
    if dangling:
        raise FileNotFoundError(
            f"{len(dangling)} files of the pool are dangling symlinks, e.g. {dangling[0]} -> "
            f"{os.readlink(dangling[0])}"
        )

    real = {}
    for f in files:
        r = os.path.realpath(f)
        if r in real:
            raise ValueError(f"file pool lists the same file twice: {real[r]} and {f} -> {r}")
        real[r] = f

    log.info(
        "file pool from %s: %d candidates, %d after include %s, %d after exclude %s, "
        "%d after min_chunk_index=%s",
        source, n0, n_inc, list(include_globs), n_exc, list(exclude_globs), len(files), min_chunk_index,
    )
    if not files:
        raise ValueError(f"no .h5 files left in the pool from {source} after filtering")
    return files


# --------------------------------------------------------------------------- scanning


@dataclass
class FileInfo:
    """Per-file metadata gathered at scan time."""

    path: str
    n_frames: int
    image_hw: tuple[int, int]
    action_dim: int | None
    proprio_dim: int | None
    segments: list[tuple[int, int]]  # inclusive (start, end) of every segment


def _per_frame(arr: np.ndarray, name: str, path: str) -> np.ndarray:
    """Return a (T,) view of a per-frame scalar stored as (T,) or (T, 1)."""
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{path}: '{name}' must have shape (T,) or (T,1), got {arr.shape}")
    return arr


def segments_from_dones(dones: np.ndarray, n_frames: int) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` segments; a segment ends at each done and at the last frame."""
    ends = list(np.flatnonzero(np.asarray(dones).reshape(-1) > 0.5))
    if not ends or ends[-1] != n_frames - 1:
        ends.append(n_frames - 1)
    segments, start = [], 0
    for end in ends:
        segments.append((start, int(end)))
        start = int(end) + 1
    return segments


def window_starts(segments: Iterable[tuple[int, int]], window: int, stride: int) -> list[int]:
    """Start frames of all windows of length ``window`` inside the segments, in order."""
    starts = []
    for start, end in segments:
        seg_len = end - start + 1
        if seg_len >= window:
            starts.extend(start + k * stride for k in range((seg_len - window) // stride + 1))
    return starts


def _declared_color_order(f: h5py.File) -> str | None:
    for attrs in (f["images"].attrs, f.attrs):
        for key in _COLOR_ATTRS:
            if key in attrs:
                v = attrs[key]
                v = v.decode() if isinstance(v, bytes) else str(v)
                return v.strip().lower()
    return None


def scan_file(
    path: str,
    *,
    color_order: str,
    load_proprio: bool,
    reward_key: str | None,
    load_action: bool = True,
) -> FileInfo:
    """Open ``path``, check the format contract and return its :class:`FileInfo`.

    Raises ``OSError`` (unreadable), ``KeyError`` (missing dataset) or ``ValueError``
    (inconsistent shapes, wrong dtype, declared channel order mismatch).
    """
    try:
        f = h5py.File(path, "r")
    except OSError as e:
        raise OSError(f"{path}: cannot open as HDF5 ({e})") from e
    with f:
        required = ["images", "dones"]
        if load_action:
            required.append("commands")
        if load_proprio:
            required.append("proprio")
        if reward_key is not None:
            required.append(reward_key)
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(f"{path}: missing dataset(s) {missing}; present: {sorted(f.keys())}")

        images = f["images"]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"{path}: 'images' must be (T,H,W,3), got {images.shape}")
        if images.dtype != np.uint8:
            raise ValueError(f"{path}: 'images' must be uint8, got {images.dtype}")
        n = int(images.shape[0])
        if n == 0:
            raise ValueError(f"{path}: file has no frames")

        declared = _declared_color_order(f)
        if declared is not None and declared != color_order:
            raise ValueError(f"{path}: images declare channel order {declared!r} but color_order={color_order!r}")

        action_dim = None
        if load_action:
            commands = f["commands"]
            if commands.ndim != 2 or commands.shape[0] != n:
                raise ValueError(f"{path}: 'commands' must be (T={n}, A), got {commands.shape}")
            action_dim = int(commands.shape[1])
        proprio_dim = None
        if load_proprio:
            proprio = f["proprio"]
            if proprio.ndim != 2 or proprio.shape[0] != n:
                raise ValueError(f"{path}: 'proprio' must be (T={n}, P), got {proprio.shape}")
            proprio_dim = int(proprio.shape[1])
        if reward_key is not None:
            ds = f[reward_key]
            if ds.shape[0] != n:
                raise ValueError(f"{path}: '{reward_key}' has {ds.shape[0]} frames, 'images' has {n}")
            if ds.dtype.kind not in "biuf":
                raise ValueError(f"{path}: '{reward_key}' must be numeric, got dtype {ds.dtype}")
            if not (ds.ndim == 1 or (ds.ndim == 2 and ds.shape[1] == 1)):
                raise ValueError(f"{path}: '{reward_key}' must have shape (T,) or (T,1), got {ds.shape}")

        dones = _per_frame(f["dones"][()], "dones", path)
        if dones.shape[0] != n:
            raise ValueError(f"{path}: 'dones' has {dones.shape[0]} frames, 'images' has {n}")

        return FileInfo(
            path=path,
            n_frames=n,
            image_hw=(int(images.shape[1]), int(images.shape[2])),
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            segments=segments_from_dones(dones, n),
        )


# --------------------------------------------------------------------------- image ops


def frames_to_tensor(frames: np.ndarray, color_order: str, image_size: tuple[int, int] | None) -> torch.Tensor:
    """uint8 (T,H,W,3) frames -> float32 (T,3,h,w) RGB in [0, 1].

    The resize is the G1 environment's op (``F.interpolate``, bilinear,
    ``align_corners=False``, no antialias) on float [0, 1] frames.
    """
    if color_order == "bgr":
        frames = frames[..., ::-1].copy()
    x = torch.from_numpy(frames).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    if image_size is not None and tuple(x.shape[-2:]) != tuple(image_size):
        x = F.interpolate(x, size=tuple(image_size), mode="bilinear", align_corners=False)
    return x


def _frames_for_cache(frames: np.ndarray, color_order: str, image_size, cache_dtype: torch.dtype) -> torch.Tensor:
    native = image_size is None or tuple(frames.shape[1:3]) == tuple(image_size)
    if native and cache_dtype == torch.uint8:
        if color_order == "bgr":
            frames = frames[..., ::-1]
        return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).contiguous()
    x = frames_to_tensor(frames, color_order, image_size)
    if cache_dtype == torch.uint8:
        return (x * 255.0).round_().clamp_(0, 255).to(torch.uint8).contiguous()
    return x.to(cache_dtype).contiguous()


class _CacheChunks(Dataset):
    """Frame chunks read and resized in DataLoader workers while building the RAM cache."""

    def __init__(self, jobs, color_order, image_size, cache_dtype):
        self.jobs = jobs
        self.color_order = color_order
        self.image_size = image_size
        self.cache_dtype = cache_dtype

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        path, start, end, dst = self.jobs[i]
        with h5py.File(path, "r") as f:
            frames = f["images"][start:end]
        return dst, _frames_for_cache(frames, self.color_order, self.image_size, self.cache_dtype)


# --------------------------------------------------------------------------- dataset


class HDF5SequenceDataset(Dataset):
    """Windows of ``window`` consecutive frames from a pool of HDF5 files.

    Args:
        files: the file pool (see :func:`list_h5_files`); a directory path is scanned with
            the default filters.
        window: frames per item.
        stride: step between window starts inside a segment.
        image_size: resize frames to ``(s, s)`` (int) or ``(h, w)``; ``None`` keeps them.
        color_order: channel order on disk, ``"bgr"`` (the G1 data) or ``"rgb"``. Items are
            always RGB. A ``channel_order``/``color_order`` attribute on the file or on
            ``images``, when present, must agree.
        load_proprio: require and return ``proprio``.
        load_action: require and return ``action`` (the ``commands``); ``False`` for
            consumers that never use actions (the tokenizer), so action-free video works.
        reward_key: dataset holding the per-frame reward; ``None`` returns no reward.
        reward_remap: optional ``{value: new_value}`` applied to exact matches of the
            (possibly overridden) rewards. Off by default.
        reward_override_dir: optional directory holding ``<basename of source>`` files whose
            ``reward`` dataset REPLACES the source rewards. Every source file must have one.
        action_dim: when given, the files' command dimension must equal it. Checked right
            after the scan, so a mismatch raises before the RAM cache is built.
        cache: hold every frame (resized) and the per-frame arrays in RAM, built in parallel
            at construction. With ``cache_dtype="uint8"`` resized frames are re-quantised to
            uint8 (|error| <= 0.5/255 vs. the streamed float frames); ``"float16"``/
            ``"float32"`` keep more precision at 2x/4x the memory. Every process that builds
            the dataset holds its own cache: N ranks on a node need N times the size logged
            at construction (a warning is logged when that exceeds the available memory).
        cache_workers: processes used to build the cache (default: up to 8 CPUs).
        max_open_files: per-process limit (>= 1) of HDF5 handles kept open between items.

    Items are dicts of tensors: ``image`` (T,3,H,W) float32 in [0,1] RGB, ``action``
    (T,A) float32 (the file's ``commands``; only with ``load_action``), ``done`` (T,) float32,
    ``is_first`` (T,) float32 (1 at t=0: the window start resets any recurrent state), plus
    ``proprio`` (T,P) float32 and ``reward`` (T,) float32 when requested.

    Attributes: ``files``, ``file_infos`` (:class:`FileInfo` per file), ``action_dim``
    (``None`` without ``load_action``),
    ``proprio_dim``, ``output_image_hw`` (H, W of returned frames) and ``summary`` (counts);
    ``window_info(i)`` gives the ``(file, start frame)`` of item ``i``.

    HDF5 handles are opened lazily and kept open per process (DataLoader workers open their
    own; handles are never pickled), up to ``max_open_files`` in LRU order.
    """

    def __init__(
        self,
        files: Sequence[str | os.PathLike] | str | os.PathLike,
        window: int,
        stride: int = 1,
        image_size: int | Sequence[int] | None = None,
        color_order: str = "bgr",
        load_proprio: bool = True,
        load_action: bool = True,
        reward_key: str | None = "rewards",
        reward_remap: Mapping[float, float] | None = None,
        reward_override_dir: str | os.PathLike | None = None,
        action_dim: int | None = None,
        cache: bool = False,
        cache_dtype: str = "uint8",
        cache_workers: int | None = None,
        max_open_files: int = 256,
    ):
        if isinstance(files, (str, os.PathLike)):
            files = list_h5_files(files)
        self.files = [os.path.abspath(os.fspath(p)) for p in files]
        if not self.files:
            raise ValueError("HDF5SequenceDataset: empty file pool")
        if len(set(self.files)) != len(self.files):
            raise ValueError("HDF5SequenceDataset: duplicate files in the pool")
        if int(window) < 1 or int(stride) < 1:
            raise ValueError(f"window and stride must be >= 1, got window={window} stride={stride}")
        if color_order not in ("bgr", "rgb"):
            raise ValueError(f"color_order must be 'bgr' or 'rgb', got {color_order!r}")
        if cache_dtype not in _CACHE_DTYPES:
            raise ValueError(f"cache_dtype must be one of {list(_CACHE_DTYPES)}, got {cache_dtype!r}")
        if reward_override_dir is not None and reward_key is None:
            raise ValueError("reward_override_dir is set but reward_key is None (no reward is returned)")
        if action_dim is not None and not load_action:
            raise ValueError("action_dim is checked against the commands, but load_action is False")
        if int(max_open_files) < 1:
            raise ValueError(f"max_open_files must be >= 1, got {max_open_files}")

        self.window = int(window)
        self.stride = int(stride)
        if image_size is None:
            self.image_size = None
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(int(s) for s in image_size)
            if len(self.image_size) != 2:
                raise ValueError(f"image_size must be an int or (h, w), got {image_size!r}")
        self.color_order = color_order
        self.load_proprio = bool(load_proprio)
        self.load_action = bool(load_action)
        self.reward_key = reward_key
        self.reward_remap = {float(k): float(v) for k, v in (reward_remap or {}).items()}
        self.reward_override_dir = Path(reward_override_dir) if reward_override_dir is not None else None
        self.max_open_files = int(max_open_files)

        self._scan()
        if action_dim is not None and self.action_dim != int(action_dim):
            raise ValueError(f"the files' commands have {self.action_dim} dims, expected action_dim={action_dim}")
        self._override_rewards = self._load_overrides() if self.reward_override_dir is not None else None

        self._handles: OrderedDict[int, h5py.File] = OrderedDict()
        self._handles_pid: int | None = None
        self._cache: dict[str, torch.Tensor] | None = None
        self.cache_dtype = cache_dtype
        if cache:
            self._build_cache(_CACHE_DTYPES[cache_dtype], cache_workers)

    # ---------------------------------------------------------------- scan

    def _scan(self) -> None:
        infos = [
            scan_file(p, color_order=self.color_order, load_proprio=self.load_proprio,
                      reward_key=None if self.reward_override_dir is not None else self.reward_key,
                      load_action=self.load_action)
            for p in self.files
        ]
        self.file_infos = infos

        def consistent(attr: str, label: str):
            values = {getattr(i, attr) for i in infos}
            if len(values) > 1:
                examples = {v: next(i.path for i in infos if getattr(i, attr) == v) for v in values}
                raise ValueError(f"files disagree on {label}: {examples}")
            return values.pop()

        self.action_dim = consistent("action_dim", "the command dimension") if self.load_action else None
        self.proprio_dim = consistent("proprio_dim", "the proprio dimension") if self.load_proprio else None
        if self.image_size is None:
            self.native_image_hw = consistent("image_hw", "the image size (set image_size to resize)")
        else:
            self.native_image_hw = None
        self.output_image_hw = self.image_size or self.native_image_hw
        self.native_image_sizes = sorted({tuple(int(s) for s in i.image_hw) for i in infos})
        self.resized = self.image_size is not None and self.native_image_sizes != [tuple(self.image_size)]

        win_file, win_start = [], []
        n_seg = n_short = short_frames = n_kept = 0
        for fi, info in enumerate(infos):
            starts = window_starts(info.segments, self.window, self.stride)
            win_file.extend([fi] * len(starts))
            win_start.extend(starts)
            for s, e in info.segments:
                n_seg += 1
                if e - s + 1 < self.window:
                    n_short += 1
                    short_frames += e - s + 1
                else:
                    n_kept += 1
        self._win_file = np.asarray(win_file, dtype=np.int32)
        self._win_start = np.asarray(win_start, dtype=np.int64)
        total = sum(i.n_frames for i in infos)
        self.summary = {
            "files": len(infos),
            "frames": total,
            "segments": n_seg,
            "segments_used": n_kept,
            "segments_shorter_than_window": n_short,
            "frames_in_short_segments": short_frames,
            "windows": len(self._win_file),
            "window": self.window,
            "stride": self.stride,
        }
        log.info(
            "%d files, %d frames, %d segments: %d windows of %d frames (stride %d); "
            "skipped %d segments shorter than the window (%d frames)",
            len(infos), total, n_seg, len(self._win_file), self.window, self.stride, n_short, short_frames,
        )
        if len(self._win_file) == 0:
            raise ValueError(f"no segment is at least {self.window} frames long; the dataset would be empty")

    def _load_overrides(self) -> list[np.ndarray]:
        names = [os.path.basename(p) for p in self.files]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"reward_override_dir matches files by basename, but basenames repeat: {dup[:5]}")
        out = []
        for path, info in zip(self.files, self.file_infos):
            ov = self.reward_override_dir / os.path.basename(path)
            if not ov.is_file():
                raise FileNotFoundError(f"no reward override for {path} (expected {ov})")
            with h5py.File(ov, "r") as f:
                if "reward" not in f:
                    raise KeyError(f"{ov}: override file has no 'reward' dataset")
                r = _per_frame(np.asarray(f["reward"][()], dtype=np.float32), "reward", str(ov))
            if r.shape[0] != info.n_frames:
                raise ValueError(f"{ov}: override has {r.shape[0]} frames, source has {info.n_frames}")
            out.append(r)
        log.info("reward override: replacing rewards of %d files from %s", len(out), self.reward_override_dir)
        return out

    # ---------------------------------------------------------------- cache

    def _build_cache(self, dtype: torch.dtype, workers: int | None, chunk_frames: int = 256) -> None:
        total = self.summary["frames"]
        offsets = np.zeros(len(self.files) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([i.n_frames for i in self.file_infos])
        if self.image_size is not None:
            h, w = self.image_size
        else:
            h, w = self.native_image_hw
        gb = total * 3 * h * w * torch.empty((), dtype=dtype).element_size() / 1e9
        procs = local_world_size()
        log.info("building RAM cache: %d frames at %dx%d %s (%.2f GB in this process, %d process(es) on this node)",
                 total, h, w, str(dtype).replace("torch.", ""), gb, procs)
        available = available_memory()
        if available is not None and gb * procs > available / 1e9:
            log.warning(
                "the RAM cache needs %.1f GB per process x %d process(es) on this node = %.1f GB, but only %.1f GB "
                "is available; set cache_dtype to uint8 (4x smaller than float32) or disable the cache",
                gb, procs, gb * procs, available / 1e9,
            )
        images = torch.empty((total, 3, h, w), dtype=dtype)

        arrays: dict[str, list[np.ndarray]] = {"action": []} if self.load_action else {}
        arrays["done"] = []
        if self.load_proprio:
            arrays["proprio"] = []
        if self.reward_key is not None:
            arrays["reward"] = []
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                if self.load_action:
                    arrays["action"].append(np.asarray(f["commands"][()], dtype=np.float32))
                arrays["done"].append(_per_frame(np.asarray(f["dones"][()], dtype=np.float32), "dones", path))
                if self.load_proprio:
                    arrays["proprio"].append(np.asarray(f["proprio"][()], dtype=np.float32))
                if self.reward_key is not None:
                    arrays["reward"].append(self._rewards_from(f, fi, slice(None)))

        jobs = []
        for fi, (path, info) in enumerate(zip(self.files, self.file_infos)):
            for s in range(0, info.n_frames, chunk_frames):
                e = min(s + chunk_frames, info.n_frames)
                jobs.append((path, s, e, int(offsets[fi] + s)))
        if workers is None:
            try:
                n_cpu = len(os.sched_getaffinity(0))
            except AttributeError:
                n_cpu = os.cpu_count() or 1
            workers = min(8, n_cpu)
        loader = DataLoader(
            _CacheChunks(jobs, self.color_order, self.image_size, dtype),
            batch_size=None, shuffle=False, num_workers=workers,
            generator=torch.Generator(),  # keep the global RNG untouched
        )
        report_every = max(1, len(jobs) // 10)
        for j, (dst, frames) in enumerate(loader):
            images[dst: dst + frames.shape[0]].copy_(frames)
            if (j + 1) % report_every == 0:
                log.info("RAM cache: %d/%d chunks", j + 1, len(jobs))

        cache = {"image": images}
        for k, parts in arrays.items():
            cache[k] = torch.from_numpy(np.concatenate(parts, axis=0))
        self._cache = cache
        self._offsets = offsets

    # ---------------------------------------------------------------- reading

    def _handle(self, fi: int) -> h5py.File:
        pid = os.getpid()
        if self._handles_pid != pid:  # fresh process (DataLoader worker): never reuse inherited handles
            self._handles = OrderedDict()
            self._handles_pid = pid
        f = self._handles.get(fi)
        if f is None:
            if len(self._handles) >= self.max_open_files:
                _, old = self._handles.popitem(last=False)
                old.close()
            f = h5py.File(self.files[fi], "r")
            self._handles[fi] = f
        else:
            self._handles.move_to_end(fi)
        return f

    def _rewards_from(self, f: h5py.File, fi: int, sl: slice) -> np.ndarray:
        if self._override_rewards is not None:
            r = self._override_rewards[fi][sl].copy()
        else:
            r = _per_frame(np.asarray(f[self.reward_key][sl], dtype=np.float32), self.reward_key, self.files[fi])
        if self.reward_remap:
            src = r.copy()
            for old, new in self.reward_remap.items():
                r[src == np.float32(old)] = new
        return r

    def __len__(self) -> int:
        return len(self._win_file)

    def window_info(self, idx: int) -> tuple[str, int]:
        """``(file path, start frame)`` of window ``idx``."""
        return self.files[int(self._win_file[idx])], int(self._win_start[idx])

    def raw_window(self, idx: int) -> dict[str, np.ndarray]:
        """Window ``idx`` without conversion or caching: ``images`` (T,H,W,3) uint8 in RGB
        order (no resize) and, with ``load_action`` / ``load_proprio``, ``commands`` (T,A) /
        ``proprio`` (T,P), in their stored dtypes."""
        fi = int(self._win_file[idx])
        sl = slice(int(self._win_start[idx]), int(self._win_start[idx]) + self.window)
        f = self._handle(fi)
        frames = f["images"][sl]
        if self.color_order == "bgr":
            frames = frames[..., ::-1]
        out = {"images": frames}
        if self.load_action:
            out["commands"] = np.asarray(f["commands"][sl])
        if self.load_proprio:
            out["proprio"] = np.asarray(f["proprio"][sl])
        return out

    def window_frame_values(self, key: str, offset: int) -> np.ndarray:
        """``key[start + offset]`` of every window (e.g. ``"commands"`` at a window's last
        frame), shape ``(len(self), ...)``, in the stored dtype. Reads each file once."""
        if not 0 <= offset < self.window:
            raise ValueError(f"offset must be in [0, window={self.window}), got {offset}")
        out = None
        for fi, path in enumerate(self.files):
            sel = np.flatnonzero(self._win_file == fi)
            if not sel.size:
                continue
            with h5py.File(path, "r") as f:
                values = np.asarray(f[key][()])[self._win_start[sel] + offset]
            if out is None:
                out = np.empty((len(self), *values.shape[1:]), dtype=values.dtype)
            out[sel] = values
        return out

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi = int(self._win_file[idx])
        start = int(self._win_start[idx])
        end = start + self.window
        if self._cache is not None:
            item = self._item_from_cache(fi, start, end)
        else:
            item = self._item_from_file(fi, start, end)
        is_first = torch.zeros(self.window, dtype=torch.float32)
        is_first[0] = 1.0
        item["is_first"] = is_first
        return item

    def _item_from_file(self, fi: int, start: int, end: int) -> dict[str, torch.Tensor]:
        f = self._handle(fi)
        sl = slice(start, end)
        item = {"image": frames_to_tensor(f["images"][sl], self.color_order, self.image_size)}
        if self.load_action:
            item["action"] = torch.from_numpy(np.asarray(f["commands"][sl])).float()
        item["done"] = torch.from_numpy(_per_frame(np.asarray(f["dones"][sl]), "dones", self.files[fi])).float()
        if self.load_proprio:
            item["proprio"] = torch.from_numpy(np.asarray(f["proprio"][sl])).float()
        if self.reward_key is not None:
            item["reward"] = torch.from_numpy(self._rewards_from(f, fi, sl))
        return item

    def _item_from_cache(self, fi: int, start: int, end: int) -> dict[str, torch.Tensor]:
        g0 = int(self._offsets[fi]) + start
        g1 = g0 + (end - start)
        cache = self._cache
        img = cache["image"][g0:g1]
        if img.dtype == torch.uint8:
            image = img.float() / 255.0
        else:
            image = img.to(torch.float32, copy=True)
        item = {"image": image}
        for k, v in cache.items():
            if k != "image":
                item[k] = v[g0:g1].clone()
        return item

    # ---------------------------------------------------------------- pickling / cleanup

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_handles_pid"] = None
        return state

    def close(self) -> None:
        if self._handles_pid == os.getpid():
            for f in self._handles.values():
                f.close()
        self._handles = OrderedDict()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
