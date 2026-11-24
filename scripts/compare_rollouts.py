#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Ground-truth trajectories next to world-model generations of the same recorded commands.

    python scripts/compare_rollouts.py $G1_DATA_ROOT/new_wm_h5_segmented_small/world_model \\
        --exclude 'patch*' --min-chunk-index 21 \\
        --tokenizer weights/tokenizer --dynamics weights/dynamics \\
        --num 8 --seed 0 --horizon 64 --out runs/compare_rollouts

For each of ``--num`` windows of recorded data, the world model is primed with the window's first
``C`` real frames and commands (``C`` = the dynamics' ``init_context_len``, 32 for G1), then
generates ``--horizon`` further frames, each conditioned on the absolute command recorded for
that frame (``commands[t]`` conditions frame ``t``, as in training). Generation is the RL
environment's: the same context prefill, Euler steps, noise level of the cached frames, KV-cache
length, reward head and bf16 autocast, run by
:meth:`~coupled_wm.envs.DreamerV4VecEnvG1Relative.reset_from_context` and
:meth:`~coupled_wm.envs.DreamerV4VecEnvG1Relative.step_command`. The recorded commands are
replayed as recorded: the environment's command clamp, which bounds the policy's commands, is not
applied. Generation constants come from the dynamics ``config.json`` (else the G1 values), as in
the RL environment; ``--env-arg`` overrides them.

Windows. The files are selected as in every ``data`` block (``--include``, ``--exclude``,
``--min-chunk-index``, ``--file-list``; ``docs/data_format.md``). A window is ``C + horizon``
consecutive frames inside one segment (no done inside it, stride 1); ``--seed`` draws ``--num``
of them, and the first ``n`` windows of a draw do not depend on ``--num``. ``--windows FILE:START``
names windows instead (``START`` = the window's first context frame).

Two world models. ``--dynamics-b`` (with ``--tokenizer-b`` when it was trained on another
tokenizer) adds a third column: the second model replays the same windows with the same
generation noise (every batch reseeds the RNG from ``--seed`` before its prefill), so two
models with the same latent shape and context length see identical noise. When the two models
have different context lengths, both generate the same frames, each primed with the frames just
before them.

Outputs in ``--out``, per window ``<k>_<file>_<start>``:

* ``.mp4`` (or ``.gif`` when neither imageio-ffmpeg nor PyAV is installed; ``--format``): ground
  truth | model A [| model B] at the tokenizer's resolution, one video frame per generated frame
  at ``--fps`` (default: the dynamics' ``control_hz``), with the step counter, the frame index in
  the file, the data's reward label and each model's reward prediction (expectation of the
  reward head's ``reward_mtp_index`` head; head ``l`` at frame ``t`` predicts ``reward[t + l]``, so
  head 0 is the generated frame's own reward, the environment's step reward, shown next to that
  frame's label);
* ``_sheet.png``: a contact sheet of every ``--sheet-every``-th generated frame (and the
  first and last), one row per column of the video;
* ``.json``: the window, the frame indices, the per-step reward labels and predictions and their
  sums.

``index.json`` records the settings, the models' generation constants and every window. Frames
are shown as the models see them: RGB, resized to the tokenizer's ``image_size`` if stored at
another size; the decoder's output is clipped to ``[0, 1]`` for display only.

Memory: one world model is loaded at a time (``--dynamics-b`` runs after ``--dynamics``), with
``--batch-size`` windows generated in parallel (G1: about 2.9 GB of bf16 weights plus 0.55 GB of
KV cache per window). The generated frames are kept in host memory as uint8 until the end.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch

try:
    import coupled_wm  # noqa: F401
except ModuleNotFoundError:  # source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coupled_wm.common.checkpoint import is_pretrained_dir, load_checkpoint, load_pretrained_config
from coupled_wm.common.config import parse_yaml
from coupled_wm.common.logging import setup_logging
from coupled_wm.data.h5_dataset import list_h5_files
from coupled_wm.envs.g1_env import DreamerV4VecEnvG1Relative, SeedWindowDataset
from coupled_wm.world_model.loading import as_model_config, checkpoint_model_config
from coupled_wm.world_model.models import (
    DYNAMICS_MODEL_TYPE,
    G1_INFERENCE,
    G1_PREPROCESSING,
    TOKENIZER_MODEL_TYPE,
)

log = logging.getLogger("compare_rollouts")

GENERATION_KEYS = ("init_context_len", "kv_cache_len", "num_inference_steps", "seq_cor_tau_idx", "reward_mtp_index")
"""Environment arguments ``--env-arg`` may set (the generation constants)."""

VIDEO_FORMATS = ("auto", "mp4", "gif")
MP4_CRF = 5  # libx264 constant rate factor: visually lossless, playable by any H.264 decoder


# ----------------------------------------------------------------------------- models


class Model:
    """One world model of the comparison (column ``key``): its checkpoints, their model configs,
    its context length and, once generated, the environment's generation constants."""

    def __init__(self, key: str, name: str, tokenizer: str, dynamics: str):
        self.key, self.name, self.tokenizer, self.dynamics = key, name, tokenizer, dynamics
        self.tokenizer_config: dict = {}
        self.dynamics_config: dict = {}
        self.context_len = 0
        self.constants: dict = {}

    @property
    def uses_proprio(self) -> bool:
        return bool(self.tokenizer_config["architecture"]["use_proprio"])

    @property
    def image_hw(self) -> tuple[int, int]:
        return tuple(int(s) for s in self.tokenizer_config["architecture"]["image_size"])


def read_model_config(path: str | Path, model_type: str) -> dict:
    """The model config of a pretrained dir or of a training checkpoint (memory-mapped)."""
    if is_pretrained_dir(path):
        return as_model_config(load_pretrained_config(path), model_type)
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path}: neither a pretrained dir (config.json + model.safetensors) nor a checkpoint")
    return checkpoint_model_config(load_checkpoint(path, mmap=True), None, model_type, path)


def resolve_context_len(dynamics_config: dict, env_args: dict) -> int:
    """``init_context_len`` as the environment resolves it: ``--env-arg``, the dynamics
    ``config.json``, else the G1 value."""
    if env_args.get("init_context_len") is not None:
        return int(env_args["init_context_len"])
    recorded = (dynamics_config.get("inference") or {}).get("init_context_len")
    return int(recorded if recorded is not None else G1_INFERENCE["init_context_len"])


def resolve_color_order(models: Sequence[Model], explicit: str | None) -> str:
    """The h5 channel order: ``--color-order``, else the one the tokenizers recorded (they must
    agree), else the G1 order."""
    if explicit is not None:
        return explicit
    recorded = {(m.tokenizer_config.get("preprocessing") or {}).get("h5_color_order") for m in models}
    recorded.discard(None)
    if len(recorded) > 1:
        raise ValueError(f"the tokenizers recorded different h5 colour orders {sorted(recorded)}: set --color-order")
    return recorded.pop() if recorded else G1_PREPROCESSING["h5_color_order"]


def parse_env_args(items: Sequence[str]) -> dict[str, Any]:
    """``KEY=VALUE`` strings (values parsed as YAML) for the generation constants."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or key not in GENERATION_KEYS:
            raise ValueError(f"--env-arg takes KEY=VALUE with KEY in {list(GENERATION_KEYS)}, got {item!r}")
        out[key] = parse_yaml(value)
    return out


def _default_name(path: str) -> str:
    parts = Path(path).resolve().parts
    return "/".join(parts[-2:])


# ----------------------------------------------------------------------------- windows


def select_windows(dataset: SeedWindowDataset, num: int, seed: int, named: Sequence[str] = ()) -> list[int]:
    """Window ids: the ``named`` ``FILE:START`` windows, else ``num`` drawn with ``seed`` (the
    first ``n`` of a draw do not depend on ``num``)."""
    if named:
        return [_named_window(dataset, spec) for spec in named]
    if num > len(dataset):
        raise ValueError(f"--num {num}: the pool has only {len(dataset)} windows of {dataset.window} frames")
    perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(int(seed)))
    return perm[:num].tolist()


def _named_window(dataset: SeedWindowDataset, spec: str) -> int:
    name, sep, start = spec.rpartition(":")
    if not sep or not name or not start.lstrip("-").isdigit():
        raise ValueError(f"--windows takes FILE:START, got {spec!r}")
    matches = [i for i, f in enumerate(dataset.files) if f == str(Path(name).resolve()) or Path(f).name == name]
    if len(matches) != 1:
        raise ValueError(f"--windows {spec!r}: {name!r} names {len(matches)} files of the pool")
    target = (dataset.files[matches[0]], int(start))
    for i in range(len(dataset)):  # windows are enumerated file by file
        if dataset.window_info(i) == target:
            return i
    raise ValueError(f"--windows {spec!r}: no window of {dataset.window} frames starts there inside one segment "
                     "(the window would cross a done or the end of the file)")


def stack_windows(dataset: SeedWindowDataset, ids: Sequence[int]) -> dict[str, torch.Tensor]:
    """Windows ``ids`` as a batch: ``image`` (B, T, 3, H, W), ``action`` (B, T, A)[, ``proprio``]."""
    items = [dataset[i] for i in ids]
    return {k: torch.stack([it[k] for it in items]) for k in items[0] if k != "index"}


def read_labels(path: str, start: int, length: int, key: str) -> np.ndarray | None:
    """Reward labels of frames ``start .. start + length - 1``, or ``None`` without ``key``."""
    with h5py.File(path, "r") as f:
        if key not in f:
            return None
        shape = f[key].shape
        labels = np.asarray(f[key][start:start + length], dtype=np.float32).reshape(-1)
    if labels.shape != (length,):
        raise ValueError(f"{path}: '{key}' must hold one value per frame, got shape {shape}")
    return labels


def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """(…, 3, H, W) uint8, or float in [0, 1] (clipped), -> (…, H, W, 3) uint8."""
    if frames.dtype != torch.uint8:
        frames = (frames.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return frames.movedim(-3, -1).cpu().numpy()


# ----------------------------------------------------------------------------- generation


def generate(model: Model, dataset: SeedWindowDataset, ids: Sequence[int], *, horizon: int, window_context: int,
             batch_size: int, seed: int, device: torch.device, data_block: dict, env_args: dict,
             verify_pairing: bool) -> tuple[np.ndarray, np.ndarray]:
    """Replay the recorded commands of windows ``ids`` in ``model``.

    Each batch prefills the ``model.context_len`` frames that precede frame ``window_context``
    of its windows, then generates frames ``window_context .. window_context + horizon - 1``.
    Returns ``(frames (N, horizon, H, W, 3) uint8, rewards (N, horizon) float32)``.
    """
    batch = min(batch_size, len(ids))
    side = model.image_hw[0]
    env = DreamerV4VecEnvG1Relative(
        num_envs=batch, data=data_block, tokenizer_ckpt=model.tokenizer, dynamics_ckpt=model.dynamics,
        device=device, episode_length=horizon, img_size=side, compile_encoder=False, loader_seed=0,
        verify_pairing=verify_pairing, **env_args)
    try:
        if env.init_context_len != model.context_len:
            raise RuntimeError(f"the environment resolved init_context_len={env.init_context_len}, expected "
                               f"{model.context_len}")
        model.constants = {**{k: getattr(env, k) for k in GENERATION_KEYS}, "num_tau_levels": env.num_tau_levels,
                           "sources": {k: env.constant_sources[k] for k in GENERATION_KEYS}}
        first = window_context - model.context_len
        frames = np.empty((len(ids), horizon, side, side, 3), dtype=np.uint8)
        rewards = np.empty((len(ids), horizon), dtype=np.float32)
        for b, lo in enumerate(range(0, len(ids), batch)):
            chunk = list(ids[lo:lo + batch])
            n = len(chunk)
            data = stack_windows(dataset, chunk + [chunk[-1]] * (batch - n))  # pad the last batch
            context = slice(first, window_context)
            torch.manual_seed(1_000_003 * seed + b)  # the same noise for every model
            env.reset_from_context(data["image"][:, context], data["action"][:, context],
                                   data["proprio"][:, context] if model.uses_proprio else None)
            for k in range(horizon):
                obs, reward = env.step_command(data["action"][:, window_context + k])
                frames[lo:lo + n, k] = frames_to_uint8(obs[:n])
                rewards[lo:lo + n, k] = reward[:n].cpu().numpy()
            log.info("%s: windows %d-%d of %d generated", model.name, lo + 1, lo + n, len(ids))
        return frames, rewards
    finally:
        env.close()
        del env
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- rendering

BACKGROUND = (16, 16, 16)
TEXT = (235, 235, 235)
MUTED = (150, 150, 150)
LABEL = (110, 214, 128)
PREDICTION = (255, 196, 64)
GAP = 4


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except (TypeError, ImportError, OSError):  # Pillow without FreeType: the fixed bitmap font
        return ImageFont.load_default()


class Renderer:
    """Draws the video frames and contact sheets for panels of ``side`` pixels."""

    def __init__(self, side: int):
        self.side = side
        self.font = _font(max(10, round(side / 19)))
        self.line = self.font.getbbox("Ag")[3] + 4

    def _column(self, frame: np.ndarray, lines: Sequence[tuple[str, tuple]]):
        from PIL import Image, ImageDraw

        col = Image.new("RGB", (self.side, self.line * len(lines) + self.side), BACKGROUND)
        draw = ImageDraw.Draw(col)
        for i, (text, color) in enumerate(lines):
            draw.text((2, i * self.line + 1), text, fill=color, font=self.font)
        col.paste(Image.fromarray(frame), (0, self.line * len(lines)))
        return col

    def video_frame(self, title: str, columns: Sequence[tuple[np.ndarray, Sequence[tuple[str, tuple]]]]):
        """One video frame: ``title`` above the ``columns`` (frame, caption lines), padded to a
        multiple of 16 pixels (H.264 macroblocks)."""
        from PIL import Image, ImageDraw

        cols = [self._column(frame, lines) for frame, lines in columns]
        width = len(cols) * self.side + (len(cols) + 1) * GAP
        height = self.line + cols[0].height + 2 * GAP
        canvas = Image.new("RGB", (16 * math.ceil(width / 16), 16 * math.ceil(height / 16)), BACKGROUND)
        ImageDraw.Draw(canvas).text((GAP, GAP), title, fill=TEXT, font=self.font)
        for i, col in enumerate(cols):
            canvas.paste(col, (GAP + i * (self.side + GAP), self.line + GAP))
        return np.asarray(canvas)

    def contact_sheet(self, title: str, header: Sequence[str],
                      rows: Sequence[tuple[str, Sequence[np.ndarray], Sequence[tuple[str, tuple]]]]):
        """A grid: one row per ``(name, frames, captions)``, one column per ``header`` entry,
        each frame captioned underneath."""
        from PIL import Image, ImageDraw

        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        name_w = max(int(measure.textlength(name, font=self.font)) for name, _, _ in rows) + 2 * GAP
        cell_h = self.side + self.line
        width = name_w + len(header) * (self.side + GAP) + GAP
        height = 2 * self.line + len(rows) * (cell_h + GAP) + GAP
        sheet = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        draw.text((GAP, 2), title, fill=TEXT, font=self.font)
        for j, text in enumerate(header):
            draw.text((name_w + j * (self.side + GAP) + 2, self.line + 2), text, fill=MUTED, font=self.font)
        for i, (name, frames, captions) in enumerate(rows):
            y = 2 * self.line + i * (cell_h + GAP)
            draw.text((GAP, y + self.side // 2 - self.line // 2), name, fill=TEXT, font=self.font)
            for j, (frame, (text, color)) in enumerate(zip(frames, captions)):
                x = name_w + j * (self.side + GAP)
                sheet.paste(Image.fromarray(frame), (x, y))
                cell = Image.new("RGB", (self.side, self.line), BACKGROUND)
                ImageDraw.Draw(cell).text((2, 1), text, fill=color, font=self.font)
                sheet.paste(cell, (x, y + self.side))
        return sheet


def video_backend(requested: str = "auto") -> str:
    """``"imageio-ffmpeg"`` or ``"pyav"`` (MP4), else ``"gif"`` (Pillow) for ``"auto"``."""
    if requested not in VIDEO_FORMATS:
        raise ValueError(f"format must be one of {VIDEO_FORMATS}, got {requested!r}")
    if requested != "gif":
        try:
            import imageio_ffmpeg

            imageio_ffmpeg.get_ffmpeg_exe()
            return "imageio-ffmpeg"
        except (ImportError, RuntimeError):
            pass
        try:
            import av  # noqa: F401

            return "pyav"
        except ImportError:
            pass
        if requested == "mp4":
            raise RuntimeError("MP4 output needs imageio-ffmpeg (pip install imageio-ffmpeg, which bundles an ffmpeg "
                               "binary) or PyAV (pip install av); --format gif needs neither")
    return "gif"


def write_video(path_stem: Path, frames: Sequence[np.ndarray], fps: float, backend: str) -> Path:
    """Write ``frames`` (H, W, 3) uint8 as ``path_stem.mp4`` (H.264) or ``.gif``; returns the path."""
    if backend == "gif":
        from PIL import Image

        path = path_stem.with_name(path_stem.name + ".gif")
        images = [Image.fromarray(f) for f in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=max(1, round(1000 / fps)), loop=0)
        return path
    path = path_stem.with_name(path_stem.name + ".mp4")
    height, width = frames[0].shape[:2]
    if backend == "imageio-ffmpeg":
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps, codec="libx264",
                                             pix_fmt_out="yuv420p", quality=None, macro_block_size=16,
                                             output_params=["-crf", str(MP4_CRF)])
        writer.send(None)
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
        writer.close()
        return path
    import av

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": str(MP4_CRF)}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def sheet_steps(horizon: int, every: int) -> list[int]:
    """Generated steps (1-based) on the contact sheet: the first, every ``every``-th, the last."""
    return sorted({1, *range(every, horizon + 1, every), horizon})


def _fmt(value: float | None, label: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:g}" if label else f"{value:.3f}"


def render_window(stem: Path, renderer: Renderer, *, file: str, start: int, window_context: int,
                  gt: np.ndarray, labels: np.ndarray | None, models: Sequence[Model], frames: dict, rewards: dict,
                  horizon: int, context_frames: int, sheet_every: int, fps: float, backend: str) -> dict:
    """Video, contact sheet and per-step JSON of one window; returns the JSON record."""
    name = Path(file).name
    gt_name = "ground truth"
    title_base = f"{name}  start {start}"

    def label_at(t: int) -> float | None:
        return None if labels is None else float(labels[t])

    video = []
    for c in range(window_context - context_frames, window_context):
        lines = [(gt_name, TEXT), (f"label {_fmt(label_at(c), label=True)}", LABEL)]
        columns = [(gt[c], lines)] + [(gt[c], [(m.name, TEXT), ("context", MUTED)]) for m in models]
        video.append(renderer.video_frame(f"{title_base}  context {c - window_context}  frame {start + c}", columns))
    for k in range(horizon):
        t = window_context + k
        columns = [(gt[t], [(gt_name, TEXT), (f"label {_fmt(label_at(t), label=True)}", LABEL)])]
        columns += [(frames[m.key][k], [(m.name, TEXT), (f"reward {_fmt(float(rewards[m.key][k]))}", PREDICTION)])
                    for m in models]
        video.append(renderer.video_frame(f"{title_base}  step {k + 1}/{horizon}  frame {start + t}", columns))
    video_path = write_video(stem, video, fps, backend)

    steps = sheet_steps(horizon, sheet_every)
    rows = [("GT", [gt[window_context + s - 1] for s in steps],
             [(f"label {_fmt(label_at(window_context + s - 1), label=True)}", LABEL) for s in steps])]
    rows += [(m.key.upper(), [frames[m.key][s - 1] for s in steps],
              [(f"reward {_fmt(float(rewards[m.key][s - 1]))}", PREDICTION) for s in steps]) for m in models]
    legend = "   ".join(f"{m.key.upper()}: {m.name}" for m in models)
    sheet = renderer.contact_sheet(f"{title_base}   {legend}", [f"step {s}  frame {start + window_context + s - 1}"
                                                                for s in steps], rows)
    sheet_path = stem.with_name(stem.name + "_sheet.png")
    sheet.save(sheet_path)

    generated = labels[window_context:window_context + horizon] if labels is not None else None
    record = {
        "file": file,
        "start": start,
        "context_frames": {m.key: [start + window_context - m.context_len, start + window_context] for m in models},
        "first_generated_frame": start + window_context,
        "frame_index": list(range(start + window_context, start + window_context + horizon)),
        "reward_label": None if generated is None else [float(v) for v in generated],
        "reward_pred": {m.key: [float(v) for v in rewards[m.key]] for m in models},
        "return": {"label": None if generated is None else float(generated.sum()),
                   **{m.key: float(rewards[m.key].sum()) for m in models}},
        "video": video_path.name,
        "contact_sheet": sheet_path.name,
    }
    stem.with_name(stem.name + ".json").write_text(json.dumps(record, indent=1) + "\n")
    return record


# ----------------------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dirs", nargs="*", help="dataset directories (*.h5 files); or use --file-list")
    p.add_argument("--tokenizer", required=True, help="tokenizer: pretrained dir or training checkpoint (latest.pt)")
    p.add_argument("--dynamics", required=True, help="dynamics: pretrained dir or training checkpoint (latest.pt)")
    p.add_argument("--dynamics-b", help="a second dynamics model, shown as a third column")
    p.add_argument("--tokenizer-b", help="the second model's tokenizer (default: --tokenizer)")
    p.add_argument("--name-a", help="column title of --dynamics (default: its last two path components)")
    p.add_argument("--name-b", help="column title of --dynamics-b")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--num", type=int, default=8, help="windows to draw (default 8)")
    p.add_argument("--seed", type=int, default=0, help="selects the windows and the generation noise")
    p.add_argument("--horizon", type=int, default=64, help="generated frames per window (default 64, the RL episode)")
    p.add_argument("--windows", nargs="+", default=[], metavar="FILE:START",
                   help="these windows instead of --num random ones (START = first context frame)")
    p.add_argument("--include", nargs="+", default=["*.h5"], help="file-name globs to keep")
    p.add_argument("--exclude", nargs="+", default=[], help="file-name globs to drop")
    p.add_argument("--min-chunk-index", type=int, default=None, help="drop chunk_<N>.h5 with N below this")
    p.add_argument("--file-list", help="text file with one h5 path per line, instead of scanning the directories")
    p.add_argument("--color-order", choices=("bgr", "rgb"), default=None,
                   help="channel order of the h5 frames (default: the tokenizer's recorded order)")
    p.add_argument("--reward-key", default="rewards", help="h5 dataset of the reward labels (shown when present)")
    p.add_argument("--env-arg", action="append", default=[], metavar="KEY=VALUE",
                   help=f"generation constant for both models, one of {', '.join(GENERATION_KEYS)} (repeatable)")
    p.add_argument("--batch-size", type=int, default=16, help="windows generated in parallel (default 16)")
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--format", choices=VIDEO_FORMATS, default="auto", help="video format (auto: mp4 if possible)")
    p.add_argument("--fps", type=float, default=None, help="video frame rate (default: the dynamics' control_hz, else 10)")
    p.add_argument("--sheet-every", type=int, default=8, help="contact sheet: every k-th generated frame (default 8)")
    p.add_argument("--context-frames", type=int, default=0,
                   help="show the last N context frames before the generated ones (default 0)")
    p.add_argument("--no-verify-pairing", action="store_true",
                   help="skip the check of the tokenizer sha256 recorded in each dynamics config.json")
    args = p.parse_args(argv)
    if not args.data_dirs and not args.file_list:
        p.error("give dataset directories or --file-list")
    if args.num < 1 or args.horizon < 1 or args.batch_size < 1 or args.sheet_every < 1 or args.context_frames < 0:
        p.error("--num, --horizon, --batch-size and --sheet-every must be >= 1, --context-frames >= 0")
    if args.tokenizer_b and not args.dynamics_b:
        p.error("--tokenizer-b needs --dynamics-b")
    if args.fps is not None and args.fps <= 0:
        p.error("--fps must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    setup_logging(0)
    torch.set_float32_matmul_precision("high")  # the RL trainer's settings
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env_args = parse_env_args(args.env_arg)
    backend = video_backend(args.format)

    models = [Model("a", args.name_a or _default_name(args.dynamics), args.tokenizer, args.dynamics)]
    if args.dynamics_b:
        models.append(Model("b", args.name_b or _default_name(args.dynamics_b), args.tokenizer_b or args.tokenizer,
                            args.dynamics_b))
    for m in models:
        m.tokenizer_config = read_model_config(m.tokenizer, TOKENIZER_MODEL_TYPE)
        m.dynamics_config = read_model_config(m.dynamics, DYNAMICS_MODEL_TYPE)
        m.context_len = resolve_context_len(m.dynamics_config, env_args)
    image_hw = models[0].image_hw
    if any(m.image_hw != image_hw for m in models) or image_hw[0] != image_hw[1]:
        raise ValueError(f"the tokenizers' image_size must be square and equal, got {[m.image_hw for m in models]}")
    color_order = resolve_color_order(models, args.color_order)
    window_context = max(m.context_len for m in models)
    fps = args.fps or float(models[0].dynamics_config.get("control_hz") or 10)
    context_frames = min(args.context_frames, window_context)

    data_block = {"dirs": args.data_dirs or None, "file_list": args.file_list, "include_globs": args.include,
                  "exclude_globs": args.exclude, "min_chunk_index": args.min_chunk_index,
                  "color_order": args.color_order, "num_workers": 0, "pin_memory": False}
    files = list_h5_files(args.data_dirs or None, include_globs=args.include, exclude_globs=args.exclude,
                          min_chunk_index=args.min_chunk_index, file_list=args.file_list)
    dataset = SeedWindowDataset(files, window=window_context + args.horizon, stride=1, image_size=image_hw,
                                color_order=color_order, load_proprio=any(m.uses_proprio for m in models),
                                reward_key=None)
    ids = select_windows(dataset, args.num, args.seed, args.windows)
    windows = [dataset.window_info(i) for i in ids]
    log.info("%d windows of %d frames (%d context + %d generated) from %d files", len(ids), dataset.window,
             window_context, args.horizon, len(files))

    frames, rewards = {}, {}
    for m in models:
        frames[m.key], rewards[m.key] = generate(
            m, dataset, ids, horizon=args.horizon, window_context=window_context, batch_size=args.batch_size,
            seed=args.seed, device=device, data_block=data_block, env_args=env_args,
            verify_pairing=not args.no_verify_pairing)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(image_hw[0])
    records = []
    for j, (i, (file, start)) in enumerate(zip(ids, windows)):
        gt = frames_to_uint8(dataset[i]["image"])
        labels = read_labels(file, start, dataset.window, args.reward_key)
        stem = out / f"{j:03d}_{Path(file).stem}_{start:06d}"
        records.append(render_window(
            stem, renderer, file=file, start=start, window_context=window_context, gt=gt, labels=labels,
            models=models, frames={k: v[j] for k, v in frames.items()}, rewards={k: v[j] for k, v in rewards.items()},
            horizon=args.horizon, context_frames=context_frames, sheet_every=args.sheet_every, fps=fps,
            backend=backend))
    dataset.close()

    index = {
        "settings": {"seed": args.seed, "num": len(ids), "horizon": args.horizon, "batch_size": args.batch_size,
                     "window_frames": dataset.window, "window_context": window_context, "color_order": color_order,
                     "data": {k: v for k, v in data_block.items() if k not in ("num_workers", "pin_memory")},
                     "device": str(device), "video_backend": backend, "fps": fps},
        "models": {m.key: {"name": m.name, "tokenizer": str(m.tokenizer), "dynamics": str(m.dynamics),
                           "context_len": m.context_len, **m.constants} for m in models},
        "windows": records,
    }
    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    log.info("wrote %d windows to %s (%s videos)", len(records), out, "mp4" if backend != "gif" else "gif")
    return index


if __name__ == "__main__":
    main()
