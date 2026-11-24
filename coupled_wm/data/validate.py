# SPDX-License-Identifier: Apache-2.0
"""Check a dataset against the HDF5 format contract and print a summary.

    python -m coupled_wm.data.validate data/g1/new_wm_h5_segmented_small/world_model \
        --image-size 256 --proprio-dim 50 --action-dim 22 --window 32 64

The contract is the key table of :mod:`coupled_wm.data.h5_dataset`. ``--no-action``,
``--no-proprio`` and ``--no-reward`` drop a requirement (``--no-action`` for tokenizer-only
video). Exit status 0 when every file satisfies the contract, 1 otherwise. Warnings flag data
that loads but is likely mislabelled (e.g. ``dones`` marking the first frame of an episode).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import h5py
import numpy as np

from coupled_wm.data.h5_dataset import list_h5_files, scan_file, window_starts

_MAX_REWARD_VALUES = 16


def _update_range(rng: dict, key: str, arr: np.ndarray) -> None:
    lo, hi = arr.min(0), arr.max(0)
    if key in rng and rng[key][0].shape == lo.shape:
        lo, hi = np.minimum(rng[key][0], lo), np.maximum(rng[key][1], hi)
    rng[key] = (lo, hi)


def _fmt(values) -> str:
    return "[" + ", ".join(f"{v:.4g}" for v in values) + "]"


def validate(args: argparse.Namespace) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    reward_key = None if args.no_reward else args.reward_key
    load_action = not args.no_action
    read_keys = {"images", "dones"}
    read_keys |= {"commands"} if load_action else set()
    read_keys |= set() if args.no_proprio else {"proprio"}
    read_keys |= {reward_key} if reward_key is not None else set()

    files = list_h5_files(
        args.data_dirs or None, include_globs=args.include, exclude_globs=args.exclude,
        min_chunk_index=args.min_chunk_index, file_list=args.file_list,
    )
    n_links = sum(os.path.islink(f) for f in files)

    infos = []
    ranges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reward_values: set[float] | None = set()
    reward_frames = reward_nonzero = 0
    other = Counter()
    color_attr = 0
    no_final_done, first_done, one_frame = [], [], Counter()
    for path in files:
        try:
            info = scan_file(path, color_order=args.color_order, load_proprio=not args.no_proprio,
                             reward_key=reward_key, load_action=load_action)
        except (OSError, KeyError, ValueError) as e:
            errors.append(str(e))
            continue
        infos.append(info)
        name = os.path.basename(path)
        n_one = sum(1 for s, e in info.segments if s == e and s > 0)
        if n_one:
            one_frame[name] = n_one
        with h5py.File(path, "r") as f:
            dones = np.asarray(f["dones"][()]).reshape(-1)
            if dones[-1] <= 0.5:
                no_final_done.append(name)
            if dones[0] > 0.5:
                first_done.append(name)
            if not np.isin(dones, (0, 1)).all():
                errors.append(f"{path}: 'dones' holds values other than 0/1: {np.unique(dones)[:8]}")
            if load_action:
                cmds = np.asarray(f["commands"][()], dtype=np.float64)
                if not np.isfinite(cmds).all():
                    errors.append(f"{path}: 'commands' has non-finite values")
                _update_range(ranges, "commands", cmds)
            if not args.no_proprio:
                prop = np.asarray(f["proprio"][()], dtype=np.float64)
                if not np.isfinite(prop).all():
                    errors.append(f"{path}: 'proprio' has non-finite values")
                _update_range(ranges, "proprio", prop)
            if reward_key is not None:
                r = np.asarray(f[reward_key][()], dtype=np.float64).reshape(-1)
                if not np.isfinite(r).all():
                    errors.append(f"{path}: '{reward_key}' has non-finite values")
                reward_frames += r.size
                reward_nonzero += int(np.count_nonzero(r))
                if reward_values is not None:
                    reward_values.update(np.unique(r).tolist())
                    if len(reward_values) > _MAX_REWARD_VALUES:
                        reward_values = None
            other.update(k for k in f.keys() if k not in read_keys)
            if any(k in f["images"].attrs or k in f.attrs for k in ("channel_order", "color_order")):
                color_attr += 1
            if args.read_images:
                images = f["images"]
                for s in range(0, images.shape[0], 256):
                    images[s: s + 256]  # raises on a corrupt chunk

    def check_same(label, values, expected):
        values = set(values)
        if len(values) > 1:
            errors.append(f"files disagree on {label}: {sorted(values)}")
        elif expected is not None and values and values != {expected}:
            errors.append(f"{label} is {values.pop()}, expected {expected}")

    check_same("image size (H, W)", [i.image_hw for i in infos],
               (args.image_size, args.image_size) if args.image_size else None)
    if load_action:
        check_same("action dim", [i.action_dim for i in infos], args.action_dim)
    if not args.no_proprio:
        check_same("proprio dim", [i.proprio_dim for i in infos], args.proprio_dim)

    print(f"files: {len(files)} ({n_links} symlinks), readable and complete: {len(infos)}")
    if infos:
        frames = sum(i.n_frames for i in infos)
        seg_lens = np.array([e - s + 1 for i in infos for s, e in i.segments])
        print(f"frames: {frames}")
        print(f"segments: {len(seg_lens)}  length min/median/max = {seg_lens.min()}/{int(np.median(seg_lens))}/{seg_lens.max()}")
        print(f"image size (H, W): {sorted({i.image_hw for i in infos})}  channel-order attribute in {color_attr} files "
              f"(assumed {args.color_order})")
        if load_action:
            print(f"action dim: {sorted({i.action_dim for i in infos})}")
        if "commands" in ranges:
            print(f"  commands min: {_fmt(ranges['commands'][0])}")
            print(f"  commands max: {_fmt(ranges['commands'][1])}")
        if "proprio" in ranges:
            lo, hi = ranges["proprio"]
            const = np.flatnonzero(lo == hi)
            print(f"proprio dim: {sorted({i.proprio_dim for i in infos})}  range [{lo.min():.4g}, {hi.max():.4g}]"
                  f"  constant dims: {const.tolist() if len(const) else 'none'}")
        if reward_key is not None:
            vals = "more than %d distinct values" % _MAX_REWARD_VALUES if reward_values is None else _fmt(sorted(reward_values))
            rate = reward_nonzero / max(reward_frames, 1)
            print(f"reward '{reward_key}': values {vals}, nonzero on {rate:.2%} of frames")
        print("other datasets (not read by the loaders): "
              + (", ".join(f"{k} in {n} files" for k, n in sorted(other.items())) or "none"))
        for w in args.window:
            n_win = sum(len(window_starts(i.segments, w, 1)) for i in infos)
            short = seg_lens[seg_lens < w]
            print(f"window {w}: {n_win} windows; {len(short)} segments shorter than the window ({short.sum()} frames) are skipped")

        if first_done:
            warnings.append(
                f"{len(first_done)} files have done=1 on frame 0, e.g. {first_done[:5]}. A done marks the LAST frame "
                "of a segment, so frame 0 becomes a 1-frame segment. If the producer marks the FIRST frame of each "
                "episode (is_first), every episode is read one frame late and windows run into the next episode: "
                "move each flag to the frame before it."
            )
        if one_frame:
            warnings.append(
                f"{sum(one_frame.values())} 1-frame segments after frame 0 in {len(one_frame)} files "
                f"(consecutive dones), e.g. {sorted(one_frame)[:5]}; they never yield a window"
            )
        if no_final_done:
            warnings.append(
                f"{len(no_final_done)} files do not end with done=1, e.g. {no_final_done[:5]}. The loader ends their "
                "last segment at the end of the file (windows never cross files), so an episode that continues in "
                "another file is cut there."
            )
        if color_attr < len(infos):
            warnings.append(
                f"{len(infos) - color_attr} of {len(infos)} files declare no channel order, so nothing checks that "
                f"they are {args.color_order} as assumed here and by every stage's color_order setting. Set "
                "images.attrs['channel_order'] = 'bgr' or 'rgb' to make the loaders check it."
            )

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors[: args.max_errors]:
        print(f"ERROR: {e}")
    if len(errors) > args.max_errors:
        print(f"... and {len(errors) - args.max_errors} more errors")
    if errors:
        print(f"FAILED: {len(errors)} errors")
    elif warnings:
        print(f"OK with {len(warnings)} warning(s): read them before training")
    else:
        print("OK")
    return 0 if not errors else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dirs", nargs="*", help="directories holding the .h5 files")
    p.add_argument("--file-list", default=None,
                   help="text file with one .h5 path per line (instead of a scan); the name filters must keep "
                        "every listed file")
    p.add_argument("--include", nargs="+", default=["*.h5"], help="file-name globs to keep")
    p.add_argument("--exclude", nargs="+", default=[], help="file-name globs to drop")
    p.add_argument("--min-chunk-index", type=int, default=None, help="drop chunk_<N>.h5 with N below this")
    p.add_argument("--image-size", type=int, default=None, help="required square frame size")
    p.add_argument("--proprio-dim", type=int, default=None)
    p.add_argument("--action-dim", type=int, default=None)
    p.add_argument("--no-action", action="store_true", help="do not require 'commands' (tokenizer-only video)")
    p.add_argument("--no-proprio", action="store_true", help="do not require 'proprio'")
    p.add_argument("--reward-key", default="rewards")
    p.add_argument("--no-reward", action="store_true", help="do not require a reward dataset")
    p.add_argument("--color-order", choices=("bgr", "rgb"), default="bgr", help="channel order the loaders will assume")
    p.add_argument("--window", type=int, nargs="+", default=[32, 64], help="window lengths to report")
    p.add_argument("--read-images", action="store_true", help="read every frame (slow) to catch corrupt chunks")
    p.add_argument("--max-errors", type=int, default=20)
    args = p.parse_args(argv)
    if not args.data_dirs and args.file_list is None:
        p.error("give at least one data dir or --file-list")
    if args.no_action and args.action_dim is not None:
        p.error("--action-dim checks the commands, which --no-action skips")
    try:
        return validate(args)
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
