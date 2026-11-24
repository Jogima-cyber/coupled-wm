# SPDX-License-Identifier: Apache-2.0
import logging
import os
import pickle

import h5py
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from conftest import frame_value
from coupled_wm.data import HDF5SequenceDataset, list_h5_files, segments_from_dones, window_starts


def paper_windows(dones, n_frames, window, stride):
    """The window loop of the paper's loaders (world-model HDF5SequenceDataset, env HDF5SequenceDatasetG1)."""
    out = []
    done_indices = np.where(dones > 0.5)[0]
    all_boundaries = list(done_indices)
    if len(all_boundaries) == 0 or all_boundaries[-1] != n_frames - 1:
        all_boundaries.append(n_frames - 1)
    start_idx = 0
    for end_idx in all_boundaries:
        segment_len = (end_idx - start_idx) + 1
        if segment_len >= window:
            for k in range((segment_len - window) // stride + 1):
                out.append(start_idx + k * stride)
        start_idx = end_idx + 1
    return out


def windows_of(ds):
    return [(os.path.basename(ds.window_info(i)[0]), ds.window_info(i)[1]) for i in range(len(ds))]


# ----------------------------------------------------------------------------- windowing


def test_windows_never_cross_dones(make_h5, tmp_path):
    make_h5(tmp_path / "a.h5", 20, dones_at=[4, 11, 19])
    ds = HDF5SequenceDataset(list_h5_files(tmp_path), window=5)
    assert windows_of(ds) == [("a.h5", s) for s in [0, 5, 6, 7, 12, 13, 14, 15]]
    for i in range(len(ds)):
        done = ds[i]["done"]
        assert done[:-1].sum() == 0  # a done may only sit on the last frame of a window


def test_windows_match_paper_loop_on_random_dones(make_h5, tmp_path):
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(1, 120))
        dones_at = sorted(set(rng.integers(0, n, int(rng.integers(0, 6))).tolist()))
        p = make_h5(tmp_path / f"t{trial}" / "x.h5", n, dones_at=dones_at)
        dones = np.zeros(n, np.float32)
        dones[dones_at] = 1
        for window, stride in [(1, 1), (7, 1), (16, 3), (40, 2)]:
            ref = paper_windows(dones, n, window, stride)
            got = window_starts(segments_from_dones(dones, n), window, stride)
            assert got == ref
            if ref:
                ds = HDF5SequenceDataset([p], window=window, stride=stride)
                assert [s for _, s in windows_of(ds)] == ref


def test_window_order_across_files(h5_pool):
    ds = HDF5SequenceDataset(list_h5_files(h5_pool), window=8)
    expected = (
        [("chunk_0000.h5", s) for s in [0, 1, 2]] + [("chunk_0000.h5", s) for s in range(10, 23)]
        + [("chunk_0001.h5", s) for s in range(5, 18)]
        + [("chunk_0002.h5", s) for s in range(0, 9)]
    )
    assert windows_of(ds) == expected


def test_short_episodes_skipped_and_counted(h5_pool):
    ds = HDF5SequenceDataset(list_h5_files(h5_pool), window=8)
    assert ds.summary["segments"] == 5
    assert ds.summary["segments_shorter_than_window"] == 1
    assert ds.summary["frames_in_short_segments"] == 5
    assert ds.summary["windows"] == len(ds) == 38


def test_window_longer_than_every_segment_raises(h5_pool):
    with pytest.raises(ValueError, match="no segment is at least"):
        HDF5SequenceDataset(list_h5_files(h5_pool), window=21)


def test_item_content_and_is_first(make_h5, tmp_path):
    make_h5(tmp_path / "a.h5", 12, file_id=3, action_dim=2, proprio_dim=2)
    ds = HDF5SequenceDataset(list_h5_files(tmp_path), window=4)
    item = ds[5]  # frames 5..8
    t = torch.arange(5, 9, dtype=torch.float32)
    assert set(item) == {"image", "action", "proprio", "reward", "done", "is_first"}
    assert item["image"].shape == (4, 3, 8, 8) and item["image"].dtype == torch.float32
    assert torch.allclose(item["action"][:, 0], 3000 + t)
    assert torch.allclose(item["proprio"][:, 1], -(3000 + t) - 0.01)
    assert torch.equal(item["reward"], (t % 3 == 0).float())
    assert item["reward"].shape == (4,) and item["done"].shape == (4,)
    assert torch.equal(item["is_first"], torch.tensor([1.0, 0.0, 0.0, 0.0]))


# ----------------------------------------------------------------------------- channels


@pytest.mark.parametrize("color_order", ["bgr", "rgb"])
def test_color_order(make_h5, tmp_path, color_order):
    make_h5(tmp_path / "a.h5", 6)
    ds = HDF5SequenceDataset(list_h5_files(tmp_path), window=6, color_order=color_order)
    img = ds[0]["image"]
    for t in range(6):
        for out_c in range(3):
            stored_c = 2 - out_c if color_order == "bgr" else out_c
            assert torch.all(img[t, out_c] == frame_value(t, stored_c) / 255.0)


def test_declared_channel_order_must_agree(make_h5, tmp_path):
    make_h5(tmp_path / "a.h5", 6, images_attrs={"channel_order": "rgb"})
    with pytest.raises(ValueError, match="channel order"):
        HDF5SequenceDataset(list_h5_files(tmp_path), window=4, color_order="bgr")
    assert len(HDF5SequenceDataset(list_h5_files(tmp_path), window=4, color_order="rgb")) == 3


def test_bad_color_order_rejected(h5_pool):
    with pytest.raises(ValueError, match="color_order"):
        HDF5SequenceDataset(list_h5_files(h5_pool), window=4, color_order="bgra")


# ----------------------------------------------------------------------------- contract errors


def test_missing_dones_raises(make_h5, tmp_path):
    p = make_h5(tmp_path / "a.h5", 10)
    with h5py.File(p, "a") as f:
        del f["dones"]
    with pytest.raises(KeyError, match="dones"):
        HDF5SequenceDataset([p], window=4)


def test_missing_proprio_raises_unless_not_loaded(make_h5, tmp_path):
    p = make_h5(tmp_path / "a.h5", 10, proprio_dim=None)
    with pytest.raises(KeyError, match="proprio"):
        HDF5SequenceDataset([p], window=4)
    item = HDF5SequenceDataset([p], window=4, load_proprio=False)[0]
    assert "proprio" not in item


def test_missing_reward_key_raises(make_h5, tmp_path):
    p = make_h5(tmp_path / "a.h5", 10, reward_key="reward")
    with pytest.raises(KeyError, match="rewards"):
        HDF5SequenceDataset([p], window=4)
    item = HDF5SequenceDataset([p], window=4, reward_key="reward")[0]
    assert item["reward"].shape == (4,)
    assert "reward" not in HDF5SequenceDataset([p], window=4, reward_key=None)[0]


def test_unreadable_file_raises(make_h5, tmp_path):
    make_h5(tmp_path / "good.h5", 10)
    (tmp_path / "bad.h5").write_bytes(b"not an hdf5 file")
    with pytest.raises(OSError, match="bad.h5"):
        HDF5SequenceDataset(list_h5_files(tmp_path), window=4)


def test_inconsistent_dims_raise(make_h5, tmp_path):
    make_h5(tmp_path / "a.h5", 10, action_dim=4)
    make_h5(tmp_path / "b.h5", 10, action_dim=5)
    with pytest.raises(ValueError, match="command dimension"):
        HDF5SequenceDataset(list_h5_files(tmp_path), window=4)


def test_mixed_image_sizes_need_resize(make_h5, tmp_path):
    make_h5(tmp_path / "a.h5", 10, image_hw=(8, 8))
    make_h5(tmp_path / "b.h5", 10, image_hw=(16, 16))
    with pytest.raises(ValueError, match="image size"):
        HDF5SequenceDataset(list_h5_files(tmp_path), window=4)
    ds = HDF5SequenceDataset(list_h5_files(tmp_path), window=4, image_size=4)
    assert ds[0]["image"].shape == ds[len(ds) - 1]["image"].shape == (4, 3, 4, 4)


def test_length_mismatch_raises(make_h5, tmp_path):
    p = make_h5(tmp_path / "a.h5", 10)
    with h5py.File(p, "a") as f:
        del f["rewards"]
        f.create_dataset("rewards", data=np.zeros(9, np.float32))
    with pytest.raises(ValueError, match="rewards"):
        HDF5SequenceDataset([p], window=4)


# ----------------------------------------------------------------------------- rewards


def test_reward_column_shape_normalised(make_h5, tmp_path):
    p = make_h5(tmp_path / "a.h5", 10, rewards=np.arange(10, dtype=np.float32)[:, None])
    ds = HDF5SequenceDataset([p], window=4)
    assert torch.equal(ds[2]["reward"], torch.tensor([2.0, 3.0, 4.0, 5.0]))


def test_reward_remap_is_opt_in_and_not_chained(make_h5, tmp_path):
    p = make_h5(tmp_path / "a.h5", 6, rewards=np.array([0, 1, 2, 10, 1, 0], np.float32))
    assert torch.equal(HDF5SequenceDataset([p], window=6)[0]["reward"], torch.tensor([0.0, 1, 2, 10, 1, 0]))
    ds = HDF5SequenceDataset([p], window=6, reward_remap={10.0: 2.0})
    assert torch.equal(ds[0]["reward"], torch.tensor([0.0, 1, 2, 2, 1, 0]))
    ds = HDF5SequenceDataset([p], window=6, reward_remap={1: 2, 2: 3})
    assert torch.equal(ds[0]["reward"], torch.tensor([0.0, 2, 3, 10, 2, 0]))


def test_reward_override_replaces(make_h5, tmp_path):
    src = tmp_path / "src"
    ov = tmp_path / "ov"
    make_h5(src / "a.h5", 8)
    make_h5(src / "b.h5", 8, file_id=1)
    ov.mkdir()
    for name, val in [("a.h5", 0.25), ("b.h5", 10.0)]:
        with h5py.File(ov / name, "w") as f:
            f["reward"] = np.full(8, val, np.float32)
            f["is_done"] = np.ones(8, np.float32)  # ignored: the override replaces, no mixing
    ds = HDF5SequenceDataset(list_h5_files(src), window=8, reward_override_dir=ov, reward_remap={10.0: 2.0})
    assert torch.equal(ds[0]["reward"], torch.full((8,), 0.25))
    assert torch.equal(ds[1]["reward"], torch.full((8,), 2.0))


def test_reward_override_missing_file_raises(make_h5, tmp_path):
    src = tmp_path / "src"
    ov = tmp_path / "ov"
    make_h5(src / "a.h5", 8)
    make_h5(src / "b.h5", 8)
    ov.mkdir()
    with h5py.File(ov / "a.h5", "w") as f:
        f["reward"] = np.zeros(8, np.float32)
    with pytest.raises(FileNotFoundError, match="b.h5"):
        HDF5SequenceDataset(list_h5_files(src), window=4, reward_override_dir=ov)


def test_reward_override_length_mismatch_raises(make_h5, tmp_path):
    src = tmp_path / "src"
    ov = tmp_path / "ov"
    make_h5(src / "a.h5", 8)
    ov.mkdir()
    with h5py.File(ov / "a.h5", "w") as f:
        f["reward"] = np.zeros(7, np.float32)
    with pytest.raises(ValueError, match="override has 7 frames"):
        HDF5SequenceDataset(list_h5_files(src), window=4, reward_override_dir=ov)


def test_reward_override_does_not_need_source_reward(make_h5, tmp_path):
    src = tmp_path / "src"
    ov = tmp_path / "ov"
    make_h5(src / "a.h5", 8, reward_key=None)
    ov.mkdir()
    with h5py.File(ov / "a.h5", "w") as f:
        f["reward"] = np.ones(8, np.float32)
    assert torch.equal(HDF5SequenceDataset([src / "a.h5"], window=8, reward_override_dir=ov)[0]["reward"],
                       torch.ones(8))
    with pytest.raises(ValueError, match="reward_key is None"):
        HDF5SequenceDataset([src / "a.h5"], window=8, reward_key=None, reward_override_dir=ov)


def test_action_dim_checked_before_the_cache(h5_pool, monkeypatch):
    files = list_h5_files(h5_pool)
    assert HDF5SequenceDataset(files, window=8, action_dim=4).action_dim == 4

    def no_cache(*args, **kwargs):
        raise AssertionError("the cache was built before the dimension check")

    monkeypatch.setattr(HDF5SequenceDataset, "_build_cache", no_cache)
    with pytest.raises(ValueError, match="expected action_dim=5"):
        HDF5SequenceDataset(files, window=8, action_dim=5, cache=True)
    with pytest.raises(ValueError, match="load_action"):
        HDF5SequenceDataset(files, window=8, action_dim=4, load_action=False)


# ----------------------------------------------------------------------------- resize / cache


def test_resize_is_the_env_op(make_h5, tmp_path):
    p = make_h5(tmp_path / "a.h5", 10, image_hw=(32, 32))
    raw = HDF5SequenceDataset([p], window=10)[0]["image"]
    resized = HDF5SequenceDataset([p], window=10, image_size=8)
    assert resized.resized and resized.native_image_sizes == [(32, 32)]
    small = resized[0]["image"]
    assert torch.equal(small, F.interpolate(raw, size=(8, 8), mode="bilinear", align_corners=False))
    kept = HDF5SequenceDataset([p], window=10, image_size=32)
    assert not kept.resized and torch.equal(kept[0]["image"], raw)


@pytest.mark.parametrize("workers", [0, 2])
def test_cache_matches_streaming(h5_pool, workers):
    files = list_h5_files(h5_pool)
    stream = HDF5SequenceDataset(files, window=8, image_size=4)
    exact = HDF5SequenceDataset(files, window=8, image_size=4, cache=True, cache_dtype="float32", cache_workers=workers)
    quant = HDF5SequenceDataset(files, window=8, image_size=4, cache=True, cache_workers=workers)
    native = HDF5SequenceDataset(files, window=8, cache=True, cache_workers=workers)
    native_stream = HDF5SequenceDataset(files, window=8)
    assert windows_of(stream) == windows_of(exact) == windows_of(quant)
    for i in range(len(stream)):
        a, b, q = stream[i], exact[i], quant[i]
        for k in a:
            if k != "image":
                assert torch.equal(a[k], b[k]) and torch.equal(a[k], q[k]), k
        assert torch.equal(a["image"], b["image"])
        assert (a["image"] - q["image"]).abs().max() <= 0.5 / 255 + 1e-7
        assert torch.equal(native[i]["image"], native_stream[i]["image"])


def test_cache_memory_is_per_process(h5_pool, monkeypatch, caplog):
    from coupled_wm.data import h5_dataset

    files = list_h5_files(h5_pool)
    nbytes = 71 * 3 * 4 * 4 * 4  # frames x 3 x 4 x 4 float32
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setattr(h5_dataset, "available_memory", lambda: 3 * nbytes)
    with caplog.at_level(logging.INFO, logger="coupled_wm.data.h5_dataset"):
        HDF5SequenceDataset(files, window=8, image_size=4, cache=True, cache_dtype="float32", cache_workers=0)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "x 4 process(es)" in warnings[0] and "uint8" in warnings[0]
    assert any("4 process(es) on this node" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    caplog.clear()
    monkeypatch.setattr(h5_dataset, "available_memory", lambda: 4 * nbytes)
    with caplog.at_level(logging.INFO, logger="coupled_wm.data.h5_dataset"):
        HDF5SequenceDataset(files, window=8, image_size=4, cache=True, cache_dtype="float32", cache_workers=0)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_cache_does_not_touch_global_rng(h5_pool):
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    HDF5SequenceDataset(list_h5_files(h5_pool), window=8, image_size=4, cache=True, cache_workers=2)
    assert torch.equal(torch.rand(3), expected)


def test_cache_items_are_copies(h5_pool):
    ds = HDF5SequenceDataset(list_h5_files(h5_pool), window=8, cache=True, cache_dtype="float32", cache_workers=0)
    item = ds[0]
    item["image"].zero_()
    item["action"].zero_()
    assert ds[0]["image"].abs().sum() > 0 and ds[0]["action"].abs().sum() > 0


# ----------------------------------------------------------------------------- handles / workers


def test_dataloader_workers_match_direct_reads(h5_pool):
    ds = HDF5SequenceDataset(list_h5_files(h5_pool), window=8)
    _ = ds[0]  # open a handle in the parent before the workers fork
    loader = DataLoader(ds, batch_size=5, shuffle=False, num_workers=2, persistent_workers=True)
    batches = list(loader)
    direct = torch.stack([ds[i]["action"] for i in range(len(ds))])
    assert torch.equal(torch.cat([b["action"] for b in batches])[: len(direct)], direct)


def test_seeded_shuffle_is_deterministic(h5_pool):
    def first_starts():
        ds = HDF5SequenceDataset(list_h5_files(h5_pool), window=8)
        torch.manual_seed(7)
        batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)))
        return batch["action"][:, 0, 0]

    assert torch.equal(first_starts(), first_starts())


@pytest.mark.parametrize("bad", [0, -1])
def test_max_open_files_must_be_positive(h5_pool, bad):
    with pytest.raises(ValueError, match="max_open_files"):
        HDF5SequenceDataset(list_h5_files(h5_pool), window=8, max_open_files=bad)


def test_handles_are_lru_bounded_and_not_pickled(h5_pool):
    ds = HDF5SequenceDataset(list_h5_files(h5_pool), window=8, max_open_files=1)
    for i in range(len(ds)):
        ds[i]
        assert len(ds._handles) <= 1
    clone = pickle.loads(pickle.dumps(ds))
    assert len(clone._handles) == 0
    assert torch.equal(clone[len(ds) - 1]["action"], ds[len(ds) - 1]["action"])
    ds.close()


def test_directory_argument(h5_pool):
    assert len(HDF5SequenceDataset(str(h5_pool), window=8)) == 38


def test_raw_window_and_window_frame_values(h5_pool):
    ds = HDF5SequenceDataset(list_h5_files(h5_pool), window=8, color_order="bgr", reward_key=None)
    for idx in (0, 7, len(ds) - 1):
        path, start = ds.window_info(idx)
        raw = ds.raw_window(idx)
        with h5py.File(path, "r") as f:
            assert np.array_equal(raw["images"], f["images"][start:start + 8][..., ::-1])
            assert np.array_equal(raw["commands"], f["commands"][start:start + 8])
            assert np.array_equal(raw["proprio"], f["proprio"][start:start + 8])
        assert raw["images"].dtype == np.uint8
        item = ds[idx]
        assert torch.equal(item["action"], torch.from_numpy(raw["commands"]).float())
        assert torch.equal(item["image"], torch.from_numpy(np.ascontiguousarray(raw["images"])).permute(0, 3, 1, 2).float() / 255)
    last = ds.window_frame_values("commands", 7)
    assert last.shape == (len(ds), ds.action_dim)
    assert np.array_equal(last, np.stack([ds.raw_window(i)["commands"][7] for i in range(len(ds))]))
    with pytest.raises(ValueError, match="offset"):
        ds.window_frame_values("commands", 8)


def test_load_action_false_allows_action_free_files(make_h5, tmp_path, h5_pool):
    p = make_h5(tmp_path / "video.h5", 12)
    with h5py.File(p, "a") as f:
        del f["commands"]
    with pytest.raises(KeyError, match="commands"):
        HDF5SequenceDataset([p], window=4)
    for cache in (False, True):
        ds = HDF5SequenceDataset([p], window=4, load_action=False, reward_key=None, cache=cache, cache_workers=0)
        assert ds.action_dim is None and "action" not in ds[0] and "action" not in ds.raw_window(0)
        assert ds[0]["proprio"].shape == (4, 3)
    full = HDF5SequenceDataset(list_h5_files(h5_pool), window=8)
    video = HDF5SequenceDataset(list_h5_files(h5_pool), window=8, load_action=False)
    assert windows_of(full) == windows_of(video)
    a, b = full[5], video[5]
    assert set(a) - set(b) == {"action"} and all(torch.equal(a[k], b[k]) for k in b)
