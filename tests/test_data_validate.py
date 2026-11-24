# SPDX-License-Identifier: Apache-2.0
import h5py
import numpy as np
import pytest

from coupled_wm.data.validate import main


def test_valid_dataset(h5_pool, capsys):
    rc = main([str(h5_pool), "--image-size", "8", "--proprio-dim", "3", "--action-dim", "4",
               "--window", "8", "--read-images"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "files: 3" in out and "frames: 71" in out
    assert "reward 'rewards': values [0, 1]" in out
    assert "window 8: 38 windows; 1 segments shorter" in out
    assert "other datasets (not read by the loaders): none" in out
    assert "files do not end with done=1" in out  # chunk_0002 has no done at all
    assert "3 of 3 files declare no channel order" in out
    assert out.strip().endswith("OK with 2 warning(s): read them before training")


def test_clean_dataset_has_no_warning(tmp_path, make_h5, capsys):
    for i in range(2):
        make_h5(tmp_path / f"chunk_{i:04d}.h5", 12, images_attrs={"channel_order": "bgr"},
                extra={"is_demo": np.zeros(12, np.float32)})
    assert main([str(tmp_path), "--window", "4"]) == 0
    out = capsys.readouterr().out
    assert "WARNING" not in out and out.strip().endswith("\nOK")
    assert "other datasets (not read by the loaders): is_demo in 2 files" in out


def test_contract_violations_reported(h5_pool, make_h5, capsys):
    make_h5(h5_pool / "chunk_0003.h5", 10, action_dim=5)
    with h5py.File(h5_pool / "chunk_0000.h5", "a") as f:
        del f["proprio"]
    rc = main([str(h5_pool), "--image-size", "16", "--action-dim", "4"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "missing dataset(s) ['proprio']" in out
    assert "image size (H, W) is (8, 8), expected (16, 16)" in out
    assert "files disagree on action dim" in out


def test_non_binary_dones_and_nan(tmp_path, make_h5, capsys):
    p = make_h5(tmp_path / "a.h5", 6, rewards=np.array([0, 1, np.nan, 0, 0, 0], np.float32))
    with h5py.File(p, "a") as f:
        f["dones"][2] = 0.5
    assert main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "'dones' holds values other than 0/1" in out and "'rewards' has non-finite values" in out


def test_empty_pool_is_an_error(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1
    assert "no .h5 files" in capsys.readouterr().out


def test_first_frame_dones_are_flagged(tmp_path, make_h5, capsys):
    """An is_first-style producer: done on the first frame of each episode, none at the end."""
    make_h5(tmp_path / "a.h5", 100, dones_at=[0, 50])
    assert main([str(tmp_path), "--window", "32"]) == 0
    out = capsys.readouterr().out
    assert "1 files have done=1 on frame 0" in out and "is_first" in out
    assert "files do not end with done=1" in out and "OK with" in out


def test_consecutive_dones_are_flagged(tmp_path, make_h5, capsys):
    make_h5(tmp_path / "a.h5", 20, dones_at=[9, 10, 19])
    assert main([str(tmp_path), "--window", "4"]) == 0
    assert "1 1-frame segments after frame 0 in 1 files" in capsys.readouterr().out


def test_no_action_for_action_free_video(tmp_path, make_h5, capsys):
    p = make_h5(tmp_path / "video.h5", 12, reward_key=None)
    with h5py.File(p, "a") as f:
        del f["commands"]
    assert main([str(tmp_path), "--no-reward"]) == 1
    assert "missing dataset(s) ['commands']" in capsys.readouterr().out
    assert main([str(tmp_path), "--no-reward", "--no-action", "--window", "4"]) == 0
    out = capsys.readouterr().out
    assert "action dim" not in out and "commands" not in out
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--no-action", "--action-dim", "4"])
