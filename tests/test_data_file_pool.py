# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from coupled_wm.data import list_h5_files


def names(files):
    return [os.path.basename(f) for f in files]


@pytest.fixture
def g1_like_dir(tmp_path, make_h5):
    d = tmp_path / "world_model"
    for name in ["chunk_0021.h5", "chunk_0003.h5", "chunk_0100.h5", "patch_2_chunk_0004.h5",
                 "patch_chunk_0001.h5", "shard_notes.h5"]:
        make_h5(d / name, 4)
    (d / "README.txt").write_text("not data")
    return d


def test_sorted_and_absolute(g1_like_dir):
    files = list_h5_files(g1_like_dir)
    assert names(files) == sorted(names(files))
    assert len(files) == 6  # no 'shard' substring filter
    assert all(os.path.isabs(f) for f in files)


def test_min_chunk_index_fullmatch(g1_like_dir):
    files = list_h5_files(g1_like_dir, min_chunk_index=21)
    assert names(files) == ["chunk_0021.h5", "chunk_0100.h5", "patch_2_chunk_0004.h5",
                            "patch_chunk_0001.h5", "shard_notes.h5"]
    assert names(list_h5_files(g1_like_dir, min_chunk_index=0)) == names(list_h5_files(g1_like_dir))


def test_paper_rl_pool_recipe(g1_like_dir):
    files = list_h5_files(g1_like_dir, include_globs=["chunk_*.h5", "patch*"], exclude_globs=["patch*"],
                          min_chunk_index=21)
    assert names(files) == ["chunk_0021.h5", "chunk_0100.h5"]


def test_include_exclude_globs(g1_like_dir):
    assert names(list_h5_files(g1_like_dir, include_globs=["patch*"])) == ["patch_2_chunk_0004.h5",
                                                                           "patch_chunk_0001.h5"]
    assert names(list_h5_files(g1_like_dir, exclude_globs=["patch*", "shard*"])) == [
        "chunk_0003.h5", "chunk_0021.h5", "chunk_0100.h5"]


def test_empty_pool_raises(g1_like_dir, tmp_path):
    with pytest.raises(ValueError, match="no .h5 files"):
        list_h5_files(g1_like_dir, include_globs=["*.hdf5"])
    with pytest.raises(FileNotFoundError):
        list_h5_files(tmp_path / "missing")


def test_multiple_dirs_keep_dir_order(tmp_path, make_h5):
    a, b = tmp_path / "b_dir", tmp_path / "a_dir"
    make_h5(a / "chunk_0001.h5", 4)
    make_h5(b / "chunk_0000.h5", 4)
    make_h5(b / "chunk_0002.h5", 4)
    files = list_h5_files([a, b])
    assert [os.path.relpath(f, tmp_path) for f in files] == [
        "b_dir/chunk_0001.h5", "a_dir/chunk_0000.h5", "a_dir/chunk_0002.h5"]


def test_symlinks_followed_and_named_by_link(tmp_path, make_h5):
    target = make_h5(tmp_path / "store" / "chunk_0005.h5", 4)
    d = tmp_path / "pool"
    d.mkdir()
    os.symlink(target, d / "patch_3_chunk_0005.h5")
    files = list_h5_files(d, min_chunk_index=21)
    assert names(files) == ["patch_3_chunk_0005.h5"]


def test_dangling_symlink_raises(tmp_path, make_h5):
    d = tmp_path / "pool"
    make_h5(d / "chunk_0000.h5", 4)
    os.symlink(tmp_path / "gone.h5", d / "patch_1_chunk_0000.h5")
    with pytest.raises(FileNotFoundError, match="dangling symlink"):
        list_h5_files(d)


def test_dangling_symlink_outside_the_pool_is_ignored(tmp_path, make_h5):
    """The paper RL pool (exclude patch*, min_chunk_index 21) of a copy whose patch_* links dangle."""
    d = tmp_path / "pool"
    make_h5(d / "chunk_0030.h5", 4)
    os.symlink(tmp_path / "gone.h5", d / "patch_4_chunk_0009.h5")
    os.symlink(tmp_path / "gone2.h5", d / "chunk_0005.h5")
    assert names(list_h5_files(d, exclude_globs=["patch*"], min_chunk_index=21)) == ["chunk_0030.h5"]
    with pytest.raises(FileNotFoundError, match="chunk_0005.h5"):
        list_h5_files(d, exclude_globs=["patch*"])  # the dangling chunk_0005 stays in this pool


def test_duplicate_real_file_raises(tmp_path, make_h5):
    target = make_h5(tmp_path / "a" / "chunk_0000.h5", 4)
    (tmp_path / "b").mkdir()
    os.symlink(target, tmp_path / "b" / "demo_chunk_0000.h5")
    with pytest.raises(ValueError, match="same file twice"):
        list_h5_files([tmp_path / "a", tmp_path / "b"])


def test_file_list_sequence_and_text(tmp_path, make_h5):
    d = tmp_path / "pool"
    for n in ["chunk_0000.h5", "chunk_0001.h5", "chunk_0002.h5"]:
        make_h5(d / n, 4)
    assert names(list_h5_files(d, file_list=["chunk_0002.h5", "chunk_0000.h5"])) == ["chunk_0002.h5",
                                                                                      "chunk_0000.h5"]
    lst = d / "pool.txt"
    lst.write_text("# paper pool\nchunk_0001.h5\n\n" + str(d / "chunk_0002.h5") + "  # absolute\n")
    assert names(list_h5_files(None, file_list=lst)) == ["chunk_0001.h5", "chunk_0002.h5"]
    with pytest.raises(FileNotFoundError, match="chunk_0009"):
        list_h5_files(d, file_list=["chunk_0009.h5"])


def test_file_list_entries_dropped_by_a_filter_raise(tmp_path, make_h5):
    d = tmp_path / "pool"
    for n in ["chunk_0003.h5", "chunk_0030.h5", "ep_7.hdf5"]:
        make_h5(d / n, 4)
    with pytest.raises(ValueError, match=r"dropped by the name filters.*chunk_0003\.h5"):
        list_h5_files(d, file_list=["chunk_0030.h5", "chunk_0003.h5"], min_chunk_index=21)
    with pytest.raises(ValueError, match="ep_7.hdf5"):
        list_h5_files(d, file_list=["chunk_0030.h5", "ep_7.hdf5"])  # default include_globs: *.h5
    assert names(list_h5_files(d, file_list=["ep_7.hdf5", "chunk_0030.h5"], include_globs=["*.h5", "*.hdf5"])) == [
        "ep_7.hdf5", "chunk_0030.h5"]


def test_file_list_ambiguous_raises(tmp_path, make_h5):
    make_h5(tmp_path / "a" / "chunk_0000.h5", 4)
    make_h5(tmp_path / "b" / "chunk_0000.h5", 4)
    with pytest.raises(ValueError, match="ambiguous"):
        list_h5_files([tmp_path / "a", tmp_path / "b"], file_list=["chunk_0000.h5"])
