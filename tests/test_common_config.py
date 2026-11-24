# SPDX-License-Identifier: Apache-2.0
import pytest

from coupled_wm.common.config import apply_overrides, load_config, save_config

CFG = """
run:
  name: test
  seed: 0
optim:
  lr: 1e-4
  betas: [0.9, 0.95]
  warmup_updates: 100
data:
  dirs: [data/g1/world_model]
  root: ${DATA_ROOT:-data}/g1
  min_chunk_index: null
"""


@pytest.fixture
def cfg_path(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(CFG)
    return p


def test_scientific_floats_are_floats(cfg_path, monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    cfg = load_config(cfg_path)
    assert cfg["optim"]["lr"] == pytest.approx(1e-4) and isinstance(cfg["optim"]["lr"], float)
    assert cfg["optim"]["warmup_updates"] == 100 and isinstance(cfg["optim"]["warmup_updates"], int)


def test_overrides(cfg_path, monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    cfg = load_config(cfg_path, ["optim.lr=3e-4", "optim.betas.1=0.99", "data.dirs=[a, b]",
                                 "data.min_chunk_index=21", "run.name=x=y", "+run.extra.depth=2"])
    assert cfg["optim"]["lr"] == pytest.approx(3e-4)
    assert cfg["optim"]["betas"] == [0.9, 0.99]
    assert cfg["data"]["dirs"] == ["a", "b"]
    assert cfg["data"]["min_chunk_index"] == 21
    assert cfg["run"]["name"] == "x=y"
    assert cfg["run"]["extra"] == {"depth": 2}


@pytest.mark.parametrize("bad", ["optim.lrr=1", "nope.x=1", "optim.betas.5=1", "optim.lr.x=1", "optim.lr"])
def test_unknown_override_keys_raise(cfg_path, bad):
    with pytest.raises((KeyError, ValueError)):
        load_config(cfg_path, [bad])


def test_env_interpolation(cfg_path, monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert load_config(cfg_path)["data"]["root"] == "data/g1"
    monkeypatch.setenv("DATA_ROOT", "/scratch/me")
    assert load_config(cfg_path)["data"]["root"] == "/scratch/me/g1"


def test_empty_variable_takes_the_default(cfg_path, tmp_path, monkeypatch):
    """bash semantics: ${VAR:-default} for unset OR empty, so an empty export cannot root paths at /."""
    monkeypatch.setenv("DATA_ROOT", "")
    assert load_config(cfg_path)["data"]["root"] == "data/g1"
    p = tmp_path / "c.yaml"
    p.write_text("a: ${CWM_TEST_EMPTY}\nb: ${CWM_TEST_EMPTY:-4}\nc: x${CWM_TEST_EMPTY}y\n")
    monkeypatch.setenv("CWM_TEST_EMPTY", "")
    assert load_config(p) == {"a": "", "b": 4, "c": "xy"}


def test_env_interpolation_missing_and_typed(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("a: ${CWM_TEST_UNSET}\nb: ${CWM_TEST_N:-4}\nc: x${CWM_TEST_N:-4}\n")
    monkeypatch.delenv("CWM_TEST_UNSET", raising=False)
    monkeypatch.delenv("CWM_TEST_N", raising=False)
    with pytest.raises(KeyError, match="CWM_TEST_UNSET"):
        load_config(p)
    monkeypatch.setenv("CWM_TEST_UNSET", "ok")
    cfg = load_config(p)
    assert cfg == {"a": "ok", "b": 4, "c": "x4"}


def test_overrides_are_interpolated(cfg_path, monkeypatch):
    monkeypatch.setenv("CWM_RUN", "abc")
    assert load_config(cfg_path, ["run.name=${CWM_RUN}"])["run"]["name"] == "abc"


def test_save_roundtrip(cfg_path, tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    cfg = load_config(cfg_path)
    cfg["run"]["tuple"] = (1, 2)
    out = tmp_path / "out" / "config.yaml"
    save_config(cfg, out)
    back = load_config(out)
    assert back["run"]["tuple"] == [1, 2]
    back["run"]["tuple"] = (1, 2)
    assert back == cfg
    assert list(back) == ["run", "optim", "data"]


def test_apply_overrides_indexes_lists():
    cfg = {"a": {"b": [1, {"c": 3}]}}
    assert apply_overrides(cfg, ["a.b.0=5", "a.b.1.c=4"]) == {"a": {"b": [5, {"c": 4}]}}


REQUIRED = object()


def test_defaults_let_overrides_reach_keys_the_yaml_omits(cfg_path, monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    defaults = {
        "run": {"name": REQUIRED, "seed": 42, "tags": []},
        "data": {"dirs": None, "min_chunk_index": None, "max_open_files": 256, "remap": None},
        "dist": {"backend": "nccl", "timeout_minutes": 30},
        "optim": None,
    }
    with pytest.raises(KeyError, match="max_open_files"):
        load_config(cfg_path, ["data.max_open_files=8"])
    cfg = load_config(cfg_path, ["data.max_open_files=8", "dist.timeout_minutes=5"], defaults=defaults)
    assert cfg["data"] == {"dirs": ["data/g1/world_model"], "min_chunk_index": None, "max_open_files": 8,
                           "remap": None, "root": "data/g1"}
    assert cfg["dist"] == {"backend": "nccl", "timeout_minutes": 5}
    assert cfg["run"]["name"] == "test" and cfg["run"]["seed"] == 0 and cfg["run"]["tags"] == []
    assert cfg["optim"]["lr"] == pytest.approx(1e-4)  # a YAML block replaces a non-mapping default
    assert defaults["dist"]["timeout_minutes"] == 30  # the defaults are not modified
    with pytest.raises(KeyError, match="data.typo"):
        load_config(cfg_path, ["data.typo=1"], defaults=defaults)


def test_defaults_keep_sentinels_and_fill_null_blocks(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dist: null\nextra: 1\n")
    cfg = load_config(p, defaults={"stage": REQUIRED, "dist": {"backend": "nccl"}})
    assert cfg["stage"] is REQUIRED  # a trainer can still detect a required key left unset
    assert cfg["dist"] == {"backend": "nccl"} and cfg["extra"] == 1
