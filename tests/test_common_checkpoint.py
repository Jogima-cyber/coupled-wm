# SPDX-License-Identifier: Apache-2.0
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from coupled_wm.common.checkpoint import (
    atomic_save,
    is_pretrained_dir,
    load_checkpoint,
    load_pretrained,
    save_pretrained,
    strip_compile_prefix,
)


def test_strip_compile_prefix():
    sd = {
        "_orig_mod.layers.0.weight": 1,
        "encoder._orig_mod._model.0.weight": 2,
        "module._orig_mod.head.bias": 3,
        "_orig_mod.module.x": 4,
        "block.module.y": 5,
        "plain": 6,
    }
    assert strip_compile_prefix(sd) == {
        "layers.0.weight": 1, "encoder._model.0.weight": 2, "head.bias": 3, "x": 4, "block.module.y": 5, "plain": 6,
    }


def test_strip_compile_prefix_collision():
    with pytest.raises(KeyError, match="w"):
        strip_compile_prefix({"module.w": 1, "w": 2})


def test_strip_compiled_module_loads_into_plain_module():
    plain = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    wrapped = nn.Module()
    wrapped._orig_mod = plain
    sd = strip_compile_prefix(wrapped.state_dict())
    fresh = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4))
    fresh.load_state_dict(sd, strict=True)


def test_atomic_save_roundtrip_and_no_tmp(tmp_path):
    path = tmp_path / "ckpt" / "latest.pt"
    atomic_save({"step": 3, "w": torch.arange(4)}, path)
    obj = load_checkpoint(path)
    assert obj["step"] == 3 and torch.equal(obj["w"], torch.arange(4))
    assert os.listdir(path.parent) == ["latest.pt"]


def test_atomic_save_follows_the_umask(tmp_path):
    old = os.umask(0o027)
    try:
        atomic_save({"a": 1}, tmp_path / "ck.pt")
    finally:
        os.umask(old)
    assert (tmp_path / "ck.pt").stat().st_mode & 0o777 == 0o640


def test_atomic_save_failure_keeps_previous(tmp_path):
    path = tmp_path / "latest.pt"
    atomic_save({"step": 1}, path)

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        atomic_save({"step": 2, "bad": Unpicklable()}, path)
    assert load_checkpoint(path)["step"] == 1
    assert os.listdir(tmp_path) == ["latest.pt"]


def test_killed_save_leaves_one_tmp_that_the_next_save_replaces(tmp_path):
    """SIGKILL mid-write (OOM, KillWait, node loss) skips every handler: the leftover must not pile up."""
    path = tmp_path / "latest.pt"
    atomic_save({"step": 1}, path)
    script = textwrap.dedent(f"""
        import os, signal, torch
        from coupled_wm.common.checkpoint import atomic_save

        def dying_save(obj, f):
            f.write(b"x" * 4096)
            f.flush()
            os.kill(os.getpid(), signal.SIGKILL)

        torch.save = dying_save
        atomic_save({{"step": 2}}, {str(path)!r})
    """)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[1]),
                                                         os.environ.get("PYTHONPATH", "")])}
    for _ in range(2):
        out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, timeout=300)
        assert out.returncode == -signal.SIGKILL, out.stderr[-2000:]
    assert sorted(os.listdir(tmp_path)) == [".latest.pt.tmp", "latest.pt"]
    assert load_checkpoint(path)["step"] == 1
    atomic_save({"step": 3}, path)
    assert os.listdir(tmp_path) == ["latest.pt"] and load_checkpoint(path)["step"] == 3


def test_save_pretrained_roundtrip(tmp_path):
    lin = nn.Linear(4, 4)
    sd = {
        "_orig_mod.w": torch.randn(3, 4, dtype=torch.bfloat16),
        "_orig_mod.steps": torch.tensor([7], dtype=torch.int64),
        "_orig_mod.t": torch.randn(4, 3).t(),  # non-contiguous
        "_orig_mod.tied_a": lin.weight,
        "_orig_mod.tied_b": lin.weight,
    }
    out = save_pretrained(tmp_path / "tok", sd, {"model_type": "dreamerv4_tokenizer", "d_model": 8,
                                                  "image_size": (256, 256)})
    assert is_pretrained_dir(out)
    loaded, cfg = load_pretrained(out)
    assert set(loaded) == {"w", "steps", "t", "tied_a", "tied_b"}
    assert loaded["w"].dtype == torch.float32 and torch.equal(loaded["w"], sd["_orig_mod.w"].float())
    assert loaded["steps"].dtype == torch.int64 and loaded["steps"].item() == 7
    assert torch.equal(loaded["t"], sd["_orig_mod.t"]) and loaded["t"].is_contiguous()
    assert torch.equal(loaded["tied_a"], lin.weight.detach()) and torch.equal(loaded["tied_b"], lin.weight.detach())
    assert cfg == {"model_type": "dreamerv4_tokenizer", "format_version": 1, "d_model": 8, "image_size": [256, 256]}
    raw = json.loads((out / "config.json").read_text())
    assert list(raw)[:2] == ["model_type", "format_version"]
    assert sorted(os.listdir(out)) == ["config.json", "model.safetensors"]


def test_save_pretrained_replaces_leftover_tmp_files(tmp_path):
    out = tmp_path / "m"
    out.mkdir()
    (out / ".model.safetensors.tmp").write_bytes(b"partial")  # left by a killed export
    (out / ".config.json.tmp").write_text("{")
    save_pretrained(out, {"w": torch.zeros(2)}, {"model_type": "x"})
    assert sorted(os.listdir(out)) == ["config.json", "model.safetensors"]
    assert torch.equal(load_pretrained(out)[0]["w"], torch.zeros(2))


def test_save_pretrained_checks(tmp_path):
    with pytest.raises(KeyError, match="model_type"):
        save_pretrained(tmp_path / "a", {"w": torch.zeros(1)}, {"d": 1})
    with pytest.raises(TypeError, match="not a tensor"):
        save_pretrained(tmp_path / "b", {"w": 3}, {"model_type": "x"})
    with pytest.raises(TypeError, match="JSON"):
        save_pretrained(tmp_path / "c", {"w": torch.zeros(1)}, {"model_type": "x", "f": object()})


def test_load_pretrained_rejects_unknown_version(tmp_path):
    out = save_pretrained(tmp_path / "m", {"w": torch.zeros(2)}, {"model_type": "x"})
    cfg = json.loads((out / "config.json").read_text())
    cfg["format_version"] = 2
    (out / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="format_version"):
        load_pretrained(out)
    with pytest.raises(FileNotFoundError):
        load_pretrained(tmp_path / "missing")
