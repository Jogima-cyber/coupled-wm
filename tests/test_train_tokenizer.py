# SPDX-License-Identifier: Apache-2.0
"""Tokenizer trainer: config, schedule, loss scaler, data order, and CPU end-to-end runs
(FSDP on gloo, tiny model, synthetic episodes, LPIPS with random weights)."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import types
import urllib.error
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data.distributed import DistributedSampler

from coupled_wm.common.checkpoint import load_checkpoint, load_pretrained
from coupled_wm.common.config import load_config
from coupled_wm.data.h5_dataset import list_h5_files
from coupled_wm.world_model import export as E
from coupled_wm.world_model import train_tokenizer as TT
from coupled_wm.world_model.loading import load_tokenizer, read_tokenizer_weights
from coupled_wm.world_model.models import G1_TOKENIZER_ARCH

REPO = Path(__file__).resolve().parents[1]
G1_CONFIG = REPO / "configs" / "world_model" / "g1_tokenizer.yaml"

TINY_ARCH = dict(image_size=[16, 16], patch_size=8, d_model=32, n_layers=4, num_heads_q=2, num_heads_kv_latent=2,
                 temporal_every=2, n_latents=4, bottleneck_dim=4, use_proprio=True, proprio_dim=3, num_proprio=1,
                 seq_len=8, context_len=0, dropout=0.0, mae_max_mask_prob=0.9)

# Random-weight LPIPS stand-in (no download); `pretrained=True` only loads the packaged linear heads.
_LPIPS_CODE = ("lambda net, trunk_weights, verbose: __import__('lpips').LPIPS(net=net, pnet_rand=True, "
               "verbose=False)")
# Rank 0 cannot download the trunk weights (a compute node without internet access).
_OFFLINE_LPIPS_CODE = ("lambda net, trunk_weights, verbose: (_ for _ in ()).throw(__import__('urllib.error').error"
                       ".URLError('[Errno 101] Network is unreachable')) if trunk_weights else "
                       "__import__('lpips').LPIPS(net=net, pnet_rand=True, verbose=False)")


# --------------------------------------------------------------------------------------
# Config and schedule
# --------------------------------------------------------------------------------------

def test_g1_config_is_the_paper_recipe():
    cfg = TT.resolve_config(load_config(G1_CONFIG))
    assert cfg["model"] == G1_TOKENIZER_ARCH
    s, o, t = cfg["schedule"], cfg["optim"], cfg["train"]
    assert (s["warmup_updates"], s["total_updates"], s["max_updates"], s["min_lr"]) == (1400, 240 * 5756 // 4, 45000,
                                                                                         1e-8)
    assert (o["lr"], o["betas"], o["eps"], o["weight_decay"], o["grad_clip"]) == (1e-4, [0.9, 0.999], 1e-8, 0.01, 1.0)
    assert (t["global_batch"], t["micro_batch_per_gpu"]) == (256, 1)
    assert TT.grad_accum_steps(256, 1, 64) == 4
    assert cfg["loss"]["lpips_weight"] == 0.2 and cfg["loss"]["proprio_weight"] == 0.01
    assert (cfg["loss"]["rms_decay"], cfg["loss"]["rms_reference_windows"]) == (0.99, 64)
    assert cfg["data"]["window"] == 64 and cfg["data"]["color_order"] == "bgr"
    # The paper checkpoint (update 45000) stores the learning rate 9.6088e-05.
    assert TT.lr_at(45000, o["lr"], s["warmup_updates"], s["total_updates"], s["min_lr"]) == pytest.approx(9.6088e-5,
                                                                                                         abs=5e-10)


def test_g1_config_selects_the_paper_files(tmp_path):
    """The include globs keep chunk_0000..chunk_0146 of the 290-file G1 directory."""
    names = [f"chunk_{i:04d}.h5" for i in range(228)]
    names += [f"patch_{k}_chunk_{i:04d}.h5" for k in range(2, 8) for i in range(12)] + ["patch_chunk_0002.h5"]
    for n in names:
        (tmp_path / n).touch()
    data = TT.resolve_config(load_config(G1_CONFIG))["data"]
    files = list_h5_files([tmp_path], include_globs=data["include_globs"], exclude_globs=data["exclude_globs"])
    assert [os.path.basename(f) for f in files] == [f"chunk_{i:04d}.h5" for i in range(147)]


def test_config_validation():
    base = {"run_dir": "r", "data": {"dirs": ["d"]}}
    TT.resolve_config(base)
    with pytest.raises(KeyError, match="optim.weight_decy"):
        TT.resolve_config({**base, "optim": {"weight_decy": 0.1}})
    with pytest.raises(KeyError, match="unknown tokenizer architecture"):
        TT.resolve_config({**base, "model": {"mlp_ratio": 4.0}})
    with pytest.raises(ValueError, match="data.window"):
        TT.resolve_config({**base, "data": {"dirs": ["d"], "window": 65}})
    with pytest.raises(ValueError, match="run_dir"):
        TT.resolve_config({"data": {"dirs": ["d"]}})
    with pytest.raises(ValueError, match="not both"):
        TT.resolve_config({**base, "resume": "x.pt", "init_from": "y"})
    with pytest.raises(KeyError, match="data.image_size"):  # removed: frames are always resized to the model's size
        TT.resolve_config({**base, "data": {"dirs": ["d"], "image_size": 128}})
    with pytest.raises(ValueError, match="save_every_minutes"):
        TT.resolve_config({**base, "checkpoint": {"save_every_minutes": 0}})
    with pytest.raises(ValueError, match="allow_pool_change"):
        TT.resolve_config({**base, "data": {"dirs": ["d"], "allow_pool_change": "yes"}})
    with pytest.raises(ValueError, match="rms_decay"):
        TT.resolve_config({**base, "loss": {"rms_decay": 1.5}})
    for bad in (0, 2.0, True):
        with pytest.raises(ValueError, match="rms_reference_windows"):
            TT.resolve_config({**base, "loss": {"rms_reference_windows": bad}})
    for bad in (0, 1.5, True):
        with pytest.raises(ValueError, match="max_nonfinite_updates"):
            TT.resolve_config({**base, "train": {"max_nonfinite_updates": bad}})


def test_grad_accum_steps():
    assert TT.grad_accum_steps(256, 1, 8) == 32
    assert TT.grad_accum_steps(256, 2, 4) == 32
    with pytest.raises(ValueError, match="not a multiple"):
        TT.grad_accum_steps(256, 1, 3)


def test_lr_schedule_shape():
    lr = lambda u: TT.lr_at(u, 1e-4, 10, 110, 1e-8)  # noqa: E731
    assert lr(0) == 0.0 and lr(5) == pytest.approx(5e-5) and lr(10) == pytest.approx(1e-4)
    decay = [lr(u) for u in range(10, 111)]
    assert all(a >= b for a, b in zip(decay, decay[1:]))
    assert lr(60) == pytest.approx(5e-5)
    assert lr(110) == lr(500) == pytest.approx(1e-8)  # floored, and constant after total_updates


# --------------------------------------------------------------------------------------
# Loss scaler and data order
# --------------------------------------------------------------------------------------

def test_rms_scaler_state_is_fp32_and_tracks_the_loss():
    scaler = TT.RMSLossScaler(decay=0.99, eps=1e-8)
    scaler("lpips", torch.tensor(0.1904, dtype=torch.bfloat16))
    for _ in range(2000):
        out = scaler("lpips", torch.tensor(0.13, dtype=torch.bfloat16))
    assert scaler.ema_sq["lpips"].dtype == torch.float32
    # A bf16 EMA stays at ~0.19 (the update falls below bf16 resolution); fp32 converges.
    assert scaler.rms()["lpips"] == pytest.approx(0.13, rel=2e-3)
    assert out.dtype == torch.bfloat16
    v = torch.tensor(0.13, dtype=torch.bfloat16)
    expect = v / (scaler.ema_sq["lpips"] * 0.99 + 0.01 * v.float() ** 2 + 1e-8).sqrt().to(torch.bfloat16)
    assert torch.equal(scaler("lpips", v), expect)

    other = TT.RMSLossScaler()
    other.load_state_dict(scaler.state_dict())
    assert {k: v.item() for k, v in other.ema_sq.items()} == {k: v.item() for k, v in scaler.ema_sq.items()}


def _normaliser_trace(decay: float, accum: int, updates: int = 40) -> list[float]:
    """Running RMS after each update when the loss jumps from 1 to 3 (one call per micro-step)."""
    scaler = TT.RMSLossScaler(decay)
    scaler("mse", torch.tensor(1.0))
    trace = []
    for _ in range(updates):
        for _ in range(accum):
            scaler("mse", torch.tensor(3.0))
        trace.append(scaler.rms()["mse"])
    return trace


# (GPUs, windows per GPU per micro-step); the first is the paper's.
_WORLDS = [(64, 1), (1, 1), (4, 1), (8, 1), (8, 4), (16, 2), (64, 4), (128, 1), (256, 1)]


def test_loss_normaliser_horizon_is_defined_in_windows():
    """The paper's 0.99 per micro-step on 64 GPUs x 1 window; the same decay per window, hence
    per update and the same trace of the running RMS, on any number of GPUs."""
    loss = TT.resolve_config(load_config(G1_CONFIG))["loss"]
    rms_decay, reference = loss["rms_decay"], loss["rms_reference_windows"]
    assert TT.rms_decay_per_micro_step(rms_decay, 64 * 1, reference) == 0.99  # bit-identical to the paper
    paper = _normaliser_trace(0.99, accum=4)
    for world, micro_batch in _WORLDS:
        windows = world * micro_batch
        accum = TT.grad_accum_steps(256, micro_batch, world)
        decay = TT.rms_decay_per_micro_step(rms_decay, windows, reference)
        assert decay ** (1 / windows) == pytest.approx(0.99 ** (1 / 64), rel=1e-14)  # per window
        assert decay ** accum == pytest.approx(0.99 ** 4, rel=1e-12)  # per update of 256 windows
        # fp32 state: rounding over up to 256 micro-steps per update
        assert _normaliser_trace(decay, accum) == pytest.approx(paper, rel=1e-3), (world, micro_batch)
    assert _normaliser_trace(0.99, accum=256) != pytest.approx(paper, rel=1e-1)  # 0.99 per micro-step on 1 GPU


@pytest.mark.parametrize("world", [1, 2, 4])
def test_sampler_is_a_prefix_of_distributed_sampler(world):
    n, global_batch, seed, epoch = 203, 8, 42, 3
    for rank in range(world):
        ours = TT.EpochWindowSampler(n, global_batch, rank, world, seed)
        ours.set_epoch(epoch)
        ref = DistributedSampler(range(n), num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True)
        ref.set_epoch(epoch)
        got = list(ours)
        assert len(got) == len(ours) == (n // global_batch) * global_batch // world
        assert got == list(ref)[:len(got)]


def test_sampler_update_blocks_and_resume():
    n, global_batch, seed = 203, 8, 7
    order = TT.EpochWindowSampler(n, global_batch, 0, 1, seed).order()
    for world in (1, 2, 4):
        per_rank = []
        for rank in range(world):
            s = TT.EpochWindowSampler(n, global_batch, rank, world, seed)
            s.set_epoch(0, start_update=5)
            per_rank.append(list(s))
        per_update = global_batch // world
        for k in range(n // global_batch - 5):
            block = sorted(i for r in per_rank for i in r[k * per_update:(k + 1) * per_update])
            assert block == sorted(order[(5 + k) * global_batch:(6 + k) * global_batch])


# --------------------------------------------------------------------------------------
# End-to-end runs (CPU, gloo)
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lpips_template():
    import lpips

    torch.manual_seed(123)
    return lpips.LPIPS(net="vgg", pnet_rand=True, verbose=False)


@pytest.fixture
def patched_lpips(monkeypatch, lpips_template):
    monkeypatch.setattr(TT, "_make_lpips", lambda net, trunk_weights, verbose: copy.deepcopy(lpips_template))


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    from conftest import write_h5

    d = tmp_path_factory.mktemp("tok_data")
    for i in range(3):
        write_h5(d / f"chunk_{i:04d}.h5", 40, dones_at=[19, 39], image_hw=(16, 16), proprio_dim=3, file_id=i)
    return d


def tiny_config(run_dir, data_dir, top=None, **blocks):
    cfg = {
        "run_dir": str(run_dir),
        "seed": 3,
        "compile": False,
        "data": {"dirs": [str(data_dir)], "window": 4, "num_workers": 0},
        "model": dict(TINY_ARCH),
        "schedule": {"warmup_updates": 2, "total_updates": 40, "max_updates": 6},
        "train": {"global_batch": 4, "micro_batch_per_gpu": 2},
        "checkpoint": {"save_every_updates": 2, "archive_every_updates": 2, "keep_archives": 2},
        "logging": {"print_every_updates": 0},
    }
    cfg.update(top or {})
    for block, values in blocks.items():
        cfg.setdefault(block, {}).update(values)
    return cfg


def run_cfg(path, cfg, *overrides):
    path = Path(path)
    path.write_text(yaml.safe_dump(cfg))
    return TT.main(["--config", str(path), *(["--set", *overrides] if overrides else [])])


def assert_same_training_state(a, b):
    for key in ("enc", "dec"):
        assert a[key].keys() == b[key].keys()
        for k in a[key]:
            assert torch.equal(a[key][k], b[key][k]), f"{key}.{k}"
    sa, sb = a["optim"]["state"], b["optim"]["state"]
    assert sa.keys() == sb.keys()
    for p in sa:
        for k in sa[p]:
            assert torch.equal(sa[p][k], sb[p][k]), f"optim {p}.{k}"
    for key in ("global_update", "epoch", "epoch_update", "rms_scaler", "skip"):
        assert a[key] == b[key], key


@pytest.fixture(scope="module")
def straight_run(tmp_path_factory, data_dir, lpips_template):
    """6 updates without interruption (an epoch of the tiny data has 102 // 4 = 25 updates)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(TT, "_make_lpips", lambda net, trunk_weights, verbose: copy.deepcopy(lpips_template))
    root = tmp_path_factory.mktemp("straight")
    out = run_cfg(root / "cfg.yaml", tiny_config(root / "run", data_dir))
    mp.undo()
    assert out == {"global_update": 6, "status": "done"}
    return root


def test_outputs_of_a_run(straight_run):
    run = straight_run / "run"
    ckpt = load_checkpoint(run / "latest.pt")
    assert ckpt["global_update"] == 6 and (ckpt["epoch"], ckpt["epoch_update"]) == (0, 6)
    assert ckpt["optim"]["state"]["encoder.latent_tokens"]["step"].item() == 6
    assert not any(k.startswith("_orig_mod") or "_fsdp" in k for k in ckpt["enc"])
    assert set(ckpt["rms_scaler"]) == {"prop", "mse", "lpips"}
    assert sorted(p.name for p in (run / "archive").iterdir()) == ["step_00000004", "step_00000006"]
    assert (run / "config.yaml").is_file() and any((run / "tb").glob("events.out.tfevents.*"))

    enc, dec, cfg = load_tokenizer(run / "pretrained")
    assert cfg["architecture"] == TT.tokenizer_arch(TINY_ARCH)
    assert cfg["provenance"]["global_update"] == 6 and cfg["provenance"]["data"]["windows"] == 102
    assert cfg["preprocessing"]["h5_color_order"] == "bgr" and cfg["preprocessing"]["proprio_dim"] == 3
    for k, v in enc.state_dict().items():
        assert torch.equal(v, ckpt["enc"][k].float()), k
    for k, v in dec.state_dict().items():
        assert torch.equal(v, ckpt["dec"][k].float()), k

    E.main(["--ckpt", str(run / "latest.pt"), "--out", str(straight_run / "cli_export")])
    a, _ = load_pretrained(run / "pretrained")
    b, cli_cfg = load_pretrained(straight_run / "cli_export")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert cli_cfg["provenance"]["global_update"] == 6


def test_trainer_normaliser_decay_follows_the_windows_per_micro_step(tmp_path, data_dir, patched_lpips, monkeypatch):
    decays = []

    class Recording(TT.RMSLossScaler):
        def __init__(self, decay, eps):
            decays.append(decay)
            super().__init__(decay, eps)

    monkeypatch.setattr(TT, "RMSLossScaler", Recording)
    cfg = tiny_config(tmp_path / "run", data_dir, schedule={"max_updates": 1})  # 1 rank x 2 windows
    run_cfg(tmp_path / "a.yaml", cfg)
    run_cfg(tmp_path / "b.yaml", {**cfg, "run_dir": str(tmp_path / "run_b")}, "loss.rms_reference_windows=2")
    assert decays == [0.99 ** (2 / 64), 0.99]


def test_resume_is_exact(straight_run, tmp_path, data_dir, patched_lpips):
    cfg = tiny_config(tmp_path / "run", data_dir, schedule={"max_updates": 3})
    assert run_cfg(tmp_path / "a.yaml", cfg)["global_update"] == 3
    out = run_cfg(tmp_path / "b.yaml", cfg, "schedule.max_updates=6")
    assert out == {"global_update": 6, "status": "done"}
    assert_same_training_state(load_checkpoint(tmp_path / "run" / "latest.pt"),
                               load_checkpoint(straight_run / "run" / "latest.pt"))


def test_stop_signal_saves_the_last_update_and_resume_is_exact(straight_run, tmp_path, data_dir, patched_lpips,
                                                               monkeypatch):
    calls = {"n": 0}
    original = TT.TokenizerTrainer._micro_step

    def micro_step(self, batch):
        calls["n"] += 1
        if calls["n"] == 7:  # first of the two micro-steps of update 3: the signal lands mid-update
            os.kill(os.getpid(), signal.SIGUSR1)
        return original(self, batch)

    monkeypatch.setattr(TT.TokenizerTrainer, "_micro_step", micro_step)
    cfg = tiny_config(tmp_path / "run", data_dir)
    assert run_cfg(tmp_path / "a.yaml", cfg) == {"global_update": 3, "status": "signal"}
    assert load_checkpoint(tmp_path / "run" / "latest.pt")["global_update"] == 3
    monkeypatch.setattr(TT.TokenizerTrainer, "_micro_step", original)
    assert run_cfg(tmp_path / "b.yaml", cfg)["global_update"] == 6
    assert_same_training_state(load_checkpoint(tmp_path / "run" / "latest.pt"),
                               load_checkpoint(straight_run / "run" / "latest.pt"))


def test_finished_run_only_exports(straight_run, tmp_path, data_dir, patched_lpips):
    run = straight_run / "run"
    stamps = [(run / f).stat().st_mtime_ns for f in ("latest.pt", "pretrained/model.safetensors")]
    cfg = tiny_config(run, data_dir)
    assert run_cfg(tmp_path / "c.yaml", cfg) == {"global_update": 6, "status": "done"}
    # pretrained/ already holds update 6: nothing is rewritten.
    assert [(run / f).stat().st_mtime_ns for f in ("latest.pt", "pretrained/model.safetensors")] == stamps


def test_relaunch_replaces_a_stale_export(tmp_path, data_dir, patched_lpips, monkeypatch):
    """An extended run killed between writing latest.pt and its export must not keep the old export."""
    run = tmp_path / "run"
    cfg = tiny_config(run, data_dir, schedule={"max_updates": 3}, checkpoint={"archive_every_updates": 0})
    assert run_cfg(tmp_path / "a.yaml", cfg)["global_update"] == 3
    assert json.loads((run / "pretrained" / "config.json").read_text())["provenance"]["global_update"] == 3

    def killed(*args, **kwargs):
        raise OSError("killed during the export")

    export = TT.save_tokenizer_pretrained
    monkeypatch.setattr(TT, "save_tokenizer_pretrained", killed)
    with pytest.raises(OSError, match="killed during the export"):
        run_cfg(tmp_path / "b.yaml", cfg, "schedule.max_updates=5")
    assert load_checkpoint(run / "latest.pt")["global_update"] == 5
    monkeypatch.setattr(TT, "save_tokenizer_pretrained", export)

    assert run_cfg(tmp_path / "c.yaml", cfg, "schedule.max_updates=5") == {"global_update": 5, "status": "done"}
    ckpt = load_checkpoint(run / "latest.pt")
    enc, dec, exported = read_tokenizer_weights(run / "pretrained")
    assert json.loads((run / "pretrained" / "config.json").read_text())["provenance"]["global_update"] == 5
    assert all(torch.equal(enc[k], ckpt["enc"][k].float()) for k in enc)
    assert all(torch.equal(dec[k], ckpt["dec"][k].float()) for k in dec)


def test_explicit_resume_path(straight_run, tmp_path, data_dir, patched_lpips):
    cfg = tiny_config(tmp_path / "run", data_dir, top={"resume": str(straight_run / "run" / "latest.pt")})
    assert run_cfg(tmp_path / "a.yaml", cfg, "schedule.max_updates=7")["global_update"] == 7
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_cfg(tmp_path / "b.yaml", tiny_config(tmp_path / "other", data_dir, top={"resume": str(tmp_path / "x.pt")}))


def test_resume_refuses_another_architecture(straight_run, tmp_path, data_dir, patched_lpips):
    cfg = tiny_config(tmp_path / "run", data_dir, top={"resume": str(straight_run / "run" / "latest.pt")},
                      model={"temporal_every": 4})
    with pytest.raises(ValueError, match="temporal_every"):
        run_cfg(tmp_path / "a.yaml", cfg, "schedule.max_updates=7")


def test_init_from_pretrained_dir(straight_run, tmp_path, data_dir, patched_lpips):
    pretrained = straight_run / "run" / "pretrained"
    cfg = tiny_config(tmp_path / "run", data_dir, top={"init_from": str(pretrained)}, schedule={"max_updates": 1})
    assert run_cfg(tmp_path / "a.yaml", cfg)["global_update"] == 1
    # Update 0 runs at learning rate 0 (warmup), so the weights are still the initial ones.
    ckpt = load_checkpoint(tmp_path / "run" / "latest.pt")
    enc, dec, _ = read_tokenizer_weights(pretrained)
    assert all(torch.equal(ckpt["enc"][k], enc[k]) for k in enc)
    assert all(torch.equal(ckpt["dec"][k], dec[k]) for k in dec)
    assert ckpt["model_config"]["provenance"]["init_from"] == "pretrained"


def test_resume_keeps_the_init_source(straight_run, tmp_path, data_dir, patched_lpips, capsys):
    run = tmp_path / "run"
    cfg = tiny_config(run, data_dir, top={"init_from": str(straight_run / "run" / "pretrained")},
                      schedule={"max_updates": 1})
    assert run_cfg(tmp_path / "a.yaml", cfg)["global_update"] == 1
    # A relaunch without init_from still records where the run started.
    assert run_cfg(tmp_path / "b.yaml", tiny_config(run, data_dir, schedule={"max_updates": 2}))["global_update"] == 2
    assert json.loads((run / "pretrained" / "config.json").read_text())["provenance"]["init_from"] == "pretrained"
    # Another init_from on a relaunch is not used, and says so.
    capsys.readouterr()
    other = tiny_config(run, data_dir, top={"init_from": str(straight_run / "run" / "latest.pt")},
                        schedule={"max_updates": 3})
    assert run_cfg(tmp_path / "c.yaml", other)["global_update"] == 3
    assert f"init_from={straight_run / 'run' / 'latest.pt'} is not used" in capsys.readouterr().out
    assert load_checkpoint(run / "latest.pt")["model_config"]["provenance"]["init_from"] == "pretrained"


def test_resume_refuses_other_data(straight_run, tmp_path, data_dir, patched_lpips):
    from conftest import write_h5

    def resume_in(name, data, *overrides):
        run = tmp_path / name
        run.mkdir()
        shutil.copy(straight_run / "run" / "latest.pt", run / "latest.pt")
        return run_cfg(tmp_path / f"{name}.yaml", tiny_config(run, data, schedule={"max_updates": 7}), *overrides)

    # Same file names, frame counts and segments, other recordings (only the proprio differs here).
    other = tmp_path / "other"
    for i in range(3):
        write_h5(other / f"chunk_{i:04d}.h5", 40, dones_at=[19, 39], image_hw=(16, 16), proprio_dim=3, file_id=i + 10)
    with pytest.raises(ValueError, match=r"\(pool_sha256 changed\).*data.allow_pool_change=true"):
        resume_in("other_pool", other)
    assert resume_in("allowed", other, "+data.allow_pool_change=true")["global_update"] == 7
    with pytest.raises(ValueError, match=r"\(window changed\)"):
        resume_in("other_window", data_dir, "data.window=5")
    # The same files in another directory are the same data.
    moved = tmp_path / "moved_data"
    shutil.copytree(data_dir, moved)
    assert resume_in("moved", moved)["global_update"] == 7
    signature = load_checkpoint(tmp_path / "moved" / "latest.pt")["data_signature"]
    assert signature == load_checkpoint(straight_run / "run" / "latest.pt")["data_signature"]
    assert json.loads((tmp_path / "moved" / "pretrained" / "config.json").read_text())["provenance"]["data"][
        "pool_sha256"] == signature["pool_sha256"]


def test_init_from_a_tokenizer_without_proprio(straight_run, tmp_path, data_dir, patched_lpips):
    """A partial start (the paper's SOAR initialisation) needs an explicit allow-list."""
    no_prop = tiny_config(tmp_path / "noprop", data_dir, model={"use_proprio": False}, schedule={"max_updates": 2})
    assert run_cfg(tmp_path / "a.yaml", no_prop)["global_update"] == 2
    source = tmp_path / "noprop" / "latest.pt"

    cfg = tiny_config(tmp_path / "run", data_dir, top={"init_from": str(source)}, schedule={"max_updates": 1})
    with pytest.raises(RuntimeError, match="missing"):
        run_cfg(tmp_path / "b.yaml", cfg)
    cfg["init_allow_missing"] = ["encoder.proprio_proj.*", "encoder.mask_proprio_token",
                                        "decoder.readout_proprio_token", "decoder.proprio_head.*",
                                        "decoder.layers.*.proprio_spatial_mask"]
    assert run_cfg(tmp_path / "c.yaml", cfg)["global_update"] == 1

    with pytest.raises(ValueError, match="temporal_every"):
        bad = tiny_config(tmp_path / "bad", data_dir, top={"init_from": str(source)}, model={"temporal_every": 4})
        run_cfg(tmp_path / "d.yaml", bad)


def test_frames_of_another_size_are_resized_to_the_model(tmp_path, data_dir, patched_lpips):
    # data_dir holds 16x16 frames
    native = tiny_config(tmp_path / "a", data_dir, schedule={"max_updates": 1})
    assert run_cfg(tmp_path / "a.yaml", native)["global_update"] == 1
    assert "h5_image_resize" not in load_checkpoint(tmp_path / "a" / "latest.pt")["model_config"]["preprocessing"]
    resized = tiny_config(tmp_path / "b", data_dir, model={"image_size": [32, 32]}, schedule={"max_updates": 1})
    assert run_cfg(tmp_path / "b.yaml", resized)["global_update"] == 1
    assert "bilinear" in load_checkpoint(tmp_path / "b" / "latest.pt")["model_config"]["preprocessing"]["h5_image_resize"]


def test_lpips_off_and_skip_detector(tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(TT, "_make_lpips", lambda *a, **k: pytest.fail("LPIPS must not be built"))
    cfg = tiny_config(tmp_path / "run", data_dir, loss={"lpips_weight": 0.0},
                      train={"skip_threshold": 1e-9, "skip_warmup_updates": 2}, schedule={"max_updates": 4})
    assert run_cfg(tmp_path / "a.yaml", cfg)["global_update"] == 4
    ckpt = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert ckpt["skip"]["skip_count"] == 2 and set(ckpt["rms_scaler"]) == {"prop", "mse"}
    # Updates 2 and 3 were skipped: the optimizer stepped twice.
    assert ckpt["optim"]["state"]["encoder.latent_tokens"]["step"].item() == 2


def _poison_encoder_grads(monkeypatch, updates):
    """Make one encoder gradient element NaN at each update of ``updates`` (counted from 0)."""
    original = TT.TokenizerTrainer._finish_update

    def finish(self):
        if self.global_update in updates:
            next(p for p in self.enc_fsdp.parameters() if p.grad is not None).grad.view(-1)[0] = float("nan")
        return original(self)

    monkeypatch.setattr(TT.TokenizerTrainer, "_finish_update", finish)


def _all_finite(ckpt):
    return all(bool(torch.isfinite(t).all()) for sd in (ckpt["enc"], ckpt["dec"]) for t in sd.values()
               if t.is_floating_point())


def test_nonfinite_updates_are_not_applied(tmp_path, data_dir, patched_lpips, monkeypatch, capsys):
    # One NaN gradient element: clipping would make every encoder gradient NaN, and the step every weight.
    _poison_encoder_grads(monkeypatch, {1, 3})
    assert run_cfg(tmp_path / "a.yaml", tiny_config(tmp_path / "run", data_dir))["global_update"] == 6
    ckpt = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert _all_finite(ckpt) and math.isfinite(ckpt["skip"]["ema_raw_loss"])
    assert ckpt["skip"]["nonfinite_count"] == 2 and ckpt["skip"]["skip_count"] == 0
    assert ckpt["optim"]["state"]["encoder.latent_tokens"]["step"].item() == 4
    assert capsys.readouterr().out.count("non-finite raw loss or grad norm") == 2
    # The save before the last one lives on as latest.prev.pt.
    assert load_checkpoint(tmp_path / "run" / "latest.prev.pt")["global_update"] == 4


def test_a_nonfinite_first_update_is_logged(tmp_path, data_dir, patched_lpips, monkeypatch):
    # The running mean of the raw loss starts at the first finite update: the logging of the first update must
    # not need it (rank 0 raised there while the other ranks waited in the next collective).
    _poison_encoder_grads(monkeypatch, {0})
    assert run_cfg(tmp_path / "a.yaml", tiny_config(tmp_path / "run", data_dir,
                                                    schedule={"max_updates": 3}))["global_update"] == 3
    ckpt = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert ckpt["skip"]["nonfinite_count"] == 1 and math.isfinite(ckpt["skip"]["ema_raw_loss"])


def test_nonfinite_updates_in_a_row_stop_the_run(tmp_path, data_dir, patched_lpips, monkeypatch):
    _poison_encoder_grads(monkeypatch, {3, 4, 5})
    cfg = tiny_config(tmp_path / "run", data_dir, train={"max_nonfinite_updates": 3})
    with pytest.raises(RuntimeError, match="3 non-finite updates in a row"):
        run_cfg(tmp_path / "a.yaml", cfg)
    ckpt = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert ckpt["global_update"] == 4 and _all_finite(ckpt)  # the save after the first skipped update


def test_save_refuses_nonfinite_weights(tmp_path, data_dir, patched_lpips, monkeypatch):
    original = TT.TokenizerTrainer._finish_update

    def finish(self):
        stats = original(self)
        if self.global_update == 3:
            with torch.no_grad():
                next(self.enc_fsdp.parameters()).view(-1)[0] = float("nan")
        return stats

    monkeypatch.setattr(TT.TokenizerTrainer, "_finish_update", finish)
    with pytest.raises(RuntimeError, match="refusing to save update 4: non-finite weights .*latest.pt keeps update 2"):
        run_cfg(tmp_path / "a.yaml", tiny_config(tmp_path / "run", data_dir))
    ckpt = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert ckpt["global_update"] == 2 and _all_finite(ckpt)


def test_timed_saves_console_lines_and_stale_temp_files(tmp_path, data_dir, patched_lpips, monkeypatch, capsys):
    run = tmp_path / "run"
    stale = [run / ".latest.pt.tmp", run / "pretrained" / ".model.safetensors.tmp99",
             run / "archive" / "step_00000002" / ".config.json.tmp99"]
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 64)
    saved = []
    original = TT.TokenizerTrainer.save

    def save(self, archive=False, export=False):
        saved.append(self.global_update)
        return original(self, archive=archive, export=export)

    monkeypatch.setattr(TT.TokenizerTrainer, "save", save)
    rare = {"save_every_updates": 1000, "archive_every_updates": 0, "at_epoch_end": False}
    cfg = tiny_config(run, data_dir, schedule={"max_updates": 7}, checkpoint={**rare, "save_every_minutes": 1e-9},
                      logging={"print_every_updates": 1000})
    assert run_cfg(tmp_path / "a.yaml", cfg)["global_update"] == 7
    assert saved == [1, 2, 3, 4, 5, 6, 7]  # every update is past the (tiny) save interval
    assert not any(path.exists() for path in stale)
    out = capsys.readouterr().out
    assert out.count("left by an interrupted save") == 3
    assert re.findall(r"update (\d+)/7 \|", out) == ["1", "2", "3", "4", "5"]  # the first 5 of the launch

    saved.clear()
    cfg = tiny_config(tmp_path / "untimed", data_dir, schedule={"max_updates": 3},
                      checkpoint={**rare, "save_every_minutes": None})
    assert run_cfg(tmp_path / "b.yaml", cfg)["global_update"] == 3
    assert saved == [3]


def test_wandb_run_is_named_after_the_run_dir(tmp_path, data_dir, patched_lpips, monkeypatch):
    calls = {}

    class Run:
        id = "abc123"

        def finish(self):
            pass

    def init(**kwargs):
        calls.update(kwargs)
        return Run()

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=init))
    cfg = tiny_config(tmp_path / "my_tokenizer", data_dir, wandb={"enabled": True}, schedule={"max_updates": 1})
    assert run_cfg(tmp_path / "a.yaml", cfg)["global_update"] == 1
    assert calls["name"] == "my_tokenizer" and calls["id"] is None
    assert load_checkpoint(tmp_path / "my_tokenizer" / "latest.pt")["wandb_run_id"] == "abc123"


def _offline_lpips(template):
    def make(net, trunk_weights, verbose):
        if trunk_weights:  # what torch.hub raises on a node without internet access
            raise urllib.error.URLError(OSError(101, "Network is unreachable"))
        return copy.deepcopy(template)
    return make


def test_lpips_download_failure_names_the_offline_fix(tmp_path, data_dir, lpips_template, monkeypatch):
    monkeypatch.setattr(TT, "_make_lpips", _offline_lpips(lpips_template))
    with pytest.raises(RuntimeError, match="--download-lpips") as info:
        run_cfg(tmp_path / "a.yaml", tiny_config(tmp_path / "run", data_dir))
    assert isinstance(info.value.__cause__, urllib.error.URLError)
    assert "Network is unreachable" in str(info.value) and "TORCH_HOME" in str(info.value)


def test_model_without_proprio(tmp_path, patched_lpips):
    from conftest import write_h5

    d = tmp_path / "data"
    write_h5(d / "chunk_0000.h5", 30, image_hw=(16, 16), proprio_dim=None)
    cfg = tiny_config(tmp_path / "run", d, model={"use_proprio": False}, schedule={"max_updates": 2})
    assert run_cfg(tmp_path / "a.yaml", cfg)["global_update"] == 2
    ckpt = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert set(ckpt["rms_scaler"]) == {"mse", "lpips"}
    assert ckpt["model_config"]["preprocessing"]["proprio_dim"] == 0


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank_env(rank, world, port):
    """torchrun's environment variables for one process (CPU only)."""
    return dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world), LOCAL_WORLD_SIZE=str(world),
                MASTER_ADDR="127.0.0.1", MASTER_PORT=port, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                PYTHONUNBUFFERED="1")


def _launch(cfg_path, world, *overrides, lpips_code=_LPIPS_CODE, cwds=None, check=True):
    """``world`` processes with torchrun's environment variables; returns ``[(returncode, output)]``."""
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from coupled_wm.world_model import train_tokenizer as T\n"
            "T._make_lpips = %s\n"
            "T.main(sys.argv[1:])\n") % (str(REPO), lpips_code)
    port = str(_free_port())
    procs = []
    for rank in range(world):
        args = [sys.executable, "-c", code, "--config", str(cfg_path), *(["--set", *overrides] if overrides else [])]
        procs.append(subprocess.Popen(args, env=_rank_env(rank, world, port), cwd=cwds[rank] if cwds else None,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outs = [p.communicate(timeout=600)[0] for p in procs]
    if check:
        for p, out in zip(procs, outs):
            assert p.returncode == 0, out[-4000:]
    return [(p.returncode, out) for p, out in zip(procs, outs)]


def test_exit_status_tells_a_stop_from_a_finished_run(tmp_path, data_dir):
    """``python -m``: a stop signal saves and exits EXIT_STOPPED, a finished run exits 0."""
    cfg = tiny_config(tmp_path / "run", data_dir, loss={"lpips_weight": 0.0}, schedule={"max_updates": 1000},
                      logging={"print_every_updates": 1})
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))

    def launch(*overrides):
        args = [sys.executable, "-m", "coupled_wm.world_model.train_tokenizer", "--config", str(path),
                *(["--set", *overrides] if overrides else [])]
        return subprocess.Popen(args, cwd=REPO, env=_rank_env(0, 1, str(_free_port())), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)

    proc, lines = launch(), []
    for line in proc.stdout:
        lines.append(line)
        if "update 2/1000 |" in line:
            proc.send_signal(signal.SIGUSR1)
    assert proc.wait(timeout=600) == TT.EXIT_STOPPED, "".join(lines[-40:])
    stopped = load_checkpoint(tmp_path / "run" / "latest.pt")["global_update"]
    assert 2 <= stopped < 1000

    proc = launch(f"schedule.max_updates={stopped + 1}")
    out = proc.communicate(timeout=600)[0]
    assert proc.returncode == 0, out[-4000:]
    assert load_checkpoint(tmp_path / "run" / "latest.pt")["global_update"] == stopped + 1


def test_lpips_failure_stops_every_rank(tmp_path, data_dir):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(tiny_config(tmp_path / "run", data_dir, dist={"timeout_minutes": 1})))
    for rank, (code, out) in enumerate(_launch(path, 2, lpips_code=_OFFLINE_LPIPS_CODE, check=False)):
        assert code != 0 and "--download-lpips" in out, out[-4000:]
        assert ("Network is unreachable" in out) == (rank == 0)


def test_ranks_must_see_the_same_checkpoint(straight_run, tmp_path, data_dir):
    """A relative run_dir from two working directories: one rank sees latest.pt, the other does not."""
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "runs" / "tok").mkdir(parents=True)
    b.mkdir()
    shutil.copy(straight_run / "run" / "latest.pt", a / "runs" / "tok" / "latest.pt")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(tiny_config("runs/tok", data_dir, schedule={"max_updates": 8},
                                               dist={"timeout_minutes": 1})))
    results = _launch(path, 2, cwds=[a, b], check=False)
    for code, out in results:
        assert code != 0 and "do not see the same start checkpoint" in out, out[-4000:]
    assert f"(cwd {a})" in results[0][1] and f"(cwd {b})" in results[0][1]
    assert load_checkpoint(a / "runs" / "tok" / "latest.pt")["global_update"] == 6


def test_resume_on_another_world_size(tmp_path, data_dir, patched_lpips):
    cfg = tiny_config(tmp_path / "run", data_dir, schedule={"max_updates": 3})
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    _launch(path, 2)
    first = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert (first["global_update"], first["world_size"]) == (3, 2) and len(first["rng"]) == 2

    # One rank loads the 2-rank optimizer state: gathered back, the Adam moments are unchanged.
    info = TT.init_distributed(require_process_group=True)
    try:
        trainer = TT.TokenizerTrainer(load_config(path, ["schedule.max_updates=5"]), info)
        trainer.setup()
        with trainer._full_state_dict_type():
            loaded = TT.FSDP.optim_state_dict(trainer.root, trainer.optim)["state"]
    finally:
        TT.cleanup()
    assert loaded.keys() == first["optim"]["state"].keys()
    for p, state in first["optim"]["state"].items():
        assert state["exp_avg_sq"].abs().sum() > 0, p
        for k in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(loaded[p][k], state[k]), f"{p}.{k}"

    _launch(path, 1, "schedule.max_updates=5")
    ckpt = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert (ckpt["global_update"], ckpt["world_size"], ckpt["epoch_update"]) == (5, 1, 5)
    steps = {s["step"].item() for s in ckpt["optim"]["state"].values()}
    assert steps == {5.0}
    assert json.loads((tmp_path / "run" / "pretrained" / "config.json").read_text())["provenance"]["global_update"] == 5


def test_resumed_launch_purges_events_after_the_checkpoint(tmp_path, data_dir, patched_lpips, monkeypatch):
    purge_steps = []

    class Recording(TT.ExperimentLogger):
        def __init__(self, *args, purge_step=None, **kwargs):
            purge_steps.append(purge_step)
            super().__init__(*args, purge_step=purge_step, **kwargs)

    monkeypatch.setattr(TT, "ExperimentLogger", Recording)
    run = tmp_path / "run"
    assert run_cfg(tmp_path / "a.yaml", tiny_config(run, data_dir, schedule={"max_updates": 2}))["global_update"] == 2
    # --set reaches keys the YAML leaves at their default (checkpoint.save_every_minutes here).
    cfg = tiny_config(run, data_dir, schedule={"max_updates": 3})
    assert run_cfg(tmp_path / "b.yaml", cfg, "checkpoint.save_every_minutes=60")["global_update"] == 3
    assert purge_steps == [0, 3]  # the resumed launch logs update 3 first
