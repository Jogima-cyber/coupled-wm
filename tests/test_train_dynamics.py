# SPDX-License-Identifier: Apache-2.0
"""Dynamics trainer: configs, schedules, and tiny end-to-end CPU runs of both stages."""

from __future__ import annotations

import copy
import logging
import math
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import coupled_wm.world_model.train_dynamics as td
from conftest import write_h5
from coupled_wm.common.checkpoint import load_checkpoint, load_pretrained, load_pretrained_config
from coupled_wm.world_model.export import export_checkpoint, save_tokenizer_pretrained
from coupled_wm.common.config import save_config
from coupled_wm.world_model.loading import (
    lineage_max_clip_len,
    load_dynamics,
    load_tokenizer,
    tokenizer_model_config,
    tokenizer_sha256,
)
from coupled_wm.world_model.models import G1_ACTION, G1_DYNAMICS_ARCH, G1_INFERENCE, build_dynamics, build_tokenizer
from coupled_wm.world_model.train_dynamics import (
    accumulation_steps,
    cosine_with_warmup,
    export_metadata,
    init_from_checkpoint,
    load_run_config,
    model_arch,
    resolve_color_order,
    resolve_config,
    rms_decay_per_micro_step,
    shortcut_layout,
    train,
    update_plan,
)

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "world_model"

TINY_TOK = dict(image_size=[32, 32], patch_size=8, in_channels=3, d_model=64, n_layers=8, num_heads_q=4,
                num_heads_kv_latent=4, temporal_every=4, n_latents=8, bottleneck_dim=4, use_proprio=True,
                proprio_dim=5, num_proprio=1, seq_len=16, context_len=0, dropout=0.0, mae_max_mask_prob=0.9)
TINY_DYN = dict(action_dim=6, num_latents=8, latent_dim=4, d_model=32, num_layers=4, num_heads=2, num_registers=2,
                num_tau_levels=16, temporal_every=2, mlp_ratio=4.0, seq_len=16, context_len=6, dropout=0.0,
                mtp_length=3, reward_hidden_dim=16, reward_num_buckets=31, reward_symlog_range=[-20.0, 20.0])


def _deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


DATA_KW = dict(image_hw=(32, 32), action_dim=6, proprio_dim=5)


def _tiny_tokenizer(out: Path, seed: int, **metadata) -> Path:
    torch.manual_seed(seed)
    arch = {**TINY_TOK, **metadata.pop("arch", {})}
    enc, dec = build_tokenizer(arch)
    return save_tokenizer_pretrained(out, enc.state_dict(), dec.state_dict(), tokenizer_model_config(arch, **metadata))


def write_pool(data: Path) -> Path:
    """The 2-file pool of :func:`assets` (16 short and 8 long windows)."""
    write_h5(data / "chunk_0000.h5", 12, file_id=0, **DATA_KW)                  # 9 short, 5 long windows
    write_h5(data / "chunk_0001.h5", 13, dones_at=[2, 12], file_id=1, **DATA_KW)  # segments 3 (too short) + 10
    return data


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    """Tiny tokenizer pretrained dirs and a 2-file data pool (16 short and 8 long windows).

    ``tokenizer_b`` has the same architecture as ``tokenizer`` and other weights;
    ``tokenizer_rgb`` records RGB files; ``tokenizer_ctx`` was trained with a 2-frame window.
    """
    root = tmp_path_factory.mktemp("dyn_assets")
    tok = _tiny_tokenizer(root / "tokenizer", 0)
    tok_b = _tiny_tokenizer(root / "tokenizer_b", 1)
    tok_rgb = _tiny_tokenizer(root / "tokenizer_rgb", 2, preprocessing={"h5_color_order": "rgb"})
    tok_ctx = _tiny_tokenizer(root / "tokenizer_ctx", 3, arch={"context_len": 2})
    data = write_pool(root / "data")
    no_reward = root / "no_reward"
    write_h5(no_reward / "chunk_0000.h5", 12, reward_key=None, **DATA_KW)
    big = root / "big_frames"
    write_h5(big / "chunk_0000.h5", 12, **{**DATA_KW, "image_hw": (48, 48)})
    return {"root": root, "tokenizer": tok, "tokenizer_b": tok_b, "tokenizer_rgb": tok_rgb, "tokenizer_ctx": tok_ctx,
            "data": data, "no_reward": no_reward, "big_frames": big}


def tiny_cfg(assets, run_dir, stage="backbone", **over):
    cfg = {
        "stage": stage,
        "run_dir": str(run_dir),
        "seed": 3,
        "tokenizer": {"path": str(assets["tokenizer"])},
        "data": {"dirs": [str(assets["data"])], "num_workers": 0,
                 "reward_key": "rewards" if stage == "reward" else None},
        "model": dict(TINY_DYN),
        "train": {"short_len": 4, "long_len": 8, "long_every": 0 if stage == "reward" else 2,
                  "global_batch": 2, "micro_batch_per_gpu": 1},
        "optim": {"lr": 1e-3},
        "schedule": {"warmup_updates": 2, "total_updates": 50, "max_updates": 12},
        "checkpoint": {"save_every_updates": 100},
        "logging": {"print_every_updates": 4},
    }
    return _deep_update(cfg, over)


def _state(path: Path) -> dict:
    return load_checkpoint(path, mmap=False)


def _assert_states_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


# ----------------------------------------------------------------------------- configs


def test_g1_configs_resolve():
    bb = load_run_config(str(CONFIGS / "g1_dynamics_backbone.yaml"))
    rw = load_run_config(str(CONFIGS / "g1_dynamics_reward.yaml"))
    assert bb["schedule"]["total_updates"] == 210 * 9122 // 4 and bb["schedule"]["warmup_updates"] == int(
        0.05 * (210 * 9122 // 4))
    assert rw["schedule"]["total_updates"] == 210 * 11561 // 4 and rw["schedule"]["warmup_updates"] == int(
        0.05 * (210 * 11561 // 4))
    assert (bb["schedule"]["max_updates"], rw["schedule"]["max_updates"]) == (49000, 3000)
    assert (bb["optim"]["lr"], rw["optim"]["lr"]) == (1e-4, 5e-5)
    assert bb["train"]["long_every"] == 8 and bb["train"]["long_len"] == 128 and rw["train"]["long_every"] == 0
    assert bb["data"]["include_globs"] == ["chunk_*.h5"] and rw["data"]["include_globs"] == ["*.h5"]
    assert rw["data"]["reward_key"] == "rewards" and rw["data"]["reward_remap"] == {10.0: 2.0}
    assert bb["tokenizer"]["context_len"] == rw["tokenizer"]["context_len"] == 0
    assert bb["data"]["color_order"] is None and rw["data"]["color_order"] is None  # the tokenizer's (G1: bgr)
    assert "image_size" not in bb["data"] and "image_size" not in rw["data"]  # frames always take the tokenizer's size
    assert (bb["data"]["start_epoch"], rw["data"]["start_epoch"]) == (0, 1)  # as the paper's two stages
    assert bb["checkpoint"]["save_every_minutes"] == rw["checkpoint"]["save_every_minutes"] == 30
    assert not bb["init_from_allow_unfinished"] and not rw["init_from_allow_unfinished"]
    bb_arch, rw_arch = model_arch(bb), model_arch(rw)
    assert not bb_arch["use_reward_head"] and rw_arch == G1_DYNAMICS_ARCH
    assert {k: v for k, v in bb_arch.items() if k != "use_reward_head"} == \
           {k: v for k, v in G1_DYNAMICS_ARCH.items() if k != "use_reward_head"}
    assert accumulation_steps(bb["train"]["global_batch"], bb["train"]["micro_batch_per_gpu"], 64) == 4
    for cfg in (bb, rw):
        assert (cfg["loss"]["rms_decay"], cfg["loss"]["rms_reference_windows"]) == (0.99, 64)
        assert cfg["train"]["shortcut_fraction"] == 0.25


def test_paper_learning_rates():
    """The schedules reproduce the learning rates stored in the paper checkpoints."""
    def lr_at(cfg_name, step):
        cfg = load_run_config(str(CONFIGS / cfg_name))
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=cfg["optim"]["lr"])
        s = cfg["schedule"]
        sched = cosine_with_warmup(opt, s["warmup_updates"], s["total_updates"], s["min_lr"])
        return cfg["optim"]["lr"] * sched.lr_lambdas[0](step)

    assert lr_at("g1_dynamics_backbone.yaml", 0) == 0.0
    assert lr_at("g1_dynamics_backbone.yaml", 17000) == pytest.approx(7.099603257465023e-05, rel=1e-12)
    assert lr_at("g1_dynamics_backbone.yaml", 33000) == pytest.approx(9.99022923122575e-05, rel=1e-12)
    assert lr_at("g1_dynamics_backbone.yaml", 49000) == pytest.approx(9.925355368869043e-05, rel=1e-12)
    assert lr_at("g1_dynamics_reward.yaml", 3000) == pytest.approx(4.942827956634923e-06, rel=1e-12)


def test_config_errors(assets, tmp_path):
    base = tiny_cfg(assets, tmp_path / "r")
    with pytest.raises(KeyError, match="train.typo"):
        resolve_config(_deep_update(base, {"train": {"typo": 1}}))
    with pytest.raises(KeyError, match="schedule.max_updates"):
        cfg = copy.deepcopy(base)
        del cfg["schedule"]["max_updates"]
        resolve_config(cfg)
    with pytest.raises(ValueError, match="reward_key"):
        resolve_config(_deep_update(tiny_cfg(assets, tmp_path / "r", stage="reward"), {"data": {"reward_key": None}}))
    with pytest.raises(ValueError, match="use_reward_head"):
        resolve_config(_deep_update(base, {"model": {"use_reward_head": True}}))
    with pytest.raises(ValueError, match="max_updates"):
        resolve_config(_deep_update(base, {"schedule": {"max_updates": 51}}))
    with pytest.raises(ValueError, match="seq_len"):
        resolve_config(_deep_update(base, {"train": {"long_len": 32}}))
    with pytest.raises(ValueError, match="shortcut_fraction"):
        resolve_config(_deep_update(base, {"train": {"shortcut_fraction": 1.0}}))
    with pytest.raises(KeyError, match="training"):
        resolve_config(_deep_update(base, {"export": {"metadata": {"training": {}}}}))
    with pytest.raises(ValueError, match="multiple"):
        accumulation_steps(256, 1, 48)
    with pytest.raises(ValueError, match="color_order"):
        resolve_config(_deep_update(base, {"data": {"color_order": "BGR"}}))
    with pytest.raises(KeyError, match="data.image_size"):  # removed: frames are always resized to the tokenizer's
        resolve_config(_deep_update(base, {"data": {"image_size": 32}}))
    with pytest.raises(ValueError, match="context_len"):
        resolve_config(_deep_update(base, {"tokenizer": {"context_len": -1}}))
    with pytest.raises(ValueError, match="start_epoch"):
        resolve_config(_deep_update(base, {"data": {"start_epoch": -1}}))
    with pytest.raises(ValueError, match="save_every_minutes"):
        resolve_config(_deep_update(base, {"checkpoint": {"save_every_minutes": 0}}))
    with pytest.raises(ValueError, match="rms_decay"):
        resolve_config(_deep_update(base, {"loss": {"rms_decay": -0.5}}))
    for bad in (0, 64.0, True):
        with pytest.raises(ValueError, match="rms_reference_windows"):
            resolve_config(_deep_update(base, {"loss": {"rms_reference_windows": bad}}))


def test_export_preset_is_checked_before_any_setup(assets, tmp_path):
    """resolve_config alone (no tokenizer, no data) rejects a preset that does not fit the model."""
    cfg = tiny_cfg(assets, tmp_path / "r", stage="reward", export={"preset": "g1"})
    cfg["tokenizer"]["path"] = str(tmp_path / "missing_tokenizer")
    with pytest.raises(ValueError, match=r"d_min_idx 7 != .*export.preset 'g1'.*export.preset: null"):
        resolve_config(cfg)
    cfg["model"]["num_tau_levels"] = 128  # d_min_idx fits now, the action layout does not
    with pytest.raises(ValueError, match=r"action.dim 22 != model.action_dim 6.*G1 data only"):
        resolve_config(cfg)
    g1_like = _deep_update(cfg, {"model": {"action_dim": 22}})
    assert resolve_config(g1_like)["export"]["preset"] == "g1"
    own = _deep_update(cfg, {"export": {"preset": None, "metadata": {"action": {"dim": 6}}}})
    assert resolve_config(own)["export"]["metadata"]["action"]["dim"] == 6


# (world size, accumulation steps): the paper's 64 x 4, 1 to 256 GPUs at global batch 256, and
# updates of fewer micro-batches than 1 / shortcut_fraction.
LAYOUTS = [(64, 4), (8, 32), (4, 64), (1, 256), (128, 2), (256, 1), (64, 1), (8, 4), (2, 2), (1, 1), (1, 2), (1, 3),
           (2, 1), (3, 1), (3, 5)]


def _update_layout(update: int, world: int, accum: int) -> list[list[bool]]:
    return [shortcut_layout(update, rank, world, accum, Fraction(1, 4)) for rank in range(world)]


@pytest.mark.parametrize("world", [1, 2, 8, 64, 256])
def test_shortcut_layout_is_the_papers_at_four_accumulation_steps(world):
    """F F F S on every rank, for any world size."""
    for update in range(6):
        assert _update_layout(update, world, 4) == [[False, False, False, True]] * world


@pytest.mark.parametrize("world,accum", LAYOUTS)
def test_shortcut_layout_of_every_update(world, accum):
    """A quarter of the micro-batches of every update (exact when the update has a multiple of 4),
    at least one flow micro-batch in every update of more than one, every 4th micro-step of a rank."""
    n = world * accum
    total = 0
    for update in range(16):
        layout = _update_layout(update, world, accum)
        count = sum(map(sum, layout))
        total += count
        assert total == (update + 1) * n // 4  # exact over the updates so far
        assert count in (n // 4, -(-n // 4))
        if n % 4 == 0:
            assert count == n // 4
        if n > 1:
            assert count < n
        for flags in layout:
            steps = [m for m, flag in enumerate(flags) if flag]
            assert all(b - a == 4 for a, b in zip(steps, steps[1:]))  # F F F S F F F S ...
    assert shortcut_layout(5, 0, world, accum, Fraction(0)) == [False] * accum


@pytest.mark.parametrize("world,accum", LAYOUTS)
def test_shortcut_fraction_is_exact_per_clip_length(world, accum):
    """Long (every 8th) and short updates each get a quarter of their micro-batches as shortcut."""
    f = Fraction(1, 4)
    n = world * accum
    long_updates = 0
    seen = {True: [0, 0], False: [0, 0]}  # long? -> [updates, shortcut micro-batches over all ranks]
    for u in range(8 * 4 * 3 + 1):
        plans = [update_plan(u, long_updates, 8, rank, world, accum, f) for rank in range(world)]
        long_update = plans[0][0]
        assert all(p[0] == long_update for p in plans) and long_update == (u > 0 and u % 8 == 0)
        long_updates += long_update
        seen[long_update][0] += 1
        seen[long_update][1] += sum(sum(p[1]) for p in plans)
        for k, (n_upd, n_short) in seen.items():
            assert n_short == n_upd * n // 4, (u, k)  # exact at every point
    assert seen[True][1] > 0 and seen[False][1] > 0
    # Without long clips the plan is the plain per-update layout.
    assert [update_plan(u, 0, 0, world - 1, world, accum, f) for u in range(12)] == \
           [(False, shortcut_layout(u, world - 1, world, accum, f)) for u in range(12)]


def test_flow_and_shortcut_make_the_same_normaliser_calls():
    """Ranks on flow and on shortcut micro-batches update the shared normalisers once each, in
    the same order, so their all-reduces stay aligned."""
    torch.manual_seed(0)
    model = build_dynamics({**TINY_DYN, "use_reward_head": True})
    names = []

    class Recording(td.RMSLossScaler):
        def __call__(self, name, value):
            names.append(name)
            return super().__call__(name, value)

    scaler = Recording()
    z, actions, rewards = torch.rand(2, 5, 8, 4) * 2 - 1, torch.randn(2, 5, 6), torch.zeros(2, 5)
    calls = {}
    for shortcut in (False, True):
        names.clear()
        levels = td.sample_noise_levels(2, 5, torch.device("cpu"), shortcut, 16)
        td.dynamics_loss(model, model, actions, z, rewards, levels, shortcut, scaler, 16, model.reward_head.heads[0])
        calls[shortcut] = list(names)
    assert calls[False] == calls[True] == ["loss_reward", "loss_dyn"]


def _normaliser_trace(decay: float, accum: int, updates: int = 40) -> list[float]:
    """Running RMS after each update when the loss jumps from 1 to 3 (one call per micro-step)."""
    scaler = td.RMSLossScaler(decay)
    scaler("loss_dyn", torch.tensor(1.0))
    trace = []
    for _ in range(updates):
        for _ in range(accum):
            scaler("loss_dyn", torch.tensor(3.0))
        trace.append(scaler.rms("loss_dyn"))
    return trace


def test_loss_normaliser_horizon_is_defined_in_clips():
    """The paper's 0.99 per micro-step on 64 GPUs x 1 clip; the same decay per clip, hence per
    update and the same trace of the running RMS, on any number of GPUs."""
    for name in ("g1_dynamics_backbone.yaml", "g1_dynamics_reward.yaml"):
        cfg = load_run_config(str(CONFIGS / name))
        loss, global_batch = cfg["loss"], cfg["train"]["global_batch"]
        assert rms_decay_per_micro_step(loss["rms_decay"], 64 * 1, loss["rms_reference_windows"]) == 0.99
        paper = _normaliser_trace(0.99, accum=4)
        for world, micro_batch in [(64, 1), (1, 1), (4, 1), (8, 1), (8, 4), (16, 2), (64, 4), (128, 1), (256, 1)]:
            clips = world * micro_batch
            accum = accumulation_steps(global_batch, micro_batch, world)
            decay = rms_decay_per_micro_step(loss["rms_decay"], clips, loss["rms_reference_windows"])
            assert decay ** (1 / clips) == pytest.approx(0.99 ** (1 / 64), rel=1e-14)  # per clip
            assert decay ** accum == pytest.approx(0.99 ** 4, rel=1e-12)  # per update of 256 clips
            # fp32 state: rounding over up to 256 micro-steps per update
            assert _normaliser_trace(decay, accum) == pytest.approx(paper, rel=1e-3), (name, world, micro_batch)
        assert _normaliser_trace(0.99, accum=256) != pytest.approx(paper, rel=1e-1)  # 0.99 per micro-step on 1 GPU


def test_trainer_normaliser_decay_follows_the_clips_per_micro_step(assets, tmp_path, monkeypatch):
    decays = []

    class Recording(td.RMSLossScaler):
        def __init__(self, decay, eps):
            decays.append(decay)
            super().__init__(decay, eps)

    monkeypatch.setattr(td, "RMSLossScaler", Recording)
    train(tiny_cfg(assets, tmp_path / "a", schedule={"max_updates": 1}))  # 1 rank x 1 clip
    train(tiny_cfg(assets, tmp_path / "b", schedule={"max_updates": 1}, loss={"rms_reference_windows": 1}))
    assert decays == [0.99 ** (1 / 64), 0.99]


def test_resolve_color_order(caplog):
    rgb = {"preprocessing": {"h5_color_order": "rgb"}}
    assert resolve_color_order(None, rgb) == "rgb"
    assert resolve_color_order(None, {"preprocessing": {}}) == resolve_color_order(None, {}) == "bgr"
    with caplog.at_level(logging.WARNING, logger=td.__name__):
        assert resolve_color_order("rgb", rgb) == "rgb" and not caplog.records
        assert resolve_color_order("bgr", rgb) == "bgr"
    assert "channel-swapped" in caplog.text


# ----------------------------------------------------------------------------- runs


def test_backbone_run_checkpoint_and_export(assets, tmp_path):
    run_dir = tmp_path / "bb"
    dyn = train(tiny_cfg(assets, run_dir, schedule={"max_updates": 6}))
    ckpt = _state(run_dir / "latest.pt")
    assert ckpt["stage"] == "backbone" and ckpt["global_update"] == 6
    assert ckpt["counters"] == {"stage_update": 6, "long_updates": 2}  # updates 2 and 4
    assert ckpt["model_config"]["architecture"]["use_reward_head"] is False
    assert not any(k.startswith(("agent_token", "reward_head")) for k in ckpt["dyn"])
    assert set(ckpt["loss_scaler"]) == {"loss_dyn"}
    assert ckpt["data_state"]["short"] == {"epoch": 0, "position": 12}
    assert ckpt["data_state"]["long"] == {"epoch": 0, "position": 4}
    assert ckpt["model_config"]["tokenizer"]["sha256"] == tokenizer_sha256(assets["tokenizer"])
    assert ckpt["scheduler"]["last_epoch"] == 6
    _assert_states_equal(ckpt["dyn"], {k: v.detach() for k, v in dyn.state_dict().items()})

    loaded, cfg = load_dynamics(run_dir / "pretrained")
    assert cfg["architecture"] == ckpt["model_config"]["architecture"]
    assert cfg["training"]["stage_update"] == 6 and cfg["training"]["data"]["files"] == 2
    assert cfg["training"]["lineage_max_clip_len"] == 8 == lineage_max_clip_len(cfg)  # the long clips
    _assert_states_equal(loaded.state_dict(), ckpt["dyn"])
    exported = export_checkpoint(run_dir / "latest.pt", tmp_path / "exported")
    _assert_states_equal(load_pretrained(exported)[0], load_pretrained(run_dir / "pretrained")[0])


def test_resume_continues_the_uninterrupted_run(assets, tmp_path):
    """Stop mid-epoch and mid long-clip pass, resume: identical to one uninterrupted run."""
    straight = tmp_path / "straight"
    train(tiny_cfg(assets, straight, schedule={"max_updates": 12}))
    split = tmp_path / "split"
    train(tiny_cfg(assets, split, schedule={"max_updates": 5}))
    assert _state(split / "latest.pt")["data_state"]["short"] == {"epoch": 0, "position": 10}
    train(tiny_cfg(assets, split, schedule={"max_updates": 12}))

    a, b = _state(straight / "latest.pt"), _state(split / "latest.pt")
    assert a["data_state"] == b["data_state"] == {"short": {"epoch": 1, "position": 8},
                                                  "long": {"epoch": 1, "position": 2}}
    assert a["counters"] == b["counters"] == {"stage_update": 12, "long_updates": 5}
    _assert_states_equal(a["dyn"], b["dyn"])
    _assert_states_equal(a["loss_scaler"], b["loss_scaler"])
    for sa, sb in zip(a["optim"]["state"].values(), b["optim"]["state"].values(), strict=True):
        _assert_states_equal(sa, sb)

    # Re-running a finished stage trains nothing.
    train(tiny_cfg(assets, split, schedule={"max_updates": 12}))
    _assert_states_equal(_state(split / "latest.pt")["dyn"], b["dyn"])


def test_resume_refuses_another_stage_or_architecture(assets, tmp_path):
    run_dir = tmp_path / "bb"
    train(tiny_cfg(assets, run_dir, schedule={"max_updates": 1}))
    with pytest.raises(ValueError, match="stage"):
        train(tiny_cfg(assets, run_dir, stage="reward", init_from=str(run_dir / "latest.pt")))
    with pytest.raises(ValueError, match="num_layers"):
        train(tiny_cfg(assets, run_dir, model={"num_layers": 2}))


@pytest.fixture(scope="module")
def backbone(assets, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("backbone")
    train(tiny_cfg(assets, run_dir, schedule={"max_updates": 4}))
    return run_dir


def test_reward_init_keeps_backbone_and_fresh_head(assets, backbone):
    arch = {**TINY_DYN, "use_reward_head": True}
    torch.manual_seed(11)
    model = build_dynamics(arch)
    fresh = copy.deepcopy(model.state_dict())
    info, scaler_state = init_from_checkpoint(model, backbone / "latest.pt", arch)
    src = _state(backbone / "latest.pt")["dyn"]
    for k, v in model.state_dict().items():
        expected = fresh[k] if k.startswith(("agent_token", "reward_head.")) else src[k]
        assert torch.equal(v, expected), k
    assert info["global_update"] == 4 and set(scaler_state) == {"loss_dyn"}

    info, scaler_state = init_from_checkpoint(build_dynamics(arch), backbone / "pretrained", arch)
    assert info["global_update"] == 4 and scaler_state is None


def test_reward_stage_run(assets, backbone, tmp_path):
    run_dir = tmp_path / "rw"
    dyn = train(tiny_cfg(assets, run_dir, stage="reward", init_from=str(backbone / "latest.pt"),
                         schedule={"max_updates": 3}))
    ckpt = _state(run_dir / "latest.pt")
    assert ckpt["stage"] == "reward" and ckpt["global_update"] == 4 + 3
    assert ckpt["counters"] == {"stage_update": 3, "long_updates": 0}
    assert ckpt["init"]["global_update"] == 4 and ckpt["scheduler"]["last_epoch"] == 3
    assert set(ckpt["loss_scaler"]) == {"loss_dyn", "loss_reward"}
    assert dyn.use_reward_head and ckpt["model_config"]["architecture"]["use_reward_head"] is True
    assert all(torch.isfinite(v).all() for v in ckpt["dyn"].values() if v.is_floating_point())
    cfg = load_pretrained_config(run_dir / "pretrained")
    assert cfg["training"]["init_from"]["global_update"] == 4 and cfg["training"]["global_update"] == 7
    assert cfg["training"]["init_from"]["tokenizer"]["sha256"] == tokenizer_sha256(assets["tokenizer"])
    assert cfg["training"]["data"]["reward_key"] == "rewards"
    # 4-frame clips here, but the backbone it starts from trained on 8-frame clips.
    assert cfg["training"]["lineage_max_clip_len"] == 8 == lineage_max_clip_len(cfg)

    # From the exported backbone dir (fresh loss scaler) instead of latest.pt.
    train(tiny_cfg(assets, tmp_path / "rw_dir", stage="reward", init_from=str(backbone / "pretrained"),
                   schedule={"max_updates": 1}))


def test_reward_stage_errors(assets, backbone, tmp_path):
    with pytest.raises(ValueError, match="init_from"):
        train(tiny_cfg(assets, tmp_path / "a", stage="reward"))
    with pytest.raises(FileNotFoundError):
        train(tiny_cfg(assets, tmp_path / "b", stage="reward", init_from=str(tmp_path / "missing.pt")))
    reward_run = tmp_path / "c"
    train(tiny_cfg(assets, reward_run, stage="reward", init_from=str(backbone / "latest.pt"),
                   schedule={"max_updates": 1}))
    with pytest.raises(ValueError, match="already has a reward head"):
        train(tiny_cfg(assets, tmp_path / "d", stage="reward", init_from=str(reward_run / "latest.pt")))
    with pytest.raises(KeyError, match="rewards"):
        train(tiny_cfg(assets, tmp_path / "e", stage="reward", init_from=str(backbone / "latest.pt"),
                       data={"dirs": [str(assets["no_reward"])]}))
    with pytest.raises(ValueError, match="d_min_idx"):
        train(tiny_cfg(assets, tmp_path / "f", stage="reward", init_from=str(backbone / "latest.pt"),
                       export={"preset": "g1"}))
    with pytest.raises(ValueError, match="action_dim"):
        train(tiny_cfg(assets, tmp_path / "g", model={"action_dim": 7}))


def test_reward_stage_refuses_another_tokenizer(assets, backbone, tmp_path, caplog):
    """The backbone records its tokenizer's sha256; the reward stage must run on the same latents."""
    arch = {**TINY_DYN, "use_reward_head": True}
    other = {"path": str(assets["tokenizer_b"]), "sha256": tokenizer_sha256(assets["tokenizer_b"])}
    for source in (backbone / "latest.pt", backbone / "pretrained"):
        with pytest.raises(ValueError, match="trained with the tokenizer"):
            init_from_checkpoint(build_dynamics(arch), source, arch, other)
    with pytest.raises(ValueError, match="trained with the tokenizer"):
        train(tiny_cfg(assets, tmp_path / "rw", stage="reward", init_from=str(backbone / "latest.pt"),
                       tokenizer={"path": str(assets["tokenizer_b"])}))
    same = {"path": str(assets["tokenizer"]), "sha256": tokenizer_sha256(assets["tokenizer"])}
    info, _ = init_from_checkpoint(build_dynamics(arch), backbone / "latest.pt", arch, same)
    assert info["tokenizer"]["sha256"] == same["sha256"]
    # A tokenizer latest.pt has no sha256: the pairing cannot be checked, which is logged.
    with caplog.at_level(logging.WARNING, logger=td.__name__):
        init_from_checkpoint(build_dynamics(arch), backbone / "latest.pt", arch, {"path": "runs/tok/latest.pt"})
    assert "make sure these are the same weights" in caplog.text


def test_shortcut_micro_batches_reach_long_clips(assets, tmp_path, monkeypatch):
    """With one micro-batch per update the long-clip updates also train the shortcut objective."""
    calls = []
    real = td.sample_noise_levels

    def spy(B, T, device, shortcut, num_tau_levels):
        calls.append((T, shortcut))
        return real(B, T, device, shortcut, num_tau_levels)

    monkeypatch.setattr(td, "sample_noise_levels", spy)
    train(tiny_cfg(assets, tmp_path / "g1", train={"global_batch": 1}, schedule={"max_updates": 17}))
    long = [s for T, s in calls if T == 8]
    short = [s for T, s in calls if T == 4]
    assert (len(long), len(short)) == (8, 9)
    assert (sum(long), sum(short)) == (2, 2)  # floor(n / 4) for each clip length


def test_logged_losses_follow_the_layout(assets, tmp_path, monkeypatch):
    """4 accumulation steps: F F F S in every update, so every update logs both objectives."""
    logged = []

    class Recording(td.ExperimentLogger):
        def scalars(self, values, step, prefix=""):
            logged.append(dict(values))
            super().scalars(values, step, prefix)

    monkeypatch.setattr(td, "ExperimentLogger", Recording)
    train(tiny_cfg(assets, tmp_path / "r", train={"global_batch": 4}, schedule={"max_updates": 3}))
    assert [v["train/shortcut_fraction"] for v in logged] == [0.25] * 3
    assert all(math.isfinite(v["train/loss_flow"]) and math.isfinite(v["train/loss_shortcut"]) for v in logged)


def test_color_order_follows_the_tokenizer(assets, tmp_path, monkeypatch, caplog):
    orders = []

    class Spy(td.HDF5SequenceDataset):
        def __init__(self, *args, **kwargs):
            orders.append(kwargs["color_order"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(td, "HDF5SequenceDataset", Spy)
    rgb_tok = {"path": str(assets["tokenizer_rgb"])}
    run_dir = tmp_path / "rgb"
    train(tiny_cfg(assets, run_dir, tokenizer=rgb_tok, schedule={"max_updates": 1}))
    assert orders == ["rgb", "rgb"]  # short and long windows
    assert _state(run_dir / "latest.pt")["model_config"]["training"]["data"]["color_order"] == "rgb"
    with caplog.at_level(logging.WARNING, logger=td.__name__):
        train(tiny_cfg(assets, tmp_path / "bgr", tokenizer=rgb_tok, data={"color_order": "bgr"},
                       schedule={"max_updates": 1}))
    assert orders[2:] == ["bgr", "bgr"] and "channel-swapped" in caplog.text


def test_encoder_context_len_defaults_to_the_tokenizer(assets, tmp_path, caplog):
    def window(cfg) -> int:
        encoder, _, used = td._load_encoder(resolve_config(cfg), torch.device("cpu"), 8)
        assert max(layer.context_len for layer in encoder.layers) == used
        return used

    base = tiny_cfg(assets, tmp_path / "r", tokenizer={"path": str(assets["tokenizer_ctx"])})
    with caplog.at_level(logging.WARNING, logger=td.__name__):
        assert window(base) == 2 and not caplog.records
        assert window(_deep_update(base, {"tokenizer": {"context_len": 0}})) == 0
    assert "overrides the temporal window the tokenizer was trained with (2)" in caplog.text
    run_dir = tmp_path / "run"
    train(_deep_update(base, {"run_dir": str(run_dir), "schedule": {"max_updates": 1}}))
    assert _state(run_dir / "latest.pt")["model_config"]["tokenizer"]["encoder_context_len"] == 2


def test_frames_are_resized_to_the_tokenizer_size(assets, tmp_path, caplog):
    cfg = tiny_cfg(assets, tmp_path / "a", data={"dirs": [str(assets["big_frames"])]}, train={"long_every": 0},
                   schedule={"max_updates": 1})
    with caplog.at_level(logging.INFO, logger=td.__name__):
        train(cfg)
    assert "resizing [[48, 48]] frames to the tokenizer's size [32, 32]" in caplog.text


def test_start_and_resume_are_logged_with_absolute_paths(assets, tmp_path, caplog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tiny_cfg(assets, "runs/bb", schedule={"max_updates": 1})
    with caplog.at_level(logging.INFO, logger=td.__name__):
        train(cfg)
        train(_deep_update(cfg, {"schedule": {"max_updates": 2}}))
    latest = (tmp_path / "runs" / "bb" / "latest.pt").resolve()
    assert f"no {latest}: starting stage backbone from scratch" in caplog.text
    assert f"resuming {latest} at stage update 1" in caplog.text


def test_wall_clock_saves_and_partial_save_cleanup(assets, tmp_path, caplog):
    def saves(run_dir, **checkpoint) -> int:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=td.__name__):
            train(tiny_cfg(assets, run_dir, schedule={"max_updates": 3}, checkpoint=checkpoint))
        return sum(r.getMessage().startswith("saved ") for r in caplog.records)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    partials = [run_dir / ".latest.pt.tmp", run_dir / ".latest.pt.k2j3h4.tmp"]  # saves killed mid-write
    for partial in partials:
        partial.write_bytes(b"partial")
    assert saves(run_dir, save_every_minutes=1e-9) == 3  # due after every update
    assert not any(p.exists() for p in partials) and "interrupted save" in caplog.text
    assert saves(tmp_path / "default") == 1  # 30 minutes: only the final save
    assert saves(tmp_path / "off", save_every_minutes=None) == 1


def test_start_epoch(assets, backbone, tmp_path):
    run_dir = tmp_path / "rw"
    train(tiny_cfg(assets, run_dir, stage="reward", init_from=str(backbone / "latest.pt"), data={"start_epoch": 1},
                   schedule={"max_updates": 1}))
    ckpt = _state(run_dir / "latest.pt")
    assert ckpt["data_state"]["short"] == {"epoch": 1, "position": 2}
    assert ckpt["model_config"]["training"]["data"]["start_epoch"] == 1


def test_lr_schedule_shape():
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
    sched = cosine_with_warmup(opt, 10, 110, 1e-8)
    f = sched.lr_lambdas[0]
    assert f(0) == 0.0 and f(5) == 0.5 and f(10) == 1.0
    assert f(60) == pytest.approx(0.5) and f(110) == pytest.approx(1e-8 / 2e-4)
    assert f(35) == pytest.approx(0.5 * (1 + math.cos(math.pi * 0.25)))


# ----------------------------------------------------------------------------- integration


def test_set_reaches_keys_the_yaml_omits(assets, tmp_path):
    path = tmp_path / "cfg.yaml"
    save_config(tiny_cfg(assets, tmp_path / "run"), path)
    cfg = load_run_config(str(path), ["data.min_chunk_index=21", "dist.timeout_minutes=5"])
    assert cfg["data"]["min_chunk_index"] == 21 and cfg["dist"]["timeout_minutes"] == 5
    with pytest.raises(KeyError, match="data.min_chunk_indx"):
        load_run_config(str(path), ["data.min_chunk_indx=21"])


def test_export_metadata_overlays_the_preset_block_by_block():
    scale = [0.5] * 22
    cfg = {"export": {"preset": "g1", "metadata": {"action": {"scale": scale}, "control_hz": 5}}}
    out = export_metadata(cfg)
    assert out["action"] == {**G1_ACTION, "scale": scale}  # dim, grip_dims and grip_range kept
    assert out["inference"] == G1_INFERENCE and out["control_hz"] == 5


def test_reward_export_without_an_action_layout_warns(assets, backbone, tmp_path, caplog):
    def warnings(**export):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=td.__name__):
            train(tiny_cfg(assets, tmp_path / str(len(export)), stage="reward", init_from=str(backbone / "latest.pt"),
                           schedule={"max_updates": 1}, export=export))
        return [r.getMessage() for r in caplog.records if "action" in r.getMessage()]

    assert any("no action block" in m for m in warnings())
    action = {"dim": 6, "scale": [1.0] * 6, "grip_dims": [5], "grip_range": [0.0, 1.0]}
    assert not warnings(metadata={"action": action})


def test_resumed_launch_purges_events_after_the_checkpoint(assets, tmp_path, monkeypatch):
    purge_steps = []

    class Recording(td.ExperimentLogger):
        def __init__(self, *args, purge_step=None, **kwargs):
            purge_steps.append(purge_step)
            super().__init__(*args, purge_step=purge_step, **kwargs)

    monkeypatch.setattr(td, "ExperimentLogger", Recording)
    run_dir = tmp_path / "bb"
    train(tiny_cfg(assets, run_dir, schedule={"max_updates": 2}))
    train(tiny_cfg(assets, run_dir, schedule={"max_updates": 3}))
    assert purge_steps == [0, 3]  # the resumed launch logs update 3 first


def test_resume_with_another_tokenizer_file_warns(assets, tmp_path, caplog):
    """Tokenizer .pt files record no sha256: a changed path is all the trainer can report."""
    enc, dec, tok_cfg = load_tokenizer(assets["tokenizer"])
    for name in ("tok_a.pt", "tok_b.pt"):
        torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "model_config": tok_cfg}, tmp_path / name)
    run_dir = tmp_path / "bb"
    train(tiny_cfg(assets, run_dir, tokenizer={"path": str(tmp_path / "tok_a.pt")}, schedule={"max_updates": 1}))
    with caplog.at_level(logging.WARNING, logger=td.__name__):
        train(tiny_cfg(assets, run_dir, tokenizer={"path": str(tmp_path / "tok_b.pt")}, schedule={"max_updates": 2}))
    assert "was trained with the tokenizer" in caplog.text and "tok_a.pt" in caplog.text
