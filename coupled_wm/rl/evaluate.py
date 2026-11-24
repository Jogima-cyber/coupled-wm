# SPDX-License-Identifier: Apache-2.0
"""Standalone evaluation of a DMO policy on the world-model environment.

Runs the deterministic policy (RSSM posterior filter + the actor's mean action, fp32 weights,
the same computation as the trainer's evaluation) on the environment of the policy's training
config, so episodes are seeded from the same pool as in training. Writes per-episode returns,
a summary, and one image strip per saved trajectory (initial frame, then every generated frame
with the env's reward for it; the last frame is the episode's terminal frame).

    python -m coupled_wm.rl.evaluate --checkpoint runs/g1_dmo/s20/best_policy.pt \\
        --episodes 64 --num-envs 16 --output-dir runs/g1_dmo/s20/eval

``--set`` overrides the config (``--set env.loader_seed=0`` pins the seed windows). Checkpoints
from the paper's code (no embedded config, wrapper-prefixed keys) need ``--config``; the paper's
policies were trained with the seed window's base command held (``env.dropped_dims_reset:
hold``), while ``configs/rl/g1_dmo.yaml`` sets ``skip_nonzero``, so add ``--set
env.dropped_dims_reset=hold`` to evaluate them in the paper's environment.
"""

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

from coupled_wm.common.checkpoint import load_checkpoint, strip_compile_prefix
from coupled_wm.common.config import apply_overrides, load_config
from coupled_wm.common.logging import setup_logging
from coupled_wm.rl.actor import build_actor
from coupled_wm.rl.dmo import POLICY_FORMAT, TRAINING_STATE_FORMAT, RSSMPolicy, make_env
from coupled_wm.rl.utils.common import seeding
from coupled_wm.rssm.rssm import build_rssm, rssm_config

log = logging.getLogger("coupled_wm.rl.evaluate")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="policy (best_policy.pt, ...) or training checkpoint")
    parser.add_argument("--config", help="config to use instead of the checkpoint's (required for paper checkpoints, "
                                         "with --set env.dropped_dims_reset=hold for the paper's environment)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=None, help="episodes to run (default: num_envs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", help="default: <checkpoint dir>/eval_<checkpoint name>")
    parser.add_argument("--save-trajectories", type=int, default=4, help="image strips of the first N envs' first episode")
    parser.add_argument("--device", default=None, help="default: cuda if available")
    return parser.parse_args(argv)


def build_policy_modules(payload, cfg, num_actions, img_size, device):
    """``(actor, rssm, drop_action_indices)`` with the checkpoint's weights, fp32, eval mode."""
    if payload.get("format") in (POLICY_FORMAT, TRAINING_STATE_FORMAT):
        trained_cfg = payload["config"]
        rssm_arch = rssm_config(payload["rssm_config"])
        drop = tuple(payload["drop_action_indices"])
        if int(payload["num_actions"]) != num_actions:
            raise ValueError(f"policy was trained with {payload['num_actions']} env actions, the env has {num_actions}")
    else:
        trained_cfg = cfg
        rssm_arch = rssm_config({**cfg["rssm"], "action_size": num_actions, "image_size": img_size})
        drop = tuple(int(i) for i in cfg["env"].get("drop_action_indices", ()) or ())
    if rssm_arch["image_size"] != img_size:
        raise ValueError(f"the RSSM was trained on {rssm_arch['image_size']}px frames, the env returns {img_size}px")

    rssm = build_rssm(rssm_arch)
    rssm.load_state_dict(strip_compile_prefix(payload["rssm"]), strict=True)
    obs_dim = rssm_arch["stochastic_size"] * rssm_arch["discrete_size"] + rssm_arch["recurrent_state_size"]
    actor = build_actor(trained_cfg["actor"], obs_dim, num_actions - len(drop), device)
    actor.load_state_dict(strip_compile_prefix(payload["actor"]), strict=True)
    return actor.eval(), rssm.to(device).eval(), drop


def save_strip(frames, rewards, path):
    """Horizontal strip of ``frames (T+1, 3, H, W)`` with ``rewards (T,)`` drawn on frames 1..T."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    tiles = []
    for t, frame in enumerate(frames):
        img = (np.clip(frame.permute(1, 2, 0).numpy(), 0.0, 1.0) * 255).astype(np.uint8)
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        draw.text((2, 2), f"t={t}", fill=(255, 255, 255), font=font)
        if t > 0:
            draw.text((2, pil.height - 12), f"R:{rewards[t - 1]:+.2f}", fill=(255, 255, 0), font=font)
        tiles.append(np.asarray(pil))
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)


@torch.no_grad()
def evaluate(env, policy, num_episodes, gamma, save_trajectories=0):
    """Run whole episodes until ``num_episodes`` have finished (all envs end together).

    Returns ``(episodes, trajectories)``: per-episode dicts (return, discounted return, length)
    and, for the first ``save_trajectories`` envs, ``(frames (T+1, 3, H, W), rewards (T,))`` of
    their first episode.
    """
    n = env.num_envs
    episodes = []
    trajectories = []
    k = min(save_trajectories, n)
    obs = env.reset(torch.arange(n)).clone()
    policy.reset()
    frames = [obs[:k].float().cpu()]
    step_rewards = []
    ret = torch.zeros(n, dtype=torch.float64)
    disc = torch.zeros(n, dtype=torch.float64)
    discount = torch.ones(n, dtype=torch.float64)
    length = torch.zeros(n, dtype=torch.long)
    while len(episodes) < num_episodes:
        action = policy.act(obs)
        next_obs, reward, done, info = env.step(action)
        done = info["dones"]
        policy.observe_dones(done)
        reward = reward.detach().double().cpu().reshape(n)
        ret += reward
        disc += discount * reward
        discount *= gamma
        length += 1
        if not trajectories and k:
            shown = info["obs_before_reset"] if bool(done.any()) else next_obs
            frames.append(shown[:k].float().cpu())
            step_rewards.append(reward[:k].clone())
        done_cpu = done.cpu()
        if bool(done_cpu.any()):
            for e in done_cpu.nonzero().flatten().tolist():
                episodes.append({"env": e, "return": ret[e].item(), "discounted_return": disc[e].item(),
                                 "length": int(length[e])})
            if not trajectories and k:
                frame_stack = torch.stack(frames, dim=1)       # (k, T+1, 3, H, W)
                reward_stack = torch.stack(step_rewards, dim=1)  # (k, T)
                trajectories = [(frame_stack[b], reward_stack[b]) for b in range(k)]
            ret[done_cpu] = 0
            disc[done_cpu] = 0
            discount[done_cpu] = 1
            length[done_cpu] = 0
        obs = next_obs.clone()
    return episodes, trajectories


def main(argv=None):
    args = parse_args(argv)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True
    setup_logging(0)

    payload = load_checkpoint(args.checkpoint)
    if args.config:
        cfg = load_config(args.config, args.set)
    elif payload.get("config") is not None:
        cfg = apply_overrides(payload["config"], args.set)
    else:
        raise SystemExit(f"{args.checkpoint} has no embedded config: pass --config")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seeding(args.seed)
    env = make_env(cfg["env"], args.num_envs, device, int(cfg["train"]["max_episode_length"]))
    actor, rssm, drop = build_policy_modules(payload, cfg, int(env.num_actions), int(env.img_size), device)
    policy = RSSMPolicy(actor, rssm, int(env.num_actions), drop, args.num_envs, device)

    num_episodes = args.episodes or args.num_envs
    gamma = float(cfg["train"].get("gamma", 0.99))
    episodes, trajectories = evaluate(env, policy, num_episodes, gamma, args.save_trajectories)

    out = Path(args.output_dir) if args.output_dir else (
        Path(args.checkpoint).parent / f"eval_{Path(args.checkpoint).stem}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["env", "return", "discounted_return", "length"])
        writer.writeheader()
        writer.writerows(episodes)
    returns = np.array([e["return"] for e in episodes])
    summary = {
        "checkpoint": str(args.checkpoint),
        "iteration": payload.get("iter"),
        "episodes": len(episodes),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "sem_return": float(returns.std(ddof=1) / math.sqrt(len(returns))) if len(returns) > 1 else None,
        "mean_discounted_return": float(np.mean([e["discounted_return"] for e in episodes])),
        "mean_length": float(np.mean([e["length"] for e in episodes])),
        "seed": args.seed,
        "num_envs": args.num_envs,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for b, (frames, rewards) in enumerate(trajectories):
        save_strip(frames, rewards.tolist(), out / f"traj_{b}.png")
    log.info("%d episodes: mean return %.4f (std %.4f); results in %s", len(episodes), summary["mean_return"],
             summary["std_return"], out)
    if hasattr(env, "close"):
        env.close()
    return summary


if __name__ == "__main__":
    main()
