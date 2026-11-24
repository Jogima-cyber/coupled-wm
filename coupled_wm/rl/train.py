# SPDX-License-Identifier: Apache-2.0
"""Train (or evaluate) a DMO policy on the G1 world-model environment.

Launch with torchrun, inside a SLURM ``srun`` step, or as a single process::

    torchrun --nproc_per_node=4 -m coupled_wm.rl.train --config configs/rl/g1_dmo.yaml \\
        --seed 20 --logdir runs/g1_dmo/s20 --no-eval

A run directory that already holds ``latest.pt`` is resumed; it must be on storage every node
sees. Unless ``train.checkpoint_replay`` is set, a resumed run refills ``train.min_replay`` replay
buffer rows with the current policy and does not rerun the ``rssm.init_udt`` warm-up (see
:mod:`coupled_wm.rl.dmo`).

Ablations: ``--no-diffusion`` (forward and backward pass of the actor's unroll both in the
RSSM, the local model, instead of a forward pass in the global model; no online RSSM
finetuning; alias ``--dreamer-dreamer-dreamer``) and ``--no-rssm-finetuning``. Evaluate a
checkpoint with ``--play --checkpoint <run>/best_policy.pt``.

Stopping: SIGTERM or SIGUSR1 makes every rank finish its iteration, write ``latest.pt`` and exit
with :data:`EXIT_STOPPED` (75); relaunching the same command resumes. torchrun forwards SIGTERM
to the training processes and kills them 30 s later, which a G1 iteration plus the save may not
fit in: send SIGUSR1 to the training processes ahead of a time limit instead (with ``srun``
tasks, ``--signal=USR1@300``).

``torch.compile`` (``train.compile`` and the G1 env's ``env.compile_encoder``) builds Triton
kernels at the first call, which needs a C compiler and the Python headers; without them, run
eagerly with ``--set train.compile=false --set env.compile_encoder=false``.
"""

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from coupled_wm.common.checkpoint import load_checkpoint
from coupled_wm.common.config import apply_overrides, load_config
from coupled_wm.common.distributed import cleanup, init_distributed
from coupled_wm.common.logging import setup_logging
from coupled_wm.rl.dmo import DMOTrainer, RunOptions

log = logging.getLogger("coupled_wm.rl.train")

EXIT_STOPPED = 75
"""Exit code of a run stopped by a signal before ``train.max_epochs`` (``EX_TEMPFAIL``)."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML config (required for training; --play defaults to the checkpoint's)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set train.max_epochs=100 (repeatable)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (rank r uses seed + r)")
    parser.add_argument("--logdir", help="run directory (config, TensorBoard logs, checkpoints)")
    parser.add_argument("--exp-name", help="experiment name for wandb (default: the run directory name)")
    parser.add_argument("--resume", metavar="CKPT",
                        help="resume from this training checkpoint (default: <logdir>/latest.pt if it exists)")
    parser.add_argument("--no-eval", action="store_true", help="no evaluation during and after training")
    parser.add_argument("--no-diffusion", "--dreamer-dreamer-dreamer", dest="no_diffusion", action="store_true",
                        help="ablation: unroll the actor in the RSSM's imagination, no online RSSM finetuning")
    parser.add_argument("--no-rssm-finetuning", action="store_true", help="ablation: keep the warm-start RSSM frozen")
    parser.add_argument("--play", action="store_true", help="evaluate --checkpoint instead of training")
    parser.add_argument("--checkpoint", help="policy or training checkpoint for --play")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True

    info = init_distributed()
    setup_logging(info.rank)
    try:
        if args.play:
            return _play(args, info)
        return _train(args, info)
    finally:
        cleanup()


def _load_cfg(args, fallback=None):
    if args.config:
        return load_config(args.config, args.set)
    if fallback is None:
        raise SystemExit("--config is required")
    return apply_overrides(fallback, args.set)


def _train(args, info):
    if not args.logdir:
        raise SystemExit("--logdir is required for training")
    cfg = _load_cfg(args)
    run = RunOptions(seed=args.seed, logdir=args.logdir, exp_name=args.exp_name, no_eval=args.no_eval,
                     no_diffusion=args.no_diffusion, no_rssm_finetuning=args.no_rssm_finetuning)
    resume = args.resume
    if resume is None and (Path(args.logdir) / "latest.pt").is_file():
        resume = str(Path(args.logdir) / "latest.pt")
    _check_resume_agreement(resume, info)
    if resume is not None:
        saved_run = load_checkpoint(resume).get("run") or {}
        for flag in ("no_diffusion", "no_rssm_finetuning"):
            if bool(saved_run.get(flag, False)) != getattr(run, flag):
                raise SystemExit(f"{resume} was trained with {flag}={saved_run.get(flag)}; pass the same flag to resume")
        run.resume = resume
    log.info("run options: %s", asdict(run))
    log.info("%s in %s", f"resuming {Path(resume).resolve()}" if resume else "new run", Path(args.logdir).resolve())
    trainer = DMOTrainer(cfg, run, info)
    if not trainer.train():
        raise SystemExit(EXIT_STOPPED)


def _check_resume_agreement(resume, info):
    """Every rank must resume from the same checkpoint or none: only rank 0 writes ``latest.pt``,
    and ranks that start fresh next to resumed ones desynchronise the DDP collectives."""
    found = resume is not None and Path(resume).is_file()
    if info.world_size > 1 and dist.is_initialized():
        flags = torch.tensor([float(found), -float(found)], device=info.device)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)  # (any rank found it, -(every rank found it))
        if flags[0].item() != -flags[1].item():
            raise SystemExit(f"ranks disagree on resuming (rank {info.rank}: {resume or 'no latest.pt in --logdir'}): "
                             "--logdir (and --resume) must be on storage that every node sees")
    if resume is not None and not found:
        raise SystemExit(f"--resume {resume}: no such file")


def _play(args, info):
    if not args.checkpoint:
        raise SystemExit("--play needs --checkpoint")
    payload = load_checkpoint(args.checkpoint)
    cfg = _load_cfg(args, fallback=payload.get("config"))
    saved_run = payload.get("run") or {}
    run = RunOptions(seed=args.seed, no_diffusion=bool(saved_run.get("no_diffusion", False)),
                     no_rssm_finetuning=bool(saved_run.get("no_rssm_finetuning", False)), play=True)
    trainer = DMOTrainer(cfg, run, info)
    loss, discounted_loss, length = trainer.play(args.checkpoint)
    if info.rank == 0:
        print(f"mean episode return = {-loss:.4f}, mean discounted return = {-discounted_loss:.4f}, "
              f"mean episode length = {length:.1f}")
    trainer.close()
    return loss, discounted_loss, length


if __name__ == '__main__':
    main()
