# GPU tests

These scripts need GPUs, the released weights and data in the documented format
([docs/data_format.md](../../docs/data_format.md)), so `pytest tests` does not collect them
(`norecursedirs` in `pyproject.toml`). Run them from the repository root. Every path is passed
on the command line or through an environment variable; nothing is hardcoded.

Each script exits non-zero if a check fails. Default output dirs are under `$TMPDIR`; pass an
explicit output dir on a scratch file system for the larger ones.

## Inputs

| variable | what |
|---|---|
| `PYTHON` | a Python with `requirements.txt` and the package's `test` extra, default `python` |
| `WEIGHTS_DIR` | the released weights: `tokenizer/`, `dynamics/` and `rssm/` pretrained dirs (`python scripts/download_weights.py --repo-id Jogima-cyber/coupled-global-local-wm-rl --out weights`) |
| `G1_DATA_ROOT` | the dir holding the G1 data dirs the configs read: `new_wm_h5_segmented_small/world_model/` and, for RSSM pretraining, `new_wm_h5_demos_segmented_small/world_model/` |
| `OUT` | a scratch dir for test outputs |

Install: `pip install -r requirements.txt && pip install -e .[test]`. With the paper's G1 data,
`PAPER_POOL=1` makes the RSSM pretraining and DMO smoke tests also check the exact sizes of the
paper's pools.

## 0. CPU suite

```bash
$PYTHON -m pytest tests -q
```

## 1. RSSM on GPU: 1 GPU, about 15 min

```bash
$PYTHON tests/gpu/rssm_gpu_smoke.py --rssm $WEIGHTS_DIR/rssm \
    --data $G1_DATA_ROOT/new_wm_h5_segmented_small/world_model/<one file>.h5
```

The RSSM compiled exactly as the DMO trainer and as pretraining compile it: CUDA-graph replays of
the training step on a fresh compile (warmup, recording and first replay, as a training run's
first steps) and of the `no_grad` validation forward; per module, compiled vs eager within twice
eager's own bf16 error; the unroll with sampled posteriors compared over 8 posterior seeds (flipped
samples printed); the straight-through flag on the DMO swap and on `--no-diffusion` imagination;
the pretraining step against the paper pretraining script's step (bitwise, with deterministic
CUDA kernels). Without `--data` the inputs are noise frames; without `--rssm` the RSSM is randomly
initialised.

## 2. Tokenizer trainer: 1 node, 4 GPUs (torchrun inside)

```bash
G1_DATA_ROOT=$G1_DATA_ROOT bash tests/gpu/smoke_tokenizer.sh 4 $OUT/smoke_tokenizer
```

From scratch on 4 GPUs, resume on 2 (optimizer state resharded), SIGUSR1 stop and resume (the
log shows the loss normaliser decay `0.99 ** (NPROC / 64)` per micro-step), export and decode of a
G1 clip. `SMOKE_COMPILE=false` skips `torch.compile`; `SMOKE_SET="..."` adds overrides (e.g. a
small model).

## 3. Dynamics trainer: 1 node, 4 GPUs (torchrun inside)

```bash
WEIGHTS_DIR=$WEIGHTS_DIR G1_DATA_ROOT=$G1_DATA_ROOT bash tests/gpu/smoke_dynamics.sh $OUT/smoke_dynamics
```

Backbone with 128-frame updates, resume, one micro-step per update (a quarter of the ranks trains
the shortcut objective while the others train flow, in the same update), SIGTERM checkpoint,
reward stage from `latest.pt` and from `pretrained/`, export and reload with the tokenizer
pairing. Several minutes per launch with the configs' compile modes; `COMPILE=0` and `TINY=1` give
a fast run.

## 4. RSSM pretraining: 1 GPU (2 for the DDP leg), about 40 GB RAM, 15-20 min

```bash
G1_DATA_ROOT=$G1_DATA_ROOT bash tests/gpu/smoke_rssm_pretrain.sh $OUT/smoke_rssm_pretrain
```

The full pool of `configs/rssm/g1_rssm_pretrain.yaml` with the shipped settings, resume, per-epoch
pretrained dirs that strict-load into the RL config's RSSM, held-out validation with the compiled
model (at least two validations, so that recorded CUDA graphs are replayed), SIGTERM save and
resume, and (with 2 GPUs) `torchrun --nproc_per_node=2`. The training and validation logs must
contain no CUDA-graph warning. Prints frames/s (paper run: about 9,970 on one H100).

## 5. DMO trainer: 1 node, 4 GPUs (torchrun inside)

```bash
WEIGHTS_DIR=$WEIGHTS_DIR G1_DATA_ROOT=$G1_DATA_ROOT bash tests/gpu/smoke_rl.sh $OUT/smoke_rl
```

Training on the real environment with `torch.compile` (the `skip_nonzero` seed-window exclusion
count is printed), `--play` on one and on all GPUs, resume (the replay buffer is refilled with the
current policy, the `init_udt` warm-up is not rerun), standalone evaluation, in-training
evaluation on all ranks, the two ablations, and the straight-through flag in DMO and under
`--no-diffusion`. The RSSM warm start is `$WEIGHTS_DIR/rssm` (`RSSM_WARMSTART` overrides it).
`NPROC` sets the GPU count; `FULL_SIZE=1` adds 2 epochs at the release sizes (128 actors,
999,936-row replay buffer) and prints rank 0's peak GPU memory.
