# Running the stages

How to run every entry point, what each stage writes and what the next stage reads. The recipe
behind the G1 configs is in [world_model_recipe.md](world_model_recipe.md); the data contract in
[data_format.md](data_format.md). Paths below assume the default `RUN_ROOT=runs`.

| stage | entry point | writes | read by |
|---|---|---|---|
| tokenizer | `coupled_wm.world_model.train_tokenizer` | `runs/g1_tokenizer/pretrained/` | dynamics (`tokenizer.path`), RL (`env.tokenizer_ckpt`) |
| dynamics backbone | `coupled_wm.world_model.train_dynamics` | `runs/g1_dynamics_backbone/latest.pt` | reward stage (`init_from`) |
| dynamics reward stage | `coupled_wm.world_model.train_dynamics` | `runs/g1_dynamics_reward/pretrained/` | RL (`env.dynamics_ckpt`) |
| world-model check | `scripts/compare_rollouts.py` | videos, contact sheets, per-step rewards | you |
| RSSM pretraining | `coupled_wm.rssm.pretrain` | `runs/g1_rssm/rssm_e10/` | RL (`rssm.pretrained`) |
| DMO RL | `coupled_wm.rl.train` | `<logdir>/best_policy.pt`, `final_policy.pt` | `coupled_wm.rl.evaluate` |

A *pretrained dir* holds `config.json` (architecture, preprocessing and, for the dynamics model,
the generation and action constants the environment reads) and `model.safetensors` (fp32 weights,
no optimizer state). The released weights have the same format.

## Conventions

**Paths.** Run every command from the repository root: relative paths in the configs are relative
to the working directory. The configs read three environment variables:

| variable | default | meaning |
|---|---|---|
| `G1_DATA_ROOT` | `data/g1` | the directory holding `new_wm_h5_segmented_small/` and `new_wm_h5_demos_segmented_small/` |
| `WEIGHTS_DIR` | `weights` | the directory holding the pretrained dirs `tokenizer/`, `dynamics/` and `rssm/` |
| `RUN_ROOT` | `runs` | where the run directories are created |

Config values may contain `${VAR}` (an error when `VAR` is unset) or `${VAR:-default}` (the
default when `VAR` is unset or empty).

**Overrides.** Every entry point takes `--config <yaml>` and any number of `--set key=value`
overrides. The value is parsed as YAML (`--set optim.lr=3e-4`, `--set data.dirs=[a,b]`,
`--set init_from=null`), integer path segments index lists (`--set env.action_scale.3=0.5`), and
a key that does not exist raises; prefix it with `+` to add a new key on purpose
(`--set '+export.metadata.action={...}'`). Each run directory receives the resolved config of
its last launch as `config.yaml`.

**Launching.** The trainers detect their launcher:

```bash
# one node, N GPUs
torchrun --nproc_per_node=N -m <module> --config ...
# several nodes (run on every node)
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=<first node>:29500 \
    -m <module> --config ...
# SLURM: one srun task per GPU (rank, world size and master address come from SLURM)
srun --ntasks-per-node=4 --gpus-per-task=1 python -m <module> --config ...
# one process (one GPU, or CPU for small tests)
python -m <module> --config ...
```

Every rank must see the same run directory, checkpoint and data files: put them on storage
shared by all nodes. Keep `TMPDIR` and the PyTorch compile caches (`TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR`, by default under `TMPDIR`) node-local, e.g. under `/tmp`: a cache shared by
several nodes on a network file system makes ranks race on compiled kernels (`CUDA driver error:
file not found`). Inside a container (Singularity/Apptainer, Docker) the commands are the
same, run inside the container with GPU access (`apptainer exec --nv ...`) and the data,
weights and run directories bind-mounted.

**Logs.** TensorBoard events go to `<run_dir>/tb/` (`tensorboard --logdir runs`). `wandb.enabled:
true` mirrors them to Weights & Biases (offline mode by default, files in `<run_dir>/wandb/`;
`pip install -e ".[wandb]"`). On resume, TensorBoard events logged after the checkpoint are purged, so
the curves continue without duplicates (RL: on the environment-step axis).

**Stopping and resuming.** Every stage writes a resumable `latest.pt` periodically (atomically:
an interrupted save never corrupts it), and relaunching the same command continues from it. SIGTERM
or SIGUSR1 received by the training processes makes them save at the next safe point and exit: with
status 75 for the tokenizer, dynamics and RL trainers (so `backbone && reward` chains stop there),
with status 0 for RSSM pretraining. A finished run exits 0; relaunching it only writes outputs
that are missing (e.g. `pretrained/`). Under `torchrun`, signal the worker processes, not the
launcher, and early: torchrun does not forward SIGUSR1, and after forwarding SIGTERM it kills the
workers 30 s later, which a world-model save (about 9 GB, gathered on rank 0) may not fit in. With
one `srun` task per GPU, `sbatch --signal=USR1@600` reaches the trainers directly. DataLoader
workers ignore both signals.

**torch.compile.** The configs compile the models as the paper did. Triton builds a small C module
at the first call, so the machine needs a C compiler (`gcc` or `clang` on `PATH`, or `$CC`) and the
Python headers. A warning names the setting when they are missing. To run eagerly instead:

| stage | eager overrides |
|---|---|
| tokenizer | `--set compile=false` |
| dynamics | `--set compile=null --set tokenizer.compile=null` |
| RSSM pretraining | `--set train.compile=false` |
| DMO RL | `--set train.compile=false --set env.compile_encoder=false` |

## 0. Released weights and data

```bash
export G1_DATA_ROOT=${G1_DATA_ROOT:-$PWD/data/g1}
python scripts/download_weights.py --repo-id Jogima-cyber/coupled-global-local-wm-rl --out weights
hf download Jogima-cyber/coupled-global-local-wm-rl --repo-type dataset --local-dir $G1_DATA_ROOT
```

`download_weights.py` downloads `tokenizer/`, `dynamics/` and `rssm/` (about 6.5 GB;
`--components` selects some), checks that each is a complete pretrained dir and that the
dynamics model was trained on this tokenizer (sha256 recorded in its `config.json`).
`hf download` downloads the G1 data (153 GB) into `G1_DATA_ROOT`;
[data_format.md](data_format.md#the-g1-dataset) shows how to download only the 106 GB that RL
reads.

## 1. Check the data

```bash
python -m coupled_wm.data.validate $G1_DATA_ROOT/new_wm_h5_segmented_small/world_model \
    --image-size 256 --proprio-dim 50 --action-dim 22 --color-order bgr --window 32 64 128
```

See [data_format.md](data_format.md#validating-a-dataset).

## 2. Tokenizer

```bash
python -m coupled_wm.world_model.train_tokenizer --download-lpips   # once, where there is internet access
torchrun --nproc_per_node=4 -m coupled_wm.world_model.train_tokenizer \
    --config configs/world_model/g1_tokenizer.yaml
```

The LPIPS loss uses torchvision's ImageNet VGG16 weights. Rank 0 downloads them into `$TORCH_HOME`
on first use; on compute nodes without internet access, run `--download-lpips` once beforehand with
the same `$TORCH_HOME`.

The tokenizer trains with FSDP. It runs on any number of GPUs for which
`train.micro_batch_per_gpu x GPUs` divides `train.global_batch` (256): 1, 2, 4, ..., 256 GPUs with
one clip per GPU.

Run directory (`runs/g1_tokenizer/`):

| path | content |
|---|---|
| `latest.pt` | resumable state: weights, optimizer, data position, loss normalisers, RNG, configs; written every 1,000 updates, after 60 minutes without a save, at epoch ends and on a stop signal; never with non-finite weights (the save fails instead) |
| `latest.prev.pt` | the `latest.pt` before the last save (a hard link: no extra space until the next save) |
| `archive/step_<N>/` | weights-only pretrained dirs every 2,000 updates (the last 5 are kept) |
| `pretrained/` | the final pretrained dir, written when `schedule.max_updates` is reached |
| `config.yaml`, `tb/` | resolved config, logs |

Resuming: a launch continues from `<run_dir>/latest.pt` when it exists, on any valid number of
GPUs (the optimizer state is resharded); data order, learning rate and loss normalisers continue
exactly. Otherwise it starts from `resume:` (the `latest.pt` of another run), else from the weights
of `init_from:` (a tokenizer pretrained dir or checkpoint; `init_allow_missing` lists parameter
globs left at their fresh initialisation), else from scratch. A resume on changed data (files,
frame counts, window, stride or colour order) is refused unless `data.allow_pool_change: true`;
train on new data in a new `run_dir`.

## 3. Dynamics backbone

```bash
torchrun --nproc_per_node=4 -m coupled_wm.world_model.train_dynamics \
    --config configs/world_model/g1_dynamics_backbone.yaml \
    --set tokenizer.path=runs/g1_tokenizer/pretrained
```

Without the `--set`, the frozen tokenizer is the released one (`$WEIGHTS_DIR/tokenizer`).
`tokenizer.path` takes a pretrained dir or a tokenizer `latest.pt`. The dynamics model trains with
DDP on any number of GPUs for which `train.micro_batch_per_gpu x GPUs` divides 256.

Run directory (`runs/g1_dynamics_backbone/`): `latest.pt` (every 1,000 updates, every 30 minutes,
on a stop signal and at the end), `pretrained/` (at the end; `checkpoint.export_on_save: true`
also writes it at every save), `config.yaml`, `tb/`. A resume refuses a changed file pool
(`data.allow_pool_change: true` continues on the new pool from the start of the current pass).

## 4. Dynamics reward stage

```bash
torchrun --nproc_per_node=4 -m coupled_wm.world_model.train_dynamics \
    --config configs/world_model/g1_dynamics_reward.yaml \
    --set tokenizer.path=runs/g1_tokenizer/pretrained
```

It starts from `init_from: runs/g1_dynamics_backbone/latest.pt` with a fresh reward head, agent
token, optimizer and schedule. It refuses a backbone that has not reached its own
`schedule.max_updates` (`init_from_allow_unfinished: true` waives this) and a backbone trained with
another tokenizer. Once started, a resume refuses an `init_from` source that has changed since
(`--set init_from=null` resumes regardless). To run both stages in a row:

```bash
torchrun ... --config configs/world_model/g1_dynamics_backbone.yaml && \
torchrun ... --config configs/world_model/g1_dynamics_reward.yaml
```

`runs/g1_dynamics_reward/pretrained/` is the environment's dynamics model. With
`export.preset: g1` its `config.json` holds the G1 generation constants and action layout.

## 5. Exporting a checkpoint

The trainers export `pretrained/` themselves. To export any training checkpoint:

```bash
python -m coupled_wm.world_model.export --ckpt runs/g1_tokenizer/latest.pt --out my_weights/tokenizer
python -m coupled_wm.world_model.export --ckpt runs/g1_dynamics_reward/latest.pt --out my_weights/dynamics \
    --tokenizer my_weights/tokenizer [--metadata my_robot.yaml]
```

`--tokenizer` records the tokenizer's sha256 for the pairing check; `--metadata` merges `config.json`
blocks (e.g. `action`, `inference`) into the export. The weights are strict-loaded and checked to be
finite before they are written.

## 6. Checking a world model

Before RL, compare the world model's generations with the recorded data:

```bash
python scripts/compare_rollouts.py $G1_DATA_ROOT/new_wm_h5_segmented_small/world_model \
    --exclude 'patch*' --min-chunk-index 21 \
    --tokenizer runs/g1_tokenizer/pretrained --dynamics runs/g1_dynamics_reward/pretrained \
    --num 8 --seed 0 --horizon 64 --out runs/compare_rollouts
```

The script draws `--num` windows of `C + horizon` consecutive recorded frames from the files
selected as in a `data` block (`--include`, `--exclude`, `--min-chunk-index`, `--file-list`; the
example selects the RL seed pool), where `C` is the dynamics' `init_context_len` (32 for G1). It
primes the world model with the first `C` frames and commands of each window, then generates the
next `--horizon` frames one at a time, each conditioned on the absolute command recorded for that
frame. The generation is the RL environment's own (`reset_from_context` and `step_command` of
`DreamerV4VecEnvG1Relative`: context prefill, Euler steps, cache-writing pass, KV-cache length,
reward head, bf16 autocast); the recorded commands are replayed without the environment's command
clamp. It runs on one GPU, one world model at a time, `--batch-size` windows in parallel (16 by
default; for G1, 2.9 GB of weights plus 0.55 GB of KV cache per window).

Outputs, per window `<k>_<file>_<start>` (`start` = the first context frame):

| file | content |
|---|---|
| `.mp4` | ground truth \| model A [\| model B] at the tokenizer's resolution, at the dynamics' `control_hz`; each frame shows the step, the frame index in the file, the data's reward label and each model's reward prediction (expectation of the reward head the environment uses) |
| `_sheet.png` | the first, every 8th (`--sheet-every`) and the last generated frame, one row per column |
| `.json` | frame indices, per-step reward labels and predictions, and their sums |

and `index.json`: the settings, each model's generation constants and every window. MP4 needs
`imageio-ffmpeg` (`pip install -e ".[video]"`, which bundles an ffmpeg binary) or PyAV; without
either the videos are GIFs (`--format` forces one).

| option | effect |
|---|---|
| `--tokenizer`, `--dynamics` | pretrained dirs or training checkpoints (`latest.pt`), so a run can be checked while it trains |
| `--dynamics-b` [`--tokenizer-b`] | a second world model as a third column, on the same windows with the same generation noise (e.g. a new model next to the released one) |
| `--seed`, `--num` | the window draw and the generation noise; the first `n` windows of a draw do not depend on `--num` |
| `--windows FILE:START ...` | these windows instead of random ones |
| `--horizon` | generated frames per window (default 64, the RL episode length) |
| `--context-frames N` | show the last `N` context frames before the generated ones |
| `--env-arg KEY=VALUE` | a generation constant (`init_context_len`, `kv_cache_len`, `num_inference_steps`, `seq_cor_tau_idx`, `reward_mtp_index`) instead of the dynamics `config.json` value |
| `--reward-key` | the h5 dataset of the reward labels (default `rewards`; omitted when absent) |

Things to check by eye: whether the generated robot follows the recorded motion, whether objects
keep their shape and position until the end of the horizon, and whether the reward prediction
rises and falls with the label.

## 7. RSSM pretraining

```bash
python -m coupled_wm.rssm.pretrain --config configs/rssm/g1_rssm_pretrain.yaml
```

One GPU, as in the paper: about 1 h per epoch on an H100, 10 epochs.
`torchrun --nproc_per_node=N` runs data parallel with the same global batch and data order. The
frames resized to 64 x 64 are held in a RAM cache built at start: about 30 GB of host memory per
process in float32 (the exact frames), 7.6 GB with `--set data.cache_dtype=uint8` (re-quantised,
error at most 0.5/255), or none with `--set data.cache=false`, which streams the same frames as the
float32 cache from disk at about half the speed. Under torchrun every process builds its own cache;
the run stops with an error before building it when the node lacks the memory.

Run directory (`runs/g1_rssm/`): `rssm_e<N>/` (a pretrained dir after every epoch; `rssm_e10/` is
the RL warm start), `latest.pt` (every 5,000 steps, at epoch ends and on a stop signal),
`config.yaml`, `tb/`. A resume must use the same model, loss, optimizer, data selection and
preprocessing; it may change the cache settings only to ones that give the same frame values.
`--set validation.holdout_every=20` holds out every 20th file for a validation loss (off by
default; the paper trained on the whole pool). Only rank 0 validates; with `N` processes it reads
each held-out batch as `N` batches of the per-process training batch, so that its compiled model
sees the training's batch shape.

## 8. DMO RL

```bash
torchrun --nproc_per_node=8 -m coupled_wm.rl.train --config configs/rl/g1_dmo.yaml \
    --seed 20 --logdir runs/g1_dmo/s20 --no-eval
```

With your own models instead of the released ones:

```bash
    --set env.tokenizer_ckpt=runs/g1_tokenizer/pretrained \
    --set env.dynamics_ckpt=runs/g1_dynamics_reward/pretrained \
    --set rssm.pretrained=runs/g1_rssm/rssm_e10
```

The paper ran seeds 20, 30, 40 and 50, each on 2 nodes x 4 H100 with `--no-eval`. Rank `r` uses
seed `seed + r`. `train.num_actors`, `train.replay_size`, `train.min_replay`, `rssm.batch_size`
and `eval.num_episodes` are global and split over the ranks; `num_actors`, `replay_size` and
`rssm.batch_size` must be multiples of the number of GPUs. The README's
[compute section](../README.md#compute-and-memory) gives the memory per GPU.

Flags:

| flag | effect |
|---|---|
| `--logdir DIR` | run directory, on storage every node sees; relaunching with the same `--logdir` resumes |
| `--seed N` | seed of rank 0 |
| `--no-eval` | no deterministic evaluation during or after training (otherwise 1,280 episodes every 50 iterations) |
| `--no-diffusion` | ablation "No Diffusion": the actor's unroll runs entirely in the local model, forward and backward, as in Dreamer: the RSSM imagines the trajectory, the gradient goes back through that imagination, and the rewards are the RSSM reward head's. Each episode's first frame is the last frame of a real 32-frame seed window, encoded and decoded by the tokenizer and given to the RSSM. The RSSM then imagines all 64 steps of the episode (`train.max_episode_length`), in four actor updates of 16 steps (`train.steps_num`): the imagined latent carries over from one update to the next, with the gradient cut at each boundary. The global world model still steps with the same actions, but only for the critic's reward targets and the logged returns. The RSSM is not finetuned (alias `--dreamer-dreamer-dreamer`) |
| `--no-rssm-finetuning` | ablation "No RSSM Finetuning": the warm-start RSSM stays frozen |
| `--resume CKPT` | resume from another training checkpoint |
| `--play --checkpoint CKPT` | evaluate a policy instead of training, with the checkpoint's config (`eval.player.num_actors` must be a multiple of the number of GPUs) |
| `--exp-name NAME` | wandb run name |

Run directory: `config.yaml`, `tb/`, `latest.pt` (every 400 iterations, at the end and on a stop
signal), `best_policy.pt` (the policy with the best mean training-episode return so far),
`final_policy.pt`, and the per-episode return histories as `.npy` files.

Resuming restores the networks, optimizers, temperature, counters, RNG states and each
environment's position in its sequence of seed windows. The replay buffer is not in `latest.pt`: with
`train.checkpoint_replay: true` each rank also saves its replay buffer (`replay/rank<r>.pt`, up to
`replay_size / world_size` rows) and a resume reloads it. Otherwise (the default) a resumed run
refills `train.min_replay` replay buffer rows with the current policy, then RSSM finetuning continues with its
usual `rssm.udt` updates per iteration; the `rssm.init_udt` warm-up runs only once per run. A run
must be resumed with the same ablation flags.

TensorBoard tags worth watching: `rewards/iter` (mean training-episode return under the global
reward head), `local_rewards/iter` (the same episodes scored by the RSSM reward head),
`entropy/ent_mean/iter` against `entropy/target_ent/iter`, `entropy/alpha/iter`, `value_loss/iter`,
`env/clamp_rate/iter` (how often the command clamp binds) and, with evaluation on,
`eval_rewards/iter`.

**Evaluation.** The standalone evaluation runs the deterministic policy on the environment of the
policy's training config (same seed pool) and writes per-episode returns, a summary and image strips
of the generated episodes:

```bash
python -m coupled_wm.rl.evaluate --checkpoint runs/g1_dmo/s20/best_policy.pt \
    --episodes 64 --num-envs 16 --output-dir runs/g1_dmo/s20/eval
```

`--set` overrides the stored config (`--set env.loader_seed=0` fixes the seed windows).

## New robots and datasets

The G1 configs are templates. For other data, change at least:

- **All stages:** the `data` blocks (directories, file selection, `color_order`;
  [data_format.md](data_format.md)). The tokenizer config selects the paper's 147 G1 files with
  `include_globs`: use `["*.h5"]` (or your own selection) for new data.
- **Tokenizer:** `model.image_size`, `model.use_proprio`, `model.proprio_dim`, and the schedule
  (`warmup_updates`, `total_updates`, `max_updates`) for the size of your data.
- **Dynamics:** `model.action_dim` and the schedules. The dynamics keys `num_latents` and
  `latent_dim` must equal the tokenizer's `n_latents` and `bottleneck_dim` (checked when the two
  are paired). `export.preset: g1` writes the G1 environment constants and is for G1 data only: set
  `export.preset: null` and give the action layout as `export.metadata.action` (`dim`, `scale`,
  `grip_dims`, `grip_range`), printed by

  ```bash
  python scripts/compute_action_envelope.py data/my_robot --grip-dims 6 7   # or --grip-dims with no value
  ```

  and, if the generation constants differ from G1's, `export.metadata.inference`. For a model
  already trained, `python -m coupled_wm.world_model.export --metadata <yaml>` adds the blocks.
- **RSSM:** `model.action_size` (the command dimension) and `data.color_order` (the tokenizer's).
- **Before RL:** check the new world model with `scripts/compare_rollouts.py`
  ([section 6](#6-checking-a-world-model)) on your data, preferably on files it did not train on
  (`--file-list` or the globs). It reads the documented format and generates with the constants
  the environment takes from the dynamics `config.json`.
- **RL:** `env.data`, `env.drop_action_indices`, the env's action layout when the dynamics
  `config.json` does not record it (`env.action_scale`, `env.grip_dims`, `env.grip_range`),
  `actor.action_scale` (one per-step scale per controlled dimension) and `train.target_entropy`.
  The target entropy must be below the policy's maximum attainable entropy, which depends on
  `actor.action_scale` and the log-std range; the trainer logs both at start-up and refuses a
  target above the maximum.

The environment logs every constant with its source and warns when a G1 default fills a value the
dynamics `config.json` does not record, when seed-window commands lie outside the clamp, and when
the dynamics model would attend over longer temporal distances than it was trained on.
