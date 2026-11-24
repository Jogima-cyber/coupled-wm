# Coupled Local and Global World Models for Efficient First Order RL

Code for the CoRL 2026 paper *Coupled Local and Global World Models for Efficient First Order RL*,
for its G1 humanoid experiment (ego-centric grasp and lift): training the global world model,
pretraining the local world model, the world-model environment, and the DMO policy optimisation.
The paper's Push-T and Go2 experiments are not part of this release.

[Paper](https://arxiv.org/abs/2602.06219) · [Project page](https://coupled-global-local-wm-rl.pages.dev/)

## Method

Policies are trained entirely inside world models learned from offline robot data, with
first-order (analytic) policy gradients. Two world models are coupled. The **global** world model is
a large DreamerV4-style model: a video tokenizer and a dynamics transformer with a reward head,
trained with DreamerV4's shortcut forcing objective on play data. It is accurate in pixel space but
too expensive to differentiate through. The **local** world model is a small DreamerV3-style RSSM:
cheap to differentiate, but less accurate. Following DMO (decoupled forward-backward model-based
policy optimisation), the global model generates every rollout, frame by frame. At each step the
actor reads the RSSM posterior of the generated frame. The backward pass differentiates the RSSM's
one-step imagined prior and reward head at that point instead of the global model. The RSSM is
pretrained on the same data and finetuned online on the global model's rollouts. The policy is
trained with the SAPO variant of the SHAC actor-critic: 16-step analytic-gradient unrolls, a
maximum-entropy objective with a learned temperature, and an ensemble of TD(λ) critics.

## What is released

| component | entry point | config |
|---|---|---|
| global world model: tokenizer training | `coupled_wm.world_model.train_tokenizer` | `configs/world_model/g1_tokenizer.yaml` |
| global world model: dynamics training (backbone, then reward head) | `coupled_wm.world_model.train_dynamics` | `configs/world_model/g1_dynamics_{backbone,reward}.yaml` |
| local world model: RSSM pretraining | `coupled_wm.rssm.pretrain` | `configs/rssm/g1_rssm_pretrain.yaml` |
| the G1 world-model environment | `coupled_wm.envs.DreamerV4VecEnvG1Relative` | `env` block of `configs/rl/g1_dmo.yaml` |
| DMO training and evaluation | `coupled_wm.rl.train`, `coupled_wm.rl.evaluate` | `configs/rl/g1_dmo.yaml` |

Pretrained weights are on the Hugging Face Hub at [`Jogima-cyber/coupled-global-local-wm-rl`](https://huggingface.co/Jogima-cyber/coupled-global-local-wm-rl), as pretrained dirs
(`config.json` + fp32 `model.safetensors`, no optimizer state):

| dir | content | parameters | size |
|---|---|---|---|
| `tokenizer/` | the paper's G1 tokenizer | 757M | 3.0 GB |
| `dynamics/` | the paper's G1 dynamics model, with reward head | 682M | 2.7 GB |
| `rssm/` | the RSSM warm start for RL, pretrained with `configs/rssm/g1_rssm_pretrain.yaml` | 172M | 0.7 GB |

The G1 data is on the Hugging Face Hub as the dataset
[`Jogima-cyber/coupled-global-local-wm-rl`](https://huggingface.co/datasets/Jogima-cyber/coupled-global-local-wm-rl):
300 HDF5 files, 153 GB, in the layout the configs read under `G1_DATA_ROOT`. The environment seeds
every episode from 32 real frames, so RL needs it too, but only its RL seed pool (207 files,
106 GB). The [G1 dataset section](docs/data_format.md#the-g1-dataset) describes the data, how
each stage uses it and how to download part of it.

## Installation

Requirements: Linux, Python 3.10, PyTorch 2.9.1 with CUDA 12.8 (the environment of the paper's
runs), and NVIDIA GPUs with bf16 support. The paper used H100 80 GB GPUs. Attention uses the
FlashAttention or memory-efficient SDPA kernels, falling back to the math kernel where neither
applies.

```bash
git clone https://github.com/Jogima-cyber/coupled-wm.git && cd coupled-wm
conda create -n coupled-wm python=3.10 && conda activate coupled-wm   # or a venv
pip install -r requirements.txt     # pinned versions, CUDA 12.8 wheels
pip install -e .                    # ".[test]" for the tests, ".[wandb]" for wandb logging
```

`torch.compile`, which every config uses as the paper did, also needs a C compiler (`gcc` or
`clang`, or `$CC`) and the Python headers at run time. Without them, each stage runs eagerly with
the overrides listed in [docs/training.md](docs/training.md#conventions).

Launch with `torchrun` (single or multiple nodes), with one `srun` task per GPU under SLURM, or as
a single process; nothing depends on a particular cluster. Inside a Singularity/Apptainer
container, run the same commands in the container (`apptainer exec --nv ...`) with the data,
weights and run directories bind-mounted on storage that every node sees.

## Quickstart

Train a policy in the released world models:

```bash
python scripts/download_weights.py --repo-id Jogima-cyber/coupled-global-local-wm-rl --out weights
hf download Jogima-cyber/coupled-global-local-wm-rl --repo-type dataset --local-dir data/g1
export WEIGHTS_DIR=$PWD/weights G1_DATA_ROOT=$PWD/data/g1

torchrun --nproc_per_node=8 -m coupled_wm.rl.train --config configs/rl/g1_dmo.yaml \
    --seed 20 --logdir runs/g1_dmo/s20 --no-eval
python -m coupled_wm.rl.evaluate --checkpoint runs/g1_dmo/s20/best_policy.pt \
    --episodes 64 --num-envs 16 --output-dir runs/g1_dmo/s20/eval
```

Train every model yourself: tokenizer, dynamics backbone, dynamics reward stage, RSSM, then RL.

```bash
export G1_DATA_ROOT=$PWD/data/g1 RUN_ROOT=$PWD/runs
python -m coupled_wm.data.validate $G1_DATA_ROOT/new_wm_h5_segmented_small/world_model \
    --image-size 256 --proprio-dim 50 --action-dim 22 --window 32 64 128

TOK=$RUN_ROOT/g1_tokenizer/pretrained
torchrun --nproc_per_node=8 -m coupled_wm.world_model.train_tokenizer \
    --config configs/world_model/g1_tokenizer.yaml
torchrun --nproc_per_node=8 -m coupled_wm.world_model.train_dynamics \
    --config configs/world_model/g1_dynamics_backbone.yaml --set tokenizer.path=$TOK
torchrun --nproc_per_node=8 -m coupled_wm.world_model.train_dynamics \
    --config configs/world_model/g1_dynamics_reward.yaml --set tokenizer.path=$TOK
python -m coupled_wm.rssm.pretrain --config configs/rssm/g1_rssm_pretrain.yaml

torchrun --nproc_per_node=8 -m coupled_wm.rl.train --config configs/rl/g1_dmo.yaml \
    --seed 20 --logdir $RUN_ROOT/g1_dmo/s20 --no-eval \
    --set env.tokenizer_ckpt=$TOK \
    --set env.dynamics_ckpt=$RUN_ROOT/g1_dynamics_reward/pretrained \
    --set rssm.pretrained=$RUN_ROOT/g1_rssm/rssm_e10
```

Every stage writes a resumable `latest.pt` and continues from it when the same command is run
again, so jobs cut by a time limit are resumed by relaunching them. A SIGTERM or SIGUSR1 sent to
the training processes makes them save and exit.

Check a newly trained world model before RL: replay recorded commands in it and compare its
generations with the recorded frames, here next to the released model.

```bash
python scripts/compare_rollouts.py $G1_DATA_ROOT/new_wm_h5_segmented_small/world_model \
    --exclude 'patch*' --min-chunk-index 21 \
    --tokenizer $TOK --dynamics $RUN_ROOT/g1_dynamics_reward/pretrained \
    --dynamics-b $WEIGHTS_DIR/dynamics --tokenizer-b $WEIGHTS_DIR/tokenizer \
    --num 8 --out $RUN_ROOT/compare_rollouts
```

For each window it writes a video (ground truth | your model | released model, with the data's
reward label and each model's reward prediction per frame), a contact sheet and the per-step
rewards as JSON ([docs/training.md](docs/training.md#6-checking-a-world-model)). The models
generate exactly as in the RL environment, on one GPU.

- [docs/training.md](docs/training.md): every entry point, its outputs, resuming and stopping,
  evaluation, and adapting the configs to a new robot.
- [docs/world_model_recipe.md](docs/world_model_recipe.md): what each stage of the paper's G1
  experiment ran, and how the configs reproduce it.
- [docs/data_format.md](docs/data_format.md): the HDF5 format every stage reads, and the G1 data.

## Compute and memory

The paper's G1 experiment:

| stage | GPUs | wall-clock | parallelism |
|---|---|---|---|
| tokenizer | 64 H100 | about 25 h (45,000 updates) | FSDP |
| dynamics backbone | 64 H100 | about 13 h (49,000 updates) | DDP |
| dynamics reward stage | 64 H100 | about 1 h (3,000 updates) | DDP |
| RSSM pretraining | 1 H100 | about 11 h (10 epochs) | single GPU |
| DMO RL | 8 H100 (2 nodes x 4) | about 9.5 h per seed (2,000 iterations) | DDP |

**Global batches.** Batch sizes in the configs are global. The world-model trainers make one
optimizer update from `train.global_batch` clips (256) and run `global_batch /
(micro_batch_per_gpu x GPUs)` gradient-accumulation steps per GPU, which must be an integer (with
one clip per GPU: any power of two up to 256 GPUs). The schedule is counted in updates, the loss
normalisers average over a fixed number of clips, and a quarter of every update's micro-batches
are shortcut micro-batches, so the recipe is the same on any number of GPUs; on `N` GPUs an
update takes roughly `64 / N` times as long as in the paper. RSSM pretraining splits its 16
sequences per step over the processes. In RL, `train.num_actors` (128), `train.replay_size`,
`train.min_replay`, `rssm.batch_size` (256) and `eval.num_episodes` are global and divided over
the GPUs; DDP averages the actor, critic and RSSM gradients over all of them.

**RL memory per GPU.** Each GPU holds the world model (2.9 GB of bf16 weights), a KV cache of
0.55 GB per environment, its share of the replay buffer (49,246 bytes per row), the RSSM with its
optimizer state (about 2.75 GB) and the RSSM finetuning batch. Every environment reset also runs
the tokenizer encoder once on the 32 context frames of all the GPU's environments. With
`env.compile_encoder: reduce-overhead`, `torch.compile` records that encoder call as a CUDA graph,
and PyTorch keeps the memory the graph works in reserved for the rest of the run (its CUDA-graph
memory pool), so that later resets can rerun the graph at the same addresses. At 64 environments
on an H100 that memory is 19.7 GiB.

| GPUs | environments per GPU | KV caches | replay buffer share |
|---|---|---|---|
| 1 | 128 | 70 GB | 45.9 GiB |
| 2 | 64 | 35 GB | 22.9 GiB |
| 4 | 32 | 17.5 GB | 11.5 GiB |
| 8 | 16 | 8.7 GB | 5.7 GiB |

One 80 GB GPU cannot hold the G1 config. With fewer than 8 GPUs, `--set train.replay_device=cpu`
moves the replay buffer to host memory, and `--set env.compile_encoder=false` runs the encoder
without `torch.compile`, so its memory is freed after each reset.
RSSM pretraining keeps the resized frames in a RAM cache of about 30 GB of host memory per process
(`data.cache: false` streams them from disk instead).

## Repository layout

```
configs/
  world_model/   tokenizer, dynamics backbone, dynamics reward stage (G1)
  rssm/          RSSM pretraining (G1)
  rl/            DMO on the G1 world model
coupled_wm/
  common/        launching (torchrun, srun, single process), configs, checkpoints, logging
  data/          the HDF5 window dataset every stage reads; the format validator
  world_model/   DreamerV4 tokenizer and dynamics: models, trainers, loading, export
  rssm/          the DreamerV3-style RSSM and its offline pretraining
  envs/          the G1 world-model environment
  rl/            DMO trainer, actor, critics, evaluation
scripts/
  download_weights.py          released weights from the Hugging Face Hub
  compute_action_envelope.py   command clamp of a new dataset, for the configs
  compare_rollouts.py          ground truth vs world-model generations of the recorded commands
docs/            data format, training recipe, running the stages
tests/           CPU tests; tests/gpu/ GPU smoke tests
```

## Tests

```bash
pip install -e ".[test]"
pytest tests -q
```

The CPU suite (about 7 minutes) uses tiny models and synthetic data. The GPU smoke tests in
[tests/gpu/](tests/gpu/README.md) run every trainer on real data with the released weights, on 1
to 4 GPUs. With the paper's G1 data, `PAPER_POOL=1` adds checks of the exact pool sizes.

## License

Apache License 2.0, see [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{amigo2026coupledlocalglobalworld,
  title         = {Coupled Local and Global World Models for Efficient First Order {RL}},
  author        = {Joseph Amigo and Rooholla Khorrambakht and Nicolas Mansard and Ludovic Righetti},
  booktitle     = {Conference on Robot Learning (CoRL)},
  year          = {2026},
  eprint        = {2602.06219},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```
