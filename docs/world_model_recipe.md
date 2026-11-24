# Training recipe (G1)

This page describes what each training stage of the paper's G1 experiment ran, and how
the release config of that stage reproduces it. [training.md](training.md) has the commands;
[data_format.md](data_format.md#the-g1-dataset) describes the data each stage reads.

| stage | config | paper hardware | paper budget | release budget |
|---|---|---|---|---|
| 1. tokenizer | `configs/world_model/g1_tokenizer.yaml` | 64 H100 | 45,000 updates, initialised from a SOAR-pretrained tokenizer | `schedule.max_updates: 45000`, from scratch (the SOAR-pretrained tokenizer is not released) |
| 2. dynamics backbone | `configs/world_model/g1_dynamics_backbone.yaml` | 64 H100 | 49,000 updates | `schedule.max_updates: 49000` |
| 3. dynamics reward stage | `configs/world_model/g1_dynamics_reward.yaml` | 64 H100 | 3,000 updates | `schedule.max_updates: 3000` |
| 4. RSSM pretraining | `configs/rssm/g1_rssm_pretrain.yaml` | 1 H100 | 10 epochs | `train.epochs: 10` |
| 5. DMO RL | `configs/rl/g1_dmo.yaml` | 8 H100 per seed | 2,000 iterations | `train.max_epochs: 2000` |

Stages 1-3 build the global world model and stage 4 the local one; stage 4 does not depend on
stages 1-3. The released weights are the paper's tokenizer and dynamics checkpoints and an RSSM
warm start pretrained with stage 4's release config.

**Budgets.** Each world-model stage of the paper ran a long nominal cosine schedule and was stopped
by the job time limit, far from the end of its schedule. The release configs keep the nominal
schedules as explicit numbers (`schedule.warmup_updates`, `schedule.total_updates`) and stop at the
update where the paper run stopped (`schedule.max_updates`). A launch resumes from
`<run_dir>/latest.pt` automatically, so a stage interrupted by a time limit is continued by
relaunching the same command. The learning rate of update `u` depends only on `u`.

**Global batch.** The world-model configs give the global batch per optimizer update (256 clips).
Each GPU runs `global_batch / (micro_batch_per_gpu x GPUs)` accumulation steps, so the paper's 64
GPUs ran 4 steps of one clip each. The sequence of updates, the learning-rate schedule, the
flow/shortcut split of every update and the horizon of the loss normalisers do not depend on the
number of GPUs; fewer GPUs take proportionally longer per update.

## 1. Tokenizer

**Model** (757.3M parameters). A block-causal transformer encoder and decoder (16 layers each,
width 1024, 16 heads) over 16 x 16 patches of 256 x 256 RGB frames. The encoder compresses each
frame and its 50-d proprioception into 256 latent tokens of width 32, bounded by a tanh; every 4th
layer attends causally over time. In training, a fraction `U(0, 0.9)` of each frame's patches is
masked (MAE style) and the proprio token is masked with probability 0.5.

**Objective.** `MSE + 0.2 LPIPS + 0.01 MSE_proprio` on 64-frame clips: pixel MSE on `[0, 1]` frames,
LPIPS-VGG on frames scaled to `[-1, 1]`, MSE on the raw proprio vector. Each term is divided by its
running RMS, an exponential moving average of the squared loss with decay 0.99 per 64 clips. An
update whose raw loss exceeds 5 times its running mean (after 200 updates) is skipped. (This
implementation also skips any update with a non-finite loss or gradient norm.)

**Optimisation.** AdamW (learning rate 1e-4, betas (0.9, 0.999), weight decay 0.01 on all
parameters), gradients clipped to norm 1.0 separately for encoder and decoder, a linear warmup over
1,400 updates followed by a cosine decay to 1e-8 at update 345,360. Global batch 256 clips of 64
frames (64 GPUs x 1 clip x 4 accumulation steps). FSDP with bf16 parameters, gradient reduction and
buffers over fp32 master weights; `torch.compile` in `reduce-overhead` mode.

**Data.** `chunk_0000.h5` to `chunk_0146.h5`: 147 files, 381,186 frames, 368,389 windows of 64
frames (stride 1). The nominal schedule length is 240 epochs of this pool: 240 x 5,756 micro-steps
per GPU / 4 = 345,360 updates.

**The paper run** was initialised from a tokenizer pretrained on SOAR data (its proprio parameters
started from random initialisation), and its Adam moments were reset once, at update 8,634. The run
was stopped by the job time limit at update 45,000,
where the learning rate was 9.61e-5, 96% of its peak: the cosine decay had barely started. It ran at
about 2 s per update on 64 H100 (about 25 h for 45,000 updates). The released tokenizer is that
checkpoint.

**The release config** trains this recipe from scratch on the same 147 files and stops at update
45,000: the SOAR-pretrained tokenizer the paper run started from is not part of the release.
`init_from` can start from other weights, such as a tokenizer you pretrained yourself. The running-RMS
state is kept in fp32, and its decay is defined per 64 clips (`loss.rms_reference_windows`), so the
normalisers average over the same number of clips at any GPU count.

## 2. Dynamics backbone

**Model** (679.8M parameters without reward head). A block-causal transformer (24 layers, width
1536, 24 heads, 4 register tokens, temporal attention every 4th layer) over the 256 x 32 latents of
each frame. It predicts the clean latents of frame `t` from noisy latents, the action `commands[t]`
and the causal history.

**Objective.** DreamerV4 shortcut forcing. Every frame gets its own noise level on a grid of
128 levels and a step size `2^-s`. A micro-batch is either *flow* (finest step, target: the clean
latents) or *shortcut* (coarser step, target: two half steps of the model itself, with a
stop-gradient on the target; the loss backpropagates through the coarse-step prediction), with
loss weight `0.9 tau + 0.1`. One micro-batch in four is a shortcut micro-batch: at the paper's
4 accumulation steps, each GPU runs flow, flow, flow, shortcut in every update. Each term is divided
by its running RMS, as in the tokenizer.

**Clips.** 32-frame clips, except every 8th update, which uses 128-frame clips (the paper's "short
T=32 + occasional long T=128"). The temporal attention has a sliding window of 96 frames, which only
binds on the 128-frame clips. The frozen tokenizer encodes each clip whole, without a temporal
window, the 128-frame clips included (it was trained on 64-frame clips).

**Optimisation.** AdamW (learning rate 1e-4, betas (0.9, 0.999), weight decay 0.1 on all parameters),
gradient clipping at 1.0, bf16 autocast over fp32 weights, DDP. Global batch 256 clips (64 GPUs x
1 clip x 4 accumulation steps). The nominal schedule is 210 epochs of the 228-file pool, 9,122
micro-batches per GPU each: 210 x 9,122 / 4 = 478,905 updates in total, with a linear warmup over
5% of them (23,945 updates).

**Data.** The 228 `chunk_*.h5` files: 593,329 frames, 583,857 windows of 32 frames and 554,987 of
128. No reward is read in this stage.

**The paper run** trained from scratch and was stopped by job time limits at update 49,000 (epoch
23 of the nominal 210), with the learning rate at 9.93e-5, just past the warmup. This is where the
reward stage starts. An update took about 1 s on 64 H100 (about 13 h for 49,000 updates).

**The release config** reproduces it. Its schedule numbers give the learning rates stored in the
paper's checkpoints at updates 17,000, 33,000 and 49,000 exactly. At an accumulation count other
than 4, the shortcut micro-batches are interleaved (every 4th micro-batch over updates, GPUs and
micro-steps) so that every update stays 25% shortcut.

## 3. Dynamics reward stage

**Model** (681.6M parameters). The backbone plus a learned agent token and a reward head that
predicts, at frame `t`, the rewards of frames `t` to `t + 7` (multi-token prediction). Each
prediction is a two-hot distribution over 255 bins spaced uniformly in symlog space on `[-20, 20]`.
The environment uses the first of the 8 predictions as the step reward.

**Training.** From the backbone's update-49,000 checkpoint, with a freshly initialised agent token
and reward head, a fresh AdamW at peak learning rate 5e-5 and a fresh schedule: 210 epochs of
11,561 micro-batches per GPU, 606,952 updates in total, warmup 30,347. Only 32-frame clips. Same
objective as the backbone plus the reward cross-entropy (targets past the end of a clip are
masked).

**Data.** All 290 files (the 228 play chunks and the 62 `patch*` files): 752,861 frames, 739,956
windows of 32 frames, rewards `{0, 1}`. `data.start_epoch: 1` reproduces the paper run's data order.

**The paper run** was a single job stopped by its time limit after 3,000 updates (global update
52,000), still inside the warmup: the learning rate was 4.94e-6, a tenth of its peak. The released
dynamics model is that checkpoint.

**The release config** reproduces it and stops at 3,000 updates. Its `pretrained/` export
carries the G1 environment constants (generation loop, action layout, control rate) because
`export.preset: g1`; for other data, see [training.md](training.md#new-robots-and-datasets).

## 4. RSSM (local world model) pretraining

**Model** (172.1M parameters). A DreamerV3-style RSSM in its XL size: 4096-d GRU state, 32
categorical variables of 32 classes, dense layers of 1024 units, CNN encoder and decoder with
channel multiplier 96 on 64 x 64 frames, 5-layer MLPs, uniform mix 0.01, and a reward head with 255
two-hot bins in symlog space on `[-20, 20]`.

**Objective.** The DreamerV3 world-model loss: image reconstruction (sum of squares), two-hot reward
negative log-likelihood, and the KL-balanced loss with dynamic 0.5, representation 0.1 and 1 free
nat. Adam at learning rate 1e-4 (eps 1e-8), no weight decay, no gradient clipping, bf16 autocast,
batches of 16 sequences of 64 rows.

**Rows.** Relative actions: the row of frame `t` holds `(frame t, commands[t] - commands[t-1],
rewards[t])`, the convention of the environment and the actor's rollout. The rewards are the
dataset's labels, the same labels the global reward head is trained on.

**Data.** The 228 play chunks and the 10 demonstration files: 238 files, 619,289 frames. Frames are
read at 256 x 256 and resized to 64 x 64 on the fly with the environment's operation (bilinear, no
antialiasing). With
windows of 65 frames (64 rows plus the previous command) there are 585,607 windows, so one epoch is
36,600 steps.

**The paper run** trained 10 epochs on one H100 in about 10.6 h; that checkpoint was the warm
start of the paper's G1 RL runs.

**The release config** trains 10 epochs and writes a pretrained dir after each one; `rssm_e10/` is
the RL warm start. The released `rssm/` weights were produced with this config. With the float32 RAM
cache (about 30 GB of host memory) it runs at about 9,950 frames/s on one H100, about 1 h per epoch.

## 5. DMO RL

**Environment.** The global world model as a batched environment: each episode is seeded from 32
real frames of the RL seed pool (`chunk_0021` to `chunk_0227`, 207 files), lasts 64 steps, and all
environments reset together. One step generates one frame with 4 shortcut steps that attend to a
KV cache of the last 32 frames, then runs one more pass whose only job is to write the new frame
into the cache (noised to signal level 0.90, as the context frames are; its prediction is
discarded), decodes the frame, and returns the reward head's prediction for it. Observations are the decoded frames resized to 64 x 64.

**Agent.** The actor and critics read the RSSM latent of the current frame (1024 stochastic +
4096 deterministic dimensions). Every iteration unrolls 16 environment steps: the forward pass is the
global model's rollout, and the backward pass goes through the RSSM's imagined prior at each step
(DMO's gradient swap) and through the RSSM reward head. The settings of the paper's G1 runs are:

| | value |
|---|---|
| policy | 19 dims: 17 upper-body joints at 0.02 rad per step and 2 grips at 1.0 per step (`tanh` scaling); base velocity dims 0-2 not controlled |
| actor | MLP 256-128-64, SiLU, Gaussian with log std in `[-5, 2]`; AdamW, learning rate 2e-3 decaying linearly to 1e-5, betas (0.7, 0.95), gradient clipping 1.0 |
| critics | 10 MLPs 256-128-64; the actor bootstraps with their mean, TD(λ) targets use the minimum of 2 random members; AdamW 5e-4 decaying to 1e-5; 8 iterations of 2 minibatches per GPU and iteration |
| return | γ 0.99, λ 0.95, 16-step unrolls, 64-step episodes |
| entropy | target -58, temperature learned by Adam at 5e-3; the reward gets the bonus `α (H + 58) / 116` |
| actors | 128 (16 per GPU) |
| RSSM finetuning | Adam 1e-4, batches of 256 sequences of 64 rows (32 per GPU), 10,000 updates once the replay buffer is first filled, then 64 per iteration; replay buffer of 999,936 rows |
| length | 2,000 iterations, about 9.5 h on 8 H100 (2 nodes x 4) per seed; seeds 20, 30, 40, 50; no evaluation during training |

**The release config** keeps these settings with two values changed: the replay buffer fill before
training is 40,960 rows (320 steps, five 64-step episodes; paper: 40,000), and episodes are never
seeded from a window whose base velocity command is nonzero (`env.dropped_dims_reset: skip_nonzero`;
paper: `hold`).
