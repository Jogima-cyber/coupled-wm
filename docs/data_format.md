# Data format

Every stage reads the same on-disk format: the tokenizer, both dynamics stages, RSSM pretraining
and the RL environment, which seeds each episode from a window of real frames. One loader
(`coupled_wm.data.h5_dataset.HDF5SequenceDataset`) serves all of them.

A dataset is a pool of HDF5 files, typically one file per chunk of recording. Each file holds
`T` consecutive frames, possibly from several episodes, as top-level arrays (`f["images"]`,
`f["commands"]`, ...):

| key | shape, dtype | content | read by |
|---|---|---|---|
| `images` | `(T, H, W, 3)` uint8 | camera frames, BGR or RGB ([colour order](#frames)) | every stage |
| `dones` | `(T,)` or `(T, 1)`, values 0/1 | 1 on the **last** frame of each episode ([episodes](#episodes-and-dones)) | every stage |
| `commands` | `(T, A)` float | absolute command that conditions frame `t` ([commands](#commands-and-actions)) | dynamics, RSSM, RL env |
| `proprio` | `(T, P)` float | proprioception of frame `t`, unnormalised ([proprio](#proprioception)) | tokenizer, dynamics and RL env, when the tokenizer uses proprio |
| `rewards` | `(T,)` or `(T, 1)` numeric | reward label of frame `t` ([rewards](#rewards)) | dynamics reward stage, RSSM pretraining |

Other datasets in a file are ignored. A file that cannot be read, or lacks a key the stage
needs, stops the run with an error naming the file; no file is skipped silently.

## Frames

- `uint8`, `(T, H, W, 3)`. Every model sees RGB floats in `[0, 1]`; the loader converts.
- **Colour order.** Each stage's `data.color_order` gives the order on disk (`bgr` or `rgb`).
  Set it in the tokenizer config; the dynamics configs and the RL env take the order the tokenizer
  recorded when theirs is `null`. RSSM pretraining has its own explicit `data.color_order`; it must
  match. Write the order into the files as an attribute, and every loader checks it against
  its config:

  ```python
  f["images"].attrs["channel_order"] = "rgb"   # or "bgr"
  ```

- **Resolution.** Store frames at the tokenizer's resolution (G1: 256 x 256; height and width
  must be multiples of the 16-pixel patch). Every stage then reads identical frames. Frames of
  another size are resized to the tokenizer's size on the fly, with bilinear interpolation
  (`align_corners=False`, no antialiasing, on float frames), by the tokenizer and dynamics
  trainers and the RL environment alike; each logs it at start-up. RSSM pretraining
  resizes the stored frames straight to its 64 x 64 input, whereas RL observations are decoder
  frames at the tokenizer's resolution resized to 64 x 64. With stored frames at the tokenizer's
  resolution the two paths match exactly; with any other size they filter the frames
  differently, so the RSSM is pretrained on slightly different images than it sees during RL.
- Frame rate: G1 is recorded at 10 Hz. The rate is recorded in the exported `config.json`;
  no computation depends on it.

## Episodes and dones

- `dones[t] = 1` marks the **last** frame of an episode (or of any contiguous segment). A file's
  last segment ends at the end of the file, with or without a done.
- Do not store an `is_first` flag in `dones`: a flag on the first frame of each episode makes every
  episode start one frame late and run into the next one. `coupled_wm.data.validate` warns
  when frame 0 of a file has `done = 1`.
- Training windows never cross a done or a file boundary, so an episode that continues in the next
  file is cut there. Segments shorter than a stage's window yield no window for that stage (the
  loader logs how many it skips). The windows are:

  | stage | window (frames) |
  |---|---|
  | tokenizer | 64 |
  | dynamics | 32, plus 128 on every 8th backbone update |
  | RSSM pretraining | 65 (`sequence_length + 1`) |
  | RL environment seeding | 32 (`init_context_len`) |

## Commands and actions

`commands` holds the **absolute** command of every controlled dimension, in the units of the robot's
low-level controller (G1: base velocity, upper-body joint position targets in radians, grip
closure in `[0, 1]`). No normalisation is applied anywhere.

**Alignment.** `commands[t]` is the command consumed together with frame `t`: the dynamics model
puts `commands[t]` and the latents of frame `t` in the same token slot, with no shift. Store the
command whose effect frame `t` shows. A logger that records pairs `(obs_t, action_t)`, where
`action_t` is sent after observing `obs_t` and produces `obs_{t+1}`, therefore stores

```
commands[t] = action_{t-1}          (commands[0]: the command in force before the first frame)
```

Actuation latency that remains in the data is learned by the world model: in the G1 data a
command's effect shows about 2 frames after it is recorded.

**Relative actions.** The policy and the RSSM work with command changes:

```
dcmd[t] = commands[t] - commands[t-1]
```

- RSSM pretraining turns a window of 65 frames into 64 rows. The row of frame `t` is
  `(frame t, dcmd[t], rewards[t])`, and the first frame of the window only supplies
  `commands[t-1]`. The first row of each window has `is_first = 1`.
- In the RL environment the policy outputs `a_t`, a per-step change, and the environment integrates
  it in fp32: `c_t = clamp(c_{t-1} + a_t)`, starting from `c_0 = commands[31]` of the seed window.
  The clamped absolute command `c_t` conditions the frame being generated, and the step reward is
  the reward head's prediction for that frame. Online RSSM finetuning pairs each generated frame
  with the action that produced it, the same convention as pretraining.
- The clamp is a symmetric per-dimension envelope `[-scale, scale]` plus a `grip_range` for grip
  dimensions. For a new dataset, compute it with
  `python scripts/compute_action_envelope.py <data_dir> --grip-dims ...`, which prints the YAML
  for the RL config's `env` block or for the dynamics export
  (see [training.md](training.md#new-robots-and-datasets)).
- Dimensions the policy does not control (`env.drop_action_indices`) receive a zero change from
  the trainer; `env.dropped_dims_reset` sets what they hold (G1: the base velocity dims, see the
  RL config).

## Proprioception

- `(T, P)` float, stored as is: the tokenizer embeds one proprio token per frame and reconstructs
  the vector (MSE term). There is no normalisation; the exported `config.json` records
  `proprio_normalization: null`.
- The tokenizer casts proprio to bf16, both as input and as reconstruction target. bf16 keeps 8
  significant bits (2.4 decimal digits), so the gap between neighbouring bf16 numbers grows with
  the value: about 0.008 near 1, but 8 just below 2048 and 16 above it. A channel that stays far from zero and moves only
  a little loses that movement: a joint encoder reading between 2043 and 2053 becomes 2040 (for
  2043) or 2048 (for every other value). Subtract such a constant offset before storing (the same
  readings minus 2048 are -5 to 5, which bf16 holds exactly) and keep channels roughly zero-centred
  and of order 1. The G1 channels lie within ±2.3.
- Without proprioception, set `model.use_proprio: false` in the tokenizer config. The dynamics
  trainer and the environment follow the tokenizer's setting.

## Rewards

- A per-frame label of frame `t`, any numeric value. Both reward heads (the dynamics model's
  8-step multi-token head and the RSSM's) predict a two-hot distribution over 255 bins spaced
  uniformly in symlog space on `[-20, 20]`.
- G1: `{0, 1}`, set by hand in a post-processing pass after collection to mark task-relevant
  behaviour; the demonstration files carry 1 on every frame.
- The reward the policy maximises is the dynamics reward head's prediction for each generated frame.
  The RSSM's reward head is pretrained on the same labels and finetuned online on the
  environment's rewards.
- Two opt-in transforms, applied by the dynamics reward stage and RSSM pretraining:
  `data.reward_remap: {old: new}` replaces exact values (the G1 configs set `{10.0: 2.0}`,
  which changes nothing on G1 data since it holds only 0 and 1), and `data.reward_override_dir`
  replaces the rewards of each file by the `reward` dataset of the same-named file in that directory
  (a missing override file is an error).

## Selecting files

Every `data` block (tokenizer, dynamics, RSSM, and `env.data` in the RL config) accepts the same keys:

| key | effect |
|---|---|
| `dirs` | directories to scan, in the given order; file names sorted within a directory; symlinks followed |
| `include_globs` | keep file names matching any of these globs (default `["*.h5"]`) |
| `exclude_globs` | then drop file names matching any of these |
| `min_chunk_index` | then drop files named exactly `chunk_<N>.h5` with `N` below this; files with any other name are always kept (with `min_chunk_index: 21`, `chunk_0004.h5` is dropped but `patch_2_chunk_0004.h5` is kept) |
| `file_list` | a list of paths, or a text file with one path per line (`#` comments), used instead of the scan; the filters must keep every listed file |

The pool order fixes the window numbering (file by file, segment by segment, start by start), so
a seeded sampler maps a seed to the same windows for the same file list. A file listed twice, or an
empty pool, is an error.

## Validating a dataset

```bash
python -m coupled_wm.data.validate data/my_robot \
    --image-size 256 --proprio-dim 50 --action-dim 22 --color-order rgb --window 32 64 128
```

The validator checks every file against the table above and prints the frame and segment counts,
the command and proprio ranges, constant proprio channels, the reward values, and how many windows
each window length yields. It exits with status 1 on any error. It also warns about data that loads
but is likely mislabelled: done on frame 0 (an `is_first` flag), files that do not end with a done,
and files without a `channel_order` attribute. The selection flags (`--include`, `--exclude`,
`--min-chunk-index`, `--file-list`) are those of the configs; `--no-action`, `--no-proprio` and
`--no-reward` drop a requirement (`--no-action` for video used by the tokenizer only);
`--read-images` reads every frame to catch corrupt chunks.

## The G1 dataset

The G1 data is the Hugging Face dataset
[`Jogima-cyber/coupled-global-local-wm-rl`](https://huggingface.co/datasets/Jogima-cyber/coupled-global-local-wm-rl)
(CC BY 4.0, 153 GB). Set `G1_DATA_ROOT` (the configs' default is `data/g1`) and download it there:

```bash
export G1_DATA_ROOT=$PWD/data/g1
hf download Jogima-cyber/coupled-global-local-wm-rl --repo-type dataset --local-dir $G1_DATA_ROOT
```

RL with the released weights reads only the RL seed pool (207 files, 106 GB):

```bash
hf download Jogima-cyber/coupled-global-local-wm-rl \
    "new_wm_h5_segmented_small/world_model/chunk_002[1-9].h5" \
    "new_wm_h5_segmented_small/world_model/chunk_00[3-9]?.h5" \
    "new_wm_h5_segmented_small/world_model/chunk_0[12]??.h5" \
    --repo-type dataset --local-dir $G1_DATA_ROOT
```

The configs expect two directories under `G1_DATA_ROOT`:

- `new_wm_h5_segmented_small/world_model/`: 290 files, the 228 play chunks `chunk_0000.h5` to
  `chunk_0227.h5` plus 62 `patch*_chunk_*.h5` files from later collection sessions. 752,861
  frames (20.9 h at 10 Hz), up to 2,700 per file, in 419 segments.
- `new_wm_h5_demos_segmented_small/world_model/`: 10 demonstration files `chunk_0000.h5` to
  `chunk_0009.h5`, 25,960 frames, reward 1 on every frame.

Every file holds `images` (256 x 256, **BGR**, no channel-order attribute), `commands` (22-d float32),
`proprio` (50-d float32), `rewards` (float32, `{0, 1}`) and `dones` (float32). The datasets
`is_demo` and `source_rrd` are not read. The commands are the inputs of the robot's low-level
controller, a controller based on the [SONIC](https://arxiv.org/abs/2511.07820) policy. Their 22
dimensions are:

| dims | command |
|---|---|
| 0-2 | base velocity `vx`, `vy`, yaw rate (not controlled by the policy) |
| 3-19 | 17 upper-body joint positions (3 waist, 14 arm), radians |
| 20-21 | 2 grip channels in `[0, 1]` |

Each stage uses a different part of the data, as in the paper:

| stage | files | size |
|---|---|---|
| tokenizer | `chunk_0000` - `chunk_0146` (147 files) | 381,186 frames |
| dynamics backbone | the 228 `chunk_*.h5` | 593,329 frames |
| dynamics reward stage | all 290 files | 752,861 frames |
| RSSM pretraining | the 228 `chunk_*.h5` and the 10 demonstration files | 619,289 frames |
| RL seed pool | `chunk_0021` - `chunk_0227` (207 files) | 530,003 windows of 32 frames |

The tokenizer selection is the data the paper's tokenizer was trained on (the other files were
added to the directory after it). Check the directory with:

```bash
python -m coupled_wm.data.validate $G1_DATA_ROOT/new_wm_h5_segmented_small/world_model \
    --image-size 256 --proprio-dim 50 --action-dim 22 --color-order bgr --window 32 64 128
```

Since the G1 files carry no channel-order attribute, the validator ends with one expected warning
saying so.
