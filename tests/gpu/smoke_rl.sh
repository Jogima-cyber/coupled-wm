#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# GPU smoke test of the DMO trainer (configs/rl/g1_dmo.yaml) on the real G1 environment, one node:
#
#   WEIGHTS_DIR=weights G1_DATA_ROOT=<dir holding new_wm_h5_segmented_small/> \
#       bash tests/gpu/smoke_rl.sh [OUT_DIR]
#
#   1. train: torchrun on NPROC GPUs (default 4), torch.compile as configured, --no-eval;
#      16 actors (global), a replay buffer fill of one 64-step episode, 20 initial + 4 per-epoch RSSM
#      updates, 4 epochs, resumable checkpoint every 2 epochs. Checks latest.pt (iteration,
#      world size, one RNG state and one env seeding state per rank, wrapper-free keys) and
#      final_policy.pt, and, under dropped_dims_reset: skip_nonzero, prints the exclusion count
#      (PAPER_POOL=1: checks the paper pool's, 25,387 of 530,003 seed windows).
#   2. --play from final_policy.pt, single process and on all GPUs (collective evaluation).
#   3. resume: the same command with max_epochs 6 continues from latest.pt to iteration 6
#      (replay buffer refilled with the current policy, the init_udt warm-up not rerun).
#   4. standalone evaluation (python -m coupled_wm.rl.evaluate) of best_policy.pt: summary.json,
#      episodes.csv and trajectory strips.
#   5. in-training evaluation on all ranks (eval.every 1) for 2 epochs: no NCCL hang.
#   6. the ablations --no-diffusion and --no-rssm-finetuning, and
#      rssm.straight_through_stochastic_grad=true in DMO and under --no-diffusion, 2 epochs each.
#   7. FULL_SIZE=1 only: the release sizes (128 actors, 999,936-row replay buffer, global RSSM batch 256)
#      for 2 epochs with a short fill: the per-rank memory of the paper config on NPROC GPUs (the
#      iteration log prints rank 0's peak GPU memory).
#
# Environment: WEIGHTS_DIR (tokenizer/, dynamics/ and rssm/ pretrained dirs, e.g. from
# scripts/download_weights.py), G1_DATA_ROOT, optional RSSM_WARMSTART (RSSM pretrained dir, default
# $WEIGHTS_DIR/rssm), PYTHON (default python), NPROC (default 4), FULL_SIZE (default 0), PAPER_POOL
# (1: the data is the paper's G1 pool, check its exact counts), EXTRA_ARGS (more --set key=value
# arguments for every training run). Exit status 0 = all checks passed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
NPROC="${NPROC:-4}"
FULL_SIZE="${FULL_SIZE:-0}"
: "${WEIGHTS_DIR:?set WEIGHTS_DIR to the dir holding the tokenizer/ and dynamics/ pretrained dirs}"
: "${G1_DATA_ROOT:?set G1_DATA_ROOT to the dir holding new_wm_h5_segmented_small/}"
RSSM_WARMSTART="${RSSM_WARMSTART:-$WEIGHTS_DIR/rssm}"
PAPER_POOL="${PAPER_POOL:-0}"
export WEIGHTS_DIR G1_DATA_ROOT
OUT="${1:-$(mktemp -d "${TMPDIR:-/tmp}/smoke_rl.XXXXXX")}"
mkdir -p "$OUT"
echo "output: $OUT"

CONFIG=configs/rl/g1_dmo.yaml
TORCHRUN=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC")
SMALL=(--set "rssm.pretrained=$RSSM_WARMSTART"
       --set train.num_actors=16 --set train.min_replay=1024 --set train.replay_size=16384
       --set rssm.init_udt=20 --set rssm.udt=4 --set train.save_interval=2
       --set eval.num_episodes=16)
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

check_state() {  # check_state <run dir> <iteration> <world size>
    "$PYTHON" - "$1" "$2" "$3" <<'EOF'
import math, sys
from pathlib import Path
from coupled_wm.common.checkpoint import load_checkpoint
run, it, world = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
s = load_checkpoint(run / "latest.pt")
assert s["format"] == "coupled_wm.rl.training_state", s["format"]
assert s["counters"]["iter"] == it, (s["counters"], it)
assert s["world_size"] == world and len(s["rng"]) == world, (s["world_size"], len(s["rng"]))
assert len(s["env"]) == world and all(e is not None for e in s["env"]), "env seeding state per rank"
assert s["counters"]["rssm_pretrained"] == (not s["run"]["no_diffusion"] and not s["run"]["no_rssm_finetuning"])
for name in ("actor", "rssm"):
    bad = [k for k in s[name] if "_orig_mod" in k or k.startswith("module.")]
    assert not bad, (name, bad[:3])
assert all(math.isfinite(float(v.float().abs().sum())) for v in s["rssm"].values())
assert math.isfinite(float(s["log_alpha"]))
for name in ("final_policy.pt", "config.yaml"):
    assert (run / name).is_file(), name
print(f"ok: {run} at iteration {it}, alpha {math.exp(float(s['log_alpha'])):.4g}")
EOF
}

echo "== 1. train on $NPROC GPUs =="
RUN="$OUT/train"
"${TORCHRUN[@]}" -m coupled_wm.rl.train --config "$CONFIG" --seed 20 --logdir "$RUN" --no-eval \
    "${SMALL[@]}" --set train.max_epochs=4 "${EXTRA[@]}" 2>&1 | tee "$OUT/train.log"
check_state "$RUN" 4 "$NPROC"
if grep -q "dropped_dims_reset=skip_nonzero:" "$OUT/train.log"; then
    grep -o "skip_nonzero: [0-9]* of [0-9]* seed windows ([0-9.]*%)" "$OUT/train.log" | head -n 1
    if [ "$PAPER_POOL" = "1" ]; then
        grep -q "25387 of 530003 seed windows (4.8%)" "$OUT/train.log" \
            || { echo "skip_nonzero: unexpected exclusion count (paper pool: 25387 of 530003)"; exit 1; }
        echo "ok: skip_nonzero excludes 25387 of 530003 seed windows (the paper pool)"
    fi
fi
test -f "$RUN/best_policy.pt" || echo "note: no best_policy.pt (not every rank finished an episode during training)"

echo "== 2. play =="
"$PYTHON" -m coupled_wm.rl.train --play --checkpoint "$RUN/final_policy.pt" \
    --set eval.player.num_actors=4 --set eval.player.games_num=4
"${TORCHRUN[@]}" -m coupled_wm.rl.train --play --checkpoint "$RUN/final_policy.pt" \
    --set "eval.player.num_actors=$((4 * NPROC))" --set "eval.player.games_num=$((4 * NPROC))"

echo "== 3. resume =="
"${TORCHRUN[@]}" -m coupled_wm.rl.train --config "$CONFIG" --seed 20 --logdir "$RUN" --no-eval \
    "${SMALL[@]}" --set train.max_epochs=6 "${EXTRA[@]}" 2>&1 | tee "$OUT/resume.log"
check_state "$RUN" 6 "$NPROC"
grep -q "warm-up is not rerun\|resuming with the checkpointed replay buffer" "$OUT/resume.log" \
    || { echo "resume: no replay buffer refill (or reload) message"; exit 1; }

echo "== 4. standalone evaluation =="
POLICY="$RUN/best_policy.pt"
test -f "$POLICY" || POLICY="$RUN/final_policy.pt"
"$PYTHON" -m coupled_wm.rl.evaluate --checkpoint "$POLICY" --num-envs 4 --episodes 8 \
    --output-dir "$OUT/eval" --save-trajectories 2
"$PYTHON" - "$OUT/eval" <<'EOF'
import json, math, sys
from pathlib import Path
out = Path(sys.argv[1])
s = json.loads((out / "summary.json").read_text())
assert s["episodes"] >= 8 and math.isfinite(s["mean_return"]) and s["mean_length"] == 64, s
assert (out / "traj_0.png").is_file() and (out / "traj_1.png").is_file() and (out / "episodes.csv").is_file()
print(f"ok: evaluation mean return {s['mean_return']:.3f} over {s['episodes']} episodes")
EOF

echo "== 5. in-training evaluation on all ranks =="
"${TORCHRUN[@]}" -m coupled_wm.rl.train --config "$CONFIG" --seed 21 --logdir "$OUT/train_eval" \
    "${SMALL[@]}" --set train.max_epochs=2 --set eval.every=1 "${EXTRA[@]}"
check_state "$OUT/train_eval" 2 "$NPROC"

echo "== 6. ablations and the straight-through swap =="
"${TORCHRUN[@]}" -m coupled_wm.rl.train --config "$CONFIG" --seed 22 --logdir "$OUT/no_diffusion" --no-eval \
    --no-diffusion "${SMALL[@]}" --set train.max_epochs=2 "${EXTRA[@]}"
check_state "$OUT/no_diffusion" 2 "$NPROC"
"${TORCHRUN[@]}" -m coupled_wm.rl.train --config "$CONFIG" --seed 23 --logdir "$OUT/no_finetuning" --no-eval \
    --no-rssm-finetuning "${SMALL[@]}" --set train.max_epochs=2 "${EXTRA[@]}"
check_state "$OUT/no_finetuning" 2 "$NPROC"
"${TORCHRUN[@]}" -m coupled_wm.rl.train --config "$CONFIG" --seed 24 --logdir "$OUT/straight_through" --no-eval \
    "${SMALL[@]}" --set train.max_epochs=2 --set rssm.straight_through_stochastic_grad=true "${EXTRA[@]}"
check_state "$OUT/straight_through" 2 "$NPROC"
"${TORCHRUN[@]}" -m coupled_wm.rl.train --config "$CONFIG" --seed 26 --logdir "$OUT/no_diffusion_straight_through" \
    --no-eval --no-diffusion "${SMALL[@]}" --set train.max_epochs=2 --set rssm.straight_through_stochastic_grad=true \
    "${EXTRA[@]}"
check_state "$OUT/no_diffusion_straight_through" 2 "$NPROC"

if [ "$FULL_SIZE" = "1" ]; then
    echo "== 7. release sizes on $NPROC GPUs =="
    "${TORCHRUN[@]}" -m coupled_wm.rl.train --config "$CONFIG" --seed 25 --logdir "$OUT/full_size" --no-eval \
        --set "rssm.pretrained=$RSSM_WARMSTART" --set train.min_replay=8192 --set rssm.init_udt=20 \
        --set train.max_epochs=2 "${EXTRA[@]}"
    check_state "$OUT/full_size" 2 "$NPROC"
fi

echo "all RL smoke checks passed ($OUT)"
