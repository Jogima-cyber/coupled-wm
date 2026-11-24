#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GPU smoke test of the dynamics trainer on the G1 data, both stages:
#
#   WEIGHTS_DIR=weights G1_DATA_ROOT=<dir holding new_wm_h5_segmented_small/> \
#       bash tests/gpu/smoke_dynamics.sh [OUT_DIR]
#
#   1. backbone, 10 updates on all GPUs (torchrun); updates 4 and 8 use 128-frame clips;
#   2. resume it to 14 updates (update 12 is another 128-frame update);
#   2b. backbone with one micro-step per update (global batch = NPROC), 9 updates: on 4+ GPUs a
#      quarter of the ranks trains the shortcut objective while the others train flow, in the
#      same update (long-clip updates 4 and 8 included); checks that the ranks stay in step;
#   3. SIGTERM to torchrun (1 process, 2 DataLoader workers): the trainer checkpoints at the end
#      of its update and exits non-zero; resume it;
#   4. reward stage from the backbone latest.pt (6 updates), and from the backbone pretrained dir;
#   5. export with `python -m coupled_wm.world_model.export`, reload with
#      coupled_wm.world_model.loading, check the tokenizer pairing and run a forward pass.
#
# Except in 2b, every update has 4 accumulation steps (3 flow + 1 shortcut micro-batch on every
# rank, as in the paper).
# Environment: WEIGHTS_DIR (holding the G1 tokenizer/ pretrained dir, e.g. from
# scripts/download_weights.py) or TOKENIZER (that dir itself), G1_DATA_ROOT or DATA_DIR
# (the world_model data dir itself), optional PYTHON (default python), NPROC (default: all visible GPUs), COMPILE=0 (no
# torch.compile; default 1 = the configs' compile modes), TINY=1 (4-layer, 256-wide dynamics
# instead of the G1 size), EXTRA_ARGS (more `--set key=value` arguments for every run), DEVICE
# (default cuda; with NPROC=1, COMPILE=0 and tiny assets the whole flow also runs on cpu).
# Output goes to OUT_DIR (default: a new temporary directory).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOKENIZER="${TOKENIZER:-${WEIGHTS_DIR:+$WEIGHTS_DIR/tokenizer}}"
DATA_DIR="${DATA_DIR:-${G1_DATA_ROOT:+$G1_DATA_ROOT/new_wm_h5_segmented_small/world_model}}"
: "${TOKENIZER:?set WEIGHTS_DIR (or TOKENIZER) to locate the G1 tokenizer pretrained dir}"
: "${DATA_DIR:?set G1_DATA_ROOT (or DATA_DIR) to locate the G1 world_model data dir}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
DEVICE="${DEVICE:-cuda}"
NPROC="${NPROC:-$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')}"
[ "$NPROC" -ge 1 ] || { echo "no GPU visible" >&2; exit 1; }
OUT="${1:-$(mktemp -d "${TMPDIR:-/tmp}/smoke_dynamics.XXXXXX")}"
mkdir -p "$OUT"
cd "$REPO"
echo "smoke_dynamics: $NPROC GPU(s), output in $OUT"

COMMON=(
  --set "tokenizer.path=$TOKENIZER"
  --set "data.dirs=['$DATA_DIR']"
  --set data.num_workers=2
  --set schedule.warmup_updates=2
  --set schedule.total_updates=1000
  --set checkpoint.save_every_updates=5
  --set logging.print_every_updates=1
)
if [ "${COMPILE:-1}" = "0" ]; then
  COMMON+=(--set compile=null --set tokenizer.compile=null)
fi
if [ "${TINY:-0}" = "1" ]; then
  COMMON+=(--set model.d_model=256 --set model.num_layers=4 --set model.num_heads=4)
fi
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
COMMON+=("${EXTRA[@]}")
BACKBONE=(--config configs/world_model/g1_dynamics_backbone.yaml "${COMMON[@]}" --set train.long_every=4)
REWARD=(--config configs/world_model/g1_dynamics_reward.yaml "${COMMON[@]}")

torchrun_train() {
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    -m coupled_wm.world_model.train_dynamics --set "train.global_batch=$((NPROC * 4))" "$@"
}

check() {  # check <latest.pt> <python expression over ck>
  "$PYTHON" - "$1" "$2" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
ok = eval(sys.argv[2], {"ck": ck})
print(f"check {sys.argv[1]}: {sys.argv[2]} -> {ok} (counters {ck['counters']}, global_update {ck['global_update']})")
sys.exit(0 if ok else 1)
EOF
}

echo "== 1. backbone, 10 updates (128-frame clips at updates 4 and 8)"
torchrun_train "${BACKBONE[@]}" --set "run_dir=$OUT/backbone" --set schedule.max_updates=10
check "$OUT/backbone/latest.pt" 'ck["counters"] == {"stage_update": 10, "long_updates": 2}'

echo "== 2. resume the backbone to 14 updates"
torchrun_train "${BACKBONE[@]}" --set "run_dir=$OUT/backbone" --set schedule.max_updates=14
check "$OUT/backbone/latest.pt" 'ck["counters"] == {"stage_update": 14, "long_updates": 3} and ck["scheduler"]["last_epoch"] == 14'

echo "== 2b. backbone with one micro-step per update (global batch $NPROC)"
torchrun_train "${BACKBONE[@]}" --set "run_dir=$OUT/one_step" --set schedule.max_updates=9 \
  --set "train.global_batch=$NPROC" 2>&1 | tee "$OUT/one_step.log"
check "$OUT/one_step/latest.pt" 'ck["counters"] == {"stage_update": 9, "long_updates": 2}'
grep -q "shortcut fraction 1/4 of the $NPROC micro-batches of an update" "$OUT/one_step.log" \
  || { echo "no shortcut layout line in $OUT/one_step.log" >&2; exit 1; }

echo "== 3. SIGTERM to torchrun (forwarded to the trainer and its DataLoader workers), then resume one more update"
SIG_RUN=(--config configs/world_model/g1_dynamics_backbone.yaml "${COMMON[@]}" --set "run_dir=$OUT/sigterm"
         --set train.global_batch=4 --set train.long_every=0 --set checkpoint.save_every_updates=1000)
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 -m coupled_wm.world_model.train_dynamics \
  "${SIG_RUN[@]}" --set schedule.max_updates=1000 > "$OUT/sigterm.log" 2>&1 &
PID=$!
for _ in $(seq 1 1800); do
  grep -q "update 2/1000" "$OUT/sigterm.log" 2>/dev/null && break
  kill -0 "$PID" 2>/dev/null || { cat "$OUT/sigterm.log"; echo "trainer exited early" >&2; exit 1; }
  sleep 1
done
kill -TERM "$PID"
RC=0
wait "$PID" || RC=$?
[ "$RC" -ne 0 ] || { echo "a run stopped before max_updates must exit non-zero" >&2; exit 1; }
grep -q "stop requested" "$OUT/sigterm.log" || { tail -20 "$OUT/sigterm.log"; echo "no stop message" >&2; exit 1; }
STOPPED=$("$PYTHON" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)['counters']['stage_update'])" "$OUT/sigterm/latest.pt")
echo "stopped at stage update $STOPPED"
"$PYTHON" -m coupled_wm.world_model.train_dynamics "${SIG_RUN[@]}" --set "schedule.max_updates=$((STOPPED + 1))"
check "$OUT/sigterm/latest.pt" "ck['counters']['stage_update'] == $((STOPPED + 1))"

echo "== 4. reward stage from the backbone latest.pt (6 updates) and from its pretrained dir (2 updates)"
torchrun_train "${REWARD[@]}" --set "run_dir=$OUT/reward" --set "init_from=$OUT/backbone/latest.pt" \
  --set schedule.max_updates=6
check "$OUT/reward/latest.pt" 'ck["counters"]["stage_update"] == 6 and ck["global_update"] == 20 and set(ck["loss_scaler"]) == {"loss_dyn", "loss_reward"}'
torchrun_train "${REWARD[@]}" --set "run_dir=$OUT/reward_from_dir" --set "init_from=$OUT/backbone/pretrained" \
  --set schedule.max_updates=2
check "$OUT/reward_from_dir/latest.pt" 'ck["counters"]["stage_update"] == 2 and ck["init"]["global_update"] == 14'

echo "== 5. export, reload, pairing, forward"
"$PYTHON" -m coupled_wm.world_model.export --ckpt "$OUT/reward/latest.pt" --out "$OUT/export" --tokenizer "$TOKENIZER"
"$PYTHON" - "$OUT" "$TOKENIZER" "$DEVICE" <<'EOF'
import sys
import torch
from coupled_wm.common.checkpoint import load_pretrained, load_pretrained_config
from coupled_wm.world_model.loading import check_pairing, load_dynamics

out, tok, device = sys.argv[1], sys.argv[2], sys.argv[3]
dyn, cfg = load_dynamics(f"{out}/export", device=device, dtype=torch.bfloat16)
check_pairing(load_pretrained_config(tok), cfg, tok)
assert cfg["inference"]["d_min_idx"] == 7 and cfg["training"]["stage"] == "reward"
assert cfg["training"]["global_update"] == 20, cfg["training"]
a, b = load_pretrained(f"{out}/export")[0], load_pretrained(f"{out}/reward/pretrained")[0]
assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "trainer export != export module"
arch = cfg["architecture"]
B, T = 1, 8
z = torch.rand(B, T, arch["num_latents"], arch["latent_dim"], device=device) * 2 - 1
act = torch.zeros(B, T, arch["action_dim"], device=device)
tau = torch.full((B, T), 96, dtype=torch.long, device=device)
step = torch.full((B, T), 7, dtype=torch.long, device=device)
with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
    pred, rew = dyn(act, z, tau, step)
assert torch.isfinite(pred).all() and rew.shape == (B, T, arch["mtp_length"], arch["reward_num_buckets"])
print("export reload OK:", {k: cfg["training"][k] for k in ("stage", "global_update", "stage_update")})
EOF

echo "smoke_dynamics: all checks passed ($OUT)"
