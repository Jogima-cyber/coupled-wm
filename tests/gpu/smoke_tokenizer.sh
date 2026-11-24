#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# GPU smoke test of the tokenizer trainer on the G1 data, on one node:
#
#   G1_DATA_ROOT=<dir holding new_wm_h5_segmented_small/> bash tests/gpu/smoke_tokenizer.sh [NPROC] [OUT_DIR]
#
#   1. scratch: torchrun on NPROC GPUs, G1 architecture and data pool (configs/world_model/
#      g1_tokenizer.yaml), global batch 2*NPROC windows (2 accumulation steps), 6 updates,
#      latest.pt + an archive every 3 updates, export of pretrained/ at the end;
#   2. resume on max(NPROC/2, 1) GPUs (4 accumulation steps; the optimizer state reshards)
#      up to update 10;
#   3. stop signal: continue towards update 1000, SIGUSR1 to the workers once update 12 is logged;
#      the last completed update must be saved and the workers must exit with status 75; the log
#      must show the loss normaliser decay 0.99 ** (NPROC / 64) per micro-step of NPROC windows;
#   4. resume from there to update 14;
#   5. export latest.pt with the export CLI, compare it with pretrained/, decode a G1 clip.
#
# Each phase checks latest.pt (update counter, world size, optimizer steps, finite loss state).
# Environment: PYTHON (default: python), SMOKE_COMPILE (torch.compile mode, default
# reduce-overhead; "false" disables it), SMOKE_SET (extra space-separated --set overrides, e.g. a
# small model for a quick check: "model.d_model=64 model.n_layers=4 model.num_heads_q=4
# model.num_heads_kv_latent=4"). Exit status 0 = all checks passed.

set -euo pipefail

PYTHON=${PYTHON:-python}
NPROC=${1:-$("$PYTHON" -c "import torch; print(torch.cuda.device_count())")}
OUT=${2:-$(mktemp -d "${TMPDIR:-/tmp}/smoke_tokenizer.XXXXXX")}
COMPILE=${SMOKE_COMPILE:-reduce-overhead}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN="$OUT/run"
NPROC2=$(( NPROC / 2 > 0 ? NPROC / 2 : 1 ))

: "${G1_DATA_ROOT:?set G1_DATA_ROOT to the directory holding new_wm_h5_segmented_small/}"
export G1_DATA_ROOT
(( NPROC >= 1 )) || { echo "no GPU found"; exit 1; }
cd "$REPO"
mkdir -p "$OUT"
echo "smoke_tokenizer: $NPROC GPU(s), resume on $NPROC2, output in $OUT"

read -r -a EXTRA <<< "${SMOKE_SET:-}"
COMMON=(
  "run_dir=$RUN"
  "train.global_batch=$((2 * NPROC))"
  "compile=$COMPILE"
  "schedule.warmup_updates=2"
  "schedule.total_updates=100"
  "checkpoint.save_every_updates=3"
  "checkpoint.archive_every_updates=3"
  "checkpoint.keep_archives=1"
  "logging.print_every_updates=1"
  ${EXTRA[@]+"${EXTRA[@]}"}
)

command_for() {  # command_for <nproc> <max_updates> [extra overrides...]: sets CMD
  local n=$1 max=$2
  shift 2
  CMD=("$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$n"
       -m coupled_wm.world_model.train_tokenizer --config configs/world_model/g1_tokenizer.yaml
       --set "${COMMON[@]}" "schedule.max_updates=$max" "$@")
}

train() {  # train <nproc> <max_updates> [extra overrides...]
  command_for "$@"
  "${CMD[@]}"
}

check() {  # check <global_update> <world_size>
  "$PYTHON" - "$RUN/latest.pt" "$1" "$2" <<'EOF'
import math, sys
import torch
path, update, world = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
assert ck["global_update"] == update, (ck["global_update"], update)
assert ck["world_size"] == world and len(ck["rng"]) == world, (ck["world_size"], world)
assert ck["epoch"] * (ck["num_windows"] // ck["global_batch"]) + ck["epoch_update"] == update
steps = {float(s["step"]) for s in ck["optim"]["state"].values()}
assert steps == {float(update - ck["skip"]["skip_count"])}, steps
assert all(math.isfinite(v) and v > 0 for v in ck["rms_scaler"].values()), ck["rms_scaler"]
assert math.isfinite(ck["skip"]["ema_raw_loss"])
bad = [k for sd in (ck["enc"], ck["dec"]) for k, t in sd.items() if not torch.isfinite(t).all()]
assert not bad, bad[:5]
print(f"  ok: update {update} on {world} rank(s); loss scales {ck['rms_scaler']}; skips {ck['skip']['skip_count']}")
EOF
}

echo "== 1. from scratch on $NPROC GPU(s) to update 6"
train "$NPROC" 6
check 6 "$NPROC"
test -f "$RUN/pretrained/model.safetensors"
test -d "$RUN/archive/step_00000006"

echo "== 2. resume on $NPROC2 GPU(s) to update 10"
train "$NPROC2" 10
check 10 "$NPROC2"

echo "== 3. stop signal at update 12"
LOG="$OUT/phase3.log"
command_for "$NPROC" 1000 "checkpoint.save_every_updates=1000" "checkpoint.archive_every_updates=0"
"${CMD[@]}" > "$LOG" 2>&1 &
LAUNCHER=$!  # the torchrun process; its children are the workers (torchrun does not forward SIGUSR1)
until grep -q "update 12/1000" "$LOG"; do
  kill -0 "$LAUNCHER" 2>/dev/null || { tail -50 "$LOG"; echo "phase 3: launcher exited early"; exit 1; }
  sleep 1
done
pkill -USR1 -P "$LAUNCHER"
set +e
wait "$LAUNCHER"
RC=$?
set -e
# The workers save, then exit with status 75 (stopped, relaunch to resume); torchrun reports a
# nonzero worker status as 1 (and may SIGTERM workers that are still exiting, after the save).
(( RC == 1 )) && grep -Eq "exitcode +: 75 " "$LOG" \
  || { tail -50 "$LOG"; echo "phase 3: expected a worker exit status 75 (launcher status $RC)"; exit 1; }
grep -q "stop signal received" "$LOG"
DECAY=$("$PYTHON" -c "import sys; print('%.10g' % 0.99 ** (int(sys.argv[1]) / 64))" "$NPROC")
grep -q "loss normaliser: EMA decay $DECAY per micro-step of $NPROC windows" "$LOG" \
  || { echo "phase 3: expected the loss normaliser decay $DECAY in $LOG"; exit 1; }
STOPPED=$("$PYTHON" -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)['global_update'])" "$RUN/latest.pt")
(( STOPPED >= 12 && STOPPED < 1000 )) || { echo "phase 3: saved update $STOPPED"; exit 1; }
check "$STOPPED" "$NPROC"

echo "== 4. resume after the signal to update $((STOPPED + 2))"
train "$NPROC" $((STOPPED + 2))
check $((STOPPED + 2)) "$NPROC"

echo "== 5. export and decode"
"$PYTHON" -m coupled_wm.world_model.export --ckpt "$RUN/latest.pt" --out "$OUT/export"
"$PYTHON" - "$RUN" "$OUT/export" <<'EOF'
import sys
import torch
from coupled_wm.common.checkpoint import load_pretrained
from coupled_wm.data import HDF5SequenceDataset, list_h5_files
from coupled_wm.world_model.loading import load_tokenizer
from coupled_wm.common.config import load_config
from coupled_wm.world_model.train_tokenizer import resolve_config

run, export = sys.argv[1], sys.argv[2]
a, cfg_a = load_pretrained(f"{run}/pretrained")
b, cfg_b = load_pretrained(export)
assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), "export differs from pretrained/"
assert cfg_a["architecture"] == cfg_b["architecture"]
device = "cuda" if torch.cuda.is_available() else "cpu"
enc, dec, cfg = load_tokenizer(export, device=device, dtype=torch.bfloat16)
data = resolve_config(load_config("configs/world_model/g1_tokenizer.yaml"))["data"]
files = list_h5_files(data["dirs"], include_globs=data["include_globs"])
ds = HDF5SequenceDataset(files[:1], window=8, color_order=data["color_order"], reward_key=None)
item = ds[0]
x = item["image"][None].to(device, torch.bfloat16)
p = item["proprio"][None].to(device, torch.bfloat16)
with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
    _, _, z = enc(x, proprio=p)
    _, x_hat, p_hat = dec(z)
arch = cfg["architecture"]
assert z.shape == (1, 8, arch["n_latents"], arch["bottleneck_dim"]), z.shape
assert x_hat.shape == x.shape and torch.isfinite(x_hat.float()).all()
mse = torch.mean((x_hat.float() - x.float()) ** 2).item()
print(f"  ok: export identical to pretrained/; decoded a G1 clip (MSE {mse:.4f} after {cfg['provenance']['global_update']} updates)")
EOF

echo "smoke_tokenizer: all checks passed ($OUT)"
