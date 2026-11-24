#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# GPU smoke test of RSSM pretraining (configs/rssm/g1_rssm_pretrain.yaml) on the real G1 data.
# Needs one GPU with the memory of the paper run (H100 80 GB), about 40 GB of RAM and ~8 GB of
# disk in the work dir; takes about 15 minutes.
#
#   G1_DATA_ROOT=<dir holding new_wm_h5_segmented_small/ and new_wm_h5_demos_segmented_small/> \
#       bash tests/gpu/smoke_rssm_pretrain.sh [work_dir]
#
# 1. The full G1 pool with the config as shipped (float32 RAM cache, bf16, torch.compile):
#    200 steps, stop; then resume with streamed frames (no cache) to step 300. Checks the pool
#    record (PAPER_POOL=1: the paper's, 238 files, 585,607 windows, 36,600 steps per epoch), the
#    counters across the resume, and prints the throughput after compilation (the paper run:
#    about 9,970 frames/s on one H100).
#    A resume with a uint8 cache (re-quantised frames) must be refused before any cache is built.
# 2. A 2-file subset of the pool (chunk_0000.h5 of the play and the demo dir): 2 epochs in two
#    pieces (stop at step 150, resume). Checks that rssm_e1/ and rssm_e2/ strict-load into
#    build_rssm(config.json) and into the RL config's rssm block, their config.json, and that
#    rssm_e2 holds the final weights.
# 2b. SIGTERM to every process of a running subset job (as a batch scheduler sends it, data
#    loader workers included) after the first progress line: the run saves latest.pt, exits 0,
#    and a relaunch resumes from that step and finishes the epoch.
# 3. Held-out validation (one file of the subset) with the compiled model (CUDA graphs under
#    no_grad, one step mark per batch): 1 epoch, a validation every 40 steps; checks that torch
#    printed no CUDA-graph warning and the validation loss recorded in rssm_e1/config.json.
# 4. With 2 or more GPUs: the subset for 1 epoch under torchrun --nproc_per_node=2 (DDP).
#
# Environment: PYTHON (python executable, default `python`), PAPER_POOL (1: the data is the
# paper's G1 pool, check its exact sizes).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-${PY:-python}}"
CONFIG=configs/rssm/g1_rssm_pretrain.yaml

: "${G1_DATA_ROOT:?set G1_DATA_ROOT to the directory holding the G1 data dirs}"
for d in new_wm_h5_segmented_small/world_model new_wm_h5_demos_segmented_small/world_model; do
    [[ -d "$G1_DATA_ROOT/$d" ]] || { echo "missing $G1_DATA_ROOT/$d" >&2; exit 1; }
done
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU visible')"

WORK="${1:-$(mktemp -d -t smoke_rssm_pretrain.XXXXXX)}"
mkdir -p "$WORK"
echo "work dir: $WORK"
SUBSET=(--set 'data.include_globs=["chunk_0000.h5"]')

last_rate() {  # frames/s of the last progress line of a log
    grep -o '[0-9]* frames/s' "$1" | tail -n 1 || true
}

no_cudagraph_warnings() {  # torch's CUDA-graph warnings (pending backwards, overwritten outputs)
    if grep -q "CUDAGraphs" "$1"; then
        grep "CUDAGraphs" "$1" >&2
        echo "CUDA-graph warnings in $1" >&2
        exit 1
    fi
}

echo "== 1. full pool: 200 steps, then resume (streamed frames) to 300"
RUN1="$WORK/full"
"$PY" -m coupled_wm.rssm.pretrain --config "$CONFIG" --set run_dir="$RUN1" \
    --set train.max_steps=200 --set train.print_every=20 2>&1 | tee "$WORK/full_a.log"
echo "throughput, cached frames (steps 160-180): $(last_rate "$WORK/full_a.log")"
no_cudagraph_warnings "$WORK/full_a.log"
"$PY" - "$RUN1" 200 "${PAPER_POOL:-0}" <<'EOF'
import sys, torch
ckpt = torch.load(f"{sys.argv[1]}/latest.pt", map_location="cpu", weights_only=False, mmap=True)
state, data, step = ckpt["state"], ckpt["data"], int(sys.argv[2])
spe = data["num_windows"] // ckpt["config"]["train"]["batch_size"]
if sys.argv[3] == "1":
    assert (data["num_files"], data["num_frames"], data["num_windows"]) == (238, 619289, 585607), data["num_files"]
    assert spe == 36600, spe
assert state == {"epoch": 0, "step": step, "global_step": step, "steps_per_epoch": spe}, state
print("full pool ok:", {k: data[k] for k in ("num_files", "num_frames", "num_windows")}, state)
EOF
"$PY" -m coupled_wm.rssm.pretrain --config "$CONFIG" --set run_dir="$RUN1" \
    --set train.max_steps=300 --set train.print_every=20 --set data.cache=false 2>&1 | tee "$WORK/full_b.log"
grep -q "resumed from .* at epoch 0, step 200" "$WORK/full_b.log" || { echo "did not resume at step 200" >&2; exit 1; }
echo "throughput, streamed frames (steps 260-280): $(last_rate "$WORK/full_b.log")"
"$PY" - "$RUN1" 300 <<'EOF'
import sys, torch
state = torch.load(f"{sys.argv[1]}/latest.pt", map_location="cpu", weights_only=False, mmap=True)["state"]
assert state["global_step"] == int(sys.argv[2]) and state["epoch"] == 0, state
print("resume ok:", state)
EOF
if "$PY" -m coupled_wm.rssm.pretrain --config "$CONFIG" --set run_dir="$RUN1" \
    --set train.max_steps=400 --set data.cache_dtype=uint8 > "$WORK/full_c.log" 2>&1; then
    echo "a resume with re-quantised (uint8) frames was accepted" >&2; exit 1
fi
grep -q "gives 'uint8-requantised' frames" "$WORK/full_c.log" || { cat "$WORK/full_c.log" >&2; exit 1; }
! grep -q "building RAM cache" "$WORK/full_c.log" || { echo "the cache was built before the refusal" >&2; exit 1; }
echo "uint8 resume refused before building a cache: ok"

echo "== 2. subset: 2 epochs in two pieces, strict reload of rssm_e1/ and rssm_e2/"
RUN2="$WORK/subset"
"$PY" -m coupled_wm.rssm.pretrain --config "$CONFIG" --set run_dir="$RUN2" "${SUBSET[@]}" \
    --set train.epochs=2 --set train.max_steps=150 --set train.print_every=50 2>&1 | tee "$WORK/subset_a.log"
"$PY" -m coupled_wm.rssm.pretrain --config "$CONFIG" --set run_dir="$RUN2" "${SUBSET[@]}" \
    --set train.epochs=2 --set train.print_every=100 2>&1 | tee "$WORK/subset_b.log"
grep -q "resumed from .* at epoch 0, step 150" "$WORK/subset_b.log" || { echo "did not resume at step 150" >&2; exit 1; }
"$PY" - "$RUN2" 1 <<'EOF'
import json, sys
from pathlib import Path
import torch
from coupled_wm.common.checkpoint import load_pretrained
from coupled_wm.common.config import load_config
from coupled_wm.rssm.rssm import build_rssm, load_rssm

run, world = Path(sys.argv[1]), int(sys.argv[2])
# The RL config's rssm block plus the sizes the trainer adds (G1 action dim, observation size).
rl = load_config("configs/rl/g1_dmo.yaml")
rl_block = {**rl["rssm"], "action_size": 22, "image_size": rl["env"]["img_size"]}
ckpt = torch.load(run / "latest.pt", map_location="cpu", weights_only=False, mmap=True)
spe, epochs = ckpt["state"]["steps_per_epoch"], ckpt["config"]["train"]["epochs"]
assert ckpt["state"] == {"epoch": epochs, "step": 0, "global_step": epochs * spe, "steps_per_epoch": spe}, ckpt["state"]
for epoch in range(1, epochs + 1):
    d = run / f"rssm_e{epoch}"
    state_dict, cfg = load_pretrained(d)
    model = build_rssm(cfg)
    model.load_state_dict(state_dict, strict=True)
    load_rssm(d, config=rl_block, device="cuda")
    assert cfg["model_type"] == "rssm" and cfg["action"]["mode"] == "relative", cfg["action"]
    assert cfg["action"]["definition"] == "commands[t] - commands[t-1]" and cfg["action"]["dim"] == 22
    assert cfg["preprocessing"]["image_size"] == [64, 64] and cfg["preprocessing"]["h5_color_order"] == "bgr"
    assert cfg["preprocessing"]["frame_values"] == "exact" and cfg["data"]["source_image_sizes"] == [[256, 256]]
    assert cfg["data"]["num_files"] == 2 and len(cfg["data"]["files"]) == 2, cfg["data"]["files"]
    prov = cfg["provenance"]
    assert (prov["epoch"], prov["global_step"], prov["world_size"]) == (epoch, epoch * spe, world), prov
    assert prov["compiled"] and prov["precision"] == "bf16 autocast"
    print(f"{d.name}: strict load ok, files {cfg['data']['files']}, global step {prov['global_step']}")
last, _ = load_pretrained(run / f"rssm_e{epochs}")
assert all(torch.equal(last[k], ckpt["model"][k].float()) for k in last), "rssm_e{N} != latest.pt weights"
print(f"subset ok: {spe} steps per epoch, world size {world}")
EOF

echo "== 2b. SIGTERM to every process of a running job: save, exit 0, resume"
RUNS="$WORK/subset_sigterm"
SIGTERM_ARGS=(--config "$CONFIG" --set run_dir="$RUNS" "${SUBSET[@]}" --set train.epochs=1 --set train.print_every=20)
set -m  # job control: the background job gets its own process group, PGID = $!
"$PY" -m coupled_wm.rssm.pretrain "${SIGTERM_ARGS[@]}" > "$WORK/sigterm_a.log" 2>&1 &
PID=$!
set +m
for _ in $(seq 900); do  # the first progress line comes after compilation
    grep -q "epoch 1 step" "$WORK/sigterm_a.log" && break
    kill -0 "$PID" 2> /dev/null || break
    sleep 1
done
kill -TERM -- "-$PID"  # the whole process group: the trainer and its data loader workers
STATUS=0
wait "$PID" || STATUS=$?
cat "$WORK/sigterm_a.log"
[[ $STATUS -eq 0 ]] || { echo "exit status $STATUS after SIGTERM" >&2; exit 1; }
grep -q "stop signal received: saved epoch 0, step" "$WORK/sigterm_a.log" || { echo "no save on SIGTERM" >&2; exit 1; }
"$PY" -m coupled_wm.rssm.pretrain "${SIGTERM_ARGS[@]}" 2>&1 | tee "$WORK/sigterm_b.log"
grep -q "resumed from .* at epoch 0, step" "$WORK/sigterm_b.log" || { echo "did not resume after SIGTERM" >&2; exit 1; }
"$PY" - "$RUNS" <<'EOF'
import sys, torch
state = torch.load(f"{sys.argv[1]}/latest.pt", map_location="cpu", weights_only=False, mmap=True)["state"]
assert state["epoch"] == 1 and state["step"] == 0, state
print("SIGTERM ok:", state)
EOF

echo "== 3. held-out validation with the compiled model"
RUNV="$WORK/subset_val"
"$PY" -m coupled_wm.rssm.pretrain --config "$CONFIG" --set run_dir="$RUNV" "${SUBSET[@]}" \
    --set train.epochs=1 --set validation.holdout_every=2 --set validation.every_steps=40 \
    --set validation.num_batches=5 --set train.print_every=50 2>&1 | tee "$WORK/subset_val.log"
grep -q "validation at step 40:" "$WORK/subset_val.log" || { echo "no validation at step 40" >&2; exit 1; }
# Validations after the first replay the recorded no_grad graphs, where a missing step mark reads
# overwritten outputs.
(( $(grep -c "validation at step" "$WORK/subset_val.log") >= 2 )) || { echo "fewer than 2 validations" >&2; exit 1; }
no_cudagraph_warnings "$WORK/subset_val.log"
"$PY" - "$RUNV" <<'EOF'
import json, math, sys
cfg = json.load(open(f"{sys.argv[1]}/rssm_e1/config.json"))
assert len(cfg["data"]["files"]) == 1 and len(cfg["data"]["validation_files"]) == 1, cfg["data"]
loss = cfg["provenance"]["validation_loss"]
assert loss is not None and math.isfinite(loss), loss
print("validation ok: loss", loss)
EOF

NGPU="$("$PY" -c "import torch; print(torch.cuda.device_count())")"
if (( NGPU >= 2 )); then
    echo "== 4. subset under torchrun with 2 processes (DDP), 1 epoch"
    RUN3="$WORK/subset_ddp"
    "$PY" -m torch.distributed.run --standalone --nproc_per_node=2 -m coupled_wm.rssm.pretrain \
        --config "$CONFIG" --set run_dir="$RUN3" "${SUBSET[@]}" --set train.epochs=1 \
        --set train.print_every=100 2>&1 | tee "$WORK/subset_ddp.log"
    "$PY" - "$RUN3" <<'EOF'
import sys
from pathlib import Path
import torch
from coupled_wm.rssm.rssm import load_rssm

run = Path(sys.argv[1])
ckpt = torch.load(run / "latest.pt", map_location="cpu", weights_only=False, mmap=True)
assert len(ckpt["rng"]) == 2 and ckpt["state"]["epoch"] == 1, ckpt["state"]
_, cfg = load_rssm(run / "rssm_e1")
assert cfg["provenance"]["world_size"] == 2
print("DDP ok:", ckpt["state"])
EOF
else
    echo "== 4. skipped (needs 2 GPUs, found $NGPU)"
fi

echo "smoke_rssm_pretrain: OK ($WORK)"
