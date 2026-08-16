#!/usr/bin/env bash
# Shard a 40-seed sweep with a configurable samples-per-step. SCRATCH ONLY.
#   usage: run_att2.sh <outdir> <envsh> <samples_per_step>
set -uo pipefail
cd /home/josh/.claude/jobs/tossdiag
OUT=$1
ENVSH=$2
SPS=$3
mkdir -p "$OUT"
for lo in 100 105 110 115 120 125 130 135; do
    hi=$((lo + 4))
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue \
        -- "$ENVSH" sweep_attempts.py --seeds "$lo-$hi" --samples-per-step "$SPS" \
        --out "$OUT/att-$lo.jsonl" > "$OUT/att-$lo.log" 2>&1 &
done
wait
echo ALLDONE
