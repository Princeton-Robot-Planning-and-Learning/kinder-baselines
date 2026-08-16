#!/usr/bin/env bash
# Shard the 40-seed attempt-count sweep across 8 processes. SCRATCH ONLY.
set -uo pipefail
cd /home/josh/.claude/jobs/tossdiag
OUT=${1:-out}
ENVSH=${2:-./env.sh}
mkdir -p "$OUT"
for lo in 100 105 110 115 120 125 130 135; do
    hi=$((lo + 4))
    systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -p OOMPolicy=continue \
        -- "$ENVSH" sweep_attempts.py --seeds "$lo-$hi" --out "$OUT/att-$lo.jsonl" \
        > "$OUT/att-$lo.log" 2>&1 &
done
wait
echo ALLDONE
