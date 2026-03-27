#!/bin/bash
# Script to run bilevel planning on DynPushPullHook2D with multiple random seeds

# Navigate to the kinder-bilevel-planning directory
cd "$(dirname "$0")/.." || exit 1

# Define the seeds to run
SEEDS=(303 304)

echo "Starting bilevel planning DynPushPullHook2D experiments..."
echo "=============================================="

# Run experiments for 5 obstructions
echo ""
echo "Running experiments for 5 obstructions (o5)..."
for seed in "${SEEDS[@]}"; do
    echo "  - Running seed ${seed}..."
    python experiments/run_experiment.py \
        env=dynpushpullhook2d-o5 \
        seed=${seed} \
        hydra.run.dir=./logs/dynpushpullhook2d-o5/seed_${seed}

    if [ $? -eq 0 ]; then
        echo "    ✓ Seed ${seed} completed successfully"
    else
        echo "    ✗ Seed ${seed} failed"
    fi
done

echo ""
echo "=============================================="
echo "All experiments completed!"
