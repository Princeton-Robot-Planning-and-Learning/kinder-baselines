#!/bin/bash
# Script to run bilevel planning on DynPushPullHook2D with multiple random seeds

# Navigate to the kinder-bilevel-planning directory
cd "$(dirname "$0")/.." || exit 1

# Define the seeds to run
SEEDS=(301 302 303 304 305)

echo "Starting bilevel planning DynPushPullHook2D experiments..."
echo "=============================================="

# Run experiments for 0 obstructions
echo ""
echo "Running experiments for 0 obstructions (o0)..."
for seed in "${SEEDS[@]}"; do
    echo "  - Running seed ${seed}..."
    python experiments/run_experiment.py \
        env=dynpushpullhook2d-o0 \
        make_videos=true \
        seed=${seed} \
        hydra.run.dir=./logs/dynpushpullhook2d-o0/seed_${seed}

    if [ $? -eq 0 ]; then
        echo "    ✓ Seed ${seed} completed successfully"
    else
        echo "    ✗ Seed ${seed} failed"
    fi
done

# Run experiments for 1 obstruction
# echo ""
# echo "Running experiments for 1 obstruction (o1)..."
# for seed in "${SEEDS[@]}"; do
#     echo "  - Running seed ${seed}..."
#     python experiments/run_experiment.py \
#         env=dynpushpullhook2d-o1 \
#         make_videos=true \
#         seed=${seed} \
#         hydra.run.dir=./logs/dynpushpullhook2d-o1/seed_${seed}

#     if [ $? -eq 0 ]; then
#         echo "    ✓ Seed ${seed} completed successfully"
#     else
#         echo "    ✗ Seed ${seed} failed"
#     fi
# done

# Run experiments for 5 obstructions
# echo ""
# echo "Running experiments for 5 obstructions (o5)..."
# for seed in "${SEEDS[@]}"; do
#     echo "  - Running seed ${seed}..."
#     python experiments/run_experiment.py \
#         env=dynpushpullhook2d-o5 \
#         seed=${seed} \
#         hydra.run.dir=./logs/dynpushpullhook2d-o5/seed_${seed}
#
#     if [ $? -eq 0 ]; then
#         echo "    ✓ Seed ${seed} completed successfully"
#     else
#         echo "    ✗ Seed ${seed} failed"
#     fi
# done

echo ""
echo "=============================================="
echo "All experiments completed!"
