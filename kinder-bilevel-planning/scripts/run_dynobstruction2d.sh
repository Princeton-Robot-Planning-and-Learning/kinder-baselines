#!/bin/bash
# Script to run bilevel planning on DynObstruction2D with multiple random seeds

# Navigate to the kinder-bilevel-planning directory
cd "$(dirname "$0")/.." || exit 1

# Define the seeds to run
SEEDS=(301 302 303 304 305)

echo "Starting bilevel planning DynObstruction2D experiments..."
echo "=============================================="

# Run experiments for 1 obstacle
echo ""
echo "Running experiments for 1 obstacle (o1)..."
for seed in "${SEEDS[@]}"; do
    echo "  - Running seed ${seed}..."
    python experiments/run_experiment.py \
        env=dynobstruction2d-o1 \
        make_videos=false \
        seed=${seed} \
        hydra.run.dir=./logs/dynobstruction2d-o1/seed_${seed}

    if [ $? -eq 0 ]; then
        echo "    ✓ Seed ${seed} completed successfully"
    else
        echo "    ✗ Seed ${seed} failed"
    fi
done

# Run experiments for 2 obstacles
# echo ""
# echo "Running experiments for 2 obstacles (o2)..."
# for seed in "${SEEDS[@]}"; do
#     echo "  - Running seed ${seed}..."
#     python experiments/run_experiment.py \
#         env=dynobstruction2d-o2 \
#         seed=${seed} \
#         hydra.run.dir=./logs/dynobstruction2d-o2/seed_${seed}

#     if [ $? -eq 0 ]; then
#         echo "    ✓ Seed ${seed} completed successfully"
#     else
#         echo "    ✗ Seed ${seed} failed"
#     fi
# done

# # Run experiments for 3 obstacles
# echo ""
# echo "Running experiments for 3 obstacles (o3)..."
# for seed in "${SEEDS[@]}"; do
#     echo "  - Running seed ${seed}..."
#     python experiments/run_experiment.py \
#         env=dynobstruction2d-o3 \
#         seed=${seed} \
#         hydra.run.dir=./logs/dynobstruction2d-o3/seed_${seed}

#     if [ $? -eq 0 ]; then
#         echo "    ✓ Seed ${seed} completed successfully"
#     else
#         echo "    ✗ Seed ${seed} failed"
#     fi
# done

echo ""
echo "=============================================="
echo "All experiments completed!"
