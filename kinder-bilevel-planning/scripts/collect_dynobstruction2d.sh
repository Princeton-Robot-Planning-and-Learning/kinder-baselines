python experiments/collect_demos_with_skills.py \
    env=dynobstruction2d-o1 \
    seed=302 \
    +num_demos=1 \
    +max_attempts=500 \
    +demo_dir=./skill_demos \
    planning_timeout=60