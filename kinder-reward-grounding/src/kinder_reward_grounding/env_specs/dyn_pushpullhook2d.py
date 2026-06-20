"""Oracle RewardSpec for DynPushPullHook2D."""

from __future__ import annotations

from kinder_reward_grounding.specs import (
    ObjectRef,
    PredicateSpec,
    ProgressMetricSpec,
    RewardSpec,
    SubgoalSpec,
)


def make_dyn_pushpullhook2d_reward_spec(
    env_id: str = "kinder/DynPushPullHook2D-o0-v0",
) -> RewardSpec:
    """Create a hand-written oracle RewardSpec for DynPushPullHook2D."""
    return RewardSpec(
        env_id=env_id,
        objects=(
            ObjectRef(role="robot", name="robot", type_name="kin_robot"),
            ObjectRef(role="tool", name="hook", type_name="hook"),
            ObjectRef(role="target", name="target_block", type_name="target_block"),
            ObjectRef(role="goal", name="middle_wall", type_name="kin_rectangle"),
        ),
        subgoals=(
            SubgoalSpec(
                name="grasp_hook",
                completion=PredicateSpec(
                    name="held",
                    objects={"obj": "tool"},
                    backend="oracle_object",
                    params={"threshold": 0.5},
                ),
                progress=ProgressMetricSpec(
                    name="distance_progress",
                    objects={"a": "robot", "b": "tool"},
                    backend="oracle_object",
                ),
                weight=2.0,
                completion_bonus=5.0,
            ),
            SubgoalSpec(
                name="bring_hook_to_target",
                completion=PredicateSpec(
                    name="keypoint_near",
                    objects={"source": "tool", "target": "target"},
                    backend="oracle_object",
                    params={"threshold": 0.12, "keypoint_type": "hook_heuristic"},
                ),
                progress=ProgressMetricSpec(
                    name="keypoint_distance_progress",
                    objects={"source": "tool", "target": "target"},
                    backend="oracle_object",
                    params={"keypoint_type": "hook_heuristic"},
                ),
                weight=3.0,
                completion_bonus=0.5,
            ),
            SubgoalSpec(
                name="move_target_to_wall",
                completion=PredicateSpec(
                    name="vertical_gap_below",
                    objects={"moving": "target", "target": "goal"},
                    backend="oracle_object",
                    params={"threshold": 0.05},
                ),
                progress=ProgressMetricSpec(
                    name="vertical_gap_progress",
                    objects={"moving": "target", "target": "goal"},
                    backend="oracle_object",
                ),
                weight=30.0,
                completion_bonus=50.0,
            ),
        ),
        composer="sequential",
        time_penalty=-0.01,
        success_bonus=50.0,
    )
