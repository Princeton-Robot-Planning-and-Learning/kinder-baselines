"""Oracle RewardSpec for Motion2D."""

from __future__ import annotations

from kinder_reward_grounding.specs import (
    ObjectRef,
    PredicateSpec,
    ProgressMetricSpec,
    RewardSpec,
    SubgoalSpec,
)


def make_motion2d_reward_spec(
    env_id: str = "kinder/Motion2D-p5-v0",
) -> RewardSpec:
    """Create a hand-written oracle RewardSpec for Motion2D."""
    return RewardSpec(
        env_id=env_id,
        objects=(
            ObjectRef(role="robot", name="robot", type_name="crv_robot"),
            ObjectRef(role="goal", name="target_region", type_name="target_region"),
            ObjectRef(
                role="passage_1_bottom",
                name="obstacle0",
                type_name="rectangle",
            ),
            ObjectRef(role="passage_1_top", name="obstacle1", type_name="rectangle"),
            ObjectRef(
                role="passage_2_bottom",
                name="obstacle2",
                type_name="rectangle",
            ),
            ObjectRef(role="passage_2_top", name="obstacle3", type_name="rectangle"),
            ObjectRef(
                role="passage_3_bottom",
                name="obstacle4",
                type_name="rectangle",
            ),
            ObjectRef(role="passage_3_top", name="obstacle5", type_name="rectangle"),
            ObjectRef(
                role="passage_4_bottom",
                name="obstacle6",
                type_name="rectangle",
            ),
            ObjectRef(role="passage_4_top", name="obstacle7", type_name="rectangle"),
            ObjectRef(
                role="passage_5_bottom",
                name="obstacle8",
                type_name="rectangle",
            ),
            ObjectRef(role="passage_5_top", name="obstacle9", type_name="rectangle"),
        ),
        subgoals=(
            SubgoalSpec(
                name="reach_passage_1",
                completion=PredicateSpec(
                    name="near_passage",
                    objects={
                        "robot": "robot",
                        "bottom": "passage_1_bottom",
                        "top": "passage_1_top",
                    },
                    backend="oracle_object",
                    params={"threshold": 0.16},
                ),
                progress=ProgressMetricSpec(
                    name="passage_progress",
                    objects={
                        "robot": "robot",
                        "bottom": "passage_1_bottom",
                        "top": "passage_1_top",
                    },
                    backend="oracle_object",
                ),
                weight=8.0,
                completion_bonus=2.0,
            ),
            SubgoalSpec(
                name="reach_passage_2",
                completion=PredicateSpec(
                    name="near_passage",
                    objects={
                        "robot": "robot",
                        "bottom": "passage_2_bottom",
                        "top": "passage_2_top",
                    },
                    backend="oracle_object",
                    params={"threshold": 0.16},
                ),
                progress=ProgressMetricSpec(
                    name="passage_progress",
                    objects={
                        "robot": "robot",
                        "bottom": "passage_2_bottom",
                        "top": "passage_2_top",
                    },
                    backend="oracle_object",
                ),
                weight=8.0,
                completion_bonus=2.0,
            ),
            SubgoalSpec(
                name="reach_passage_3",
                completion=PredicateSpec(
                    name="near_passage",
                    objects={
                        "robot": "robot",
                        "bottom": "passage_3_bottom",
                        "top": "passage_3_top",
                    },
                    backend="oracle_object",
                    params={"threshold": 0.16},
                ),
                progress=ProgressMetricSpec(
                    name="passage_progress",
                    objects={
                        "robot": "robot",
                        "bottom": "passage_3_bottom",
                        "top": "passage_3_top",
                    },
                    backend="oracle_object",
                ),
                weight=8.0,
                completion_bonus=2.0,
            ),
            SubgoalSpec(
                name="reach_passage_4",
                completion=PredicateSpec(
                    name="near_passage",
                    objects={
                        "robot": "robot",
                        "bottom": "passage_4_bottom",
                        "top": "passage_4_top",
                    },
                    backend="oracle_object",
                    params={"threshold": 0.16},
                ),
                progress=ProgressMetricSpec(
                    name="passage_progress",
                    objects={
                        "robot": "robot",
                        "bottom": "passage_4_bottom",
                        "top": "passage_4_top",
                    },
                    backend="oracle_object",
                ),
                weight=8.0,
                completion_bonus=2.0,
            ),
            SubgoalSpec(
                name="reach_passage_5",
                completion=PredicateSpec(
                    name="near_passage",
                    objects={
                        "robot": "robot",
                        "bottom": "passage_5_bottom",
                        "top": "passage_5_top",
                    },
                    backend="oracle_object",
                    params={"threshold": 0.16},
                ),
                progress=ProgressMetricSpec(
                    name="passage_progress",
                    objects={
                        "robot": "robot",
                        "bottom": "passage_5_bottom",
                        "top": "passage_5_top",
                    },
                    backend="oracle_object",
                ),
                weight=8.0,
                completion_bonus=2.0,
            ),
            SubgoalSpec(
                name="reach_goal",
                completion=PredicateSpec(
                    name="near",
                    objects={"a": "robot", "b": "goal"},
                    backend="oracle_object",
                    params={"threshold": 0.18},
                ),
                progress=ProgressMetricSpec(
                    name="distance_progress",
                    objects={"a": "robot", "b": "goal"},
                    backend="oracle_object",
                ),
                weight=5.0,
                completion_bonus=10.0,
            ),
        ),
        composer="sequential",
        time_penalty=-0.01,
        success_bonus=50.0,
    )
