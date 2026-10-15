"""Regression checks for MPC isolation and persistent grasp frames."""

# Fixtures shadow names; tests inspect private rollout state.
# pylint: disable=redefined-outer-name,protected-access

import numpy as np
import pybullet as p
import pytest

from kinder_pddlstream_planning.limbrepositioning3d.run import (
    build_stream_context,
    create_env,
    reset_to_start,
)
from kinder_pddlstream_planning.limbrepositioning3d.stream import (
    LimbConf,
    MPCConfig,
    NoFeasibleMotion,
    PredictiveSamplingMPC,
    plan_grasp_motion,
    plan_limb_motion,
    sample_grasp,
)
from kinder_pddlstream_planning.limbrepositioning3d.utils import (
    advance_corrected,
    capture_state,
    is_grasping,
    release_grasp,
    restore_state,
    robot_motion_violation,
    saved_sim_state,
)


@pytest.fixture(
    params=("isolated-right-arm", "isolated-left-leg", "isolated-right-leg")
)
def grasped(request):
    """A deterministic grasped state with an accepted approach."""
    sim = create_env(request.param, muscle_tone="none")
    try:
        sim.reset(seed=0)
        ctx = build_stream_context(
            sim,
            motion_seed=0,
            mpc=MPCConfig(
                num_rollouts=4,
                horizon=2,
                num_control_points=2,
                action_repeat=2,
                directed_gains=(),
                feedback_gains=(),
            ),
        )
        reset_to_start(ctx)
        scene = sim.config.scene
        sampled = next(sample_grasp(ctx, "limb"), None)
        assert sampled is not None
        grasp = sampled[0]
        candidates = plan_grasp_motion(
            ctx,
            "limb",
            grasp,
            LimbConf(tuple(scene.limb_init_joint_positions)),
            LimbConf(tuple(scene.limb_goal_joint_positions)),
        )
        candidate = next(candidates, None)
        assert candidate is not None
        _, _, initial = candidate
        candidates.close()
        restore_state(sim, initial)
        yield sim, ctx, capture_state(sim)
    finally:
        sim.close()


def test_candidate_search_does_not_change_committed_dynamics(grasped):
    """Discarded rollouts must not change committed dynamics."""
    sim, ctx, initial = grasped
    reference = []
    for _ in range(20):
        for _ in range(12):
            advance_corrected(sim, [0.1] * 7)
        reference.append(capture_state(sim))

    restore_state(sim, initial)
    goal = np.asarray(sim.config.scene.limb_goal_joint_positions)
    with PredictiveSamplingMPC(ctx, goal, initial) as mpc:
        for expected in reference:
            before = capture_state(sim)
            mpc.step()  # Recommendation unused; commands stay identical.
            after = capture_state(sim)
            for field in (
                "robot_positions",
                "robot_velocities",
                "limb_positions",
                "limb_velocities",
            ):
                np.testing.assert_array_equal(
                    getattr(before, field), getattr(after, field)
                )
            for _ in range(12):
                advance_corrected(sim, [0.1] * 7)
            actual = capture_state(sim)
            for field in (
                "robot_positions",
                "robot_velocities",
                "limb_positions",
                "limb_velocities",
            ):
                np.testing.assert_allclose(
                    getattr(actual, field), getattr(expected, field), atol=1e-9, rtol=0
                )
        rollout_client = mpc._rollout_sim.physics_client_id
    assert not p.isConnected(physicsClientId=rollout_client)


def test_snapshot_preserves_loaded_grasp_frame(grasped):
    """Restoring a loaded grasp keeps its original weld frames."""
    sim, _, _ = grasped
    for _ in range(120):
        advance_corrected(sim, [0.2] * 7)
    state = capture_state(sim)
    frame = state.grasp_frame
    assert frame is not None
    release_grasp(sim)
    restore_state(sim, state)
    restored = capture_state(sim).grasp_frame
    assert restored is not None
    assert restored.parent == frame.parent
    assert restored.child == frame.child
    assert restored.max_force == frame.max_force


def test_saved_ungrasped_state_removes_new_weld(grasped):
    """A temporary grasp must not survive restoring an ungrasped state."""
    sim, _, initial = grasped
    release_grasp(sim)
    with saved_sim_state(sim):
        restore_state(sim, initial)
        assert is_grasping(sim)
    assert not is_grasping(sim)


def test_motion_attempt_budget_returns_control_to_search(grasped):
    """A spent attempt budget yields no trajectory and records its reason."""
    sim, ctx, initial = grasped
    ctx.mpc.max_motion_time = 0.0
    goal = LimbConf(tuple(sim.config.scene.limb_goal_joint_positions))
    assert not list(plan_limb_motion(ctx, initial, goal))
    assert ctx.motion_outcomes["motion attempt exceeded 0 s"] == 1
    np.testing.assert_array_equal(
        sim.limb.get_joint_positions(), initial.limb_positions
    )


def test_all_invalid_candidates_are_rejected(grasped, monkeypatch):
    """MPC must not commit the least bad candidate when every one is invalid."""
    sim, ctx, initial = grasped
    goal = np.asarray(sim.config.scene.limb_goal_joint_positions)
    with PredictiveSamplingMPC(ctx, goal, initial) as mpc:
        monkeypatch.setattr(mpc, "_score", lambda *_args: (1, 0.0))
        with pytest.raises(NoFeasibleMotion):
            mpc.step()
    np.testing.assert_array_equal(
        sim.limb.get_joint_positions(), initial.limb_positions
    )


def test_robot_velocity_limit_is_checked(grasped):
    """The robot speed guard detects an over-limit joint velocity."""
    sim, ctx, initial = grasped
    velocities = [0.0] * len(initial.robot_positions)
    velocities[0] = ctx.mpc.max_robot_velocity + 0.1
    sim.robot.arm.set_joints(initial.robot_positions, joint_velocities=velocities)
    assert robot_motion_violation(ctx) == "robot velocity limit"


def test_rollout_preserves_custom_belief_constraints(grasped):
    """The rollout context keeps patched belief checks and true limits."""
    sim, ctx, initial = grasped
    checked = []

    def reject(positions):
        checked.append(positions)
        return False

    ctx.in_believed_limits = reject  # type: ignore[method-assign]
    ctx.true_joint_limits = ([-0.01] * 6, [0.01] * 6)
    goal = np.asarray(sim.config.scene.limb_goal_joint_positions)
    with PredictiveSamplingMPC(ctx, goal, initial) as mpc:
        invalid, _ = mpc._score(initial, np.zeros((2, 7)))
        assert invalid
        assert checked
        assert mpc._rollout_ctx.true_joint_limits == ctx.true_joint_limits
        assert mpc._rollout_ctx.sim is not ctx.sim
