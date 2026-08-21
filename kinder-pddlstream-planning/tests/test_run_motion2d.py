"""Tests for the PDDLStream Motion2D integration."""

import numpy as np
from kinder.envs.kinematic2d.motion2d import (
    ObjectCentricMotion2DEnv,
    TargetRegionType,
)
from kinder.envs.kinematic2d.object_types import CRVRobotType
from kinder.envs.kinematic2d.structs import SE2Pose
from PIL import Image as PILImage
from relational_structs import ObjectCentricState

from kinder_pddlstream_planning.motion2d.run import (
    plan_motion2d,
    solve_and_execute,
    waypoints_to_actions,
)


def _make_close_passage_scenario(
    sim: ObjectCentricMotion2DEnv, seed: int
) -> ObjectCentricState:
    """Reset `sim` with the robot and target close to the passage gap.

    The default placements sit far apart, making the search slow but no stronger.
    """
    state, _ = sim.reset(seed=seed)
    robot = state.get_objects(CRVRobotType)[0]
    target = state.get_objects(TargetRegionType)[0]
    bottom_obstacle = next(o for o in state if o.name == "obstacle0")
    top_obstacle = next(o for o in state if o.name == "obstacle1")

    obstacle_x = state.get(bottom_obstacle, "x")
    obstacle_width = state.get(bottom_obstacle, "width")
    gap_lower_y = state.get(bottom_obstacle, "y") + state.get(bottom_obstacle, "height")
    gap_upper_y = state.get(top_obstacle, "y")
    gap_mid_y = (gap_lower_y + gap_upper_y) / 2

    state.set(robot, "x", obstacle_x - 0.3)
    state.set(robot, "y", gap_mid_y)
    state.set(target, "x", obstacle_x + obstacle_width + 0.05)
    state.set(target, "y", gap_mid_y - 0.1)
    state.set(target, "width", 0.2)
    state.set(target, "height", 0.2)

    sim.reset(options={"init_state": state})
    return state


def test_waypoints_to_actions_respects_step_bounds():
    """Every action lies in the action space and lands on the final pose.

    This hand-spaces a pose plan rather than relying on the planner.
    """
    sim = ObjectCentricMotion2DEnv(num_passages=0)
    sim.reset(seed=0)
    max_dx, max_dy = sim.action_space.high[0], sim.action_space.high[1]
    pose_plan = [
        SE2Pose(0.2, 0.2, 0.0),
        SE2Pose(0.2 + max_dx, 0.2 + max_dy, 0.0),
        SE2Pose(0.2 + 2 * max_dx, 0.2 + 2 * max_dy, 0.0),
    ]
    actions = waypoints_to_actions(sim, pose_plan)
    assert len(actions) == len(pose_plan) - 1
    for action in actions:
        assert sim.action_space.contains(action)
    total_dxdy = np.sum(np.array(actions)[:, :2], axis=0)
    expected = [pose_plan[-1].x - pose_plan[0].x, pose_plan[-1].y - pose_plan[0].y]
    np.testing.assert_allclose(total_dxdy, expected, atol=1e-4)


def test_solve_and_execute_open_field():
    """With no obstacles, the target is reached almost instantly."""
    assert solve_and_execute(num_passages=0, seed=0, max_time=15.0)


def test_solve_and_execute_saves_gif(tmp_path):
    """`gif_path` produces a multi-frame GIF of the rollout."""
    gif_path = tmp_path / "motion2d.gif"
    assert solve_and_execute(
        num_passages=0,
        seed=0,
        max_time=15.0,
        gif_path=gif_path,
    )
    assert gif_path.exists()
    with PILImage.open(gif_path) as gif:
        assert gif.n_frames > 1


def test_solve_and_execute_narrow_passage():
    """A single RRT-Connect `connect` call must thread one narrow passage."""
    sim = ObjectCentricMotion2DEnv(num_passages=1)
    state = _make_close_passage_scenario(sim, seed=0)
    pose_plan = plan_motion2d(sim, state, max_time=30.0)
    assert pose_plan is not None
    reached_target = False
    for action in waypoints_to_actions(sim, pose_plan):
        _, _, terminated, _, _ = sim.step(action)
        if terminated:
            reached_target = True
            break
    assert reached_target


def test_solve_and_execute_multiple_passages():
    """Regression test: several passages with the default placements.

    The straight-line `connect` this replaced failed almost always here.
    """
    assert solve_and_execute(num_passages=3, seed=0, max_time=30.0)
