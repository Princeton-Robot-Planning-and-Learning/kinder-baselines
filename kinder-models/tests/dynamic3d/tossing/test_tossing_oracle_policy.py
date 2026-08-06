"""Tests for the tossing oracle policy."""

import kinder
import numpy as np
from conftest import MAKE_VIDEOS  # pylint: disable=import-error
from gymnasium.wrappers import RecordVideo
from relational_structs import ObjectCentricState

from kinder_models.dynamic3d.tossing.oracle_policy import (
    ORACLE_THROW_STANDOFF,
    Tossing3DOraclePolicy,
)
from kinder_models.dynamic3d.utils import WAYPOINT_TOL

kinder.register_all_environments()


def test_tossing3d_oracle_policy():
    """The oracle solves Tossing3D-o1 end to end.

    Three controllers, chosen from the abstract state alone, put the cube in the goal
    region -- which is the environment's own success criterion, so this asserts
    _check_goals() rather than a hand-written box test.
    """

    # Create the environment.
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array", scene_bg=False)
    if MAKE_VIDEOS:
        env.unwrapped._object_centric_env.set_render_camera(  # type: ignore # pylint: disable=protected-access
            "task_view"
        )
        env = RecordVideo(env, "unit_test_videos", name_prefix="Tossing3D-oracle")
    sim = env.unwrapped._object_centric_env  # pylint: disable=protected-access

    # The policy builds its own abstractor and controllers from the environment.
    policy = Tossing3DOraclePolicy(sim, env.action_space)

    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    assert isinstance(state, ObjectCentricState)
    assert not sim._check_goals()  # pylint: disable=protected-access

    # Run whatever the oracle asks for until it says the goal is reached. The bound is
    # the plan length the domain admits (pick, drive, toss) and not a search budget:
    # exceeding it means the oracle re-selected a controller, which is a failure.
    executed = []
    for _ in range(3):
        selection = policy.get_next_controller(state)
        if selection is None:
            break
        name, controller, params = selection
        executed.append(name)
        controller.reset(state, params)
        for _ in range(400):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(obs)
            controller.observe(state)
            if controller.terminated():
                break
        else:
            assert False, f"Controller {name} did not terminate"

    assert executed == ["pick_shelf", "move_to_throw_pose", "toss_from_windup"]
    assert sim._check_goals()  # pylint: disable=protected-access
    assert policy.get_next_controller(state) is None

    # The oracle hands move_to_throw_pose an explicit 1.35 m standoff, which is outside
    # the MOVE_TO_TARGET_DISTANCE_BOUNDS that controller *samples* from. A supplied
    # parameter is not a sampled one, and this pins that: the base really did stop at
    # the requested standoff rather than at a clamped 0.5-0.6 m. The base does not move
    # again after the drive, so the final state carries the same pose.
    robot = policy._get_robot(state)  # pylint: disable=protected-access
    target_bin = state.get_object_from_name("bin_0")
    standoff = np.hypot(
        state.get(target_bin, "x") - state.get(robot, "pos_base_x"),
        state.get(target_bin, "y") - state.get(robot, "pos_base_y"),
    )
    assert abs(standoff - ORACLE_THROW_STANDOFF) < WAYPOINT_TOL

    env.close()
