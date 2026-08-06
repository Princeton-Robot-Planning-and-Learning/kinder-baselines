"""Tests for the tossing oracle policy."""

import kinder
import numpy as np
from conftest import MAKE_VIDEOS  # pylint: disable=import-error
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from relational_structs import ObjectCentricState

from kinder_models.dynamic3d.shelf import parameterized_skills as shelf_skills
from kinder_models.dynamic3d.tossing.oracle_policy import (
    ORACLE_PICK_DISTANCE,
    ORACLE_PICK_ROTATION,
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

    # Run whatever the oracle asks for. The bound is the plan length the domain admits
    # (pick, drive, toss); a policy that re-selected a controller would be caught by the
    # executed-sequence assertion and by the final get_next_controller() call, not by
    # this loop, which merely stops.
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
    # The cube is already down when the toss controller terminates: release happens at
    # 0.46 of the swing profile, leaving over half of it -- longer than the flight -- to
    # run before terminated() is True. No settling loop is needed, but if the profile or
    # the release fraction is retuned, this is the assertion that will start failing.
    assert sim._check_goals()  # pylint: disable=protected-access
    assert policy.get_next_controller(state) is None

    # The oracle hands move_to_throw_pose an explicit 1.35 m standoff rather than
    # letting it sample one from TOSS_TARGET_DISTANCE_BOUNDS = (1.25, 1.45). Nothing
    # clamps or re-samples a supplied parameter, and this pins that: the base really did
    # stop at the requested standoff, not merely somewhere inside the sampler's range.
    # The base does not move again after the drive, so the final state carries the same
    # pose.
    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    target_bin = state.get_object_from_name("bin_0")
    standoff = np.hypot(
        state.get(target_bin, "x") - state.get(robot, "pos_base_x"),
        state.get(target_bin, "y") - state.get(robot, "pos_base_y"),
    )
    assert abs(standoff - ORACLE_THROW_STANDOFF) < WAYPOINT_TOL

    env.close()


def test_oracle_pick_parameters_match_the_sampler():
    """The oracle's grasp literals are PickShelfController's own rng-123 draw.

    The package states their provenance in a docstring and cannot check it, since it
    holds them as constants rather than drawing them. This checks it.
    """
    env = kinder.make("kinder/Tossing3D-o1-v0", scene_bg=False)
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    assert isinstance(state, ObjectCentricState)

    robot = state.get_objects(MujocoTidyBotRobotObjectType)[0]
    cube = state.get_object_from_name("cube_0")
    controllers = shelf_skills.create_lifted_controllers(env.action_space)
    controller = controllers["pick_shelf"].ground((robot, cube))

    params = controller.sample_parameters(state, np.random.default_rng(123))
    assert np.allclose(params, [ORACLE_PICK_DISTANCE, ORACLE_PICK_ROTATION])

    env.close()
