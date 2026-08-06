"""Tests for tossing state_abstractions.py."""

import kinder
import numpy as np
from conftest import MAKE_VIDEOS  # pylint: disable=import-error
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from relational_structs import GroundAtom, ObjectCentricState

from kinder_models.dynamic3d.tossing.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.dynamic3d.tossing.state_abstractions import (
    InGoalRegion,
    Tossing3DStateAbstractor,
)


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def test_tossing3d_state_abstraction():
    """Tests for Tossing3DStateAbstractor()."""
    kinder.register_all_environments()
    num_objects = 1
    env = kinder.make(f"kinder/Tossing3D-o{num_objects}-v0", render_mode="rgb_array")
    if MAKE_VIDEOS:
        env.unwrapped._object_centric_env.set_render_camera("task_view")  # type: ignore # pylint: disable=protected-access
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix="Tossing3D-state-abstraction",
        )
    sim = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    abstractor = Tossing3DStateAbstractor(sim)

    # Check the state abstraction in the initial state. The robot's hand should be
    # empty, the cube should be on the ground on the robot's side of the barrier, and
    # the cube should not yet be in the goal region.
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    assert isinstance(state, ObjectCentricState)
    abstract_state = abstractor.state_abstractor(state)
    robot = _get_robot_from_state(state)
    assert str(sorted(abstract_state.atoms)) == (
        f"[(HandEmpty {robot.name}), "
        f"(OnGround cube_0), "
        f"(Reachable cube_0 cuboid_barrier)]"
    )

    # The goal is for the cube to land in the goal region, which is exactly the
    # environment's own success criterion.
    cube = state.get_object_from_name("cube_0")
    goal = abstractor.goal_deriver(state)
    assert goal.atoms == {GroundAtom(InGoalRegion, [cube])}
    assert not goal.check_state(state)
    assert not sim._check_goals()  # pylint: disable=protected-access

    # InGoalRegion is checked against the environment's own region object, which is the
    # task JSON's "ranges" entry inflated by the ground placement threshold (0.05 m) in
    # every direction. Putting the cube just outside the literal 2.10 but inside the
    # inflated 2.15 is what tells the two apart: a predicate written against the file
    # would score this real success as a failure.
    landed_state = state.copy()
    landed_state.set(cube, "x", 2.14)
    landed_state.set(cube, "y", 0.0)
    landed_state.set(cube, "z", 0.025)
    landed_atoms = abstractor.state_abstractor(landed_state).atoms
    assert GroundAtom(InGoalRegion, [cube]) in landed_atoms

    # Drive the base to a throw standoff from the bin. NearBin is the one predicate
    # here with no upstream precedent, so it is checked against the controller that is
    # supposed to establish it rather than only against the initial state.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    target_bin = state.get_object_from_name("bin_0")
    object_parameters = (robot, target_bin)
    controller = lifted_controller.ground(object_parameters)
    throw_standoff = 1.35
    controller.reset(state, np.array([throw_standoff, 0.0]))
    for _ in range(300):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # The robot is now standing where it can throw from, and nothing else has changed:
    # the cube is untouched and still on the robot's side of the barrier.
    abstract_state = abstractor.state_abstractor(state)
    assert str(sorted(abstract_state.atoms)) == (
        f"[(HandEmpty {robot.name}), "
        f"(NearBin {robot.name} bin_0), "
        f"(OnGround cube_0), "
        f"(Reachable cube_0 cuboid_barrier)]"
    )

    env.close()
