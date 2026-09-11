"""Tossing planning geometry, cube bindings and physics-state restoration."""

import kinder
import numpy as np
import pybullet as p
from bilevel_planning.structs import RelationalAbstractGoal, RelationalAbstractState
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv, TidyBot3DEnv
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
)
from kinder.envs.dynamic3d.objects.fixtures import FixedCuboid
from kinder.envs.dynamic3d.objects.primitive_objects import Cuboid
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    MoveToTossLocationAndTossController,
)
from kinder_models.dynamic3d.tossing.state_abstractions import (
    HandEmpty,
    Holding,
    MovableInGoalRegion,
    MovableIsDownX,
    OnGround,
)
from kinder_models.dynamic3d.utils import (
    PyBulletSim,
    get_overhead_kinematic2ds,
    get_overhead_object_se2_pose,
    get_target_robot_pose_from_parameters,
)
from relational_structs import GroundAtom
from relational_structs.spaces import ObjectCentricBoxSpace
from tomsgeoms2d.structs import Rectangle

from kinder_bilevel_planning.env_models import create_bilevel_planning_models


def test_fixed_tossing_obstacles_match_both_planning_scenes(monkeypatch) -> None:
    """Fixture typing, arm collision boxes and base footprints all use real geometry."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array")
    captured = []
    original_init = PyBulletSim.__init__

    def capture(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        captured.append(self)

    monkeypatch.setattr(PyBulletSim, "__init__", capture)
    try:
        obs, _ = env.reset(seed=0)
        assert isinstance(env.observation_space, ObjectCentricBoxSpace)
        assert isinstance(env.unwrapped, TidyBot3DEnv)
        state = env.observation_space.devectorize(obs)
        models = create_bilevel_planning_models(
            "tidybot3d_tossing3D",
            env.observation_space,
            env.action_space,
            num_objects=1,
        )
        planner = captured[-1]
        sim = env.unwrapped._object_centric_env  # pylint: disable=protected-access
        assert isinstance(sim, ObjectCentricTidyBot3DEnv)
        fixtures = sim._fixtures_dict  # pylint: disable=protected-access
        bin_obj = state.get_object_from_name("bin_0")
        barrier = state.get_object_from_name("cuboid_barrier")
        cube = state.get_object_from_name("cube_0")
        assert bin_obj.is_instance(MujocoMovableObjectType)
        assert barrier.is_instance(MujocoFixtureObjectType)
        assert (
            GroundAtom(MovableIsDownX, [cube, barrier])
            in models.state_abstractor(state).atoms
        )
        geoms = get_overhead_kinematic2ds(state, planner.bounding_boxes)
        planner.set_state(state)
        boxes = planner._boxes  # pylint: disable=protected-access
        assert len(boxes) == 1
        assert len(planner._static_boxes) == 6  # pylint: disable=protected-access
        assert len(planner.static_base_obstacles) == 6
        assert set(boxes.values()) <= planner.get_collision_bodies()
        for name, fixture in fixtures.items():
            if not isinstance(fixture, FixedCuboid) or not isinstance(
                fixture.primitive, Cuboid
            ):
                continue
            dims = fixture.primitive.get_bounding_box_dimensions()
            assert planner.bounding_boxes[name] == dims
            geom = geoms[name]
            assert isinstance(geom, Rectangle)
            assert np.allclose([geom.width, geom.height], dims[:2])
            low, high = p.getAABB(
                boxes[name], physicsClientId=planner.physics_client_id
            )
            assert np.allclose(np.subtract(high, low), dims, atol=1e-7)
        assert planner.has_bin("bin_0")
        robot = state.get_object_from_name("robot")
        controller = MoveToTossLocationAndTossController(
            [robot, cube, barrier], pybullet_sim=planner
        )
        for _ in range(20):
            params = controller.sample_parameters(state, np.random.default_rng(_))
            target = get_target_robot_pose_from_parameters(
                get_overhead_object_se2_pose(state, bin_obj), params[0], params[1]
            )
            assert target.x + 0.275 < state.get(barrier, "x") - 0.03

    finally:
        for planner in captured:
            planner.close()
        env.close()  # type: ignore[no-untyped-call]


def test_tossing_two_cubes_have_independent_operator_effects() -> None:
    """One completed toss must not satisfy or delete the other cube's facts."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Tossing3D-o2-v0", scene_bg=False)
    assert isinstance(env.unwrapped, TidyBot3DEnv)
    try:
        obs, _ = env.reset(seed=3)
        models = create_bilevel_planning_models(
            "tidybot3d_tossing3D",
            env.observation_space,
            env.action_space,
            num_objects=2,
        )
        state = models.observation_to_state(obs)
        robot = state.get_object_from_name("robot")
        barrier = state.get_object_from_name("cuboid_barrier")
        cubes = [state.get_object_from_name(f"cube_{i}") for i in range(2)]
        goal = models.goal_deriver(state)
        assert isinstance(goal, RelationalAbstractGoal)
        assert goal.atoms == {GroundAtom(MovableInGoalRegion, [c]) for c in cubes}
        abstract = models.state_abstractor(state)
        atoms = set(abstract.atoms)
        skills = {s.operator.name: s for s in models.skills}
        for index, cube in enumerate(cubes):
            pick = skills["pick_cube"].ground((robot, cube, barrier))
            toss = skills["move_to_toss_location_and_toss"].ground(
                (robot, cube, barrier)
            )
            assert pick.controller.objects[1] == cube
            assert toss.controller.objects[1] == cube
            assert pick.operator.preconditions <= atoms
            atoms.difference_update(pick.operator.delete_effects)
            atoms.update(pick.operator.add_effects)
            assert GroundAtom(Holding, [robot, cube]) in atoms
            assert toss.operator.preconditions <= atoms
            atoms.difference_update(toss.operator.delete_effects)
            atoms.update(toss.operator.add_effects)
            assert GroundAtom(HandEmpty, [robot]) in atoms
            assert GroundAtom(MovableInGoalRegion, [cube]) in atoms
            if index == 0:
                other = cubes[1]
                assert GroundAtom(OnGround, [other]) in atoms
                assert GroundAtom(MovableIsDownX, [other, barrier]) in atoms
                assert GroundAtom(MovableInGoalRegion, [other]) not in atoms
                assert not goal.check_abstract_state(
                    RelationalAbstractState(atoms, abstract.objects)
                )
        assert goal.check_abstract_state(
            RelationalAbstractState(atoms, abstract.objects)
        )
    finally:
        env.close()  # type: ignore[no-untyped-call]


def test_tossing_transition_restores_hidden_gripper_state() -> None:
    """A failed close-gripper branch cannot contaminate replay of an open branch."""
    kinder.register_all_environments()
    env = kinder.make("kinder/Tossing3D-o2-v0", scene_bg=False)
    assert isinstance(env.unwrapped, TidyBot3DEnv)
    try:
        obs, _ = env.reset(seed=3)
        models = create_bilevel_planning_models(
            "tidybot3d_tossing3D",
            env.observation_space,
            env.action_space,
            num_objects=2,
        )
        start = models.observation_to_state(obs)
        close_action = np.zeros(18, dtype=np.float32)
        close_action[10] = 1.0
        open_action = np.zeros(18, dtype=np.float32)
        expected = models.transition_fn(start, open_action)
        branch = start
        for _ in range(10):
            branch = models.transition_fn(branch, close_action)
        replayed = models.transition_fn(start.copy(), open_action)
        for obj in expected:
            np.testing.assert_array_equal(expected[obj], replayed[obj])
    finally:
        env.close()  # type: ignore[no-untyped-call]
