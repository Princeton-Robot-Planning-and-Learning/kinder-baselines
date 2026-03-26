"""State abstractions for the Sweep3D environment."""

import numpy as np
from bilevel_planning.structs import (
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from kinder.envs.dynamic3d.object_types import (
    MujocoDrawerObjectType,
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoTidyBotRobotObjectType,
)
from kinder.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv
from relational_structs import (
    GroundAtom,
    ObjectCentricState,
    Predicate,
)

from kinder_models.dynamic3d.ground.parameterized_skills import PyBulletSim

# Predicates.
DrawerOpen = Predicate("DrawerOpen", [MujocoDrawerObjectType])
DrawerClosed = Predicate("DrawerClosed", [MujocoDrawerObjectType])
InDrawer = Predicate("InDrawer", [MujocoMovableObjectType, MujocoDrawerObjectType])
OnTable = Predicate("OnTable", [MujocoMovableObjectType])
Holding = Predicate("Holding", [MujocoTidyBotRobotObjectType, MujocoMovableObjectType])
HandEmpty = Predicate("HandEmpty", [MujocoTidyBotRobotObjectType])


class Sweep3DStateAbstractor:
    """State abstractor for the Sweep3D environment."""

    def __init__(self, sim: ObjectCentricTidyBot3DEnv) -> None:
        """Initialize the state abstractor."""
        initial_state, _ = sim.reset()  # just need to access the objects
        self._pybullet_sim = PyBulletSim(initial_state, rendering=False)
        self._robot_name = sim.robot_name

    def state_abstractor(self, state: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        atoms: set[GroundAtom] = set()

        # Sync the pybullet simulator.
        self._pybullet_sim.set_state(state)

        # Uncomment to debug.
        # from pybullet_helpers.camera import capture_image
        # img = capture_image(
        #     self._pybullet_sim.physics_client_id,
        #     image_width=512,
        #     image_height=512,
        #     camera_yaw=90,
        #     camera_distance=2.5,
        #     camera_pitch=-20,
        #     camera_target=(0, 0, 0),
        # )
        # import imageio.v2 as iio
        # iio.imsave("pybullet_sim.png", img)
        # import ipdb; ipdb.set_trace()

        # Extract the relevant objects.
        robot = state.get_object_from_name(self._robot_name)
        fixtures = state.get_objects(MujocoFixtureObjectType)
        movables = state.get_objects(MujocoMovableObjectType)
        drawers = state.get_objects(MujocoDrawerObjectType)
        all_mujoco_objects = set(fixtures) | set(movables) | set(drawers)

        # drawer open
        open_drawer = None
        for drawer in drawers:
            if drawer.name == "kitchen_island_drawer_s1c1":
                if state.get(drawer, "pos") > 0.04:
                    atoms.add(GroundAtom(DrawerOpen, [drawer]))
                    open_drawer = drawer
                else:
                    atoms.add(GroundAtom(DrawerClosed, [drawer]))
        # OnTable.
        GraspThreshold = 0.1
        for target in movables:
            z = state.get(target, "z")
            if z > 0.45:
                if target.name == "wiper_0":
                    if state.get(robot, "pos_gripper") > GraspThreshold:
                        continue
                atoms.add(GroundAtom(OnTable, [target]))

        # HandEmpty.
        handempty_tol = 1e-3
        gripper_val = state.get(robot, "pos_gripper")
        if np.isclose(gripper_val, 0.0, atol=handempty_tol):
            atoms.add(GroundAtom(HandEmpty, [robot]))

        # Holding.
        # checking the ee pose and target pose.

        gripper_val = state.get(robot, "pos_gripper")
        if gripper_val > GraspThreshold:
            for target in movables:
                if target.name == "wiper_0":
                    atoms.add(GroundAtom(Holding, [robot, target]))

        # InDrawer.
        if open_drawer is not None:
            for movable in movables:
                if state.get(movable, "z") < 0.4:
                    atoms.add(GroundAtom(InDrawer, [movable, open_drawer]))

        objects = {robot} | all_mujoco_objects
        return RelationalAbstractState(atoms, objects)

    # def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
    #     """The goal is to sweep the target into the drawer."""
    #     wiper = state.get_object_from_name("wiper_0")
    #     cubes = state.get_objects(MujocoMovableObjectType)
    #     drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
    #     robot = state.get_object_from_name(self._robot_name)
    #     atoms = {
    #         GroundAtom(Holding, [robot, wiper]),
    #         GroundAtom(DrawerOpen, [drawer]),
    #     }
    #     for cube in cubes:
    #         atoms.add(GroundAtom(InDrawer, [cube, drawer]))
    #     return RelationalAbstractGoal(atoms, self.state_abstractor)

    def goal_deriver(self, state: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to sweep the target into the drawer."""
        wiper = state.get_object_from_name("wiper_0")
        cubes = state.get_objects(MujocoMovableObjectType)
        drawer = state.get_object_from_name("kitchen_island_drawer_s1c1")
        robot = state.get_object_from_name(self._robot_name)
        atoms = {
            GroundAtom(HandEmpty, [robot]),
            GroundAtom(DrawerOpen, [drawer]),
            GroundAtom(OnTable, [wiper]),
        }
        for cube in cubes:
            atoms.add(GroundAtom(OnTable, [cube]))
        return RelationalAbstractGoal(atoms, self.state_abstractor)
