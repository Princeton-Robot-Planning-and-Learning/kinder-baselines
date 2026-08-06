"""A privileged solver for Tossing3D: pick the cube, drive to a throw pose, toss.

The policy is a three-way branch on the abstract state, because the domain admits
exactly one plan shape. What makes it an oracle rather than a fixed script is the
continuous parameters -- the part a learner would have to find -- and every one of them
is a value this repository already publishes:

- ORACLE_PICK_DISTANCE and ORACLE_PICK_ROTATION are the pair that
  PickShelfController.sample_parameters draws from np.random.default_rng(123), the rng
  test_pick_ground_toss constructs to parameterize this exact grasp. With one cube in
  the scene the sampler's rejection loop has nothing to reject against, so its first
  draw is accepted. They are written out as literals because the linter bans np.random
  in package code, and test_oracle_pick_parameters_match_the_sampler pins them against
  the sampler itself.
- ORACLE_THROW_STANDOFF is test_pick_ground_toss's own target_distance for the drive,
  and the arm configurations for the toss are that test's own two confs, already module
  constants in parameterized_skills.

Note that the standoff is passed explicitly and is deliberately outside
MOVE_TO_TARGET_DISTANCE_BOUNDS, which MoveToThrowPoseController samples from. Those
bounds are the grasping range; nothing constrains a caller-supplied parameter to them,
and 1.35 m is the only standoff in this repository demonstrated to land a cube in the
goal region.

There is no other scripted policy in the monorepo to follow, so the shape here is the
smallest one that keeps the existing test idiom intact: this class only *chooses* the
next ground controller and its parameters, and the caller runs the same step/observe/
terminated loop the parameterized-skill tests already run. Nothing here steps the
environment.
"""

from typing import Any

import numpy as np
from bilevel_planning.structs import GroundParameterizedController
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder.envs.dynamic3d.object_types import (
    MujocoMovableObjectType,
    MujocoTidyBotRobotObjectType,
)
from numpy.typing import NDArray
from relational_structs import GroundAtom, Object, ObjectCentricState

from kinder_models.dynamic3d.shelf.parameterized_skills import (
    create_lifted_controllers as shelf_create_lifted_controllers,
)
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    TOSS_RELEASE_ARM_CONF,
    TOSS_WINDUP_ARM_CONF,
    create_lifted_controllers,
)
from kinder_models.dynamic3d.tossing.state_abstractions import (
    BIN_NAME_PREFIX,
    CUBE_NAME_PREFIX,
    Holding,
    NearBin,
    Tossing3DStateAbstractor,
)
from kinder_models.dynamic3d.utils import PyBulletSim

# The grasp, as drawn by PickShelfController's own sampler; see the module docstring.
ORACLE_PICK_DISTANCE = 0.5682351863248143
ORACLE_PICK_ROTATION = -0.7008563047585579

# The base pose to throw from, relative to the bin.
ORACLE_THROW_STANDOFF = 1.35
ORACLE_THROW_ROTATION = 0.0


class Tossing3DOraclePolicy:
    """Chooses the next ground controller and its parameters from the state alone."""

    def __init__(
        self,
        sim: ObjectCentricTidyBot3DEnv,
        action_space: Any,
        pybullet_sim: PyBulletSim | None = None,
        throw_standoff: float = ORACLE_THROW_STANDOFF,
    ) -> None:
        """Initialize the oracle policy.

        The pick comes from the shelf package: Tossing3D has no pick controller of its
        own, which is why the two pick-and-toss tests reach into shelf for it too.
        """
        self._abstractor = Tossing3DStateAbstractor(sim)
        self._controllers = create_lifted_controllers(
            action_space, pybullet_sim=pybullet_sim
        )
        self._shelf_controllers = shelf_create_lifted_controllers(action_space)
        self._throw_standoff = throw_standoff

    def get_next_controller(
        self, state: ObjectCentricState
    ) -> tuple[str, GroundParameterizedController, NDArray[np.float64]] | None:
        """The next controller to run, its objects already ground, and its parameters.

        Returns None once the goal holds, which is the only terminal condition: a missed
        toss leaves the cube on the far side of the barrier, where the policy asks for
        the pick again and that grasp fails to plan. There is deliberately no recovery
        branch, because the domain admits no recovery, and a fallback skill would hide
        the irreversibility the environment exists to exhibit.
        """
        if self._abstractor.goal_deriver(state).check_state(state):
            return None

        robot = self._get_robot(state)
        cube = self._get_movable_by_prefix(state, CUBE_NAME_PREFIX)
        target_bin = self._get_movable_by_prefix(state, BIN_NAME_PREFIX)

        atoms = self._abstractor.state_abstractor(state).atoms
        holding = GroundAtom(Holding, [robot, cube]) in atoms
        near_bin = GroundAtom(NearBin, [robot, target_bin]) in atoms

        if holding and near_bin:
            name = "toss_from_windup"
            lifted = self._controllers[name]
            objects: tuple[Object, ...] = (robot,)
            params = np.array([TOSS_WINDUP_ARM_CONF, TOSS_RELEASE_ARM_CONF])
        elif holding:
            name = "move_to_throw_pose"
            lifted = self._controllers[name]
            objects = (robot, target_bin, cube)
            params = np.array([self._throw_standoff, ORACLE_THROW_ROTATION])
        else:
            name = "pick_shelf"
            lifted = self._shelf_controllers[name]
            objects = (robot, cube)
            params = np.array([ORACLE_PICK_DISTANCE, ORACLE_PICK_ROTATION])

        return name, lifted.ground(objects), params

    @staticmethod
    def _get_robot(state: ObjectCentricState) -> Object:
        robots = state.get_objects(MujocoTidyBotRobotObjectType)
        assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
        return robots[0]

    @staticmethod
    def _get_movable_by_prefix(state: ObjectCentricState, prefix: str) -> Object:
        matches = [
            obj
            for obj in state.get_objects(MujocoMovableObjectType)
            if obj.name.startswith(prefix)
        ]
        assert len(matches) == 1, f"Expected 1 {prefix}, got {len(matches)}"
        return matches[0]
