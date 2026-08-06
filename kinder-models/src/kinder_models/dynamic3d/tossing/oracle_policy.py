"""A privileged solver for Tossing3D: pick the cube, drive to a throw pose, toss.

The policy is a three-way branch on the abstract state, because the domain admits
exactly one plan shape. What makes it an oracle rather than a fixed script is the
continuous parameters -- the part a learner would have to find -- and every one of them
is a value this repository already publishes:

- ORACLE_PICK_DISTANCE and ORACLE_PICK_ROTATION are the pair that
  PickShelfController.sample_parameters draws from np.random.default_rng(123), the rng
  test_pick_ground_toss constructs to parameterize this exact grasp. The rejection loop
  rejects a draw only when the base pose it implies lands within 0.6 m of a *different*
  cube, so with one cube in the scene the first draw is always accepted and the pair is
  a function of the seed alone -- it does not depend on the cube's pose. They are
  literals rather than a draw made here so that the oracle is reproducible without
  carrying an rng, and
  test_oracle_pick_parameters_match_the_sampler pins them against the sampler itself.
- ORACLE_THROW_STANDOFF is test_pick_ground_toss's own target_distance for the drive,
  and the arm configurations for the toss are that test's own two confs, already module
  constants in parameterized_skills.

Note that the standoff is passed explicitly rather than sampled. It sits inside
TOSS_TARGET_DISTANCE_BOUNDS, the range MoveToThrowPoseController draws from, but nothing
constrains a caller-supplied parameter to that range either way. 1.35 m is the standoff
test_pick_ground_toss drives to, and the test beside this module is what shows it lands
the cube in the goal region -- that test prints the landing position without checking
the goal.

The one precedent in the monorepo is kinder-ds-policies, whose Transport3DScriptedPolicy
(policies/kinematic3d/transport3d.py) drives the same kind of parameterized skills. It
is a different interface, not a different implementation of this one: a StatefulPolicy
maps a vectorized observation straight to an action and owns the step/observe/terminated
bookkeeping internally, whereas this class only *chooses* the next ground controller and
its parameters and leaves that loop to the caller -- the same loop the parameterized-
skill tests already run. Nothing here steps the environment, though constructing one
does reset it -- see __init__.

The policy is written for the one-cube variant, Tossing3D-o1: with more cubes there is a
choice of which to throw and which order to throw them in, and this makes neither.
"""

import numpy as np
from bilevel_planning.structs import GroundParameterizedController
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv
from kinder.envs.dynamic3d.object_types import (
    MujocoMovableObjectType,
    MujocoTidyBotRobotObjectType,
)
from kinder.envs.dynamic3d.robots.tidybot_robot_env import TidyBot3DRobotActionSpace
from numpy.typing import NDArray
from relational_structs import Array, GroundAtom, Object, ObjectCentricState

from kinder_models.dynamic3d.shelf import parameterized_skills as shelf_skills
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

# The name of the chosen controller, the controller already ground on its objects, and
# the continuous parameters to reset it with. Named because the full form does not fit
# in 88 columns.
_ControllerSelection = tuple[
    str, GroundParameterizedController[ObjectCentricState, Array], NDArray[np.float64]
]


class Tossing3DOraclePolicy:
    """Chooses the next ground controller and its parameters for Tossing3D-o1.

    Branches on the abstract state rather than on raw features -- it still reads the
    state directly to find its objects -- which means it inherits
    Tossing3DStateAbstractor's dependence on the live simulator: InGoalRegion is checked
    against the environment's own goal region, not against the state argument.
    """

    def __init__(
        self,
        sim: ObjectCentricTidyBot3DEnv,
        action_space: TidyBot3DRobotActionSpace,
        pybullet_sim: PyBulletSim | None = None,
        throw_standoff: float = ORACLE_THROW_STANDOFF,
    ) -> None:
        """Initialize the oracle policy.

        The pick comes from the shelf package: Tossing3D has no pick controller of its
        own, which is why the two pick-and-toss tests reach into shelf for it too.

        Construct this *before* the episode starts. Tossing3DStateAbstractor resets sim
        to read its objects, so building a policy mid-episode discards that episode, and
        it allocates a PyBullet client that lives as long as the policy does.

        pybullet_sim is deliberately not forwarded to the shelf factory, which accepts
        the same keyword. PickShelfController.reset writes base_link_to_held_obj onto
        the sim it plans in, so sharing one across the pick and the toss would be a
        behaviour change rather than only an allocation saving.
        """
        self._abstractor = Tossing3DStateAbstractor(sim)
        self._controllers = create_lifted_controllers(
            action_space, pybullet_sim=pybullet_sim
        )
        self._shelf_controllers = shelf_skills.create_lifted_controllers(action_space)
        self._throw_standoff = throw_standoff

    def get_next_controller(
        self, state: ObjectCentricState
    ) -> _ControllerSelection | None:
        """The next controller to run, its objects already ground, and its parameters.

        Returns None once the goal holds, which is the only terminal condition. There is
        deliberately no recovery branch: a toss that carries the cube past the barrier is
        unrecoverable, and a fallback skill would hide the irreversibility the
        environment exists to exhibit. Note what that costs: this never reads Reachable,
        the predicate state_abstractions added to make that irreversibility expressible,
        so it still selects pick_shelf for a cube it cannot reach, and
        PickShelfController.reset then fails its `assert base_motion_plan is not None`.
        The caller sees an AssertionError, not a signal it can act on. A toss that falls
        short leaves the cube on the near side instead, where the policy simply asks for
        the pick again, so a caller that cannot tolerate an unbounded retry should cap
        the number of selections. Neither miss is exercised by a test.
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
        """Find the one movable whose name starts with prefix.

        The bin, the cube and the barrier are all MujocoMovableObjectType, so they can
        only be told apart by name, as the state abstractor does. Note how narrowly
        "cuboid_barrier" misses the "cube" prefix. The assertion is what confines this
        policy to Tossing3D-o1: with two cubes it fires rather than picking one.
        """
        matches = [
            obj
            for obj in state.get_objects(MujocoMovableObjectType)
            if obj.name.startswith(prefix)
        ]
        assert len(matches) == 1, f"Expected 1 {prefix}, got {len(matches)}"
        return matches[0]
