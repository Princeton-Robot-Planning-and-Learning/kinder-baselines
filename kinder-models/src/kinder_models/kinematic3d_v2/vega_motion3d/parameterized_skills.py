"""Parameterized skills for the VegaMotion3D environment.

The only skill is moving the arm to the target, so this is pure motion planning: sample
an end-effector pose at the target, solve IK for a goal configuration, plan a collision-
free joint path to it, and emit the path as bounded joint deltas.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Callable, Sequence

import numpy as np
import pybullet as p
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from kinder.envs.kinematic3d_v2.base_env import ArmJointDeltaActionSpace
from kinder.envs.kinematic3d_v2.object_types import (
    ARM_NUM_JOINTS,
    Kinematic3Dv2ArmRobotType,
    Kinematic3Dv2PointType,
)
from kinder.envs.kinematic3d_v2.vega_motion3d import (
    ObjectCentricVegaMotion3DEnv,
    VegaMotion3DObjectCentricState,
)
from prpl_kinematics.collision import PyBulletCollisionChecker
from prpl_kinematics.planning import BiRRTPlanner
from prpl_kinematics.planning.configuration_space import ConfigurationSpace
from prpl_kinematics.planning.motion_planner import MotionPlanner
from prpl_kinematics.tree.kinematic_tree import Configuration
from relational_structs import Object, ObjectCentricState, Variable
from spatialmath import SE3, SO3

# Sampled end-effector orientations are drawn within this angle of the orientation the
# arm holds at home. Vega's analytic IK solves a non-SRS 7R arm, so orientation matters
# a great deal for whether a reachable-looking target actually admits a solution:
# measured over the environment's target bounds, uniformly random orientations solve
# about 27% of the time, this cone about 95%, and the home orientation alone about 97%.
# A cone keeps the sampler diverse without spending most draws on failures.
DEFAULT_ORIENTATION_PERTURBATION = np.pi / 6


def ompl_is_available() -> bool:
    """Whether the optional ompl package can be imported.

    ompl is an extra of prpl_kinematics (``prpl_kinematics[planning]``) because it
    publishes wheels for fewer platforms than everything else.
    """
    return importlib.util.find_spec("ompl") is not None


def create_motion_planner(
    space: ConfigurationSpace,
    collision_fn: Callable[[Configuration], bool],
    rng: np.random.Generator,
    timeout: float = 5.0,
    prefer_ompl: bool = True,
) -> MotionPlanner:
    """An OMPL planner when ompl is installed, otherwise a BiRRT planner.

    Both satisfy the same ``MotionPlanner`` protocol. Note that OMPL's RNG is
    process-global rather than per-instance, so ``rng`` seeds only the BiRRT fallback;
    call ``prpl_kinematics.planning.seed_ompl`` once per process to make OMPL runs
    reproducible.
    """
    if prefer_ompl and ompl_is_available():
        planning = importlib.import_module("prpl_kinematics.planning")
        planner: MotionPlanner = planning.OMPLPlanner(
            space, collision_fn, timeout=timeout
        )
        return planner
    return BiRRTPlanner(space, collision_fn, rng)


def create_collision_fn(
    sim: ObjectCentricVegaMotion3DEnv,
) -> Callable[[Configuration], bool]:
    """A self-collision check for the robot in ``sim``.

    This builds a checker over the environment's kinematic tree rather than reusing the
    environment's own, which is not exposed. The tree is shared, so the two stay in
    agreement; the cost is one extra PyBullet client held for the process lifetime.
    """
    physics_client_id = p.connect(p.DIRECT)
    collision_checker = PyBulletCollisionChecker(physics_client_id)
    collision_checker.load(sim.tree)
    collision_checker.ignore(sim.robot.allowed_collision_pairs)
    return collision_checker.in_collision


class GroundMoveToTargetController(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Controller for moving the robot arm to the target."""

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricVegaMotion3DEnv,
        planner: MotionPlanner,
        orientation_perturbation: float = DEFAULT_ORIENTATION_PERTURBATION,
    ) -> None:
        super().__init__(objects)
        self._sim = sim
        self._planner = planner
        self._orientation_perturbation = orientation_perturbation
        self._manipulator = sim.robot.manipulators[sim.config.manipulator]
        self._space = sim.robot.groups[self._manipulator.group]
        self._robot, self._target = objects
        self._current_params: np.ndarray | None = None
        self._current_plan: list[np.ndarray] | None = None
        self._current_state: ObjectCentricState | None = None

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> np.ndarray:
        assert isinstance(x, VegaMotion3DObjectCentricState)
        self._sim.set_state(x)

        # Sample an end-effector orientation within a cone around the home orientation.
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = self._orientation_perturbation * rng.random()
        rotation = np.array(SO3.EulerVec(angle * axis))
        home_pose = self._sim.target_reach_pose(x.target_position)
        target_pose = SE3.Rt(rotation @ np.array(home_pose.R), x.target_position)

        # Solve IK for a configuration whose end effector reaches that pose.
        solution = self._manipulator.ik.solve(target_pose, self._sim.configuration)
        if solution is None:
            raise TrajectorySamplingFailure(f"IK failed for target pose {target_pose}")

        return self._space.to_vector(solution)

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._current_plan is not None and len(self._current_plan) == 0

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, VegaMotion3DObjectCentricState)
        self._sim.set_state(self._current_state)

        # Generate the motion plan if it doesn't exist yet.
        if self._current_plan is None:
            start = self._sim.configuration
            goal = dict(start)
            goal.update(self._space.to_configuration(self._current_params))
            path = self._planner.plan(start, goal)
            if path is None:
                raise TrajectorySamplingFailure("Motion planning failed")

            # Densify so that consecutive waypoints are within one action of each other.
            # Planners return waypoints at whatever spacing search produced, which can
            # exceed what a single action can cover.
            max_step = self._sim.config.max_action_mag / 2
            vectors = [self._space.to_vector(config) for config in path]
            plan: list[np.ndarray] = []
            for previous, following in zip(vectors[:-1], vectors[1:], strict=True):
                plan.extend(self._space.interpolate(previous, following, max_step))
            self._current_plan = plan

        # Pop the next target joint positions from the plan.
        assert self._current_plan is not None
        target_joints = self._current_plan.pop(0)

        # Every Vega arm joint is bounded, so a plain difference is the correct delta;
        # no wrapping is possible on these joints.
        current_joints = np.asarray(self._current_state.arm_joint_positions)
        delta = target_joints - current_joints
        max_magnitude = self._sim.config.max_action_mag
        action = np.clip(delta, -max_magnitude, max_magnitude)

        return action.astype(np.float32)

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x


def create_lifted_controllers(
    action_space: ArmJointDeltaActionSpace,
    sim: ObjectCentricVegaMotion3DEnv,
    rng: np.random.Generator | None = None,
    prefer_ompl: bool = True,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for VegaMotion3D."""
    del action_space  # the action space is implied by the environment

    if rng is None:
        rng = np.random.default_rng(0)
    planner = create_motion_planner(
        sim.robot.groups[sim.robot.manipulators[sim.config.manipulator].group],
        create_collision_fn(sim),
        rng,
        prefer_ompl=prefer_ompl,
    )

    class MoveToTargetController(GroundMoveToTargetController):
        """Controller for moving the robot arm to the target."""

        def __init__(self, objects):
            super().__init__(objects, sim, planner)

    # Create variables for lifted controllers.
    robot = Variable("?robot", Kinematic3Dv2ArmRobotType)
    target = Variable("?target", Kinematic3Dv2PointType)

    move_to_target_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveToTargetController,
            Box(-np.inf, np.inf, (ARM_NUM_JOINTS,)),
        )
    )
    return {
        "move_to_target": move_to_target_controller,
    }
