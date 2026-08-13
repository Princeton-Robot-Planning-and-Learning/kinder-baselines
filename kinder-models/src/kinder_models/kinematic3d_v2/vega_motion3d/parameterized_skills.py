"""Parameterized skills for the VegaMotion3D environment.

The only skill is moving the arm to the target, so this is pure motion planning: sample
a collision-free goal configuration whose end effector lies in the target sphere, plan a
collision-free joint path to it, and emit the path as bounded joint deltas.
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

# Goal configurations are drawn uniformly within a per-joint window around the current
# configuration and accepted when the end effector lands inside the target sphere, so
# acceptance is the fraction of the window whose end effector is on target: roughly
# 0.2-0.3% per draw at ~0.6 ms per forward-kinematics call, i.e. a few hundred
# milliseconds per successful sample. This budget makes exhaustion (which raises
# TrajectorySamplingFailure) a several-sigma event rather than a routine one.
DEFAULT_NUM_GOAL_CANDIDATES = 10_000


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
        collision_fn: Callable[[Configuration], bool],
        goal_joint_window: float | None = None,
        num_goal_candidates: int = DEFAULT_NUM_GOAL_CANDIDATES,
    ) -> None:
        super().__init__(objects)
        self._sim = sim
        self._planner = planner
        self._collision_fn = collision_fn
        # The window defaults to the environment's target-witness window: targets are
        # placed at the end effector of a configuration within that window of home, so
        # sampling the same window around the (initially home) current configuration is
        # guaranteed a solution for every environment-generated target.
        if goal_joint_window is None:
            goal_joint_window = sim.config.target_witness_joint_delta
        self._goal_joint_window = goal_joint_window
        self._num_goal_candidates = num_goal_candidates
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

        # Sample goal configurations directly: draw arm configurations within a
        # per-joint window around the current configuration and accept ones that are
        # collision-free with the end effector inside the target sphere. Sampling near
        # the current configuration keeps goals on the current arm branch, so the
        # planned motion is commensurate with the goal's distance; solving IK at a
        # sampled end-effector orientation instead can return solutions on a far
        # shoulder branch and demand swings of hundreds of degrees (issue #110).
        target = np.asarray(x.target_position)
        current = np.asarray(x.arm_joint_positions)
        lower, upper = self._space.bounds()
        sample_lower = np.clip(current - self._goal_joint_window, lower, upper)
        sample_upper = np.clip(current + self._goal_joint_window, lower, upper)
        base_config = dict(self._sim.configuration)
        target_radius = self._sim.config.target_radius
        ee_frame = self._manipulator.ee_frame
        for _ in range(self._num_goal_candidates):
            joints = rng.uniform(sample_lower, sample_upper)
            config = base_config | self._space.to_configuration(joints)
            position = self._sim.tree.forward_kinematics(ee_frame, config).t
            if np.linalg.norm(position - target) >= target_radius:
                continue
            if self._collision_fn(config):
                continue
            return joints
        raise TrajectorySamplingFailure(
            f"No goal configuration found for target {tuple(target)} within "
            f"{self._goal_joint_window} rad per joint of the current configuration "
            f"after {self._num_goal_candidates} samples"
        )

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
    collision_fn = create_collision_fn(sim)
    planner = create_motion_planner(
        sim.robot.groups[sim.robot.manipulators[sim.config.manipulator].group],
        collision_fn,
        rng,
        prefer_ompl=prefer_ompl,
    )

    class MoveToTargetController(GroundMoveToTargetController):
        """Controller for moving the robot arm to the target."""

        def __init__(self, objects):
            super().__init__(objects, sim, planner, collision_fn)

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
