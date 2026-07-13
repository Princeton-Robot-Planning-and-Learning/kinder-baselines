"""Parameterized skills for the Franka FR3 FrankaPickPlace3D environment.

The FR3 is a fixed-base arm mounted on a desk, so unlike the TidyBot skills there is no
base navigation: each skill is a sequence of arm/gripper phases. Waypoint configurations
are computed at reset time with FR3IKSolver (exact MuJoCo kinematics) and executed with
trapezoidal joint-space velocity profiles under proportional control.
"""

from typing import Any

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from gymnasium.spaces import Box
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoFR3RobotObjectType,
    MujocoMovableObjectType,
)
from kinder.envs.dynamic3d.robots.fr3_robot_env import (
    FR3RobotActionSpace,
    FR3RobotEnv,
)
from relational_structs import (
    Array,
    Object,
    ObjectCentricState,
    Variable,
)

from kinder_models.dynamic3d.fr3_ik_solver import FR3IKSolver
from kinder_models.dynamic3d.utils import (
    MAX_SAMPLER_ATTEMPTS,
    _compute_per_joint_profile,
)

# Height (z) of the fixed base mount above the ground. Matches
# robots.fr3.robot.mount_height in FrankaPickPlace3D task JSONs; the mount
# pose in the state only carries (x, y, yaw).
FR3_MOUNT_HEIGHT = 0.75

# Vertical clearance of the pre-grasp/pre-place waypoints above the grasp and
# place poses.
APPROACH_HEIGHT = 0.12

# Extra height above the resting cube height when releasing it.
PLACE_HEIGHT_OFFSET = 0.02

# Number of control steps to hold the gripper command while closing/opening.
# pos_gripper reads back the commanded value for the FR3, so closure cannot be
# detected from the state; instead the command is held long enough for the
# fingers to settle.
GRIPPER_CLOSE_STEPS = 15
GRIPPER_OPEN_STEPS = 10

# Proportional gain for tracking profile targets; deltas are clipped to the
# action space bounds by the caller-provided action space.
ARM_KP = 2.0

# Grasp-point offset bounds relative to the target object center (x and y).
PICK_OFFSET_BOUNDS = (-0.01, 0.01)

# Placement region on the desk, desk-local (x_min, y_min, x_max, y_max).
# Matches goal_region in FrankaPickPlace3D-o1.json.
GOAL_REGION_DESK_LOCAL = (-0.05, 0.12, 0.15, 0.27)

# Minimum distance between a sampled placement and any other movable object.
PLACE_COLLISION_THRESHOLD = 0.08

# Arm joint velocity and acceleration limits (rad/s, rad/s²) for the profile
# executor. Conservative relative to the FR3 datasheet limits so that the
# per-step position deltas stay well inside the action bounds at 10 Hz.
_FR3_MAX_VEL = np.deg2rad(np.array([80.0, 80.0, 80.0, 80.0, 70.0, 70.0, 70.0]))
_FR3_MAX_ACCEL = np.deg2rad(np.array([297.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0]))


class _MovePhase:
    """Follow a trapezoidal joint-space profile to a target configuration."""

    def __init__(self, target_conf: np.ndarray, gripper: float) -> None:
        self.target_conf = target_conf
        self.gripper = gripper
        self.trajectory: np.ndarray | None = None
        self.direction: np.ndarray = np.zeros(7)
        self.start_conf: np.ndarray = np.zeros(7)
        self.step_idx = 0


class _GripPhase:
    """Hold an arm configuration while commanding the gripper."""

    def __init__(self, hold_conf: np.ndarray, gripper: float, num_steps: int) -> None:
        self.hold_conf = hold_conf
        self.gripper = gripper
        self.num_steps = num_steps
        self.step_idx = 0


class _FrankaArmController(GroundParameterizedController[ObjectCentricState, Array]):
    """Base class executing a phase sequence computed at reset time.

    The object parameters always start with the robot; subclasses define the rest and
    build the phase list in _create_phases().
    """

    def __init__(self, *args, ik_solver: FR3IKSolver | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ik_solver = ik_solver
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None
        self._phases: list[_MovePhase | _GripPhase] | None = None
        self._phase_idx = 0
        # Pinch-point orientation in the base frame: same as at the home
        # configuration, i.e. pointing straight down for a top-down grasp.
        self._down_quat: np.ndarray | None = None

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        if self._ik_solver is None:
            self._ik_solver = FR3IKSolver()
        self._last_state = x
        self._current_params = np.asarray(params, dtype=np.float32)
        _, self._down_quat = self._ik_solver.get_site_pose(FR3RobotEnv.HOME_QPOS)
        self._phases = self._create_phases(x)
        self._phase_idx = 0

    def _create_phases(self, x: ObjectCentricState) -> list[_MovePhase | _GripPhase]:
        raise NotImplementedError

    def terminated(self) -> bool:
        assert self._phases is not None
        return self._phase_idx >= len(self._phases)

    def step(self) -> Array:
        assert self._phases is not None
        curr = np.array(self._get_current_arm_conf())
        while self._phase_idx < len(self._phases):
            phase = self._phases[self._phase_idx]
            if isinstance(phase, _MovePhase):
                if phase.trajectory is None:
                    phase.start_conf = curr.copy()
                    phase.trajectory, phase.direction = _compute_per_joint_profile(
                        curr, phase.target_conf, _FR3_MAX_VEL, _FR3_MAX_ACCEL
                    )
                if phase.step_idx >= len(phase.trajectory):
                    self._phase_idx += 1
                    continue
                s = float(phase.trajectory[phase.step_idx])
                target = phase.start_conf + phase.direction * s
                phase.step_idx += 1
                return self._make_action(curr, target, phase.gripper)
            assert isinstance(phase, _GripPhase)
            if phase.step_idx >= phase.num_steps:
                self._phase_idx += 1
                continue
            phase.step_idx += 1
            return self._make_action(curr, phase.hold_conf, phase.gripper)
        # All phases done: hold the final configuration.
        last_phase = self._phases[-1]
        hold = (
            last_phase.target_conf
            if isinstance(last_phase, _MovePhase)
            else last_phase.hold_conf
        )
        return self._make_action(curr, hold, self._final_gripper())

    def observe(self, x: ObjectCentricState) -> None:
        self._last_state = x

    def _final_gripper(self) -> float:
        assert self._phases is not None
        return self._phases[-1].gripper

    def _make_action(
        self, curr: np.ndarray, target: np.ndarray, gripper: float
    ) -> Array:
        action = np.zeros(8, dtype=np.float32)
        # Delta joint targets, clipped to the FR3 action bounds.
        action[:7] = np.clip(ARM_KP * (target - curr), -0.1, 0.1)
        action[7] = gripper
        return action

    def _get_current_arm_conf(self) -> list[float]:
        x = self._last_state
        assert x is not None
        robot = self.objects[0]
        return [x.get(robot, f"pos_arm_joint{i}") for i in range(1, 8)]

    def _world_to_base(self, pos_world: np.ndarray) -> np.ndarray:
        """Transform a world position into the arm base (mount) frame."""
        x = self._last_state
        assert x is not None
        robot = self.objects[0]
        mount = np.array(
            [
                x.get(robot, "pos_base_x"),
                x.get(robot, "pos_base_y"),
                FR3_MOUNT_HEIGHT,
            ]
        )
        yaw = x.get(robot, "pos_base_rot")
        delta = pos_world - mount
        cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)
        return np.array(
            [
                cos_yaw * delta[0] - sin_yaw * delta[1],
                sin_yaw * delta[0] + cos_yaw * delta[1],
                delta[2],
            ]
        )

    def _solve_ik(self, pos_world: np.ndarray, init_conf: np.ndarray) -> np.ndarray:
        assert self._ik_solver is not None and self._down_quat is not None
        target_base = self._world_to_base(pos_world)
        conf = self._ik_solver.solve(target_base, self._down_quat, init_conf)
        reached, _ = self._ik_solver.get_site_pose(conf)
        assert (
            np.linalg.norm(reached - target_base) < 1e-3
        ), f"IK failed to reach {pos_world}"
        return conf


class PickFrankaController(_FrankaArmController):
    """Top-down pick: approach above the target, descend, close, lift.

    The object parameters are:
        robot: The robot itself.
        object: The target object.

    The continuous parameters are an (x, y) grasp-point offset relative to the
    target object center.
    """

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        del x  # any offset within bounds is valid
        return rng.uniform(*PICK_OFFSET_BOUNDS, size=2)

    def _create_phases(self, x: ObjectCentricState) -> list[_MovePhase | _GripPhase]:
        assert self._current_params is not None
        offset_x, offset_y = self._current_params
        target = self.objects[1]
        grasp_world = np.array(
            [
                x.get(target, "x") + offset_x,
                x.get(target, "y") + offset_y,
                x.get(target, "z"),
            ]
        )
        pre_grasp_world = grasp_world + np.array([0.0, 0.0, APPROACH_HEIGHT])

        curr = np.array(self._get_current_arm_conf())
        pre_grasp_conf = self._solve_ik(pre_grasp_world, curr)
        grasp_conf = self._solve_ik(grasp_world, pre_grasp_conf)

        return [
            _MovePhase(pre_grasp_conf, gripper=0.0),
            _MovePhase(grasp_conf, gripper=0.0),
            _GripPhase(grasp_conf, gripper=1.0, num_steps=GRIPPER_CLOSE_STEPS),
            _MovePhase(pre_grasp_conf, gripper=1.0),
        ]


class PlaceFrankaController(_FrankaArmController):
    """Place a held object on the desk: transfer, descend, open, retract home.

    The object parameters are:
        robot: The robot itself.
        object: The held object.
        surface: The desk fixture to place on.

    The continuous parameters are the desk-local (x, y) placement position.
    """

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        x_min, y_min, x_max, y_max = GOAL_REGION_DESK_LOCAL
        surface = self.objects[2]
        held = self.objects[1]
        for _ in range(MAX_SAMPLER_ATTEMPTS):
            place_x = rng.uniform(x_min, x_max)
            place_y = rng.uniform(y_min, y_max)
            place_world = np.array(
                [
                    x.get(surface, "x") + place_x,
                    x.get(surface, "y") + place_y,
                ]
            )
            collision = False
            for other in x.get_objects(MujocoMovableObjectType):
                if other.name == held.name:
                    continue
                other_xy = np.array([x.get(other, "x"), x.get(other, "y")])
                if np.linalg.norm(place_world - other_xy) < PLACE_COLLISION_THRESHOLD:
                    collision = True
                    break
            if not collision:
                return np.array([place_x, place_y])
        raise ValueError("No valid parameters found")

    def _create_phases(self, x: ObjectCentricState) -> list[_MovePhase | _GripPhase]:
        assert self._current_params is not None
        place_x, place_y = self._current_params
        held = self.objects[1]
        surface = self.objects[2]
        # Release with the held object slightly above its resting height.
        place_z = FR3_MOUNT_HEIGHT + x.get(held, "bb_z") / 2 + PLACE_HEIGHT_OFFSET
        place_world = np.array(
            [
                x.get(surface, "x") + place_x,
                x.get(surface, "y") + place_y,
                place_z,
            ]
        )
        pre_place_world = place_world + np.array([0.0, 0.0, APPROACH_HEIGHT])

        curr = np.array(self._get_current_arm_conf())
        pre_place_conf = self._solve_ik(pre_place_world, curr)
        place_conf = self._solve_ik(place_world, pre_place_conf)

        return [
            _MovePhase(pre_place_conf, gripper=1.0),
            _MovePhase(place_conf, gripper=1.0),
            _GripPhase(place_conf, gripper=0.0, num_steps=GRIPPER_OPEN_STEPS),
            _MovePhase(pre_place_conf, gripper=0.0),
            _MovePhase(FR3RobotEnv.HOME_QPOS.copy(), gripper=0.0),
        ]


def create_lifted_controllers(
    action_space: FR3RobotActionSpace,
    init_constant_state: ObjectCentricState | None = None,
    ik_solver: FR3IKSolver | None = None,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for the FrankaPickPlace3D env."""
    del action_space, init_constant_state  # not used

    if ik_solver is None:
        ik_solver = FR3IKSolver()

    # Create wrapper classes that capture the shared IK solver.
    class PickController(PickFrankaController):
        """Pick controller with a pre-configured IK solver."""

        def __init__(self, objects: list[Object]) -> None:
            super().__init__(ik_solver=ik_solver, objects=objects)

    class PlaceController(PlaceFrankaController):
        """Place controller with a pre-configured IK solver."""

        def __init__(self, objects: list[Object]) -> None:
            super().__init__(ik_solver=ik_solver, objects=objects)

    # Pick controller.
    robot = Variable("?robot", MujocoFR3RobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)

    # Parameter space: (x, y) grasp-point offset.
    pick_params_space = Box(
        low=np.array([PICK_OFFSET_BOUNDS[0]] * 2, dtype=np.float32),
        high=np.array([PICK_OFFSET_BOUNDS[1]] * 2, dtype=np.float32),
        dtype=np.float32,
    )

    lifted_pick_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            PickController,
            params_space=pick_params_space,
        )
    )

    # Place controller.
    robot = Variable("?robot", MujocoFR3RobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)
    surface = Variable("?surface", MujocoFixtureObjectType)

    # Parameter space: desk-local (x, y) placement position.
    x_min, y_min, x_max, y_max = GOAL_REGION_DESK_LOCAL
    place_params_space = Box(
        low=np.array([x_min, y_min], dtype=np.float32),
        high=np.array([x_max, y_max], dtype=np.float32),
        dtype=np.float32,
    )

    lifted_place_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target, surface],
            PlaceController,
            params_space=place_params_space,
        )
    )

    return {
        "pick": lifted_pick_controller,
        "place": lifted_place_controller,
    }
