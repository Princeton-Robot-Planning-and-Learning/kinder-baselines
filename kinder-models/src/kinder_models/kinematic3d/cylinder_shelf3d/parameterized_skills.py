"""Parameterized skills for the CylinderShelf3D environment.

Two canned grasp modes are supported, selected by ``grasp_mode``:
"side" (the default) approaches near-horizontally and grips the
cylinder's side near its top; "top_down" descends vertically and grips
the upper section from above. Because cylinders are rotationally
symmetric, the approach angle around the cylinder axis is a free
parameter in both modes, so the pick samples the full circle when
choosing where to stage the base. Note that in top-down mode the
gripper-and-forearm column above the held cylinder needs ~0.65 m of
clear height, which no finite opening of the standard shelf provides —
top-down placement is only feasible onto surfaces with essentially open
height above them.

Neither skill uses arm motion planning. All arm motion is planar:
joints 3, 5, and 7 stay at their retract values, joint 1 absorbs only
the small lateral correction that keeps the arm plane through the
target, and the motion happens in joints 2, 4, and 6 — the pitch
joints — so the gripper travels forward, up, and down in the vertical
plane without twisting. Joint trajectories are solved by numerical
continuation along prescribed in-plane end-effector paths, and every
configuration is collision-checked against the robot base, the shelf,
and (while grasped) the held cylinder. The pick's stow and the place's
retreat replay their approach configurations in reverse. Because the
grasp is planar, entering the shelf at the grasp's own pitch keeps the
cylinder upright through the placement.
"""

from typing import Any, Sequence

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from kinder.envs.kinematic3d.cylinder_shelf3d import (
    CylinderShelf3DObjectCentricState,
    Kinematic3DRobotType,
    ObjectCentricCylinderShelf3DEnv,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DFixtureType,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
    extend_joints_to_include_fingers,
)
from pybullet_helpers.geometry import (
    Pose,
    SE2Pose,
    matrix_from_quat,
    multiply_poses,
    set_pose,
)
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.joint import JointPositions, get_jointwise_difference
from pybullet_helpers.motion_planning import (
    remap_joint_position_plan_to_constant_distance,
    remap_se2_pose_plan_to_constant_distance,
    run_single_arm_mobile_base_motion_planning,
)
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from relational_structs import (
    Object,
    ObjectCentricState,
    Variable,
)
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from kinder_models.kinematic3d.constants import (
    GRIPPER_CLOSE_THRESHOLD,
    GRIPPER_OPEN_THRESHOLD,
    HOME_JOINT_POSITIONS,
)
from kinder_models.kinematic3d.utils import (
    get_target_robot_pose_from_parameters,
    step_toward_se2_waypoint,
)

# constants
# Grasp modes. "side" (the default) approaches near-horizontally and grips
# the cylinder's side near its top; "top_down" descends vertically and
# grips the cylinder's upper section from above. Both are planar canned
# motions through joints 2, 4, 6 (with the joint-1 lateral correction).
GRASP_MODES = ("side", "top_down")
MOVE_TO_TARGET_DISTANCE_BOUNDS = (0.78, 0.88)
# The vertical top-down reach is infeasible close to the base (with
# joints 3/5/7 pinned at their retract values the wrist cannot point
# straight down near the body); the feasible band starts around 0.6 m
# and shorter cylinders need the far end.
TOP_DOWN_DISTANCE_BOUNDS = (0.68, 0.78)
# Cylinders are rotationally symmetric: approach from any angle.
MOVE_TO_TARGET_ROT_BOUNDS = (-np.pi, np.pi)
PLACE_X_OFFSET_BOUNDS = (-0.15, 0.15)
PLACE_Y_OFFSET_BOUNDS = (-0.05, 0.1)
# The staging distance is a sampled parameter: low board targets need the
# base farther back (a close base forces the folded arm into it), high
# targets need it closer (reach). Infeasible draws fail the sample's
# collision or reach checks and the planner resamples.
PLACE_BASE_DISTANCE_BOUNDS = (0.65, 0.88)
# Headroom (beyond the standing cylinder) an opening must have for the
# place approach to slide the cylinder in: the pre-place waypoint enters
# slightly lifted, so a bare height-sized opening is not placeable.
PLACE_VERTICAL_CLEARANCE = 0.04
# In top-down mode the gripper column above the held cylinder — measured
# at 0.61-0.66 m in pitch-90 planar configurations, dominated by the
# forearm hanging directly overhead — must also fit in the opening.
# (With the standard shelf this exceeds every finite opening, so a
# top-down place is only feasible on surfaces with essentially open
# height above them.)
TOP_DOWN_PLACE_HEADROOM = 0.68

# Side-grasp geometry. The approach axis is tilted down by
# SIDE_GRASP_PITCH from horizontal (a purely horizontal approach is
# unreachable for the arm at low grasp heights). The grasp point sits
# GRASP_DEPTH_BELOW_TOP below the cylinder top, and the end effector
# stops GRASP_AXIS_STANDOFF short of the cylinder axis along the
# approach so the fingers straddle the cylinder without touching it.
SIDE_GRASP_PITCH = np.deg2rad(15)
GRASP_DEPTH_BELOW_TOP = 0.05
GRASP_AXIS_STANDOFF = 0.02
# Top-down grasp geometry: the approach is straight down, the grasp point
# sits TOP_DOWN_GRASP_DEPTH below the cylinder top so the fingers overlap
# the upper section, and the same GRASP_AXIS_STANDOFF backs the end
# effector off along the (vertical) approach.
TOP_DOWN_GRASP_PITCH = np.pi / 2
TOP_DOWN_GRASP_DEPTH = 0.03
# How far behind the grasp point (along the approach axis) the reach's
# pre-grasp waypoint sits; the pitch ramps from the retract posture's
# value down to SIDE_GRASP_PITCH before the pre-grasp, so the final
# approach segment is a pure translation.
PRE_GRASP_BACKOFF = 0.10
# The place approach's pre-place waypoint: backed off along the shelf
# approach axis and lifted slightly, so the final segment slides the
# cylinder in over the shelf layer.
PRE_PLACE_BACKOFF = 0.10
PRE_PLACE_LIFT = 0.02
# Spacing of the continuation targets along the in-plane reach path.
PLANAR_PATH_STEP = 0.025

# The arm joints that move during the planar reach: joint 1 (index 0)
# for the small lateral plane correction, and the pitch joints 2, 4, 6
# (indices 1, 3, 5). Joints 3, 5, 7 stay at their retract values.
_PLANAR_FREE_JOINT_INDICES = (0, 1, 3, 5)

# Arm links up to and including the bracelet; later links belong to the
# gripper, which legitimately touches a held object.
_ARM_BODY_LINK_MAX = 7


def get_grasp_approach(approach_yaw: float, pitch: float) -> np.ndarray:
    """Unit vector of a grasp approach axis: world-frame yaw, pitched down."""
    return np.array(
        [
            np.cos(approach_yaw) * np.cos(pitch),
            np.sin(approach_yaw) * np.cos(pitch),
            -np.sin(pitch),
        ]
    )


def get_side_grasp_approach(approach_yaw: float) -> np.ndarray:
    """Unit vector of the side-grasp approach axis for a world-frame yaw."""
    return get_grasp_approach(approach_yaw, SIDE_GRASP_PITCH)


def _pick_distance_bounds(grasp_mode: str) -> tuple[float, float]:
    """Base staging distance bounds for the pick, per grasp mode."""
    if grasp_mode == "top_down":
        return TOP_DOWN_DISTANCE_BOUNDS
    return MOVE_TO_TARGET_DISTANCE_BOUNDS


def get_grasp_geometry(grasp_mode: str) -> tuple[float, float]:
    """(approach pitch, grasp depth below the cylinder top) for a mode."""
    if grasp_mode == "side":
        return SIDE_GRASP_PITCH, GRASP_DEPTH_BELOW_TOP
    if grasp_mode == "top_down":
        return TOP_DOWN_GRASP_PITCH, TOP_DOWN_GRASP_DEPTH
    raise ValueError(f"Unknown grasp mode: {grasp_mode}")


def _approach_pitch(rotation_matrix: np.ndarray) -> float:
    """Angle of the tool z axis below horizontal (positive = pointing down)."""
    approach = rotation_matrix[:, 2]
    return float(np.arctan2(-approach[2], np.linalg.norm(approach[:2])))


def _solve_planar_joints(
    arm: SingleArmPyBulletRobot,
    target_position: np.ndarray,
    target_pitch: float,
    seed_joints: list[float],
) -> list[float] | None:
    """Solve for a planar arm configuration by damped Gauss-Newton.

    Only the free planar joints (1, 2, 4, 6) are adjusted, starting from
    ``seed_joints``; the residual is the 3D end-effector position error
    plus the approach-pitch error. Returns the 7-DOF joint values, or
    None if the solve does not converge.
    """
    free = list(_PLANAR_FREE_JOINT_INDICES)
    joints = np.array(seed_joints[:7], dtype=float)
    epsilon = 1e-4

    def _forward(q: np.ndarray) -> tuple[np.ndarray, float]:
        arm.set_joints(extend_joints_to_include_fingers(q.tolist()))
        ee = arm.get_end_effector_pose()
        rot = matrix_from_quat(ee.orientation)
        return np.array(ee.position), _approach_pitch(rot)

    for _ in range(100):
        position, pitch = _forward(joints)
        residual = np.append(position - target_position, pitch - target_pitch)
        if np.max(np.abs(residual)) < 1e-4:
            return joints.tolist()
        jacobian = np.zeros((4, len(free)))
        for column, joint_index in enumerate(free):
            perturbed = joints.copy()
            perturbed[joint_index] += epsilon
            p_position, p_pitch = _forward(perturbed)
            jacobian[:3, column] = (p_position - position) / epsilon
            jacobian[3, column] = (p_pitch - pitch) / epsilon
        gram = jacobian.T @ jacobian + 1e-8 * np.eye(len(free))
        step = np.linalg.solve(gram, -jacobian.T @ residual)
        step = np.clip(step, -0.2, 0.2)
        joints[free] += step
    return None


def _plan_planar_reach(
    sim: ObjectCentricCylinderShelf3DEnv,
    start_joints: list[float],
    path_targets: list[tuple[np.ndarray, float]],
    extra_collision_ids: set[int] | None = None,
    held_object_id: int | None = None,
    held_object_transform: Pose | None = None,
) -> list[JointPositions]:
    """Continuation-solve a planar joint trajectory along an in-plane path.

    Each (position, pitch) target is solved from the previous solution, so the
    trajectory follows the prescribed end-effector path through the pitch joints without
    branch jumps. Every configuration is checked for collision between the arm and the
    robot base, between the arm and any extra collision bodies (e.g. the shelf), and —
    when an object is held — between the held object and the arm body, the base, and the
    extra bodies.
    """
    arm = sim.robot.arm
    base_body_id = sim.robot.base.robot_id
    physics_client_id = sim.physics_client_id
    plan: list[JointPositions] = []
    seed = list(start_joints[:7])
    for position, pitch in path_targets:
        solution = _solve_planar_joints(arm, position, pitch, seed)
        if solution is None:
            raise TrajectorySamplingFailure(
                f"Planar reach solve failed at {np.round(position, 3)}"
            )
        lower = arm.joint_lower_limits[:7]
        upper = arm.joint_upper_limits[:7]
        limit_margin = 0.03
        if any(
            value < low + limit_margin or value > high - limit_margin
            for value, low, high in zip(solution, lower, upper)
        ):
            raise TrajectorySamplingFailure(
                "Planar reach solution exceeds a joint limit"
            )
        full_joints = extend_joints_to_include_fingers(solution)
        arm.set_joints(full_joints)
        if held_object_id is not None:
            assert held_object_transform is not None
            held_pose = multiply_poses(
                arm.get_end_effector_pose(), held_object_transform
            )
            set_pose(held_object_id, held_pose, physics_client_id)
        if check_body_collisions(arm.robot_id, base_body_id, physics_client_id):
            raise TrajectorySamplingFailure(
                "Arm collides with the robot base along the reach"
            )
        for body_id in extra_collision_ids or set():
            if check_body_collisions(arm.robot_id, body_id, physics_client_id):
                raise TrajectorySamplingFailure(
                    "Arm collides with the scene along the reach"
                )
            if held_object_id is not None and check_body_collisions(
                held_object_id, body_id, physics_client_id
            ):
                raise TrajectorySamplingFailure(
                    "Held object collides with the scene along the reach"
                )
        if held_object_id is not None and _held_object_collides(sim, held_object_id):
            raise TrajectorySamplingFailure(
                "Held object collides with the robot along the reach"
            )
        plan.append(full_joints)
        seed = solution
    return plan


def _held_object_collides(
    sim: ObjectCentricCylinderShelf3DEnv, held_object_id: int
) -> bool:
    """Check the held object against the arm body (excluding the gripper, which
    legitimately touches it) and the robot base."""
    arm_id = sim.robot.arm.robot_id
    for link in range(-1, _ARM_BODY_LINK_MAX + 1):
        if check_body_collisions(
            held_object_id,
            arm_id,
            sim.physics_client_id,
            link2=link,
            perform_collision_detection=(link == -1),
        ):
            return True
    return check_body_collisions(
        held_object_id,
        sim.robot.base.robot_id,
        sim.physics_client_id,
        perform_collision_detection=False,
    )


def _interpolated_path_targets(
    start_position: np.ndarray,
    start_pitch: float,
    mid_position: np.ndarray,
    end_position: np.ndarray,
    end_pitch: float = SIDE_GRASP_PITCH,
) -> list[tuple[np.ndarray, float]]:
    """Dense (position, pitch) targets for the two-leg planar reach.

    Leg one runs from the start to the mid waypoint (the pre-grasp or
    pre-place) while the pitch ramps to ``end_pitch``; leg two pushes
    straight in to the end position at constant pitch.
    """
    targets: list[tuple[np.ndarray, float]] = []
    leg_one = float(np.linalg.norm(mid_position - start_position))
    num_one = max(5, int(np.ceil(leg_one / PLANAR_PATH_STEP)))
    for step in range(1, num_one + 1):
        t = step / num_one
        position = start_position + t * (mid_position - start_position)
        # Quadratic pitch schedule: extend first, pitch late. Pitching
        # down while the end effector is still close to the body folds
        # the elbow (joint 4) past its hard limit; extending first
        # straightens it before the wrist comes down.
        pitch = start_pitch + t * t * (end_pitch - start_pitch)
        targets.append((position, pitch))
    leg_two = float(np.linalg.norm(end_position - mid_position))
    num_two = max(3, int(np.ceil(leg_two / PLANAR_PATH_STEP)))
    for step in range(1, num_two + 1):
        t = step / num_two
        position = mid_position + t * (end_position - mid_position)
        targets.append((position, end_pitch))
    return targets


# Controllers.
class GroundPickController(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Controller for picking up a cylinder with a planar grasp.

    ``grasp_mode`` selects the canned approach: "side" (default) grips the
    cylinder's side near its top; "top_down" descends vertically onto it.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricCylinderShelf3DEnv,
        grasp_mode: str = "side",
    ) -> None:
        super().__init__(objects)
        if grasp_mode not in GRASP_MODES:
            raise ValueError(f"Unknown grasp mode: {grasp_mode}")
        self._grasp_mode = grasp_mode
        self._sim = sim
        self._joint_infos = sim.robot.arm.get_arm_joint_infos()[:7]
        self._robot, self._target = objects
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._reach_configurations: list[JointPositions] | None = None
        self._current_plan: list[SE2Pose] | None = None
        self._current_state: ObjectCentricState | None = None
        self._navigated: bool = False
        self._pre_grasp: bool = False
        self._closed_gripper: bool = False
        self._lifted: bool = False
        self._last_gripper_state: float = 0.0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        """Sample the base staging distance and the approach angle."""
        assert isinstance(x, CylinderShelf3DObjectCentricState)
        bounds = _pick_distance_bounds(self._grasp_mode)
        distance = rng.uniform(*bounds)  # type: ignore
        rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
        return np.array([distance, rot])

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._lifted

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, CylinderShelf3DObjectCentricState)

        # Generate the motion plan if it doesn't exist yet.
        if self._current_plan is None:
            self._sim.set_state(self._current_state)

            target_pose = self._current_state.get_object_pose(
                self.objects[1].name
            ).to_se2()
            target_base_pose = get_target_robot_pose_from_parameters(
                target_pose, self._current_params[0], self._current_params[1]
            )
            # Run base motion planning to the target pose.
            base_plan = run_single_arm_mobile_base_motion_planning(
                self._sim.robot,
                self._sim.robot.base.get_pose(),
                target_base_pose,
                collision_bodies=self._sim._get_collision_object_ids(),  # pylint: disable=protected-access
                seed=0,  # for determinism
            )

            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")

            # Remap the plan to ensure we stay within action limits.
            base_plan = remap_se2_pose_plan_to_constant_distance(
                base_plan,
                max_distance=self._sim.config.max_action_mag,
            )

            # Store the plan (excluding the first state which is the current state).
            self._current_plan = base_plan[1:]

        if not self._navigated:
            # Step toward the next waypoint within action limits.
            assert self._current_plan is not None
            delta_lst, exhausted = step_toward_se2_waypoint(
                self._current_state.base_pose,
                self._current_plan,
                self._sim.config.max_action_mag,
            )
            if exhausted:
                self._navigated = True

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = delta_lst + [0.0] * 7 + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        if self._navigated and not self._pre_grasp:
            # Generate the planar reach plan if it doesn't exist yet.
            if self._current_arm_joint_plan is None:
                self._sim.set_state(self._current_state)

                # Side grasp aligned with the achieved base heading, so the
                # gripper reaches straight out from wherever the base ended
                # up around the cylinder.
                cylinder_pose = self._current_state.get_object_pose(
                    self.objects[1].name
                )
                base_pose = self._current_state.base_pose
                approach_yaw = np.arctan2(
                    cylinder_pose.position[1] - base_pose.y,
                    cylinder_pose.position[0] - base_pose.x,
                )
                half_height = self._current_state.get(self.objects[1], "half_extent_z")
                grasp_pitch, grasp_depth = get_grasp_geometry(self._grasp_mode)
                approach = get_grasp_approach(approach_yaw, grasp_pitch)
                grasp_point = np.array(
                    [
                        cylinder_pose.position[0],
                        cylinder_pose.position[1],
                        cylinder_pose.position[2] + half_height - grasp_depth,
                    ]
                )
                grasp_position = grasp_point - GRASP_AXIS_STANDOFF * approach
                pre_grasp_position = grasp_position - PRE_GRASP_BACKOFF * approach

                # The reach starts from the retract posture: joints 3, 5, 7
                # keep their retract values throughout, so the continuation
                # must be seeded from the retract configuration.
                home_joints = HOME_JOINT_POSITIONS.tolist()
                self._sim.robot.arm.set_joints(home_joints)
                home_ee = self._sim.robot.arm.get_end_effector_pose()
                home_pitch = _approach_pitch(matrix_from_quat(home_ee.orientation))
                path_targets = _interpolated_path_targets(
                    np.array(home_ee.position),
                    home_pitch,
                    pre_grasp_position,
                    grasp_position,
                    end_pitch=grasp_pitch,
                )
                reach_plan = _plan_planar_reach(
                    self._sim,
                    home_joints,
                    path_targets,
                )
                # Remember the reach so the stow can replay it in reverse.
                self._reach_configurations = reach_plan

                current_joints = extend_joints_to_include_fingers(
                    list(self._current_state.joint_positions)
                )
                joint_plan = [current_joints] + reach_plan
                if not np.allclose(current_joints[:7], home_joints[:7], atol=1e-3):
                    # The arm is not at the retract posture; go there first
                    # so the planar reach starts from its seed.
                    joint_plan = [current_joints, home_joints] + reach_plan

                # Remap the plan to ensure we stay within action limits.
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    self._sim.robot.arm,
                    max_distance=self._sim.config.max_action_mag / 2,
                )

                # Store the plan (excluding the first state which is the current state).
                self._current_arm_joint_plan = joint_plan[1:]
            # Pop the next target joint positions from the plan.
            assert self._current_arm_joint_plan is not None
            target_joints = self._current_arm_joint_plan.pop(0)
            if len(self._current_arm_joint_plan) == 0:
                self._pre_grasp = True
            # Compute delta joint positions.
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        if self._pre_grasp and not self._closed_gripper:
            if (
                self._get_current_robot_gripper_pose() > GRIPPER_CLOSE_THRESHOLD
                and np.isclose(
                    self._get_current_robot_gripper_pose(),
                    self._last_gripper_state,
                    atol=0.02,
                )
            ):
                self._closed_gripper = True
            action_lst = [0.0] * 10 + [-1.0]
            action = np.array(action_lst, dtype=np.float32)
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action

        if self._closed_gripper and not self._lifted:
            # Generate the stow plan if it doesn't exist yet: the reach
            # configurations replayed in reverse (so the cylinder retraces
            # the same planar path out of the grasp region), stopping at
            # the last configuration where the held cylinder stays clear
            # of the arm body and the base. That configuration is the
            # carrying pose — continuing all the way to the retract
            # posture would press the cylinder into the lower arm.
            if self._current_retract_plan is None:
                self._sim.set_state(self._current_state)
                assert self._reach_configurations is not None
                grasped_object_id = (
                    self._sim._grasped_object_id  # pylint: disable=protected-access
                )
                grasped_object_transform = (
                    self._sim._grasped_object_transform  # pylint: disable=protected-access
                )
                assert grasped_object_id is not None
                assert grasped_object_transform is not None
                arm = self._sim.robot.arm
                stow_configs: list[JointPositions] = []
                candidates = list(reversed(self._reach_configurations)) + [
                    HOME_JOINT_POSITIONS.tolist()
                ]
                for candidate in candidates:
                    arm.set_joints(candidate)
                    held_pose = multiply_poses(
                        arm.get_end_effector_pose(), grasped_object_transform
                    )
                    set_pose(grasped_object_id, held_pose, self._sim.physics_client_id)
                    if _held_object_collides(self._sim, grasped_object_id):
                        break
                    stow_configs.append(candidate)
                if not stow_configs:
                    raise TrajectorySamplingFailure(
                        "No collision-free carrying pose along the stow"
                    )
                current_joints = extend_joints_to_include_fingers(
                    list(self._current_state.joint_positions)
                )
                joint_plan = [current_joints] + stow_configs

                # Remap the plan to ensure we stay within action limits.
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    self._sim.robot.arm,
                    max_distance=self._sim.config.max_action_mag / 2,
                )

                # Store the plan (excluding the first state which is the current state).
                self._current_retract_plan = joint_plan[1:]
            # Pop the next target joint positions from the plan.
            assert self._current_retract_plan is not None
            target_joints = self._current_retract_plan.pop(0)
            if len(self._current_retract_plan) == 0:
                self._lifted = True
            # Compute delta joint positions.
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._current_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        return x.get(robot_obj, "finger_state")


class GroundPlaceController(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Controller for placing a cylinder upright on the shelf with a planar arm motion.

    Like the pick, the arm moves only through joint 1 (small lateral plane correction)
    and the pitch joints 2, 4, 6: the approach enters the shelf at the same pitch the
    cylinder was grasped with, which is what keeps the cylinder upright, and the retreat
    replays the approach in reverse after the release. Every configuration is collision-
    checked against the robot base, the shelf, and (until the release) the held
    cylinder.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricCylinderShelf3DEnv,
        grasp_mode: str = "side",
    ) -> None:
        super().__init__(objects)
        if grasp_mode not in GRASP_MODES:
            raise ValueError(f"Unknown grasp mode: {grasp_mode}")
        self._grasp_mode = grasp_mode
        self._sim = sim
        self._joint_infos = sim.robot.arm.get_arm_joint_infos()[:7]
        self._robot, self._target, self._target_shelf = objects
        self._current_params: np.ndarray | None = None
        self._current_approach_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._approach_configurations: list[JointPositions] | None = None
        self._current_plan: list[SE2Pose] | None = None
        self._current_state: ObjectCentricState | None = None
        self._navigated: bool = False
        self._approached: bool = False
        self._opened_gripper: bool = False
        self._lifted: bool = False

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        """Sample the placement offset on the shelf."""
        assert isinstance(x, CylinderShelf3DObjectCentricState)
        place_x_offset = rng.uniform(*PLACE_X_OFFSET_BOUNDS)  # type: ignore
        place_y_offset = rng.uniform(*PLACE_Y_OFFSET_BOUNDS)  # type: ignore
        base_distance = rng.uniform(*PLACE_BASE_DISTANCE_BOUNDS)
        return np.array([place_x_offset, place_y_offset, base_distance])

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._lifted

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, CylinderShelf3DObjectCentricState)

        # Generate the base motion plan if it doesn't exist yet.
        if self._current_plan is None:
            self._sim.set_state(self._current_state)

            grasped_object_id = (
                self._sim._grasped_object_id  # pylint: disable=protected-access
            )
            grasped_object_transform = (
                self._sim._grasped_object_transform  # pylint: disable=protected-access
            )
            assert grasped_object_transform is not None

            target_surface_pose = self._current_state.get_object_pose(
                self.objects[2].name
            )
            target_pose_temp_se2 = target_surface_pose.to_se2()
            target_place_pose_se2 = SE2Pose(
                target_pose_temp_se2.x + self._current_params[0],
                target_pose_temp_se2.y + self._current_params[1],
                target_pose_temp_se2.rot,
            )
            target_base_pose = get_target_robot_pose_from_parameters(
                target_place_pose_se2, self._current_params[2], np.pi / 2
            )
            all_collision_ids = (
                self._sim._get_collision_object_ids()  # pylint: disable=protected-access
            )
            # Run base motion planning to the target pose.
            base_plan = run_single_arm_mobile_base_motion_planning(
                self._sim.robot,
                self._sim.robot.base.get_pose(),
                target_base_pose,
                collision_bodies=all_collision_ids - {grasped_object_id},
                seed=0,  # for determinism
                held_object=grasped_object_id,
                base_link_to_held_obj=grasped_object_transform,
            )

            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")

            # Remap the plan to ensure we stay within action limits.
            base_plan = remap_se2_pose_plan_to_constant_distance(
                base_plan,
                max_distance=self._sim.config.max_action_mag,
            )

            # Store the plan (excluding the first state which is the current state).
            self._current_plan = base_plan[1:]

        if not self._navigated:
            # Step toward the next waypoint within action limits.
            assert self._current_plan is not None
            delta_lst, exhausted = step_toward_se2_waypoint(
                self._current_state.base_pose,
                self._current_plan,
                self._sim.config.max_action_mag,
            )
            if exhausted:
                self._navigated = True

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = delta_lst + [0.0] * 7 + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        if self._navigated and not self._approached:
            # Generate the planar approach plan if it doesn't exist yet.
            if self._current_approach_plan is None:
                self._sim.set_state(self._current_state)

                grasped_object_id = (
                    self._sim._grasped_object_id  # pylint: disable=protected-access
                )
                grasped_object_transform = (
                    self._sim._grasped_object_transform  # pylint: disable=protected-access
                )
                assert grasped_object_id is not None
                assert grasped_object_transform is not None

                # Compute the desired object placement pose: standing
                # upright on a shelf board, at the sampled offset from the
                # shelf center, resting just above the board surface
                # (within the env's placement distance threshold) so the
                # release registers as a placement.
                target_surface_pose = self._current_state.get_object_pose(
                    self.objects[2].name
                )
                half_height = self._current_state.get(self.objects[1], "half_extent_z")
                # Choose the lowest present board whose opening fits the
                # standing cylinder plus approach clearance. With the
                # default full shelf that is a regular gap; with an inner
                # board omitted (see shelf_omitted_layers) a tall cylinder
                # lands on the board below the merged opening.
                cylinder_height = 2 * half_height
                if self._grasp_mode == "top_down":
                    required = cylinder_height + TOP_DOWN_PLACE_HEADROOM
                else:
                    required = cylinder_height + PLACE_VERTICAL_CLEARANCE
                for surface_z, opening in self._sim.config.get_layer_openings():
                    if opening >= required:
                        target_surface_z = surface_z
                        break
                else:
                    raise TrajectorySamplingFailure(
                        "No shelf opening fits the cylinder for a "
                        f"{self._grasp_mode} place"
                    )
                desired_object_position = (
                    target_surface_pose.position[0] + self._current_params[0],
                    target_surface_pose.position[1] - 0.05 + self._current_params[1],
                    target_surface_z + half_height + 0.004,
                )

                # The cylinder was grasped from an arbitrary angle around
                # its axis. Choose the placement yaw so the end effector
                # approaches the shelf head-on (from -y, the shelf front):
                # rotating the (upright) cylinder about its own axis
                # rotates the attached grasp with it, and any yaw keeps
                # the cylinder upright. Because the grasp itself was a
                # planar side grasp, the resulting end-effector pose lies
                # in the arm's sagittal plane at the grasp's pitch, so the
                # planar solver can reach it.
                object_to_ee = grasped_object_transform.invert()
                approach_in_object = matrix_from_quat(object_to_ee.orientation)[:, 2]
                if np.linalg.norm(approach_in_object[:2]) < 1e-6:
                    # Vertical (top-down) grasp: the approach has no
                    # horizontal component, so any placement yaw keeps the
                    # entry head-on.
                    placement_yaw = 0.0
                else:
                    grasp_yaw_in_object = np.arctan2(
                        approach_in_object[1], approach_in_object[0]
                    )
                    placement_yaw = np.pi / 2 - grasp_yaw_in_object
                qx, qy, qz, qw = Rotation.from_euler("z", placement_yaw).as_quat()
                desired_object_pose = Pose(desired_object_position, (qx, qy, qz, qw))
                place_pose = multiply_poses(desired_object_pose, object_to_ee)
                place_rotation = matrix_from_quat(place_pose.orientation)
                place_pitch = _approach_pitch(place_rotation)
                approach_world = place_rotation[:, 2]

                # Pre-place waypoint. In side mode, back off along the
                # (near-horizontal) approach axis plus a little extra
                # height. In top-down mode the approach is vertical, and
                # backing off along it would raise the wrist column into
                # the board above the opening — back off horizontally out
                # the shelf front (the entry direction) instead.
                if self._grasp_mode == "top_down":
                    backoff_direction = np.array([0.0, 1.0, 0.0])
                else:
                    backoff_direction = approach_world
                pre_place_position = (
                    np.array(place_pose.position)
                    - PRE_PLACE_BACKOFF * backoff_direction
                    + np.array([0.0, 0.0, PRE_PLACE_LIFT])
                )

                ee_start = self._sim.robot.arm.get_end_effector_pose()
                start_pitch = _approach_pitch(matrix_from_quat(ee_start.orientation))
                path_targets = _interpolated_path_targets(
                    np.array(ee_start.position),
                    start_pitch,
                    pre_place_position,
                    np.array(place_pose.position),
                    end_pitch=place_pitch,
                )
                extra_collision_ids = (
                    self._sim._get_collision_object_ids()  # pylint: disable=protected-access
                ) - {grasped_object_id}
                approach_plan = _plan_planar_reach(
                    self._sim,
                    list(self._current_state.joint_positions),
                    path_targets,
                    extra_collision_ids=extra_collision_ids,
                    held_object_id=grasped_object_id,
                    held_object_transform=grasped_object_transform,
                )
                # Remember the approach so the retreat can replay it in
                # reverse.
                self._approach_configurations = approach_plan

                current_joints = extend_joints_to_include_fingers(
                    list(self._current_state.joint_positions)
                )
                joint_plan = [current_joints] + approach_plan

                # Remap the plan to ensure we stay within action limits.
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    self._sim.robot.arm,
                    max_distance=self._sim.config.max_action_mag / 2,
                )

                # Store the plan (excluding the first state which is the current state).
                self._current_approach_plan = joint_plan[1:]
            # Pop the next target joint positions from the plan.
            assert self._current_approach_plan is not None
            target_joints = self._current_approach_plan.pop(0)
            if len(self._current_approach_plan) == 0:
                self._approached = True
            # Compute delta joint positions.
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        if self._approached and not self._opened_gripper:
            if self._get_current_robot_gripper_pose() < GRIPPER_OPEN_THRESHOLD:
                self._opened_gripper = True
            action_lst = [0.0] * 10 + [1.0]
            action = np.array(action_lst, dtype=np.float32)
            return action

        if self._opened_gripper and not self._lifted:
            # Generate the retreat plan if it doesn't exist yet: the
            # approach configurations replayed in reverse (backing the
            # gripper out of the shelf along the way it came in), ending
            # at the home posture.
            if self._current_retract_plan is None:
                assert self._approach_configurations is not None
                current_joints = extend_joints_to_include_fingers(
                    list(self._current_state.joint_positions)
                )
                joint_plan = (
                    [current_joints]
                    + list(reversed(self._approach_configurations))
                    + [HOME_JOINT_POSITIONS.tolist()]
                )

                # Remap the plan to ensure we stay within action limits.
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    self._sim.robot.arm,
                    max_distance=self._sim.config.max_action_mag / 2,
                )

                # Store the plan (excluding the first state which is the current state).
                self._current_retract_plan = joint_plan[1:]
            # Pop the next target joint positions from the plan.
            assert self._current_retract_plan is not None
            target_joints = self._current_retract_plan.pop(0)
            if len(self._current_retract_plan) == 0:
                self._lifted = True
            # Compute delta joint positions.
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._current_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        return x.get(robot_obj, "finger_state")


def create_lifted_controllers(
    action_space: Kinematic3DRobotActionSpace,
    sim: ObjectCentricCylinderShelf3DEnv,
    grasp_mode: str = "side",
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for CylinderShelf3D.

    ``grasp_mode`` ("side" or "top_down") selects the pick's canned planar
    approach and the matching place entry behavior.
    """
    del action_space
    if grasp_mode not in GRASP_MODES:
        raise ValueError(f"Unknown grasp mode: {grasp_mode}")

    # Create partial controller classes that include the sim
    class PickController(GroundPickController):
        """Controller for picking up a cylinder."""

        def __init__(self, objects):
            super().__init__(objects, sim, grasp_mode=grasp_mode)

    class PlaceController(GroundPlaceController):
        """Controller for placing a cylinder."""

        def __init__(self, objects):
            super().__init__(objects, sim, grasp_mode=grasp_mode)

    # Create variables for lifted controllers
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)

    # Lifted controllers
    pick_distance_bounds = _pick_distance_bounds(grasp_mode)
    pick_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target],
        PickController,
        Box(
            low=np.array(
                [
                    pick_distance_bounds[0],
                    MOVE_TO_TARGET_ROT_BOUNDS[0],
                ]
            ),
            high=np.array(
                [
                    pick_distance_bounds[1],
                    MOVE_TO_TARGET_ROT_BOUNDS[1],
                ]
            ),
        ),
    )

    # Create variables for lifted controllers
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    target_shelf = Variable("?target_shelf", Kinematic3DFixtureType)

    # lifted place controller
    place_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target, target_shelf],
        PlaceController,
        Box(
            low=np.array(
                [
                    PLACE_X_OFFSET_BOUNDS[0],
                    PLACE_Y_OFFSET_BOUNDS[0],
                    PLACE_BASE_DISTANCE_BOUNDS[0],
                ]
            ),
            high=np.array(
                [
                    PLACE_X_OFFSET_BOUNDS[1],
                    PLACE_Y_OFFSET_BOUNDS[1],
                    PLACE_BASE_DISTANCE_BOUNDS[1],
                ]
            ),
        ),
    )

    return {
        "pick": pick_controller,
        "place": place_controller,
    }
