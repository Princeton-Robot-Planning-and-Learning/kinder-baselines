"""Parameterized skills for the CylinderShelf3D environment.

Three skills cover pick-and-place of a tall cylinder with a planar side
grasp:

* ``MoveToPreGrasp`` stages the base around the cylinder (the approach
  angle is a free parameter because cylinders are rotationally
  symmetric) and reaches the arm from the retract posture to the
  pre-grasp pose, a short standoff behind the grasp along the approach
  axis.
* ``Grasp`` covers the last mile: it pushes the gripper straight in from
  the pre-grasp pose to the grasp, closes, and stows the arm to a
  carrying pose. It takes no parameters. Splitting it out lets a planner
  treat just this part as magic and hand it to a teleoperator or a
  learned policy while the approach stays planned.
* ``Place`` navigates to the shelf and slides the cylinder in upright.

Neither the reach nor the place uses arm motion planning. All arm motion
is planar: joints 3, 5, and 7 stay at their retract values, joint 1
absorbs only the small lateral correction that keeps the arm plane
through the target, and the motion happens in joints 2, 4, and 6 — the
pitch joints — so the gripper travels forward, up, and down in the
vertical plane without twisting. Joint trajectories are solved by
numerical continuation along prescribed in-plane end-effector paths, and
every configuration is collision-checked against the robot base, the
shelf, and (while grasped) the held cylinder. The grasp's stow and the
place's retreat replay their approach configurations in reverse. Because
the grasp is planar, entering the shelf at the grasp's own pitch keeps
the cylinder upright through the placement.
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
from pybullet_helpers.joint import (
    JointInfo,
    JointPositions,
    get_jointwise_difference,
)
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
from kinder_models.magic import OutcomePredictor

# constants
MOVE_TO_TARGET_DISTANCE_BOUNDS = (0.78, 0.88)
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

# Side-grasp geometry. The approach axis is tilted down by
# SIDE_GRASP_PITCH from horizontal (a purely horizontal approach is
# unreachable for the arm at low grasp heights). The grasp point sits
# GRASP_DEPTH_BELOW_TOP below the cylinder top, and the end effector
# stops GRASP_AXIS_STANDOFF short of the cylinder axis along the
# approach so the fingers straddle the cylinder without touching it.
SIDE_GRASP_PITCH = np.deg2rad(15)
GRASP_DEPTH_BELOW_TOP = 0.05
GRASP_AXIS_STANDOFF = 0.02
# How far behind the grasp point (along the approach axis) the reach's
# pre-grasp waypoint sits; the pitch ramps from the retract posture's
# value down to SIDE_GRASP_PITCH before the pre-grasp, so the final
# approach segment is a pure translation.
PRE_GRASP_BACKOFF = 0.10
# How close (m) the end effector must be to the pre-grasp position for the
# robot to count as being at the pre-grasp pose.
PRE_GRASP_POSITION_TOL = 0.03
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


def get_side_grasp_approach(approach_yaw: float) -> np.ndarray:
    """Unit vector of the side-grasp approach axis for a world-frame yaw."""
    return np.array(
        [
            np.cos(approach_yaw) * np.cos(SIDE_GRASP_PITCH),
            np.sin(approach_yaw) * np.cos(SIDE_GRASP_PITCH),
            -np.sin(SIDE_GRASP_PITCH),
        ]
    )


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
) -> tuple[list[tuple[np.ndarray, float]], int]:
    """Dense (position, pitch) targets for the two-leg planar reach.

    Leg one runs from the start to the mid waypoint (the pre-grasp or
    pre-place) while the pitch ramps to ``end_pitch``; leg two pushes
    straight in to the end position at constant pitch. Also returns the
    number of leg-one targets, so callers can split the path at the mid
    waypoint.
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
    return targets, num_one


# Controllers.
def get_grasp_positions(
    state: CylinderShelf3DObjectCentricState, target_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """(grasp position, pre-grasp position) of the side grasp on ``target_name``.

    The approach is aligned with the base heading toward the cylinder so the
    gripper reaches straight out from wherever the base is around it. The
    grasp position is where the end effector sits at the grasp; the
    pre-grasp position is PRE_GRASP_BACKOFF behind it along the approach
    axis.
    """
    cylinder_pose = state.get_object_pose(target_name)
    base_pose = state.base_pose
    approach_yaw = np.arctan2(
        cylinder_pose.position[1] - base_pose.y,
        cylinder_pose.position[0] - base_pose.x,
    )
    target = state.get_object_from_name(target_name)
    half_height = state.get(target, "half_extent_z")
    approach = get_side_grasp_approach(approach_yaw)
    grasp_point = np.array(
        [
            cylinder_pose.position[0],
            cylinder_pose.position[1],
            cylinder_pose.position[2] + half_height - GRASP_DEPTH_BELOW_TOP,
        ]
    )
    grasp_position = grasp_point - GRASP_AXIS_STANDOFF * approach
    pre_grasp_position = grasp_position - PRE_GRASP_BACKOFF * approach
    return grasp_position, pre_grasp_position


def is_at_pre_grasp(
    sim: ObjectCentricCylinderShelf3DEnv,
    state: CylinderShelf3DObjectCentricState,
    target_name: str,
) -> bool:
    """Whether the (empty) gripper is within PRE_GRASP_POSITION_TOL of the pre-grasp
    position for ``target_name``. The sim is set to ``state`` to read the end effector
    pose."""
    if state.grasped_object is not None:
        return False
    sim.set_state(state)
    ee_position = np.array(sim.robot.arm.get_end_effector_pose().position)
    _, pre_grasp_position = get_grasp_positions(state, target_name)
    return bool(
        np.linalg.norm(ee_position - pre_grasp_position) < PRE_GRASP_POSITION_TOL
    )


def _plan_reach(
    sim: ObjectCentricCylinderShelf3DEnv,
    state: CylinderShelf3DObjectCentricState,
    target_name: str,
) -> tuple[list[JointPositions], int]:
    """Solve the planar reach from the retract posture to the grasp.

    The sim is set to ``state`` first. Returns the reach configurations and
    the index of the pre-grasp configuration within them; the configurations
    after that index are the final straight-in approach. Raises
    ``TrajectorySamplingFailure`` when the reach is infeasible.
    """
    sim.set_state(state)
    grasp_position, pre_grasp_position = get_grasp_positions(state, target_name)
    # The reach starts from the retract posture: joints 3, 5, 7 keep their
    # retract values throughout, so the continuation must be seeded from
    # the retract configuration.
    home_joints = HOME_JOINT_POSITIONS.tolist()
    sim.robot.arm.set_joints(home_joints)
    home_ee = sim.robot.arm.get_end_effector_pose()
    home_pitch = _approach_pitch(matrix_from_quat(home_ee.orientation))
    path_targets, num_leg_one = _interpolated_path_targets(
        np.array(home_ee.position),
        home_pitch,
        pre_grasp_position,
        grasp_position,
    )
    return _plan_planar_reach(sim, home_joints, path_targets), num_leg_one - 1


def _plan_stow(
    sim: ObjectCentricCylinderShelf3DEnv,
    state: CylinderShelf3DObjectCentricState,
    reach_configurations: list[JointPositions],
) -> list[JointPositions]:
    """Replay the reach in reverse while the held cylinder stays clear.

    The sim is set to ``state`` (which must hold the cylinder) first. The
    stow stops at the last configuration where the cylinder stays clear of
    the arm body and the base: that configuration is the carrying pose.
    Continuing all the way to the retract posture would press the cylinder
    into the lower arm.
    """
    sim.set_state(state)
    grasped_object_id = sim._grasped_object_id  # pylint: disable=protected-access
    grasped_object_transform = (
        sim._grasped_object_transform  # pylint: disable=protected-access
    )
    assert grasped_object_id is not None
    assert grasped_object_transform is not None
    arm = sim.robot.arm
    stow_configs: list[JointPositions] = []
    candidates = list(reversed(reach_configurations)) + [HOME_JOINT_POSITIONS.tolist()]
    for candidate in candidates:
        arm.set_joints(candidate)
        held_pose = multiply_poses(
            arm.get_end_effector_pose(), grasped_object_transform
        )
        set_pose(grasped_object_id, held_pose, sim.physics_client_id)
        if _held_object_collides(sim, grasped_object_id):
            break
        stow_configs.append(candidate)
    if not stow_configs:
        raise TrajectorySamplingFailure(
            "No collision-free carrying pose along the stow"
        )
    return stow_configs


def _predict_grasp_outcome(
    sim: ObjectCentricCylinderShelf3DEnv,
    x: CylinderShelf3DObjectCentricState,
    target_name: str,
) -> ObjectCentricState:
    """Predict the state after grasping ``target_name`` from ``x``.

    ``x`` must already have the base staged (the arm may be anywhere). The
    planar reach is solved, the grasp is registered by the sim's own grasp
    rule at the reach's final configuration, and the arm ends at the
    carrying pose (the last collision-free configuration of the reversed
    reach).
    """
    reach_plan, _ = _plan_reach(sim, x, target_name)
    at_grasp = x.copy()
    _set_arm_joints(at_grasp, reach_plan[-1])
    sim.set_state(at_grasp)
    close_action = np.array([0.0] * 10 + [-1.0], dtype=np.float32)
    grasped, _, _, _, _ = sim.step(close_action)
    assert isinstance(grasped, CylinderShelf3DObjectCentricState)
    if grasped.grasped_object != target_name:
        raise TrajectorySamplingFailure("Grasp did not register at the reach")

    stow_configs = _plan_stow(sim, grasped, reach_plan)
    carry_joints = stow_configs[-1]
    grasp_transform = grasped.grasped_object_transform
    assert grasp_transform is not None
    arm = sim.robot.arm
    arm.set_joints(carry_joints)
    held_pose = multiply_poses(arm.get_end_effector_pose(), grasp_transform)
    predicted = grasped.copy()
    _set_arm_joints(predicted, carry_joints)
    _set_object_pose(predicted, target_name, held_pose)
    sim.set_state(predicted)
    return sim.get_state()


def _remap_joint_plan(
    sim: ObjectCentricCylinderShelf3DEnv,
    state: CylinderShelf3DObjectCentricState,
    configurations: list[JointPositions],
) -> list[JointPositions]:
    """Remap ``[current joints] + configurations`` to the action limits, excluding the
    current joints."""
    current_joints = extend_joints_to_include_fingers(list(state.joint_positions))
    joint_plan = remap_joint_position_plan_to_constant_distance(
        [current_joints] + configurations,
        sim.robot.arm,
        max_distance=sim.config.max_action_mag / 2,
    )
    return joint_plan[1:]


def _arm_action(
    joint_infos: list[JointInfo],
    target_joints: JointPositions,
    state: CylinderShelf3DObjectCentricState,
) -> np.ndarray:
    """Kinder action moving the arm joints toward ``target_joints``."""
    delta_lst = get_jointwise_difference(
        joint_infos, target_joints[:7], state.joint_positions
    )
    return np.array([0.0] * 3 + delta_lst + [0.0], dtype=np.float32)


class GroundMoveToPreGraspController(
    GroundParameterizedController[ObjectCentricState, np.ndarray],
    OutcomePredictor[ObjectCentricState],
):
    """Stage the base around a cylinder and reach the arm to its pre-grasp pose.

    The parameters are the base staging distance and the approach angle
    around the cylinder. The base is motion-planned to the staged pose, then
    the arm follows the first leg of the planar reach from the retract
    posture to the pre-grasp waypoint. The controller also predicts its own
    outcome (:meth:`predict_outcome`).
    """

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricCylinderShelf3DEnv,
    ) -> None:
        super().__init__(objects)
        self._sim = sim
        self._joint_infos = sim.robot.arm.get_arm_joint_infos()[:7]
        self._robot, self._target = objects
        self._current_params: np.ndarray | None = None
        self._current_plan: list[SE2Pose] | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_state: ObjectCentricState | None = None
        self._navigated: bool = False
        self._reached: bool = False

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        """Sample the base staging distance and the approach angle."""
        assert isinstance(x, CylinderShelf3DObjectCentricState)
        distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)  # type: ignore
        rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
        return np.array([distance, rot])

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_arm_joint_plan = None
        self._current_state = x
        self._navigated = False
        self._reached = False

    def terminated(self) -> bool:
        return self._reached

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, CylinderShelf3DObjectCentricState)

        # Generate the base motion plan if it doesn't exist yet.
        if self._current_plan is None:
            self._sim.set_state(self._current_state)
            target_pose = self._current_state.get_object_pose(
                self._target.name
            ).to_se2()
            target_base_pose = get_target_robot_pose_from_parameters(
                target_pose, self._current_params[0], self._current_params[1]
            )
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
            delta_lst, exhausted = step_toward_se2_waypoint(
                self._current_state.base_pose,
                self._current_plan,
                self._sim.config.max_action_mag,
            )
            if exhausted:
                self._navigated = True
            return np.array(delta_lst + [0.0] * 7 + [0.0], dtype=np.float32)

        if not self._reached:
            # Generate the reach plan (leg one only) if it doesn't exist yet.
            if self._current_arm_joint_plan is None:
                reach_plan, pre_grasp_index = _plan_reach(
                    self._sim, self._current_state, self._target.name
                )
                configurations = reach_plan[: pre_grasp_index + 1]
                home_joints = HOME_JOINT_POSITIONS.tolist()
                if not np.allclose(
                    self._current_state.joint_positions, home_joints[:7], atol=1e-3
                ):
                    # The arm is not at the retract posture; go there first
                    # so the planar reach starts from its seed.
                    configurations = [home_joints] + configurations
                self._current_arm_joint_plan = _remap_joint_plan(
                    self._sim, self._current_state, configurations
                )
            target_joints = self._current_arm_joint_plan.pop(0)
            if not self._current_arm_joint_plan:
                self._reached = True
            return _arm_action(self._joint_infos, target_joints, self._current_state)

        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x

    def predict_outcome(self, x: ObjectCentricState, params: Any) -> ObjectCentricState:
        """Predict the state at the pre-grasp pose without simulating the motion.

        The base is staged where ``params`` put it (base motion planning is
        skipped; the staged pose only has to be collision-free) and the arm is
        set to the pre-grasp configuration of the planar reach.
        """
        assert isinstance(x, CylinderShelf3DObjectCentricState)
        target_se2 = x.get_object_pose(self._target.name).to_se2()
        staged_base_pose = get_target_robot_pose_from_parameters(
            target_se2, params[0], params[1]
        )
        staged = x.copy()
        _set_base_pose(staged, staged_base_pose)
        _set_arm_joints(staged, HOME_JOINT_POSITIONS.tolist())
        self._sim.set_state(staged)
        if (
            self._sim._robot_or_held_object_collision_exists()  # pylint: disable=protected-access
        ):
            raise TrajectorySamplingFailure("Staged base pose is in collision")
        reach_plan, pre_grasp_index = _plan_reach(self._sim, staged, self._target.name)
        predicted = staged.copy()
        _set_arm_joints(predicted, reach_plan[pre_grasp_index])
        self._sim.set_state(predicted)
        return self._sim.get_state()


class GroundGraspController(
    GroundParameterizedController[ObjectCentricState, np.ndarray],
    OutcomePredictor[ObjectCentricState],
):
    """Close the last mile of a side grasp from the pre-grasp pose.

    Takes no parameters. The arm pushes straight in along the approach axis
    from the pre-grasp pose to the grasp (the final leg of the planar reach,
    re-solved from the current state), closes the gripper, then stows: the
    reach configurations are replayed in reverse, stopping at the last
    configuration where the held cylinder stays clear of the arm body and
    the base — the carrying pose. The controller also predicts its own
    outcome (:meth:`predict_outcome`), which is what lets a planner treat
    the grasp as a magic skill and hand it to a teleoperator.
    """

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricCylinderShelf3DEnv,
    ) -> None:
        super().__init__(objects)
        self._sim = sim
        self._joint_infos = sim.robot.arm.get_arm_joint_infos()[:7]
        self._robot, self._target = objects
        self._current_approach_plan: list[JointPositions] | None = None
        self._current_stow_plan: list[JointPositions] | None = None
        self._reach_configurations: list[JointPositions] | None = None
        self._current_state: ObjectCentricState | None = None
        self._approached: bool = False
        self._closed_gripper: bool = False
        self._lifted: bool = False
        self._last_gripper_state: float = 0.0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        """The grasp has no free parameters."""
        del x, rng
        return np.zeros(0)

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        del params
        self._current_approach_plan = None
        self._current_stow_plan = None
        self._reach_configurations = None
        self._current_state = x
        self._approached = False
        self._closed_gripper = False
        self._lifted = False
        self._last_gripper_state = 0.0

    def terminated(self) -> bool:
        return self._lifted

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert isinstance(self._current_state, CylinderShelf3DObjectCentricState)

        if not self._approached:
            # Generate the final approach if it doesn't exist yet: the
            # configurations after the pre-grasp waypoint of the planar
            # reach. The full reach is kept so the stow can replay it.
            if self._current_approach_plan is None:
                reach_plan, pre_grasp_index = _plan_reach(
                    self._sim, self._current_state, self._target.name
                )
                self._reach_configurations = reach_plan
                self._current_approach_plan = _remap_joint_plan(
                    self._sim, self._current_state, reach_plan[pre_grasp_index + 1 :]
                )
            target_joints = self._current_approach_plan.pop(0)
            if not self._current_approach_plan:
                self._approached = True
            return _arm_action(self._joint_infos, target_joints, self._current_state)

        if not self._closed_gripper:
            finger_state = self._current_state.finger_state
            if finger_state > GRIPPER_CLOSE_THRESHOLD and np.isclose(
                finger_state, self._last_gripper_state, atol=0.02
            ):
                self._closed_gripper = True
            self._last_gripper_state = finger_state
            return np.array([0.0] * 10 + [-1.0], dtype=np.float32)

        if not self._lifted:
            if self._current_stow_plan is None:
                assert self._reach_configurations is not None
                stow_configs = _plan_stow(
                    self._sim, self._current_state, self._reach_configurations
                )
                self._current_stow_plan = _remap_joint_plan(
                    self._sim, self._current_state, stow_configs
                )
            target_joints = self._current_stow_plan.pop(0)
            if not self._current_stow_plan:
                self._lifted = True
            return _arm_action(self._joint_infos, target_joints, self._current_state)

        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x

    def predict_outcome(self, x: ObjectCentricState, params: Any) -> ObjectCentricState:
        """Predict the post-grasp state (holding, at the carrying pose) from ``x``."""
        del params
        assert isinstance(x, CylinderShelf3DObjectCentricState)
        return _predict_grasp_outcome(self._sim, x, self._target.name)


def _set_base_pose(state: ObjectCentricState, base_pose: SE2Pose) -> None:
    """Write an SE2 base pose into the robot's features."""
    robot = state.get_object_from_name("robot")
    state.set(robot, "pos_base_x", base_pose.x)
    state.set(robot, "pos_base_y", base_pose.y)
    state.set(robot, "pos_base_rot", base_pose.rot)


def _set_arm_joints(state: ObjectCentricState, joints: JointPositions) -> None:
    """Write the first seven entries of ``joints`` into the robot's features."""
    robot = state.get_object_from_name("robot")
    for index, value in enumerate(joints[:7]):
        state.set(robot, f"joint_{index + 1}", value)


def _set_object_pose(state: ObjectCentricState, name: str, pose: Pose) -> None:
    """Write a pose into an object's pose features."""
    obj = state.get_object_from_name(name)
    for feature, value in zip(("pose_x", "pose_y", "pose_z"), pose.position):
        state.set(obj, feature, value)
    for feature, value in zip(
        ("pose_qx", "pose_qy", "pose_qz", "pose_qw"), pose.orientation
    ):
        state.set(obj, feature, value)


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
    ) -> None:
        super().__init__(objects)
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
                for surface_z, opening in self._sim.config.get_layer_openings():
                    if opening >= cylinder_height + PLACE_VERTICAL_CLEARANCE:
                        target_surface_z = surface_z
                        break
                else:
                    raise TrajectorySamplingFailure(
                        "No shelf opening fits the cylinder"
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

                # Back off along the (near-horizontal) approach axis, plus
                # a little extra height, for the pre-place waypoint.
                pre_place_position = (
                    np.array(place_pose.position)
                    - PRE_PLACE_BACKOFF * approach_world
                    + np.array([0.0, 0.0, PRE_PLACE_LIFT])
                )

                ee_start = self._sim.robot.arm.get_end_effector_pose()
                start_pitch = _approach_pitch(matrix_from_quat(ee_start.orientation))
                path_targets, _ = _interpolated_path_targets(
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
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for CylinderShelf3D."""
    del action_space

    # Create partial controller classes that include the sim
    class MoveToPreGraspController(GroundMoveToPreGraspController):
        """Controller for staging the base and reaching the pre-grasp pose."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    class GraspController(GroundGraspController):
        """Controller for the last mile of the grasp."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    class PlaceController(GroundPlaceController):
        """Controller for placing a cylinder."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    # Create variables for lifted controllers
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)

    # Lifted controllers
    move_to_pre_grasp_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveToPreGraspController,
            Box(
                low=np.array(
                    [
                        MOVE_TO_TARGET_DISTANCE_BOUNDS[0],
                        MOVE_TO_TARGET_ROT_BOUNDS[0],
                    ]
                ),
                high=np.array(
                    [
                        MOVE_TO_TARGET_DISTANCE_BOUNDS[1],
                        MOVE_TO_TARGET_ROT_BOUNDS[1],
                    ]
                ),
            ),
        )
    )
    grasp_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target], GraspController
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
        "move_to_pre_grasp": move_to_pre_grasp_controller,
        "grasp": grasp_controller,
        "place": place_controller,
    }
