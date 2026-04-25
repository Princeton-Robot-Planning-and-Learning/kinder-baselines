"""Parameterized skills for the PrplLab3D environment."""

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
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DFixtureType,
)
from kinder.envs.kinematic3d.prpl3d import (
    Kinematic3DRobotType,
    ObjectCentricPrplLab3DEnv,
    PrplLab3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
)
from pybullet_helpers.geometry import Pose, SE2Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.joint import JointPositions, get_jointwise_difference
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from relational_structs import (
    Object,
    ObjectCentricState,
    Variable,
)

from kinder_models.kinematic3d.base_controllers import BasePlaceController
from kinder_models.kinematic3d.constants import (
    GRIPPER_CLOSE_THRESHOLD,
    HOME_JOINT_POSITIONS,
)
from kinder_models.kinematic3d.utils import get_target_robot_pose_from_parameters

MOVE_TO_TARGET_DISTANCE_BOUNDS = (0.45, 0.6)
MOVE_TO_TARGET_ROT_BOUNDS = (-np.pi / 4, np.pi / 4)
PLACE_X_OFFSET_BOUNDS = (-0.3, 0.3)
PLACE_Y_OFFSET_BOUNDS = (-0.1, 0.1)

COUNTER_X_FROM_FIXTURE = 1.3   # fixture base x is -1.0; counter centre x ≈ 0.3
COUNTER_Y_FROM_FIXTURE = -0.4  # fixture base y is  2.0; counter front y ≈ 1.6

# Cubes are on the floor: approach from directly above.
_PRE_GRASP_Z_OFFSET = 0.05   # m above cube centre for pre-grasp
_GRASP_Z_OFFSET = 0.005      # m above cube centre for final grasp
_FLOOR_GRASP_RPY = (np.pi, 0, np.pi / 2)  # downward-facing gripper


class GroundPickController(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Pick a cube from the floor using a top-down approach."""

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricPrplLab3DEnv,
    ) -> None:
        super().__init__(objects)
        self._sim = sim
        self._joint_infos = sim.robot.arm.get_arm_joint_infos()[:7]
        self._robot, self._target = objects
        self._current_params: np.ndarray | None = None
        self._current_base_plan: list[SE2Pose] | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_state: ObjectCentricState | None = None
        self._navigated: bool = False
        self._descended: bool = False   # arm at grasp pose (pre-grasp + descent done)
        self._closed_gripper: bool = False
        self._lifted: bool = False
        self._last_gripper_state: float = 0.0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        assert isinstance(x, PrplLab3DObjectCentricState)
        distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)  # type: ignore
        rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
        return np.array([distance, rot])

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_base_plan = None
        self._current_arm_joint_plan = None
        self._current_retract_plan = None
        self._current_state = x
        self._navigated = False
        self._descended = False
        self._closed_gripper = False
        self._lifted = False
        self._last_gripper_state = 0.0

    def terminated(self) -> bool:
        return self._lifted

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, PrplLab3DObjectCentricState)

        # ── Phase 1: navigate base to cube vicinity ───────────────────────────
        if not self._navigated:
            if self._current_base_plan is None:
                self._sim.set_state(self._current_state)
                target_pose = self._current_state.get_object_pose(
                    self.objects[1].name
                ).to_se2()
                target_base_pose = get_target_robot_pose_from_parameters(
                    target_pose, self._current_params[0], self._current_params[1]
                )
                base_plan = run_single_arm_mobile_base_motion_planning(
                    self._sim.robot,
                    self._sim.robot.base.get_pose(),
                    target_base_pose,
                    collision_bodies=self._sim._get_collision_object_ids(),  # pylint: disable=protected-access
                    seed=0,
                )
                if base_plan is None:
                    raise TrajectorySamplingFailure("Base motion planning failed")
                self._current_base_plan = base_plan[1:]

            target_base_pose = self._current_base_plan.pop(0)
            if len(self._current_base_plan) == 0:
                self._navigated = True
            delta = target_base_pose - self._current_state.base_pose
            return np.array([delta.x, delta.y, delta.rot] + [0.0] * 7 + [0.0], dtype=np.float32)

        # ── Phase 2: top-down approach + descent to grasp ────────────────────
        if self._navigated and not self._descended:
            if self._current_arm_joint_plan is None:
                self._sim.set_state(self._current_state)

                cube_pos = self._current_state.get_object_pose(self.objects[1].name).position
                pre_grasp_pose = Pose.from_rpy(
                    (cube_pos[0], cube_pos[1], cube_pos[2] + _PRE_GRASP_Z_OFFSET),
                    _FLOOR_GRASP_RPY,
                )
                grasp_pose = Pose.from_rpy(
                    (cube_pos[0], cube_pos[1], cube_pos[2] + _GRASP_Z_OFFSET),
                    _FLOOR_GRASP_RPY,
                )

                # Move to pre-grasp (above cube).
                try:
                    joint_plan1 = run_smooth_motion_planning_to_pose(
                        pre_grasp_pose,
                        self._sim.robot.arm,
                        collision_ids=set(),
                        end_effector_frame_to_plan_frame=Pose.identity(),
                        seed=0,
                        max_candidate_plans=1,
                    )
                except InverseKinematicsError:
                    joint_plan1 = None
                if joint_plan1 is None:
                    raise TrajectorySamplingFailure("Motion planning to pre-grasp failed")

                # Descend to grasp pose.
                self._sim.robot.arm.set_joints(joint_plan1[-1])
                joint_distance_fn = create_joint_distance_fn(self._sim.robot.arm)
                try:
                    joint_plan2 = smoothly_follow_end_effector_path(
                        self._sim.robot.arm,
                        [pre_grasp_pose, grasp_pose],
                        initial_joints=self._sim.robot.arm.get_joint_positions(),
                        collision_ids=set(),
                        joint_distance_fn=joint_distance_fn,
                        max_smoothing_iters_per_step=1,
                        include_start=False,
                    )
                except InverseKinematicsError:
                    joint_plan2 = None
                if joint_plan2 is None:
                    raise TrajectorySamplingFailure("Descent path following failed")

                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan1 + joint_plan2,
                    self._sim.robot.arm,
                    max_distance=self._sim.config.max_action_mag / 2,
                )
                self._current_arm_joint_plan = joint_plan[1:]

            target_joints = self._current_arm_joint_plan.pop(0)
            if len(self._current_arm_joint_plan) == 0:
                self._descended = True
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )
            return np.array([0.0] * 3 + delta_lst + [0.0], dtype=np.float32)

        # ── Phase 3: close gripper ─────────────────────────────────────────────
        if self._descended and not self._closed_gripper:
            current_grip = self._get_current_gripper_state()
            if (
                current_grip > GRIPPER_CLOSE_THRESHOLD
                and np.isclose(current_grip, self._last_gripper_state, atol=0.02)
            ):
                self._closed_gripper = True
            self._last_gripper_state = current_grip
            return np.array([0.0] * 10 + [-1.0], dtype=np.float32)

        # ── Phase 4: retract arm to home ──────────────────────────────────────
        if self._closed_gripper and not self._lifted:
            if self._current_retract_plan is None:
                self._sim.set_state(self._current_state)
                grasped_id = self._sim._grasped_object_id  # pylint: disable=protected-access
                grasped_tf = self._sim._grasped_object_transform  # pylint: disable=protected-access
                all_ids = self._sim._get_collision_object_ids()  # pylint: disable=protected-access
                joint_plan = run_motion_planning(
                    self._sim.robot.arm,
                    initial_positions=self._sim.robot.arm.get_joint_positions(),
                    target_positions=HOME_JOINT_POSITIONS.tolist(),
                    collision_bodies=all_ids - {grasped_id},
                    seed=0,
                    physics_client_id=self._sim.physics_client_id,
                    held_object=grasped_id,
                    base_link_to_held_obj=grasped_tf,
                )
                if joint_plan is None:
                    raise TrajectorySamplingFailure("Retract motion planning failed")
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    self._sim.robot.arm,
                    max_distance=self._sim.config.max_action_mag / 2,
                )
                self._current_retract_plan = joint_plan[1:]

            target_joints = self._current_retract_plan.pop(0)
            if len(self._current_retract_plan) == 0:
                self._lifted = True
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )
            return np.array([0.0] * 3 + delta_lst + [0.0], dtype=np.float32)

        raise ValueError("Invalid pick controller state")

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x

    def _get_current_gripper_state(self) -> float:
        assert self._current_state is not None
        robot_obj = self._current_state.get_object_from_name("robot")
        return self._current_state.get(robot_obj, "finger_state")


class GroundPlaceController(BasePlaceController):
    """Place a cube onto the PRPL lab countertop."""

    def __init__(self, objects: Sequence[Object], sim: ObjectCentricPrplLab3DEnv) -> None:
        # The PRPL lab URDF has a dense mesh; give smooth MP enough time to
        # find a collision-free path around the cabinet geometry.
        super().__init__(objects, sim, smooth_mp_max_time=10.0)

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        assert isinstance(x, PrplLab3DObjectCentricState)
        x_offset = rng.uniform(*PLACE_X_OFFSET_BOUNDS)  # type: ignore
        y_offset = rng.uniform(*PLACE_Y_OFFSET_BOUNDS)  # type: ignore
        return np.array([x_offset, y_offset])

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x
        self._navigated = False
        self._pre_place = False
        self._opened_gripper = False
        self._lifted = False
        self._current_arm_joint_plan = None
        self._current_retract_plan = None
        self._target_place_pose_world = None
        self._pre_place_pose_world = None

    def terminated(self) -> bool:
        return self._lifted

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, PrplLab3DObjectCentricState)

        if self._current_plan is None:
            self._sim.set_state(self._current_state)

            grasped_object_id = self._sim._grasped_object_id  # pylint: disable=protected-access
            grasped_object_transform = self._sim._grasped_object_transform  # pylint: disable=protected-access
            if grasped_object_transform is None:
                raise TrajectorySamplingFailure("Nothing grasped at place time")
            assert grasped_object_transform is not None

            # Compute placement pose relative to the prpl_lab fixture.
            fixture_pose = self._current_state.get_object_pose(self.objects[2].name)
            target_x = fixture_pose.position[0] + COUNTER_X_FROM_FIXTURE + self._current_params[0]
            target_y = fixture_pose.position[1] + COUNTER_Y_FROM_FIXTURE + self._current_params[1]
            block_half_z = self._sim.config.block_half_extents[2]
            target_z = self._sim._counter_top_z + block_half_z

            desired_object_pose = Pose((target_x, target_y, target_z), (0, 0, 0, 1))

            # Derive EE z from the grasp transform; use a fixed downward orientation.
            ee_z = multiply_poses(
                desired_object_pose, grasped_object_transform.invert()
            ).position[2]
            self._target_place_pose_world = Pose.from_rpy(
                (target_x, target_y, ee_z + 1e-3),
                (-np.pi / 2, np.pi, 0),
            )
            self._pre_place_pose_world = Pose(
                (
                    self._target_place_pose_world.position[0],
                    self._target_place_pose_world.position[1] - 0.1,
                    self._target_place_pose_world.position[2] + 0.02,
                ),
                self._target_place_pose_world.orientation,
            )

            target_base_pose = get_target_robot_pose_from_parameters(
                SE2Pose(target_x, target_y, 0.0), 0.9, np.pi / 2
            )
            all_collision_ids = self._sim._get_collision_object_ids()  # pylint: disable=protected-access
            base_plan = run_single_arm_mobile_base_motion_planning(
                self._sim.robot,
                self._sim.robot.base.get_pose(),
                target_base_pose,
                collision_bodies=all_collision_ids - {grasped_object_id},
                seed=0,
                held_object=grasped_object_id,
                base_link_to_held_obj=grasped_object_transform,
            )
            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")
            self._current_plan = base_plan[1:]

        if not self._navigated:
            return self.navigate()
        if self._navigated and not self._pre_place:
            return self.pre_place()
        if self._pre_place and not self._opened_gripper:
            return self.open_gripper()
        if self._opened_gripper and not self._lifted:
            return self.lift()

        raise ValueError("Invalid place controller state")


def create_lifted_controllers(
    action_space: Kinematic3DRobotActionSpace,
    sim: ObjectCentricPrplLab3DEnv,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for PrplLab3D."""
    del action_space

    class PickController(GroundPickController):
        def __init__(self, objects):
            super().__init__(objects, sim)

    class PlaceController(GroundPlaceController):
        def __init__(self, objects):
            super().__init__(objects, sim)

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)

    pick_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target],
        PickController,
        Box(
            low=np.array([MOVE_TO_TARGET_DISTANCE_BOUNDS[0], MOVE_TO_TARGET_ROT_BOUNDS[0]]),
            high=np.array([MOVE_TO_TARGET_DISTANCE_BOUNDS[1], MOVE_TO_TARGET_ROT_BOUNDS[1]]),
        ),
    )

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    target_fixture = Variable("?target_fixture", Kinematic3DFixtureType)

    place_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target, target_fixture],
        PlaceController,
        Box(
            low=np.array([PLACE_X_OFFSET_BOUNDS[0], PLACE_Y_OFFSET_BOUNDS[0]]),
            high=np.array([PLACE_X_OFFSET_BOUNDS[1], PLACE_Y_OFFSET_BOUNDS[1]]),
        ),
    )

    return {
        "pick": pick_controller,
        "place": place_controller,
    }
