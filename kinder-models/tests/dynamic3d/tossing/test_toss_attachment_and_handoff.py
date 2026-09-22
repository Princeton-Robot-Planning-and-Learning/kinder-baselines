"""Regression coverage for observed cargo geometry and windup handoff."""

# These regressions inspect planning geometry and phase state directly.
# pylint: disable=protected-access

import json
from pathlib import Path

import numpy as np
import pybullet as p
import pytest
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from kinder.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoObjectTypeFeatures,
    MujocoStaticColliderType,
    MujocoTidyBotRobotObjectType,
)
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
    check_collisions_with_held_object,
)
from pybullet_helpers.link import get_link_state
from relational_structs import Object, ObjectCentricState

from kinder_models.dynamic3d.tossing.parameterized_skills import (
    MoveToTossLocationAndTossController,
)
from kinder_models.dynamic3d.tossing.toss_swing import (
    TOSS_RELEASE_ARM_CONFIGURATION,
    TOSS_WINDUP_ARM_CONFIGURATION,
)


@pytest.fixture(name="recorded_controller", params=[0, 1])
def recorded_controller_fixture(request):
    """Real failed initializations: the held cube used to block its own windup."""
    rows = json.loads(
        (Path(__file__).with_name("held_cube_regression_states.json")).read_text(
            encoding="utf-8"
        )
    )
    row = rows[request.param]
    data = {}
    for name, values in row["before_state"].items():
        if name == "robot":
            kind = MujocoTidyBotRobotObjectType
        elif name in ("cube_0", "bin_0"):
            kind = MujocoMovableObjectType
        elif name.startswith("collider:"):
            kind = MujocoStaticColliderType
        else:
            kind = MujocoFixtureObjectType
        data[Object(name, kind)] = np.asarray(values, dtype=np.float32)
    state = ObjectCentricState(data, MujocoObjectTypeFeatures)
    controller = MoveToTossLocationAndTossController(
        tuple(
            state.get_object_from_name(n) for n in ("robot", "cube_0", "cuboid_barrier")
        )
    )
    params = np.asarray(row["selected_params"], dtype=float)
    params[2] = np.deg2rad(params[2])
    try:
        controller.reset(state, params)
        yield controller, state
    finally:
        if controller._pybullet_sim is not None:
            controller._pybullet_sim.close()


def test_saved_false_collision_and_rigid_grasp(recorded_controller):
    """Both formerly rejected states initialize and preserve the measured grasp."""
    controller, state = recorded_controller
    sim = controller._pybullet_sim
    assert controller._swing is not None and len(controller._windup_trajectory) > 0
    cube, held = controller._configure_held_cube(state)
    observed = get_pose(held, sim.physics_client_id)
    # Re-applying the attachment in the same frame must not snap the cube to a
    # nominal grasp or confuse link-frame and center-of-mass transforms.
    sim.set_state(state, held_object=cube)
    assert get_pose(held, sim.physics_client_id).allclose(observed)
    moved = state.copy()
    final = controller._current_base_motion_plan[-1]
    robot = controller.objects[0]
    for name, value in (
        ("pos_base_x", final.x),
        ("pos_base_y", final.y),
        ("pos_base_rot", final.theta()),
    ):
        moved.set(robot, name, value)
    sim.set_state(moved, held_object=cube)
    ee = get_link_state(
        sim.robot.robot_id, sim.robot.end_effector_id, sim.physics_client_id
    ).com_pose
    assert get_pose(held, sim.physics_client_id).allclose(
        multiply_poses(ee, sim.base_link_to_held_obj)
    )
    assert not get_pose(held, sim.physics_client_id).allclose(observed, atol=1e-3)


def test_attached_cube_still_blocks_obstacle_motion(recorded_controller):
    """A cube-only obstacle collision rejects the complete planned windup."""
    controller, state = recorded_controller
    sim = controller._pybullet_sim
    cube, held = controller._configure_held_cube(state)
    moved = state.copy()
    robot = controller.objects[0]
    final = controller._current_base_motion_plan[-1]
    for name, value in (
        ("pos_base_x", final.x),
        ("pos_base_y", final.y),
        ("pos_base_rot", final.theta()),
    ):
        moved.set(robot, name, value)
    sim.set_state(moved, held_object=cube)
    pose = get_pose(held, sim.physics_client_id)
    half = np.array([state.get(cube, f"bb_{a}") for a in ("x", "y", "z")]) / 2
    shape = p.createCollisionShape(
        p.GEOM_SPHERE, radius=0.003, physicsClientId=sim.physics_client_id
    )
    obstacle = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=shape, physicsClientId=sim.physics_client_id
    )
    # Place a tiny obstacle against a cube face that does not touch the arm.
    for axis in range(3):
        for sign in (-1, 1):
            offset = np.zeros(3)
            offset[axis] = sign * (half[axis] + 0.002)
            point = multiply_poses(pose, Pose(tuple(offset))).position
            p.resetBasePositionAndOrientation(
                obstacle, point, [0, 0, 0, 1], physicsClientId=sim.physics_client_id
            )
            if check_body_collisions(
                held, obstacle, sim.physics_client_id
            ) and not check_body_collisions(
                sim.robot.robot_id, obstacle, sim.physics_client_id
            ):
                break
        else:
            continue
        break
    else:
        pytest.fail("No isolated cube face was available for the obstacle fixture")
    joints = sim.get_robot_joints()
    assert not check_collisions_with_held_object(
        sim.robot, {obstacle}, sim.physics_client_id, None, None, joints
    )
    assert check_collisions_with_held_object(
        sim.robot,
        {obstacle},
        sim.physics_client_id,
        held,
        sim.base_link_to_held_obj,
        joints,
    )
    sim._static_colliders["attachment_regression_obstacle"] = obstacle
    with pytest.raises(TrajectorySamplingFailure, match="Motion planning failed"):
        controller._plan_arm_toss(state, final)


def test_windup_waits_for_observation_and_starts_at_observed_pose(
    recorded_controller, monkeypatch
):
    """Elapsed profile time alone cannot start a moving or inaccurate throw."""
    controller, _ = recorded_controller
    phase = controller.MoveToTossLocationAndTossControllerPhase
    controller._phase = phase.WINDUP
    controller._windup_step_idx = len(controller._windup_trajectory) - 1
    current = np.asarray(TOSS_WINDUP_ARM_CONFIGURATION).copy()
    current[0] += 0.002  # Small real tracking offset, retained in the timed swing.
    monkeypatch.setattr(
        controller, "_get_current_robot_arm_conf", lambda: current.tolist() + [0.0] * 6
    )
    controller._previous_windup_arm_conf = current.copy()
    controller._action_windup()
    assert controller._phase is phase.WINDUP  # Last command is not its observation.
    controller._previous_windup_arm_conf = current + 0.02
    hold = controller._action_windup()
    assert controller._phase is phase.WINDUP
    assert hold.shape == (11,)  # Position-only hold has no feedforward channel.
    current[0] += 0.1
    controller._previous_windup_arm_conf = current.copy()
    controller._action_windup()
    assert controller._phase is phase.WINDUP  # Stationary but off target.
    current[0] -= 0.1
    controller._previous_windup_arm_conf = current.copy()
    # The physical collision behavior is tested above. Isolate this test to the
    # observation/time boundary and verify the actual planned start reaches timing.
    monkeypatch.setattr(controller, "_configure_held_cube", lambda _: (None, 0))
    monkeypatch.setattr(
        "kinder_models.dynamic3d.tossing.parameterized_skills.run_motion_planning",
        lambda robot, start, target, **kwargs: [start, target],
    )
    controller._release_arm_configuration = (
        list(TOSS_RELEASE_ARM_CONFIGURATION) + [0.0] * 6
    )
    controller._action_windup()
    assert controller._phase is phase.SWING
    assert np.array_equal(controller._swing.start_joint_angles, current)
    assert controller._swing_step_idx == 1
