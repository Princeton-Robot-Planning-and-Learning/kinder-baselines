"""Interactively record arm waypoints for the CylinderShelf3D pick skill.

Opens the CylinderShelf3D environment in the PyBullet GUI with a canonical
pick scene: the robot base at the origin facing +x, and a cylinder staged
directly in front of it at the given standoff distance. Joint sliders move
the arm; buttons record the current configuration as a waypoint, undo the
last one, toggle the gripper, or save and exit.

The saved JSON contains the staged scene parameters and, per waypoint, the
arm joint positions, the finger state, and the end-effector pose (forward
kinematics) — everything needed to replay the demonstrated reach in the
pick skill.

Usage (from the kinder-models directory):

    python scripts/record_cylinder_shelf3d_waypoints.py \
        --output src/kinder_models/kinematic3d/cylinder_shelf3d/reach_waypoints.json

Buttons:
    Record waypoint  — snapshot the current arm configuration.
    Undo last        — remove the most recently recorded waypoint.
    Toggle gripper   — open/close the fingers (finger state is recorded
                       with each waypoint).
    Save and quit    — write the JSON file and exit.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
from kinder.envs.kinematic3d.cylinder_shelf3d import (
    CylinderShelf3DEnvConfig,
    ObjectCentricCylinderShelf3DEnv,
)
from kinder.envs.kinematic3d.utils import extend_joints_to_include_fingers
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.gui import visualize_pose
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot


def _read_button(button_id: int, physics_client_id: int) -> float:
    try:
        return p.readUserDebugParameter(button_id, physicsClientId=physics_client_id)
    except p.error:
        return 0.0


def main() -> None:
    """Run the interactive waypoint recorder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cylinder_shelf3d_waypoints.json"),
        help="Where to write the recorded waypoints JSON.",
    )
    parser.add_argument(
        "--cylinder-height",
        type=float,
        default=CylinderShelf3DEnvConfig().cylinder_heights[0],
        help="Height of the staged cylinder.",
    )
    parser.add_argument(
        "--standoff",
        type=float,
        default=0.6,
        help="Distance from the base origin to the staged cylinder axis.",
    )
    args = parser.parse_args()

    config = CylinderShelf3DEnvConfig(
        cylinder_heights=(args.cylinder_height,),
    )
    env = ObjectCentricCylinderShelf3DEnv(
        num_cylinders=1, config=config, use_gui=True, realistic_bg=False
    )
    env.reset(seed=0)
    client = env.physics_client_id

    # Stage the canonical pick scene: base at the origin facing +x, the
    # cylinder straight ahead at the standoff distance.
    cylinder_position = (args.standoff, 0.0, args.cylinder_height / 2)
    cylinder_id = env._cylinders["cylinder0"]  # pylint: disable=protected-access
    set_pose(cylinder_id, Pose(cylinder_position), client)

    robot = env.robot
    arm = robot.arm
    assert isinstance(arm, FingeredSingleArmPyBulletRobot)
    arm_joint_names = arm.arm_joint_names[:7]
    initial_joints = arm.get_joint_positions()

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, True, physicsClientId=client)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.4,
        cameraYaw=50,
        cameraPitch=-25,
        cameraTargetPosition=(args.standoff / 2, 0, 0.3),
        physicsClientId=client,
    )

    slider_ids = []
    for i, joint_name in enumerate(arm_joint_names):
        lower, upper = arm.get_joint_limits_from_name(joint_name)
        if np.isinf(lower):
            lower = -2 * np.pi
        if np.isinf(upper):
            upper = 2 * np.pi
        slider_ids.append(
            p.addUserDebugParameter(
                paramName=joint_name,
                rangeMin=lower,
                rangeMax=upper,
                startValue=initial_joints[i],
                physicsClientId=client,
            )
        )
    record_button = p.addUserDebugParameter(
        "Record waypoint", 0, -1, 0, physicsClientId=client
    )
    undo_button = p.addUserDebugParameter("Undo last", 0, -1, 0, physicsClientId=client)
    gripper_button = p.addUserDebugParameter(
        "Toggle gripper", 0, -1, 0, physicsClientId=client
    )
    save_button = p.addUserDebugParameter(
        "Save and quit", 0, -1, 0, physicsClientId=client
    )

    waypoints: list[dict] = []
    waypoint_frame_ids: list[set[int]] = []
    last_record = _read_button(record_button, client)
    last_undo = _read_button(undo_button, client)
    last_gripper = _read_button(gripper_button, client)
    last_save = _read_button(save_button, client)
    gripper_closed = False

    print(f"Recording waypoints; will save to {args.output}")
    print("Move the joint sliders, then click 'Record waypoint' for each pose.")

    while True:
        joint_positions = []
        for slider_id in slider_ids:
            try:
                joint_positions.append(
                    p.readUserDebugParameter(slider_id, physicsClientId=client)
                )
            except p.error:
                break
        if len(joint_positions) != len(slider_ids):
            time.sleep(0.02)
            continue
        arm.set_joints(extend_joints_to_include_fingers(joint_positions))
        closed_state = arm.closed_fingers_state if gripper_closed else 0.0
        arm.set_finger_state(closed_state)

        value = _read_button(gripper_button, client)
        if value != last_gripper:
            last_gripper = value
            gripper_closed = not gripper_closed
            print(f"Gripper {'closed' if gripper_closed else 'open'}")

        value = _read_button(record_button, client)
        if value != last_record:
            last_record = value
            ee_pose = arm.get_end_effector_pose()
            waypoints.append(
                {
                    "joints": list(joint_positions),
                    "finger_state": float(arm.get_finger_state()),
                    "ee_position": list(ee_pose.position),
                    "ee_orientation": list(ee_pose.orientation),
                }
            )
            waypoint_frame_ids.append(visualize_pose(ee_pose, physics_client_id=client))
            print(
                f"Recorded waypoint {len(waypoints)}: "
                f"ee={np.round(ee_pose.position, 3).tolist()}"
            )

        value = _read_button(undo_button, client)
        if value != last_undo:
            last_undo = value
            if waypoints:
                waypoints.pop()
                for frame_id in waypoint_frame_ids.pop():
                    p.removeUserDebugItem(frame_id, physicsClientId=client)
                print(f"Removed last waypoint; {len(waypoints)} remain")
            else:
                print("No waypoints to remove")

        value = _read_button(save_button, client)
        if value != last_save:
            data = {
                "scene": {
                    "base_pose": [0.0, 0.0, 0.0],
                    "cylinder_position": list(cylinder_position),
                    "cylinder_height": args.cylinder_height,
                    "cylinder_radius": config.cylinder_radius,
                    "standoff": args.standoff,
                },
                "waypoints": waypoints,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(waypoints)} waypoints to {args.output}")
            break

        time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main()
