"""Tests for the PDDLStream MassTransport3D integration."""

# pylint: disable=protected-access

import itertools

import numpy as np
import pytest
from kinder.envs.dynamic3d.kinematic_masstransport3d import (
    ObjectCentricKinematicMassTransport3DEnv,
)
from pybullet_helpers.geometry import Pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions

from kinder_pddlstream_planning.masstransport3d import stream
from kinder_pddlstream_planning.masstransport3d.run import build_stream_context


@pytest.fixture(name="mass_env", scope="module")
def _mass_env():
    """A reset 4-item kinematic MassTransport3D env."""
    sim = ObjectCentricKinematicMassTransport3DEnv(
        num_items=4, realistic_bg=False, allow_state_access=True
    )
    state, _ = sim.reset(seed=0)
    yield sim, state


def _first_carry_conf(ctx, grasp_name="wall-x-"):
    grasp = next(g for g in ctx.tote_grasps if g.name == grasp_name)
    (carry,) = next(stream.sample_carry_conf(ctx, grasp))
    return grasp, carry


def test_slots_sit_inside_the_tote_clear_of_its_walls(mass_env):
    """Items in the slots rest on the tote floor without touching its walls."""
    sim, state = mass_env
    ctx = build_stream_context(sim, state)
    tote_pose = state.get_object_pose("tote")
    item_id = sim._object_name_to_pybullet_id(
        "item0"
    )  # pylint: disable=protected-access
    for slot in ctx.slots.values():
        set_pose(item_id, multiply_poses(tote_pose, slot), sim.physics_client_id)
        assert not check_body_collisions(item_id, sim.tote_id, sim.physics_client_id)
        assert check_body_collisions(
            item_id,
            sim.tote_id,
            sim.physics_client_id,
            distance_threshold=sim.config.min_placement_dist,
        )
    sim.reset(seed=0)


def test_destination_spots_are_spaced_reachable_and_off_the_tote(mass_env):
    """Spots fit every item, clear the tote, and leave room for the fingers."""
    sim, state = mass_env
    ctx = build_stream_context(sim, state)
    assert len(ctx.dest_spots) >= len(sim.item_grid_cells())
    for a, b in itertools.combinations(ctx.dest_spots, 2):
        assert stream.test_cfree_pose_pose(ctx, "item0", a, "item1", b)
    for spot in ctx.dest_spots:
        assert stream.test_cfree_pose_tote(ctx, "item0", spot, ctx.dest_tote_poses[0])
    # Spot every corner of the grid from the destination's fixed base conf.
    corners = [ctx.dest_spots[0], ctx.dest_spots[-1]]
    grasp = ctx.item_grasps["item0"][1]
    for spot in corners:
        outputs = list(
            stream.inverse_kinematics_stream(
                ctx, "item0", spot, grasp, "dst", ctx.table_bases["dst"]
            )
        )
        assert outputs, f"spot {spot.position} is out of reach"
    sim.reset(seed=0)


def test_carry_torque_rejects_a_heavy_load(mass_env):
    """The tote alone can be carried; a tote full of heavy items cannot."""
    sim, state = mass_env
    ctx = build_stream_context(sim, state)
    grasp, carry = _first_carry_conf(ctx)
    assert stream.test_carry_torque(ctx, grasp, carry, fluents=())
    heavy = build_stream_context(
        sim, state, item_masses={name: 5.0 for name in ctx.item_names}
    )
    full = [("InSlot", name, slot) for name, slot in zip(ctx.item_names, heavy.slots)]
    assert not stream.test_carry_torque(heavy, grasp, carry, fluents=full)


def test_tote_base_conf_puts_the_tote_in_the_carry_frame(mass_env):
    """At the computed base conf, the carry conf's low joints grasp the tote."""
    sim, state = mass_env
    ctx = build_stream_context(sim, state)
    grasp, carry = _first_carry_conf(ctx)
    tote_pose = state.get_object_pose("tote")
    outputs = list(
        stream.tote_inverse_kinematics_stream(ctx, tote_pose, "src", grasp, carry)
    )
    assert outputs, "the initial tote pose should be reachable"
    base, _ = outputs[0]
    sim.robot.set_base(base.pose)
    sim.robot.arm.set_joints(carry.low_joints)
    expected = multiply_poses(tote_pose, grasp.transform)
    assert sim.robot.arm.get_end_effector_pose().allclose(expected, atol=1e-2)
    assert base.pose.y < 0.0, "the base should stand on the source side"
    sim.reset(seed=0)


def test_cfree_pose_pose_leaves_room_for_the_fingers(mass_env):
    """Items closer than the finger gap conflict; items on different levels do not."""
    sim, state = mass_env
    ctx = build_stream_context(sim, state)
    a = Pose((0.5, 0.5, 0.43))
    assert not stream.test_cfree_pose_pose(
        ctx, "item0", a, "item1", Pose((0.55, 0.5, 0.43))
    )
    assert stream.test_cfree_pose_pose(
        ctx, "item0", a, "item1", Pose((0.62, 0.5, 0.43))
    )
    assert stream.test_cfree_pose_pose(ctx, "item0", a, "item1", Pose((0.5, 0.5, 0.6)))
    assert np.isclose(ctx.min_item_gap, 0.1)
