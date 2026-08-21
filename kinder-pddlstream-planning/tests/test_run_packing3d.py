"""Tests for the PDDLStream Packing3D integration."""

import itertools

import pytest
from kinder.envs.kinematic3d.packing3d import (
    ObjectCentricPacking3DEnv,
    Packing3DEnvConfig,
)
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions

from kinder_pddlstream_planning.packing3d.run import (
    HOME_BASE_POSE,
    build_stream_context,
)
from kinder_pddlstream_planning.packing3d.stream import (
    _placement_xy_window,
    _rack_grid_positions,
    sample_place_pose,
)


@pytest.fixture(name="packing_env", scope="module")
def _packing_env():
    """A reset 3-part Packing3D env: one cuboid plus two right-triangle parts."""
    sim = ObjectCentricPacking3DEnv(
        num_parts=3,
        config=Packing3DEnvConfig(robot_base_home_pose=HOME_BASE_POSE),
        use_gui=False,
        allow_state_access=True,
    )
    state, _ = sim.reset(seed=0)
    yield sim, state


def _grid_for(ctx, name):
    """The grid `sample_place_pose` enumerates before falling back to continuous."""
    bounds = ctx.part_xy_bounds[name]
    window = _placement_xy_window(ctx.rack_pose, ctx.rack_interior_half_extents, bounds)
    return _rack_grid_positions(window, bounds)


def _iter_candidate_poses(ctx, name, limit):
    """The first `limit` placement poses proposed for `name`."""
    return [pose for (pose,) in itertools.islice(sample_place_pose(ctx, name), limit)]


def test_placement_candidates_clear_the_rack_walls(packing_env):
    """Every placement clears the rack's walls, resting just above its floor.

    Regression test for a window sized off the rack's outer half extents.
    """
    sim, state = packing_env
    ctx = build_stream_context(sim, state)
    rack_id = sim._rack_id  # pylint: disable=protected-access
    floor_link, wall_links = 0, (1, 2, 3, 4)

    for name in ctx.part_names:
        body_id = sim._object_name_to_pybullet_id(  # pylint: disable=protected-access
            name
        )
        original = get_pose(body_id, sim.physics_client_id)
        try:
            for pose in _iter_candidate_poses(ctx, name, limit=32):
                set_pose(body_id, pose, sim.physics_client_id)
                for wall in wall_links:
                    assert not check_body_collisions(
                        body_id,
                        rack_id,
                        sim.physics_client_id,
                        link2=wall,
                    ), f"{name} at {pose.position} touches rack wall link {wall}"
                # Not resting on the floor, which would make the sim revert the descent,
                # but within the min_placement_dist the release check requires.
                assert not check_body_collisions(
                    body_id, rack_id, sim.physics_client_id, link2=floor_link
                ), f"{name} at {pose.position} is flush against the rack floor"
                assert check_body_collisions(
                    body_id,
                    rack_id,
                    sim.physics_client_id,
                    link2=floor_link,
                    distance_threshold=sim.config.min_placement_dist,
                ), f"{name} at {pose.position} is too high to release"
        finally:
            set_pose(body_id, original, sim.physics_client_id)


def test_placement_candidates_stay_inside_the_rack(packing_env):
    """Every placement satisfies the env's own containment check."""
    sim, state = packing_env
    ctx = build_stream_context(sim, state)

    for name in ctx.part_names:
        body_id = sim._object_name_to_pybullet_id(  # pylint: disable=protected-access
            name
        )
        original = get_pose(body_id, sim.physics_client_id)
        try:
            for pose in _iter_candidate_poses(ctx, name, limit=32):
                set_pose(body_id, pose, sim.physics_client_id)
                assert sim.get_state().is_part_inside_rack(
                    name
                ), f"{name} at {pose.position} is not inside the rack"
        finally:
            set_pose(body_id, original, sim.physics_client_id)


def test_grid_can_seat_every_part_at_once(packing_env):
    """The grid admits a collision-free arrangement of all three parts.

    The cavity fits two bounding boxes, so the third relies on triangle shape.
    """
    sim, state = packing_env
    ctx = build_stream_context(sim, state)
    grids = {name: _grid_for(ctx, name) for name in ctx.part_names}
    body_ids = {
        # pylint: disable=protected-access
        name: sim._object_name_to_pybullet_id(name)
        for name in ctx.part_names
    }
    originals = {
        name: get_pose(body_id, sim.physics_client_id)
        for name, body_id in body_ids.items()
    }
    try:
        found = False
        for cells in itertools.product(*(grids[n] for n in ctx.part_names)):
            for name, (x, y) in zip(ctx.part_names, cells):
                set_pose(
                    body_ids[name],
                    Pose((x, y, ctx.place_z[name])),
                    sim.physics_client_id,
                )
            if all(
                not check_body_collisions(
                    body_ids[one], body_ids[other], sim.physics_client_id
                )
                for one, other in itertools.combinations(ctx.part_names, 2)
            ):
                found = True
                break
        assert found, "no grid arrangement seats all parts without overlap"
    finally:
        for name, pose in originals.items():
            set_pose(body_ids[name], pose, sim.physics_client_id)


def test_placement_window_degrades_when_part_exceeds_cavity():
    """An oversized part falls back to a single centered candidate.

    An inverted range would instead sample outside the rack entirely.
    """
    rack_pose = Pose((0.3, 0.0, 0.095))
    interior = (0.09, 0.14, 0.02)
    oversized = (-0.5, 0.5, -0.5, 0.5)
    lower_x, upper_x, lower_y, upper_y = _placement_xy_window(
        rack_pose, interior, oversized
    )
    assert lower_x == upper_x == pytest.approx(rack_pose.position[0])
    assert lower_y == upper_y == pytest.approx(rack_pose.position[1])
    assert _rack_grid_positions((lower_x, upper_x, lower_y, upper_y), oversized) == [
        (rack_pose.position[0], rack_pose.position[1])
    ]
