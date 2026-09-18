"""Select tight cells on the furnished production sampler's feasible-bin boundary."""

import argparse
import json
from pathlib import Path

import kinder
import numpy as np
from kinder.envs.dynamic3d.envs import _object_world_axis_aligned_bbox
from shapely import Polygon, box, union_all
from shapely.geometry import mapping

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--regions", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed", type=int, default=10300)
args = parser.parse_args()
if args.output.exists() or args.output.with_suffix(".polygons.json").exists():
    parser.error("Output already exists")
kinder.register_all_environments()
env = kinder.make("kinder/Tossing3D-o1-v0", allow_state_access=True, scene_bg=True)
env.reset(seed=args.seed)
scene = env.unwrapped._object_centric_env
bin_geometry = scene.get_object("bin_0")
assert np.allclose(
    [bin_geometry.length, bin_geometry.width, bin_geometry.height], [0.3, 0.3, 0.2]
), "Update the boundary fixture for changed bin geometry"
regions = json.loads(args.regions.read_text())
obstacles = [
    _object_world_axis_aligned_bbox(obj._get_object_centric_data())
    for name, obj in scene._objects_dict.items()
    if name not in {"cube_0", "bin_0"}
]
obstacles += [
    _object_world_axis_aligned_bbox(c)
    for c in scene._get_static_collision_boxes().values()
]
obstacles += scene._get_static_mesh_placement_bounds()
cells = {}
polygons = {}
for family, region in regions.items():
    x0, y0, x1, y1 = region["ranges"][0]
    half = np.array([0.15, 0.15])
    clearance = 0.005
    vertices = [
        np.array(p)
        for p in (
            (x0 + 0.155, y0 + 0.155),
            (x1 - 0.155, y0 + 0.155),
            (x1 - 0.155, y1 - 0.155),
            (x0 + 0.155, y1 - 0.155),
        )
    ]
    for normal, offset in scene._placement_room_halfspaces():
        threshold = offset + abs(normal) @ half + clearance
        clipped = []
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            da, db = normal @ a - threshold, normal @ b - threshold
            if da >= 0:
                clipped.append(a)
            if (da >= 0) != (db >= 0):
                clipped.append(a + da / (da - db) * (b - a))
        vertices = clipped
    blocked = [
        box(o[0] - 0.155, o[1] - 0.155, o[3] + 0.155, o[4] + 0.155)
        for o in obstacles
        if o[5] > 0 and o[2] < 0.2
    ]
    feasible = Polygon(vertices).difference(union_all(blocked))
    polygons[family] = mapping(feasible)
    # Inset 5 mm: the 2 mm-wide center cell stays safely on the feasible side.
    safe = feasible.buffer(-0.005, join_style=2)
    pieces = [safe] if safe.geom_type == "Polygon" else list(safe.geoms)
    for pi, piece in enumerate(pieces):
        for ri, ring in enumerate([piece.exterior, *piece.interiors]):
            points = list(ring.coords)[:-1]
            for i, (a, b) in enumerate(zip(points, points[1:] + points[:1])):
                for typ, p in [("vertex", a), ("edge", (np.array(a) + b) / 2)]:
                    x, y = p
                    w = 0.156
                    cells[f"{family}-p{pi}-r{ri}-{typ}{i}"] = {
                        "target": "ground",
                        "ranges": [[x - w, y - w, x + w, y + w]],
                        "yaw_ranges": region["yaw_ranges"],
                    }
args.output.write_text(json.dumps(cells, indent=2))
args.output.with_suffix(".polygons.json").write_text(json.dumps(polygons, indent=2))
print({family: sum(k.startswith(family) for k in cells) for family in regions})
env.close()
