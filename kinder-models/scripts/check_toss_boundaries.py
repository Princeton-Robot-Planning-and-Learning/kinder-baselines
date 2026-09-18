"""Physical witness search at placement edges; run with --regions and --output.

The input maps placement-family names to production ground-region definitions.
Each case narrows a region near a corner or edge midpoint; the production sampler
still enforces footprint, room and furniture clearance. Empty cells are reported
separately, never counted as solved. This is a finite audit, not a universal proof.
"""

import argparse
import copy
import itertools
import json
from pathlib import Path

from kinder_models.dynamic3d.tossing.coverage import run_trial


def boundary_cells(region: dict, width: float = 0.34) -> list[tuple[str, dict]]:
    """Eight perimeter cells per rectangle, including all corners.

    Width 0.34 leaves a 3 cm center band after the 30 cm bin footprint and
    the production sampler's 5 mm clearance on each side are accounted for.
    """
    cells = []
    for i, (x0, y0, x1, y1) in enumerate(region["ranges"]):
        if x1 - x0 < width or y1 - y0 < width:
            raise ValueError("Region must be wider than the boundary cell")
        for ix, iy in itertools.product(range(3), repeat=2):
            if ix == iy == 1:
                continue
            left = (x0, (x0 + x1 - width) / 2, x1 - width)[ix]
            bottom = (y0, (y0 + y1 - width) / 2, y1 - width)[iy]
            cell = copy.deepcopy(region)
            cell["ranges"] = [[left, bottom, left + width, bottom + width]]
            cell["yaw_ranges"] = [region["yaw_ranges"][i]]
            cells.append((f"r{i}-x{ix}-y{iy}", cell))
    return cells


def main() -> None:
    """Search a bounded candidate bank and preserve every physical attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10300)
    parser.add_argument(
        "--precut-regions",
        action="store_true",
        help="Treat each named region as an already selected boundary cell.",
    )
    parser.add_argument(
        "--cube-region",
        type=Path,
        help="Instead probe cube-region edges in each bin placement family.",
    )
    args = parser.parse_args()
    regions = json.loads(args.regions.read_text())
    cube_region = json.loads(args.cube_region.read_text()) if args.cube_region else None
    short = [(d, 140, t) for d in (1.35, 1.3, 1.4) for t in (780, 760, 800)]
    long = [
        (d, s, t)
        for d, s in ((2.5, 360), (2.6, 380), (2.4, 340))
        for t in (500, 480, 520)
    ]
    results = []
    with args.output.open("x") as stream:

        def record(value: dict) -> None:
            stream.write(json.dumps(value) + "\n")
            stream.flush()

        record(
            dict(
                kind="config",
                regions=regions,
                seed=args.seed,
                candidates=short + long,
                max_effort=3,
                rotation=0,
                cube_region=cube_region,
            )
        )
        for family, region in regions.items():
            cells = (
                [("cell", region)]
                if args.precut_regions
                else (
                    boundary_cells(cube_region, width=0.09)
                    if cube_region
                    else boundary_cells(region)
                )
            )
            for case, cell in cells:
                outcome = "uncovered"
                witness = None
                candidates = long + short if "evaluation" in family else short + long
                for distance, speed, release in candidates:
                    row = run_trial(
                        seed=args.seed,
                        distance=distance,
                        speed=speed,
                        release_ms=release,
                        max_effort=3,
                        bin_reset_region=region if cube_region else cell,
                        cube_reset_region=cell if cube_region else None,
                    )
                    record(dict(kind="trial", family=family, case=case, **row))
                    if row["status"] == "success":
                        outcome, witness = "solved", row
                        break
                    if row["status"] == "error":
                        outcome = (
                            "empty_cell"
                            if row.get("stage") == "reset"
                            and "No feasible ground placement region"
                            in row.get("error", "")
                            else "error"
                        )
                        break
                    if row["status"].startswith("pickup_"):
                        break  # Toss parameters cannot repair the identical pickup.
                result = dict(
                    family=family, case=case, outcome=outcome, witness=witness
                )
                results.append(result)
                record(dict(kind="case_result", **result))
                print(f"{family} {case}: {outcome}", flush=True)
        record(dict(kind="summary", cases=results))
    if any(r["outcome"] != "solved" for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
