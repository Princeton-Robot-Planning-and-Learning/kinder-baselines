"""Challenge library and similarity scoring for the drawing feature."""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Trail segment type (same format the executor produces).
# Keys: x1 y1 x2 y2 r g b
# ---------------------------------------------------------------------------
TrailSegment = dict[str, float]


# ---------------------------------------------------------------------------
# Helper: build trail segments from a list of waypoints + a single colour.
# ---------------------------------------------------------------------------
def _segments_from_waypoints(
    waypoints: list[tuple[float, float]],
    r: int, g: int, b: int,
    closed: bool = False,
) -> list[TrailSegment]:
    pts = list(waypoints)
    if closed and len(pts) >= 2 and pts[-1] != pts[0]:
        pts.append(pts[0])
    segs: list[TrailSegment] = []
    for i in range(len(pts) - 1):
        segs.append({
            "x1": pts[i][0], "y1": pts[i][1],
            "x2": pts[i + 1][0], "y2": pts[i + 1][1],
            "r": r, "g": g, "b": b,
        })
    return segs


# ---------------------------------------------------------------------------
# Challenge library
# ---------------------------------------------------------------------------
CHALLENGES: list[dict] = [
    {
        "id": "square",
        "name": "Square",
        "difficulty": "easy",
        "description": "Draw a red square (side length 1, centered at the origin).",
        "hint": (
            "Move to one corner, put the pen down, "
            "then visit the other three corners and come back."
        ),
        "target_trail": _segments_from_waypoints(
            [(-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5)],
            r=255, g=0, b=0, closed=True,
        ),
    },
    {
        "id": "triangle",
        "name": "Triangle",
        "difficulty": "easy",
        "description": "Draw a blue triangle.",
        "hint": "Three points, three moves, close the shape!",
        "target_trail": _segments_from_waypoints(
            [(0.0, 1.0), (1.0, -0.5), (-1.0, -0.5)],
            r=0, g=80, b=255, closed=True,
        ),
    },
    {
        "id": "letter_l",
        "name": "Letter L",
        "difficulty": "easy",
        "description": "Draw a green letter L.",
        "hint": "Two straight lines: one going down, one going right.",
        "target_trail": _segments_from_waypoints(
            [(0.5, 1.0), (0.5, -0.5), (-0.5, -0.5)],
            r=0, g=200, b=0, closed=False,
        ),
    },
    {
        "id": "diamond",
        "name": "Diamond",
        "difficulty": "medium",
        "description": "Draw a red diamond (rotated square).",
        "hint": "Four diagonal moves connecting top, right, bottom, left.",
        "target_trail": _segments_from_waypoints(
            [(0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)],
            r=255, g=0, b=0, closed=True,
        ),
    },
    {
        "id": "zigzag",
        "name": "Zigzag",
        "difficulty": "medium",
        "description": "Draw an orange zigzag line from left to right.",
        "hint": "Alternate between moving up-right and down-right.",
        "target_trail": _segments_from_waypoints(
            [
                (1.0, 0.0), (0.5, 0.5), (0.0, 0.0),
                (-0.5, 0.5), (-1.0, 0.0),
            ],
            r=255, g=140, b=0, closed=False,
        ),
    },
    {
        "id": "house",
        "name": "House",
        "difficulty": "hard",
        "description": "Draw a house: red square base with a blue triangle roof.",
        "hint": "You will need to change pen colour halfway through. "
                "Draw the square first, then pen up, move to the roof start, "
                "change to blue, and draw the triangle.",
        "target_trail": (
            # base (red square)
            _segments_from_waypoints(
                [(0.5, -0.5), (0.5, 0.5), (0.0, 0.5), (0.0, -0.5)],
                r=255, g=0, b=0, closed=True,
            )
            +
            # roof (blue triangle)
            _segments_from_waypoints(
                [(0.0, -0.5), (-1.0, 0.0), (0.0, 0.5)],
                r=0, g=80, b=255, closed=False,
            )
        ),
    },
    {
        "id": "star",
        "name": "Star",
        "difficulty": "hard",
        "description": "Draw a yellow five-pointed star.",
        "hint": (
            "A star is drawn by connecting every other vertex of a regular pentagon."
        ),
        "target_trail": _segments_from_waypoints(
            [
                (0.0, 1.0),
                (-0.5, -0.5),
                (1.0, 0.5),
                (-1.0, 0.5),
                (0.5, -0.5),
            ],
            r=220, g=180, b=0, closed=True,
        ),
    },
]

_CHALLENGE_BY_ID = {c["id"]: c for c in CHALLENGES}


def get_challenge(challenge_id: str) -> dict | None:
    """Return a challenge dict by id, or None."""
    return _CHALLENGE_BY_ID.get(challenge_id)


def list_challenges() -> list[dict]:
    """Return the list of challenges (without the full trail data)."""
    return [
        {
            "id": c["id"],
            "name": c["name"],
            "difficulty": c["difficulty"],
            "description": c["description"],
            "hint": c["hint"],
        }
        for c in CHALLENGES
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _sample_points(
    trail: list[TrailSegment], spacing: float = 0.02
) -> list[tuple[float, float, int, int, int]]:
    """Sample evenly-spaced (x, y, r, g, b) points along a trail."""
    pts: list[tuple[float, float, int, int, int]] = []
    for seg in trail:
        dx = seg["x2"] - seg["x1"]
        dy = seg["y2"] - seg["y1"]
        length = math.sqrt(dx * dx + dy * dy)
        n = max(int(length / spacing), 1)
        for i in range(n + 1):
            t = i / n
            pts.append((
                seg["x1"] + t * dx,
                seg["y1"] + t * dy,
                int(seg["r"]), int(seg["g"]), int(seg["b"]),
            ))
    return pts


def score_trail(
    student_trail: list[TrailSegment],
    target_trail: list[TrailSegment],
    tolerance: float = 0.12,
) -> dict:
    """Score a student trail against a target.

    Returns a dict with ``score`` (0-100) and a ``breakdown`` dict.
    """
    target_pts = _sample_points(target_trail)
    student_pts = _sample_points(student_trail)

    if not target_pts:
        return {"score": 100 if not student_pts else 0,
                "breakdown": {"coverage": 100, "precision": 100, "color": 100}}
    if not student_pts:
        return {"score": 0,
                "breakdown": {"coverage": 0, "precision": 0, "color": 0}}

    # Build simple arrays for distance calc.
    t_xy = [(p[0], p[1]) for p in target_pts]
    s_xy = [(p[0], p[1]) for p in student_pts]

    # Coverage: fraction of target points within *tolerance* of any student point.
    covered = 0
    color_diffs: list[float] = []
    for tp in target_pts:
        best_dist = float("inf")
        best_j = 0
        for j, sp_xy in enumerate(s_xy):
            d = math.hypot(sp_xy[0] - tp[0], sp_xy[1] - tp[1])
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_dist <= tolerance:
            covered += 1
            sp = student_pts[best_j]
            cdiff = (
                abs(tp[2] - sp[2]) + abs(tp[3] - sp[3]) + abs(tp[4] - sp[4])
            ) / (3.0 * 255.0)
            color_diffs.append(1.0 - cdiff)

    coverage = covered / len(target_pts)

    # Precision: fraction of student points near a target point.
    precise = 0
    for sp_xy in s_xy:
        for tp_xy in t_xy:
            if math.hypot(tp_xy[0] - sp_xy[0], tp_xy[1] - sp_xy[1]) <= tolerance:
                precise += 1
                break
    precision = precise / len(student_pts)

    color_score = sum(color_diffs) / len(color_diffs) if color_diffs else 0.0

    raw = 0.45 * coverage + 0.30 * precision + 0.25 * color_score
    final = round(raw * 100)

    return {
        "score": final,
        "breakdown": {
            "coverage": round(coverage * 100),
            "precision": round(precision * 100),
            "color": round(color_score * 100),
        },
    }
