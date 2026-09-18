"""Boundary-case generation must preserve geometry and expose empty cells."""

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[3] / "scripts" / "check_toss_boundaries.py"
    spec = importlib.util.spec_from_file_location("check_toss_boundaries", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boundary_cells_cover_corners_and_edge_midpoints() -> None:
    """All eight perimeter locations are generated inside the original region."""
    region = {
        "target": "ground",
        "ranges": [[2.6, -2.3, 3.42, 2.3]],
        "yaw_ranges": [[0, 0]],
    }
    cells = _module().boundary_cells(region)
    assert len(cells) == 8
    assert not any(name.endswith("x1-y1") for name, _ in cells)
    for _, cell in cells:
        x0, y0, x1, y1 = cell["ranges"][0]
        assert 2.6 <= x0 < x1 <= 3.42
        assert -2.3 <= y0 < y1 <= 2.3
        assert x1 - x0 == pytest.approx(0.34)
        assert y1 - y0 == pytest.approx(0.34)
        assert cell["yaw_ranges"] == [[0, 0]]
    assert region["ranges"] == [[2.6, -2.3, 3.42, 2.3]]


def test_boundary_cells_reject_too_small_regions() -> None:
    """Do not silently expand a requested region to accommodate a test cell."""
    with pytest.raises(ValueError, match="wider"):
        _module().boundary_cells({"ranges": [[0, 0, 0.1, 0.1]]})
