"""Grid serialization and accounting must not hide unsuccessful trials."""

import json
import sys

import pytest

from kinder_models.dynamic3d.tossing import coverage


def test_summary_includes_every_failure_in_denominator() -> None:
    """Failed attempts remain in the denominator, including unexpected errors."""
    rows = [{"status": s} for s in ["success", "toss_miss", "pickup_failed", "error"]]
    assert coverage.summarize(rows)["success_fraction"] == 0.25
    assert coverage.summarize(rows)["trials"] == 4
    assert coverage.summarize([])["success_fraction"] is None


def test_cli_grid_and_no_overwrite(tmp_path, monkeypatch) -> None:
    """All Cartesian combinations are written and existing results are protected."""
    output = tmp_path / "coverage.jsonl"
    calls = []

    def trial(**kwargs):
        calls.append(kwargs)
        return {**kwargs, "status": "toss_miss"}

    monkeypatch.setattr(coverage, "run_trial", trial)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coverage",
            "--output",
            str(output),
            "--seeds",
            "1",
            "2",
            "--speeds",
            "350",
            "360",
            "--efforts",
            "1",
            "3",
        ],
    )
    coverage.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(calls) == 8
    assert rows[0]["kind"] == "config" and rows[-1]["kind"] == "summary"
    assert len([r for r in rows if r["kind"] == "trial"]) == 8
    assert rows[-1]["by_effort"]["1.0"]["trials"] == 4
    with pytest.raises(FileExistsError):
        coverage.main()
    assert len(calls) == 8


def test_cli_reports_errors_with_nonzero_exit(tmp_path, monkeypatch) -> None:
    """Automation can distinguish a miss from a broken experiment."""
    monkeypatch.setattr(coverage, "run_trial", lambda **kw: {**kw, "status": "error"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coverage",
            "--output",
            str(tmp_path / "error.jsonl"),
            "--seeds",
            "1",
            "--speeds",
            "350",
            "--efforts",
            "1",
        ],
    )
    with pytest.raises(SystemExit, match="1"):
        coverage.main()


def test_solution_coverage_requires_a_witness_for_every_scene() -> None:
    """Many successes on one scene must not hide a missing or unattempted scene."""
    rows = [{"seed": 1, "max_effort": 3.0, "status": "success"}] * 10
    result = coverage.solution_coverage(rows, seeds=[1, 2], efforts=[3.0])
    assert result[0]["covered"]
    assert not result[1]["covered"]
    assert result[1]["witness"] is None and result[1]["attempts"] == 0


def test_find_solutions_exits_nonzero_for_uncovered_scene(
    tmp_path, monkeypatch
) -> None:
    """A witness on seed one cannot turn the whole coverage gate green."""
    calls = []

    def trial(**kw):
        calls.append(kw)
        return {**kw, "status": "success" if kw["seed"] == 1 else "toss_miss"}

    monkeypatch.setattr(coverage, "run_trial", trial)
    output = tmp_path / "witnesses.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coverage",
            "--output",
            str(output),
            "--find-solutions",
            "--seeds",
            "1",
            "2",
            "--speeds",
            "350",
            "360",
            "--efforts",
            "3",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        coverage.main()
    assert len(calls) == 3  # one successful attempt on seed one; both on seed two
    summary = json.loads(output.read_text().splitlines()[-1])
    assert [r["covered"] for r in summary["solution_coverage"]] == [True, False]
