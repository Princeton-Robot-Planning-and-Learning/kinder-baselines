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
