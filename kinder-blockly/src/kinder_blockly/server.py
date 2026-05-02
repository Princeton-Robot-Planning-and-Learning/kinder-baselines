"""Flask server for the Blockly visual programming interface."""

import base64
import io
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, request, stream_with_context
from PIL import Image

from kinder_blockly.challenges import get_challenge, list_challenges, score_trail
from kinder_blockly.executor import (
    FrameLabel,
    PaintBucket,
    PenEvent,
    TrailSegment,
    execute_program,
    render_initial_frame,
    validate_program,
)

STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["PROPAGATE_EXCEPTIONS"] = True

_stop_event = threading.Event()

EXECUTION_TIMEOUT_S = 120.0


@app.route("/")
def index() -> Response:
    """Serve the main page."""
    return app.send_static_file("index.html")


def _encode_frame(frame: "np.ndarray[Any, Any]") -> str:
    """Encode a numpy RGB frame as a base64 JPEG string."""
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Environment endpoints ────────────────────────────────────────


@app.route("/reset", methods=["POST"])
def reset_env() -> Response:
    """Reset the environment and return the initial frame."""
    data = request.get_json() or {}
    seed = data.get("seed", 0)
    try:
        frame = render_initial_frame(seed=seed)
        return Response(
            json.dumps({"frame": _encode_frame(frame)}),
            status=200,
            mimetype="application/json",
        )
    except Exception as exc:  # pylint: disable=broad-except
        return Response(
            json.dumps({"error": str(exc)}),
            status=500,
            mimetype="application/json",
        )


@app.route("/run", methods=["POST"])
def run_program() -> Response:
    """Execute a Blockly program, streaming one NDJSON line per frame."""
    data = request.get_json()
    if data is None:
        return Response(
            json.dumps(
                {
                    "type": "done",
                    "error": "Invalid JSON",
                    "trail": [],
                    "pen_events": [],
                    "infinite_loop": False,
                }
            )
            + "\n",
            status=400,
            mimetype="application/x-ndjson",
        )

    program = data.get("program", {})
    seed = data.get("seed", 0)
    paint_buckets: list[PaintBucket] = data.get("paint_buckets", [])

    # Fast abstract pre-check — no physics needed.
    validation = validate_program(program)
    if validation.get("error") or validation.get("infinite_loop"):
        return Response(
            json.dumps(
                {
                    "type": "done",
                    "error": validation.get("error"),
                    "infinite_loop": validation.get("infinite_loop", False),
                    "error_block_id": validation.get("error_block_id"),
                    "trail": [],
                    "pen_events": [],
                }
            )
            + "\n",
            status=200,
            mimetype="application/x-ndjson",
        )

    _stop_event.clear()

    def generate() -> Iterator[str]:
        trail: list[TrailSegment] = []
        pen_events: list[PenEvent] = []
        frame_labels: list[FrameLabel] = []
        infinite_loop: list[bool] = [False]
        visited_buckets: set[str] = set()
        frame_index = 0
        error: str | None = None
        deadline = time.monotonic() + EXECUTION_TIMEOUT_S

        try:
            for frame in execute_program(
                program,
                seed=seed,
                trail_out=trail,
                pen_events_out=pen_events,
                frame_labels_out=frame_labels,
                infinite_loop_out=infinite_loop,
                stop_event=_stop_event,
                paint_buckets=paint_buckets,
                visited_buckets_out=visited_buckets,
            ):
                idx = frame_index
                label = frame_labels[idx] if idx < len(frame_labels) else None
                yield json.dumps(
                    {
                        "type": "frame",
                        "frame": _encode_frame(frame),
                        "index": frame_index,
                        "label": label,
                    }
                ) + "\n"
                frame_index += 1
                if time.monotonic() > deadline:
                    _stop_event.set()
                    error = (
                        "Phew, I'm exhausted!! That program took too long to run. "
                        "Try fewer moves, or check for a loop that's too long!"
                    )
                    break
        except Exception as exc:  # pylint: disable=broad-except
            error = str(exc)

        yield json.dumps(
            {
                "type": "done",
                "error": error,
                "infinite_loop": infinite_loop[0],
                "trail": trail,
                "pen_events": pen_events,
                "visited_buckets": list(visited_buckets),
            }
        ) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.route("/stop", methods=["POST"])
def stop_program() -> Response:
    """Signal the running execute_program to stop after the current block."""
    _stop_event.set()
    return Response(json.dumps({"ok": True}), status=200, mimetype="application/json")


# ── Challenge endpoints ──────────────────────────────────────────


@app.route("/challenges", methods=["GET"])
def challenges_list() -> Response:
    """Return the list of available challenges (without trail data)."""
    return Response(
        json.dumps({"challenges": list_challenges()}),
        status=200,
        mimetype="application/json",
    )


@app.route("/challenges/<challenge_id>", methods=["GET"])
def challenge_detail(challenge_id: str) -> Response:
    """Return a single challenge including its target trail."""
    challenge = get_challenge(challenge_id)
    if challenge is None:
        return Response(
            json.dumps({"error": f"Unknown challenge: {challenge_id}"}),
            status=404,
            mimetype="application/json",
        )
    return Response(
        json.dumps(challenge),
        status=200,
        mimetype="application/json",
    )


@app.route("/score", methods=["POST"])
def score() -> Response:
    """Score a student trail against a challenge target.

    Body: ``{"challenge_id": "...", "student_trail": [...]}``
    """
    data = request.get_json()
    if data is None:
        return Response(
            json.dumps({"error": "Invalid JSON"}),
            status=400,
            mimetype="application/json",
        )

    challenge_id = data.get("challenge_id")
    student_trail = data.get("student_trail", [])

    challenge = get_challenge(challenge_id)
    if challenge is None:
        return Response(
            json.dumps({"error": f"Unknown challenge: {challenge_id}"}),
            status=404,
            mimetype="application/json",
        )

    result = score_trail(student_trail, challenge["target_trail"])
    return Response(
        json.dumps(result),
        status=200,
        mimetype="application/json",
    )


def main() -> None:
    """Run the development server."""
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()
