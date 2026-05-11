"""Flask server for the Blockly visual programming interface."""

import base64
import io
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import sentry_sdk
from flask import Flask, Response, g, request, stream_with_context
from PIL import Image
from sentry_sdk.integrations.flask import FlaskIntegration

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

# Initialise Sentry before creating the Flask app so the integration can hook
# request handlers. No-ops when SENTRY_DSN is unset (e.g. local dev), so this
# never gates the server on having Sentry configured.
_SENTRY_DSN = os.environ.get("SENTRY_DSN")
if _SENTRY_DSN:
    sentry_sdk.init(
        dsn=_SENTRY_DSN,
        integrations=[FlaskIntegration()],
        environment=os.environ.get("SENTRY_ENVIRONMENT", "production"),
        release=os.environ.get("FLY_IMAGE_REF") or os.environ.get("SENTRY_RELEASE"),
        # Capture all exceptions; this is a low-volume service.
        sample_rate=1.0,
        # No performance tracing — we have k6 + Fly metrics for that.
        traces_sample_rate=0.0,
        send_default_pii=False,
    )

STATIC_DIR = Path(__file__).parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["PROPAGATE_EXCEPTIONS"] = True

EXECUTION_TIMEOUT_S = 120.0

# Per-session stop-event registry. Keys are session IDs read from the
# ``kb_session`` cookie; values are the Event a streaming /run watches. /stop
# sets the caller's Event only, so one student pressing Stop cannot cancel
# another student's run. The lock guards the dict itself.
_SESSION_COOKIE = "kb_session"
_SESSION_COOKIE_TTL_S = 86400
_session_stops: dict[str, threading.Event] = {}
_session_stops_lock = threading.Lock()


def _acquire_session_stop(session_id: str) -> threading.Event:
    """Return a fresh (cleared) stop Event registered under *session_id*."""
    event = threading.Event()
    with _session_stops_lock:
        _session_stops[session_id] = event
    return event


def _release_session_stop(session_id: str) -> None:
    """Remove a session's stop Event from the registry."""
    with _session_stops_lock:
        _session_stops.pop(session_id, None)


def _peek_session_stop(session_id: str) -> threading.Event | None:
    """Return the current stop Event for *session_id*, or None if absent."""
    with _session_stops_lock:
        return _session_stops.get(session_id)


@app.before_request
def _attach_session_id() -> None:
    """Read the session cookie, minting a new id if the client doesn't have one."""
    g.session_id = request.cookies.get(_SESSION_COOKIE) or str(uuid.uuid4())
    g.session_cookie_was_set = _SESSION_COOKIE in request.cookies


@app.after_request
def _ensure_session_cookie(response: Response) -> Response:
    """Issue the session cookie on the first response a client receives.

    Setting it server-side means existing frontends work without modification — their
    next request automatically carries the cookie.
    """
    if not getattr(g, "session_cookie_was_set", True):
        response.set_cookie(
            _SESSION_COOKIE,
            g.session_id,
            max_age=_SESSION_COOKIE_TTL_S,
            samesite="Lax",
            secure=request.is_secure,
            httponly=True,
        )
    return response


@app.route("/")
def index() -> Response:
    """Serve the main page."""
    return app.send_static_file("index.html")


@app.route("/healthz")
def healthz() -> Response:
    """Liveness probe.

    Returns 200 once this worker has a warm pybullet env.
    """
    # Avoid importing executor internals at module load to keep this cheap;
    # introspect the worker state lazily here.
    from kinder_blockly.executor import _W  # pylint: disable=import-outside-toplevel

    status = "ok" if _W.env is not None else "warming"
    code = 200 if _W.env is not None else 503
    return Response(
        json.dumps({"status": status}), status=code, mimetype="application/json"
    )


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
    paint_buckets = data.get("paint_buckets", [])
    try:
        frame = render_initial_frame(seed=seed, paint_buckets=paint_buckets)
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

    session_id = g.session_id
    stop_event = _acquire_session_stop(session_id)

    def generate() -> Iterator[str]:
        trail: list[TrailSegment] = []
        pen_events: list[PenEvent] = []
        frame_labels: list[FrameLabel] = []
        infinite_loop: list[bool] = [False]
        visited_buckets: set[str] = set()
        spawned_buckets: list[PaintBucket] = []
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
                stop_event=stop_event,
                paint_buckets=paint_buckets,
                visited_buckets_out=visited_buckets,
                spawned_buckets_out=spawned_buckets,
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
                    stop_event.set()
                    error = (
                        "Phew, I'm exhausted!! That program took too long to run. "
                        "Try fewer moves, or check for a loop that's too long!"
                    )
                    break
        except Exception as exc:  # pylint: disable=broad-except
            error = str(exc)
        finally:
            _release_session_stop(session_id)

        yield json.dumps(
            {
                "type": "done",
                "error": error,
                "infinite_loop": infinite_loop[0],
                "trail": trail,
                "pen_events": pen_events,
                "visited_buckets": list(visited_buckets),
                "spawned_buckets": spawned_buckets,
            }
        ) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.route("/stop", methods=["POST"])
def stop_program() -> Response:
    """Signal the caller's running execute_program to stop after the current block.

    Only this session's stop event is set. Other students' runs are unaffected.
    """
    event = _peek_session_stop(g.session_id)
    if event is not None:
        event.set()
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
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
