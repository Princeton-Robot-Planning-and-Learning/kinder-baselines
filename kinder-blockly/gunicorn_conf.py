"""Gunicorn configuration for the Blockly server.

Workload notes:
- Each request to /run is CPU-bound (pybullet stepping + image encoding) and
  may stream for up to EXECUTION_TIMEOUT_S seconds. We use sync workers
  because async (gevent / asyncio) cannot release the GIL during pybullet C
  calls — it would not improve concurrency for this workload.
- preload_app must be False: pybullet's physics client is created per
  process and cannot be shared across forked children.
- Each worker pre-warms its own pybullet env at startup so the first
  student request does not pay the URDF-load latency.
"""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"

workers = int(os.environ.get("GUNICORN_WORKERS", "2"))
worker_class = "sync"
threads = 1

# Streaming /run is capped server-side at 120s; give gunicorn a margin so it
# does not kill a legitimate long-running request before the app does.
timeout = 180
graceful_timeout = 30
keepalive = 5

preload_app = False

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")


def post_worker_init(worker):  # type: ignore[no-untyped-def]
    """Pre-warm the pybullet env so /healthz starts returning 200 quickly.

    Without this the env is created lazily on the first /run or /reset, which means
    health checks fail for the first few seconds after a deploy.
    """
    # Imported lazily so this file can be loaded by tooling without dragging
    # pybullet into every process that just reads the config.
    # pylint: disable=import-outside-toplevel
    from kinder_blockly.executor import _ensure_worker_initialized

    _ensure_worker_initialized()
    worker.log.info("pybullet env pre-warmed for worker pid=%s", worker.pid)
