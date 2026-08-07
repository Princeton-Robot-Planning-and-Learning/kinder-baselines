"""Utilities for locating the local pddlstream checkout.

pddlstream is not pip-installable, so its checkout is put on `sys.path` here.
"""

import os
import sys
from pathlib import Path

_PDDLSTREAM_PATH_ENV_VAR = "PDDLSTREAM_PATH"
_DEFAULT_PDDLSTREAM_PATH = Path("~/pddlstream").expanduser()


def ensure_pddlstream_importable() -> None:
    """Add the pddlstream checkout to sys.path if it is not already importable."""
    try:
        import pddlstream  # pylint: disable=unused-import,import-outside-toplevel

        return
    except ImportError:
        pass

    pddlstream_path = Path(
        os.environ.get(_PDDLSTREAM_PATH_ENV_VAR, _DEFAULT_PDDLSTREAM_PATH)
    ).expanduser()
    if not (pddlstream_path / "pddlstream").is_dir():
        raise ModuleNotFoundError(
            "Could not find a pddlstream checkout. Set the "
            f"{_PDDLSTREAM_PATH_ENV_VAR} environment variable to the root of a "
            "pddlstream checkout (the directory containing the `pddlstream` and "
            f"`downward` subdirectories). Tried: {pddlstream_path}"
        )
    sys.path.insert(0, str(pddlstream_path))
