"""Configure pytest."""

import os

# Must be set before anything imports mujoco or OpenGL: under osmesa the dynamic3d
# environments fail to register, and the failure is silent until make() raises
# NameNotFound. Hence the imports below sit after module-level code.
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"

# pylint: disable=wrong-import-position
from _pytest.config import Config
from _pytest.config.argparsing import Parser

# Global variable that gets set with the command line --make-videos flag.
MAKE_VIDEOS = False

# Global variable that gets set with the command line --save-demos flag.
SAVE_DEMOS = False


def pytest_addoption(parser: Parser) -> None:
    """Register custom command-line options for pytest."""
    parser.addoption(
        "--make-videos",
        action="store_true",
        default=False,
        help="Enable video generation during tests",
    )
    parser.addoption(
        "--save-demos",
        action="store_true",
        default=False,
        help="Save demo pickle files for each unit test",
    )


def pytest_configure(config: Config) -> None:
    """Set global configuration values after command-line options are parsed."""
    global MAKE_VIDEOS, SAVE_DEMOS  # pylint:disable=global-statement
    MAKE_VIDEOS = config.getoption("--make-videos")
    SAVE_DEMOS = config.getoption("--save-demos")
