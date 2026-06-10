"""Renderer for the lab's Obstruction2D visualizer bundles.

Defines ``render_state(state)``, which the bilevel_planning visualizer execs and
calls on each clicked node. It draws an Obstruction2D state (robot, blocks,
target surface) as a top-down 2D scene using kinder's pure ``render_2dstate``
function -- no live environment needed, since the pickled states are
self-contained. Works for both the walkthrough and pyramid bundles.

Pass this file to the visualizer with ``--renderer``.
"""

from kinder.envs.utils import render_2dstate

# World bounds for kinder/Obstruction2D-*-v0 (from the env config).
WORLD_MIN_X = 0.0
WORLD_MAX_X = 1.618033988749895
WORLD_MIN_Y = 0.0
WORLD_MAX_Y = 1.0
RENDER_DPI = 300


def render_state(state):
    """Return an HxWx3 uint8 RGB image of an Obstruction2D state."""
    return render_2dstate(
        state,
        world_min_x=WORLD_MIN_X,
        world_max_x=WORLD_MAX_X,
        world_min_y=WORLD_MIN_Y,
        world_max_y=WORLD_MAX_Y,
        render_dpi=RENDER_DPI,
    )
