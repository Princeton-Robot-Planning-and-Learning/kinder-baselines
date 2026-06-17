"""Inline visualization for the Colab version of the bilevel-planning lab.

You do **not** edit this file. It replaces the local ``bilevel_planning.visualizer``
web app with functions that draw Obstruction2D states directly in the notebook
using kinder's pure ``render_2dstate``:

* ``render_state`` -- one state to an RGB image,
* ``show_storyboard`` -- a few key frames of a trajectory as inline images, and
* ``animate_states`` -- the full trajectory as an inline animation.

``run_sesame`` returns a ``plan`` whose ``states`` are the full per-step
trajectory the planner found, so ``animate_states(plan.states, ...)`` shows the
robot actually moving -- including routing a carried block around obstacles --
with no re-rollout needed. The bounds and DPI mirror ``lab/renderer.py``; the
static-object merge at render time mirrors ``part1_stacking/make_video.py``.

(The notebook's setup cell handles the clone + 2D-only install before importing
this module; see ``build_notebooks.py``.)
"""

from __future__ import annotations

from typing import Any, Sequence

# The lab packages and the plotting stack are imported inside the functions that
# use them, not at module top, so importing this module right after install in a
# fresh runtime stays cheap and order-independent.
# pylint: disable=import-outside-toplevel


# World bounds for kinder/Obstruction2D-*-v0 (same values as lab/renderer.py).
WORLD_MIN_X = 0.0
WORLD_MAX_X = 1.618033988749895
WORLD_MIN_Y = 0.0
WORLD_MAX_Y = 1.0
RENDER_DPI = 200


def render_state(state: Any, constant_state: Any = None, dpi: int = RENDER_DPI):
    """Return an HxWx3 uint8 RGB image of an Obstruction2D state.

    ``constant_state`` (the env's static objects -- table, walls) is merged in
    only for rendering, mirroring how the env and the web visualizer build the
    full scene from a stored state.
    """
    from kinder.envs.utils import render_2dstate

    if constant_state is not None:
        state = state.copy()
        state.data.update(constant_state.data)
    return render_2dstate(
        state,
        None,
        world_min_x=WORLD_MIN_X,
        world_max_x=WORLD_MAX_X,
        world_min_y=WORLD_MIN_Y,
        world_max_y=WORLD_MAX_Y,
        render_dpi=dpi,
    )


def show_storyboard(
    states: Sequence[Any],
    constant_state: Any = None,
    max_panels: int = 6,
    dpi: int = 120,
):
    """Draw a handful of evenly-spaced frames from a trajectory as inline images.

    ``states`` is typically ``plan.states`` (the full per-step trajectory). When
    there are more than ``max_panels`` states, this subsamples evenly -- always
    keeping the first and last -- so the storyboard stays readable. Use
    ``animate_states`` to watch the full motion.
    """
    import matplotlib.pyplot as plt

    states = list(states)
    n = len(states)
    if n <= max_panels:
        idxs = list(range(n))
    else:
        idxs = sorted(
            {round(i * (n - 1) / (max_panels - 1)) for i in range(max_panels)}
        )
    fig, axes = plt.subplots(1, len(idxs), figsize=(4 * len(idxs), 4))
    if len(idxs) == 1:
        axes = [axes]
    for ax, i in zip(axes, idxs):
        ax.imshow(render_state(states[i], constant_state, dpi=dpi))
        ax.set_title("start" if i == 0 else ("end" if i == n - 1 else f"step {i}"))
        ax.axis("off")
    fig.tight_layout()
    return fig


def animate_states(
    states: Sequence[Any],
    constant_state: Any = None,
    fps: int = 12,
    dpi: int = RENDER_DPI,
):
    """Animate a trajectory of states (e.g. ``plan.states``) as an inline video.

    Renders each state to a frame and returns an object Jupyter/Colab displays as an
    HTML5 animation, so the robot is seen moving step by step.
    """
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    from matplotlib import animation

    frames = [render_state(s, constant_state, dpi) for s in states]
    frames.extend([frames[-1]] * fps)  # hold the final frame for ~1s

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    im = ax.imshow(frames[0])
    fig.tight_layout()

    def _update(i):
        im.set_array(frames[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=1000 / fps, blit=True
    )
    plt.close(fig)  # don't also show the static first frame
    return HTML(anim.to_jshtml())  # type: ignore[no-untyped-call]
