"""Shared rendering/GIF helpers used by the per-environment run scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def render_frame(sim: Any) -> NDArray[np.uint8]:
    """Render one RGB frame of the environment."""
    frame = sim.render()
    if frame is None:
        raise RuntimeError("Environment returned no frame in rgb_array mode.")
    return np.asarray(frame)


def save_gif(
    path: str | Path, frames: list[NDArray[np.uint8]], duration: int = 100
) -> None:
    """Save rendered rgb_array frames as a GIF."""
    from PIL import Image as PILImage  # pylint: disable=import-outside-toplevel

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [PILImage.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved {len(frames)} frames to: {path}")
