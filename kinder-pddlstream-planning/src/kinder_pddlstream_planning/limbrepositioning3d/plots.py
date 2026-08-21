"""Plots of one LimbRepositioning3D rollout, from the npz `run.py --trajectory-dir`
writes.

Run standalone to redraw without re-running the planner:

    python -m kinder_pddlstream_planning.limbrepositioning3d.plots outputs/traj/*.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

plt.switch_backend("Agg")

# Categorical slots in their fixed order, plus the chart's chrome and status inks.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"


def _style(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Recessive grid and axes, ink on the chart surface."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=PRIMARY_INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel("seconds", color=SECONDARY_INK, fontsize=9)
    ax.set_ylabel(ylabel, color=SECONDARY_INK, fontsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED_INK, labelsize=8)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(AXIS)


def _plot(
    ax: plt.Axes, times: NDArray, values: NDArray, names: list[str], legend: bool = True
) -> None:
    """One line per column of `values`, in the fixed categorical order."""
    for index, name in enumerate(names):
        ax.plot(
            times,
            values[:, index],
            color=SERIES[index % len(SERIES)],
            linewidth=1.6,
            label=name,
        )
    if legend and len(names) > 1:
        _legend(ax)


def _legend(ax: plt.Axes) -> None:
    """Outside the panel, so it never lands on a trace or a limit line."""
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        frameon=False,
        labelcolor=SECONDARY_INK,
    )


def _threshold(
    ax: plt.Axes,
    value: float,
    label: str,
    color: str = CRITICAL,
    mirror: bool = False,
) -> None:
    """A dashed reference line, mirrored below zero for a signed quantity.

    Skipped when it is so far out that drawing it would flatten the traces; the panel
    title carries the peak against it either way.
    """
    low, high = ax.get_ylim()
    if abs(value) > 3.0 * max(abs(low), abs(high)):
        return
    levels = [value, -value] if mirror and value > 0.0 else [value]
    for level in levels:
        ax.axhline(level, color=color, linestyle="--", linewidth=1.2)
    low, high = min([low] + levels), max([high] + levels)
    pad = 0.05 * (high - low)
    ax.set_ylim(low - pad, high + pad)
    ax.annotate(
        label,
        xy=(0.995, value),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=7,
        color=color,
    )


def _peak(title: str, values: NDArray, limit: float, unit: str) -> str:
    """The panel title, with its peak read against the limit it is checked on."""
    return f"{title} - peak {np.abs(values).max():.1f} of {limit:.0f} {unit}"


def _joint_labels(names: NDArray) -> list[str]:
    return [f"{index} {name}" for index, name in enumerate(names)]


def plot_rollout(npz_path: str | Path, png_path: str | Path | None = None) -> Path:
    """Draw one rollout's torques, wrenches, and margins, and save it beside the npz."""
    npz_path = Path(npz_path)
    png_path = Path(png_path) if png_path is not None else npz_path.with_suffix(".png")
    data = np.load(npz_path, allow_pickle=False)

    positions = data["limb_positions"]
    times = np.arange(len(positions)) * float(data["dt"])
    limb_names = _joint_labels(data["limb_joint_names"])
    robot_names = [f"joint {i + 1}" for i in range(data["commanded_torques"].shape[1])]
    margin = np.minimum(
        positions - data["limb_lower_limits"], data["limb_upper_limits"] - positions
    )
    wrench = data["grasp_wrenches"]

    fig, axes = plt.subplots(4, 2, figsize=(16, 17), facecolor=SURFACE)
    fig.suptitle(
        f"{data['variant']}: what the rollout put on the person and the robot",
        color=PRIMARY_INK,
        fontsize=13,
        x=0.02,
        ha="left",
    )

    ax = axes[0, 0]
    _plot(ax, times, data["human_total"], limb_names)
    _style(
        ax,
        _peak(
            "Torque on the person's joints, total",
            data["human_total"],
            float(data["human_torque_limit"]),
            "N*m",
        ),
        "N*m",
    )
    _threshold(ax, float(data["human_torque_limit"]), "limit", mirror=True)

    ax = axes[0, 1]
    _plot(ax, times, data["human_robot_induced"], limb_names)
    _style(
        ax,
        _peak(
            "The robot's share of it, over the person's static load",
            data["human_robot_induced"],
            float(data["robot_induced_torque_limit"]),
            "N*m",
        ),
        "N*m",
    )
    _threshold(ax, float(data["robot_induced_torque_limit"]), "limit", mirror=True)

    ax = axes[1, 0]
    _plot(ax, times, margin, limb_names)
    _style(
        ax,
        f"Room left before the nearest anatomical limit - "
        f"closest {margin.min():.2f} rad",
        "rad",
    )
    _threshold(ax, 0.0, "at the limit")

    ax = axes[1, 1]
    _plot(ax, times, data["limb_velocities"], limb_names)
    _style(ax, "How fast the limb is being moved", "rad/s")

    ax = axes[2, 0]
    _plot(ax, times, data["commanded_torques"], robot_names)
    utilization = np.abs(data["commanded_torques"]) / np.abs(data["robot_torque_upper"])
    _style(
        ax,
        f"Torque the robot is asked for, hold plus correction - peak "
        f"{100 * utilization.max():.0f}% of a joint's effort limit",
        "N*m",
    )
    for limit in np.unique(np.abs(data["robot_torque_upper"])):
        _threshold(ax, float(limit), "effort limit", mirror=True)

    ax = axes[2, 1]
    ax.plot(times, data["goal_error"], color=SERIES[0], linewidth=1.6)
    _style(ax, "Distance from the limb's goal configuration", "rad")
    _threshold(ax, float(data["goal_atol"]), "goal tolerance", color=GOOD)

    ax = axes[3, 0]
    _plot(ax, times, wrench[:, :3], ["force x", "force y", "force z"])
    ax.plot(
        times,
        np.linalg.norm(wrench[:, :3], axis=1),
        color=SERIES[3],
        linewidth=1.6,
        label="magnitude",
    )
    _style(ax, "Force the grasp transmits into the limb", "N")
    _legend(ax)

    ax = axes[3, 1]
    _plot(ax, times, wrench[:, 3:], ["moment x", "moment y", "moment z"])
    ax.plot(
        times,
        np.linalg.norm(wrench[:, 3:], axis=1),
        color=SERIES[3],
        linewidth=1.6,
        label="magnitude",
    )
    _style(ax, "Moment the grasp transmits into the limb", "N*m")
    _legend(ax)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(png_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Plots written to {png_path.resolve()}")
    return png_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_paths", nargs="+", type=str)
    args = parser.parse_args()
    for npz_path in args.npz_paths:
        plot_rollout(npz_path)


if __name__ == "__main__":
    main()
