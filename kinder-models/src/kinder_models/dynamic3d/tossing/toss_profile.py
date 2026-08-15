"""How hard a toss is thrown.

The limits are hand-tuned for throwing rather than derived from the arm's own, and are
demonstrated on the real TidyBot (`yixuanhuang98/tidybot_real`, `robot/kinova.py`).
"""

import numpy as np

# Deliberately over-driving _ARM_MAX_VELOCITY: a toss throws hard on purpose.
TOSS_MAX_VELOCITY = np.deg2rad(140.0)
TOSS_MAX_ACCELERATION = np.deg2rad(300.0)
TOSS_MAX_DECELERATION = np.deg2rad(200.0)


def toss_profile_limits(
    release_speed: float = TOSS_MAX_VELOCITY,
) -> tuple[float, float, float]:
    """The (max_vel, max_accel, max_decel) triple a toss at release_speed is timed by.

    One factor on all three, so this is an effort and not a speed cap: raising max_vel
    alone turns the profile triangular and moves the release into the acceleration
    phase. Clamped at 1, the real arm's own ceiling.
    """
    effort = min(max(release_speed / TOSS_MAX_VELOCITY, 0.0), 1.0)
    return (
        TOSS_MAX_VELOCITY * effort,
        TOSS_MAX_ACCELERATION * effort,
        TOSS_MAX_DECELERATION * effort,
    )
