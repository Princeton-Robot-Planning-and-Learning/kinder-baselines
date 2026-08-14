"""The numeric cores of the dynamic3d state abstractions, free of any simulator import.

Every dynamic3d state abstractor answers most of its predicates with a few lines of
arithmetic over pose features, and those lines are the part worth sharing: shelf's
abstractor and tossing's carry the same gripper-open test, the same on-ground test and
the same grasped-and-lifted test, written out twice.

Two properties, both deliberate, and both the reason this is a module rather than more
private methods on the abstractors:

- **It takes plain floats, not an `ObjectCentricState`.** An abstractor is the only way
  to ask upstream whether a predicate holds, it answers for every predicate at once, and
  it needs a state type a consumer with its own state representation cannot construct.
  Arithmetic over floats has none of those constraints.
- **It imports numpy and nothing else**, so importing it pulls in neither `kinder.envs`,
  PyBullet nor MuJoCo. `utils.py` cannot offer that -- it exports `PyBulletSim` -- which
  is why the tolerances moved here and are re-exported from there rather than the other
  way around. `test_predicate_checks.py` asserts the property in a subprocess, so it
  fails loudly if a future import erodes it.

What is *not* here is anything needing the simulator: the forward kinematics behind
`Holding` and the goal-region lookup behind `MovableInGoalRegion` both do, so the
abstractor keeps them and calls in here for the rest.
"""

from typing import Sequence

import numpy as np

# State-abstraction tolerances, shared by every dynamic3d state_abstractor. They live
# here rather than in utils.py so that reading a threshold does not cost a PyBullet
# import; utils.py re-exports them, so its own callers are unaffected.
GRIPPER_OPEN_COMMAND_TOLERANCE = 1e-3
ON_GROUND_TOLERANCE = 0.05
GRIPPER_GRASPING_THRESHOLD = 0.1
MINIMUM_HOLDING_HEIGHT = 0.1
END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE = 0.05


def check_gripper_open(pos_gripper: float) -> bool:
    """Whether the gripper is commanded open, which implies an empty hand.

    Reads the command, not finger pose, so this and Holding are not complementary.
    """
    return bool(np.isclose(pos_gripper, 0.0, atol=GRIPPER_OPEN_COMMAND_TOLERANCE))


def check_on_ground(z: float, bounding_box_height: float, qx: float, qy: float) -> bool:
    """Whether a movable rests flat on the ground.

    Flat because the bounding box is pose-independent, so the bottom-face arithmetic
    only holds while axis-aligned. A toss cannot predict this.
    """
    return bool(
        np.isclose(z - bounding_box_height / 2, 0.0, atol=ON_GROUND_TOLERANCE)
        and np.isclose(qx, 0.0, atol=ON_GROUND_TOLERANCE)
        and np.isclose(qy, 0.0, atol=ON_GROUND_TOLERANCE)
    )


def check_grasped_and_lifted(pos_gripper: float, z: float) -> bool:
    """Whether the gripper is closed and the object is off the ground.

    The simulator-free part of Holding. On its own it is weaker than Holding: it admits
    a closed gripper and an airborne object that are not attached to each other -- a
    thrown object mid-flight, say. Pair it with check_end_effector_at_object, which
    needs forward kinematics, for the full test.
    """
    return bool(pos_gripper > GRIPPER_GRASPING_THRESHOLD and z > MINIMUM_HOLDING_HEIGHT)


def check_end_effector_at_object(
    end_effector_position: Sequence[float], object_position: Sequence[float]
) -> bool:
    """Whether the end effector is at an object, per axis rather than by distance."""
    return bool(
        all(
            abs(float(ee) - float(obj)) < END_EFFECTOR_TO_OBJECT_HOLDING_TOLERANCE
            for ee, obj in zip(end_effector_position, object_position)
        )
    )


def check_is_down_x(x: float, other_x: float) -> bool:
    """Whether a movable is at lower x than another, read live rather than fixed."""
    return bool(x < other_x)


def check_reach_interval_hits_box(
    base_x: float,
    reach_min: float,
    reach_max: float,
    box_x_min: float,
    box_x_max: float,
) -> bool:
    """Whether a throw from base_x can land inside [box_x_min, box_x_max].

    A throw from a fixed windup displaces the object by a distance set by the toss
    parameters and not by the scene, so the set of x it can reach from base_x is the
    interval [base_x + reach_min, base_x + reach_max]. The pose is throwable-from iff
    that interval meets the box -- iff *some* toss parameterisation scores from here.
    A controller with a single fixed toss passes reach_min == reach_max, which
    degenerates to a point and is the same test.

    **Deriving the acceptance band this way, rather than accepting the range the base
    sampler draws from, is the whole point of the function.** The two are easy to
    conflate and the failure is silent: if a "robot is at a throw pose" predicate
    accepts exactly the standoffs the sampler can produce, then the move-to-throw-pose
    operator's only add effect holds after every attempt by construction. A learner
    that trains a per-skill success classifier on that label sees a single class and
    never improves on its uniform prior, and nothing about the run looks wrong.
    """
    return bool(base_x + reach_min <= box_x_max and box_x_min <= base_x + reach_max)
