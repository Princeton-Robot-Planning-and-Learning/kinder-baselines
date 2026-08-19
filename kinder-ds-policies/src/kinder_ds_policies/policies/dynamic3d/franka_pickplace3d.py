"""Domain-specific policy for the FrankaPickPlace3D environment.

This policy runs a scripted sequence of the pick and place parameterized skills from
kinder-models: for each cube, pick it and place it in the goal region on the desk. The
logic mirrors the test in kinder-models/tests/dynamic3d/franka_pickplace.
"""

from typing import Any

import numpy as np
from kinder.envs.dynamic3d.object_types import (
    MujocoFR3RobotObjectType,
    MujocoMovableObjectType,
)
from kinder.envs.dynamic3d.robots.fr3_robot_env import FR3RobotActionSpace
from kinder_models.dynamic3d.franka_pickplace.parameterized_skills import (
    create_lifted_controllers,
)
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_ds_policies.policies.base import PolicyFailure, StatefulPolicy

__all__ = ["create_domain_specific_policy"]


class FrankaPickPlace3DScriptedPolicy(StatefulPolicy):
    """A stateful scripted policy for FrankaPickPlace3D.

    This policy maintains state between calls to track which controller is currently
    executing and the overall progress through the task.
    """

    def __init__(
        self,
        observation_space: ObjectCentricBoxSpace,
        action_space: FR3RobotActionSpace,
        seed: int = 123,
    ) -> None:
        self._observation_space = observation_space
        self._action_space = action_space
        self._rng = np.random.default_rng(seed)

        # Create controllers; the shared FR3 IK solver is built internally.
        self._controllers = create_lifted_controllers(self._action_space)

        # State tracking.
        self._current_controller: Any = None
        self._skill_sequence: list[tuple[str, tuple[Object, ...]]] = []
        self._skill_index = 0
        self._initialized = False
        self._finished = False

    def reset(self) -> None:
        """Reset the policy state for a new episode."""
        self._current_controller = None
        self._skill_sequence = []
        self._skill_index = 0
        self._initialized = False
        self._finished = False

    def is_finished(self) -> bool:
        """Whether the scripted skill sequence has fully executed.

        The environment's goal is a positional region check that fires while the cube is
        still gripped during the place descent, before release. Demo collection uses
        this to keep stepping until the place skill has also opened the gripper and
        retracted, so the demo captures the full placement.
        """
        return self._finished

    def _build_skill_sequence(self, state: ObjectCentricState) -> None:
        """Build the skill sequence based on the initial state."""
        robots = state.get_objects(MujocoFR3RobotObjectType)
        assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
        robot = list(robots)[0]
        desk = state.get_object_from_name("desk_1")

        # Pick and place each cube in the goal region on the desk.
        self._skill_sequence = []
        cubes = sorted(
            (o for o in state.get_objects(MujocoMovableObjectType)),
            key=lambda o: o.name,
        )
        for cube in cubes:
            self._skill_sequence.append(("pick", (robot, cube)))
            self._skill_sequence.append(("place", (robot, cube, desk)))

    def _start_next_skill(self, state: ObjectCentricState) -> bool:
        """Start the next skill in the sequence.

        Returns True if a skill was started, False if the sequence is complete.
        """
        if self._skill_index >= len(self._skill_sequence):
            return False

        skill_name, objects = self._skill_sequence[self._skill_index]
        lifted_controller = self._controllers[skill_name]
        self._current_controller = lifted_controller.ground(objects)
        try:
            params = self._current_controller.sample_parameters(state, self._rng)
            self._current_controller.reset(state, params)
        except Exception as e:
            raise PolicyFailure(f"Controller reset failed: {e}") from e
        return True

    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute action using the scripted controller sequence."""
        state = self._observation_space.devectorize(observation)

        # Initialize on first call.
        if not self._initialized:
            self._build_skill_sequence(state)
            if not self._start_next_skill(state):
                raise PolicyFailure("No skills to execute.")
            self._initialized = True

        # Observe the result of the last action.
        assert self._current_controller is not None
        self._current_controller.observe(state)

        # Check if the current controller is done; move to the next skill.
        while self._current_controller.terminated():
            self._skill_index += 1
            if not self._start_next_skill(state):
                # All skills complete - return zero action (hold still).
                self._finished = True
                shape = self._action_space.shape
                assert shape is not None
                return np.zeros(shape, dtype=np.float32)

        try:
            action = self._current_controller.step()
        except Exception as e:
            raise PolicyFailure(f"Controller step failed: {e}") from e
        return np.asarray(action, dtype=np.float32)


def create_domain_specific_policy(
    observation_space: ObjectCentricBoxSpace,
    action_space: FR3RobotActionSpace,
    seed: int = 123,
) -> FrankaPickPlace3DScriptedPolicy:
    """Create a domain-specific policy for FrankaPickPlace3D.

    Args:
        observation_space: The observation space used to devectorize
            observations.
        action_space: The action space (required for controller creation).
        seed: Random seed for the skill parameter samplers.

    Returns:
        A stateful policy that maps observations to actions.
    """
    return FrankaPickPlace3DScriptedPolicy(
        observation_space=observation_space,
        action_space=action_space,
        seed=seed,
    )
