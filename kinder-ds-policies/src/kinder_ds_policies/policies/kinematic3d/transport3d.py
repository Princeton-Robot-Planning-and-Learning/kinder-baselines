"""Domain-specific policy for Transport3D environment.

This policy implements a scripted sequence of pick and place operations using the
parameterized skills from kinder-models. The logic mirrors the test in kinder-
models/tests/kinematic3d/transport3d.
"""

import os
from typing import Any

import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from kinder.envs.kinematic3d.transport3d import ObjectCentricTransport3DEnv
from kinder.envs.kinematic3d.utils import (
    Kinematic3DObjectCentricState,
    Kinematic3DRobotActionSpace,
)
from kinder_models.kinematic3d.transport3d.parameterized_skills import (
    create_lifted_controllers,
)
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_ds_policies.policies.base import PolicyFailure, StatefulPolicy

__all__ = ["create_domain_specific_policy"]

_SkillSpec = tuple[str, tuple[Object, ...], NDArray[np.float32]]
_SkillPair = list[_SkillSpec]


def _place_approach_candidates() -> list[tuple[float, float]]:
    """(stand-off distance, approach rotation) candidates for the place skill.

    Ordered by |rot| then by distance so that index 0 is exactly the skill's original
    hard-coded (0.65, 0.0) approach: if that works nothing changes. Rotating the park
    position around the target surface is what makes a blocked approach solvable -- the
    robot always faces the target, so the place itself stays valid.
    """
    rots = [0.0]
    for k in range(1, 7):
        rots.extend([k * np.pi / 12, -k * np.pi / 12])
    return [(d, r) for r in rots for d in (0.65, 0.55, 0.72)]


class Transport3DScriptedPolicy(StatefulPolicy):
    """A stateful scripted policy for Transport3D.

    This policy maintains state between calls to track which controller is currently
    executing and the overall progress through the task.
    """

    #: (distance, rot) base-approach candidates for `place`, |rot|-ascending.
    PLACE_APPROACHES = _place_approach_candidates()

    def __init__(
        self,
        observation_space: ObjectCentricBoxSpace,
        action_space: Kinematic3DRobotActionSpace,
        num_cubes: int,
        seed: int = 123,
        birrt_extend_num_interp: int = 25,
        smooth_mp_max_time: float = 120.0,
        smooth_mp_max_candidate_plans: int = 20,
    ) -> None:
        self._observation_space = observation_space
        self._action_space = action_space
        self._num_cubes = num_cubes
        self._rng = np.random.default_rng(seed)

        # Create simulator for controllers.
        self._sim = ObjectCentricTransport3DEnv(
            num_cubes=num_cubes, use_gui=False, allow_state_access=True
        )

        # Create controllers with settings for motion smoothness.
        self._controllers = create_lifted_controllers(
            self._action_space,
            self._sim,
            birrt_extend_num_interp=birrt_extend_num_interp,
            smooth_mp_max_time=smooth_mp_max_time,
            smooth_mp_max_candidate_plans=smooth_mp_max_candidate_plans,
        )

        # State tracking (initialized via reset()).
        self._current_controller: Any = None
        self._pending_cube_pairs: list[_SkillPair] = []
        self._box_pair: _SkillPair = []
        self._current_pair: _SkillPair | None = None
        self._current_is_box = False
        self._box_started = False
        self._pair_skill_idx = 0
        self._consecutive_defers = 0
        self._initialized = False
        self.reset()

    def reset(self) -> None:
        """Reset the policy state for a new episode."""
        self._current_controller = None
        # A "pair" is [ (pick...), (place...) ] executed back-to-back while
        # holding the object. Cube pairs are reorderable; the box pair is last.
        self._pending_cube_pairs = []
        self._box_pair = []
        self._current_pair = None
        self._current_is_box = False
        self._box_started = False
        self._pair_skill_idx = 0
        self._consecutive_defers = 0
        self._initialized = False

    def _build_pairs(self, state: ObjectCentricState) -> None:
        """Build (pick, place) pairs: one per cube (reorderable), box move last."""
        robot = state.get_object_from_name("robot")
        box0 = state.get_object_from_name("box0")
        table = state.get_object_from_name("table")
        place_offsets = [
            np.array([0.0, -0.06], dtype=np.float32),
            np.array([0.0, 0.06], dtype=np.float32),
        ]
        self._pending_cube_pairs = []
        for i in range(self._num_cubes):
            cube = state.get_object_from_name(f"cube{i}")
            self._pending_cube_pairs.append(
                [
                    ("pick", (robot, cube), np.array([0.5, 0.0], dtype=np.float32)),
                    (
                        "place",
                        (robot, cube, box0),
                        place_offsets[i % len(place_offsets)],
                    ),
                ]
            )
        self._box_pair = [
            ("pick", (robot, box0), np.array([0.5, 0.0], dtype=np.float32)),
            ("place", (robot, box0, table), np.array([0.0, 0.0], dtype=np.float32)),
        ]

    def _resample_params(self, name: str, base_params: NDArray) -> NDArray:
        """Perturb params to find a feasible approach after a planning failure.

        For a pick, the params are (base_distance, base_rot) around the target; rotating
        the park position tries a different, possibly-reachable side. For a place, the
        params are the in-box placement offset; the base approach itself is varied
        separately (see PLACE_APPROACHES).
        """
        if name == "pick":
            dist = float(self._rng.uniform(0.45, 0.62))
            rot = float(self._rng.uniform(-np.pi, np.pi))
            return np.array([dist, rot], dtype=np.float32)
        ox, oy = float(base_params[0]), float(base_params[1])
        return np.array(
            [
                ox + float(self._rng.uniform(-0.03, 0.03)),
                oy + float(self._rng.uniform(-0.03, 0.03)),
            ],
            dtype=np.float32,
        )

    def _ground_and_reset(self, name, objs, params, state, attempt: int = 0) -> bool:
        """Ground a controller and reset it; return False if reset sampling fails.

        For `place`, the base stand-off pose is fixed in the skill (distance
        0.65, rot 0.0), which is infeasible whenever that one spot is blocked or
        out of arm reach. Walk a deterministic list of (distance, rot) approach
        candidates ordered by |rot| so attempt 0 reproduces the original
        behavior exactly and later attempts try progressively different sides.
        """
        controller = self._controllers[name].ground(objs)
        # Use the env's real per-joint action limit when re-timing arm plans.
        # The default remap uses a weighted-L1 budget half the size, which
        # inflates every arm motion several-fold; measured, that alone pushed
        # the 90th-percentile successful episode past the 1200-step cap.
        setattr(controller, "_use_linf_arm_remap", True)
        # Let a place retract re-plan without the bodies it is resting against
        # (after release the gripper is inside the box and the planner rejects
        # its own start configuration).
        setattr(controller, "_retract_rescue", True)
        if name == "place":
            dist, rot = self.PLACE_APPROACHES[attempt % len(self.PLACE_APPROACHES)]
            # Consumed via getattr() by GroundPlaceController.step().
            setattr(controller, "_approach_distance", dist)
            setattr(controller, "_approach_rot", rot)
        try:
            controller.reset(state, params)
        except TrajectorySamplingFailure:
            self._current_controller = None
            return False
        self._current_controller = controller
        return True

    @staticmethod
    def _can_resample(name: str, controller: Any) -> bool:
        """Whether restarting this skill would preserve physical task progress."""
        if name == "pick":
            # Once the gripper has closed around the object, a fresh pick would
            # discard the held-object state and can no longer be replayed safely.
            return not bool(getattr(controller, "_closed_gripper", False))
        if name == "place":
            # Once release has completed, a fresh place has no grasp transform.
            return not bool(getattr(controller, "_opened_gripper", False))
        return False

    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute an action, adapting skill order and params to the scene.

        Greedily executes whichever object is currently reachable; if a pick cannot be
        planned even after resampling, the cube is deferred and retried after other
        objects move (which can clear the blocked approach). The box move is always
        performed last.
        """
        state = self._observation_space.devectorize(observation)
        assert isinstance(state, Kinematic3DObjectCentricState)
        self._sim.set_state(state)

        if not self._initialized:
            self._build_pairs(state)
            self._initialized = True

        # T3D_BASELINE=1 disables the adaptive behavior (for A/B comparison):
        # one planning attempt, no param resampling, no deferral -> original.
        adaptive = os.environ.get("T3D_BASELINE") != "1"
        max_resample = 4 if adaptive else 1
        zeros = np.zeros(self._action_space.shape, dtype=np.float32)

        while True:
            # Select a pair to work on.
            if self._current_pair is None:
                if self._pending_cube_pairs:
                    self._current_pair = self._pending_cube_pairs.pop(0)
                    self._current_is_box = False
                elif not self._box_started:
                    self._current_pair = self._box_pair
                    self._current_is_box = True
                    self._box_started = True
                else:
                    return zeros  # task complete
                self._pair_skill_idx = 0
                self._current_controller = None

            name, objs, params = self._current_pair[self._pair_skill_idx]

            # Start / advance the controller for the current skill in the pair.
            if self._current_controller is None:
                self._ground_and_reset(name, objs, params, state)
            if self._current_controller is not None:
                self._current_controller.observe(state)
                while self._current_controller.terminated():
                    self._pair_skill_idx += 1
                    if self._pair_skill_idx >= len(self._current_pair):
                        self._current_pair = None
                        self._current_controller = None
                        break
                    name, objs, params = self._current_pair[self._pair_skill_idx]
                    if not self._ground_and_reset(name, objs, params, state):
                        break
                    self._current_controller.observe(state)
                if self._current_pair is None:
                    self._consecutive_defers = 0
                    continue  # pair finished; get the next one

            # Try to produce an action, resampling on planning failure. `place`
            # gets a bigger budget because its retries walk a deterministic
            # approach-angle list (a blocked base goal makes BiRRT fail
            # immediately, so extra attempts are cheap).
            name, objs, base_params = self._current_pair[self._pair_skill_idx]
            attempts = max_resample
            if adaptive and name == "place":
                attempts = len(self.PLACE_APPROACHES)
            for _attempt in range(attempts):
                if self._current_controller is not None:
                    try:
                        action = self._current_controller.step()
                        self._consecutive_defers = 0
                        return action
                    except TrajectorySamplingFailure as err:
                        if not self._can_resample(name, self._current_controller):
                            raise PolicyFailure(
                                f"{name.capitalize()} recovery failed after an "
                                "irreversible grasp or release."
                            ) from err
                self._ground_and_reset(
                    name,
                    objs,
                    self._resample_params(name, base_params),
                    state,
                    attempt=_attempt + 1,
                )

            # Could not plan this skill even after resampling.
            is_pick = self._pair_skill_idx == 0
            if (
                adaptive
                and is_pick
                and not self._current_is_box
                and self._consecutive_defers <= len(self._pending_cube_pairs) + 1
            ):
                # Defer this cube: retry it after other objects move (their
                # removal can open the blocked base approach).
                self._pending_cube_pairs.append(self._current_pair)
                self._current_pair = None
                self._current_controller = None
                self._consecutive_defers += 1
                continue

            raise PolicyFailure("Sampling failed after resampling and deferral.")


def create_domain_specific_policy(
    observation_space: ObjectCentricBoxSpace,
    action_space: Kinematic3DRobotActionSpace,
    num_cubes: int = 2,
    seed: int = 123,
    birrt_extend_num_interp: int = 25,
    smooth_mp_max_time: float = 10.0,
    smooth_mp_max_candidate_plans: int = 1,
) -> StatefulPolicy:
    """Create a domain-specific policy for Transport3D.

    Args:
        observation_space: The observation space used to devectorize observations.
        action_space: The action space (required for controller creation).
        num_cubes: Number of cubes in the environment.
        seed: Random seed for controller parameter sampling.
        birrt_extend_num_interp: Number of interpolation steps for BiRRT extend.
            Defaults to 25.
        smooth_mp_max_time: Maximum time for motion planning smoothing.
            Defaults to 10.0.
        smooth_mp_max_candidate_plans: Maximum candidate plans for smoothing.
            Defaults to 1.

    Returns:
        A policy that maps observations to actions.
    """
    policy = Transport3DScriptedPolicy(
        observation_space=observation_space,
        action_space=action_space,
        num_cubes=num_cubes,
        seed=seed,
        birrt_extend_num_interp=birrt_extend_num_interp,
        smooth_mp_max_time=smooth_mp_max_time,
        smooth_mp_max_candidate_plans=smooth_mp_max_candidate_plans,
    )

    return policy
