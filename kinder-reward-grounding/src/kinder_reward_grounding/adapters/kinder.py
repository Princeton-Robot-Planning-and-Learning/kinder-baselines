"""KinDER-specific object-centric adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kinder_reward_grounding.utils.geometry import (
    hook_heuristic_keypoints,
    point_to_oriented_rectangle_distance,
)


@dataclass(frozen=True)
class KinDERSceneState:
    """Object-centric KinDER state split into dynamic and constant objects."""

    dynamic_state: Any
    constant_state: Any


class KinDERObjectCentricAdapter:
    """Adapter for KinDER constant-object environments.

    This adapter wraps the KinDER object-centric vectorization bridge:
    flat planner state -> ObjectCentricState.
    """

    def decode(self, env: Any, flat_state: np.ndarray) -> KinDERSceneState:
        """Decode a flat observation into an object-centric state."""
        return KinDERSceneState(
            dynamic_state=env.unwrapped.observation_space.devectorize(flat_state),
            constant_state=self.get_constant_state(env),
        )

    def get(self, state: Any, obj: Any, feature: str) -> Any:
        """Get a feature value for an object."""
        if isinstance(state, KinDERSceneState):
            return self._state_for_object(state, obj).get(obj, feature)
        return state.get(obj, feature)

    def get_object_by_name(self, state: Any, name: str) -> Any:
        """Get an object from state by object name."""
        states = (
            (state.dynamic_state, state.constant_state)
            if isinstance(state, KinDERSceneState)
            else (state,)
        )
        for object_state in states:
            for obj in object_state:
                if obj.name == name:
                    return obj
        raise KeyError(f"Object not found in state: {name}")

    def get_dynamic_object_by_name(self, state: Any, name: str) -> Any:
        """Get a dynamic object from state by object name."""
        dynamic_state = (
            state.dynamic_state if isinstance(state, KinDERSceneState) else state
        )
        for obj in dynamic_state:
            if obj.name == name:
                return obj
        raise KeyError(f"Dynamic object not found in state: {name}")

    def get_constant_state(self, env: Any) -> Any:
        """Get the KinDER initial constant state."""
        # KinDER currently exposes constant state only through this internal
        # object-centric bridge.
        return env.unwrapped._object_centric_env.initial_constant_state  # pylint: disable=protected-access

    def get_constant_by_name(self, env: Any, name: str) -> tuple[Any, Any]:
        """Get a constant object and its state by object name."""
        constant_state = self.get_constant_state(env)
        for obj in constant_state:
            if obj.name == name:
                return constant_state, obj
        raise KeyError(f"Constant object not found: {name}")

    def position(self, state: Any, obj: Any) -> np.ndarray:
        """Get a 2D object center position."""
        return np.array(
            [float(self.get(state, obj, "x")), float(self.get(state, obj, "y"))],
            dtype=float,
        )

    def distance(self, state: Any, obj_a: Any, obj_b: Any) -> float:
        """Compute Euclidean center distance between two objects."""
        return float(
            np.linalg.norm(self.position(state, obj_a) - self.position(state, obj_b))
        )

    def vertical_gap(self, state: Any, obj_a: Any, obj_b: Any) -> float:
        """Compute vertical geometric gap between two rectangle-like objects."""
        y_a = float(self.get(state, obj_a, "y"))
        h_a = float(self.get(state, obj_a, "height"))
        y_b = float(self.get(state, obj_b, "y"))
        h_b = float(self.get(state, obj_b, "height"))

        center_gap = abs(y_a - y_b)
        extent_gap = center_gap - h_a / 2.0 - h_b / 2.0
        return float(max(0.0, extent_gap))

    def passage_center(
        self,
        state: Any,
        bottom_obstacle: Any,
        top_obstacle: Any,
    ) -> np.ndarray:
        """Return the center of a vertical passage from its obstacle pair."""
        x = float(self.get(state, bottom_obstacle, "x"))
        bottom_y = float(self.get(state, bottom_obstacle, "y"))
        bottom_height = float(self.get(state, bottom_obstacle, "height"))
        top_y = float(self.get(state, top_obstacle, "y"))
        top_height = float(self.get(state, top_obstacle, "height"))

        return np.array(
            [
                x,
                (bottom_y + bottom_height / 2.0 + top_y - top_height / 2.0) / 2.0,
            ]
        )

    def passage_distance(
        self,
        state: Any,
        robot: Any,
        bottom_obstacle: Any,
        top_obstacle: Any,
    ) -> float:
        """Compute source-center distance to a vertical passage center."""
        robot_position = self.position(state, robot)
        passage_center = self.passage_center(
            state,
            bottom_obstacle,
            top_obstacle,
        )
        return float(np.linalg.norm(robot_position - passage_center))

    def min_keypoint_distance(
        self,
        state: Any,
        source: Any,
        target: Any,
        keypoint_type: str = "heuristic",
    ) -> float:
        """Compute source-keypoint to target distance.

        For DynPushPullHook2D hooks, ``hook_heuristic`` samples useful points
        around the L-shape and measures each point against the target rectangle.
        """
        if keypoint_type == "hook_heuristic":
            center = self.position(state, source)
            theta = float(self.get(state, source, "theta"))
            width = float(self.get(state, source, "width"))
            length_side1 = float(self.get(state, source, "length_side1"))
            length_side2 = float(self.get(state, source, "length_side2"))

            target_center = self.position(state, target)
            target_theta = float(self.get(state, target, "theta"))
            target_width = float(self.get(state, target, "width"))
            target_height = float(self.get(state, target, "height"))

            keypoints = hook_heuristic_keypoints(
                center,
                theta,
                width,
                length_side1,
                length_side2,
            )
            return float(
                min(
                    point_to_oriented_rectangle_distance(
                        point,
                        target_center,
                        target_theta,
                        target_width,
                        target_height,
                    )
                    for point in keypoints
                )
            )

        return self.distance(state, source, target)

    def intersects(self, state: Any, obj_a: Any, obj_b: Any) -> bool:
        """Return whether two objects intersect.

        Placeholder implementation. Exact geometry intersection should be added
        by object-specific geometry plugins.
        """
        return self.distance(state, obj_a, obj_b) <= 0.0

    def _state_for_object(self, state: KinDERSceneState, obj: Any) -> Any:
        """Return the backing state that owns an object."""
        if self._contains_object(state.dynamic_state, obj):
            return state.dynamic_state
        if self._contains_object(state.constant_state, obj):
            return state.constant_state
        raise KeyError(f"Object not found in scene state: {obj}")

    @staticmethod
    def _contains_object(object_state: Any, obj: Any) -> bool:
        """Return whether an object-centric state contains an object."""
        return any(candidate == obj for candidate in object_state)
