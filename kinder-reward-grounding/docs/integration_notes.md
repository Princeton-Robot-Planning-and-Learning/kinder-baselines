# Integration Notes

This package is expected to connect to the KinDER / KinderTrajOpt pipeline at
reward evaluation or trajectory scoring time.

## Expected integration point

Trajectory optimization code needs a scalar reward function for transitions:

```text
state, action, next_state, env_reward, terminated, env -> reward
```

`RewardEvaluator` provides that boundary. It wraps a `RewardSpec`, an
environment adapter, and a reward composer. The optimizer can call the evaluator
while scoring candidate trajectories, without needing to know how subgoals,
predicates, or object bindings are represented.

## KinDER adapter role

`KinDERObjectCentricAdapter` decodes flat KinDER planner states into
object-centric state, finds named objects, and exposes object features such as
position, distance, vertical gap, and heuristic keypoint distance.

That adapter is the current bridge from KinDER state to reward grounding logic.
If a 3D environment uses a different state representation, it should get its
own adapter rather than changing the generic reward evaluator.

## VLM grounding role

The future VLM integration should sit behind `VLMGrounder`. It should not be
called directly from trajectory optimization code. The intended shape is:

```text
scene / rendered observation + query
    -> VLMGrounder.score(...)
    -> normalized grounding score
    -> predicate or progress metric
    -> RewardEvaluator
```

The current `MockVLMGrounder` is only for deterministic tests and offline
development.

## Practical notes

- Keep reward specs inspectable and serializable enough for collaborator review.
- Keep planner-side code dependent on `RewardEvaluator`, not on individual
  VLM implementations.
- Use cached or mocked VLM outputs for CI; real VLM calls should be optional.
- Treat reward scale and completion thresholds as experiment configuration.
