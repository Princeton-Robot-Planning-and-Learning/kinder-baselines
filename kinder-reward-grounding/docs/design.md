# Design

This prototype represents reward grounding as a small pipeline:

```text
Task / language goal
    -> subgoal specification
    -> predicate library
    -> progress score
    -> composite reward
    -> trajectory scoring / MPC
```

## Pipeline

1. Task / language goal

   A user-level task such as "reach the goal" or "use the hook to move the
   target" is the starting point. Today this is not parsed from free-form
   language by a model; it is encoded manually in Python.

2. Subgoal specification

   A `RewardSpec` lists task objects and ordered `SubgoalSpec` entries. Each
   subgoal has a completion predicate, a dense progress metric, a weight, and a
   completion bonus.

3. Predicate library

   Predicates return booleans such as "object is held", "two objects are near",
   or "vertical gap is below threshold". The current predicates use
   object-centric oracle state through an adapter.

4. Progress score

   Progress metrics return scalar dense shaping terms, usually positive when a
   useful distance or gap decreases across a transition.

5. Composite reward

   A reward composer combines progress, completion bonuses, time penalties, and
   success bonuses. The current implemented composer is sequential: only the
   first subgoal incomplete before the transition contributes progress.

6. Trajectory scoring / MPC

   `RewardEvaluator` exposes the final scalar reward function in the form
   expected by trajectory optimization code. MPC can call this reward function
   repeatedly while scoring candidate action sequences.

## Boundaries

- `adapters/` owns environment-specific state decoding and object queries.
- `predicates/` owns boolean subgoal completion checks.
- `rewards/` owns dense progress metrics, reward composition, and evaluation.
- `grounders/` owns future VLM-style scene/query scoring interfaces.
- `env_specs/` currently owns hand-written, oracle-style task specs.

The intended long-term shape is that a VLM or VLM-assisted module proposes or
scores grounded subgoals, while deterministic reward evaluation remains easy to
test and inspect.
