# Current Scope

The current version validates the reward grounding interface using
hand-crafted / oracle-style subgoals.

## What is implemented

- Structured `RewardSpec`, `SubgoalSpec`, predicate, and progress metric data
  models.
- Oracle object-centric predicates and progress metrics for 2D KinDER-style
  tasks.
- A sequential composer that turns subgoal evaluations into scalar rewards.
- A `RewardEvaluator` compatible with MPC / trajectory scoring.
- A deterministic `MockVLMGrounder` for tests.

## What is not implemented yet

- Real VLM inference is not implemented yet.
- VLM prompting, image construction, scene serialization, and calibration are
  not implemented.
- Automatic language-to-subgoal generation is not implemented.
- 3D reward grounding is not validated yet.

## Environment target

The current main environment target is 2D. The active examples are Motion2D and
DynPushPullHook2D-style reward specifications.

3D feasibility and VLM grounding are planned next.
