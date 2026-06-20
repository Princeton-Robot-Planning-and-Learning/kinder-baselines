# Roadmap

## Phase 0: hand-crafted reward grounding prototype

- Define structured reward specs.
- Implement oracle predicates and progress metrics.
- Keep the 2D demo runnable.
- Add documentation that clearly states the VLM gap.

## Phase 1: 2D evaluation with statistics script

- Add repeatable 2D evaluation runs across seeds.
- Report success rate, final distance, best distance, and step count.
- Compare sparse reward, oracle reward, and spec-grounded reward where useful.

## Phase 2: VLM grounding interface

- Replace the placeholder with a real `VLMGrounder` implementation.
- Decide the scene/query representation passed to the VLM.
- Add deterministic mocks and cached fixtures for tests.
- Keep planner-side code independent of the VLM server/GPU setup.

## Phase 3: 3D feasibility check

- Identify 3D KinDER tasks where reward grounding is meaningful.
- Check available object state, rendered views, and geometry interfaces.
- Prototype simple 3D predicates or VLM scoring hooks.

## Phase 4: 3D + VLM reward grounding evaluation

- Evaluate VLM-grounded rewards in selected 3D environments.
- Measure reliability, reward calibration, failure modes, and runtime cost.
- Compare against oracle specs and sparse baselines.
