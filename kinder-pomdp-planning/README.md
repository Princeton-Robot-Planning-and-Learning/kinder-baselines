# POMDP Planning Baselines for KinDER

Online POMDP planning with [POMCPOW](https://arxiv.org/abs/1709.06196) (Sunberg & Kochenderfer, ICAPS 2018), covering the PuckPushing environments.

## Installation

```bash
uv pip install -e ".[develop]"
```

## Solver

`kinder_pomdp_planning.pomcpow` is domain-free. A domain implements the `POMDPModel` protocol: sample a transition and its reward, sample an observation and give its likelihood, say when a state is terminal, propose a new action, and estimate a leaf's value. New actions at a tree node come from an `ActionProposer`, which defaults to the model's own sampler. Passing another proposer (for example a learned one) changes only where the search looks.

## Environments

| Environment | Hidden parameter | Action |
|---|---|---|
| `puckpushing` (`DynPuckPushing2D`, `PuckPushing3D`) | Sliding friction of each floor surface (laminate, wood, felt) | One push: release heading and speed |

### PuckPushing

A robot pushes a puck through a floor maze to a goal. The maze has three routes: a slippery laminate corridor with a hole past its far end, a slow felt corridor with a hole in its middle, and a slot through the middle that the robot cannot enter. The slot shot reaches the goal only if the floor is slippery enough. The puck fails if its center passes over a hole, and it can also be stranded where the robot cannot get behind it.

- **State:** puck and robot positions, plus the hidden friction vector.
- **Transitions:** the analytic slide model from `kinder.envs.puckpushing_utils`, with the envs' execution noise on heading and speed.
- **Observations:** noisy release speed and stop position.
- **Rewards:** -1 per push and -0.5 per meter driven to the push station; +20 at the goal; -50 in a hole.
- **Default proposer:** uniformly random feasible pushes.
- **Leaf values:** a greedy rollout along the shortest path to the goal. Its speeds assume the belief-mean frictions and its slides use the particle's own.
- **Belief:** a particle filter over the frictions, updated after each real push.

```bash
# 2D (PyMunk) or 3D (PyBullet) env, 1000 simulations per push.
python -m kinder_pomdp_planning.puckpushing.run --env 2d --seed 0 --sims 1000
python -m kinder_pomdp_planning.puckpushing.run --env 3d --seed 0 --sims 1000 --out log.json
```
