"""Minimal reproduction: is the refiner's transition function faithful to execution?

SCRATCH ONLY. No planner, no sampling, no controllers.

Replays ONE fixed action sequence two ways from the same reset:
  A) continuously            -- exactly what `env.step` does during execution
  B) reset-per-step          -- exactly what `transition_fn` does during refinement,
                                i.e. sim.set_state(x_prev); sim.step(u)
and reports where the two diverge. If B tracks A, set_state is faithful and refinement's
acceptance means something. If B departs from A, every sample the refiner accepts was
validated against dynamics the environment will not reproduce.

Run under each kindergarden variant to compare.
"""

import argparse
import json
import os

import kinder
import numpy as np
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv  # noqa: F401

kinder.register_all_environments()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from kinder_bilevel_planning.env_models import create_bilevel_planning_models  # noqa: E402

CUBE = "cube_0"


def cube(state):  # noqa: ANN001, ANN201
    o = state.get_object_from_name(CUBE)
    return [float(state.get(o, f)) for f in ("x", "y", "z")]


def grip(env):  # noqa: ANN001, ANN201
    return [
        float(v)
        for v in env.unwrapped._object_centric_env._robot_env.sim.data.mj_data.qpos[31:39]  # noqa: SLF001
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--actions", type=str, required=True, help="json list of actions")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--label", type=str, default="")
    args = ap.parse_args()

    actions = [np.asarray(a, dtype=np.float32) for a in json.load(open(args.actions))]

    # ---- A: continuous execution
    envA = kinder.make("kinder/Tossing3D-o1-v0", render_mode=None)
    obsA, _ = envA.reset(seed=args.seed)
    models = create_bilevel_planning_models(
        "tidybot3d_tossing3D", envA.observation_space, envA.action_space, num_objects=1
    )
    trajA = [cube(models.observation_to_state(obsA))]
    gripA = [grip(envA)]
    for u in actions:
        obsA, _, _, _, _ = envA.step(u)
        trajA.append(cube(models.observation_to_state(obsA)))
        gripA.append(grip(envA))
    simA = envA.unwrapped._object_centric_env  # noqa: SLF001
    scoredA = bool(simA._check_goals())  # noqa: SLF001

    # ---- B: reset-per-step -- the refiner's OWN transition function, verbatim.
    envB = kinder.make("kinder/Tossing3D-o1-v0", render_mode=None)
    obsB, _ = envB.reset(seed=args.seed)
    x = models.observation_to_state(obsB)
    trajB = [cube(x)]
    gripB: list[list[float]] = []
    for u in actions:
        x = models.transition_fn(x, u)
        trajB.append(cube(x))
    # Score path B's final cube with the same ground fixture the goal check uses.
    gf = simA._ground_fixture  # noqa: SLF001
    scoredB = bool(
        gf.check_in_region(
            np.asarray(trajB[-1], dtype=np.float32),
            "blocks_goal_region",
            simA._robot_env,  # noqa: SLF001
        )
    )

    A = np.asarray(trajA)
    B = np.asarray(trajB)
    n = min(len(A), len(B))
    per = np.max(np.abs(A[:n] - B[:n]), axis=1)
    nz = np.nonzero(per > 1e-6)[0]
    out = {
        "label": args.label,
        "seed": args.seed,
        "n_actions": len(actions),
        "continuous_final_cube": trajA[-1],
        "resetperstep_final_cube": trajB[-1],
        "continuous_scored": scoredA,
        "resetperstep_scored": scoredB,
        "max_divergence": float(per.max()),
        "first_step_over_1e-6": int(nz[0]) if len(nz) else None,
        "per_step": per.tolist(),
        "grip_continuous": gripA,
        "grip_resetperstep": gripB,
        "traj_continuous": trajA,
        "traj_resetperstep": trajB,
    }
    with open(args.out, "w") as f:
        f.write(json.dumps(out) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "label",
                    "seed",
                    "n_actions",
                    "continuous_final_cube",
                    "resetperstep_final_cube",
                    "continuous_scored",
                    "resetperstep_scored",
                    "max_divergence",
                    "first_step_over_1e-6",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
