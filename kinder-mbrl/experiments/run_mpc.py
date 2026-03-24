"""Random-shooting Model Predictive Control (MPC) for Motion2D-p0.

At each step, samples a set of random action sequences, rolls each one out over
a fixed horizon to estimate the resulting cost, and executes the first action of
the lowest-cost sequence in the real environment.

Planning rollouts can be performed with three configurations:

  1. Simulator (default):
       cost = state_cost(final_state)   [Euclidean distance to goal]

  2. World model only (--use_world_model):
       cost = state_cost(final_state)   [same metric, predicted state]

  3. World model + termination classifier (--use_world_model --term_checkpoint):
       reward_t = -1 + term_prob_t      [soft sparse reward per step]
       cost     = -sum_t(reward_t)      [mirrors KinderTrajOptProblem.get_traj_cost]
       Rollout for a candidate stops early when term_prob > --term_threshold,
       mirroring the early-exit on termination in get_traj_cost.

Usage:
    # Simulator-based MPC
    python experiments/run_mpc.py

    # World-model-based MPC (distance cost)
    python experiments/run_mpc.py --use_world_model --checkpoint output/wm.pt

    # World-model-based MPC with termination classifier (soft-reward cost)
    python experiments/run_mpc.py --use_world_model --checkpoint output/wm.pt \\
        --term_checkpoint output/term.pt
"""

import argparse
import os

import kinder
import numpy as np
from PIL import Image as PILImage

from kinder_mbrl.planning import (
    load_termination_classifier,
    load_world_model,
    state_cost,
    wm_get_next_state,
    wm_get_termination_prob,
)


def main() -> None:
    """Run random-shooting MPC on Motion2D-p0 and save a GIF of the rollout.

    See module docstring for the three planning configurations.
    In all cases the real environment is used to execute the chosen action and
    advance the true state.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_world_model",
        action="store_true",
        help="Use learned world model for planning rollouts instead of the simulator.",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/wm.pt",
        help="Path to world model checkpoint (used with --use_world_model).",
    )
    parser.add_argument(
        "--term_checkpoint",
        default="",
        help="Path to termination classifier checkpoint (term.pt). When given "
        "alongside --use_world_model, replaces the distance-based planning cost "
        "with a soft termination-probability cost.",
    )
    parser.add_argument(
        "--term_threshold",
        type=float,
        default=0.5,
        help="Termination probability threshold for early exit during planning "
        "rollout (used with --term_checkpoint). Default: 0.5.",
    )
    parser.add_argument("--num_candidates", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wm_model, wm_norms = None, None
    if args.use_world_model:
        print(f"Loading world model from {args.checkpoint} ...")
        wm_model, wm_norms = load_world_model(args.checkpoint)
        print("World model loaded.")

    term_model, term_norms = None, None
    if args.term_checkpoint:
        if not args.use_world_model:
            raise ValueError("--term_checkpoint requires --use_world_model.")
        print(f"Loading termination classifier from {args.term_checkpoint} ...")
        term_model, term_norms = load_termination_classifier(args.term_checkpoint)
        print("Termination classifier loaded.")

    use_term_cost = term_model is not None

    kinder.register_all_environments()

    env = kinder.make(
        "kinder/Motion2D-p0-v0",
        render_mode="rgb_array",
        allow_state_access=True,
    )

    unwrapped = env.unwrapped

    print("Observation shape:", env.observation_space.shape)
    print("Action shape:     ", env.action_space.shape)
    print(
        f"Planning cost: {'soft termination probability' if use_term_cost else 'distance to goal'}"
    )

    rng = np.random.default_rng(0)

    obs, _ = env.reset(seed=args.seed)
    frames = [env.render()]  # type: ignore

    for step in range(args.max_steps):
        current_state = unwrapped.get_state()  # type: ignore

        raw = rng.uniform(
            low=env.action_space.low[:2],  # type: ignore
            high=env.action_space.high[:2],  # type: ignore
            size=(args.num_candidates, args.horizon, 2),
        ).astype(np.float32)
        action_sequences = np.zeros(
            (args.num_candidates, args.horizon, env.action_space.shape[0]),  # type: ignore # pylint: disable=line-too-long
            dtype=np.float32,
        )
        action_sequences[:, :, :2] = raw

        best_cost = float("inf")
        best_idx = 0
        for cand_idx in range(args.num_candidates):
            state = current_state
            cost_val: float

            if use_term_cost:
                # Mirror KinderTrajOptProblem.get_traj_cost for sparse rewards:
                #   reward_t = -1 + term_prob_t
                #     ≈ -1 far from goal (term_prob ≈ 0), ≈ 0 at goal (term_prob ≈ 1)
                #   cost = -total_reward  (lower is better)
                # Stop accumulating once the predicted termination fires, just
                # as get_traj_cost breaks on the first terminated=True step.
                total_reward = 0.0
                for t_idx in range(args.horizon):
                    state = wm_get_next_state(
                        state,
                        action_sequences[cand_idx, t_idx],
                        wm_model,  # type: ignore
                        wm_norms,  # type: ignore
                    )
                    prob = wm_get_termination_prob(
                        state, term_model, term_norms  # type: ignore
                    )
                    total_reward += -1.0 + prob
                    if prob > args.term_threshold:
                        break
                cost_val = -total_reward
            else:
                for t_idx in range(args.horizon):
                    if args.use_world_model:
                        state = wm_get_next_state(
                            state,
                            action_sequences[cand_idx, t_idx],
                            wm_model,  # type: ignore
                            wm_norms,  # type: ignore
                        )
                    else:
                        state = unwrapped.get_next_state(  # type: ignore
                            state, action_sequences[cand_idx, t_idx]
                        )
                cost_val = state_cost(state)

            if cost_val < best_cost:
                best_cost = cost_val
                best_idx = cand_idx

        unwrapped.set_state(current_state)  # type: ignore
        obs, _, terminated, truncated, _ = env.step(action_sequences[best_idx, 0])  # type: ignore # pylint: disable=line-too-long
        frames.append(env.render())

        if terminated or truncated:
            print(f"Reached goal in {step + 1} steps!")
            break
    else:
        print(
            f"Did not reach goal within {args.max_steps} steps "
            f"(final cost: {state_cost(obs):.3f})."
        )

    pil_frames = [PILImage.fromarray(f) for f in frames]  # type: ignore
    if use_term_cost:
        gif_name = "mpc_wm_term.gif"
    elif args.use_world_model:
        gif_name = "mpc_wm.gif"
    else:
        gif_name = "mpc.gif"
    output_path = f"output/{gif_name}"
    os.makedirs("output", exist_ok=True)
    pil_frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,
        loop=0,
    )
    print(f"GIF saved to {output_path}")

    env.close()  # type: ignore


if __name__ == "__main__":
    main()
