"""Random-shooting Model Predictive Control (MPC) for Motion2D-p0.

At each step, samples a set of random action sequences, rolls each one out over
a fixed horizon to estimate the resulting cost, and executes the first action of
the lowest-cost sequence in the real environment.

Planning rollouts can be performed with either:
  - The ground-truth simulator (default): uses unwrapped.get_next_state().
  - A learned world model (--use_world_model): uses an MLPDynamics checkpoint
    trained by train_world_model.py to predict full state deltas.

Usage:
    # Simulator-based MPC
    python experiments/run_mpc.py

    # World-model-based MPC
    python experiments/run_mpc.py --use_world_model --checkpoint output/wm.pt
"""

import argparse
import os

import kinder
import numpy as np
from PIL import Image as PILImage

from kinder_mbrl.planning import load_world_model, state_cost, wm_get_next_state


def main() -> None:
    """Run random-shooting MPC on Motion2D-p0 and save a GIF of the rollout.

    Planning rollouts can use either the ground-truth simulator (default) or a learned
    world model (--use_world_model). In both cases the real environment is used to
    execute the chosen action and advance the true state.
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

    kinder.register_all_environments()

    env = kinder.make(
        "kinder/Motion2D-p0-v0",
        render_mode="rgb_array",
        allow_state_access=True,
    )

    unwrapped = env.unwrapped

    print("Observation shape:", env.observation_space.shape)
    print("Action shape:     ", env.action_space.shape)

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
            (args.num_candidates, args.horizon, env.action_space.shape[0]),  # type: ignore
            dtype=np.float32,
        )
        action_sequences[:, :, :2] = raw

        best_cost = float("inf")
        best_idx = 0
        for cand_idx in range(args.num_candidates):
            state = current_state
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
        obs, _, terminated, truncated, _ = env.step(action_sequences[best_idx, 0])  # type: ignore
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
    gif_name = "mpc_wm.gif" if args.use_world_model else "mpc.gif"
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
