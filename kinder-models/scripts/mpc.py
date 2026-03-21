import argparse
import os

import kinder
import numpy as np
import torch
from PIL import Image as PILImage

from state_wm import MLPDynamics

ROBOT_STATE_DIM = 9


def cost(state: np.ndarray) -> float:
    """Euclidean distance from robot to target region center."""
    robot_xy = state[:2]
    target_xy = state[9:11]
    return float(np.linalg.norm(robot_xy - target_xy))


def load_world_model(checkpoint: str):
    """Load trained MLPDynamics and its normalizers from a checkpoint."""
    ckpt  = torch.load(checkpoint, weights_only=False)
    model = MLPDynamics(ckpt["state_dim"], ckpt["action_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norms = {
        "s_mean": ckpt["s_norm"]["mean"],
        "s_std":  ckpt["s_norm"]["std"],
        "a_mean": ckpt["a_norm"]["mean"],
        "a_std":  ckpt["a_norm"]["std"],
        "d_mean": ckpt["d_norm"]["mean"],
        "d_std":  ckpt["d_norm"]["std"],
    }
    return model, norms


def wm_get_next_state(
    state: np.ndarray,
    action: np.ndarray,
    model: MLPDynamics,
    norms: dict,
) -> np.ndarray:
    """Predict next state using the learned world model."""
    s_in = torch.tensor(
        (state - norms["s_mean"]) / norms["s_std"], dtype=torch.float32
    )
    a_in = torch.tensor(
        (action - norms["a_mean"]) / norms["a_std"], dtype=torch.float32
    )
    with torch.no_grad():
        d_pred = model(s_in.unsqueeze(0), a_in.unsqueeze(0)).squeeze(0).numpy()
    delta = d_pred * norms["d_std"] + norms["d_mean"]
    next_state = state.copy()
    next_state[:ROBOT_STATE_DIM] += delta
    return next_state


def main() -> None:
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
    obs, info = env.reset(seed=42)

    # kinder.make() wraps the environment in gymnasium wrappers.
    # Access the unwrapped env to use the state interface.
    unwrapped = env.unwrapped

    print("Observation shape:", env.observation_space.shape)
    print("Action shape:     ", env.action_space.shape)

    num_candidates = 50
    horizon = 5
    max_steps = 300
    rng = np.random.default_rng(0)

    obs, info = env.reset(seed=42)
    frames = [env.render()]

    for step in range(max_steps):
        current_state = unwrapped.get_state()

        # Sample random action sequences (only dx, dy matter; zero out the rest).
        raw = rng.uniform(
            low=env.action_space.low[:2],
            high=env.action_space.high[:2],
            size=(num_candidates, horizon, 2),
        ).astype(np.float32)
        action_sequences = np.zeros(
            (num_candidates, horizon, env.action_space.shape[0]), dtype=np.float32
        )
        action_sequences[:, :, :2] = raw

        # Evaluate each candidate by rolling out forward.
        best_cost = float("inf")
        best_idx = 0
        for i in range(num_candidates):
            state = current_state
            for t in range(horizon):
                if args.use_world_model:
                    state = wm_get_next_state(
                        state, action_sequences[i, t], wm_model, wm_norms
                    )
                else:
                    state = unwrapped.get_next_state(state, action_sequences[i, t])
            c = cost(state)
            if c < best_cost:
                best_cost = c
                best_idx = i

        # Restore the real state and execute the best first action.
        unwrapped.set_state(current_state)
        obs, reward, terminated, truncated, info = env.step(action_sequences[best_idx, 0])
        frames.append(env.render())

        if terminated or truncated:
            print(f"Reached goal in {step + 1} steps!")
            break
    else:
        print(f"Did not reach goal within {max_steps} steps (final cost: {cost(obs):.3f}).")

    pil_frames = [PILImage.fromarray(f) for f in frames]
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

    env.close()


if __name__ == "__main__":
    main()
