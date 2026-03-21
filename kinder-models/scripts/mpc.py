import numpy as np
import matplotlib.pyplot as plt

import kinder

import os
from PIL import Image as PILImage

def cost(state: np.ndarray) -> float:
    """Euclidean distance from robot to target region center."""
    robot_xy = state[:2]
    target_xy = state[9:11]
    return float(np.linalg.norm(robot_xy - target_xy))

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

    # Evaluate each candidate by simulating forward.
    best_cost = float("inf")
    best_idx = 0
    for i in range(num_candidates):
        state = current_state
        for t in range(horizon):
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
output_path = "output/mpc.gif"
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
