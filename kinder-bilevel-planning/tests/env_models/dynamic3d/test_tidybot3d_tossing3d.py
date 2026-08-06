"""Tests for tidybot3d_tossing3D.py."""

import kinder
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models

kinder.register_all_environments()


def test_tidybot3d_tossing_bilevel_planning():
    """Tests for bilevel planning in the Tossing3D environment.

    The plan shape is fixed: pick the cube off the ground, drive to a throw standoff
    from the bin, and toss. What bilevel planning has to supply is the grasp standoff
    and the throw standoff; the toss itself samples nothing, so the two demonstrated
    arm confs are constants that only have to survive refinement.
    """
    num_objects = 1
    env = kinder.make(f"kinder/Tossing3D-o{num_objects}-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos", name_prefix="TidyBot3D-tossing3d")

    seed = 123
    obs, info = env.reset(seed=seed)
    total_reward = 0

    env_models = create_bilevel_planning_models(
        "tidybot3d_tossing3D",
        env.observation_space,
        env.action_space,
        num_objects=num_objects,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=seed,
        max_abstract_plans=1,
        samples_per_step=1,
        planning_timeout=120.0,
        max_skill_horizon=400,
    )

    agent.reset(obs, info)
    for _ in range(4000):
        action = agent.step()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        agent.update(obs, reward, terminated or truncated, info)
        if (
            terminated
            or truncated
            or len(agent._current_plan) == 0  # pylint: disable=protected-access
        ):
            break

    else:
        assert False, "Did not terminate successfully"

    # Running out of plan is not success. The refiner only ever returns a trajectory it
    # simulated into the goal, but it simulates through transition_fn, which restores a
    # state into a *separate* simulator; that the same actions also reach the goal when
    # replayed in the real env is a different claim and is the one worth pinning.
    assert terminated, "Executed the whole plan without the environment terminating"
    # And the planner's own abstraction agrees with the environment: InGoalRegion is
    # checked against the model sim's goal region, so this catches the two drifting
    # apart, which is what would make a refined plan and a real success diverge.
    final_state = env_models.observation_to_state(obs)
    assert env_models.goal_deriver(final_state).check_state(final_state)

    env.close()
