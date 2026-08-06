"""Tests for the tossing oracle policy."""

import kinder
from conftest import MAKE_VIDEOS  # pylint: disable=import-error
from gymnasium.wrappers import RecordVideo
from relational_structs import ObjectCentricState

from kinder_models.dynamic3d.tossing.oracle_policy import Tossing3DOraclePolicy

kinder.register_all_environments()


def test_tossing3d_oracle_policy():
    """The oracle solves Tossing3D-o1 end to end.

    Three controllers, chosen from the abstract state alone, put the cube in the goal
    region -- which is the environment's own success criterion, so this asserts
    _check_goals() rather than a hand-written box test.
    """

    # Create the environment.
    env = kinder.make("kinder/Tossing3D-o1-v0", render_mode="rgb_array", scene_bg=False)
    if MAKE_VIDEOS:
        env.unwrapped._object_centric_env.set_render_camera(  # type: ignore # pylint: disable=protected-access
            "task_view"
        )
        env = RecordVideo(env, "unit_test_videos", name_prefix="Tossing3D-oracle")
    sim = env.unwrapped._object_centric_env  # pylint: disable=protected-access

    # The policy builds its own abstractor and controllers from the environment.
    policy = Tossing3DOraclePolicy(sim, env.action_space)

    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    assert isinstance(state, ObjectCentricState)
    assert not sim._check_goals()  # pylint: disable=protected-access

    # Run whatever the oracle asks for until it says the goal is reached. The bound is
    # the plan length the domain admits (pick, drive, toss) and not a search budget:
    # exceeding it means the oracle re-selected a controller, which is a failure.
    executed = []
    for _ in range(3):
        selection = policy.get_next_controller(state)
        if selection is None:
            break
        name, controller, params = selection
        executed.append(name)
        controller.reset(state, params)
        for _ in range(400):
            action = controller.step()
            obs, _, _, _, _ = env.step(action)
            state = env.observation_space.devectorize(obs)
            controller.observe(state)
            if controller.terminated():
                break
        else:
            assert False, f"Controller {name} did not terminate"

    assert executed == ["pick_shelf", "move_to_throw_pose", "toss_from_windup"]
    assert sim._check_goals()  # pylint: disable=protected-access
    assert policy.get_next_controller(state) is None

    env.close()
