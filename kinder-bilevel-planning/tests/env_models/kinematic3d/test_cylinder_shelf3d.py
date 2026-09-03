"""Tests for cylinder_shelf3d.py."""

import json
import time
from pathlib import Path

import kinder
import numpy as np
import pytest
from bilevel_planning.structs import RelationalAbstractState
from kinder.envs.kinematic3d.cylinder_shelf3d import (
    CylinderShelf3DEnv,
    CylinderShelf3DEnvConfig,
)
from kinder_models.structs import SkillCall
from pybullet_helpers.geometry import Pose, SE2Pose
from relational_structs import GroundAtom

from kinder_bilevel_planning.agent import BilevelPlanningAgent
from kinder_bilevel_planning.env_models import create_bilevel_planning_models
from kinder_bilevel_planning.injection import (
    place_params_from_ir,
    run_injected_sesame,
    skeleton_from_ir,
)

kinder.register_all_environments()


def _make_agent(env, magic_skills, **kwargs):
    env_models = create_bilevel_planning_models(
        "cylinder_shelf3d",
        env.observation_space,
        env.action_space,
        magic_skills=magic_skills,
        **kwargs,
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=123,
        max_abstract_plans=10,
        samples_per_step=3,
        planning_timeout=600.0,
        max_skill_horizon=400,
    )
    return env_models, agent


def _predicate(env_models, name):
    return next(p for p in env_models.predicates if p.name == name)


def test_cylinder_shelf3d_magic_grasp_planning_and_execution():
    """With Grasp magic, the plan reaches the pre-grasp pose with planned motion, holds
    exactly one SkillCall for the grasp whose predicted state satisfies Holding, and
    executing the plan with the SkillCall carried out as a teleport reaches the goal."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0", allow_state_access=True)
    env_models, agent = _make_agent(env, magic_skills=("Grasp",))
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)

    trajectory = agent.plan()
    calls = [
        (i, x, u) for i, (x, u) in enumerate(trajectory) if isinstance(u, SkillCall)
    ]
    assert len(calls) == 1
    index, pre_state, call = calls[0]
    assert call.skill_name == "Grasp"
    assert [o.name for o in call.objects] == ["robot", "cylinder0"]
    # The approach to the pre-grasp pose is planned motion, not magic.
    assert index > 0
    assert all(isinstance(u, np.ndarray) for _, u in trajectory[:index])

    robot = pre_state.get_object_from_name("robot")
    cylinder = pre_state.get_object_from_name("cylinder0")
    at_pre_grasp = GroundAtom(_predicate(env_models, "AtPreGrasp"), [robot, cylinder])
    holding = GroundAtom(_predicate(env_models, "Holding"), [robot, cylinder])
    before = env_models.state_abstractor(pre_state)
    after = env_models.state_abstractor(call.predicted_state)
    assert isinstance(before, RelationalAbstractState)
    assert at_pre_grasp in before.atoms and holding not in before.atoms
    assert holding in after.atoms and at_pre_grasp not in after.atoms
    assert env_models.transition_fn(pre_state, call) == call.predicted_state

    # Execute: regular actions step the env; the SkillCall teleports it.
    terminated = False
    for _, action in trajectory:
        if isinstance(action, SkillCall):
            env.unwrapped.set_state(
                env.observation_space.vectorize(action.predicted_state)
            )
            continue
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    assert terminated, "Executing the magic plan did not reach the goal"
    env.close()


def test_cylinder_shelf3d_magic_skills_validation():
    """Unknown skill names are rejected at model construction."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0")
    with pytest.raises(ValueError, match="Unknown magic skill"):
        create_bilevel_planning_models(
            "cylinder_shelf3d",
            env.observation_space,
            env.action_space,
            magic_skills=("Pick",),
        )
    env.close()


def test_cylinder_shelf3d_without_magic_has_no_skill_calls():
    """Without magic skills the plan is a plain low-level action sequence that reaches
    the goal."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0")
    _, agent = _make_agent(env, magic_skills=())
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)
    trajectory = agent.plan()
    assert trajectory
    assert all(isinstance(u, np.ndarray) for _, u in trajectory)
    terminated = False
    for _, action in trajectory:
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    assert terminated
    env.close()


def test_cylinder_shelf3d_fixed_place_params():
    """With place_params the plan sets the cylinder down at the given offset from the
    shelf centre."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0", allow_state_access=True)
    offset = (-0.10, -0.05)
    _, agent = _make_agent(env, magic_skills=("Grasp",), place_params=[offset])
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)

    terminated = False
    for _, action in agent.plan():
        if isinstance(action, SkillCall):
            obs = env.observation_space.vectorize(action.predicted_state)
            env.unwrapped.set_state(obs)
            continue
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    assert terminated, "Executing the plan did not reach the goal"

    state = env.observation_space.devectorize(obs)
    shelf_pose = state.get_object_pose("shelf")
    cylinder = state.get_object_from_name("cylinder0")
    placed_xy = (state.get(cylinder, "pose_x"), state.get(cylinder, "pose_y"))
    expected_xy = (
        shelf_pose.position[0] + offset[0],
        shelf_pose.position[1] - 0.05 + offset[1],
    )
    assert np.allclose(placed_xy, expected_xy, atol=0.02), (placed_xy, expected_xy)
    env.close()


def test_cylinder_shelf3d_place_params_validation():
    """One place parameter entry per cylinder, or a ValueError."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0")
    with pytest.raises(ValueError, match="place_params has 2 entries"):
        create_bilevel_planning_models(
            "cylinder_shelf3d",
            env.observation_space,
            env.action_space,
            place_params=[(0.0, 0.0), None],
        )
    env.close()


def test_cylinder_shelf3d_two_cylinders_first_skeleton():
    """MoveToPreGrasp requires and deletes NotAtPreGrasp, so the abstract planner
    cannot chain two MoveToPreGrasps and its first skeleton for two cylinders is
    refinable: planning succeeds with a single abstract plan, one sample per step
    (the placements are fully fixed), and one MoveToPreGrasp + Grasp per
    cylinder."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o2-v0", allow_state_access=True)
    env_models = create_bilevel_planning_models(
        "cylinder_shelf3d",
        env.observation_space,
        env.action_space,
        num_objects=2,
        magic_skills=("MoveToPreGrasp", "Grasp"),
        place_params=[(-0.10, -0.05, 0.86), (0.10, -0.05, 0.86)],
    )
    agent = BilevelPlanningAgent(
        env_models,
        seed=123,
        max_abstract_plans=1,
        samples_per_step=1,
        planning_timeout=600.0,
        max_skill_horizon=400,
    )
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)
    calls = [u for _, u in agent.plan() if isinstance(u, SkillCall)]
    assert [c.skill_name for c in calls] == ["MoveToPreGrasp", "Grasp"] * 2
    assert {c.objects[1].name for c in calls} == {"cylinder0", "cylinder1"}
    env.close()


def test_cylinder_shelf3d_plans_with_base_clearance():
    """base_clearance reaches the skills' base motion planner and planning still
    succeeds with a few centimetres of it."""
    env = kinder.make("kinder/KinematicCylinderShelf3D-o1-v0", allow_state_access=True)
    _, agent = _make_agent(env, magic_skills=("Grasp",), base_clearance=0.05)
    obs, info = env.reset(seed=123)
    agent.reset(obs, info)
    assert len(agent.plan()) > 0
    env.close()


def _real_restock_config_and_params():
    """The measured physical restock scene plus the fixed per-cylinder parameters a
    high-level planner would inject (see test_injected_skeleton...)."""
    board_half = 0.0127 / 2
    deep_center = (0.9075, 1.49)
    shallow_center, shallow_yaw = (0.40, 1.28), 0.25

    def zigzag(center, yaw, pitch, dy):
        out = []
        for lx, ly in [(-pitch, -dy), (0.0, dy), (pitch, -dy)]:
            c, s = np.cos(yaw), np.sin(yaw)
            out.append((center[0] + c * lx - s * ly, center[1] + s * lx + c * ly))
        return out

    spots = zigzag(deep_center, 0.0, 0.13, 0.06) + zigzag(
        shallow_center, shallow_yaw, 0.13, 0.07
    )
    config = CylinderShelf3DEnvConfig(
        shelf_pose=Pose((1.63, 1.51, 0.0)),
        shelf_layer_zs=(0.100 - board_half, 0.538 - board_half, 0.800 - board_half),
        cylinder_heights=(0.29, 0.208, 0.233, 0.12, 0.125, 0.10),
        cylinder_radii=(0.0375, 0.0375, 0.0375, 0.0375, 0.035, 0.0325),
        boxes=(
            (0.71, 1.105, 1.34125, 1.63875, 0.215),
            (0.20, 0.60, 1.12, 1.44, 0.115, shallow_yaw),
        ),
        cylinder_init_regions=tuple((x, x, y, y) for x, y in spots),
        robot_base_home_pose=SE2Pose(1.48, 0.67, 1.54),
        robot_base_pose_lower_bound=SE2Pose(-0.2, -0.2, -np.pi),
        robot_base_pose_upper_bound=SE2Pose(2.0, 2.0, np.pi),
        x_lb=-0.2,
        x_ub=2.0,
        y_lb=-0.2,
        y_ub=2.0,
    )
    pitch45 = np.deg2rad(45)
    grasp_params = [
        (pitch45, 0.03), (pitch45, 0.05), (pitch45, 0.03),
        (pitch45, 0.015), (pitch45, 0.05), (pitch45, 0.015),
    ]
    # Staging rot pi/2 parks the base south of the cylinder, heading at it; the
    # shallow box is set down rotated by 0.25, so its cylinders are approached
    # along the box's own normal (pi/2 + 0.25) and the chassis aligns with it.
    move_params = [
        (0.83, np.pi / 2), (0.88, np.pi / 2), (0.83, np.pi / 2),
        (0.72, np.pi / 2 + shallow_yaw), (0.78, np.pi / 2 + shallow_yaw),
        (0.78, np.pi / 2 + shallow_yaw),
    ]
    place_params = [
        (-0.13, -0.05, 0.80, 0), (0.0, -0.05, 0.80, 0), (0.13, -0.05, 0.80, 0),
        (-0.13, -0.05, 0.80, 1), (0.0, -0.05, 0.80, 1), (0.13, -0.05, 0.80, 1),
    ]
    return config, grasp_params, move_params, place_params


def test_injected_skeleton_boxed_scene():
    """A fully specified intermediate representation — the skill skeleton plus fixed
    per-cylinder move/grasp/place parameters — refines with one sample per step and no
    abstract search, solving the boxed real-restock scene in a single pass."""
    config, grasp_params, move_params, place_params = _real_restock_config_and_params()
    env = CylinderShelf3DEnv(num_cylinders=6, config=config, allow_state_access=True)
    env_models = create_bilevel_planning_models(
        "cylinder_shelf3d",
        env.observation_space,
        env.action_space,
        num_objects=6,
        config=config,
        place_params=place_params,
        grasp_params=grasp_params,
        move_params=move_params,
        carry_lift_z=0.27,
    )
    obs, _ = env.reset(seed=0)
    x0 = env_models.observation_to_state(obs)
    skeleton = []
    for name in ("cylinder3", "cylinder4", "cylinder5",
                 "cylinder0", "cylinder1", "cylinder2"):
        skeleton.append(("MoveToPreGrasp", ("robot", name)))
        skeleton.append(("Grasp", ("robot", name)))
        skeleton.append(("Place", ("robot", name, "shelf")))
    start = time.monotonic()
    plan, _ = run_injected_sesame(
        env_models, x0, skeleton, samples_per_step=1, timeout=300.0
    )
    wall = time.monotonic() - start
    assert plan is not None, "Injected skeleton failed to refine"
    goal = env_models.goal_deriver(x0)
    assert goal.check_state(plan.states[-1])
    # The point of injection is speed: no search, one sample per step.
    assert wall < 120.0, f"Injected refinement too slow: {wall:.1f}s"


def test_injected_skeleton_from_alphatamp_ir():
    """End-to-end across the repo seam: a plan_ir.json emitted by alphatamp's restock3d
    deploy kit (checked in under fixtures/) drives planning in the boxed real-restock
    scene. The IR supplies the decisions — pick order, board layer, placement x — and
    this side supplies its own calibration for how to move (staging poses, grasp
    geometry, insertion depth, base standoff)."""
    ir = json.loads(
        (Path(__file__).parent / "fixtures" / "plan_ir.json").read_text()
    )
    config, grasp_params, move_params, _ = _real_restock_config_and_params()
    # The IR's object table must describe the same physical objects, in the same
    # order, as this scene's cylinders.
    for i, obj in enumerate(ir["objects"]):
        assert obj["cylinder"] == f"cylinder{i}"
        assert config.cylinder_heights[i] == pytest.approx(obj["height"])
    place_params = place_params_from_ir(ir, y_offset=-0.05, base_distance=0.80)
    env = CylinderShelf3DEnv(num_cylinders=6, config=config, allow_state_access=True)
    env_models = create_bilevel_planning_models(
        "cylinder_shelf3d",
        env.observation_space,
        env.action_space,
        num_objects=6,
        config=config,
        place_params=place_params,
        grasp_params=grasp_params,
        move_params=move_params,
        carry_lift_z=0.27,
    )
    obs, _ = env.reset(seed=0)
    x0 = env_models.observation_to_state(obs)
    plan, _ = run_injected_sesame(
        env_models, x0, skeleton_from_ir(ir), samples_per_step=1, timeout=300.0
    )
    assert plan is not None, "IR-driven skeleton failed to refine"
    goal = env_models.goal_deriver(x0)
    assert goal.check_state(plan.states[-1])
