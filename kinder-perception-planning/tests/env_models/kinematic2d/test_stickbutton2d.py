"""Tests for perception-based stickbutton2d planning models.

Uses a mock VLM (OrderedResponseModel) whose responses are generated
from the ground-truth bilevel planning state abstractor, so we can
verify the full pipeline: render -> prompt -> parse -> abstract state.
"""

import tempfile
from pathlib import Path

import kinder
from prpl_llm_utils.cache import FilePretrainedLargeModelCache
from prpl_llm_utils.models import OrderedResponseModel
from prpl_llm_utils.structs import Response
from relational_structs import GroundAtom, Object, ObjectCentricState, Predicate

from kinder.envs.kinematic2d.object_types import CircleType, CRVRobotType, RectangleType
from kinder_bilevel_planning.env_models.kinematic2d.stickbutton2d import (
    create_bilevel_planning_models as create_gt_models,
)
from kinder_perception_planning.env_models.kinematic2d.stickbutton2d import (
    _PREDICATE_DESCRIPTIONS,
    create_perception_planning_models,
)
from kinder_perception_planning.vlm_utils import (
    _build_atom_labelling_prompt,
    _parse_vlm_response,
)

kinder.register_all_environments()


def _make_vlm_response_for_atoms(
    candidate_atoms: list[GroundAtom],
    true_atoms: set[GroundAtom],
) -> str:
    """Build a VLM response string that labels each candidate atom."""
    lines = []
    for atom in candidate_atoms:
        value = "True" if atom in true_atoms else "False"
        lines.append(f"{atom}: {value}.")
    return "\n".join(lines)


def _build_candidate_atoms_for_state(state: ObjectCentricState, env_models):
    """Reconstruct the same candidate atom list that the perception
    state_abstractor would build."""
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    Grasped = pred_name_to_pred["Grasped"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    Pressed = pred_name_to_pred["Pressed"]
    RobotAboveButton = pred_name_to_pred["RobotAboveButton"]
    StickAboveButton = pred_name_to_pred["StickAboveButton"]
    AboveNoButton = pred_name_to_pred["AboveNoButton"]

    robot = state.get_objects(CRVRobotType)[0]
    stick = state.get_objects(RectangleType)[0]
    buttons = state.get_objects(CircleType)

    candidate_atoms: list[GroundAtom] = []
    candidate_atoms.append(GroundAtom(Grasped, [robot, stick]))
    candidate_atoms.append(GroundAtom(HandEmpty, [robot]))
    for button in buttons:
        candidate_atoms.append(GroundAtom(Pressed, [button]))
        candidate_atoms.append(GroundAtom(RobotAboveButton, [robot, button]))
        candidate_atoms.append(GroundAtom(StickAboveButton, [stick, button]))
    candidate_atoms.append(GroundAtom(AboveNoButton, []))
    return candidate_atoms


def _create_oracle_vlm(
    gt_models,
    perception_models,
    states: list[ObjectCentricState],
) -> OrderedResponseModel:
    """Create an OrderedResponseModel that returns ground-truth
    responses for each state."""
    responses = []
    for state in states:
        gt_abstract = gt_models.state_abstractor(state)
        candidate_atoms = _build_candidate_atoms_for_state(state, perception_models)
        response_text = _make_vlm_response_for_atoms(
            candidate_atoms, gt_abstract.atoms
        )
        responses.append(Response(response_text, metadata={}))
    cache_dir = Path(tempfile.mkdtemp())
    cache = FilePretrainedLargeModelCache(cache_dir)
    return OrderedResponseModel(responses, cache)


def _make_mock_vlm_and_models(env, num_buttons):
    """Helper to create a mock VLM and perception models."""
    cache_dir = Path(tempfile.mkdtemp())
    cache = FilePretrainedLargeModelCache(cache_dir)
    placeholder = [Response("placeholder", metadata={})]
    mock_vlm = OrderedResponseModel(placeholder, cache)
    perception_models = create_perception_planning_models(
        env.observation_space,
        env.action_space,
        num_buttons=num_buttons,
        vlm=mock_vlm,
    )
    return mock_vlm, perception_models


def _prime_mock_vlm(mock_vlm, gt_models, perception_models, states):
    """Load oracle responses into the mock VLM."""
    oracle_vlm = _create_oracle_vlm(gt_models, perception_models, states)
    mock_vlm._responses = oracle_vlm._responses  # pylint: disable=protected-access
    mock_vlm._next_response_idx = 0  # pylint: disable=protected-access


def test_stickbutton2d_vlm_state_abstractor_initial():
    """Test VLM state abstractor on the initial state (hand empty, above
    no button, nothing pressed)."""
    env = kinder.make("kinder/StickButton2D-b2-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    gt_models = create_gt_models(
        env.observation_space, env.action_space, num_buttons=2
    )
    state = gt_models.observation_to_state(obs)
    gt_abstract = gt_models.state_abstractor(state)

    mock_vlm, perception_models = _make_mock_vlm_and_models(env, num_buttons=2)
    _prime_mock_vlm(mock_vlm, gt_models, perception_models, [state])

    vlm_abstract = perception_models.state_abstractor(state)

    assert vlm_abstract.atoms == gt_abstract.atoms
    assert vlm_abstract.objects == gt_abstract.objects

    # Verify specifics: hand empty, above no button, nothing pressed.
    pred_names = {a.predicate.name for a in vlm_abstract.atoms}
    assert "HandEmpty" in pred_names
    assert "AboveNoButton" in pred_names
    assert "Pressed" not in pred_names
    assert "Grasped" not in pred_names

    env.close()


def test_stickbutton2d_vlm_state_abstractor_grasped():
    """Test VLM state abstractor when robot is holding the stick."""
    env = kinder.make("kinder/StickButton2D-b1-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    gt_models = create_gt_models(
        env.observation_space, env.action_space, num_buttons=1
    )
    state = gt_models.observation_to_state(obs)

    robot = state.get_objects(CRVRobotType)[0]
    stick = state.get_objects(RectangleType)[0]

    # Position robot to grasp the stick.
    state_grasped = state.copy()
    stick_x = state.get(stick, "x")
    stick_y = state.get(stick, "y")
    arm_joint = state.get(robot, "arm_joint")
    gripper_w = state.get(robot, "gripper_width")
    state_grasped.set(robot, "x", stick_x)
    state_grasped.set(robot, "y", stick_y + arm_joint + gripper_w / 2 + 0.01)
    state_grasped.set(robot, "vacuum", 1.0)

    gt_abstract = gt_models.state_abstractor(state_grasped)

    mock_vlm, perception_models = _make_mock_vlm_and_models(env, num_buttons=1)
    _prime_mock_vlm(mock_vlm, gt_models, perception_models, [state_grasped])

    vlm_abstract = perception_models.state_abstractor(state_grasped)

    assert vlm_abstract.atoms == gt_abstract.atoms

    # Verify Grasped is true.
    pred_name_to_pred = {p.name: p for p in perception_models.predicates}
    Grasped = pred_name_to_pred["Grasped"]
    assert GroundAtom(Grasped, [robot, stick]) in vlm_abstract.atoms

    env.close()


def test_stickbutton2d_vlm_state_abstractor_button_pressed():
    """Test VLM state abstractor when a button is pressed."""
    env = kinder.make("kinder/StickButton2D-b1-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    gt_models = create_gt_models(
        env.observation_space, env.action_space, num_buttons=1
    )
    state = gt_models.observation_to_state(obs)

    buttons = state.get_objects(CircleType)
    button0 = buttons[0]

    # Set button0 to pressed color (green).
    state_pressed = state.copy()
    state_pressed.set(button0, "color_r", 0.0)
    state_pressed.set(button0, "color_g", 0.9)
    state_pressed.set(button0, "color_b", 0.0)

    gt_abstract = gt_models.state_abstractor(state_pressed)

    mock_vlm, perception_models = _make_mock_vlm_and_models(env, num_buttons=1)
    _prime_mock_vlm(mock_vlm, gt_models, perception_models, [state_pressed])

    vlm_abstract = perception_models.state_abstractor(state_pressed)

    assert vlm_abstract.atoms == gt_abstract.atoms

    pred_name_to_pred = {p.name: p for p in perception_models.predicates}
    Pressed = pred_name_to_pred["Pressed"]
    assert GroundAtom(Pressed, [button0]) in vlm_abstract.atoms

    env.close()


def test_stickbutton2d_vlm_state_abstractor_robot_above_button():
    """Test VLM state abstractor when robot is above a button."""
    env = kinder.make("kinder/StickButton2D-b1-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    gt_models = create_gt_models(
        env.observation_space, env.action_space, num_buttons=1
    )
    state = gt_models.observation_to_state(obs)

    robot = state.get_objects(CRVRobotType)[0]
    button0 = state.get_objects(CircleType)[0]

    # Position robot directly over button0.
    state_above = state.copy()
    state_above.set(robot, "x", state.get(button0, "x"))
    state_above.set(robot, "y", state.get(button0, "y"))

    gt_abstract = gt_models.state_abstractor(state_above)

    mock_vlm, perception_models = _make_mock_vlm_and_models(env, num_buttons=1)
    _prime_mock_vlm(mock_vlm, gt_models, perception_models, [state_above])

    vlm_abstract = perception_models.state_abstractor(state_above)

    assert vlm_abstract.atoms == gt_abstract.atoms

    pred_name_to_pred = {p.name: p for p in perception_models.predicates}
    RobotAboveButton = pred_name_to_pred["RobotAboveButton"]
    assert GroundAtom(RobotAboveButton, [robot, button0]) in vlm_abstract.atoms
    # AboveNoButton should NOT be present.
    assert not any(
        a.predicate.name == "AboveNoButton" for a in vlm_abstract.atoms
    )

    env.close()


def test_stickbutton2d_vlm_state_abstractor_stick_above_button():
    """Test VLM state abstractor when stick is above a button."""
    env = kinder.make("kinder/StickButton2D-b1-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    gt_models = create_gt_models(
        env.observation_space, env.action_space, num_buttons=1
    )
    state = gt_models.observation_to_state(obs)

    stick = state.get_objects(RectangleType)[0]
    button0 = state.get_objects(CircleType)[0]

    # Position stick directly over button0.
    state_above = state.copy()
    state_above.set(stick, "x", state.get(button0, "x"))
    state_above.set(stick, "y", state.get(button0, "y"))

    gt_abstract = gt_models.state_abstractor(state_above)

    mock_vlm, perception_models = _make_mock_vlm_and_models(env, num_buttons=1)
    _prime_mock_vlm(mock_vlm, gt_models, perception_models, [state_above])

    vlm_abstract = perception_models.state_abstractor(state_above)

    assert vlm_abstract.atoms == gt_abstract.atoms

    pred_name_to_pred = {p.name: p for p in perception_models.predicates}
    StickAboveButton = pred_name_to_pred["StickAboveButton"]
    assert GroundAtom(StickAboveButton, [stick, button0]) in vlm_abstract.atoms

    env.close()


def test_stickbutton2d_vlm_prompt_construction():
    """Test that the VLM prompt is correctly constructed."""
    Grasped = Predicate("Grasped", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    Pressed = Predicate("Pressed", [CircleType])
    AboveNoButton = Predicate("AboveNoButton", [])

    robot = Object("robot", CRVRobotType)
    stick = Object("stick", RectangleType)
    button0 = Object("button0", CircleType)

    candidate_atoms = [
        GroundAtom(Grasped, [robot, stick]),
        GroundAtom(HandEmpty, [robot]),
        GroundAtom(Pressed, [button0]),
        GroundAtom(AboveNoButton, []),
    ]

    prompt = _build_atom_labelling_prompt(candidate_atoms, _PREDICATE_DESCRIPTIONS)

    assert "perception system" in prompt
    assert "Grasped" in prompt
    assert "HandEmpty" in prompt
    assert "Pressed" in prompt
    assert "AboveNoButton" in prompt
    assert "(Grasped robot stick)" in prompt
    assert "(HandEmpty robot)" in prompt
    assert "(Pressed button0)" in prompt
    assert "(AboveNoButton)" in prompt


def test_stickbutton2d_vlm_response_parsing():
    """Test VLM response parsing for stickbutton predicates."""
    Grasped = Predicate("Grasped", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    Pressed = Predicate("Pressed", [CircleType])
    AboveNoButton = Predicate("AboveNoButton", [])

    robot = Object("robot", CRVRobotType)
    stick = Object("stick", RectangleType)
    button0 = Object("button0", CircleType)

    candidate_atoms = [
        GroundAtom(Grasped, [robot, stick]),
        GroundAtom(HandEmpty, [robot]),
        GroundAtom(Pressed, [button0]),
        GroundAtom(AboveNoButton, []),
    ]

    # Scenario: hand empty, no button pressed, above no button.
    vlm_output = (
        "(Grasped robot stick): False.\n"
        "(HandEmpty robot): True.\n"
        "(Pressed button0): False.\n"
        "(AboveNoButton): True."
    )
    true_atoms = _parse_vlm_response(vlm_output, candidate_atoms)

    assert GroundAtom(HandEmpty, [robot]) in true_atoms
    assert GroundAtom(AboveNoButton, []) in true_atoms
    assert GroundAtom(Grasped, [robot, stick]) not in true_atoms
    assert GroundAtom(Pressed, [button0]) not in true_atoms
    assert len(true_atoms) == 2

    # Scenario: grasped, button pressed.
    vlm_output2 = (
        "(Grasped robot stick): True.\n"
        "(HandEmpty robot): False.\n"
        "(Pressed button0): True.\n"
        "(AboveNoButton): False."
    )
    true_atoms2 = _parse_vlm_response(vlm_output2, candidate_atoms)
    assert GroundAtom(Grasped, [robot, stick]) in true_atoms2
    assert GroundAtom(Pressed, [button0]) in true_atoms2
    assert len(true_atoms2) == 2


def test_stickbutton2d_goal_deriver():
    """Test goal deriver in perception planning models."""
    env = kinder.make("kinder/StickButton2D-b2-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    mock_vlm, perception_models = _make_mock_vlm_and_models(env, num_buttons=2)
    state = perception_models.observation_to_state(obs)
    goal = perception_models.goal_deriver(state)

    assert len(goal.atoms) == 2
    goal_atom_strs = {str(a) for a in goal.atoms}
    assert "(Pressed button0)" in goal_atom_strs
    assert "(Pressed button1)" in goal_atom_strs

    env.close()


def test_stickbutton2d_observation_to_state():
    """Test observation_to_state in perception planning models."""
    env = kinder.make("kinder/StickButton2D-b1-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    mock_vlm, perception_models = _make_mock_vlm_and_models(env, num_buttons=1)
    state = perception_models.observation_to_state(obs)
    assert isinstance(hash(state), int)
    assert perception_models.state_space.contains(state)

    env.close()


def test_stickbutton2d_vlm_multiple_buttons():
    """Test VLM state abstractor with multiple buttons (b2) to ensure
    per-button atoms are handled correctly."""
    env = kinder.make("kinder/StickButton2D-b2-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    gt_models = create_gt_models(
        env.observation_space, env.action_space, num_buttons=2
    )
    state = gt_models.observation_to_state(obs)
    gt_abstract = gt_models.state_abstractor(state)

    mock_vlm, perception_models = _make_mock_vlm_and_models(env, num_buttons=2)
    _prime_mock_vlm(mock_vlm, gt_models, perception_models, [state])

    vlm_abstract = perception_models.state_abstractor(state)

    assert vlm_abstract.atoms == gt_abstract.atoms
    assert vlm_abstract.objects == gt_abstract.objects

    env.close()
