"""Tests for perception-based motion2d planning models.

Uses a mock VLM (OrderedResponseModel) whose responses are generated
from the ground-truth bilevel planning state abstractor, so we can
verify the full pipeline: render → prompt → parse → abstract state.

Tests marked with ``@pytest.mark.real_vlm`` call the OpenAI API and are
skipped when OPENAI_API_KEY is not set.
"""

import logging
import os
import tempfile
from pathlib import Path

import kinder
import pytest
from kinder.envs.kinematic2d.motion2d import (
    ObjectCentricMotion2DEnv,
    RectangleType,
    TargetRegionType,
)
from kinder.envs.kinematic2d.object_types import CRVRobotType
from kinder_bilevel_planning.env_models.kinematic2d.motion2d import (
    create_bilevel_planning_models as create_gt_models,)
from prpl_llm_utils.cache import FilePretrainedLargeModelCache
from prpl_llm_utils.models import OrderedResponseModel
from prpl_llm_utils.structs import Response
from relational_structs import GroundAtom, Object, ObjectCentricState, Predicate

from kinder_perception_planning.env_models.kinematic2d.motion2d import (
    _PREDICATE_DESCRIPTIONS,
    create_perception_planning_models,
)
from kinder_perception_planning.vlm_utils import (
    _build_atom_labelling_prompt,
    _parse_vlm_response,
    create_vlm,
    query_vlm_for_atom_vals,
)

_HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY", ""))
real_vlm = pytest.mark.skipif(not _HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")

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
    """Reconstruct the same candidate atom list that the perception state_abstractor
    would build."""
    pred_name_to_pred = {p.name: p for p in env_models.predicates}
    AtTgt = pred_name_to_pred["AtTgt"]
    NotAtTgt = pred_name_to_pred["NotAtTgt"]
    AtPassage = pred_name_to_pred["AtPassage"]
    NotAtPassage = pred_name_to_pred["NotAtPassage"]
    NotAtAnyPassage = pred_name_to_pred["NotAtAnyPassage"]

    robot = state.get_objects(CRVRobotType)[0]
    target_region = state.get_objects(TargetRegionType)[0]
    obstacles = state.get_objects(RectangleType)

    candidate_atoms: list[GroundAtom] = []
    candidate_atoms.append(GroundAtom(AtTgt, [robot, target_region]))
    candidate_atoms.append(GroundAtom(NotAtTgt, [robot, target_region]))
    candidate_atoms.append(GroundAtom(NotAtAnyPassage, [robot]))
    for obs1 in obstacles:
        for obs2 in obstacles:
            if obs1 != obs2:
                candidate_atoms.append(GroundAtom(AtPassage, [robot, obs1, obs2]))
                candidate_atoms.append(GroundAtom(NotAtPassage, [robot, obs1, obs2]))
    return candidate_atoms


def _create_oracle_vlm(
    gt_models,
    perception_models,
    states: list[ObjectCentricState],
) -> OrderedResponseModel:
    """Create an OrderedResponseModel that returns ground-truth responses for each
    state."""
    responses = []
    for state in states:
        gt_abstract = gt_models.state_abstractor(state)
        candidate_atoms = _build_candidate_atoms_for_state(state, perception_models)
        response_text = _make_vlm_response_for_atoms(candidate_atoms, gt_abstract.atoms)
        responses.append(Response(response_text, metadata={}))
    cache_dir = Path(tempfile.mkdtemp())
    cache = FilePretrainedLargeModelCache(cache_dir)
    return OrderedResponseModel(responses, cache)


def test_motion2d_vlm_state_abstractor_initial():
    """Test that VLM state abstractor matches ground truth on the initial state."""
    env = kinder.make("kinder/Motion2D-p2-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    gt_models = create_gt_models(
        env.observation_space, env.action_space, num_passages=2
    )
    state = gt_models.observation_to_state(obs)
    gt_abstract = gt_models.state_abstractor(state)

    # Create a mock VLM that returns the ground-truth response.
    cache_dir = Path(tempfile.mkdtemp())
    cache = FilePretrainedLargeModelCache(cache_dir)
    # We need to pre-build the perception models first to get candidate atoms,
    # but the state_abstractor closure captures the vlm reference. We create
    # a placeholder, build the models, then swap in the real mock.
    placeholder_responses = [Response("placeholder", metadata={})]
    mock_vlm = OrderedResponseModel(placeholder_responses, cache)

    perception_models = create_perception_planning_models(
        env.observation_space, env.action_space, num_passages=2, vlm=mock_vlm
    )

    # Now build the real oracle VLM and replace it.
    oracle_vlm = _create_oracle_vlm(gt_models, perception_models, [state])
    # Replace the vlm in the closure — the state_abstractor captures `vlm`
    # via the nonlocal variable in the enclosing scope. We update the
    # OrderedResponseModel's internal response list instead.
    mock_vlm._responses = oracle_vlm._responses  # pylint: disable=protected-access
    mock_vlm._next_response_idx = 0  # pylint: disable=protected-access

    vlm_abstract = perception_models.state_abstractor(state)

    # The VLM-based abstract state should match the ground truth.
    assert vlm_abstract.atoms == gt_abstract.atoms
    assert vlm_abstract.objects == gt_abstract.objects

    env.close()


def test_motion2d_vlm_state_abstractor_at_target():
    """Test VLM state abstractor when robot is at the target region."""
    env = kinder.make("kinder/Motion2D-p2-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    gt_models = create_gt_models(
        env.observation_space, env.action_space, num_passages=2
    )
    state = gt_models.observation_to_state(obs)

    # Move robot to center of target region.
    robot = state.get_objects(CRVRobotType)[0]
    target_region = state.get_objects(TargetRegionType)[0]
    target_x = state.get(target_region, "x")
    target_y = state.get(target_region, "y")
    target_width = state.get(target_region, "width")
    target_height = state.get(target_region, "height")
    state_at_target = state.copy()
    state_at_target.set(robot, "x", target_x + target_width / 2)
    state_at_target.set(robot, "y", target_y + target_height / 2)

    gt_abstract = gt_models.state_abstractor(state_at_target)

    # Create mock VLM with oracle response.
    cache_dir = Path(tempfile.mkdtemp())
    cache = FilePretrainedLargeModelCache(cache_dir)
    placeholder = [Response("placeholder", metadata={})]
    mock_vlm = OrderedResponseModel(placeholder, cache)

    perception_models = create_perception_planning_models(
        env.observation_space, env.action_space, num_passages=2, vlm=mock_vlm
    )

    oracle_vlm = _create_oracle_vlm(gt_models, perception_models, [state_at_target])
    mock_vlm._responses = oracle_vlm._responses  # pylint: disable=protected-access
    mock_vlm._next_response_idx = 0  # pylint: disable=protected-access

    vlm_abstract = perception_models.state_abstractor(state_at_target)

    # Verify AtTgt is true.
    pred_name_to_pred = {p.name: p for p in perception_models.predicates}
    AtTgt = pred_name_to_pred["AtTgt"]
    NotAtTgt = pred_name_to_pred["NotAtTgt"]

    assert GroundAtom(AtTgt, [robot, target_region]) in vlm_abstract.atoms
    assert GroundAtom(NotAtTgt, [robot, target_region]) not in vlm_abstract.atoms
    assert vlm_abstract.atoms == gt_abstract.atoms

    env.close()


def test_motion2d_vlm_state_abstractor_at_passage():
    """Test VLM state abstractor when robot is at a passage."""
    env = kinder.make("kinder/Motion2D-p2-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    gt_models = create_gt_models(
        env.observation_space, env.action_space, num_passages=2
    )
    state = gt_models.observation_to_state(obs)

    # Move robot into a passage between obstacle0 and obstacle1.
    robot = state.get_objects(CRVRobotType)[0]
    obstacles = state.get_objects(RectangleType)
    obj_name_to_obj = {o.name: o for o in obstacles}
    obstacle0 = obj_name_to_obj["obstacle0"]
    obstacle1 = obj_name_to_obj["obstacle1"]

    state_at_passage = state.copy()
    obs0_x = state.get(obstacle0, "x")
    obs0_width = state.get(obstacle0, "width")
    obs0_y = state.get(obstacle0, "y")
    obs1_y = state.get(obstacle1, "y")
    obs0_height = state.get(obstacle0, "height")
    # Position robot in the gap.
    state_at_passage.set(robot, "x", obs0_x + obs0_width / 2)
    state_at_passage.set(robot, "y", (obs0_y + obs0_height + obs1_y) / 2)

    gt_abstract = gt_models.state_abstractor(state_at_passage)

    # Create mock VLM with oracle response.
    cache_dir = Path(tempfile.mkdtemp())
    cache = FilePretrainedLargeModelCache(cache_dir)
    placeholder = [Response("placeholder", metadata={})]
    mock_vlm = OrderedResponseModel(placeholder, cache)

    perception_models = create_perception_planning_models(
        env.observation_space, env.action_space, num_passages=2, vlm=mock_vlm
    )

    oracle_vlm = _create_oracle_vlm(gt_models, perception_models, [state_at_passage])
    mock_vlm._responses = oracle_vlm._responses  # pylint: disable=protected-access
    mock_vlm._next_response_idx = 0  # pylint: disable=protected-access

    vlm_abstract = perception_models.state_abstractor(state_at_passage)

    # Verify passage-related atoms match ground truth.
    assert vlm_abstract.atoms == gt_abstract.atoms

    env.close()


def test_motion2d_vlm_prompt_construction():
    """Test that the VLM prompt is constructed correctly."""
    AtTgt = Predicate("AtTgt", [CRVRobotType, TargetRegionType])
    NotAtTgt = Predicate("NotAtTgt", [CRVRobotType, TargetRegionType])

    robot = Object("robot", CRVRobotType)
    target_region = Object("target_region", TargetRegionType)

    candidate_atoms = [
        GroundAtom(AtTgt, [robot, target_region]),
        GroundAtom(NotAtTgt, [robot, target_region]),
    ]

    prompt = _build_atom_labelling_prompt(candidate_atoms, _PREDICATE_DESCRIPTIONS)

    # Verify prompt structure.
    assert "perception system" in prompt
    assert "AtTgt" in prompt
    assert "NotAtTgt" in prompt
    assert "True." in prompt
    assert "False." in prompt
    assert "(AtTgt robot target_region)" in prompt
    assert "(NotAtTgt robot target_region)" in prompt


def test_motion2d_vlm_response_parsing():
    """Test that VLM response parsing works correctly."""
    AtTgt = Predicate("AtTgt", [CRVRobotType, TargetRegionType])
    NotAtTgt = Predicate("NotAtTgt", [CRVRobotType, TargetRegionType])
    NotAtAnyPassage = Predicate("NotAtAnyPassage", [CRVRobotType])

    robot = Object("robot", CRVRobotType)
    target_region = Object("target_region", TargetRegionType)

    candidate_atoms = [
        GroundAtom(AtTgt, [robot, target_region]),
        GroundAtom(NotAtTgt, [robot, target_region]),
        GroundAtom(NotAtAnyPassage, [robot]),
    ]

    # Test parsing a response where AtTgt is false, NotAtTgt and
    # NotAtAnyPassage are true.
    vlm_output = (
        "(AtTgt robot target_region): False.\n"
        "(NotAtTgt robot target_region): True.\n"
        "(NotAtAnyPassage robot): True."
    )
    true_atoms = _parse_vlm_response(vlm_output, candidate_atoms)

    assert GroundAtom(AtTgt, [robot, target_region]) not in true_atoms
    assert GroundAtom(NotAtTgt, [robot, target_region]) in true_atoms
    assert GroundAtom(NotAtAnyPassage, [robot]) in true_atoms
    assert len(true_atoms) == 2

    # Test with AtTgt true.
    vlm_output2 = (
        "(AtTgt robot target_region): True.\n"
        "(NotAtTgt robot target_region): False.\n"
        "(NotAtAnyPassage robot): True."
    )
    true_atoms2 = _parse_vlm_response(vlm_output2, candidate_atoms)
    assert GroundAtom(AtTgt, [robot, target_region]) in true_atoms2
    assert GroundAtom(NotAtTgt, [robot, target_region]) not in true_atoms2
    assert len(true_atoms2) == 2


def test_motion2d_goal_deriver():
    """Test goal deriver in perception planning models."""
    env = kinder.make("kinder/Motion2D-p2-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    cache_dir = Path(tempfile.mkdtemp())
    cache = FilePretrainedLargeModelCache(cache_dir)
    mock_vlm = OrderedResponseModel([Response("", metadata={})], cache)

    perception_models = create_perception_planning_models(
        env.observation_space, env.action_space, num_passages=2, vlm=mock_vlm
    )
    state = perception_models.observation_to_state(obs)
    goal = perception_models.goal_deriver(state)
    assert len(goal.atoms) == 1
    goal_atom = next(iter(goal.atoms))
    assert str(goal_atom) == "(AtTgt robot target_region)"

    env.close()


def test_motion2d_observation_to_state():
    """Test observation_to_state in perception planning models."""
    env = kinder.make("kinder/Motion2D-p2-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    cache_dir = Path(tempfile.mkdtemp())
    cache = FilePretrainedLargeModelCache(cache_dir)
    mock_vlm = OrderedResponseModel([Response("", metadata={})], cache)

    perception_models = create_perception_planning_models(
        env.observation_space, env.action_space, num_passages=2, vlm=mock_vlm
    )
    state = perception_models.observation_to_state(obs)
    assert isinstance(hash(state), int)
    assert perception_models.state_space.contains(state)

    env.close()


# ---------------------------------------------------------------------------
# Real VLM tests (require OPENAI_API_KEY)
# ---------------------------------------------------------------------------


def _setup_real_vlm_test(env_id, num_passages, vlm_model="gpt-4o"):
    """Shared setup: create env, ground-truth models, real VLM, and
    perception models."""
    env = kinder.make(env_id, render_mode="rgb_array")
    gt_models = create_gt_models(
        env.observation_space, env.action_space, num_passages=num_passages
    )
    vlm = create_vlm(vlm_model, cache_dir=Path(tempfile.mkdtemp()))
    perception_models = create_perception_planning_models(
        env.observation_space, env.action_space, num_passages=num_passages, vlm=vlm
    )
    return env, gt_models, perception_models


@real_vlm
def test_motion2d_real_vlm_initial_state():
    """Query a real VLM on the initial state and compare against ground truth.

    The initial state is unambiguous: the robot is far from both the
    target and any passage, so the VLM should get this right.
    """
    env, gt_models, perception_models = _setup_real_vlm_test(
        "kinder/Motion2D-p2-v0", num_passages=2
    )
    obs, _ = env.reset(seed=123)
    state = gt_models.observation_to_state(obs)

    gt_abstract = gt_models.state_abstractor(state)
    vlm_abstract = perception_models.state_abstractor(state)

    logging.info("Ground-truth atoms: %s", gt_abstract.atoms)
    logging.info("VLM atoms:          %s", vlm_abstract.atoms)

    # The key easy predicates the VLM must get right.
    pred_names = {p.name: p for p in perception_models.predicates}
    robot = state.get_objects(CRVRobotType)[0]
    target_region = state.get_objects(TargetRegionType)[0]

    NotAtTgt = pred_names["NotAtTgt"]
    AtTgt = pred_names["AtTgt"]

    # Robot is clearly not at the target in the initial state.
    assert GroundAtom(NotAtTgt, [robot, target_region]) in vlm_abstract.atoms
    assert GroundAtom(AtTgt, [robot, target_region]) not in vlm_abstract.atoms

    env.close()


@real_vlm
def test_motion2d_real_vlm_at_target():
    """Query a real VLM when the robot is positioned inside the target region."""
    env, gt_models, perception_models = _setup_real_vlm_test(
        "kinder/Motion2D-p2-v0", num_passages=2
    )
    obs, _ = env.reset(seed=123)
    state = gt_models.observation_to_state(obs)

    robot = state.get_objects(CRVRobotType)[0]
    target_region = state.get_objects(TargetRegionType)[0]

    # Move robot to the centre of the target region.
    state_at_target = state.copy()
    state_at_target.set(
        robot,
        "x",
        state.get(target_region, "x") + state.get(target_region, "width") / 2,
    )
    state_at_target.set(
        robot,
        "y",
        state.get(target_region, "y") + state.get(target_region, "height") / 2,
    )

    gt_abstract = gt_models.state_abstractor(state_at_target)
    vlm_abstract = perception_models.state_abstractor(state_at_target)

    logging.info("Ground-truth atoms: %s", gt_abstract.atoms)
    logging.info("VLM atoms:          %s", vlm_abstract.atoms)

    pred_names = {p.name: p for p in perception_models.predicates}
    AtTgt = pred_names["AtTgt"]
    NotAtTgt = pred_names["NotAtTgt"]

    assert GroundAtom(AtTgt, [robot, target_region]) in vlm_abstract.atoms
    assert GroundAtom(NotAtTgt, [robot, target_region]) not in vlm_abstract.atoms

    env.close()


@real_vlm
def test_motion2d_real_vlm_query_returns_parseable_response():
    """Verify that the raw VLM response is parseable (every candidate atom gets a
    True/False label) without asserting correctness of each atom."""
    env = kinder.make("kinder/Motion2D-p2-v0", render_mode="rgb_array")
    obs, _ = env.reset(seed=123)

    gt_models = create_gt_models(
        env.observation_space, env.action_space, num_passages=2
    )
    state = gt_models.observation_to_state(obs)

    robot = state.get_objects(CRVRobotType)[0]
    target_region = state.get_objects(TargetRegionType)[0]
    obstacles = state.get_objects(RectangleType)

    pred_names = {p.name: p for p in gt_models.predicates}
    AtTgt = pred_names["AtTgt"]
    NotAtTgt = pred_names["NotAtTgt"]
    NotAtAnyPassage = pred_names["NotAtAnyPassage"]
    AtPassage = pred_names["AtPassage"]
    NotAtPassage = pred_names["NotAtPassage"]

    candidate_atoms: list[GroundAtom] = [
        GroundAtom(AtTgt, [robot, target_region]),
        GroundAtom(NotAtTgt, [robot, target_region]),
        GroundAtom(NotAtAnyPassage, [robot]),
    ]
    for obs1 in obstacles:
        for obs2 in obstacles:
            if obs1 != obs2:
                candidate_atoms.append(GroundAtom(AtPassage, [robot, obs1, obs2]))
                candidate_atoms.append(GroundAtom(NotAtPassage, [robot, obs1, obs2]))

    # Render the scene.
    sim = ObjectCentricMotion2DEnv(num_passages=2)
    sim.reset(options={"init_state": state.copy()})
    rendered = sim.render()
    assert rendered is not None

    vlm = create_vlm("gpt-4o", cache_dir=Path(tempfile.mkdtemp()))
    true_atoms = query_vlm_for_atom_vals(
        vlm, rendered, candidate_atoms, _PREDICATE_DESCRIPTIONS
    )

    logging.info(
        "VLM returned %d true atoms out of %d candidates",
        len(true_atoms),
        len(candidate_atoms),
    )
    for atom in true_atoms:
        logging.info("  TRUE: %s", atom)

    # Basic sanity: we should get *some* atoms back and they should all
    # be a subset of the candidates.
    assert len(true_atoms) > 0
    assert true_atoms.issubset(set(candidate_atoms))

    # AtTgt and NotAtTgt are mutually exclusive — at most one should be true.
    at_tgt = GroundAtom(AtTgt, [robot, target_region])
    not_at_tgt = GroundAtom(NotAtTgt, [robot, target_region])
    assert not (
        at_tgt in true_atoms and not_at_tgt in true_atoms
    ), "AtTgt and NotAtTgt should be mutually exclusive"

    env.close()
