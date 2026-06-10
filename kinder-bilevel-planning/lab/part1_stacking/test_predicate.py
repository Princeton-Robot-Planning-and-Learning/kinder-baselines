"""Unit test for the new ``On`` predicate, in isolation."""

import kinder
from part1_stacking.models import create_stacking_models
from relational_structs import GroundAtom


def _setup():
    kinder.register_all_environments()
    env = kinder.make("kinder/Obstruction2D-o1-v0")
    env_models = create_stacking_models(
        env.observation_space, env.action_space, num_obstructions=1
    )
    obs, _ = env.reset(seed=0)
    state = env_models.observation_to_state(obs).copy()
    return env_models, state


def test_on_predicate_fires_when_stacked():
    """``On(obstruction0, target_block)`` holds exactly when stacked, not beside."""
    env_models, state = _setup()
    pred = {p.name: p for p in env_models.predicates}
    On, OnTable = pred["On"], pred["OnTable"]
    target = state.get_object_from_name("target_block")
    obstruction = state.get_object_from_name("obstruction0")

    # Side by side on the table: no On atom, both on the table.
    state.set(target, "x", 0.5)
    state.set(target, "width", 0.16)
    state.set(target, "height", 0.09)
    state.set(obstruction, "x", 0.2)
    state.set(obstruction, "width", 0.1)
    state.set(obstruction, "height", 0.09)
    atoms = env_models.state_abstractor(state).atoms
    assert GroundAtom(On, [obstruction, target]) not in atoms
    assert GroundAtom(OnTable, [obstruction]) in atoms
    assert GroundAtom(OnTable, [target]) in atoms

    # Obstruction resting on top of the target block: On fires, not OnTable.
    target_top = state.get(target, "y") + state.get(target, "height")
    state.set(obstruction, "x", 0.53)  # within the target's x-span [0.5, 0.66]
    state.set(obstruction, "y", target_top)
    atoms = env_models.state_abstractor(state).atoms
    assert GroundAtom(On, [obstruction, target]) in atoms
    assert GroundAtom(OnTable, [obstruction]) not in atoms
    assert GroundAtom(OnTable, [target]) in atoms  # target still on the table
