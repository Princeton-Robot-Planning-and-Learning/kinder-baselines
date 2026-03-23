"""VLM utilities for predicate grounding.

Renders environment states as images and queries a vision-language model
to determine which ground atoms (predicates) are true.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import PIL.Image
from numpy.typing import NDArray
from prpl_llm_utils.cache import FilePretrainedLargeModelCache
from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.models import OpenAIModel  # noqa: F401
from relational_structs import GroundAtom


def create_vlm(
    model_name: str,
    cache_dir: Optional[Path] = None,
) -> OpenAIModel:
    """Create a VLM instance with file-based caching."""
    if cache_dir is None:
        cache_dir = Path("./vlm_cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache = FilePretrainedLargeModelCache(cache_dir)
    return OpenAIModel(model_name, cache)


def query_vlm_for_atom_vals(
    vlm: PretrainedLargeModel,
    rendered_image: NDArray[np.uint8],
    candidate_atoms: list[GroundAtom],
    predicate_descriptions: dict[str, str],
) -> set[GroundAtom]:
    """Query a VLM to determine which ground atoms are true in the scene.

    Args:
        vlm: The vision-language model to query.
        rendered_image: RGB image of the current scene (H, W, 3), uint8.
        candidate_atoms: All possible ground atoms to evaluate.
        predicate_descriptions: Maps predicate name to a natural-language
            description of what it means for it to be true.

    Returns:
        The subset of candidate_atoms that the VLM judges to be true.
    """
    if len(candidate_atoms) == 0:
        return set()

    prompt = _build_atom_labelling_prompt(candidate_atoms, predicate_descriptions)
    logging.debug("VLM prompt:\n%s", prompt)

    # Convert numpy image to PIL for prpl_llm_utils API.
    pil_image = PIL.Image.fromarray(rendered_image)

    response = vlm.query(
        prompt,
        imgs=[pil_image],
        hyperparameters={"temperature": 0.0},
        seed=0,
    )
    vlm_output = response.text
    logging.debug("VLM response:\n%s", vlm_output)

    true_atoms = _parse_vlm_response(vlm_output, candidate_atoms)
    return true_atoms


def _build_atom_labelling_prompt(
    candidate_atoms: list[GroundAtom],
    predicate_descriptions: dict[str, str],
) -> str:
    """Build a prompt asking the VLM to label each atom as True or False."""
    lines = [
        "You are a perception system for a robot. You will be shown an image "
        "of a 2D scene. Your task is to determine the truth value of each "
        "predicate listed below.",
        "",
        "Predicate descriptions:",
    ]
    for pred_name, desc in predicate_descriptions.items():
        lines.append(f"  - {pred_name}: {desc}")

    lines.append("")
    lines.append(
        "For each predicate instance below, respond with EXACTLY one line in "
        "the format:"
    )
    lines.append("  <predicate_instance>: True.")
    lines.append("  or")
    lines.append("  <predicate_instance>: False.")
    lines.append("")
    lines.append("Predicates to evaluate:")

    for atom in candidate_atoms:
        lines.append(f"  {atom}")

    return "\n".join(lines)


def _parse_vlm_response(
    vlm_output: str,
    candidate_atoms: list[GroundAtom],
) -> set[GroundAtom]:
    """Parse VLM response to extract which atoms are true.

    Expects each line to be formatted as:
        <atom_str>: True.
    or
        <atom_str>: False.
    """
    true_atoms: set[GroundAtom] = set()
    atom_str_to_atom = {str(atom): atom for atom in candidate_atoms}

    response_lines = vlm_output.strip().split("\n")

    for line in response_lines:
        line = line.strip()
        if not line:
            continue

        # Find the last colon to split atom name from truth value.
        colon_idx = line.rfind(":")
        if colon_idx == -1:
            logging.warning("Skipping malformed VLM output line: %s", line)
            continue

        atom_str = line[:colon_idx].strip()
        value_str = line[colon_idx + 1 :].strip().rstrip(".").lower()

        if atom_str in atom_str_to_atom and value_str == "true":
            true_atoms.add(atom_str_to_atom[atom_str])
        elif atom_str not in atom_str_to_atom:
            logging.warning(
                "VLM returned unknown atom '%s'. Known: %s",
                atom_str,
                list(atom_str_to_atom.keys()),
            )

    return true_atoms
