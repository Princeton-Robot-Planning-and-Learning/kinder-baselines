"""Loader for environment-specific in-context examples."""

from pathlib import Path
from typing import Any, Optional


def get_in_context_examples(
    env_class_name: str, env_name: str, env: Optional[Any] = None
) -> str:
    """Load in-context examples for a given environment.

    Args:
        env_class_name: Category of the environment (e.g., "kinematic2d", "dynamic3d")
        env_name: Specific environment name (e.g., "motion2d", "tidybot3d")
        env: Optional environment object (needed for tidybot3d to determine task type)

    Returns:
        Markdown string containing in-context examples, or empty string if not found.
    """
    # Handle special cases for tidybot3d variants
    if env_class_name == "dynamic3d" and env is not None:
        # Extract task name from env.spec.id
        # Example: env.spec.id = "kinder/SweepIntoDrawer3D-o5-v0"
        # We want to extract "SweepIntoDrawer3D"
        if hasattr(env, "spec") and env.spec is not None and hasattr(env.spec, "id"):
            env_id = env.spec.id
            # Remove "kinder/" prefix if present
            if "/" in env_id:
                env_id = env_id.split("/")[-1]
            
            # Extract task name (part before variant and version)
            # e.g., "SweepIntoDrawer3D-o5-v0" -> "SweepIntoDrawer3D"
            task_name = env_id.split("-")[0]

            # Map task type to markdown file
            if "Shelf3D" in task_name:
                env_file_name = "shelf3d"
            elif "SweepIntoDrawer3D" in task_name:
                env_file_name = "sweepintodrawer3d"
            else:
                raise NotImplementedError(
                    f"Unknown task type '{task_name}' for dynamic3d environment. "
                    f"Expected 'Shelf3D' or 'SweepIntoDrawer3D'."
                )
        else:
            raise NotImplementedError(
                f"Cannot determine task type for dynamic3d environment. "
                f"Environment must have spec.id attribute."
            )
    else:
        env_file_name = env_name

    # Construct path to the markdown file
    current_dir = Path(__file__).parent
    examples_path = (
        current_dir / "in_context_examples" / env_class_name / f"{env_file_name}.md"
    )

    # Try to load the file
    if examples_path.exists():
        with open(examples_path, "r", encoding="utf-8") as f:
            return f.read()

    # If not found, return empty string
    return ""


def get_in_context_examples_by_env_id(env_id: str) -> str:
    """Load in-context examples based on environment ID.

    Args:
        env_id: Full environment ID (e.g., "kinder/Motion2D-p0-v0")

    Returns:
        Markdown string containing in-context examples, or empty string if not found.
    """
    # Parse environment ID to extract env_name
    # Format: "kinder/EnvName-variant-v0" or just "EnvName-variant-v0"
    if "/" in env_id:
        env_id = env_id.split("/")[-1]  # Remove "kinder/" prefix if present

    # Extract base environment name (before first hyphen)
    if "-" in env_id:
        env_name = env_id.split("-")[0].lower()
    else:
        env_name = env_id.lower()

    # Map environment names to categories
    # This mapping is based on the directory structure
    kinematic2d_envs = {
        "motion2d",
        "stickbutton2d",
        "clutteredretrieval2d",
        "clutteredstorage2d",
        "obstruction2d",
        "pushpullhook2d",
    }
    kinematic3d_envs = {
        "basemotion3d",
        "transport3d",
        "ground3d",
        "obstruction3d",
        "packing3d",
        "shelf3d",
        "table3d",
        "motion3d",
    }
    dynamic2d_envs = {
        "dynpushpullhook2d",
        "dynobstruction2d",
        "dynpusht2d",
        "dynscooppour2d",
    }
    dynamic3d_envs = {"tidybot3d", "shelf3d_real"}

    # Determine category
    env_class_name: Optional[str] = None
    if env_name in kinematic2d_envs:
        env_class_name = "kinematic2d"
    elif env_name in kinematic3d_envs:
        env_class_name = "kinematic3d"
    elif env_name in dynamic2d_envs:
        env_class_name = "dynamic2d"
    elif env_name in dynamic3d_envs:
        env_class_name = "dynamic3d"

    if env_class_name is None:
        return ""

    return get_in_context_examples(env_class_name, env_name)
