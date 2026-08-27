#!/usr/bin/env python3
"""Install all dependencies."""

import subprocess
import sys
from pathlib import Path

from generate_topological_order import get_topological_order

# Unreleased dependencies to install before the packages. Installing these
# first means the later `uv pip install -e .[develop]` calls see the
# packages' version pins already satisfied and leave them in place. uv
# rejects URL dependencies that arrive transitively (a package's git pin on
# kindergarden would break every package that depends on that package), so
# the git pin lives here instead of in the pyprojects.
#
# kindergarden: pinned to the commit that adds the CylinderShelf3D env,
# which is not yet in a PyPI release. Remove at the next kindergarden
# release.
PREINSTALL_REQUIREMENTS = [
    "kindergarden @ git+https://github.com/Princeton-Robot-Planning-and-Learning/"
    "kindergarden.git@bf420c4404168c13c3ac3270c19918810aa0d4c7",
]


def install_package(package_path: Path) -> bool:
    """Install a single package quickly with minimal output."""
    if not package_path.exists() or not (package_path / "pyproject.toml").exists():
        return True  # Skip missing packages silently

    try:
        # Install the package in development mode
        subprocess.run(
            ["uv", "pip", "install", "-e", ".[develop]"],
            cwd=package_path,
            check=True,
            capture_output=True,
        )
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_path.name}", file=sys.stderr)
        print(f"   Command: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"   Return code: {e.returncode}", file=sys.stderr)
        if e.stdout:
            print(f"   Stdout:\n{e.stdout.decode()}", file=sys.stderr)
        if e.stderr:
            print(f"   Stderr:\n{e.stderr.decode()}", file=sys.stderr)
        return False


def main():
    """Install all packages in the correct order."""
    repo_root = Path(__file__).parents[1]
    install_order = get_topological_order(repo_root)

    for requirement in PREINSTALL_REQUIREMENTS:
        print(f"Preinstalling {requirement.split(' @ ')[0]}...", end=" ", flush=True)
        try:
            subprocess.run(
                ["uv", "pip", "install", requirement],
                cwd=repo_root,
                check=True,
                capture_output=True,
            )
            print("✅")
        except subprocess.CalledProcessError as e:
            print("❌")
            print(f"❌ Failed to preinstall {requirement}", file=sys.stderr)
            if e.stderr:
                print(f"   Stderr:\n{e.stderr.decode()}", file=sys.stderr)
            sys.exit(1)

    print(f"Installing {len(install_order)} packages...")

    for package_name in install_order:
        package_path = repo_root / package_name
        print(f"Installing {package_name}...", end=" ", flush=True)

        if install_package(package_path):
            print("✅")
        else:
            print("❌")
            sys.exit(1)

    print("🎉 All packages installed successfully!")


if __name__ == "__main__":
    main()
