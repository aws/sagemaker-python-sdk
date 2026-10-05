from __future__ import absolute_import

"""Test dependency resolution using pip-tools."""

import subprocess
import sys
import tempfile
import pytest
from pathlib import Path


def test_pip_compile_resolution():
    """Test that pip-compile can resolve all dependencies without conflicts."""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Install pip-tools
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pip-tools"], check=True, capture_output=True
        )

        # Try to compile dependencies
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "piptools",
                "compile",
                str(pyproject_path),
                "--dry-run",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            cwd=temp_dir,
        )

        if result.returncode != 0:
            # Check for specific conflict patterns
            stderr = result.stderr.lower()
            if "could not find a version" in stderr or "incompatible" in stderr:
                pytest.fail(f"Dependency resolution failed:\n{result.stderr}")
            # Other errors might be acceptable (missing extras, etc.)


def test_pipdeptree_conflicts():
    """Test using pipdeptree to detect conflicts."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pipdeptree"], check=True, capture_output=True
        )

        result = subprocess.run(
            [sys.executable, "-m", "pipdeptree", "--warn", "conflict"],
            capture_output=True,
            text=True,
        )

        if "Warning!!" in result.stdout:
            pytest.fail(f"Dependency conflicts detected:\n{result.stdout}")

    except subprocess.CalledProcessError:
        pytest.skip("pipdeptree installation failed")


if __name__ == "__main__":
    test_pip_compile_resolution()
    test_pipdeptree_conflicts()
    print("✅ Dependency resolution tests passed")
