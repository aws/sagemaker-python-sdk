from __future__ import absolute_import

"""Integration test to verify dependency compatibility."""

import subprocess
import sys
import tempfile
import os
import pytest
from pathlib import Path


def test_dependency_compatibility():
    """Test that all dependencies in pyproject.toml are compatible."""
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    assert pyproject_path.exists(), "pyproject.toml not found"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a fresh virtual environment
        venv_path = os.path.join(temp_dir, "test_env")
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)

        # Get pip path for the virtual environment
        if sys.platform == "win32":
            pip_path = os.path.join(venv_path, "Scripts", "pip")
        else:
            pip_path = os.path.join(venv_path, "bin", "pip")

        # Install dependencies
        result = subprocess.run(
            [pip_path, "install", "-e", str(project_root)], capture_output=True, text=True
        )

        if result.returncode != 0:
            pytest.fail(f"Dependency installation failed:\n{result.stderr}")

        # Check for conflicts
        check_result = subprocess.run([pip_path, "check"], capture_output=True, text=True)

        if check_result.returncode != 0:
            pytest.fail(f"Dependency conflicts found:\n{check_result.stdout}")


def test_numpy_pandas_compatibility():
    """Test specific NumPy-pandas compatibility."""
    try:
        import numpy as np
        import pandas as pd

        # Test basic operations
        arr = np.array([1, 2, 3])
        df = pd.DataFrame({"col": arr})

        # This should not raise the dtype size error
        result = df.values
        assert isinstance(result, np.ndarray)

    except ImportError:
        pytest.skip("NumPy or pandas not available")
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            pytest.fail(f"NumPy-pandas compatibility issue: {e}")
        raise


def test_critical_imports():
    """Test that critical packages can be imported without conflicts."""
    critical_packages = ["numpy", "pandas", "boto3", "sagemaker_core", "protobuf", "cloudpickle"]

    failed_imports = []

    for package in critical_packages:
        try:
            __import__(package)
        except ImportError:
            # Skip if package not installed
            continue
        except Exception as e:
            failed_imports.append(f"{package}: {e}")

    if failed_imports:
        pytest.fail("Import failures:\n" + "\n".join(failed_imports))
