# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Requirements Manager class to pull in client dependencies from a .txt or .yml file"""
from __future__ import absolute_import
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


class RequirementsManager:
    """Manages dependency installation by detecting file types"""

    def detect_file_exists(self, dependencies: str = None) -> str:
        """Detects the type of file dependencies will be installed from

        If a req.txt or conda.yml file is provided, it verifies their existence and
        returns the local file path

        Args:
            dependencies (str): Local path where dependencies file exists.

        Returns:
            file path of the existing or generated dependencies file
        """
        dependencies = self._capture_from_local_runtime()

        # Dependencies specified as either req.txt or conda_env.yml
        if dependencies.endswith(".txt"):
            self._install_requirements_txt()
        elif dependencies.endswith(".yml"):
            self._update_conda_env_in_path()
        else:
            raise ValueError(f'Invalid dependencies provided: "{dependencies}"')

    def _install_requirements_txt(self):
        """Install requirements.txt file using pip"""
        logger.info("Running command to pip install")
        subprocess.run("pip install -r requirements.txt", shell=True, check=True)
        logger.info("Command ran successfully")

    def _update_conda_env_in_path(self):
        """Update conda env using conda yml file"""
        logger.info("Updating conda env")
        subprocess.run("conda env update -f conda_in_process.yml", shell=True, check=True)
        logger.info("Conda env updated successfully")

    def _get_active_conda_env_name(self) -> str:
        """Returns the conda environment name from the set environment variable. None otherwise."""
        return os.getenv("CONDA_DEFAULT_ENV")

    def _get_active_conda_env_prefix(self) -> str:
        """Returns the conda prefix from the set environment variable. None otherwise."""
        return os.getenv("CONDA_PREFIX")

    def _capture_from_local_runtime(self) -> str:
        """Generates dependencies list from the user's local runtime.

        Raises RuntimeEnvironmentError if not able to.

        Currently supports: conda environments
        """

        # Try to capture dependencies from the conda environment, if any.
        conda_env_name = self._get_active_conda_env_name()
        logger.info("Found conda_env_name: '%s'", conda_env_name)
        conda_env_prefix = None

        if conda_env_name is None:
            conda_env_prefix = self._get_active_conda_env_prefix()

        if conda_env_name is None and conda_env_prefix is None:
            raise ValueError("No conda environment seems to be active.")

        if conda_env_name == "base":
            logger.warning(
                "We recommend using an environment other than base to "
                "isolate your project dependencies from conda dependencies"
            )

        local_dependencies_path = os.path.join(os.getcwd(), "inf_env_snapshot.yml")

        return local_dependencies_path
