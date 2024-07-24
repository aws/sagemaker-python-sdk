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
    """Transformers build logic with ModelBuilder()"""

    def detect_file_exists(self, dependencies: str = None) -> str:
        """Creates snapshot of the user's environment
 
        If a req.txt or conda.yml file is provided, it verifies their existence and
        returns the local file path
 
        Args:
            dependencies (str): Local path where dependencies file exists.
 
        Returns:
            file path of the existing or generated dependencies file
        """
 
        # No additional dependencies specified
        if dependencies is None:
            return None
 
        # Dependencies specified as either req.txt or conda_env.yml
        if dependencies.endswith(".txt"):
            self._install_requirements_txt
        elif dependencies.endswith(".yml"):
            self._update_conda_env_in_path
        else:
            raise ValueError(f'Invalid dependencies provided: "{dependencies}"')
        
    def _install_requirements_txt(self):
        """Install requirements.txt file using pip"""
        logger.info("Running command to pip install")
        subprocess.run("pip install -r /home/ec2-user/SageMaker/require.txt")
        logger.info("Command ran successfully")
 
    def _update_conda_env_in_path(self):
        """Update conda env using conda yml file"""
        logger.info("Updating conda env")
        subprocess.run("conda env update -p '/home/ec2-user/anaconda3/envs/pytorch_p310' --file=conda_in_process.yml")
        logger.info("Conda env updated successfully")

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
            logger.info("Found conda_env_prefix: '%s'", conda_env_prefix)
            if conda_env_prefix is None and self._get_studio_image_uri() is not None:
                logger.info(
                    "Neither conda env name or prefix is set. Running Studio fallback logic"
                )
                # Fallback for Studio Notebooks since conda env is not activated to use as a
                # Jupyter kernel from images.
                # TODO: Remove after fixing the behavior for Studio Notebooks.
                which_python = self._get_which_python()
                prefix_candidate = which_python.replace("/bin/python", "")
                conda_env_list = self._get_conda_envs_list()
                if (
                    conda_env_list.find(prefix_candidate + "\n") > 0
                ):  # need "\n" to match exact prefix; -1 for not found.
                    conda_env_prefix = prefix_candidate
 
        if conda_env_name is None and conda_env_prefix is None:
            raise ValueError("No conda environment seems to be active.")
 
        if conda_env_name == "base":
            logger.warning(
                "We recommend using an environment other than base to "
                "isolate your project dependencies from conda dependencies"
            )
 
        local_dependencies_path = os.path.join(os.getcwd(), "inf_env_snapshot.yml")
        with tempfile.NamedTemporaryFile(suffix=".yml", prefix="tmp_export") as tmp_file:
            if conda_env_name is not None:
                self._export_conda_env_from_env_name(conda_env_name, tmp_file.name)
            else:
                self._export_conda_env_from_prefix(conda_env_prefix, tmp_file.name)
            data = self._replace_sagemaker_in_conda_env_yml(tmp_file.read())
            with open(local_dependencies_path, "wb") as file:
                file.write(data)
 
        return local_dependencies_path
