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
"""Scrapper utilities to support repacking of models."""
from __future__ import absolute_import

import logging
import os
import shutil
import tarfile
import tempfile
from typing import List, Union, Optional, TYPE_CHECKING
from sagemaker.core import image_uris
from sagemaker.core.training.configs import InputData
# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    pass
from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.mlops.workflow.steps import (
    TrainingStep,
    Step,
)
from sagemaker.core.utils import (
    _save_model,
    download_file_from_url,
    custom_extractall_tarfile,
)
from sagemaker.mlops.workflow.retry import RetryPolicy

if TYPE_CHECKING:
    from sagemaker.mlops.workflow.step_collections import StepCollection

logger = logging.getLogger(__name__)

FRAMEWORK_VERSION = "1.2-1"
INSTANCE_TYPE = "ml.m5.large"
REPACK_SCRIPT = "_repack_model.py"
REPACK_SCRIPT_LAUNCHER = "_repack_script_launcher.sh"
LAUNCH_REPACK_SCRIPT_CMD = """
#!/bin/bash

var_inference_script="${SM_HP_INFERENCE_SCRIPT}"
var_model_archive="${SM_HP_MODEL_ARCHIVE}"
var_source_dir="${SM_HP_SOURCE_DIR}"

python _repack_model.py \
--inference_script "${var_inference_script}" \
--model_archive "${var_model_archive}" \
--source_dir "${var_source_dir}"
"""


class _RepackModelStep(TrainingStep):
    """Repacks model artifacts with custom inference entry points.

    The SDK automatically adds this step to pipelines that have RegisterModelSteps with models
    that have a custom entry point.
    """

    def __init__(
        self,
        name: str,
        sagemaker_session,
        role,
        model_data: str,
        entry_point: str,
        display_name: str = None,
        description: str = None,
        source_dir: str = None,
        requirements: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, "StepCollection"]]] = None,
        retry_policies: Optional[List[RetryPolicy]] = None,
        subnets=None,
        security_group_ids=None,
        **kwargs,
    ):
        """Base class initializer.

        Args:
            name (str): The name of the training step.
            sagemaker_session (sagemaker.session.Session): Session object which manages
                    interactions with Amazon SageMaker APIs and any other AWS services needed. If
                    not specified, the model trainer creates one using the default
                    AWS configuration chain.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                    SageMaker training jobs and APIs that create Amazon SageMaker
                    endpoints use this role to access training data and model
                    artifacts. After the endpoint is created, the inference code
                    might use the IAM role, if it needs to access an AWS resource.
            model_data (str): The S3 location of a SageMaker model data `.tar.gz` file.
            entry_point (str): Path (absolute or relative) to the local Python
                    source file which should be executed as the entry point to
                    inference. If ``source_dir`` is specified, then ``entry_point``
                    must point to a file located at the root of ``source_dir``.
                    If 'git_config' is provided, 'entry_point' should be
                    a relative location to the Python source file in the Git repo.

                    Example:
                        With the following GitHub repo directory structure:

                        >>> |----- README.md
                        >>> |----- src
                        >>>         |----- train.py
                        >>>         |----- test.py

                        You can assign entry_point='src/train.py'.
            display_name (str): The display name of this `_RepackModelStep` step (default: None).
            description (str): The description of this `_RepackModelStep` (default: None).
            source_dir (str): A relative location to a directory with other training
                or model hosting source code aside from the entry point
                file (default: None). Structure within this directory is
                preserved when repacking on Amazon SageMaker.
            requirements (str): Path to a requirements.txt file containing Python
                    dependencies to install in the container (default: None).
                    The file will be processed by ModelTrainer during repacking.

                    .. admonition:: Example

                        >>> _RepackModelStep(requirements='requirements.txt')
            depends_on (List[Union[str, Step, StepCollection]]): The list of `Step`/`StepCollection`
                names or `Step` instances or `StepCollection` instances that the current `Step`
                depends on (default: None).
            retry_policies (List[RetryPolicy]): The list of retry policies for the current step
                (default: None).
            subnets (list[str]): List of subnet ids. If not specified, the re-packing
                    job will be created without VPC config (default: None).
            security_group_ids (list[str]): List of security group ids. If not
                specified, the re-packing job will be created without VPC config (default: None).
            **kwargs: additional arguments for the repacking job.
        """
        self._model_data = model_data
        self.sagemaker_session = sagemaker_session
        self.role = role
        self._entry_point = entry_point
        self._entry_point_basename = os.path.basename(self._entry_point)
        self._source_dir = source_dir
        self._requirements = requirements

        # Prepare source directory with repack scripts
        self._prepare_for_repacking()
        
        # Handle requirements.txt like ModelTrainer
        requirements_file = self._requirements if self._requirements and self._requirements.endswith('.txt') else None

        # Configure ModelTrainer components for repacking
        from sagemaker.core.training.configs import SourceCode, Compute, Networking
        
        source_code = SourceCode(
            source_dir=self._source_dir,
            entry_script=REPACK_SCRIPT_LAUNCHER,
            requirements=requirements_file,
        )
        
        compute = Compute(
            instance_type=kwargs.pop("instance_type", None) or INSTANCE_TYPE,
        )
        
        networking = None
        if subnets or security_group_ids:
            networking = Networking(
                subnets=subnets,
                security_group_ids=security_group_ids,
            )
        
        # Get region-appropriate sklearn inference image
        training_image = image_uris.retrieve(
            framework="sklearn",
            region=self.sagemaker_session.boto_region_name,
            version=FRAMEWORK_VERSION,
            image_scope="inference",
            instance_type=compute.instance_type
        )
        
        # Lazy import to avoid circular dependency
        from sagemaker.train import ModelTrainer
        
        repacker = ModelTrainer(
            training_image=training_image,
            source_code=source_code,
            compute=compute,
            networking=networking,
            sagemaker_session=self.sagemaker_session,
            role=self.role,
            hyperparameters={
                "inference_script": self._entry_point_basename,
                "model_archive": self._model_data,
                "source_dir": self._source_dir,
                # Requirements are handled by SourceCode.requirements, not hyperparameters
            },
            **kwargs,
        )
        
        inputs = [InputData(channel_name="training", data_source=self._model_data)]

        # Initialize the parent TrainingStep with the ModelTrainer configuration
        super(_RepackModelStep, self).__init__(
            name=name,
            display_name=display_name,
            description=description,
            depends_on=depends_on,
            retry_policies=retry_policies,
            step_args=repacker.train(input_data_config=inputs, wait=False, logs=False),
        )

    def _prepare_for_repacking(self):
        """Prepares the source for the model trainer."""
        if self._source_dir is None:
            self._establish_source_dir()

        self._inject_repack_script_and_launcher()

    def _establish_source_dir(self):
        """If the source_dir is None, creates it for the repacking job.

        It performs the following:
            1) creates a source directory
            2) copies the inference_entry_point inside it
            3) copies the repack_model.py inside it
            4) sets the source dir for the repacking model trainer
        """
        self._source_dir = tempfile.mkdtemp()
        # Note: source_dir will be set when creating the ModelTrainer instance

        shutil.copy2(self._entry_point, os.path.join(self._source_dir, self._entry_point_basename))
        self._entry_point = self._entry_point_basename

    def _inject_repack_script_and_launcher(self):
        """Injects the _repack_model.py script and _repack_script_launcher.sh

        into S3 or local source directory.

        Note: The bash file is needed because if not supplied, the SKLearn
        training job will auto install all dependencies listed in requirements.txt.
        However, this auto install behavior is not expected in _RepackModelStep,
        since it should just simply repack the model along with other supplied files,
        e.g. the requirements.txt.

        If the source_dir is an S3 path:
            1) downloads the source_dir tar.gz
            2) extracts it
            3) copies the _repack_model.py script into the extracted directory
            4) creates the _repack_script_launcher.sh in the extracted dir
            5) rezips the directory
            6) overwrites the S3 source_dir with the new tar.gz

        If the source_dir is a local path:
            1) copies the _repack_model.py script into the source dir
            2) creates the _repack_script_launcher.sh in the source dir
        """
        fname = os.path.join(os.path.dirname(__file__), REPACK_SCRIPT)
        if self._source_dir.lower().startswith("s3://"):
            with tempfile.TemporaryDirectory() as tmp:
                targz_contents_dir = os.path.join(tmp, "extracted")

                old_targz_path = os.path.join(tmp, "old.tar.gz")
                download_file_from_url(self._source_dir, old_targz_path, self.sagemaker_session)

                with tarfile.open(name=old_targz_path, mode="r:gz") as t:
                    custom_extractall_tarfile(t, targz_contents_dir)

                shutil.copy2(fname, os.path.join(targz_contents_dir, REPACK_SCRIPT))
                with open(
                    os.path.join(targz_contents_dir, REPACK_SCRIPT_LAUNCHER), "w"
                ) as launcher_file:
                    launcher_file.write(LAUNCH_REPACK_SCRIPT_CMD)

                new_targz_path = os.path.join(tmp, "new.tar.gz")
                with tarfile.open(new_targz_path, mode="w:gz") as t:
                    t.add(targz_contents_dir, arcname=os.path.sep)

                _save_model(self._source_dir, new_targz_path, self.sagemaker_session, kms_key=None)
        else:
            shutil.copy2(fname, os.path.join(self._source_dir, REPACK_SCRIPT))
            with open(os.path.join(self._source_dir, REPACK_SCRIPT_LAUNCHER), "w") as launcher_file:
                launcher_file.write(LAUNCH_REPACK_SCRIPT_CMD)

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that are used to call `create_training_job`.

        This first prepares the source bundle for repackinglby placing artifacts
        in locations which the training container will make available to the
        repacking script and then gets the arguments for the training job.
        """
        self._prepare_for_repacking()
        return super(_RepackModelStep, self).arguments

    @property
    def properties(self):
        """A Properties object representing the DescribeTrainingJobResponse data model."""
        return self._properties
