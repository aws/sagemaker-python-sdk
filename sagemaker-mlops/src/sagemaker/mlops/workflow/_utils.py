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
# Suffix appended to an ``s3://`` source_dir object name to derive the SDK-owned
# staging key. The caller's original object is never overwritten (issue #6481).
STAGED_SOURCE_DIR_SUFFIX = "-sagemaker-repack.tar.gz"
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

# Static list of regions where Experiments (Eureka) is Generally Available.
# Note: Experiments is not expanding to new regions, so this list is static.
EUREKA_GA_REGIONS = frozenset(
    [
        "us-east-1",  # iad (N. Virginia)
        "us-east-2",  # cmh (Ohio)
        "us-west-1",  # sfo (N. California)
        "us-west-2",  # pdx (Oregon)
        "ca-central-1",  # yul (Montreal)
        "eu-west-1",  # dub (Dublin)
        "eu-west-2",  # lhr (London)
        "eu-west-3",  # cdg (Paris)
        "eu-central-1",  # fra (Frankfurt)
        "eu-north-1",  # arn (Stockholm)
        "eu-south-1",  # mxp (Milan)
        "eu-south-2",  # zaz (Spain)
        "ap-northeast-1",  # nrt (Tokyo)
        "ap-northeast-2",  # icn (Seoul)
        "ap-northeast-3",  # kix (Osaka)
        "ap-southeast-1",  # sin (Singapore)
        "ap-southeast-2",  # syd (Sydney)
        "ap-southeast-3",  # cgk (Jakarta)
        "ap-south-1",  # bom (Mumbai)
        "ap-east-1",  # hkg (Hong Kong)
        "sa-east-1",  # gru (São Paulo)
        "af-south-1",  # cpt (Cape Town)
        "me-south-1",  # bah (Bahrain)
        "il-central-1",  # tlv (Tel Aviv)
        "cn-north-1",  # bjs (Beijing)
        "cn-northwest-1",  # zhy (Ningxia)
    ]
)


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
        # The SDK-owned copy of the source that the repack helper files are written
        # into. ``None`` until the first ``_prepare_for_repacking()`` call.
        self._staged_source_dir = None

        # Stage an SDK-owned copy of the source with the repack scripts in it
        self._prepare_for_repacking()

        # Handle requirements.txt like ModelTrainer
        requirements_file = (
            self._requirements
            if self._requirements and self._requirements.endswith(".txt")
            else None
        )

        # Configure ModelTrainer components for repacking
        from sagemaker.core.training.configs import SourceCode, Compute, Networking

        source_code = SourceCode(
            source_dir=self._staged_source_dir,
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
            instance_type=compute.instance_type,
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
        """Stages an SDK-owned copy of the source for the model trainer.

        The caller-provided ``source_dir`` is never modified: the repack helper
        files are written into a copy that the SDK owns -- a temporary directory
        for a local ``source_dir``, a new S3 object for an ``s3://`` one. Staging
        runs at most once, so repeated ``arguments`` access neither re-stages nor
        re-uploads (see issue #6481).
        """
        if self._staged_source_dir is not None:
            return

        if self._source_dir is not None and self._source_dir.lower().startswith("s3://"):
            self._staged_source_dir = self._stage_s3_source_dir()
            return

        if self._source_dir is None:
            # _establish_source_dir already builds a directory that the SDK owns,
            # so the helper files can go straight into it.
            self._establish_source_dir()
            self._staged_source_dir = self._source_dir
        else:
            self._staged_source_dir = tempfile.mkdtemp()
            shutil.copytree(self._source_dir, self._staged_source_dir, dirs_exist_ok=True)

        self._inject_repack_script_and_launcher(self._staged_source_dir)

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

    def _inject_repack_script_and_launcher(self, target_dir):
        """Injects the _repack_model.py script and _repack_script_launcher.sh

        into a local directory that the SDK owns.

        Note: The bash file is needed because if not supplied, the SKLearn
        training job will auto install all dependencies listed in requirements.txt.
        However, this auto install behavior is not expected in _RepackModelStep,
        since it should just simply repack the model along with other supplied files,
        e.g. the requirements.txt.

        It performs the following:
            1) copies the _repack_model.py script into ``target_dir``
            2) creates the _repack_script_launcher.sh in ``target_dir``

        Args:
            target_dir (str): A local directory owned by the SDK. This is never
                the caller-provided ``source_dir`` (see issue #6481).
        """
        fname = os.path.join(os.path.dirname(__file__), REPACK_SCRIPT)
        shutil.copy2(fname, os.path.join(target_dir, REPACK_SCRIPT))
        # newline="\n" forces LF endings so the bash launcher stays runnable in the
        # Linux repack container even when written from a Windows host (see issue #3762).
        with open(
            os.path.join(target_dir, REPACK_SCRIPT_LAUNCHER), "w", newline="\n"
        ) as launcher_file:
            launcher_file.write(LAUNCH_REPACK_SCRIPT_CMD)

    def _stage_s3_source_dir(self):
        """Stages an ``s3://`` source archive into a new, SDK-owned S3 object.

        It performs the following:
            1) downloads the source_dir tar.gz
            2) extracts it
            3) copies the _repack_model.py script into the extracted directory
            4) creates the _repack_script_launcher.sh in the extracted dir
            5) rezips the directory
            6) uploads the new tar.gz under a new key, alongside the original

        The caller's object is read but never overwritten (see issue #6481). The
        staging key is derived from the original one, so it stays in the caller's
        bucket and prefix and inherits the same access and encryption policies.

        Returns:
            str: The S3 URI of the staged source archive.
        """
        staged_uri = self._staged_s3_source_dir_uri()
        with tempfile.TemporaryDirectory() as tmp:
            targz_contents_dir = os.path.join(tmp, "extracted")

            old_targz_path = os.path.join(tmp, "old.tar.gz")
            download_file_from_url(self._source_dir, old_targz_path, self.sagemaker_session)

            with tarfile.open(name=old_targz_path, mode="r:gz") as t:
                custom_extractall_tarfile(t, targz_contents_dir)

            self._inject_repack_script_and_launcher(targz_contents_dir)

            new_targz_path = os.path.join(tmp, "new.tar.gz")
            with tarfile.open(new_targz_path, mode="w:gz") as t:
                t.add(targz_contents_dir, arcname=os.path.sep)

            _save_model(staged_uri, new_targz_path, self.sagemaker_session, kms_key=None)

        return staged_uri

    def _staged_s3_source_dir_uri(self):
        """Derives the staging S3 URI for an ``s3://`` source_dir.

        Returns:
            str: The original URI with its object name suffixed, keeping the
                caller's bucket and prefix.
        """
        prefix, _, object_name = self._source_dir.rpartition("/")
        for extension in (".tar.gz", ".tgz"):
            if object_name.lower().endswith(extension):
                object_name = object_name[: -len(extension)]
                break
        return "/".join([prefix, object_name + STAGED_SOURCE_DIR_SUFFIX])

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that are used to call `create_training_job`.

        The source bundle is staged once, by the constructor, so repeated access
        here neither re-stages the source nor touches the caller's ``source_dir``
        (see issue #6481).
        """
        self._prepare_for_repacking()
        return super(_RepackModelStep, self).arguments

    @property
    def properties(self):
        """A Properties object representing the DescribeTrainingJobResponse data model."""
        return self._properties
