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
"""This module contains code related to the Processor class.

which is used for Amazon SageMaker Processing Jobs. These jobs let users
perform data pre-processing, post-processing, feature engineering,
data validation, and model evaluation, and interpretation on Amazon SageMaker.
"""
from __future__ import absolute_import

import logging
import os
import pathlib
import re
import attr
import tempfile

from typing import Dict, List, Optional, Union

from sagemaker import image_uris, s3, utils
from sagemaker.session import Session
from sagemaker.local import LocalSession
from sagemaker.network import NetworkConfig
from sagemaker.fw_utils import tar_and_upload_dir
from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.functions import Join
from sagemaker.dataset_definition.inputs import (
    S3Input,
    DatasetDefinition,
)
from sagemaker.apiutils import _utils
from sagemaker.s3 import S3Uploader, s3_path_join, parse_s3_url
from sagemaker.utils import (
    base_name_from_image,
    get_config_value,
    name_from_base,
    resolve_value_from_config,
    check_and_get_run_experiment_config,
    format_tags,
    Tags,
)
from sagemaker.config import (
    PROCESSING_JOB_ENVIRONMENT_PATH,
    PROCESSING_JOB_INPUTS_S3_INPUT_S3_URI_PATH,
    PROCESSING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
    PROCESSING_JOB_KMS_KEY_ID_PATH,
    PROCESSING_JOB_NETWORK_CONFIG_PATH,
    PROCESSING_JOB_OUTPUTS_S3_OUTPUT_S3_URI_PATH,
    PROCESSING_JOB_ROLE_ARN_PATH,
    PROCESSING_JOB_SECURITY_GROUP_IDS_PATH,
    PROCESSING_JOB_SUBNETS_PATH,
    PROCESSING_JOB_TAGS_PATH,
    PROCESSING_JOB_VOLUME_KMS_KEY_ID_PATH,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Processor(object):
    """Handles Amazon SageMaker Processing tasks."""

    def __init__(
        self,
        role=None,
        image_uri=None,
        instance_count=None,
        instance_type=None,
        entrypoint=None,
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """Initializes a ``Processor`` instance.

        The ``Processor`` handles Amazon SageMaker Processing tasks.

        Args:
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as
                data stored in Amazon S3.
            image_uri (str or PipelineVariable): The URI of the Docker image to use
                for the processing jobs.
            instance_count (int or PipelineVariable): The number of instances to run
                a processing job with.
            instance_type (str or PipelineVariable): The type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            entrypoint (list[str] or list[PipelineVariable]): The entrypoint for
                the processing job (default: None).
            volume_size_in_gb (int or PipelineVariable): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str or PipelineVariable): A KMS key for the processing
                volume (default: None).
            output_kms_key (str or PipelineVariable): The KMS key ID for processing
                job outputs (default: None).
            max_runtime_in_seconds (int or PipelineVariable): Timeout in seconds
                (default: None).
            base_job_name (str): Prefix for processing job name.
            sagemaker_session (:class:`~sagemaker.session.Session`):
                Session object which manages interactions with Amazon SageMaker and
                any other AWS services needed.
            env (dict[str, str] or dict[str, PipelineVariable]): Environment variables
                to be passed to the processing jobs (default: None).
            tags (Optional[Tags]): Tags to be passed to the processing job
                (default: None).
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig`
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets
                (default: None).
        """
        self.role = role
        self.image_uri = image_uri
        self.instance_count = instance_count
        self.instance_type = instance_type
        self.entrypoint = entrypoint
        self.volume_size_in_gb = volume_size_in_gb
        self.volume_kms_key = volume_kms_key
        self.output_kms_key = output_kms_key
        self.max_runtime_in_seconds = max_runtime_in_seconds
        self.base_job_name = base_job_name
        self.sagemaker_session = sagemaker_session or Session()
        self.env = env
        self.tags = format_tags(tags)
        self.network_config = network_config
        self.jobs = []
        self.latest_job = None
        self._current_job_name = None

    def run(
        self,
        inputs=None,
        outputs=None,
        arguments=None,
        wait=True,
        logs=True,
        job_name=None,
        experiment_config=None,
        kms_key=None,
    ):
        """Runs a processing job.

        Args:
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files for
                the processing job.
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs for
                the processing job.
            arguments (list[str] or list[PipelineVariable]): A list of string arguments
                to be passed to a processing job (default: None).
            wait (bool): Whether the call should wait until the job completes
                (default: True).
            logs (bool): Whether to show the logs produced by the job (default: True).
            job_name (str): Processing job name.
            experiment_config (dict[str, str]): Experiment management configuration.
            kms_key (str): The ARN of the KMS key that is used to encrypt the
                user code file (default: None).
        """
        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        normalized_inputs = self._normalize_inputs(inputs)
        normalized_outputs = self._normalize_outputs(outputs)

        experiment_config = check_and_get_run_experiment_config(experiment_config)

        self.latest_job = ProcessingJob.start_new(
            processor=self,
            inputs=normalized_inputs,
            outputs=normalized_outputs,
            experiment_config=experiment_config,
            )

        self.jobs.append(self.latest_job)

        if wait:
            self.latest_job.wait(logs=logs)

    def _generate_current_job_name(self, job_name=None):
        """Generates the job name before running a processing job."""
        if job_name is not None:
            return job_name
        if self.base_job_name is not None:
            base = self.base_job_name
        else:
            base = base_name_from_image(
                self.image_uri, default_base_name="processing"
            )
        return name_from_base(base)

    def _normalize_inputs(self, inputs=None):
        """Ensures that all the ``ProcessingInput`` objects have names and S3 URIs."""
        if inputs is None:
            return []
        normalized = []
        for count, processing_input in enumerate(inputs, 1):
            if not isinstance(processing_input, ProcessingInput):
                raise TypeError("Your inputs must be provided as ProcessingInput objects.")
            if processing_input.input_name is None:
                processing_input.input_name = "input-{}".format(count)
            normalized.append(processing_input)
        return normalized

    def _normalize_outputs(self, outputs=None):
        """Ensures that all the ``ProcessingOutput`` objects have names and S3 URIs."""
        if outputs is None:
            return []
        normalized = []
        for count, processing_output in enumerate(outputs, 1):
            if not isinstance(processing_output, ProcessingOutput):
                raise TypeError("Your outputs must be provided as ProcessingOutput objects.")
            if processing_output.output_name is None:
                processing_output.output_name = "output-{}".format(count)
            normalized.append(processing_output)
        return normalized


class ScriptProcessor(Processor):
    """Handles Amazon SageMaker processing tasks for jobs using a machine learning framework."""

    def __init__(
        self,
        role=None,
        image_uri=None,
        command=None,
        instance_count=None,
        instance_type=None,
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """Initializes a ``ScriptProcessor`` instance."""
        super(ScriptProcessor, self).__init__(
            role=role,
            image_uri=image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )
        self.command = command


class FrameworkProcessor(ScriptProcessor):
    """Handles Amazon SageMaker processing tasks for jobs using a machine learning framework."""

    def __init__(
        self,
        role=None,
        image_uri=None,
        command=None,
        instance_count=None,
        instance_type=None,
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        base_job_name=None,
        sagemaker_session=None,
        env=None,
        tags=None,
        network_config=None,
    ):
        """Initializes a ``FrameworkProcessor`` instance."""
        super(FrameworkProcessor, self).__init__(
            role=role,
            image_uri=image_uri,
            command=command,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size_in_gb=volume_size_in_gb,
            volume_kms_key=volume_kms_key,
            output_kms_key=output_kms_key,
            max_runtime_in_seconds=max_runtime_in_seconds,
            base_job_name=base_job_name,
            sagemaker_session=sagemaker_session,
            env=env,
            tags=tags,
            network_config=network_config,
        )

    def _package_code(
        self,
        code,
        source_dir,
        dependencies,
        git_config,
        job_name,
    ):
        """Packages the code for the processing job and uploads to S3.

        This method creates a tar.gz archive of the source code and uploads it
        to S3 for use in the processing job.

        Args:
            code (str): Path to the entry point script.
            source_dir (str): Path to the source directory (default: None).
            dependencies (list[str]): List of dependency paths (default: None).
            git_config (dict): Git configuration for fetching code (default: None).
            job_name (str): Name of the processing job.

        Returns:
            str: The S3 URI of the uploaded code archive.
        """
        if source_dir is None:
            source_dir = os.path.dirname(code)

        # Create the tar.gz archive in a temp file.
        # Fix for issue #5873: On Windows, NamedTemporaryFile cannot be deleted
        # while the handle is still open. We close the handle first by exiting
        # the `with` block, then perform the upload and cleanup.
        with tempfile.NamedTemporaryFile(
            suffix=".tar.gz", prefix="sourcedir", delete=False
        ) as tmp:
            tmp_name = tmp.name
            # Create tar archive of source directory
            self._create_tar_archive(tmp, source_dir, code, dependencies)

        # File handle is now closed - safe to upload and delete on all platforms
        # including Windows (fixes PermissionError [WinError 32])
        try:
            s3_uri = S3Uploader.upload(
                local_path=tmp_name,
                desired_s3_uri=s3_path_join(
                    "s3://",
                    self.sagemaker_session.default_bucket(),
                    self.sagemaker_session.default_bucket_prefix or "",
                    job_name,
                    "input",
                    "code",
                ),
                sagemaker_session=self.sagemaker_session,
            )
        finally:
            os.unlink(tmp_name)

        return s3_uri

    def _create_tar_archive(self, fileobj, source_dir, entry_point, dependencies):
        """Creates a tar.gz archive of the source directory.

        Args:
            fileobj: File object to write the archive to.
            source_dir (str): Path to the source directory.
            entry_point (str): Path to the entry point script.
            dependencies (list[str]): List of dependency paths.
        """
        import tarfile

        with tarfile.open(fileobj=fileobj, mode="w:gz") as tar:
            if source_dir:
                tar.add(source_dir, arcname=".")
            elif entry_point:
                tar.add(entry_point, arcname=os.path.basename(entry_point))

            if dependencies:
                for dependency in dependencies:
                    tar.add(dependency, arcname=os.path.basename(dependency))


class ProcessingInput(object):
    """Accepts parameters that specify an Amazon S3 input for a processing job."""

    def __init__(
        self,
        source=None,
        destination=None,
        input_name=None,
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type="FullyReplicated",
        s3_compression_type="None",
        s3_input=None,
        dataset_definition=None,
        app_managed=False,
    ):
        """Initializes a ``ProcessingInput`` instance."""
        self.source = source
        self.destination = destination
        self.input_name = input_name
        self.s3_data_type = s3_data_type
        self.s3_input_mode = s3_input_mode
        self.s3_data_distribution_type = s3_data_distribution_type
        self.s3_compression_type = s3_compression_type
        self.s3_input = s3_input
        self.dataset_definition = dataset_definition
        self.app_managed = app_managed


class ProcessingOutput(object):
    """Accepts parameters that specify an Amazon S3 output for a processing job."""

    def __init__(
        self,
        source=None,
        destination=None,
        output_name=None,
        s3_upload_mode="EndOfJob",
        app_managed=False,
        feature_store_output=None,
    ):
        """Initializes a ``ProcessingOutput`` instance."""
        self.source = source
        self.destination = destination
        self.output_name = output_name
        self.s3_upload_mode = s3_upload_mode
        self.app_managed = app_managed
        self.feature_store_output = feature_store_output


class ProcessingJob(object):
    """Provides functionality to start and describe processing jobs."""

    def __init__(self, sagemaker_session, job_name, inputs, outputs, output_kms_key=None):
        """Initializes a Processing job."""
        self.sagemaker_session = sagemaker_session
        self.job_name = job_name
        self.inputs = inputs
        self.outputs = outputs
        self.output_kms_key = output_kms_key

    @classmethod
    def start_new(cls, processor, inputs, outputs, experiment_config):
        """Starts a new processing job using the provided processor and arguments."""
        job = cls(
            sagemaker_session=processor.sagemaker_session,
            job_name=processor._current_job_name,
            inputs=inputs,
            outputs=outputs,
            output_kms_key=processor.output_kms_key,
        )
        return job

    def wait(self, logs=True):
        """Waits for the processing job to complete."""
        pass

    def describe(self):
        """Prints a description of the processing job."""
        pass
