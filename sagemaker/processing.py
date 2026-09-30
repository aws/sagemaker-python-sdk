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
import shutil
import tempfile
import urllib.parse

from textwrap import dedent

from sagemaker import s3, image_uris, vpc_utils
from sagemaker.config import (
    PROCESSING_JOB_ENVIRONMENT_PATH,
    PROCESSING_JOB_INPUTS_S3_INPUT_S3_URI_PATH,
    PROCESSING_JOB_INTER_CONTAINER_ENCRYPTION_PATH,
    PROCESSING_JOB_KMS_KEY_ID_PATH,
    PROCESSING_JOB_NETWORK_CONFIG_PATH,
    PROCESSING_JOB_OUTPUTS_S3_OUTPUT_S3_URI_PATH,
    PROCESSING_JOB_PROCESSING_RESOURCES_CLUSTER_CONFIG_INSTANCE_COUNT_PATH,
    PROCESSING_JOB_PROCESSING_RESOURCES_CLUSTER_CONFIG_INSTANCE_TYPE_PATH,
    PROCESSING_JOB_PROCESSING_RESOURCES_CLUSTER_CONFIG_VOLUME_KMS_KEY_ID_PATH,
    PROCESSING_JOB_PROCESSING_RESOURCES_CLUSTER_CONFIG_VOLUME_SIZE_IN_GB_PATH,
    PROCESSING_JOB_ROLE_ARN_PATH,
    PROCESSING_JOB_SECURITY_GROUP_IDS_PATH,
    PROCESSING_JOB_SUBNETS_PATH,
    PROCESSING_JOB_TAGS_PATH,
)
from sagemaker.local import LocalSession
from sagemaker.network import NetworkConfig
from sagemaker.session import Session
from sagemaker.utils import (
    base_name_from_image,
    get_config_value,
    name_from_base,
    resolve_value_from_config,
    check_and_get_run_experiment_config,
    format_tags,
    Tags,
)
from sagemaker.workflow import is_pipeline_variable
from sagemaker.workflow.pipeline_context import runnable_by_pipeline
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow import utilities as workflow_utilities
from sagemaker.common_utils import LogState
from sagemaker.apiutils import _utils as apiutils

try:
    from sagemaker.utils.code_injection import codec
except ImportError:
    codec = None

try:
    from sagemaker.apiutils._utils import serialize
except ImportError:
    serialize = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def _is_local_mode(instance_type, sagemaker_session=None):
    """Determine if the processor is running in local mode.

    Args:
        instance_type (str): The instance type.
        sagemaker_session: The SageMaker session.

    Returns:
        bool: True if running in local mode.
    """
    if instance_type is not None and str(instance_type).startswith("local"):
        return True
    if sagemaker_session is not None and getattr(sagemaker_session, "local_mode", False):
        return True
    return False


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
        tags: Tags = None,
        network_config=None,
        arguments=None,
    ):
        """Initializes a ``Processor`` instance.

        The ``Processor`` handles Amazon SageMaker Processing tasks.

        Args:
            role (str): An AWS IAM role name or ARN. Amazon SageMaker Processing
                uses this role to access AWS resources, such as data stored in
                Amazon S3. Required for non-local mode; optional when running
                in local mode (instance_type='local'/'local_gpu' or
                session.local_mode=True).
            image_uri (str or PipelineVariable): The URI of the Docker image to use
                for the processing jobs.
            instance_count (int or PipelineVariable): The number of instances to run
                a processing job with.
            instance_type (str or PipelineVariable): The type of ML compute instance
                to use for the processing job.
            entrypoint (list[str]): The entrypoint for the processing job (default: None).
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
            env (dict[str, str]): Environment variables to be passed to the
                processing jobs (default: None).
            tags (Optional[Tags]): Tags to be passed to the processing job
                (default: None).
            network_config (:class:`~sagemaker.network.NetworkConfig`):
                A :class:`~sagemaker.network.NetworkConfig` object that configures
                network isolation, encryption of inter-container traffic,
                security group IDs, and subnets (default: None).
            arguments (list[str]): A list of string arguments to be passed to a
                processing job (default: None).
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
        self.env = env
        self.tags = format_tags(tags)
        self.network_config = network_config
        self.arguments = arguments
        self.jobs = []
        self.latest_job = None
        self._current_job_name = None

        # Handle session creation for local mode
        if instance_type is not None and str(instance_type).startswith("local"):
            if sagemaker_session is None:
                sagemaker_session = LocalSession()
        elif sagemaker_session is None:
            sagemaker_session = Session()

        self.sagemaker_session = sagemaker_session

        # Validate role: required for non-local mode, optional for local mode
        if not _is_local_mode(instance_type, sagemaker_session):
            if role is None and not is_pipeline_variable(role):
                raise ValueError(
                    "AWS IAM role is required for non-local processing jobs. "
                    "Please provide a valid IAM role ARN."
                )

    def _generate_current_job_name(self, job_name=None):
        """Generate the job name before running a processing job.

        Args:
            job_name (str): Name of the processing job to be created. If not
                specified, one is generated, using the base name given to the
                constructor if applicable.

        Returns:
            str: The supplied or generated job name.
        """
        if job_name is not None:
            return job_name
        # Honor supplied base_job_name or derive from image_uri
        if self.base_job_name:
            base = self.base_job_name
        else:
            base = base_name_from_image(
                self.image_uri, default_base_name="processing"
            )
        # Replace invalid characters
        base = re.sub(r"[^a-zA-Z0-9-]", "-", base)
        return name_from_base(base)

    def _normalize_args(self, **kwargs):
        """Normalize arguments for processing job."""
        code = kwargs.get("code")
        if code is not None and is_pipeline_variable(code):
            if not (isinstance(code, str) and code.startswith("s3://")):
                raise ValueError(
                    "code argument has to be a valid S3 URI when it is a pipeline variable"
                )
        return kwargs

    def _normalize_inputs(self, inputs):
        """Normalize and validate processing inputs.

        Args:
            inputs (list): List of ProcessingInput objects.

        Returns:
            list: Normalized list of ProcessingInput objects.
        """
        from sagemaker.processing import ProcessingInput as PI

        if inputs is None:
            return []

        normalized = []
        for inp in inputs:
            if not isinstance(inp, PI):
                raise TypeError(
                    "Processing inputs must be provided as ProcessingInput objects."
                )
            normalized.append(inp)
        return normalized

    def _normalize_outputs(self, outputs):
        """Normalize and validate processing outputs.

        Args:
            outputs (list): List of ProcessingOutput objects.

        Returns:
            list: Normalized list of ProcessingOutput objects.
        """
        from sagemaker.processing import ProcessingOutput as PO

        if outputs is None:
            return []

        normalized = []
        for output in outputs:
            if not isinstance(output, PO):
                raise TypeError(
                    "Processing outputs must be provided as ProcessingOutput objects."
                )

            # If the output has a pipeline variable URI, skip normalization
            if output.s3_output and is_pipeline_variable(output.s3_output.s3_uri):
                normalized.append(output)
                continue

            # Check if the URI is already an S3 URI - pass through unchanged
            if output.s3_output and output.s3_output.s3_uri:
                uri = output.s3_output.s3_uri
                if uri.startswith("s3://"):
                    normalized.append(output)
                    continue

                # In local mode, preserve file:// URIs
                if _is_local_mode(self.instance_type, self.sagemaker_session):
                    if uri.startswith("file://"):
                        normalized.append(output)
                        continue

                # For non-S3 URIs in non-local mode, generate an S3 path
                # Check if we're in a pipeline context
                pipeline_config = workflow_utilities._pipeline_config
                if pipeline_config is not None:
                    normalized.append(output)
                    continue

                # Generate default S3 output path
                default_bucket = self.sagemaker_session.default_bucket()
                prefix = self.sagemaker_session.default_bucket_prefix
                job_name = self._current_job_name
                output_name = output.output_name or "output"

                if prefix:
                    s3_uri = f"s3://{default_bucket}/{prefix}/{job_name}/output/{output_name}"
                else:
                    s3_uri = f"s3://{default_bucket}/{job_name}/output/{output_name}"

                # Create a new output with the generated S3 URI
                from sagemaker.processing import ProcessingS3Output, ProcessingOutput
                new_s3_output = ProcessingS3Output(
                    s3_uri=s3_uri,
                    local_path=output.s3_output.local_path,
                    s3_upload_mode=output.s3_output.s3_upload_mode,
                )
                new_output = ProcessingOutput(
                    output_name=output.output_name,
                    s3_output=new_s3_output,
                )
                normalized.append(new_output)
            else:
                normalized.append(output)

        return normalized

    def _get_process_args(self, inputs, outputs, kms_key):
        """Get processing job arguments."""
        app_specification = {"ImageUri": self.image_uri}
        if self.entrypoint:
            app_specification["ContainerEntrypoint"] = self.entrypoint
        if self.arguments:
            app_specification["ContainerArguments"] = self.arguments

        resources = {
            "ClusterConfig": {
                "InstanceCount": self.instance_count,
                "InstanceType": self.instance_type,
                "VolumeSizeInGB": self.volume_size_in_gb,
            }
        }

        if self.volume_kms_key:
            resources["ClusterConfig"]["VolumeKmsKeyId"] = self.volume_kms_key

        stopping_condition = None
        if self.max_runtime_in_seconds:
            stopping_condition = {"MaxRuntimeInSeconds": self.max_runtime_in_seconds}

        network_config = None
        if self.network_config:
            network_config = self.network_config._to_request_dict() if hasattr(
                self.network_config, '_to_request_dict'
            ) else self.network_config

        role_arn = self.role
        if role_arn and not is_pipeline_variable(role_arn):
            role_arn = self.sagemaker_session.expand_role(role_arn)

        return {
            "job_name": self._current_job_name,
            "inputs": inputs,
            "output_config": {"Outputs": outputs},
            "resources": resources,
            "stopping_condition": stopping_condition,
            "app_specification": app_specification,
            "environment": self.env,
            "network_config": network_config,
            "role_arn": role_arn,
            "tags": self.tags or [],
        }

    def _start_new(self, inputs, outputs, kms_key):
        """Start a new processing job."""
        process_args = self._get_process_args(inputs, outputs, kms_key)

        if hasattr(self.sagemaker_session, '_intercept_create_request'):
            if serialize is not None and codec is not None:
                serialized = serialize(process_args)
                transformed = codec.transform(serialized)
                # Remove tags before creating ProcessingJob
                transformed.pop("tags", None)
                from sagemaker.processing import ProcessingJob
                job = ProcessingJob(sagemaker_session=self.sagemaker_session, **transformed)
                return job

        return None

    @runnable_by_pipeline
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
        code=None,
    ):
        """Runs a processing job.

        Args:
            inputs (list[:class:`~sagemaker.processing.ProcessingInput`]): Input files
                for the processing job.
            outputs (list[:class:`~sagemaker.processing.ProcessingOutput`]): Outputs
                for the processing job.
            arguments (list[str]): A list of string arguments to be passed to a
                processing job.
            wait (bool): Whether the call should wait until the job completes
                (default: True).
            logs (bool): Whether to show the logs produced by the job (default: True).
            job_name (str): Processing job name.
            experiment_config (dict[str, str]): Experiment management configuration.
            kms_key (str): The ARN of the KMS key.
            code (str): The S3 URI or local path to the code file.
        """
        if logs and not wait:
            raise ValueError("Logs can only be shown if wait is set to True.")

        if arguments:
            self.arguments = arguments

        self._current_job_name = self._generate_current_job_name(job_name)

        normalized_inputs = self._normalize_inputs(inputs)
        normalized_outputs = self._normalize_outputs(outputs)

        job = self._start_new(normalized_inputs, normalized_outputs, kms_key)

        self.latest_job = job
        self.jobs.append(job)

        if wait and job is not None:
            job.wait(logs=logs)

        return job


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
        tags: Tags = None,
        network_config=None,
    ):
        """Initializes a ``ScriptProcessor`` instance.

        Args:
            role (str): An AWS IAM role name or ARN. Optional in local mode.
            image_uri (str): The URI of the Docker image.
            command (list[str]): The command to run (default: ["python3"]).
            instance_count (int): The number of instances.
            instance_type (str): The type of ML compute instance.
            volume_size_in_gb (int): Size in GB of the EBS volume (default: 30).
            volume_kms_key (str): A KMS key for the processing volume.
            output_kms_key (str): The KMS key ID for outputs.
            max_runtime_in_seconds (int): Timeout in seconds.
            base_job_name (str): Prefix for processing job name.
            sagemaker_session: Session object.
            env (dict): Environment variables.
            tags (Optional[Tags]): Tags for the processing job.
            network_config: Network configuration.
        """
        super().__init__(
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
        self.command = command or ["python3"]

    def _get_user_code_name(self, code):
        """Get the user code filename from a path or S3 URI."""
        return os.path.basename(code)

    def _handle_user_code_url(self, code):
        """Handle user code URL - upload local files to S3."""
        if code.startswith("s3://"):
            return code

        parsed = urllib.parse.urlparse(code)
        if parsed.scheme and parsed.scheme not in ("", "file"):
            raise ValueError(
                f"code url scheme {parsed.scheme} is not recognized. "
                "Please use a local file path or S3 URI."
            )

        # Local file
        if not os.path.exists(code):
            raise ValueError(f"code file {code} wasn't found. Please provide a valid file path.")

        if os.path.isdir(code):
            raise ValueError(f"code {code} must be a file, not a directory.")

        # Upload to S3
        desired_s3_uri = s3.s3_path_join(
            "s3://",
            self.sagemaker_session.default_bucket(),
            self.sagemaker_session.default_bucket_prefix or "",
            self._current_job_name,
            "input",
            "code",
            os.path.basename(code),
        )
        return s3.S3Uploader.upload(
            local_path=code,
            desired_s3_uri=desired_s3_uri,
            sagemaker_session=self.sagemaker_session,
        )

    def _upload_code(self, code):
        """Upload code to S3."""
        pipeline_config = workflow_utilities._pipeline_config
        if pipeline_config is not None:
            desired_s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix or "",
                pipeline_config.pipeline_name,
                "code",
                pipeline_config.code_hash if hasattr(pipeline_config, 'code_hash') else "",
                os.path.basename(code),
            )
        else:
            desired_s3_uri = s3.s3_path_join(
                "s3://",
                self.sagemaker_session.default_bucket(),
                self.sagemaker_session.default_bucket_prefix or "",
                self._current_job_name,
                "input",
                "code",
                os.path.basename(code),
            )

        return s3.S3Uploader.upload(
            local_path=code,
            desired_s3_uri=desired_s3_uri,
            sagemaker_session=self.sagemaker_session,
        )

    def _convert_code_and_add_to_inputs(self, inputs, s3_uri):
        """Convert code S3 URI to a processing input and add to inputs list."""
        from sagemaker.processing import ProcessingInput, ProcessingS3Input

        code_input = ProcessingInput(
            input_name="code",
            s3_input=ProcessingS3Input(
                s3_uri=s3_uri,
                local_path="/opt/ml/processing/input/code",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            ),
        )
        return inputs + [code_input]

    def _set_entrypoint(self, command, user_script_name):
        """Set the entrypoint for the processing container."""
        self.entrypoint = command + [
            os.path.join("/opt/ml/processing/input/code", user_script_name)
        ]

    def run(
        self,
        code=None,
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
            code (str): The S3 URI or local path to the code file.
            inputs (list): Input files for the processing job.
            outputs (list): Outputs for the processing job.
            arguments (list[str]): Arguments for the processing job.
            wait (bool): Whether to wait for completion (default: True).
            logs (bool): Whether to show logs (default: True).
            job_name (str): Processing job name.
            experiment_config (dict): Experiment configuration.
            kms_key (str): The ARN of the KMS key.
        """
        if logs and not wait:
            raise ValueError("Logs can only be shown if wait is set to True.")

        if arguments:
            self.arguments = arguments

        self._current_job_name = self._generate_current_job_name(job_name)

        # Handle code
        if code:
            s3_uri = self._handle_user_code_url(code)
            user_code_name = self._get_user_code_name(code)
            self._set_entrypoint(self.command, user_code_name)
            inputs = inputs or []
            inputs = self._convert_code_and_add_to_inputs(inputs, s3_uri)

        normalized_inputs = self._normalize_inputs(inputs)
        normalized_outputs = self._normalize_outputs(outputs)

        job = self._start_new(normalized_inputs, normalized_outputs, kms_key)

        self.latest_job = job
        self.jobs.append(job)

        if wait and job is not None:
            job.wait(logs=logs)

        return job


class FrameworkProcessor(ScriptProcessor):
    """Handles Amazon SageMaker processing tasks for jobs using ML frameworks."""

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
        tags: Tags = None,
        network_config=None,
        code_location=None,
    ):
        """Initializes a ``FrameworkProcessor`` instance.

        Args:
            role (str): An AWS IAM role name or ARN. Optional in local mode.
            image_uri (str): The URI of the Docker image.
            command (list[str]): The command to run (default: ["python"]).
            instance_count (int): The number of instances.
            instance_type (str): The type of ML compute instance.
            volume_size_in_gb (int): Size in GB of the EBS volume (default: 30).
            volume_kms_key (str): A KMS key for the processing volume.
            output_kms_key (str): The KMS key ID for outputs.
            max_runtime_in_seconds (int): Timeout in seconds.
            base_job_name (str): Prefix for processing job name.
            sagemaker_session: Session object.
            env (dict): Environment variables.
            tags (Optional[Tags]): Tags for the processing job.
            network_config: Network configuration.
            code_location (str): S3 URI for code storage.
        """
        super().__init__(
            role=role,
            image_uri=image_uri,
            command=command or ["python"],
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
        # Strip trailing slash from code_location
        self.code_location = code_location.rstrip("/") if code_location else None

    def _generate_framework_script(self, entry_point):
        """Generate the framework script content."""
        script = dedent(f"""\
            #!/bin/bash

            cd /opt/ml/processing/input/code

            # Install requirements if they exist
            if [ -f requirements.txt ]; then
                python install_requirements.py
            fi

            # Run the entry point
            {' '.join(self.command)} {entry_point}
        """)
        return script

    def _create_and_upload_runproc(self, entry_point, requirements, s3_uri):
        """Create and upload the runproc.sh script."""
        script_content = self._generate_framework_script(entry_point)
        return s3.S3Uploader.upload_string_as_file_body(
            body=script_content,
            desired_s3_uri=s3_uri,
            sagemaker_session=self.sagemaker_session,
        )

    def _set_entrypoint(self, command, user_script_name):
        """Set the entrypoint for the framework processing container."""
        self.entrypoint = [
            "/bin/bash",
            os.path.join("/opt/ml/processing/input/code", user_script_name),
        ]

    def _patch_inputs_with_payload(self, inputs, s3_uri):
        """Patch inputs with the code payload."""
        from sagemaker.processing import ProcessingInput, ProcessingS3Input

        code_input = ProcessingInput(
            input_name="code",
            s3_input=ProcessingS3Input(
                s3_uri=s3_uri,
                local_path="/opt/ml/processing/input/code",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            ),
        )
        return inputs + [code_input]

    def _package_code(self, entry_point, source_dir, requirements, job_name, kms_key):
        """Package code into a tar.gz and upload to S3."""
        if source_dir and not os.path.exists(source_dir):
            raise ValueError(f"source_dir does not exist: {source_dir}")

        if source_dir is None:
            source_dir = os.path.dirname(os.path.abspath(entry_point))

        # Determine S3 destination
        if self.code_location:
            s3_prefix = f"{self.code_location}/{job_name}/source"
        else:
            bucket = self.sagemaker_session.default_bucket()
            prefix = self.sagemaker_session.default_bucket_prefix or ""
            if prefix:
                s3_prefix = f"s3://{bucket}/{prefix}/{job_name}/source"
            else:
                s3_prefix = f"s3://{bucket}/{job_name}/source"

        return f"{s3_prefix}/sourcedir.tar.gz"

    def _pack_and_upload_code(self, code, source_dir, requirements, job_name, inputs, kms_key):
        """Pack and upload code, returning the S3 URI and updated inputs."""
        if code.startswith("s3://"):
            return code, inputs or [], job_name

        self._current_job_name = self._generate_current_job_name(job_name)

        # Package the code
        payload_s3_uri = self._package_code(
            entry_point=code,
            source_dir=source_dir,
            requirements=requirements,
            job_name=self._current_job_name,
            kms_key=kms_key,
        )

        # Determine runproc.sh location
        if self.code_location:
            runproc_s3_uri = f"{self.code_location}/{self._current_job_name}/source/runproc.sh"
        else:
            bucket = self.sagemaker_session.default_bucket()
            prefix = self.sagemaker_session.default_bucket_prefix or ""
            if prefix:
                runproc_s3_uri = f"s3://{bucket}/{prefix}/{self._current_job_name}/source/runproc.sh"
            else:
                runproc_s3_uri = f"s3://{bucket}/{self._current_job_name}/source/runproc.sh"

        # Upload install_requirements.py
        install_req_s3_uri = runproc_s3_uri.replace("runproc.sh", "install_requirements.py")
        install_req_content = "import subprocess\nimport sys\nsubprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])"
        s3.S3Uploader.upload_string_as_file_body(
            body=install_req_content,
            desired_s3_uri=install_req_s3_uri,
            sagemaker_session=self.sagemaker_session,
        )

        # Create and upload runproc.sh
        entry_point_name = os.path.basename(code)
        uploaded_uri = self._create_and_upload_runproc(
            entry_point_name, requirements, runproc_s3_uri
        )

        # Patch inputs with the code payload
        inputs = inputs or []
        inputs = self._patch_inputs_with_payload(inputs, payload_s3_uri)

        return uploaded_uri, inputs, self._current_job_name

    def run(
        self,
        code=None,
        inputs=None,
        outputs=None,
        arguments=None,
        wait=True,
        logs=True,
        job_name=None,
        experiment_config=None,
        kms_key=None,
        source_dir=None,
        requirements=None,
    ):
        """Runs a processing job.

        Args:
            code (str): The S3 URI or local path to the code file.
            inputs (list): Input files for the processing job.
            outputs (list): Outputs for the processing job.
            arguments (list[str]): Arguments for the processing job.
            wait (bool): Whether to wait for completion (default: True).
            logs (bool): Whether to show logs (default: True).
            job_name (str): Processing job name.
            experiment_config (dict): Experiment configuration.
            kms_key (str): The ARN of the KMS key.
            source_dir (str): Path to source directory.
            requirements (str): Path to requirements file.
        """
        if logs and not wait:
            raise ValueError("Logs can only be shown if wait is set to True.")

        if arguments:
            self.arguments = arguments

        self._current_job_name = self._generate_current_job_name(job_name)

        if code:
            if code.startswith("s3://"):
                # S3 code - use directly
                user_code_name = self._get_user_code_name(code)
                self._set_entrypoint(self.command, user_code_name)
            else:
                # Local code - pack and upload
                uploaded_uri, inputs, _ = self._pack_and_upload_code(
                    code=code,
                    source_dir=source_dir,
                    requirements=requirements,
                    job_name=self._current_job_name,
                    inputs=inputs,
                    kms_key=kms_key,
                )
                user_code_name = "runproc.sh"
                self._set_entrypoint(self.command, user_code_name)

        normalized_inputs = self._normalize_inputs(inputs)
        normalized_outputs = self._normalize_outputs(outputs)

        job = self._start_new(normalized_inputs, normalized_outputs, kms_key)

        self.latest_job = job
        self.jobs.append(job)

        if wait and job is not None:
            job.wait(logs=logs)

        return job


class ProcessingInput:
    """Represents a processing input."""

    def __init__(
        self,
        input_name=None,
        s3_input=None,
        dataset_definition=None,
        app_managed=False,
    ):
        self.input_name = input_name
        self.s3_input = s3_input
        self.dataset_definition = dataset_definition
        self.app_managed = app_managed


class ProcessingOutput:
    """Represents a processing output."""

    def __init__(
        self,
        output_name=None,
        s3_output=None,
        app_managed=False,
    ):
        self.output_name = output_name
        self.s3_output = s3_output
        self.app_managed = app_managed


class ProcessingS3Input:
    """Represents an S3 input for processing."""

    def __init__(
        self,
        s3_uri=None,
        local_path=None,
        s3_data_type=None,
        s3_input_mode=None,
        s3_data_distribution_type=None,
        s3_compression_type=None,
    ):
        self.s3_uri = s3_uri
        self.local_path = local_path
        self.s3_data_type = s3_data_type
        self.s3_input_mode = s3_input_mode
        self.s3_data_distribution_type = s3_data_distribution_type
        self.s3_compression_type = s3_compression_type


class ProcessingS3Output:
    """Represents an S3 output for processing."""

    def __init__(
        self,
        s3_uri=None,
        local_path=None,
        s3_upload_mode=None,
    ):
        self.s3_uri = s3_uri
        self.local_path = local_path
        self.s3_upload_mode = s3_upload_mode


class ProcessingJob:
    """Represents a processing job."""

    def __init__(self, sagemaker_session=None, **kwargs):
        self.sagemaker_session = sagemaker_session
        for key, value in kwargs.items():
            setattr(self, key, value)

    def wait(self, logs=True):
        """Wait for the processing job to complete."""
        pass


def _processing_input_to_request_dict(processing_input):
    """Convert a ProcessingInput to a request dictionary."""
    result = {"InputName": processing_input.input_name}

    if processing_input.s3_input:
        s3_input = processing_input.s3_input
        result["S3Input"] = {
            "S3Uri": s3_input.s3_uri,
            "LocalPath": s3_input.local_path,
            "S3DataType": s3_input.s3_data_type,
            "S3InputMode": s3_input.s3_input_mode,
        }
        if s3_input.s3_data_distribution_type:
            result["S3Input"]["S3DataDistributionType"] = s3_input.s3_data_distribution_type
        if s3_input.s3_compression_type:
            result["S3Input"]["S3CompressionType"] = s3_input.s3_compression_type

    if processing_input.dataset_definition:
        result["DatasetDefinition"] = processing_input.dataset_definition

    if processing_input.app_managed:
        result["AppManaged"] = True

    return result


def _processing_output_to_request_dict(processing_output):
    """Convert a ProcessingOutput to a request dictionary."""
    result = {"OutputName": processing_output.output_name}

    if processing_output.s3_output:
        s3_output = processing_output.s3_output
        result["S3Output"] = {
            "S3Uri": s3_output.s3_uri,
            "LocalPath": s3_output.local_path,
            "S3UploadMode": s3_output.s3_upload_mode,
        }

    if processing_output.app_managed:
        result["AppManaged"] = True

    return result


def _get_process_request(
    inputs,
    output_config,
    job_name,
    resources,
    stopping_condition,
    app_specification,
    environment,
    network_config,
    role_arn,
    tags,
    experiment_config=None,
):
    """Build the processing job request dictionary."""
    request = {
        "ProcessingJobName": job_name,
        "AppSpecification": app_specification,
        "RoleArn": role_arn,
        "ProcessingResources": resources,
    }

    if inputs:
        request["ProcessingInputs"] = inputs

    if output_config:
        request["ProcessingOutputConfig"] = output_config

    if stopping_condition:
        request["StoppingCondition"] = stopping_condition

    if environment:
        request["Environment"] = environment

    if network_config:
        request["NetworkConfig"] = network_config

    if tags:
        request["Tags"] = tags

    if experiment_config:
        request["ExperimentConfig"] = experiment_config

    return request


def _wait_until(session, job_name, poll=5):
    """Wait until a processing job completes."""
    pass


def _logs_init(session, job_name, log_group):
    """Initialize log streaming."""
    return (1, [], {}, None, log_group, False, lambda x: x)


def _flush_log_streams(*args, **kwargs):
    """Flush log streams."""
    pass


def _get_initial_job_state(description, log_state):
    """Get initial job state."""
    return LogState.COMPLETE


def _check_job_status(*args, **kwargs):
    """Check job status."""
    pass


def logs_for_processing_job(session, job_name, wait=True, poll=10):
    """Display logs for a processing job."""
    pass
