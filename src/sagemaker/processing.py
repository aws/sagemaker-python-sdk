# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This module contains code related to the Processor class, which is used
for Processing jobs. These jobs let users perform data pre-processing,
post-processing, feature engineering, data validation, and model evaluation
and interpretation on SageMaker.
"""
from __future__ import print_function, absolute_import

import os

from six.moves.urllib.parse import urlparse

from sagemaker.job import _Job
from sagemaker.utils import base_name_from_image, name_from_base
from sagemaker.session import Session
from sagemaker.s3 import S3Uploader
from sagemaker.network import NetworkConfig  # noqa: F401 # pylint: disable=unused-import


class Processor(object):
    """Handles Amazon SageMaker processing tasks."""

    def __init__(
        self,
        role,
        image_uri,
        instance_count,
        instance_type,
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
        """Initialize a ``Processor`` instance. The Processor handles Amazon
        SageMaker processing tasks.

        Args:
            role (str): An AWS IAM role name or ARN. The Amazon SageMaker training jobs
                and APIs that create Amazon SageMaker endpoints use this role
                to access training data and model artifacts. After the endpoint
                is created, the inference code might use the IAM role, if it
                needs to access an AWS resource.
            image_uri (str): The uri of the image to use for the processing
                jobs started by the Processor.
            instance_count (int): The number of instances to run
                the Processing job with.
            instance_type (str): Type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            entrypoint ([str]): The entrypoint for the processing job.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing
                volume.
            output_kms_key (str): The KMS key id for all ProcessingOutputs.
            max_runtime_in_seconds (int): Timeout in seconds
                After this amount of time Amazon SageMaker terminates the job
                regardless of its current status.
            base_job_name (str): Prefix for processing name. If not specified,
                the processor generates a default job name, based on the
                training image name and current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the processor creates one
                using the default AWS configuration chain.
            env (dict): Environment variables to be passed to the processing job.
            tags ([dict]): List of tags to be passed to the processing job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
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
        self.tags = tags
        self.network_config = network_config

        self.jobs = []
        self.latest_job = None
        self._current_job_name = None
        self.arguments = None

    def run(
        self,
        inputs=None,
        outputs=None,
        arguments=None,
        wait=True,
        logs=True,
        job_name=None,
        experiment_config=None,
    ):
        """Run a processing job.

        Args:
            inputs ([sagemaker.processing.ProcessingInput]): Input files for the processing
                job. These must be provided as ProcessingInput objects.
            outputs ([sagemaker.processing.ProcessingOutput]): Outputs for the processing
                job. These can be specified as either a path string or a ProcessingOutput
                object.
            arguments ([str]): A list of string arguments to be passed to a
                processing job.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
                Dictionary contains three optional keys,
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.

        """
        if logs and not wait:
            raise ValueError(
                """Logs can only be shown if wait is set to True.
                Please either set wait to True or set logs to False."""
            )

        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        normalized_inputs = self._normalize_inputs(inputs)
        normalized_outputs = self._normalize_outputs(outputs)
        self.arguments = arguments

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
        # Honor supplied base_job_name or generate it.
        if self.base_job_name:
            base_name = self.base_job_name
        else:
            base_name = base_name_from_image(self.image_uri)

        return name_from_base(base_name)

    def _normalize_inputs(self, inputs=None):
        """Ensure that all the ProcessingInput objects have names and S3 uris.

        Args:
            inputs ([sagemaker.processing.ProcessingInput]): A list of ProcessingInput
                objects to be normalized.

        Returns:
            [sagemaker.processing.ProcessingInput]: The list of normalized
            ProcessingInput objects.
        """
        # Initialize a list of normalized ProcessingInput objects.
        normalized_inputs = []
        if inputs is not None:
            # Iterate through the provided list of inputs.
            for count, file_input in enumerate(inputs, 1):
                if not isinstance(file_input, ProcessingInput):
                    raise TypeError("Your inputs must be provided as ProcessingInput objects.")
                # Generate a name for the ProcessingInput if it doesn't have one.
                if file_input.input_name is None:
                    file_input.input_name = "input-{}".format(count)
                # If the source is a local path, upload it to S3
                # and save the S3 uri in the ProcessingInput source.
                parse_result = urlparse(file_input.source)
                if parse_result.scheme != "s3":
                    desired_s3_uri = os.path.join(
                        "s3://",
                        self.sagemaker_session.default_bucket(),
                        self._current_job_name,
                        "input",
                        file_input.input_name,
                    )
                    s3_uri = S3Uploader.upload(
                        local_path=file_input.source,
                        desired_s3_uri=desired_s3_uri,
                        session=self.sagemaker_session,
                    )
                    file_input.source = s3_uri
                normalized_inputs.append(file_input)
        return normalized_inputs

    def _normalize_outputs(self, outputs=None):
        """Ensure that all the outputs are ProcessingOutput objects with
        names and S3 uris.

        Args:
            outputs ([sagemaker.processing.ProcessingOutput]): A list
                of outputs to be normalized. Can be either strings or
                ProcessingOutput objects.

        Returns:
            [sagemaker.processing.ProcessingOutput]: The list of normalized
                ProcessingOutput objects.
        """
        # Initialize a list of normalized ProcessingOutput objects.
        normalized_outputs = []
        if outputs is not None:
            # Iterate through the provided list of outputs.
            for count, output in enumerate(outputs, 1):
                if not isinstance(output, ProcessingOutput):
                    raise TypeError("Your outputs must be provided as ProcessingOutput objects.")
                # Generate a name for the ProcessingOutput if it doesn't have one.
                if output.output_name is None:
                    output.output_name = "output-{}".format(count)
                # If the output's destination is not an s3_uri, create one.
                parse_result = urlparse(output.destination)
                if parse_result.scheme != "s3":
                    s3_uri = os.path.join(
                        "s3://",
                        self.sagemaker_session.default_bucket(),
                        self._current_job_name,
                        "output",
                        output.output_name,
                    )
                    output.destination = s3_uri
                normalized_outputs.append(output)
        return normalized_outputs


class ScriptProcessor(Processor):
    """Handles Amazon SageMaker processing tasks for jobs using a machine learning framework."""

    def __init__(
        self,
        role,
        image_uri,
        command,
        instance_count,
        instance_type,
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
        """Initialize a ``ScriptProcessor`` instance. The ScriptProcessor
        handles Amazon SageMaker processing tasks for jobs using script mode.

        Args:
            role (str): An AWS IAM role name or ARN. The Amazon SageMaker training jobs
                and APIs that create Amazon SageMaker endpoints use this role
                to access training data and model artifacts. After the endpoint
                is created, the inference code might use the IAM role, if it
                needs to access an AWS resource.
            image_uri (str): The uri of the image to use for the processing
                jobs started by the Processor.
            command ([str]): The command to run, along with any command-line flags.
                Example: ["python3", "-v"].
            instance_count (int): The number of instances to run
                the Processing job with.
            instance_type (str): Type of EC2 instance to use for
                processing, for example, 'ml.c4.xlarge'.
            volume_size_in_gb (int): Size in GB of the EBS volume
                to use for storing data during processing (default: 30).
            volume_kms_key (str): A KMS key for the processing
                volume.
            output_kms_key (str): The KMS key id for all ProcessingOutputs.
            max_runtime_in_seconds (int): Timeout in seconds.
                After this amount of time Amazon SageMaker terminates the job
                regardless of its current status.
            base_job_name (str): Prefix for processing name. If not specified,
                the processor generates a default job name, based on the
                training image name and current timestamp.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the processor creates one
                using the default AWS configuration chain.
            env (dict): Environment variables to be passed to the processing job.
            tags ([dict]): List of tags to be passed to the processing job.
            network_config (sagemaker.network.NetworkConfig): A NetworkConfig
                object that configures network isolation, encryption of
                inter-container traffic, security group IDs, and subnets.
        """
        self._CODE_CONTAINER_BASE_PATH = "/opt/ml/processing/input/"
        self._CODE_CONTAINER_INPUT_NAME = "code"
        self.command = command

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

    def run(
        self,
        code,
        inputs=None,
        outputs=None,
        arguments=None,
        wait=True,
        logs=True,
        job_name=None,
        experiment_config=None,
    ):
        """Run a processing job with Script Mode.

        Args:
            code (str): This can be an S3 uri or a local path to either
                a directory or a file with the user's script to run.
            inputs ([sagemaker.processing.ProcessingInput]): Input files for the processing
                job. These must be provided as ProcessingInput objects.
            outputs ([str or sagemaker.processing.ProcessingOutput]): Outputs for the processing
                job. These can be specified as either a path string or a ProcessingOutput
                object.
            arguments ([str]): A list of string arguments to be passed to a
                processing job.
            wait (bool): Whether the call should wait until the job completes (default: True).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
            job_name (str): Processing job name. If not specified, the processor generates
                a default job name, based on the image name and current timestamp.
            experiment_config (dict[str, str]): Experiment management configuration.
                Dictionary contains three optional keys,
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
        """
        self._current_job_name = self._generate_current_job_name(job_name=job_name)

        user_code_s3_uri = self._handle_user_code_url(code)
        user_script_name = self._get_user_code_name(code)

        inputs_with_code = self._convert_code_and_add_to_inputs(inputs, user_code_s3_uri)

        self._set_entrypoint(self.command, user_script_name)

        normalized_inputs = self._normalize_inputs(inputs_with_code)
        normalized_outputs = self._normalize_outputs(outputs)
        self.arguments = arguments

        self.latest_job = ProcessingJob.start_new(
            processor=self,
            inputs=normalized_inputs,
            outputs=normalized_outputs,
            experiment_config=experiment_config,
        )
        self.jobs.append(self.latest_job)
        if wait:
            self.latest_job.wait(logs=logs)

    def _get_user_code_name(self, code):
        """Gets the basename of the user's code from the URL the customer provided.

        Args:
            code (str): A URL to the user's code.

        Returns:
            str: The basename of the user's code.

        """
        code_url = urlparse(code)
        return os.path.basename(code_url.path)

    def _handle_user_code_url(self, code):
        """Gets the S3 URL containing the user's code.

           Inspects the scheme the customer passed in ("s3://" for code in S3, "file://" or nothing
           for absolute or local file paths. Uploads the code to S3 if the code is a local file.

        Args:
            code (str): A URL to the customer's code.

        Returns:
            str: The S3 URL to the customer's code.

        """
        code_url = urlparse(code)
        if code_url.scheme == "s3":
            user_code_s3_uri = code
        elif code_url.scheme == "" or code_url.scheme == "file":
            # Validate that the file exists locally and is not a directory.
            if not os.path.exists(code):
                raise ValueError(
                    """code {} wasn't found. Please make sure that the file exists.
                    """.format(
                        code
                    )
                )
            if not os.path.isfile(code):
                raise ValueError(
                    """code {} must be a file, not a directory. Please pass a path to a file.
                    """.format(
                        code
                    )
                )
            user_code_s3_uri = self._upload_code(code)
        else:
            raise ValueError(
                "code {} url scheme {} is not recognized. Please pass a file path or S3 url".format(
                    code, code_url.scheme
                )
            )
        return user_code_s3_uri

    def _upload_code(self, code):
        """Uploads a code file or directory specified as a string
        and returns the S3 uri.

        Args:
            code (str): A file or directory to be uploaded to S3.

        Returns:
            str: The S3 uri of the uploaded file or directory.

        """
        desired_s3_uri = os.path.join(
            "s3://",
            self.sagemaker_session.default_bucket(),
            self._current_job_name,
            "input",
            self._CODE_CONTAINER_INPUT_NAME,
        )
        return S3Uploader.upload(
            local_path=code, desired_s3_uri=desired_s3_uri, session=self.sagemaker_session
        )

    def _convert_code_and_add_to_inputs(self, inputs, s3_uri):
        """Creates a ProcessingInput object from an S3 uri and adds it to the list of inputs.

        Args:
            inputs ([sagemaker.processing.ProcessingInput]): List of ProcessingInput objects.
            s3_uri (str): S3 uri of the input to be added to inputs.

        Returns:
            [sagemaker.processing.ProcessingInput]: A new list of ProcessingInput objects, with
                the ProcessingInput object created from s3_uri appended to the list.

        """
        code_file_input = ProcessingInput(
            source=s3_uri,
            destination=os.path.join(
                self._CODE_CONTAINER_BASE_PATH, self._CODE_CONTAINER_INPUT_NAME
            ),
            input_name=self._CODE_CONTAINER_INPUT_NAME,
        )
        return (inputs or []) + [code_file_input]

    def _set_entrypoint(self, command, user_script_name):
        """Sets the entrypoint based on the user's script and corresponding executable.

        Args:
            user_script_name (str): A filename with an extension.
        """
        user_script_location = os.path.join(
            self._CODE_CONTAINER_BASE_PATH, self._CODE_CONTAINER_INPUT_NAME, user_script_name
        )
        self.entrypoint = command + [user_script_location]


class ProcessingJob(_Job):
    """Provides functionality to start, describe, and stop processing jobs."""

    def __init__(self, sagemaker_session, job_name, inputs, outputs, output_kms_key=None):
        """Initializes a Processing job.

        Args:
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
            job_name (str): Name of the Processing job.
            inputs ([sagemaker.processing.ProcessingInput]): A list of ProcessingInput objects.
            outputs ([sagemaker.processing.ProcessingOutput]): A list of ProcessingOutput objects.
            output_kms_key (str): The output kms key associated with the job. Defaults to None
                if not provided.

        """
        self.inputs = inputs
        self.outputs = outputs
        self.output_kms_key = output_kms_key
        super(ProcessingJob, self).__init__(sagemaker_session=sagemaker_session, job_name=job_name)

    @classmethod
    def start_new(cls, processor, inputs, outputs, experiment_config):
        """Start a new processing job using the provided inputs and outputs.

        Args:
            processor (sagemaker.processing.Processor): The Processor instance
                that started the job.
            inputs ([sagemaker.processing.ProcessingInput]): A list of ProcessingInput objects.
            outputs ([sagemaker.processing.ProcessingOutput]): A list of ProcessingOutput objects.
            experiment_config (dict[str, str]): Experiment management configuration.
                Dictionary contains three optional keys,
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.

        Returns:
            sagemaker.processing.ProcessingJob: The instance of ProcessingJob created
                using the current job name.

        """
        # Initialize an empty dictionary for arguments to be passed to sagemaker_session.process.
        process_request_args = {}

        # Add arguments to the dictionary.
        process_request_args["inputs"] = [input._to_request_dict() for input in inputs]

        process_request_args["output_config"] = {
            "Outputs": [output._to_request_dict() for output in outputs]
        }
        if processor.output_kms_key is not None:
            process_request_args["output_config"]["KmsKeyId"] = processor.output_kms_key

        process_request_args["experiment_config"] = experiment_config
        process_request_args["job_name"] = processor._current_job_name

        process_request_args["resources"] = {
            "ClusterConfig": {
                "InstanceType": processor.instance_type,
                "InstanceCount": processor.instance_count,
                "VolumeSizeInGB": processor.volume_size_in_gb,
            }
        }

        if processor.volume_kms_key is not None:
            process_request_args["resources"]["ClusterConfig"][
                "VolumeKmsKeyId"
            ] = processor.volume_kms_key

        if processor.max_runtime_in_seconds is not None:
            process_request_args["stopping_condition"] = {
                "MaxRuntimeInSeconds": processor.max_runtime_in_seconds
            }
        else:
            process_request_args["stopping_condition"] = None

        process_request_args["app_specification"] = {"ImageUri": processor.image_uri}
        if processor.arguments is not None:
            process_request_args["app_specification"]["ContainerArguments"] = processor.arguments
        if processor.entrypoint is not None:
            process_request_args["app_specification"]["ContainerEntrypoint"] = processor.entrypoint

        process_request_args["environment"] = processor.env

        if processor.network_config is not None:
            process_request_args["network_config"] = processor.network_config._to_request_dict()
        else:
            process_request_args["network_config"] = None

        process_request_args["role_arn"] = processor.sagemaker_session.expand_role(processor.role)

        process_request_args["tags"] = processor.tags

        # Print the job name and the user's inputs and outputs as lists of dictionaries.
        print()
        print("Job Name: ", process_request_args["job_name"])
        print("Inputs: ", process_request_args["inputs"])
        print("Outputs: ", process_request_args["output_config"]["Outputs"])

        # Call sagemaker_session.process using the arguments dictionary.
        processor.sagemaker_session.process(**process_request_args)

        return cls(
            processor.sagemaker_session,
            processor._current_job_name,
            inputs,
            outputs,
            processor.output_kms_key,
        )

    @classmethod
    def from_processing_name(cls, sagemaker_session, processing_job_name):
        """Initializes a Processing job from a Processing job name.

        Args:
            processing_job_name (str): Name of the processing job.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.

        Returns:
            sagemaker.processing.ProcessingJob: The instance of ProcessingJob created
                using the current job name.
        """
        job_desc = sagemaker_session.describe_processing_job(job_name=processing_job_name)

        return cls(
            sagemaker_session=sagemaker_session,
            job_name=processing_job_name,
            inputs=[
                ProcessingInput(
                    source=processing_input["S3Input"]["S3Uri"],
                    destination=processing_input["S3Input"]["LocalPath"],
                    input_name=processing_input["InputName"],
                    s3_data_type=processing_input["S3Input"].get("S3DataType"),
                    s3_input_mode=processing_input["S3Input"].get("S3InputMode"),
                    s3_data_distribution_type=processing_input["S3Input"].get(
                        "S3DataDistributionType"
                    ),
                    s3_compression_type=processing_input["S3Input"].get("S3CompressionType"),
                )
                for processing_input in job_desc["ProcessingInputs"]
            ],
            outputs=[
                ProcessingOutput(
                    source=job_desc["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                        "LocalPath"
                    ],
                    destination=job_desc["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                        "S3Uri"
                    ],
                    output_name=job_desc["ProcessingOutputConfig"]["Outputs"][0]["OutputName"],
                )
            ],
            output_kms_key=job_desc["ProcessingOutputConfig"].get("KmsKeyId"),
        )

    @classmethod
    def from_processing_arn(cls, sagemaker_session, processing_job_arn):
        """Initializes a Processing job from a Processing ARN.

        Args:
            processing_job_arn (str): ARN of the processing job.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.

        Returns:
            sagemaker.processing.ProcessingJob: The instance of ProcessingJob created
                using the current job name.
        """
        processing_job_name = processing_job_arn.split(":")[5][
            len("processing-job/") :
        ]  # This is necessary while the API only vends an arn.
        return cls.from_processing_name(
            sagemaker_session=sagemaker_session, processing_job_name=processing_job_name
        )

    def _is_local_channel(self, input_url):
        """Used for Local Mode. Not yet implemented.
        Args:
            input_url (str):
        """
        raise NotImplementedError

    def wait(self, logs=True):
        if logs:
            self.sagemaker_session.logs_for_processing_job(self.job_name, wait=True)
        else:
            self.sagemaker_session.wait_for_processing_job(self.job_name)

    def describe(self):
        """Prints out a response from the DescribeProcessingJob API call."""
        return self.sagemaker_session.describe_processing_job(self.job_name)

    def stop(self):
        """Stops the processing job."""
        self.sagemaker_session.stop_processing_job(self.name)


class ProcessingInput(object):
    """Accepts parameters that specify an S3 input for a processing job and provides
    a method to turn those parameters into a dictionary."""

    def __init__(
        self,
        source,
        destination,
        input_name=None,
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type="FullyReplicated",
        s3_compression_type="None",
    ):
        """Initialize a ``ProcessingInput`` instance. ProcessingInput accepts parameters
        that specify an S3 input for a processing job and provides a method
        to turn those parameters into a dictionary.

        Args:
            source (str): The source for the input. If a local path is provided, it will
                automatically be uploaded to s3 under:
                "s3://<default-bucket-name>/<job-name>/input/<input-name>".
            destination (str): The destination of the input.
            input_name (str): The user-provided name for the input. If a name
                is not provided, one will be generated (eg. "input-1").
            s3_data_type (str): Valid options are "ManifestFile" or "S3Prefix".
            s3_input_mode (str): Valid options are "Pipe" or "File".
            s3_data_distribution_type (str): Valid options are "FullyReplicated"
                or "ShardedByS3Key".
            s3_compression_type (str): Valid options are "None" or "Gzip".
        """
        self.source = source
        self.destination = destination
        self.input_name = input_name
        self.s3_data_type = s3_data_type
        self.s3_input_mode = s3_input_mode
        self.s3_data_distribution_type = s3_data_distribution_type
        self.s3_compression_type = s3_compression_type

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        # Create the request dictionary.
        s3_input_request = {
            "InputName": self.input_name,
            "S3Input": {
                "S3Uri": self.source,
                "LocalPath": self.destination,
                "S3DataType": self.s3_data_type,
                "S3InputMode": self.s3_input_mode,
                "S3DataDistributionType": self.s3_data_distribution_type,
            },
        }

        # Check the compression type, then add it to the dictionary.
        if self.s3_compression_type == "Gzip" and self.s3_input_mode != "Pipe":
            raise ValueError("Data can only be gzipped when the input mode is Pipe.")
        if self.s3_compression_type is not None:
            s3_input_request["S3Input"]["S3CompressionType"] = self.s3_compression_type

        # Return the request dictionary.
        return s3_input_request


class ProcessingOutput(object):
    """Accepts parameters that specify an S3 output for a processing job and provides
    a method to turn those parameters into a dictionary."""

    def __init__(self, source, destination=None, output_name=None, s3_upload_mode="EndOfJob"):
        """Initialize a ``ProcessingOutput`` instance. ProcessingOutput accepts parameters that
        specify an S3 output for a processing job and provides a method to turn
        those parameters into a dictionary.

        Args:
            source (str): The source for the output.
            destination (str): The destination of the output. If a destination
                is not provided, one will be generated:
                "s3://<default-bucket-name>/<job-name>/output/<output-name>".
            output_name (str): The name of the output. If a name
                is not provided, one will be generated (eg. "output-1").
            s3_upload_mode (str): Valid options are "EndOfJob" or "Continuous".
        """
        self.source = source
        self.destination = destination
        self.output_name = output_name
        self.s3_upload_mode = s3_upload_mode

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        # Create the request dictionary.
        s3_output_request = {
            "OutputName": self.output_name,
            "S3Output": {
                "S3Uri": self.destination,
                "LocalPath": self.source,
                "S3UploadMode": self.s3_upload_mode,
            },
        }

        # Return the request dictionary.
        return s3_output_request
