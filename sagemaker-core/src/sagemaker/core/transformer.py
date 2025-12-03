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
"""Placeholder docstring"""
from __future__ import absolute_import

from typing import Union, Optional, Dict
import logging
import time
import json

from botocore import exceptions
from sagemaker.core.helper.session_helper import Session
from sagemaker.core import s3
from sagemaker.core._studio import _append_project_tags
from sagemaker.core.config import (
    TRANSFORM_JOB_ENVIRONMENT_PATH,
    TRANSFORM_OUTPUT_KMS_KEY_ID_PATH,
    TRANSFORM_JOB_KMS_KEY_ID_PATH,
    TRANSFORM_RESOURCES_VOLUME_KMS_KEY_ID_PATH,
    SAGEMAKER,
    TRANSFORM_JOB,
    TAGS,
    KMS_KEY_ID,
    VOLUME_KMS_KEY_ID,
    TRANSFORM_JOB_VOLUME_KMS_KEY_ID_PATH,
)

from sagemaker.core.shapes import BatchDataCaptureConfig
from sagemaker.core.resources import TransformJob, Model
from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.core.workflow.functions import Join
from sagemaker.core.workflow.pipeline_context import runnable_by_pipeline, PipelineSession
from sagemaker.core.workflow import is_pipeline_variable
from sagemaker.core.workflow.execution_variables import ExecutionVariables
from sagemaker.core.common_utils import (
    base_name_from_image,
    name_from_base,
    check_and_get_run_experiment_config,
    resolve_value_from_config,
    resolve_class_attribute_from_config,
    resolve_nested_dict_value_from_config,
    format_tags,
    Tags,
    _wait_until,
    _flush_log_streams,
    _logs_init,
    LogState,
    _check_job_status,
    _get_initial_job_state,
)
from sagemaker.core.utils.utils import serialize
from sagemaker.core.config.config_utils import _append_sagemaker_config_tags

logger = LOGGER = logging.getLogger("sagemaker")


class Transformer(object):
    """A class for handling creating and interacting with Amazon SageMaker transform jobs."""

    JOB_CLASS_NAME = "transform-job"

    def __init__(
        self,
        model_name: Union[str, PipelineVariable],
        instance_count: Union[int, PipelineVariable],
        instance_type: Union[str, PipelineVariable],
        strategy: Optional[Union[str, PipelineVariable]] = None,
        assemble_with: Optional[Union[str, PipelineVariable]] = None,
        output_path: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        accept: Optional[Union[str, PipelineVariable]] = None,
        max_concurrent_transforms: Optional[Union[int, PipelineVariable]] = None,
        max_payload: Optional[Union[int, PipelineVariable]] = None,
        tags: Optional[Tags] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        base_transform_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
    ):
        """Initialize a ``Transformer``.

        Args:
            model_name (str or PipelineVariable): Name of the SageMaker model being
                used for the transform job.
            instance_count (int or PipelineVariable): Number of EC2 instances to use.
            instance_type (str or PipelineVariable): Type of EC2 instance to use, for example,
                'ml.c4.xlarge'.
            strategy (str or PipelineVariable): The strategy used to decide how to batch records
                in a single request (default: None). Valid values: 'MultiRecord'
                and 'SingleRecord'.
            assemble_with (str or PipelineVariable): How the output is assembled (default: None).
                Valid values: 'Line' or 'None'.
            output_path (str or PipelineVariable): S3 location for saving the transform result. If
                not specified, results are stored to a default bucket.
            output_kms_key (str or PipelineVariable): Optional. KMS key ID for encrypting the
                transform output (default: None).
            accept (str or PipelineVariable): The accept header passed by the client to
                the inference endpoint. If it is supported by the endpoint,
                it will be the format of the batch transform output.
            max_concurrent_transforms (int or PipelineVariable): The maximum number of HTTP requests
                to be made to each individual transform container at one time.
            max_payload (int or PipelineVariable): Maximum size of the payload in a single HTTP
                request to the container in MB.
            tags (Optional[Tags]): Tags for labeling a transform job (default: None).
                For more, see the SageMaker API documentation for
                `Tag <https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html>`_.
            env (dict[str, str] or dict[str, PipelineVariable]): Environment variables to be set
                for use during the transform job (default: None).
            base_transform_job_name (str): Prefix for the transform job when the
                :meth:`~sagemaker.transformer.Transformer.transform` method
                launches. If not specified, a default prefix will be generated
                based on the training image name that was used to train the
                model associated with the transform job.
            sagemaker_session (sagemaker.core.helper.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed.
            volume_kms_key (str or PipelineVariable): Optional. KMS key ID for encrypting
                the volume attached to the ML compute instance (default: None).
        """
        self.model_name = model_name
        self.strategy = strategy

        self.output_path = output_path
        self.accept = accept
        self.assemble_with = assemble_with

        self.instance_count = instance_count
        self.instance_type = instance_type

        self.max_concurrent_transforms = max_concurrent_transforms
        self.max_payload = max_payload
        self.tags = format_tags(tags)

        self.base_transform_job_name = base_transform_job_name
        self._current_job_name = None
        self.latest_transform_job = None
        self._reset_output_path = False

        self.sagemaker_session = sagemaker_session or Session()
        self.volume_kms_key = resolve_value_from_config(
            volume_kms_key,
            TRANSFORM_RESOURCES_VOLUME_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.output_kms_key = resolve_value_from_config(
            output_kms_key,
            TRANSFORM_OUTPUT_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        self.env = resolve_value_from_config(
            env,
            TRANSFORM_JOB_ENVIRONMENT_PATH,
            sagemaker_session=self.sagemaker_session,
        )

    @runnable_by_pipeline
    def transform(
        self,
        data: Union[str, PipelineVariable],
        data_type: Union[str, PipelineVariable] = "S3Prefix",
        content_type: Optional[Union[str, PipelineVariable]] = None,
        compression_type: Optional[Union[str, PipelineVariable]] = None,
        split_type: Optional[Union[str, PipelineVariable]] = None,
        job_name: Optional[str] = None,
        input_filter: Optional[Union[str, PipelineVariable]] = None,
        output_filter: Optional[Union[str, PipelineVariable]] = None,
        join_source: Optional[Union[str, PipelineVariable]] = None,
        experiment_config: Optional[Dict[str, str]] = None,
        model_client_config: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        batch_data_capture_config: BatchDataCaptureConfig = None,
        wait: bool = True,
        logs: bool = True,
    ):
        """Start a new transform job.

        Args:
            data (str or PipelineVariable): Input data location in S3.
            data_type (str or PipelineVariable): What the S3 location defines (default: 'S3Prefix').
                Valid values:

                * 'S3Prefix' - the S3 URI defines a key name prefix. All objects with this prefix
                    will be used as inputs for the transform job.

                * 'ManifestFile' - the S3 URI points to a single manifest file listing each S3
                    object to use as an input for the transform job.

            content_type (str or PipelineVariable): MIME type of the input data (default: None).
            compression_type (str or PipelineVariable): Compression type of the input data, if
                compressed (default: None). Valid values: 'Gzip', None.
            split_type (str or PipelineVariable): The record delimiter for the input object
                (default: 'None'). Valid values: 'None', 'Line', 'RecordIO', and
                'TFRecord'.
            job_name (str): job name (default: None). If not specified, one will
                be generated.
            input_filter (str or PipelineVariable): A JSONPath to select a portion of the input to
                pass to the algorithm container for inference. If you omit the
                field, it gets the value '$', representing the entire input.
                For CSV data, each row is taken as a JSON array,
                so only index-based JSONPaths can be applied, e.g. $[0], $[1:].
                CSV data should follow the `RFC format <https://tools.ietf.org/html/rfc4180>`_.
                See `Supported JSONPath Operators
                <https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html#data-processing-operators>`_
                for a table of supported JSONPath operators.
                For more information, see the SageMaker API documentation for
                `CreateTransformJob
                <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
                Some examples: "$[1:]", "$.features" (default: None).
            output_filter (str or PipelineVariable): A JSONPath to select a portion of the
                joined/original output to return as the output.
                For more information, see the SageMaker API documentation for
                `CreateTransformJob
                <https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateTransformJob.html>`_.
                Some examples: "$[1:]", "$.prediction" (default: None).
            join_source (str or PipelineVariable): The source of data to be joined to the transform
                output. It can be set to 'Input' meaning the entire input record
                will be joined to the inference result. You can use OutputFilter
                to select the useful portion before uploading to S3. (default:
                None). Valid values: Input, None.
            experiment_config (dict[str, str]): Experiment management configuration.
                Optionally, the dict can contain three keys:
                'ExperimentName', 'TrialName', and 'TrialComponentDisplayName'.
                The behavior of setting these keys is as follows:
                * If `ExperimentName` is supplied but `TrialName` is not a Trial will be
                automatically created and the job's Trial Component associated with the Trial.
                * If `TrialName` is supplied and the Trial already exists the job's Trial Component
                will be associated with the Trial.
                * If both `ExperimentName` and `TrialName` are not supplied the trial component
                will be unassociated.
                * `TrialComponentDisplayName` is used for display in Studio.
                * Both `ExperimentName` and `TrialName` will be ignored if the Transformer instance
                is built with :class:`~sagemaker.workflow.pipeline_context.PipelineSession`.
                However, the value of `TrialComponentDisplayName` is honored for display in Studio.
            model_client_config (dict[str, str] or dict[str, PipelineVariable]): Model
                configuration. Dictionary contains two optional keys,
                'InvocationsTimeoutInSeconds', and 'InvocationsMaxRetries'.
                (default: ``None``).
            batch_data_capture_config (BatchDataCaptureConfig): Configuration object which
                specifies the configurations related to the batch data capture for the transform job
                (default: ``None``).
            batch_data_capture_config (BatchDataCaptureConfig): Configuration object which
                specifies the configurations related to the batch data capture for the transform job
                (default: ``None``).
            wait (bool): Whether the call should wait until the job completes
                (default: ``True``).
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is ``True`` (default: ``True``).
        Returns:
            None or pipeline step arguments in case the Transformer instance is built with
            :class:`~sagemaker.workflow.pipeline_context.PipelineSession`
        """
        from sagemaker.core.workflow.utilities import _pipeline_config

        local_mode = self.sagemaker_session.local_mode
        if not local_mode and not is_pipeline_variable(data) and not data.startswith("s3://"):
            raise ValueError("Invalid S3 URI: {}".format(data))

        if job_name is not None:
            self._current_job_name = job_name
        else:
            base_name = self.base_transform_job_name

            if base_name is None:
                base_name = (
                    "transform-job"
                    if is_pipeline_variable(self.model_name)
                    else self._retrieve_base_name()
                )

            self._current_job_name = name_from_base(base_name)

        if self.output_path is None or self._reset_output_path is True:
            if _pipeline_config:
                self.output_path = Join(
                    on="/",
                    values=[
                        "s3:/",
                        self.sagemaker_session.default_bucket(),
                        *(
                            # don't include default_bucket_prefix if it is None or ""
                            [self.sagemaker_session.default_bucket_prefix]
                            if self.sagemaker_session.default_bucket_prefix
                            else []
                        ),
                        _pipeline_config.pipeline_name,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                        _pipeline_config.step_name,
                    ],
                )
            else:
                self.output_path = s3.s3_path_join(
                    "s3://",
                    self.sagemaker_session.default_bucket(),
                    self.sagemaker_session.default_bucket_prefix,
                    self._current_job_name,
                )
            self._reset_output_path = True

        experiment_config = check_and_get_run_experiment_config(experiment_config)

        batch_data_capture_config = resolve_class_attribute_from_config(
            None,
            batch_data_capture_config,
            "kms_key_id",
            TRANSFORM_JOB_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )

        transform_args = self._get_transform_args(
            data,
            data_type,
            content_type,
            compression_type,
            split_type,
            input_filter,
            output_filter,
            join_source,
            experiment_config,
            model_client_config,
            batch_data_capture_config,
        )

        # Apply config resolution and create transform job
        tags = _append_project_tags(format_tags(transform_args["tags"]))
        tags = _append_sagemaker_config_tags(
            self.sagemaker_session, tags, "{}.{}.{}".format(SAGEMAKER, TRANSFORM_JOB, TAGS)
        )

        batch_data_capture_config = resolve_class_attribute_from_config(
            None,
            transform_args["batch_data_capture_config"],
            "kms_key_id",
            TRANSFORM_JOB_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )

        output_config = resolve_nested_dict_value_from_config(
            transform_args["output_config"],
            [KMS_KEY_ID],
            TRANSFORM_OUTPUT_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )

        resource_config = resolve_nested_dict_value_from_config(
            transform_args["resource_config"],
            [VOLUME_KMS_KEY_ID],
            TRANSFORM_JOB_VOLUME_KMS_KEY_ID_PATH,
            sagemaker_session=self.sagemaker_session,
        )

        env = resolve_value_from_config(
            direct_input=transform_args["env"],
            config_path=TRANSFORM_JOB_ENVIRONMENT_PATH,
            default_value=None,
            sagemaker_session=self.sagemaker_session,
        )

        transform_request = self._get_transform_request(
            job_name=transform_args["job_name"],
            model_name=transform_args["model_name"],
            strategy=transform_args["strategy"],
            max_concurrent_transforms=transform_args["max_concurrent_transforms"],
            max_payload=transform_args["max_payload"],
            env=env,
            input_config=transform_args["input_config"],
            output_config=output_config,
            resource_config=resource_config,
            experiment_config=transform_args["experiment_config"],
            tags=tags,
            data_processing=transform_args["data_processing"],
            model_client_config=transform_args["model_client_config"],
            batch_data_capture_config=batch_data_capture_config,
        )

        # convert Unassigned() type in sagemaker-core to None
        serialized_request = serialize(transform_request)

        if isinstance(self.sagemaker_session, PipelineSession):
            self.sagemaker_session._intercept_create_request(serialized_request, None, "transform")
            return

        def submit(request):
            logger.info("Creating transform job with name: %s", transform_args["job_name"])
            logger.debug("Transform request: %s", json.dumps(request, indent=4))
            self.sagemaker_session.sagemaker_client.create_transform_job(**request)

        self.sagemaker_session._intercept_create_request(serialized_request, submit, "transform")

        from sagemaker.core.utils.code_injection.codec import transform as transform_util

        transformed = transform_util(serialized_request, "CreateTransformJobRequest")
        self.latest_transform_job = TransformJob(**transformed)

        if wait:
            self.latest_transform_job.wait(logs=logs)

    def delete_model(self):
        """Delete the corresponding SageMaker model for this Transformer."""
        model = Model.get(model_name=self.model_name, session=self.sagemaker_session.boto_session)
        if model:
            model.delete()

    def _retrieve_base_name(self):
        """Placeholder docstring"""
        image_uri = self._retrieve_image_uri()

        if image_uri:
            return base_name_from_image(image_uri, default_base_name=Transformer.JOB_CLASS_NAME)

        return self.model_name

    def _retrieve_image_uri(self):
        """Placeholder docstring"""
        try:
            model = Model.get(
                model_name=self.model_name,
                session=self.sagemaker_session.boto_session,
                region=self.sagemaker_session.boto_region_name,
            )
            if not model:
                return None
            model_desc = model.__dict__

            primary_container = getattr(model_desc, "primary_container", None)
            if primary_container:
                return getattr(primary_container, "image", None)

            containers = getattr(model_desc, "containers", None)
            if containers:
                return getattr(containers[0], "image", None)

            return None

        except exceptions.ClientError:
            raise ValueError(
                "Failed to fetch model information for %s. "
                "Please ensure that the model exists. "
                "Local instance types require locally created models." % self.model_name
            )

    def wait(self, logs=True):
        """Placeholder docstring"""
        self._ensure_last_transform_job()
        self.latest_transform_job.wait(logs=logs)

    def stop_transform_job(self, wait=True):
        """Stop latest running batch transform job."""
        self._ensure_last_transform_job()
        self.latest_transform_job.stop()
        if wait:
            self.latest_transform_job.wait()

    def _ensure_last_transform_job(self):
        """Placeholder docstring"""
        if self.latest_transform_job is None:
            raise ValueError("No transform job available")

    @classmethod
    def attach(cls, transform_job_name, sagemaker_session=None):
        """Attach an existing transform job to a new Transformer instance

        Args:
            transform_job_name (str): Name for the transform job to be attached.
            sagemaker_session (sagemaker.core.helper.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one will be created using
                the default AWS configuration chain.

        Returns:
            sagemaker.transformer.Transformer: The Transformer instance with the
            specified transform job attached.
        """
        sagemaker_session = sagemaker_session or Session()

        transform_job = TransformJob.get(
            transform_job_name=transform_job_name, session=sagemaker_session
        )
        if not transform_job:
            raise ValueError(f"Transform job {transform_job_name} not found")
        job_details = transform_job.__dict__
        init_params = cls._prepare_init_params_from_job_description(job_details)
        transformer = cls(sagemaker_session=sagemaker_session, **init_params)
        transformer.latest_transform_job = TransformJob.get(
            transform_job_name=init_params["base_transform_job_name"], session=sagemaker_session
        )

        return transformer

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        """Convert the transform job description to init params.

        It can be handled by the class constructor.

        Args:
            job_details (dict): the returned job details from a
                describe_transform_job API call.

        Returns:
            dict: The transformed init_params
        """
        init_params = dict()

        init_params["model_name"] = job_details.get("model_name")
        if job_details.get("transform_resources"):
            init_params["instance_count"] = job_details["transform_resources"].instance_count
            init_params["instance_type"] = job_details["transform_resources"].instance_type
            init_params["volume_kms_key"] = getattr(
                job_details["transform_resources"], "volume_kms_key_id", None
            )
        init_params["strategy"] = job_details.get("batch_strategy")
        if job_details.get("transform_output"):
            init_params["assemble_with"] = getattr(
                job_details["transform_output"], "assemble_with", None
            )
            init_params["output_path"] = job_details["transform_output"].s3_output_path
            init_params["output_kms_key"] = getattr(
                job_details["transform_output"], "kms_key_id", None
            )
            init_params["accept"] = getattr(job_details["transform_output"], "accept", None)
        init_params["max_concurrent_transforms"] = job_details.get("max_concurrent_transforms")
        init_params["max_payload"] = job_details.get("max_payload_in_mb")
        init_params["base_transform_job_name"] = job_details.get("transform_job_name")

        return init_params

    def _get_transform_args(
        self,
        data,
        data_type,
        content_type,
        compression_type,
        split_type,
        input_filter,
        output_filter,
        join_source,
        experiment_config,
        model_client_config,
        batch_data_capture_config,
    ):
        """Get transform job arguments."""
        config = self._load_config(data, data_type, content_type, compression_type, split_type)
        data_processing = self._prepare_data_processing(input_filter, output_filter, join_source)

        transform_args = config.copy()
        transform_args.update(
            {
                "job_name": self._current_job_name,
                "model_name": self.model_name,
                "strategy": self.strategy,
                "max_concurrent_transforms": self.max_concurrent_transforms,
                "max_payload": self.max_payload,
                "env": self.env,
                "experiment_config": experiment_config,
                "model_client_config": model_client_config,
                "tags": self.tags,
                "data_processing": data_processing,
                "batch_data_capture_config": batch_data_capture_config,
            }
        )

        return transform_args

    def _load_config(self, data, data_type, content_type, compression_type, split_type):
        """Load configuration for transform job."""
        input_config = self._format_inputs_to_input_config(
            data, data_type, content_type, compression_type, split_type
        )

        output_config = self._prepare_output_config(
            self.output_path,
            self.output_kms_key,
            self.assemble_with,
            self.accept,
        )

        resource_config = self._prepare_resource_config(
            self.instance_count, self.instance_type, self.volume_kms_key
        )

        return {
            "input_config": input_config,
            "output_config": output_config,
            "resource_config": resource_config,
        }

    def _format_inputs_to_input_config(
        self, data, data_type, content_type, compression_type, split_type
    ):
        """Format inputs to input config."""
        from sagemaker.core.shapes import TransformDataSource, TransformS3DataSource

        config = {
            "data_source": TransformDataSource(
                s3_data_source=TransformS3DataSource(s3_data_type=data_type, s3_uri=data)
            )
        }

        if content_type is not None:
            config["content_type"] = content_type

        if compression_type is not None:
            config["compression_type"] = compression_type

        if split_type is not None:
            config["split_type"] = split_type

        return config

    def _prepare_output_config(self, s3_path, kms_key_id, assemble_with, accept):
        """Prepare output config."""
        config = {"s3_output_path": s3_path}

        if kms_key_id is not None:
            config["kms_key_id"] = kms_key_id

        if assemble_with is not None:
            config["assemble_with"] = assemble_with

        if accept is not None:
            config["accept"] = accept

        return config

    def _prepare_resource_config(self, instance_count, instance_type, volume_kms_key):
        """Prepare resource config."""
        config = {"instance_count": instance_count, "instance_type": instance_type}

        if volume_kms_key is not None:
            config["volume_kms_key_id"] = volume_kms_key

        return config

    def _prepare_data_processing(self, input_filter, output_filter, join_source):
        """Prepare data processing config."""
        from sagemaker.core.shapes import DataProcessing

        if input_filter is None and output_filter is None and join_source is None:
            return None

        return DataProcessing(
            input_filter=input_filter, output_filter=output_filter, join_source=join_source
        )

    def _get_transform_request(
        self,
        job_name,
        model_name,
        strategy,
        max_concurrent_transforms,
        max_payload,
        env,
        input_config,
        output_config,
        resource_config,
        experiment_config,
        tags,
        data_processing,
        model_client_config=None,
        batch_data_capture_config: BatchDataCaptureConfig = None,
    ):
        """Construct a dict for creating an Amazon SageMaker transform job."""
        from sagemaker.core.shapes import TransformInput, TransformOutput, TransformResources

        transform_request = {
            "TransformJobName": job_name,
            "ModelName": model_name,
            "TransformInput": TransformInput(**input_config),
            "TransformOutput": TransformOutput(**output_config),
            "TransformResources": TransformResources(**resource_config),
        }

        if strategy is not None:
            transform_request["BatchStrategy"] = strategy

        if max_concurrent_transforms is not None:
            transform_request["MaxConcurrentTransforms"] = max_concurrent_transforms

        if max_payload is not None:
            transform_request["MaxPayloadInMB"] = max_payload

        if env is not None:
            transform_request["Environment"] = env

        if tags is not None:
            transform_request["Tags"] = tags

        if data_processing is not None:
            transform_request["DataProcessing"] = data_processing

        if experiment_config and len(experiment_config) > 0:
            transform_request["ExperimentConfig"] = experiment_config

        if model_client_config and len(model_client_config) > 0:
            transform_request["ModelClientConfig"] = model_client_config

        if batch_data_capture_config is not None:
            transform_request["DataCaptureConfig"] = batch_data_capture_config

        return transform_request


def logs_for_transform_job(sagemaker_session, job_name, wait=False, poll=10):
    """Display logs for a given training job, optionally tailing them until job is complete.

    If the output is a tty or a Jupyter cell, it will be color-coded
    based on which instance the log entry is from.

    Args:
        job_name (str): Name of the transform job to display the logs for.
        wait (bool): Whether to keep looking for new log entries until the job completes
            (default: False).
        poll (int): The interval in seconds between polling for new log entries and job
            completion (default: 5).

    Raises:
        ValueError: If the transform job fails.
    """

    description = _wait_until(
        lambda: TransformJob.get(transform_job_name=job_name, session=sagemaker_session).__dict__,
        poll,
    )

    instance_count, stream_names, positions, client, log_group, dot, color_wrap = _logs_init(
        sagemaker_session.boto_session, description, job="Transform"
    )

    state = _get_initial_job_state(description, "TransformJobStatus", wait)

    # The loop below implements a state machine that alternates between checking the job status
    # and reading whatever is available in the logs at this point. Note, that if we were
    # called with wait == False, we never check the job status.
    #
    # If wait == TRUE and job is not completed, the initial state is TAILING
    # If wait == FALSE, the initial state is COMPLETE (doesn't matter if the job really is
    # complete).
    #
    # The state table:
    #
    # STATE               ACTIONS                        CONDITION             NEW STATE
    # ----------------    ----------------               -----------------     ----------------
    # TAILING             Read logs, Pause, Get status   Job complete          JOB_COMPLETE
    #                                                    Else                  TAILING
    # JOB_COMPLETE        Read logs, Pause               Any                   COMPLETE
    # COMPLETE            Read logs, Exit                                      N/A
    #
    # Notes:
    # - The JOB_COMPLETE state forces us to do an extra pause and read any items that got to
    #   Cloudwatch after the job was marked complete.
    last_describe_job_call = time.time()
    while True:
        _flush_log_streams(
            stream_names,
            instance_count,
            client,
            log_group,
            job_name,
            positions,
            dot,
            color_wrap,
        )
        if state == LogState.COMPLETE:
            break

        time.sleep(poll)

        if state == LogState.JOB_COMPLETE:
            state = LogState.COMPLETE
        elif time.time() - last_describe_job_call >= 30:
            transform_job = TransformJob.get(transform_job_name=job_name, session=sagemaker_session)
            description = transform_job.__dict__ if transform_job else None
            last_describe_job_call = time.time()

            status = description["TransformJobStatus"]

            if status in ("Completed", "Failed", "Stopped"):
                print()
                state = LogState.JOB_COMPLETE

    if wait:
        _check_job_status(job_name, description, "TransformJobStatus")
        if dot:
            print()
