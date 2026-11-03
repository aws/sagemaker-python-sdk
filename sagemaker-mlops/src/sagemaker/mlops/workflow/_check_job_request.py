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
"""Shared CreateProcessingJob request builder for the check steps.

`QualityCheckStep` and `ClarifyCheckStep` both expose their baselining job as a
`CreateProcessingJob` request dict. The two requests are built identically apart
from the inputs, the output and the default job name, so the shaping lives here
to keep the field names in one place and in sync with the SageMaker API.
"""

from __future__ import absolute_import

from typing import List

from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.shapes import ProcessingInput, ProcessingOutput

_DEFAULT_MAX_RUNTIME_IN_SECONDS = 86400
_DEFAULT_VOLUME_SIZE_IN_GB = 30


def _to_processing_input_dict(processing_input: ProcessingInput) -> dict:
    """Converts a ProcessingInput into its `ProcessingInputs` entry.

    Args:
        processing_input (ProcessingInput): The input to convert.

    Returns:
        dict: A `ProcessingInput` API entry.
    """
    input_dict = {
        "InputName": processing_input.input_name,
    }
    if processing_input.s3_input:
        input_dict["S3Input"] = {
            "S3Uri": processing_input.s3_input.s3_uri,
            "LocalPath": processing_input.s3_input.local_path,
            "S3DataType": getattr(processing_input.s3_input, "s3_data_type", "S3Prefix"),
            "S3InputMode": getattr(processing_input.s3_input, "s3_input_mode", "File"),
        }
    return input_dict


def _build_check_job_request(
    processing_inputs: List[ProcessingInput],
    processing_output: ProcessingOutput,
    processor,
    check_job_config,
    default_job_name: str,
) -> RequestType:
    """Builds the CreateProcessingJob request for a check step's baselining job.

    Args:
        processing_inputs (List[ProcessingInput]): The baselining job inputs.
        processing_output (ProcessingOutput): The baselining job output.
        processor (sagemaker.core.processing.Processor): The baselining processor.
        check_job_config (CheckJobConfig): The check job config of the step.
        default_job_name (str): The job name to fall back to when the processor
            has not generated one yet.

    Returns:
        RequestType: A `CreateProcessingJob` request dict.
    """
    # `KmsKeyId` belongs on ProcessingOutputConfig, not on ProcessingS3Output -
    # the API applies it to every output. ProcessingS3Output accepts only S3Uri,
    # LocalPath and S3UploadMode.
    output_config = {
        "Outputs": [
            {
                "OutputName": processing_output.output_name,
                "S3Output": {
                    "S3Uri": processing_output.s3_output.s3_uri,
                    "LocalPath": processing_output.s3_output.local_path,
                    "S3UploadMode": processing_output.s3_output.s3_upload_mode,
                },
            }
        ],
    }
    if check_job_config.output_kms_key:
        output_config["KmsKeyId"] = check_job_config.output_kms_key

    cluster_config = {
        "InstanceCount": processor.instance_count,
        "InstanceType": processor.instance_type,
        "VolumeSizeInGB": getattr(processor, "volume_size_in_gb", _DEFAULT_VOLUME_SIZE_IN_GB),
    }
    if check_job_config.volume_kms_key:
        cluster_config["VolumeKmsKeyId"] = check_job_config.volume_kms_key

    request_dict = {
        "ProcessingInputs": [
            _to_processing_input_dict(processing_input)
            for processing_input in processing_inputs or []
        ],
        "ProcessingOutputConfig": output_config,
        "ProcessingJobName": processor._current_job_name or default_job_name,
        "ProcessingResources": {
            "ClusterConfig": cluster_config,
        },
        "AppSpecification": {
            "ImageUri": processor.image_uri,
        },
        "RoleArn": processor.role,
        "StoppingCondition": {
            "MaxRuntimeInSeconds": getattr(processor, "max_runtime_in_seconds", None)
            or _DEFAULT_MAX_RUNTIME_IN_SECONDS
        },
    }

    # Add optional fields if they exist
    if processor.env:
        request_dict["Environment"] = processor.env
    if processor.network_config:
        request_dict["NetworkConfig"] = processor.network_config
    if processor.entrypoint:
        request_dict["AppSpecification"]["ContainerEntrypoint"] = processor.entrypoint
    if processor.arguments:
        request_dict["AppSpecification"]["ContainerArguments"] = processor.arguments
    if check_job_config.tags:
        request_dict["Tags"] = check_job_config.tags

    return request_dict
