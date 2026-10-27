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
"""Unit tests for the shared check-step CreateProcessingJob request builder."""

from __future__ import absolute_import

from unittest.mock import Mock

import botocore.session
import pytest
from botocore.validate import ParamValidator

from sagemaker.mlops.workflow._check_job_request import _build_check_job_request

_OUTPUT_KMS_KEY = "arn:aws:kms:us-east-1:123456789012:key/output-key-id"
_VOLUME_KMS_KEY = "arn:aws:kms:us-east-1:123456789012:key/volume-key-id"
_TAGS = [{"Key": "test", "Value": "check-step"}]


@pytest.fixture
def processing_input():
    processing_input = Mock()
    processing_input.input_name = "baseline_dataset_input"
    processing_input.s3_input = Mock()
    processing_input.s3_input.s3_uri = "s3://bucket/input/data.csv"
    processing_input.s3_input.local_path = "/opt/ml/processing/input"
    processing_input.s3_input.s3_data_type = "S3Prefix"
    processing_input.s3_input.s3_input_mode = "File"
    return processing_input


@pytest.fixture
def processing_output():
    processing_output = Mock()
    processing_output.output_name = "quality_check_output"
    processing_output.s3_output = Mock()
    processing_output.s3_output.s3_uri = "s3://bucket/output"
    processing_output.s3_output.local_path = "/opt/ml/processing/output"
    processing_output.s3_output.s3_upload_mode = "EndOfJob"
    return processing_output


@pytest.fixture
def processor():
    processor = Mock()
    processor._current_job_name = "baseline-job-2026-01-01"
    processor.instance_count = 1
    processor.instance_type = "ml.m5.xlarge"
    processor.volume_size_in_gb = 30
    processor.image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/monitor:latest"
    processor.role = "arn:aws:iam::123456789012:role/SageMakerRole"
    processor.max_runtime_in_seconds = 3600
    processor.env = None
    processor.network_config = None
    processor.entrypoint = None
    processor.arguments = None
    return processor


def _check_job_config(output_kms_key=None, volume_kms_key=None, tags=None):
    check_job_config = Mock()
    check_job_config.output_kms_key = output_kms_key
    check_job_config.volume_kms_key = volume_kms_key
    check_job_config.tags = tags
    return check_job_config


def _build(processing_input, processing_output, processor, check_job_config):
    return _build_check_job_request(
        processing_inputs=[processing_input],
        processing_output=processing_output,
        processor=processor,
        check_job_config=check_job_config,
        default_job_name="baseline-job",
    )


def test_request_matches_create_processing_job_shape(
    processing_input, processing_output, processor
):
    """The request validates against the CreateProcessingJob service model.

    This is the guard for issue #6495: `KmsKeyId` on ProcessingS3Output and a
    `Tags` list in the wrong place are both rejected here.
    """
    request = _build(
        processing_input,
        processing_output,
        processor,
        _check_job_config(
            output_kms_key=_OUTPUT_KMS_KEY,
            volume_kms_key=_VOLUME_KMS_KEY,
            tags=_TAGS,
        ),
    )

    input_shape = (
        botocore.session.get_session()
        .get_service_model("sagemaker")
        .operation_model("CreateProcessingJob")
        .input_shape
    )
    errors = ParamValidator().validate(request, input_shape)
    assert not errors.has_errors(), errors.generate_report()


def test_output_kms_key_on_output_config(processing_input, processing_output, processor):
    """output_kms_key lands on ProcessingOutputConfig, not on ProcessingS3Output."""
    request = _build(
        processing_input,
        processing_output,
        processor,
        _check_job_config(output_kms_key=_OUTPUT_KMS_KEY),
    )

    output_config = request["ProcessingOutputConfig"]
    assert output_config["KmsKeyId"] == _OUTPUT_KMS_KEY
    assert "KmsKeyId" not in output_config["Outputs"][0]["S3Output"]


def test_tags_on_request_root(processing_input, processing_output, processor):
    """Tags land at the root of the request."""
    request = _build(processing_input, processing_output, processor, _check_job_config(tags=_TAGS))

    assert request["Tags"] == _TAGS


@pytest.mark.parametrize("tags", [None, []])
def test_tags_omitted_when_unset(processing_input, processing_output, processor, tags):
    """Tags is omitted rather than sent empty."""
    request = _build(processing_input, processing_output, processor, _check_job_config(tags=tags))

    assert "Tags" not in request


def test_inputs_and_output_are_shaped(processing_input, processing_output, processor):
    """Inputs and the output keep their API field names."""
    request = _build(processing_input, processing_output, processor, _check_job_config())

    assert request["ProcessingInputs"] == [
        {
            "InputName": "baseline_dataset_input",
            "S3Input": {
                "S3Uri": "s3://bucket/input/data.csv",
                "LocalPath": "/opt/ml/processing/input",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        }
    ]
    assert request["ProcessingOutputConfig"] == {
        "Outputs": [
            {
                "OutputName": "quality_check_output",
                "S3Output": {
                    "S3Uri": "s3://bucket/output",
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ],
    }


def test_input_without_s3_input_is_name_only(processing_output, processor):
    """An input with no s3_input contributes only its name."""
    processing_input = Mock()
    processing_input.input_name = "no_s3_input"
    processing_input.s3_input = None

    request = _build(processing_input, processing_output, processor, _check_job_config())

    assert request["ProcessingInputs"] == [{"InputName": "no_s3_input"}]


def test_empty_inputs_are_allowed(processing_output, processor):
    """A step with no inputs produces an empty ProcessingInputs list."""
    request = _build_check_job_request(
        processing_inputs=[],
        processing_output=processing_output,
        processor=processor,
        check_job_config=_check_job_config(),
        default_job_name="baseline-job",
    )

    assert request["ProcessingInputs"] == []


def test_default_job_name_used_when_processor_has_none(
    processing_input, processing_output, processor
):
    """The default job name is used when the processor has not generated one."""
    processor._current_job_name = None

    request = _build(processing_input, processing_output, processor, _check_job_config())

    assert request["ProcessingJobName"] == "baseline-job"


def test_optional_processor_fields_are_added(processing_input, processing_output, processor):
    """Environment, NetworkConfig and container overrides are added when set."""
    processor.env = {"KEY": "value"}
    processor.network_config = {"EnableNetworkIsolation": True}
    processor.entrypoint = ["python3"]
    processor.arguments = ["--flag"]

    request = _build(processing_input, processing_output, processor, _check_job_config())

    assert request["Environment"] == {"KEY": "value"}
    assert request["NetworkConfig"] == {"EnableNetworkIsolation": True}
    assert request["AppSpecification"]["ContainerEntrypoint"] == ["python3"]
    assert request["AppSpecification"]["ContainerArguments"] == ["--flag"]


def test_defaults_when_processor_values_missing(processing_input, processing_output, processor):
    """Volume size and max runtime fall back to their defaults."""
    processor.max_runtime_in_seconds = None
    del processor.volume_size_in_gb

    request = _build(processing_input, processing_output, processor, _check_job_config())

    assert request["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"] == 30
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 86400
