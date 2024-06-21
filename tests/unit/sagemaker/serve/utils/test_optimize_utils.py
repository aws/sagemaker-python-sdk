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
from __future__ import absolute_import

from unittest.mock import Mock

import pytest

from sagemaker.enums import Tag
from sagemaker.serve.utils.optimize_utils import (
    _generate_optimized_model,
    _is_inferentia_or_trainium,
    _update_environment_variables,
    _is_image_compatible_with_optimization_job,
    _extract_speculative_draft_model_provider,
    _validate_optimization_inputs,
    _extracts_and_validates_speculative_model_source,
)

mock_optimization_job_output = {
    "OptimizationJobArn": "arn:aws:sagemaker:us-west-2:312206380606:"
    "optimization-job/modelbuilderjob-6b09ffebeb0741b8a28b85623fd9c968",
    "OptimizationJobStatus": "COMPLETED",
    "OptimizationJobName": "modelbuilderjob-6b09ffebeb0741b8a28b85623fd9c968",
    "ModelSource": {
        "S3": {
            "S3Uri": "s3://jumpstart-private-cache-alpha-us-west-2/meta-textgeneration/"
            "meta-textgeneration-llama-3-8b/artifacts/inference-prepack/v1.0.1/"
        }
    },
    "OptimizationEnvironment": {
        "ENDPOINT_SERVER_TIMEOUT": "3600",
        "HF_MODEL_ID": "/opt/ml/model",
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_ENV": "1",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
    },
    "DeploymentInstanceType": "ml.g5.48xlarge",
    "OptimizationConfigs": [
        {
            "ModelQuantizationConfig": {
                "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124",
                "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"},
            }
        }
    ],
    "OutputConfig": {
        "S3OutputLocation": "s3://dont-delete-ss-jarvis-integ-test-312206380606-us-west-2/"
    },
    "OptimizationOutput": {
        "RecommendedInferenceImage": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124"
    },
    "RoleArn": "arn:aws:iam::312206380606:role/service-role/AmazonSageMaker-ExecutionRole-20230707T131628",
    "StoppingCondition": {"MaxRuntimeInSeconds": 36000},
    "ResponseMetadata": {
        "RequestId": "17ae151f-b51d-4194-8ba9-edbba068c90b",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "17ae151f-b51d-4194-8ba9-edbba068c90b",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "1380",
            "date": "Thu, 20 Jun 2024 19:25:53 GMT",
        },
        "RetryAttempts": 0,
    },
}


@pytest.mark.parametrize(
    "instance, expected",
    [
        ("ml.trn1.2xlarge", True),
        ("ml.inf2.xlarge", True),
        ("ml.c7gd.4xlarge", False),
    ],
)
def test_is_inferentia_or_trainium(instance, expected):
    assert _is_inferentia_or_trainium(instance) == expected


@pytest.mark.parametrize(
    "image_uri, expected",
    [
        (
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124",
            True,
        ),
        (
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-neuronx-sdk2.18.2",
            True,
        ),
        (
            None,
            True,
        ),
        (
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:"
            "2.1.1-tgi2.0.0-gpu-py310-cu121-ubuntu22.04",
            False,
        ),
        (None, True),
    ],
)
def test_is_image_compatible_with_optimization_job(image_uri, expected):
    assert _is_image_compatible_with_optimization_job(image_uri) == expected


def test_generate_optimized_model():
    pysdk_model = Mock()
    pysdk_model.model_data = {
        "S3DataSource": {
            "S3Uri": "s3://jumpstart-private-cache-alpha-us-west-2/meta-textgeneration/"
            "meta-textgeneration-llama-3-8b/artifacts/inference-prepack/v1.0.1/"
        }
    }

    optimized_model = _generate_optimized_model(pysdk_model, mock_optimization_job_output)

    assert (
        optimized_model.image_uri
        == mock_optimization_job_output["OptimizationOutput"]["RecommendedInferenceImage"]
    )
    assert optimized_model.env == mock_optimization_job_output["OptimizationEnvironment"]
    assert (
        optimized_model.model_data["S3DataSource"]["S3Uri"]
        == mock_optimization_job_output["ModelSource"]["S3"]
    )
    assert optimized_model.instance_type == mock_optimization_job_output["DeploymentInstanceType"]
    pysdk_model.add_tags.assert_called_once_with(
        {
            "Key": Tag.OPTIMIZATION_JOB_NAME,
            "Value": mock_optimization_job_output["OptimizationJobName"],
        }
    )


@pytest.mark.parametrize(
    "env, new_env, output_env",
    [
        ({"a": "1"}, {"b": "2"}, {"a": "1", "b": "2"}),
        (None, {"b": "2"}, {"b": "2"}),
        ({"a": "1"}, None, {"a": "1"}),
        (None, None, None),
    ],
)
def test_update_environment_variables(env, new_env, output_env):
    assert _update_environment_variables(env, new_env) == output_env


@pytest.mark.parametrize(
    "speculative_decoding_config, expected_model_provider",
    [
        ({"ModelProvider": "SageMaker"}, "sagemaker"),
        ({"ModelProvider": "Custom"}, "custom"),
        ({"ModelSource": "s3://"}, "custom"),
        (None, None),
    ],
)
def test_extract_speculative_draft_model_provider(
    speculative_decoding_config, expected_model_provider
):
    assert (
        _extract_speculative_draft_model_provider(speculative_decoding_config)
        == expected_model_provider
    )


@pytest.mark.parametrize(
    "output_path, instance, quantization_config, compilation_config",
    [
        (
            None,
            None,
            {"OverrideEnvironment": {"TENSOR_PARALLEL_DEGREE": 4}},
            {"OverrideEnvironment": {"TENSOR_PARALLEL_DEGREE": 4}},
        ),
        (None, None, {"OverrideEnvironment": {"TENSOR_PARALLEL_DEGREE": 4}}, None),
        (None, None, None, {"OverrideEnvironment": {"TENSOR_PARALLEL_DEGREE": 4}}),
        ("output_path", None, None, {"OverrideEnvironment": {"TENSOR_PARALLEL_DEGREE": 4}}),
        (None, "instance_type", None, {"OverrideEnvironment": {"TENSOR_PARALLEL_DEGREE": 4}}),
    ],
)
def test_validate_optimization_inputs(
    output_path, instance, quantization_config, compilation_config
):

    with pytest.raises(ValueError):
        _validate_optimization_inputs(
            output_path, instance, quantization_config, compilation_config
        )


def test_extract_speculative_draft_model_s3_uri():
    res = _extracts_and_validates_speculative_model_source({"ModelSource": "s3://"})
    assert res == "s3://"


def test_extract_speculative_draft_model_s3_uri_ex():
    with pytest.raises(ValueError):
        _extracts_and_validates_speculative_model_source({"ModelSource": None})
