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
)

mock_optimization_job_output = {
    "OptimizationJobName": "optimization_job_name",
    "RecommendedInferenceImage": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
    "huggingface-pytorch-tgi-inference:2.1.1-tgi2.0.0-gpu-py310-cu121-ubuntu22.04",
    "OptimizationJobStatus": "COMPLETED",
    "OptimizationEnvironment": {
        "SAGEMAKER_PROGRAM": "inference.py",
        "ENDPOINT_SERVER_TIMEOUT": "3600",
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_ENV": "1",
        "HF_MODEL_ID": "/opt/ml/model",
        "MAX_INPUT_LENGTH": "4095",
        "MAX_TOTAL_TOKENS": "4096",
        "MAX_BATCH_PREFILL_TOKENS": "8192",
        "MAX_CONCURRENT_REQUESTS": "512",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
    },
    "ModelSource": {
        "S3": "s3://jumpstart-private-cache-prod-us-west-2/meta-textgeneration/"
        "meta-textgeneration-llama-3-8b/artifacts/inference-prepack/v2.0.0/"
    },
    "DeploymentInstanceType": "ml.m5.xlarge",
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
    pysdk_model.model_data = {"S3DataSource": {"S3Uri": "s3://foo/bar"}}

    optimized_model = _generate_optimized_model(pysdk_model, mock_optimization_job_output)

    assert optimized_model.image_uri == mock_optimization_job_output["RecommendedInferenceImage"]
    assert optimized_model.env == mock_optimization_job_output["OptimizationEnvironment"]
    assert (
        optimized_model.model_data["S3DataSource"]["S3Uri"]
        == mock_optimization_job_output["ModelSource"]["S3"]
    )
    assert optimized_model.instance_type == mock_optimization_job_output["DeploymentInstanceType"]
    pysdk_model.add_tags.assert_called_once_with(
        {
            "key": Tag.OPTIMIZATION_JOB_NAME,
            "value": mock_optimization_job_output["OptimizationJobName"],
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
