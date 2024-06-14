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
    _is_speculation_enabled,
    _extract_supported_deployment_config,
    _is_inferentia_or_trainium,
    _is_compatible_with_optimization_job,
)

mock_optimization_job_output = {
    "OptimizationJobName": "optimization_job_name",
    "RecommendedInferenceImage": "763104351884.dkr.ecr.us-west-2.amazonaws.com/"
    "huggingface-pytorch-tgi-inference:2.1.1-tgi2.0.0-gpu-py310-cu121-ubuntu22.04",
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
    "instance, image_uri, expected",
    [
        (
            "ml.g5.12xlarge",
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124",
            True,
        ),
        (
            "ml.trn1.2xlarge",
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-neuronx-sdk2.18.2",
            True,
        ),
        (
            "ml.inf2.xlarge",
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-neuronx-sdk2.18.2",
            True,
        ),
        (
            "ml.c7gd.4xlarge",
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:"
            "2.1.1-tgi2.0.0-gpu-py310-cu121-ubuntu22.04",
            False,
        ),
    ],
)
def test_is_compatible_with_optimization_job(instance, image_uri, expected):
    assert _is_compatible_with_optimization_job(instance, image_uri) == expected


@pytest.mark.parametrize(
    "deployment_configs, expected",
    [
        (
            [
                {
                    "InstanceType": "ml.c7gd.4xlarge",
                    "DeploymentArgs": {
                        "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124"
                    },
                    "AccelerationConfigs": [
                        {
                            "type": "acceleration",
                            "enabled": True,
                            "spec": {"compiler": "a", "version": "1"},
                        }
                    ],
                }
            ],
            None,
        ),
        (
            [
                {
                    "InstanceType": "ml.g5.12xlarge",
                    "DeploymentArgs": {
                        "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124"
                    },
                    "AccelerationConfigs": [
                        {
                            "type": "speculation",
                            "enabled": True,
                        }
                    ],
                }
            ],
            {
                "InstanceType": "ml.g5.12xlarge",
                "DeploymentArgs": {
                    "ImageUri": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124"
                },
                "AccelerationConfigs": [
                    {
                        "type": "speculation",
                        "enabled": True,
                    }
                ],
            },
        ),
        (None, None),
    ],
)
def test_extract_supported_deployment_config(deployment_configs, expected):
    assert _extract_supported_deployment_config(deployment_configs, True) == expected


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
    "deployment_config, expected",
    [
        (
            {
                "AccelerationConfigs": [
                    {
                        "type": "acceleration",
                        "enabled": True,
                        "spec": {"compiler": "a", "version": "1"},
                    }
                ],
            },
            False,
        ),
        (
            {
                "AccelerationConfigs": [
                    {
                        "type": "speculation",
                        "enabled": True,
                    }
                ],
            },
            True,
        ),
        (None, False),
    ],
)
def test_is_speculation_enabled(deployment_config, expected):
    assert _is_speculation_enabled(deployment_config) is expected
