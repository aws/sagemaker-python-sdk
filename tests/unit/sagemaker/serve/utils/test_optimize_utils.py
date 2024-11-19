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

import unittest
from unittest.mock import Mock, patch

import pytest

from sagemaker.enums import Tag
from sagemaker.serve.utils.optimize_utils import (
    _generate_optimized_model,
    _update_environment_variables,
    _is_image_compatible_with_optimization_job,
    _extract_speculative_draft_model_provider,
    _extracts_and_validates_speculative_model_source,
    _is_s3_uri,
    _generate_additional_model_data_sources,
    _generate_channel_name,
    _extract_optimization_config_and_env,
    _is_optimized,
    _custom_speculative_decoding,
    _is_inferentia_or_trainium,
    _is_draft_model_gated,
    _deployment_config_contains_draft_model,
    _jumpstart_speculative_decoding,
)
from tests.unit.sagemaker.serve.constants import (
    GATED_DRAFT_MODEL_CONFIG,
    NON_GATED_DRAFT_MODEL_CONFIG,
    OPTIMIZED_DEPLOYMENT_CONFIG_WITH_GATED_DRAFT_MODEL,
    NON_OPTIMIZED_DEPLOYMENT_CONFIG,
)

mock_optimization_job_output = {
    "OptimizationJobArn": "arn:aws:sagemaker:us-west-2:312206380606:optimization-job/"
    "modelbuilderjob-3cbf9c40b63c455d85b60033f9a01691",
    "OptimizationJobStatus": "COMPLETED",
    "OptimizationJobName": "modelbuilderjob-3cbf9c40b63c455d85b60033f9a01691",
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
    "DeploymentInstanceType": "ml.g5.2xlarge",
    "OptimizationConfigs": [
        {
            "ModelQuantizationConfig": {
                "Image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124",
                "OverrideEnvironment": {"OPTION_QUANTIZE": "awq"},
            }
        }
    ],
    "OutputConfig": {"S3OutputLocation": "s3://quicksilver-model-data/llama-3-8b/quantized-1/"},
    "OptimizationOutput": {
        "RecommendedInferenceImage": "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124"
    },
    "RoleArn": "arn:aws:iam::312206380606:role/service-role/AmazonSageMaker-ExecutionRole-20240116T151132",
    "StoppingCondition": {"MaxRuntimeInSeconds": 36000},
    "ResponseMetadata": {
        "RequestId": "a95253d5-c045-4708-8aac-9f0d327515f7",
        "HTTPStatusCode": 200,
        "HTTPHeaders": {
            "x-amzn-requestid": "a95253d5-c045-4708-8aac-9f0d327515f7",
            "content-type": "application/x-amz-json-1.1",
            "content-length": "1371",
            "date": "Fri, 21 Jun 2024 04:27:42 GMT",
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
    assert (
        optimized_model.model_data["S3DataSource"]["S3Uri"]
        == mock_optimization_job_output["OutputConfig"]["S3OutputLocation"]
    )
    assert optimized_model.instance_type == mock_optimization_job_output["DeploymentInstanceType"]
    pysdk_model.add_tags.assert_called_once_with(
        {
            "Key": Tag.OPTIMIZATION_JOB_NAME,
            "Value": mock_optimization_job_output["OptimizationJobName"],
        }
    )


def test_is_optimized():
    model = Mock()

    model._tags = {"Key": Tag.OPTIMIZATION_JOB_NAME}
    assert _is_optimized(model) is True

    model._tags = [{"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER}]
    assert _is_optimized(model) is True

    model._tags = [{"Key": Tag.FINE_TUNING_MODEL_PATH}]
    assert _is_optimized(model) is False


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
        ({"ModelProvider": "JumpStart"}, "jumpstart"),
        ({"ModelProvider": "asdf"}, "auto"),
        ({"ModelProvider": "Auto"}, "auto"),
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


def test_extract_speculative_draft_model_s3_uri():
    res = _extracts_and_validates_speculative_model_source({"ModelSource": "s3://"})
    assert res == "s3://"


def test_extract_speculative_draft_model_s3_uri_ex():
    with pytest.raises(ValueError):
        _extracts_and_validates_speculative_model_source({"ModelSource": None})


def test_generate_channel_name():
    assert _generate_channel_name(None) is not None

    additional_model_data_sources = _generate_additional_model_data_sources(
        "s3://jumpstart-private-cache-alpha-us-west-2/meta-textgeneration/", "channel_name", True
    )

    assert _generate_channel_name(additional_model_data_sources) == "channel_name"


def test_generate_additional_model_data_sources():
    model_source = _generate_additional_model_data_sources(
        "s3://jumpstart-private-cache-alpha-us-west-2/meta-textgeneration/", "channel_name", True
    )

    assert model_source == [
        {
            "ChannelName": "channel_name",
            "S3DataSource": {
                "S3Uri": "s3://jumpstart-private-cache-alpha-us-west-2/meta-textgeneration/",
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
                "ModelAccessConfig": {"AcceptEula": True},
            },
        }
    ]

    model_source = _generate_additional_model_data_sources(
        "s3://jumpstart-private-cache-alpha-us-west-2/meta-textgeneration/", "channel_name", False
    )

    assert model_source == [
        {
            "ChannelName": "channel_name",
            "S3DataSource": {
                "S3Uri": "s3://jumpstart-private-cache-alpha-us-west-2/meta-textgeneration/",
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            },
        }
    ]


@pytest.mark.parametrize(
    "s3_uri, expected",
    [
        (
            "s3://jumpstart-private-cache-alpha-us-west-2/meta-textgeneration/"
            "meta-textgeneration-llama-3-8b/artifacts/inference-prepack/v1.0.1/",
            True,
        ),
        ("invalid://", False),
    ],
)
def test_is_s3_uri(s3_uri, expected):
    assert _is_s3_uri(s3_uri) == expected


@pytest.mark.parametrize(
    "draft_model_config, expected",
    [
        (GATED_DRAFT_MODEL_CONFIG, True),
        (NON_GATED_DRAFT_MODEL_CONFIG, False),
    ],
)
def test_is_draft_model_gated(draft_model_config, expected):
    assert _is_draft_model_gated(draft_model_config) is expected


@pytest.mark.parametrize(
    "quantization_config, compilation_config, expected_config, expected_quant_env, expected_compilation_env",
    [
        (
            None,
            {
                "OverrideEnvironment": {
                    "OPTION_TENSOR_PARALLEL_DEGREE": "2",
                }
            },
            {
                "ModelCompilationConfig": {
                    "OverrideEnvironment": {
                        "OPTION_TENSOR_PARALLEL_DEGREE": "2",
                    }
                },
            },
            None,
            {
                "OPTION_TENSOR_PARALLEL_DEGREE": "2",
            },
        ),
        (
            {
                "OverrideEnvironment": {
                    "OPTION_TENSOR_PARALLEL_DEGREE": "2",
                }
            },
            None,
            {
                "ModelQuantizationConfig": {
                    "OverrideEnvironment": {
                        "OPTION_TENSOR_PARALLEL_DEGREE": "2",
                    }
                },
            },
            {
                "OPTION_TENSOR_PARALLEL_DEGREE": "2",
            },
            None,
        ),
        (None, None, None, None, None),
    ],
)
def test_extract_optimization_config_and_env(
    quantization_config,
    compilation_config,
    expected_config,
    expected_quant_env,
    expected_compilation_env,
):
    assert _extract_optimization_config_and_env(quantization_config, compilation_config) == (
        expected_config,
        expected_quant_env,
        expected_compilation_env,
    )


@pytest.mark.parametrize(
    "deployment_config",
    [
        (OPTIMIZED_DEPLOYMENT_CONFIG_WITH_GATED_DRAFT_MODEL, True),
        (NON_OPTIMIZED_DEPLOYMENT_CONFIG, False),
        (None, False),
    ],
)
def deployment_config_contains_draft_model(deployment_config, expected):
    assert _deployment_config_contains_draft_model(deployment_config)


class TestJumpStartSpeculativeDecodingConfig(unittest.TestCase):

    @patch("sagemaker.model.Model")
    def test_with_no_js_model_id(self, mock_model):
        mock_model.env = {}
        mock_model.additional_model_data_sources = None
        speculative_decoding_config = {"ModelSource": "JumpStart"}

        with self.assertRaises(ValueError) as _:
            _jumpstart_speculative_decoding(mock_model, speculative_decoding_config)

    @patch(
        "sagemaker.jumpstart.utils.accessors.JumpStartModelsAccessor.get_jumpstart_gated_content_bucket",
        return_value="js_gated_content_bucket",
    )
    @patch(
        "sagemaker.jumpstart.utils.accessors.JumpStartModelsAccessor.get_jumpstart_content_bucket",
        return_value="js_content_bucket",
    )
    @patch(
        "sagemaker.jumpstart.utils.accessors.JumpStartModelsAccessor.get_model_specs",
        return_value=Mock(),
    )
    @patch("sagemaker.model.Model")
    def test_with_gated_js_model(
        self,
        mock_model,
        mock_model_specs,
        mock_js_content_bucket,
        mock_js_gated_content_bucket,
    ):
        mock_sagemaker_session = Mock()
        mock_sagemaker_session.boto_region_name = "us-west-2"

        mock_model.env = {}
        mock_model.additional_model_data_sources = None
        speculative_decoding_config = {
            "ModelSource": "JumpStart",
            "ModelID": "meta-textgeneration-llama-3-2-1b",
            "AcceptEula": True,
        }

        mock_model_specs.return_value.to_json.return_value = {
            "gated_bucket": True,
            "hosting_prepacked_artifact_key": "hosting_prepacked_artifact_key",
        }

        _jumpstart_speculative_decoding(
            mock_model, speculative_decoding_config, mock_sagemaker_session
        )

        expected_env_var = {
            "OPTION_SPECULATIVE_DRAFT_MODEL": "/opt/ml/additional-model-data-sources/draft_model/"
        }
        self.maxDiff = None

        self.assertEqual(
            mock_model.additional_model_data_sources,
            [
                {
                    "ChannelName": "draft_model",
                    "S3DataSource": {
                        "S3Uri": f"s3://{mock_js_gated_content_bucket.return_value}/hosting_prepacked_artifact_key",
                        "S3DataType": "S3Prefix",
                        "CompressionType": "None",
                        "ModelAccessConfig": {"AcceptEula": True},
                    },
                }
            ],
        )

        mock_model.add_tags.assert_called_once_with(
            {"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": "jumpstart"}
        )
        self.assertEqual(mock_model.env, expected_env_var)

    @patch(
        "sagemaker.serve.utils.optimize_utils.get_eula_message", return_value="Accept eula message"
    )
    @patch(
        "sagemaker.jumpstart.utils.accessors.JumpStartModelsAccessor.get_jumpstart_gated_content_bucket",
        return_value="js_gated_content_bucket",
    )
    @patch(
        "sagemaker.jumpstart.utils.accessors.JumpStartModelsAccessor.get_jumpstart_content_bucket",
        return_value="js_content_bucket",
    )
    @patch(
        "sagemaker.jumpstart.utils.accessors.JumpStartModelsAccessor.get_model_specs",
        return_value=Mock(),
    )
    @patch("sagemaker.model.Model")
    def test_with_gated_js_model_and_accept_eula_false(
        self,
        mock_model,
        mock_model_specs,
        mock_js_content_bucket,
        mock_js_gated_content_bucket,
        mock_eula_message,
    ):
        mock_sagemaker_session = Mock()
        mock_sagemaker_session.boto_region_name = "us-west-2"

        mock_model.env = {}
        mock_model.additional_model_data_sources = None
        speculative_decoding_config = {
            "ModelSource": "JumpStart",
            "ModelID": "meta-textgeneration-llama-3-2-1b",
            "AcceptEula": False,
        }

        mock_model_specs.return_value.to_json.return_value = {
            "gated_bucket": True,
            "hosting_prepacked_artifact_key": "hosting_prepacked_artifact_key",
        }

        self.assertRaisesRegex(
            ValueError,
            f"{mock_eula_message.return_value} Set `AcceptEula`=True in "
            f"speculative_decoding_config once acknowledged.",
            _jumpstart_speculative_decoding,
            mock_model,
            speculative_decoding_config,
            mock_sagemaker_session,
        )


class TestCustomSpeculativeDecodingConfig(unittest.TestCase):

    @patch("sagemaker.model.Model")
    def test_with_s3_hf(self, mock_model):
        mock_model.env = {}
        mock_model.additional_model_data_sources = None
        speculative_decoding_config = {
            "ModelSource": "s3://bucket/djl-inference-2024-07-02-00-03-32-127/code"
        }

        res_model = _custom_speculative_decoding(mock_model, speculative_decoding_config)

        mock_model.add_tags.assert_called_once_with(
            {"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": "custom"}
        )

        self.assertEqual(
            res_model.env,
            {"OPTION_SPECULATIVE_DRAFT_MODEL": "/opt/ml/additional-model-data-sources/draft_model"},
        )
        self.assertEqual(
            res_model.additional_model_data_sources,
            [
                {
                    "ChannelName": "draft_model",
                    "S3DataSource": {
                        "S3Uri": "s3://bucket/djl-inference-2024-07-02-00-03-32-127/code",
                        "S3DataType": "S3Prefix",
                        "CompressionType": "None",
                    },
                }
            ],
        )

    @patch("sagemaker.model.Model")
    def test_with_s3_js(self, mock_model):
        mock_model.env = {}
        mock_model.additional_model_data_sources = None
        speculative_decoding_config = {
            "ModelSource": "s3://bucket/huggingface-pytorch-tgi-inference"
        }

        res_model = _custom_speculative_decoding(mock_model, speculative_decoding_config, True)

        self.assertEqual(
            res_model.additional_model_data_sources,
            [
                {
                    "ChannelName": "draft_model",
                    "S3DataSource": {
                        "S3Uri": "s3://bucket/huggingface-pytorch-tgi-inference",
                        "S3DataType": "S3Prefix",
                        "CompressionType": "None",
                        "ModelAccessConfig": {"AcceptEula": True},
                    },
                }
            ],
        )

    @patch("sagemaker.model.Model")
    def test_with_non_s3(self, mock_model):
        mock_model.env = {}
        mock_model.additional_model_data_sources = None
        speculative_decoding_config = {"ModelSource": "huggingface-pytorch-tgi-inference"}

        res_model = _custom_speculative_decoding(mock_model, speculative_decoding_config, False)

        self.assertIsNone(res_model.additional_model_data_sources)
        self.assertEqual(
            res_model.env,
            {"OPTION_SPECULATIVE_DRAFT_MODEL": "huggingface-pytorch-tgi-inference"},
        )

        mock_model.add_tags.assert_called_once_with(
            {"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": "custom"}
        )
