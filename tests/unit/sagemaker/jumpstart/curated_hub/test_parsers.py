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

import json
import pytest
import datetime
from sagemaker.jumpstart.enums import ModelSpecKwargType, NamingConventionType
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.curated_hub.interfaces import DescribeHubContentResponse
from sagemaker.jumpstart.curated_hub.parsers import (
    get_model_spec_arg_keys,
    make_model_specs_from_describe_hub_content_response,
)
from tests.unit.sagemaker.jumpstart.constants import (
    SPECIAL_MODEL_SPECS_DICT,
    HUB_MODEL_DOCUMENT_DICTS,
)

gemma_model_spec = SPECIAL_MODEL_SPECS_DICT["gemma-model-2b-v1_1_0"]
gemma_model_document = HUB_MODEL_DOCUMENT_DICTS["huggingface-llm-gemma-2b-instruct"]


@pytest.mark.parametrize(
    ("arg_type,naming_convention,expected"),
    [
        pytest.param(ModelSpecKwargType.FIT, NamingConventionType.UPPER_CAMEL_CASE, []),
        pytest.param(ModelSpecKwargType.FIT, NamingConventionType.SNAKE_CASE, []),
        pytest.param(ModelSpecKwargType.MODEL, NamingConventionType.UPPER_CAMEL_CASE, []),
        pytest.param(ModelSpecKwargType.MODEL, NamingConventionType.SNAKE_CASE, []),
        pytest.param(
            ModelSpecKwargType.ESTIMATOR,
            NamingConventionType.UPPER_CAMEL_CASE,
            [
                "EncryptInterContainerTraffic",
                "MaxRuntimeInSeconds",
                "DisableOutputCompression",
                "ModelDir",
            ],
        ),
        pytest.param(
            ModelSpecKwargType.ESTIMATOR,
            NamingConventionType.SNAKE_CASE,
            [
                "encrypt_inter_container_traffic",
                "max_runtime_in_seconds",
                "disable_output_compression",
                "model_dir",
            ],
        ),
        pytest.param(
            ModelSpecKwargType.DEPLOY,
            NamingConventionType.UPPER_CAMEL_CASE,
            ["ModelDataDownloadTimeout", "ContainerStartupHealthCheckTimeout"],
        ),
        pytest.param(
            ModelSpecKwargType.DEPLOY,
            NamingConventionType.SNAKE_CASE,
            ["model_data_download_timeout", "container_startup_health_check_timeout"],
        ),
    ],
)
def test_get_model_spec_arg_keys(arg_type, naming_convention, expected):
    assert get_model_spec_arg_keys(arg_type, naming_convention) == expected


def test_make_model_specs_from_describe_hub_content_response():
    response = DescribeHubContentResponse(
        {
            "CreationTime": datetime.datetime.today(),
            "DocumentSchemaVersion": "1.2.3",
            "FailureReason": "failed with no apparent reason",
            "HubName": "test-hub-123",
            "HubArn": "arn:aws:sagemaker:us-west-2:012345678910:hub/test-hub-123",
            "HubContentArn": "arn:aws:sagemaker:us-west-2:012345678910:hub-content/"
            "test-gemma-model-2b-instruct",
            "HubContentDependencies": [
                {
                    "DependencyCopyPath": "test-copy-path",
                    "DependencyOriginPath": "test_origin_path",
                },
                {},
            ],
            "HubContentDescription": "this is my cool hub content description.",
            "HubContentDisplayName": "Cool Content Name",
            "HubContentType": "Model",
            "HubContentDocument": json.dumps(gemma_model_document),
            "HubContentMarkdown": "markdown",
            "HubContentName": "huggingface-llm-gemma-2b-instruct",
            "HubContentStatus": "Test",
            "HubContentVersion": "1.1.0",
            "HubContentSearchKeywords": [],
        }
    )
    test_specs = make_model_specs_from_describe_hub_content_response(response)
    test_specs.supported_inference_instance_types.sort()
    test_specs.supported_training_instance_types.sort()
    # Known mismatched fields
    gemma_model_spec["hosting_ecr_uri"] = (
        "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-"
        "inference:2.1.1-tgi1.4.2-gpu-py310-cu121-ubuntu22.04"
    )
    gemma_model_spec["training_ecr_uri"] = (
        "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training"
        ":2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
    )

    expected_specs = JumpStartModelSpecs(gemma_model_spec, is_hub_content=True)
    expected_specs.training_artifact_key = (
        "s3://jumpstart-cache-prod-us-west-2/" + expected_specs.training_artifact_key
    )
    expected_specs.hosting_artifact_key = (
        "s3://jumpstart-cache-prod-us-west-2/" + expected_specs.hosting_artifact_key
    )
    expected_specs.hosting_prepacked_artifact_key = (
        "s3://jumpstart-cache-prod-us-west-2/" + expected_specs.hosting_prepacked_artifact_key
    )
    expected_specs.training_prepacked_script_key = (
        "s3://jumpstart-cache-prod-us-west-2/" + expected_specs.training_prepacked_script_key
    )
    expected_specs.hosting_eula_key = (
        "s3://jumpstart-cache-prod-us-west-2/" + expected_specs.hosting_eula_key
    )
    expected_specs.training_script_key = (
        "s3://jumpstart-cache-prod-us-west-2/" + expected_specs.training_script_key
    )
    expected_specs.hosting_script_key = (
        "s3://jumpstart-cache-prod-us-west-2/" + expected_specs.hosting_script_key
    )
    expected_specs.supported_inference_instance_types.sort()
    expected_specs.supported_training_instance_types.sort()
    expected_specs.hosting_instance_type_variants = {
        "regional_aliases": None,
        "aliases": {
            "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface"
            "-pytorch-tgi-inference:2.1.1-tgi1.4.0-gpu-py310-cu121-ubuntu20.04"
        },
        "variants": {
            "g4dn": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "g5": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "local_gpu": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "p2": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "p3": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "p3dn": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "p4d": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "p4de": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "p5": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "ml.g5.12xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
            "ml.g5.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "4"}}},
            "ml.g5.48xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
            "ml.p4d.24xlarge": {"properties": {"environment_variables": {"SM_NUM_GPUS": "8"}}},
        },
    }

    expected_specs.training_instance_type_variants = {
        "regional_aliases": None,
        "aliases": {
            "gpu_ecr_uri_1": "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-"
            "training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04"
        },
        "variants": {
            "g4dn": {
                "properties": {
                    "image_uri": "$gpu_ecr_uri_1",
                    "gated_model_key_env_var_value": "huggingface-training/g4dn/v1.0.0/train-"
                    "huggingface-llm-gemma-2b-instruct.tar.gz",
                }
            },
            "g5": {
                "properties": {
                    "image_uri": "$gpu_ecr_uri_1",
                    "gated_model_key_env_var_value": "huggingface-training/g5/v1.0.0/train-huggingface"
                    "-llm-gemma-2b-instruct.tar.gz",
                }
            },
            "local_gpu": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "p2": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "p3": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "p3dn": {
                "properties": {
                    "image_uri": "$gpu_ecr_uri_1",
                    "gated_model_key_env_var_value": "huggingface-training/p3dn/v1.0.0/train-"
                    "huggingface-llm-gemma-2b-instruct.tar.gz",
                }
            },
            "p4d": {
                "properties": {
                    "image_uri": "$gpu_ecr_uri_1",
                    "gated_model_key_env_var_value": "huggingface-training/p4d/v1.0.0/train-"
                    "huggingface-llm-gemma-2b-instruct.tar.gz",
                }
            },
            "p4de": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
            "p5": {"properties": {"image_uri": "$gpu_ecr_uri_1"}},
        },
    }

    assert test_specs.to_json() == expected_specs.to_json()
