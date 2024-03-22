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

import pytest
from sagemaker.jumpstart.enums import ModelSpecKwargType, NamingConventionType
from sagemaker.jumpstart.curated_hub.parsers import (
    get_model_spec_arg_keys,
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
            ],
        ),
        pytest.param(
            ModelSpecKwargType.ESTIMATOR,
            NamingConventionType.SNAKE_CASE,
            [
                "encrypt_inter_container_traffic",
                "max_runtime_in_seconds",
                "disable_output_compression",
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
    # response = DescribeHubContentResponse({
    #     "CreationTime": datetime.today(),
    #     "DocumentSchemaVersion": "1.2.3",
    #     "FailureReason": "failed with no apparent reason",
    #     "HubName": "test-hub-123",
    #     "HubArn": "arn:aws:sagemaker:us-west-2:012345678910:hub/test-hub-123",
    #     "HubContentArn": "arn:aws:sagemaker:us-west-2:012345678910:hub-content/test-gemma-model-2b-instruct",
    #     "HubContentDependencies": [
    #         {"DependencyCopyPath": "test-copy-path", "DependencyOriginPath": "test_origin_path"},
    #         {},
    #     ],
    #     "HubContentDescription": "this is my cool hub content description.",
    #     "HubContentDisplayName": "Cool Content Name",
    #     "HubContentType": "Model",
    #     "HubContentDocument": gemma_model_document,
    #     "HubContentMarkdown": "markdown",
    #     "HubContentName": "huggingface-llm-gemma-2b-instruct",
    #     "HubContentStatus": "Test",
    #     "HubContentVersion": "1.1.0",
    #     "HubContentSearchKeywords": [],
    # })
    # test_specs = make_model_specs_from_describe_hub_content_response(response)
    # expected_specs = JumpStartModelSpecs(gemma_model_spec)
    # assert test_specs == expected_specs
    pass
