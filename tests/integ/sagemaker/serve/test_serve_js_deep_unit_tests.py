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
from unittest.mock import MagicMock, patch, ANY

from sagemaker.session import Session
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.resource_requirements import ResourceRequirements

ROLE_NAME = "SageMakerRole"


def test_js_model_with_optimize_speculative_decoding_config_gated_requests_are_expected(
    sagemaker_session,
):
    with patch.object(
        Session, "create_model", return_value="mock_model"
    ) as mock_create_model, patch.object(
        Session, "endpoint_from_production_variants"
    ) as mock_endpoint_from_production_variants:
        iam_client = sagemaker_session.boto_session.client("iam")
        role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

        schema_builder = SchemaBuilder("test", "test")
        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-1-8b-instruct",
            schema_builder=schema_builder,
            sagemaker_session=sagemaker_session,
            role_arn=role_arn,
        )

        optimized_model = model_builder.optimize(
            instance_type="ml.g5.xlarge",  # set to small instance in case a network call is made
            speculative_decoding_config={
                "ModelProvider": "JumpStart",
                "ModelID": "meta-textgeneration-llama-3-2-1b",
                "AcceptEula": True,
            },
            accept_eula=True,
        )

        optimized_model.deploy()

        mock_create_model.assert_called_once_with(
            name=ANY,
            role=ANY,
            container_defs={
                "Image": ANY,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "ENDPOINT_SERVER_TIMEOUT": "3600",
                    "MODEL_CACHE_ROOT": "/opt/ml/model",
                    "SAGEMAKER_ENV": "1",
                    "HF_MODEL_ID": "/opt/ml/model",
                    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                    "OPTION_SPECULATIVE_DRAFT_MODEL": "/opt/ml/additional-model-data-sources/draft_model/",
                },
                "AdditionalModelDataSources": [
                    {
                        "ChannelName": "draft_model",
                        "S3DataSource": {
                            "S3Uri": ANY,
                            "S3DataType": "S3Prefix",
                            "CompressionType": "None",
                            "ModelAccessConfig": {"AcceptEula": True},
                        },
                    }
                ],
                "ModelDataSource": {
                    "S3DataSource": {
                        "S3Uri": ANY,
                        "S3DataType": "S3Prefix",
                        "CompressionType": "None",
                        "ModelAccessConfig": {"AcceptEula": True},
                    }
                },
            },
            vpc_config=None,
            enable_network_isolation=True,
            tags=ANY,
        )
        mock_endpoint_from_production_variants.assert_called_once()


def test_js_model_with_optimize_sharding_and_resource_requirements_requests_are_expected(
    sagemaker_session,
):
    with patch.object(
        Session,
        "wait_for_optimization_job",
        return_value={"OptimizationJobName": "mock_optimization_job"},
    ), patch.object(
        Session, "create_model", return_value="mock_model"
    ) as mock_create_model, patch.object(
        Session, "endpoint_from_production_variants", return_value="mock_endpoint_name"
    ) as mock_endpoint_from_production_variants, patch.object(
        Session, "create_inference_component"
    ) as mock_create_inference_component:
        iam_client = sagemaker_session.boto_session.client("iam")
        role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

        sagemaker_session.sagemaker_client.create_optimization_job = MagicMock()

        schema_builder = SchemaBuilder("test", "test")
        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-1-8b-instruct",
            schema_builder=schema_builder,
            sagemaker_session=sagemaker_session,
            role_arn=role_arn,
        )

        optimized_model = model_builder.optimize(
            instance_type="ml.g5.xlarge",  # set to small instance in case a network call is made
            sharding_config={"OverrideEnvironment": {"OPTION_TENSOR_PARALLEL_DEGREE": "8"}},
            accept_eula=True,
        )

        optimized_model.deploy(
            resources=ResourceRequirements(requests={"memory": 196608, "num_accelerators": 8})
        )

        mock_create_model.assert_called_once_with(
            name=ANY,
            role=ANY,
            container_defs={
                "Image": ANY,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "ENDPOINT_SERVER_TIMEOUT": "3600",
                    "MODEL_CACHE_ROOT": "/opt/ml/model",
                    "SAGEMAKER_ENV": "1",
                    "HF_MODEL_ID": "/opt/ml/model",
                    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                    "OPTION_TENSOR_PARALLEL_DEGREE": "8",
                },
                "ModelDataSource": {
                    "S3DataSource": {
                        "S3Uri": ANY,
                        "S3DataType": "S3Prefix",
                        "CompressionType": "None",
                        "ModelAccessConfig": {"AcceptEula": True},
                    }
                },
            },
            vpc_config=None,
            enable_network_isolation=False,  # should be set to false
            tags=ANY,
        )
        mock_endpoint_from_production_variants.assert_called_once_with(
            name=ANY,
            production_variants=ANY,
            tags=ANY,
            kms_key=ANY,
            vpc_config=ANY,
            enable_network_isolation=False,
            role=ANY,
            live_logging=False,  # this should be set to false for IC
            wait=True,
        )
        mock_create_inference_component.assert_called_once()


def test_js_model_with_optimize_quantization_on_pre_optimized_model_requests_are_expected(
    sagemaker_session,
):
    with patch.object(
        Session,
        "wait_for_optimization_job",
        return_value={"OptimizationJobName": "mock_optimization_job"},
    ), patch.object(
        Session, "create_model", return_value="mock_model"
    ) as mock_create_model, patch.object(
        Session, "endpoint_from_production_variants", return_value="mock_endpoint_name"
    ) as mock_endpoint_from_production_variants:
        iam_client = sagemaker_session.boto_session.client("iam")
        role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

        sagemaker_session.sagemaker_client.create_optimization_job = MagicMock()

        schema_builder = SchemaBuilder("test", "test")
        model_builder = ModelBuilder(
            model="meta-textgeneration-llama-3-1-8b-instruct",
            schema_builder=schema_builder,
            sagemaker_session=sagemaker_session,
            role_arn=role_arn,
        )

        optimized_model = model_builder.optimize(
            instance_type="ml.g5.xlarge",  # set to small instance in case a network call is made
            quantization_config={
                "OverrideEnvironment": {
                    "OPTION_QUANTIZE": "fp8",
                },
            },
            accept_eula=True,
        )

        optimized_model.deploy()

        mock_create_model.assert_called_once_with(
            name=ANY,
            role=ANY,
            container_defs={
                "Image": ANY,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "ENDPOINT_SERVER_TIMEOUT": "3600",
                    "MODEL_CACHE_ROOT": "/opt/ml/model",
                    "SAGEMAKER_ENV": "1",
                    "HF_MODEL_ID": "/opt/ml/model",
                    "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
                    "OPTION_QUANTIZE": "fp8",
                },
                "ModelDataSource": {
                    "S3DataSource": {
                        "S3Uri": ANY,
                        "S3DataType": "S3Prefix",
                        "CompressionType": "None",
                        "ModelAccessConfig": {"AcceptEula": True},
                    }
                },
            },
            vpc_config=None,
            enable_network_isolation=True,  # should be set to false
            tags=ANY,
        )
        mock_endpoint_from_production_variants.assert_called_once()
