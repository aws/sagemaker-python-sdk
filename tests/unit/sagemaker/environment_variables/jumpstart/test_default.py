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


import boto3
from mock.mock import patch, Mock
import pytest

from sagemaker import environment_variables
from sagemaker.jumpstart.utils import get_jumpstart_gated_content_bucket
from sagemaker.jumpstart.enums import JumpStartModelType

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec, get_special_model_spec


mock_client = boto3.client("s3")
mock_session = Mock(s3_client=mock_client)


@patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_default_environment_variables(
    patched_get_model_specs, patched_validate_model_id_and_get_type
):

    patched_get_model_specs.side_effect = get_spec_from_base_spec
    patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

    model_id = "pytorch-eqa-bert-base-cased"
    region = "us-west-2"

    vars = environment_variables.retrieve_default(
        region=region, model_id=model_id, model_version="*", sagemaker_session=mock_session
    )
    assert vars == {
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        "SAGEMAKER_ENV": "1",
        "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600",
        "ENDPOINT_SERVER_TIMEOUT": "3600",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
    }

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version="*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
    )

    patched_get_model_specs.reset_mock()

    vars = environment_variables.retrieve_default(
        region=region, model_id=model_id, model_version="1.*", sagemaker_session=mock_session
    )
    assert vars == {
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        "SAGEMAKER_ENV": "1",
        "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600",
        "ENDPOINT_SERVER_TIMEOUT": "3600",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
    }

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version="1.*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
    )

    patched_get_model_specs.reset_mock()

    with pytest.raises(KeyError):
        environment_variables.retrieve_default(
            region=region,
            model_id="blah",
            model_version="*",
        )

    with pytest.raises(ValueError):
        environment_variables.retrieve_default(
            region="mars-south-1",
            model_id=model_id,
            model_version="*",
        )

    with pytest.raises(ValueError):
        environment_variables.retrieve_default(
            model_version="*",
        )

    with pytest.raises(ValueError):
        environment_variables.retrieve_default(
            model_id=model_id,
        )


@patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_sdk_environment_variables(
    patched_get_model_specs, patched_validate_model_id_and_get_type
):

    patched_get_model_specs.side_effect = get_spec_from_base_spec
    patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

    model_id = "pytorch-eqa-bert-base-cased"
    region = "us-west-2"

    vars = environment_variables.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        include_aws_sdk_env_vars=False,
        sagemaker_session=mock_session,
    )
    assert vars == {
        "ENDPOINT_SERVER_TIMEOUT": "3600",
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_ENV": "1",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
    }

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version="*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
    )

    patched_get_model_specs.reset_mock()

    vars = environment_variables.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="1.*",
        include_aws_sdk_env_vars=False,
        sagemaker_session=mock_session,
    )
    assert vars == {
        "ENDPOINT_SERVER_TIMEOUT": "3600",
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_ENV": "1",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
    }

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version="1.*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
    )

    patched_get_model_specs.reset_mock()

    with pytest.raises(KeyError):
        environment_variables.retrieve_default(
            region=region,
            model_id="blah",
            model_version="*",
            include_aws_sdk_env_vars=False,
        )

    with pytest.raises(ValueError):
        environment_variables.retrieve_default(
            region="mars-south-1",
            model_id=model_id,
            model_version="*",
            include_aws_sdk_env_vars=False,
        )

    with pytest.raises(ValueError):
        environment_variables.retrieve_default(
            model_version="*",
            include_aws_sdk_env_vars=False,
        )

    with pytest.raises(ValueError):
        environment_variables.retrieve_default(
            model_id=model_id,
            include_aws_sdk_env_vars=False,
        )


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_sdk_environment_variables_1_artifact_all_variants(patched_get_model_specs):

    patched_get_model_specs.side_effect = get_special_model_spec

    model_id = "gemma-model-1-artifact"
    region = "us-west-2"

    assert {
        "SageMakerGatedModelS3Uri": f"s3://{get_jumpstart_gated_content_bucket(region)}/"
        "huggingface-training/train-huggingface-llm-gemma-7b-instruct.tar.gz"
    } == environment_variables.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        include_aws_sdk_env_vars=False,
        sagemaker_session=mock_session,
        instance_type="ml.p3.2xlarge",
        script="training",
    )


@patch("sagemaker.jumpstart.artifacts.environment_variables.JUMPSTART_LOGGER")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_sdk_environment_variables_no_gated_env_var_available(
    patched_get_model_specs, patched_jumpstart_logger
):

    patched_get_model_specs.side_effect = get_special_model_spec

    model_id = "gemma-model"
    region = "us-west-2"

    assert {} == environment_variables.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        include_aws_sdk_env_vars=False,
        sagemaker_session=mock_session,
        instance_type="ml.p3.2xlarge",
        script="training",
    )

    patched_jumpstart_logger.warning.assert_called_once_with(
        "'gemma-model' does not support ml.p3.2xlarge instance type for "
        "training. Please use one of the following instance types: "
        "ml.g5.12xlarge, ml.g5.24xlarge, ml.g5.48xlarge, ml.p4d.24xlarge."
    )

    # assert that supported instance types succeed
    assert {
        "SageMakerGatedModelS3Uri": f"s3://{get_jumpstart_gated_content_bucket(region)}/"
        "huggingface-training/g5/v1.0.0/train-huggingface-llm-gemma-7b-instruct.tar.gz"
    } == environment_variables.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        include_aws_sdk_env_vars=False,
        sagemaker_session=mock_session,
        instance_type="ml.g5.24xlarge",
        script="training",
    )


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_sdk_environment_variables_instance_type_overrides(patched_get_model_specs):

    patched_get_model_specs.side_effect = get_special_model_spec

    model_id = "env-var-variant-model"
    region = "us-west-2"

    # assert that we can override default environment variables
    vars = environment_variables.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        include_aws_sdk_env_vars=False,
        sagemaker_session=mock_session,
        instance_type="ml.g5.48xlarge",
    )
    assert vars == {
        "ENDPOINT_SERVER_TIMEOUT": "3600",
        "HF_MODEL_ID": "/opt/ml/model",
        "MAX_INPUT_LENGTH": "1024",
        "MAX_TOTAL_TOKENS": "2048",
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_ENV": "1",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
        "SM_NUM_GPUS": "80",
    }

    # assert that we can add environment variables
    vars = environment_variables.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        include_aws_sdk_env_vars=False,
        sagemaker_session=mock_session,
        instance_type="ml.p4d.24xlarge",
    )
    assert vars == {
        "ENDPOINT_SERVER_TIMEOUT": "3600",
        "HF_MODEL_ID": "/opt/ml/model",
        "MAX_INPUT_LENGTH": "1024",
        "MAX_TOTAL_TOKENS": "2048",
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_ENV": "1",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
        "SM_NUM_GPUS": "8",
        "YODEL": "NACEREMA",
    }

    # assert that we can return default env variables for unrecognized instance
    vars = environment_variables.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        include_aws_sdk_env_vars=False,
        sagemaker_session=mock_session,
        instance_type="ml.p002.xlarge",
    )
    assert vars == {
        "ENDPOINT_SERVER_TIMEOUT": "3600",
        "HF_MODEL_ID": "/opt/ml/model",
        "MAX_INPUT_LENGTH": "1024",
        "MAX_TOTAL_TOKENS": "2048",
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_ENV": "1",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
        "SM_NUM_GPUS": "8",
    }
