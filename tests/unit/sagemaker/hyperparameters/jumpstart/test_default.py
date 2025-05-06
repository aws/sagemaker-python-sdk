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

from sagemaker import hyperparameters
from sagemaker.jumpstart.enums import JumpStartModelType

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec, get_special_model_spec


mock_client = boto3.client("s3")
region = "us-west-2"
mock_session = Mock(s3_client=mock_client, boto_region_name=region)


@patch("sagemaker.jumpstart.utils.validate_model_id_and_get_type")
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_default_hyperparameters(
    patched_get_model_specs, patched_validate_model_id_and_get_type
):

    patched_get_model_specs.side_effect = get_spec_from_base_spec
    patched_validate_model_id_and_get_type.return_value = JumpStartModelType.OPEN_WEIGHTS

    model_id = "pytorch-eqa-bert-base-cased"
    region = "us-west-2"

    params = hyperparameters.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        sagemaker_session=mock_session,
    )
    assert params == {
        "train_only_top_layer": "True",
        "epochs": "5",
        "learning_rate": "0.001",
        "batch_size": "4",
        "reinitialize_top_layer": "Auto",
    }

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version="*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        hub_arn=None,
        sagemaker_session=mock_session,
    )

    patched_get_model_specs.reset_mock()

    params = hyperparameters.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="1.*",
        sagemaker_session=mock_session,
    )
    assert params == {
        "train_only_top_layer": "True",
        "epochs": "5",
        "learning_rate": "0.001",
        "batch_size": "4",
        "reinitialize_top_layer": "Auto",
    }

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version="1.*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        hub_arn=None,
        sagemaker_session=mock_session,
    )

    patched_get_model_specs.reset_mock()

    params = hyperparameters.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="1.*",
        include_container_hyperparameters=True,
        sagemaker_session=mock_session,
    )
    assert params == {
        "train_only_top_layer": "True",
        "epochs": "5",
        "learning_rate": "0.001",
        "batch_size": "4",
        "reinitialize_top_layer": "Auto",
        "sagemaker_submit_directory": "/opt/ml/input/data/code/sourcedir.tar.gz",
        "sagemaker_program": "transfer_learning.py",
        "sagemaker_container_log_level": "20",
    }

    patched_get_model_specs.assert_called_once_with(
        region=region,
        model_id=model_id,
        version="1.*",
        s3_client=mock_client,
        model_type=JumpStartModelType.OPEN_WEIGHTS,
        hub_arn=None,
        sagemaker_session=mock_session,
    )

    patched_get_model_specs.reset_mock()

    with pytest.raises(KeyError):
        hyperparameters.retrieve_default(
            region=region,
            model_id="blah",
            model_version="*",
        )

    with pytest.raises(ValueError):
        hyperparameters.retrieve_default(
            region="mars-south-1",
            model_id=model_id,
            model_version="*",
        )

    with pytest.raises(ValueError):
        hyperparameters.retrieve_default(
            model_version="*",
        )

    with pytest.raises(ValueError):
        hyperparameters.retrieve_default(
            model_id=model_id,
        )


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_sdk_hyperparameters_instance_type_overrides(patched_get_model_specs):

    patched_get_model_specs.side_effect = get_special_model_spec

    model_id = "variant-model"
    region = "us-west-2"

    # assert that we can add hyperparameters to default
    vars = hyperparameters.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        sagemaker_session=mock_session,
        instance_type="ml.p2.48xlarge",
    )
    assert vars == {
        "adam-learning-rate": "0.05",
        "batch-size": "4",
        "epochs": "3",
        "num_bag_sets": "5",
        "num_stack_levels": "6",
        "refit_full": "False",
        "sagemaker_container_log_level": "20",
        "sagemaker_program": "transfer_learning.py",
        "sagemaker_submit_directory": "/opt/ml/input/data/code/sourcedir.tar.gz",
        "save_space": "False",
        "set_best_to_refit_full": "False",
        "verbosity": "2",
    }

    # assert that we can override default environment variables (instance family + instance type
    # specific)
    vars = hyperparameters.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        sagemaker_session=mock_session,
        instance_type="ml.p2.12xlarge",
    )
    assert vars == {
        "adam-learning-rate": "0.05",
        "batch-size": "1",
        "epochs": "3",
        "num_bag_sets": "1",
        "num_stack_levels": "0",
        "refit_full": "False",
        "eval_metric": "auto",
        "num_bag_folds": "0",
        "presets": "medium_quality",
        "auto_stack": "False",
        "sagemaker_container_log_level": "20",
        "sagemaker_program": "transfer_learning.py",
        "sagemaker_submit_directory": "/opt/ml/input/data/code/sourcedir.tar.gz",
        "save_space": "False",
        "set_best_to_refit_full": "False",
        "verbosity": "2",
    }

    # assert that we can return default hyperparameters for unrecognized instance
    vars = hyperparameters.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
        sagemaker_session=mock_session,
        instance_type="ml.p9999.48xlarge",
    )

    assert vars == {"epochs": "3", "adam-learning-rate": "0.05", "batch-size": "4"}
