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


from mock.mock import patch
import pytest

from sagemaker import environment_variables

from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec


@patch("sagemaker.jumpstart.accessors.JumpStartModelsCache.get_model_specs")
def test_jumpstart_default_environment_variables(patched_get_model_specs):

    patched_get_model_specs.side_effect = get_spec_from_base_spec

    model_id = "pytorch-eqa-bert-base-cased"
    region = "us-west-2"

    vars = environment_variables.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="*",
    )
    assert vars == {
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        "SAGEMAKER_ENV": "1",
        "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
    }

    patched_get_model_specs.assert_called_once_with(region=region, model_id=model_id, version="*")

    patched_get_model_specs.reset_mock()

    vars = environment_variables.retrieve_default(
        region=region,
        model_id=model_id,
        model_version="1.*",
    )
    assert vars == {
        "MODEL_CACHE_ROOT": "/opt/ml/model",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        "SAGEMAKER_ENV": "1",
        "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600",
        "SAGEMAKER_MODEL_SERVER_WORKERS": "1",
        "SAGEMAKER_PROGRAM": "inference.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
    }

    patched_get_model_specs.assert_called_once_with(region=region, model_id=model_id, version="1.*")

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
