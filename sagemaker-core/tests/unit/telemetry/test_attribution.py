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
import os
import pytest
from sagemaker.core.telemetry.attribution import (
    _CREATED_BY_ENV_VAR,
    Attribution,
    set_attribution,
)


@pytest.fixture(autouse=True)
def clean_env():
    yield
    if _CREATED_BY_ENV_VAR in os.environ:
        del os.environ[_CREATED_BY_ENV_VAR]


def test_set_attribution_sagemaker_agent_plugin():
    set_attribution(Attribution.SAGEMAKER_AGENT_PLUGIN)
    assert os.environ[_CREATED_BY_ENV_VAR] == Attribution.SAGEMAKER_AGENT_PLUGIN.value


def test_set_attribution_invalid_type_raises():
    with pytest.raises(TypeError):
        set_attribution("awslabs/agent-plugins/sagemaker-ai")
