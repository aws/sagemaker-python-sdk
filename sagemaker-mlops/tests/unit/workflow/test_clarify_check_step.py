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
"""Unit tests for workflow clarify_check_step."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock

from sagemaker.mlops.workflow.clarify_check_step import (
    DataBiasCheckConfig, ModelBiasCheckConfig, ModelExplainabilityCheckConfig
)


def test_data_bias_check_config_init():
    data_config = Mock()
    bias_config = Mock()
    
    config = DataBiasCheckConfig(
        data_config=data_config,
        data_bias_config=bias_config
    )
    assert config.data_config == data_config
    assert config.data_bias_config == bias_config


def test_model_bias_check_config_init():
    data_config = Mock()
    bias_config = Mock()
    model_config = Mock()
    label_config = Mock()
    
    config = ModelBiasCheckConfig(
        data_config=data_config,
        data_bias_config=bias_config,
        model_config=model_config,
        model_predicted_label_config=label_config
    )
    assert config.model_config == model_config


def test_model_explainability_check_config_init():
    data_config = Mock()
    model_config = Mock()
    explainability_config = Mock()
    
    config = ModelExplainabilityCheckConfig(
        data_config=data_config,
        model_config=model_config,
        explainability_config=explainability_config
    )
    assert config.explainability_config == explainability_config
