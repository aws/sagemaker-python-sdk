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
"""Unit tests for workflow quality_check_step."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock

from sagemaker.mlops.workflow.quality_check_step import (
    DataQualityCheckConfig, ModelQualityCheckConfig
)
from sagemaker.mlops.workflow.steps import StepTypeEnum


def test_data_quality_check_config_init():
    config = DataQualityCheckConfig(
        baseline_dataset="s3://bucket/data.csv",
        dataset_format={"csv": {"header": True}}
    )
    assert config.baseline_dataset == "s3://bucket/data.csv"
    assert config.dataset_format == {"csv": {"header": True}}


def test_model_quality_check_config_init():
    config = ModelQualityCheckConfig(
        baseline_dataset="s3://bucket/data.csv",
        dataset_format={"csv": {"header": True}},
        problem_type="BinaryClassification"
    )
    assert config.problem_type == "BinaryClassification"
