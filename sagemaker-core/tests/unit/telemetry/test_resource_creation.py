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
from unittest.mock import MagicMock
from sagemaker.core.utils.utils import Unassigned
from sagemaker.core.telemetry.resource_creation import _RESOURCE_ARN_ATTRIBUTES, get_resource_arn


# Each entry: (class_name, arn_attr, arn_value)
_RESOURCE_TEST_CASES = [
    (
        "TrainingJob",
        "training_job_arn",
        "arn:aws:sagemaker:us-west-2:123456789012:training-job/my-job",
    ),
]


def test_get_resource_arn_none_response():
    assert get_resource_arn(None) is None


def test_get_resource_arn_unknown_type():
    assert get_resource_arn("some string") is None
    assert get_resource_arn(42) is None


@pytest.mark.parametrize("class_name,arn_attr,arn_value", _RESOURCE_TEST_CASES)
def test_get_resource_arn_with_valid_arn(class_name, arn_attr, arn_value):
    mock_resource = MagicMock()
    mock_resource.__class__.__name__ = class_name
    setattr(mock_resource, arn_attr, arn_value)
    assert get_resource_arn(mock_resource) == arn_value


@pytest.mark.parametrize("class_name,arn_attr,arn_value", _RESOURCE_TEST_CASES)
def test_get_resource_arn_with_unassigned(class_name, arn_attr, arn_value):
    mock_resource = MagicMock()
    mock_resource.__class__.__name__ = class_name
    setattr(mock_resource, arn_attr, Unassigned())
    assert get_resource_arn(mock_resource) is None


@pytest.mark.parametrize("class_name,arn_attr,arn_value", _RESOURCE_TEST_CASES)
def test_get_resource_arn_with_none_arn(class_name, arn_attr, arn_value):
    mock_resource = MagicMock()
    mock_resource.__class__.__name__ = class_name
    setattr(mock_resource, arn_attr, None)
    assert get_resource_arn(mock_resource) is None


# Verify string keys in _RESOURCE_ARN_ATTRIBUTES match actual class names
@pytest.mark.parametrize("class_name,arn_attr,arn_value", _RESOURCE_TEST_CASES)
def test_resource_class_name_matches_dict_key(class_name, arn_attr, arn_value):
    from sagemaker.core.resources import TrainingJob

    _CLASS_MAP = {
        "TrainingJob": TrainingJob,
    }
    cls = _CLASS_MAP.get(class_name)
    assert cls is not None, f"No class found for key '{class_name}'"
    assert cls.__name__ == class_name
    assert class_name in _RESOURCE_ARN_ATTRIBUTES
