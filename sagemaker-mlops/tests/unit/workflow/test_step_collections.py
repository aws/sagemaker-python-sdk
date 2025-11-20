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
"""Unit tests for workflow step_collections."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock

from sagemaker.mlops.workflow.step_collections import StepCollection


def test_step_collection_init():
    step1 = Mock()
    step1.name = "step1"
    step2 = Mock()
    step2.name = "step2"
    
    collection = StepCollection(name="test-collection", steps=[step1, step2])
    assert collection.name == "test-collection"
    assert len(collection.steps) == 2


def test_step_collection_request_structure():
    step = Mock()
    step.name = "step1"
    step.to_request.return_value = {"Name": "step1"}
    
    collection = StepCollection(name="test-collection", steps=[step])
    request = collection.request_dicts()
    assert len(request) == 1
