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
"""This module contains code to test SageMaker ``ModelPackageGroup``"""
from __future__ import absolute_import

import pytest


@pytest.mark.skip("data inconsistency P61661075")
def test_pipeline_execution_arn(static_model_package_group_context, static_pipeline_execution_arn):
    pipeline_execution_arn = static_model_package_group_context.pipeline_execution_arn()

    assert pipeline_execution_arn == static_pipeline_execution_arn
