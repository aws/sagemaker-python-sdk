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

from unittest.mock import MagicMock, patch

from sagemaker.transformer import Transformer
from tests.unit.sagemaker.workflow.test_mechanism.test_code.test_pipeline_var_compatibility_template import (
    PipelineVarCompatiTestTemplate,
)
from tests.unit.sagemaker.workflow.test_mechanism.test_code import MockProperties


# These tests provide the incomplete default arg dict
# within which some class or target func parameters are missing.
# The test template will fill in those missing args
# Note: the default args should not include PipelineVariable objects
@patch(
    "sagemaker.workflow.steps.Properties",
    MagicMock(return_value=MockProperties(step_name="MyStep")),
)
def test_transformer_compatibility():
    default_args = dict(
        clazz_args=dict(),
        func_args=dict(),
    )
    test_template = PipelineVarCompatiTestTemplate(
        clazz=Transformer,
        default_args=default_args,
    )
    test_template.check_compatibility()
