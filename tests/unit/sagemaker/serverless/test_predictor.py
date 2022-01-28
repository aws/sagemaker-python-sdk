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

from sagemaker.serverless import LambdaPredictor

FUNCTION_NAME = "my-function"


def test_deprecated_for_class_lamda_predictor():
    with pytest.warns(DeprecationWarning) as w:
        LambdaPredictor()
        msg = (
            "LambdaPredictor is a no-op in sagemaker>=v2.66.3.\n"
            "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details."
        )
        assert str(w[-1].message) == msg
