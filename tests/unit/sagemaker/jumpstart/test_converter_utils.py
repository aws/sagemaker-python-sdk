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

from sagemaker.jumpstart.converter_utils import (
    camel_to_snake,
    snake_to_upper_camel,
    walk_and_apply_json,
)
from tests.unit.sagemaker.jumpstart.constants import BASE_HUB_NOTEBOOK_DOCUMENT


def test_naming_convention_type_camel_to_snake():
    string_one = "TestUpperCamelCase"
    assert camel_to_snake(string_one) == "test_upper_camel_case"

    string_two = "TestCaseTwo2"
    assert camel_to_snake(string_two) == "test_case_two2"

    string_three = "testCaseCamelString"
    assert camel_to_snake(string_three) == "test_case_camel_string"


def test_naming_convention_type_snake_to_upper_camel():
    string_one = "test_snake_case"
    assert snake_to_upper_camel(string_one) == "TestSnakeCase"

    string_two = "test_case_two2"
    assert snake_to_upper_camel(string_two) == "TestCaseTwo2"


def test_naming_convention_type_interchangable():
    string_one_snake = "test_snake_case"
    string_one_snake_to_upper_camel = snake_to_upper_camel(string_one_snake)
    assert string_one_snake == camel_to_snake(string_one_snake_to_upper_camel)

    string_one_snake_to_camel_to_snake = camel_to_snake(string_one_snake_to_upper_camel)
    assert string_one_snake_to_camel_to_snake == string_one_snake


def test_walk_and_apply_json():
    assert walk_and_apply_json(BASE_HUB_NOTEBOOK_DOCUMENT, camel_to_snake) == {
        "notebook_location": "s3://sagemaker-test-objects-do-not-delete/tensorflow-notebooks/tensorflow-ic-bit-s-r101x3-ilsvrc2012-classification-1-inference.ipynb",
        "dependencies": [
            {
                "dependency_origin_path": "sagemaker-test-objects-do-not-delete/tensorflow-notebooks/tensorflow-ic-bit-s-r101x3-ilsvrc2012-classification-1-inference.ipynb",
                "dependency_copy_path": "sagemaker-hubs-us-west-2-802376408542/default-hub-1667253603.746/Notebook/pentest-3-notebook-1667933000.49/0.0.1/notebook.ipynb",
                "dependency_type": "Notebook",
            }
        ],
    }
