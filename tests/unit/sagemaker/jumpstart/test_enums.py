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

from sagemaker.jumpstart.enums import NamingConventionType

def test_naming_convention_type_upper_camel_to_snake():
    string_one = "TestUpperCamelCase"
    assert NamingConventionType.upper_camel_to_snake(string_one) == "test_upper_camel_case"
    
    string_two = "TestCaseTwo2"
    assert NamingConventionType.upper_camel_to_snake(string_two) == "test_case_two2"

def test_naming_convention_type_snake_to_upper_camel():
    string_one = "test_snake_case"
    assert NamingConventionType.snake_to_upper_camel(string_one) == "TestSnakeCase"
    
    string_two = "test_case_two2"
    assert NamingConventionType.snake_to_upper_camel(string_two) == "TestCaseTwo2"

def test_naming_convention_type_interchangable():
    string_one_snake = "test_snake_case"
    string_one_snake_to_upper_camel = NamingConventionType.snake_to_upper_camel(string_one_snake)
    assert string_one_snake == NamingConventionType.upper_camel_to_snake(string_one_snake_to_upper_camel)

    string_one_snake_to_upper_camel_to_snake = NamingConventionType.upper_camel_to_snake(string_one_snake_to_upper_camel)
    assert string_one_snake_to_upper_camel_to_snake == string_one_snake