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
"""This module contains utilities related to SageMaker JumpStart CuratedHub."""
from __future__ import absolute_import

import re
from typing import Any, Dict, List


def camel_to_snake(camel_case_string: str) -> str:
    """Converts camelCaseString or UpperCamelCaseString to snake_case_string."""
    snake_case_string = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_string)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake_case_string).lower()


def snake_to_upper_camel(snake_case_string: str) -> str:
    """Converts snake_case_string to UpperCamelCaseString."""
    upper_camel_case_string = "".join(word.title() for word in snake_case_string.split("_"))
    return upper_camel_case_string


def walk_and_apply_json(json_obj: Dict[Any, Any], apply, keys_to_skip: List[str] = None) -> Dict[Any, Any]:
    """Recursively walks a json object and applies a given function to the keys."""
    if keys_to_skip is None:
        keys_to_skip = []

    def _walk_and_apply_json(json_obj):
      new_object = None
      if isinstance(json_obj, dict):
          new_object = {}
          for key, value in json_obj.items():
              new_key = apply(key)
              new_value = value
              if key not in keys_to_skip:
                  new_value = _walk_and_apply_json(value)
              new_object[new_key] = new_value
      elif isinstance(json_obj, list):
          new_object = []
          for obj in json_obj:
              new_object.append(_walk_and_apply_json(obj))
      else:
          new_object = json_obj
      return new_object

    return _walk_and_apply_json(json_obj)

