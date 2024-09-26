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
# pylint: skip-file
"""This module contains utilities related to SageMaker JumpStart Hub."""
from __future__ import absolute_import

import re
from typing import Any, Dict, List, Optional


def camel_to_snake(camel_case_string: str) -> str:
    """Converts camelCaseString or UpperCamelCaseString to snake_case_string."""
    snake_case_string = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_string)
    if "-" in snake_case_string:
        # remove any hyphen from the string for accurate conversion.
        snake_case_string = snake_case_string.replace("-", "")
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake_case_string).lower()


def snake_to_upper_camel(snake_case_string: str) -> str:
    """Converts snake_case_string to UpperCamelCaseString."""
    upper_camel_case_string = "".join(word.title() for word in snake_case_string.split("_"))
    return upper_camel_case_string


def walk_and_apply_json(
    json_obj: Dict[Any, Any], apply, stop_keys: Optional[List[str]] = ["metrics"]
) -> Dict[Any, Any]:
    """Recursively walks a json object and applies a given function to the keys.

    stop_keys (Optional[list[str]]): List of field keys that should stop the application function.
        Any children of these keys will not have the application function applied to them.
    """

    def _walk_and_apply_json(json_obj, new):
        if isinstance(json_obj, dict) and isinstance(new, dict):
            for key, value in json_obj.items():
                new_key = apply(key)
                if (stop_keys and new_key not in stop_keys) or stop_keys is None:
                    if isinstance(value, dict):
                        new[new_key] = {}
                        _walk_and_apply_json(value, new=new[new_key])
                    elif isinstance(value, list):
                        new[new_key] = []
                        for item in value:
                            _walk_and_apply_json(item, new=new[new_key])
                    else:
                        new[new_key] = value
                else:
                    new[new_key] = value
        elif isinstance(json_obj, dict) and isinstance(new, list):
            new.append(_walk_and_apply_json(json_obj, new={}))
        elif isinstance(json_obj, list) and isinstance(new, dict):
            new.update(json_obj)
        elif isinstance(json_obj, list) and isinstance(new, list):
            new.append(json_obj)
        elif isinstance(json_obj, str) and isinstance(new, list):
            new.append(json_obj)
        return new

    return _walk_and_apply_json(json_obj, new={})
