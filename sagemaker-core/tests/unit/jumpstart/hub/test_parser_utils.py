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

from sagemaker.core.jumpstart.hub.parser_utils import (
    camel_to_snake,
    snake_to_upper_camel,
    walk_and_apply_json,
)


VARIANTS = {
    "Variants": {
        "g5": {
            "Properties": {
                "ImageUri": "image",
                "EnvironmentVariables": {
                    "SM_VLLM_MAX_MODEL_LEN": "131072",
                    "HF_HUB_OFFLINE": "1",
                },
                "Metrics": [{"Name": "loss", "Regex": "loss=(.*)"}],
            }
        }
    }
}


def test_walk_and_apply_json_camel_to_snake_preserves_environment_variable_names():
    result = walk_and_apply_json(VARIANTS, camel_to_snake)

    properties = result["variants"]["g5"]["properties"]
    assert properties["image_uri"] == "image"
    assert properties["environment_variables"] == {
        "SM_VLLM_MAX_MODEL_LEN": "131072",
        "HF_HUB_OFFLINE": "1",
    }
    assert properties["metrics"] == [{"Name": "loss", "Regex": "loss=(.*)"}]


def test_walk_and_apply_json_snake_to_upper_camel_preserves_environment_variable_names():
    snake = walk_and_apply_json(VARIANTS, camel_to_snake)

    result = walk_and_apply_json(snake, snake_to_upper_camel)

    properties = result["Variants"]["G5"]["Properties"]
    assert properties["ImageUri"] == "image"
    assert properties["EnvironmentVariables"] == {
        "SM_VLLM_MAX_MODEL_LEN": "131072",
        "HF_HUB_OFFLINE": "1",
    }


def test_walk_and_apply_json_round_trip_preserves_environment_variable_names():
    """The private hub pipeline converts the same document several times in both directions."""
    result = walk_and_apply_json(VARIANTS, camel_to_snake)
    result = walk_and_apply_json(result, snake_to_upper_camel)
    result = walk_and_apply_json(result, camel_to_snake)
    result = walk_and_apply_json(result, camel_to_snake)

    assert result["variants"]["g5"]["properties"]["environment_variables"] == {
        "SM_VLLM_MAX_MODEL_LEN": "131072",
        "HF_HUB_OFFLINE": "1",
    }


def test_walk_and_apply_json_explicit_stop_keys_still_honored():
    result = walk_and_apply_json(
        {"Outer": {"Inner": {"KeepMe": 1}}}, camel_to_snake, stop_keys=["inner"]
    )

    assert result == {"outer": {"inner": {"KeepMe": 1}}}


def test_walk_and_apply_json_no_stop_keys_converts_everything():
    result = walk_and_apply_json(VARIANTS, camel_to_snake, stop_keys=None)

    properties = result["variants"]["g5"]["properties"]
    assert "SM_VLLM_MAX_MODEL_LEN" not in properties["environment_variables"]
    assert properties["metrics"] == [{"name": "loss", "regex": "loss=(.*)"}]
