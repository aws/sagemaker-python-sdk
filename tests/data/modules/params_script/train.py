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
"""Script to test hyperparameters contract."""
from __future__ import absolute_import

import argparse
import json
import os

EXPECTED_HYPERPARAMETERS = {
    "integer": 1,
    "boolean": True,
    "float": 3.14,
    "string": "Hello World",
    "list": [1, 2, 3],
    "dict": {
        "string": "value",
        "integer": 3,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "boolean": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test Hyperparameters")
    parser.add_argument(
        "--string",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--integer",
        type=int,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--float",
        type=float,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--boolean",
        type=lambda x: json.loads(x),
        default=None,
        required=True,
    )
    parser.add_argument(
        "--list",
        type=lambda x: json.loads(x),
        default=None,
        required=True,
    )
    parser.add_argument(
        "--dict",
        type=lambda x: json.loads(x),
        default=None,
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    assert isinstance(args.string, str)
    assert isinstance(args.integer, int)
    assert isinstance(args.boolean, bool)
    assert isinstance(args.float, float)
    assert isinstance(args.list, list)
    assert isinstance(args.dict, dict)

    assert args.string == EXPECTED_HYPERPARAMETERS["string"]
    assert args.integer == EXPECTED_HYPERPARAMETERS["integer"]
    assert args.boolean == EXPECTED_HYPERPARAMETERS["boolean"]
    assert args.float == EXPECTED_HYPERPARAMETERS["float"]
    assert args.list == EXPECTED_HYPERPARAMETERS["list"]
    assert args.dict == EXPECTED_HYPERPARAMETERS["dict"]

    assert os.environ["SM_HP_STRING"] == EXPECTED_HYPERPARAMETERS["string"]
    assert int(os.environ["SM_HP_INTEGER"]) == EXPECTED_HYPERPARAMETERS["integer"]
    assert float(os.environ["SM_HP_FLOAT"]) == EXPECTED_HYPERPARAMETERS["float"]
    assert json.loads(os.environ["SM_HP_BOOLEAN"]) == EXPECTED_HYPERPARAMETERS["boolean"]
    assert json.loads(os.environ["SM_HP_LIST"]) == EXPECTED_HYPERPARAMETERS["list"]
    assert json.loads(os.environ["SM_HP_DICT"]) == EXPECTED_HYPERPARAMETERS["dict"]


if __name__ == "__main__":
    main()
