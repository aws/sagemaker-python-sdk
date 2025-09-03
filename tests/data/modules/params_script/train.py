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
from typing import List, Dict, Any
from dataclasses import dataclass
from omegaconf import OmegaConf

EXPECTED_HYPERPARAMETERS = {
    "integer": 1,
    "boolean": True,
    "float": 3.14,
    "string": "Hello World",
    "list": [1, 2, 3],
    "dict": {
        "string": "value",
        "integer": 3,
        "float": 3.14,
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

    params = json.loads(os.environ["SM_HPS"])
    print(f"SM_HPS: {params}")
    assert params["string"] == EXPECTED_HYPERPARAMETERS["string"]
    assert params["integer"] == EXPECTED_HYPERPARAMETERS["integer"]
    assert params["boolean"] == EXPECTED_HYPERPARAMETERS["boolean"]
    assert params["float"] == EXPECTED_HYPERPARAMETERS["float"]
    assert params["list"] == EXPECTED_HYPERPARAMETERS["list"]
    assert params["dict"] == EXPECTED_HYPERPARAMETERS["dict"]

    assert isinstance(params, dict)
    assert isinstance(params["string"], str)
    assert isinstance(params["integer"], int)
    assert isinstance(params["boolean"], bool)
    assert isinstance(params["float"], float)
    assert isinstance(params["list"], list)
    assert isinstance(params["dict"], dict)

    params = json.loads(os.environ["SM_TRAINING_ENV"])["hyperparameters"]
    print(f"SM_TRAINING_ENV -> hyperparameters: {params}")
    assert params["string"] == EXPECTED_HYPERPARAMETERS["string"]
    assert params["integer"] == EXPECTED_HYPERPARAMETERS["integer"]
    assert params["boolean"] == EXPECTED_HYPERPARAMETERS["boolean"]
    assert params["float"] == EXPECTED_HYPERPARAMETERS["float"]
    assert params["list"] == EXPECTED_HYPERPARAMETERS["list"]
    assert params["dict"] == EXPECTED_HYPERPARAMETERS["dict"]

    assert isinstance(params, dict)
    assert isinstance(params["string"], str)
    assert isinstance(params["integer"], int)
    assert isinstance(params["boolean"], bool)
    assert isinstance(params["float"], float)
    assert isinstance(params["list"], list)
    assert isinstance(params["dict"], dict)

    # Local JSON - DictConfig OmegaConf
    params = OmegaConf.load("hyperparameters.json")

    print(f"Local hyperparameters.json: {params}")
    assert params.string == EXPECTED_HYPERPARAMETERS["string"]
    assert params.integer == EXPECTED_HYPERPARAMETERS["integer"]
    assert params.boolean == EXPECTED_HYPERPARAMETERS["boolean"]
    assert params.float == EXPECTED_HYPERPARAMETERS["float"]
    assert params.list == EXPECTED_HYPERPARAMETERS["list"]
    assert params.dict == EXPECTED_HYPERPARAMETERS["dict"]
    assert params.dict.string == EXPECTED_HYPERPARAMETERS["dict"]["string"]
    assert params.dict.integer == EXPECTED_HYPERPARAMETERS["dict"]["integer"]
    assert params.dict.boolean == EXPECTED_HYPERPARAMETERS["dict"]["boolean"]
    assert params.dict.float == EXPECTED_HYPERPARAMETERS["dict"]["float"]
    assert params.dict.list == EXPECTED_HYPERPARAMETERS["dict"]["list"]
    assert params.dict.dict == EXPECTED_HYPERPARAMETERS["dict"]["dict"]

    @dataclass
    class DictConfig:
        string: str
        integer: int
        boolean: bool
        float: float
        list: List[int]
        dict: Dict[str, Any]

    @dataclass
    class HPConfig:
        string: str
        integer: int
        boolean: bool
        float: float
        list: List[int]
        dict: DictConfig

    # Local JSON - Structured OmegaConf
    hp_config: HPConfig = OmegaConf.merge(
        OmegaConf.structured(HPConfig), OmegaConf.load("hyperparameters.json")
    )
    print(f"Local hyperparameters.json - Structured: {hp_config}")
    assert hp_config.string == EXPECTED_HYPERPARAMETERS["string"]
    assert hp_config.integer == EXPECTED_HYPERPARAMETERS["integer"]
    assert hp_config.boolean == EXPECTED_HYPERPARAMETERS["boolean"]
    assert hp_config.float == EXPECTED_HYPERPARAMETERS["float"]
    assert hp_config.list == EXPECTED_HYPERPARAMETERS["list"]
    assert hp_config.dict == EXPECTED_HYPERPARAMETERS["dict"]
    assert hp_config.dict.string == EXPECTED_HYPERPARAMETERS["dict"]["string"]
    assert hp_config.dict.integer == EXPECTED_HYPERPARAMETERS["dict"]["integer"]
    assert hp_config.dict.boolean == EXPECTED_HYPERPARAMETERS["dict"]["boolean"]
    assert hp_config.dict.float == EXPECTED_HYPERPARAMETERS["dict"]["float"]
    assert hp_config.dict.list == EXPECTED_HYPERPARAMETERS["dict"]["list"]
    assert hp_config.dict.dict == EXPECTED_HYPERPARAMETERS["dict"]["dict"]

    # Local YAML - Structured OmegaConf
    hp_config: HPConfig = OmegaConf.merge(
        OmegaConf.structured(HPConfig), OmegaConf.load("hyperparameters.yaml")
    )
    print(f"Local hyperparameters.yaml - Structured: {hp_config}")
    assert hp_config.string == EXPECTED_HYPERPARAMETERS["string"]
    assert hp_config.integer == EXPECTED_HYPERPARAMETERS["integer"]
    assert hp_config.boolean == EXPECTED_HYPERPARAMETERS["boolean"]
    assert hp_config.float == EXPECTED_HYPERPARAMETERS["float"]
    assert hp_config.list == EXPECTED_HYPERPARAMETERS["list"]
    assert hp_config.dict == EXPECTED_HYPERPARAMETERS["dict"]
    assert hp_config.dict.string == EXPECTED_HYPERPARAMETERS["dict"]["string"]
    assert hp_config.dict.integer == EXPECTED_HYPERPARAMETERS["dict"]["integer"]
    assert hp_config.dict.boolean == EXPECTED_HYPERPARAMETERS["dict"]["boolean"]
    assert hp_config.dict.float == EXPECTED_HYPERPARAMETERS["dict"]["float"]
    assert hp_config.dict.list == EXPECTED_HYPERPARAMETERS["dict"]["list"]
    assert hp_config.dict.dict == EXPECTED_HYPERPARAMETERS["dict"]["dict"]
    print(f"hyperparameters.yaml -> hyperparameters: {hp_config}")

    # HP Dict - Structured OmegaConf
    hp_dict = json.loads(os.environ["SM_HPS"])
    hp_config: HPConfig = OmegaConf.merge(OmegaConf.structured(HPConfig), OmegaConf.create(hp_dict))
    print(f"SM_HPS - Structured: {hp_config}")
    assert hp_config.string == EXPECTED_HYPERPARAMETERS["string"]
    assert hp_config.integer == EXPECTED_HYPERPARAMETERS["integer"]
    assert hp_config.boolean == EXPECTED_HYPERPARAMETERS["boolean"]
    assert hp_config.float == EXPECTED_HYPERPARAMETERS["float"]
    assert hp_config.list == EXPECTED_HYPERPARAMETERS["list"]
    assert hp_config.dict == EXPECTED_HYPERPARAMETERS["dict"]
    assert hp_config.dict.string == EXPECTED_HYPERPARAMETERS["dict"]["string"]
    assert hp_config.dict.integer == EXPECTED_HYPERPARAMETERS["dict"]["integer"]
    assert hp_config.dict.boolean == EXPECTED_HYPERPARAMETERS["dict"]["boolean"]
    assert hp_config.dict.float == EXPECTED_HYPERPARAMETERS["dict"]["float"]
    assert hp_config.dict.list == EXPECTED_HYPERPARAMETERS["dict"]["list"]
    assert hp_config.dict.dict == EXPECTED_HYPERPARAMETERS["dict"]["dict"]
    print(f"SM_HPS -> hyperparameters: {hp_config}")


if __name__ == "__main__":
    main()
