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
"""Utility functions for SageMaker training recipes Tests."""
from __future__ import absolute_import

import pytest
from unittest.mock import patch

import yaml
from urllib.request import urlretrieve
from tempfile import NamedTemporaryFile

from sagemaker.modules.train.sm_recipes.utils import (
    _load_base_recipe,
    _get_args_from_recipe,
    _load_recipes_cfg,
    _configure_gpu_args,
    _configure_trainium_args,
    _get_trainining_recipe_gpu_model_name_and_script,
)
from sagemaker.modules.utils import _run_clone_command_silent
from sagemaker.modules.configs import Compute


@pytest.fixture(scope="module")
def training_recipes_cfg():
    return _load_recipes_cfg()


@pytest.fixture(scope="module")
def temporary_recipe():
    data = {
        "trainer": {"num_nodes": 2, "max_epochs": 10},
        "model": {"model_type": "llama_v3", "num_classes": 10, "num_layers": 10},
    }
    with NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        with open(f.name, "w") as file:
            yaml.dump(data, file)
        yield f.name


def test_load_base_recipe_with_overrides(temporary_recipe, training_recipes_cfg):
    expected_epochs = 20
    expected_layers = 15

    recipe_overrides = {
        "trainer": {"max_epochs": expected_epochs},
        "model": {"num_layers": expected_layers},
    }

    load_recipe = _load_base_recipe(
        training_recipe=temporary_recipe,
        recipe_overrides=recipe_overrides,
        training_recipes_cfg=training_recipes_cfg,
    )

    assert (
        load_recipe["trainer"]["max_epochs"] == expected_epochs
        and load_recipe["model"]["num_layers"] == expected_layers
    )


@pytest.mark.parametrize(
    "test_case",
    [
        {"recipe_type": "local"},
        {"recipe_type": "sagemaker"},
        {"recipe_type": "url"},
        {"recipe_type": "not_found"},
    ],
)
@patch("sagemaker.modules.train.sm_recipes.utils.urlretrieve")
@patch("sagemaker.modules.train.sm_recipes.utils._run_clone_command_silent")
def test_load_base_recipe_types(
    mock_clone, mock_retrieve, temporary_recipe, training_recipes_cfg, test_case
):
    recipe_type = test_case["recipe_type"]

    if recipe_type == "not_found":
        with pytest.raises(ValueError):
            _load_base_recipe(
                training_recipe="not_found",
                recipe_overrides=None,
                training_recipes_cfg=training_recipes_cfg,
            )

    if recipe_type == "local":
        load_recipe = _load_base_recipe(
            training_recipe=temporary_recipe,
            recipe_overrides=None,
            training_recipes_cfg=training_recipes_cfg,
        )
        assert load_recipe is not None
        assert "trainer" in load_recipe

    if recipe_type == "sagemaker":
        mock_clone.side_effect = _run_clone_command_silent
        load_recipe = _load_base_recipe(
            training_recipe="training/llama/p4_hf_llama3_70b_seq8k_gpu",
            recipe_overrides=None,
            training_recipes_cfg=training_recipes_cfg,
        )
        assert load_recipe is not None
        assert "trainer" in load_recipe
        assert mock_clone.call_args.args[0] == training_recipes_cfg.get("launcher_repo")

    if recipe_type == "url":
        url = "https://raw.githubusercontent.com/aws-neuron/neuronx-distributed-training/refs/heads/main/examples/conf/hf_llama3_8B_config.yaml"  # noqa
        mock_retrieve.side_effect = urlretrieve
        load_recipe = _load_base_recipe(
            training_recipe=url,
            recipe_overrides=None,
            training_recipes_cfg=training_recipes_cfg,
        )
        assert load_recipe is not None
        assert "trainer" in load_recipe
        assert mock_retrieve.call_args.args[0] == url


@pytest.mark.parametrize(
    "test_case",
    [
        {"type": "gpu", "instance_type": "ml.p4d.24xlarge"},
        {"type": "trn", "instance_type": "ml.trn1.32xlarge"},
        {"type": "cpu", "instance_type": "ml.c5.4xlarge"},
    ],
)
@patch("sagemaker.modules.train.sm_recipes.utils._configure_gpu_args")
@patch("sagemaker.modules.train.sm_recipes.utils._configure_trainium_args")
def test_get_args_from_recipe_compute(
    mock_trainium_args, mock_gpu_args, temporary_recipe, test_case
):
    compute = Compute(instance_type=test_case["instance_type"])
    if test_case["type"] == "gpu":
        mock_gpu_args.side_effect = _configure_gpu_args

        args = _get_args_from_recipe(
            training_recipe=temporary_recipe,
            compute=compute,
            region_name="us-west-2",
            recipe_overrides=None,
            requirements=None,
        )
        assert mock_gpu_args.call_count == 1
        assert mock_trainium_args.call_count == 0

    if test_case["type"] == "trn":
        mock_trainium_args.side_effect = _configure_trainium_args

        args = _get_args_from_recipe(
            training_recipe=temporary_recipe,
            compute=compute,
            region_name="us-west-2",
            recipe_overrides=None,
            requirements=None,
        )
        assert mock_gpu_args.call_count == 0
        assert mock_trainium_args.call_count == 1

    if test_case["type"] == "cpu":
        with pytest.raises(ValueError):
            args = _get_args_from_recipe(
                training_recipe=temporary_recipe,
                compute=compute,
                region_name="us-west-2",
                recipe_overrides=None,
                requirements=None,
            )
            assert mock_gpu_args.call_count == 0
            assert mock_trainium_args.call_count == 0
            assert args is None


@pytest.mark.parametrize(
    "test_case",
    [
        {"model_type": "llama_v4", "script": "llama_pretrain.py", "model_base_name": "llama"},
        {
            "model_type": "llama_v3",
            "script": "llama_pretrain.py",
            "model_base_name": "llama",
        },
        {
            "model_type": "mistral",
            "script": "mistral_pretrain.py",
            "model_base_name": "mistral",
        },
        {
            "model_type": "deepseek_llamav3",
            "script": "deepseek_pretrain.py",
            "model_base_name": "deepseek",
        },
        {
            "model_type": "deepseek_qwenv2",
            "script": "deepseek_pretrain.py",
            "model_base_name": "deepseek",
        },
    ],
)
def test_get_trainining_recipe_gpu_model_name_and_script(test_case):
    model_type = test_case["model_type"]
    script = test_case["script"]
    model_base_name, script = _get_trainining_recipe_gpu_model_name_and_script(model_type)
    assert model_base_name == test_case["model_base_name"]
    assert script == test_case["script"]
