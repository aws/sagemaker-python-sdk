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
from unittest.mock import patch, MagicMock

import yaml
from omegaconf import OmegaConf
from urllib.request import urlretrieve
from tempfile import NamedTemporaryFile

from sagemaker.modules.train.sm_recipes.utils import (
    _load_base_recipe,
    _get_args_from_recipe,
    _load_recipes_cfg,
    _configure_gpu_args,
    _configure_trainium_args,
    _get_trainining_recipe_gpu_model_name_and_script,
    _is_nova_recipe,
    _is_llmft_recipe,
    _get_args_from_nova_recipe,
    _get_args_from_llmft_recipe,
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


@patch("sagemaker.modules.train.sm_recipes.utils._get_args_from_nova_recipe")
def test_get_args_from_recipe_with_nova_and_role(mock_get_args_from_nova_recipe, temporary_recipe):
    # Set up mock return value
    mock_args = {"hyperparameters": {}}
    mock_dir = MagicMock()
    mock_get_args_from_nova_recipe.return_value = (mock_args, mock_dir)

    # Create a Nova recipe with distillation data
    recipe = OmegaConf.create(
        {"training_config": {"distillation_data": True, "kms_key": "alias/my-kms-key"}}
    )
    compute = Compute(instance_type="ml.g5.xlarge")
    role = "arn:aws:iam::123456789012:role/SageMakerRole"

    # Mock the Nova recipe detection to return True
    with patch("sagemaker.modules.train.sm_recipes.utils._is_nova_recipe", return_value=True):
        _get_args_from_recipe(
            training_recipe=recipe,
            compute=compute,
            region_name="us-west-2",
            recipe_overrides=None,
            requirements=None,
            role=role,
        )

        # Verify _get_args_from_nova_recipe was called with the role parameter
        mock_get_args_from_nova_recipe.assert_called_once_with(recipe, compute, role=role)


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
        {
            "model_type": "gpt_oss",
            "script": "custom_pretrain.py",
            "model_base_name": "custom_model",
        },
    ],
)
def test_get_trainining_recipe_gpu_model_name_and_script(test_case):
    model_type = test_case["model_type"]
    script = test_case["script"]
    model_base_name, script = _get_trainining_recipe_gpu_model_name_and_script(model_type)
    assert model_base_name == test_case["model_base_name"]
    assert script == test_case["script"]


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "recipe": {
                "run": {
                    "model_type": "amazon.nova",
                    "model_name_or_path": "some-model",
                }
            },
            "is_nova": True,
        },
        {
            "recipe": {
                "run": {
                    "model_type": "amazon.nova.other",
                    "model_name_or_path": "some-model",
                }
            },
            "is_nova": True,
        },
        {"recipe": {"run": {"model_type": "amazon.nova.other"}}, "is_nova": False},
        {
            "recipe": {"run": {"model_type": "other.model", "model_name_or_path": "some-model"}},
            "is_nova": False,
        },
        {
            "recipe": {"training_config": {"distillation_data": "s3://bucket/distillation-data"}},
            "is_nova": True,
        },
        {
            "recipe": {"training_config": {"some_other_field": "value"}},
            "is_nova": False,
        },
    ],
    ids=[
        "nova_model",
        "nova_model_subtype",
        "nova_missing_model_path",
        "non_nova_model",
        "distillation_data",
        "no_distillation_data",
    ],
)
def test_is_nova_recipe(test_case):
    recipe = OmegaConf.create(test_case["recipe"])
    is_nova = _is_nova_recipe(recipe)
    assert is_nova == test_case["is_nova"]


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "recipe": {
                "run": {"model_type": "amazon.nova", "model_name_or_path": "dummy-test"},
            },
            "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
            "expected_args": {
                "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
                "hyperparameters": {"base_model": "dummy-test"},
                "training_image": None,
                "source_code": None,
                "distributed": None,
            },
        },
        {
            "recipe": {
                "run": {
                    "model_type": "amazon.nova",
                    "model_name_or_path": "s3://bucket/path/to/model",
                },
            },
            "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
            "expected_args": {
                "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
                "hyperparameters": {"base_model_location": "s3://bucket/path/to/model"},
                "training_image": None,
                "source_code": None,
                "distributed": None,
            },
        },
        {
            "recipe": {
                "run": {
                    "model_type": "amazon.nova",
                    "model_name_or_path": "s3://bucket/path/to/model",
                    "replicas": 4,
                },
            },
            "compute": Compute(instance_type="ml.m5.xlarge"),
            "expected_args": {
                "compute": Compute(instance_type="ml.m5.xlarge", instance_count=4),
                "hyperparameters": {"base_model_location": "s3://bucket/path/to/model"},
                "training_image": None,
                "source_code": None,
                "distributed": None,
            },
        },
        {
            "recipe": {
                "run": {
                    "model_type": "amazon.nova",
                    "model_name_or_path": "s3://bucket/path/to/model",
                    "replicas": 2,
                },
            },
            "compute": Compute(instance_type="ml.m5.xlarge", instance_count=4),
            "expected_args": {
                "compute": Compute(instance_type="ml.m5.xlarge", instance_count=4),
                "hyperparameters": {"base_model_location": "s3://bucket/path/to/model"},
                "training_image": None,
                "source_code": None,
                "distributed": None,
            },
        },
    ],
)
def test_get_args_from_nova_recipe(test_case):
    recipe = OmegaConf.create(test_case["recipe"])
    args, _ = _get_args_from_nova_recipe(recipe=recipe, compute=test_case["compute"])
    assert args == test_case["expected_args"]


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "recipe": {
                "training_config": {
                    "distillation_data": "s3://bucket/distillation-data",
                    "kms_key": "alias/my-kms-key",
                }
            },
            "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
            "role": "arn:aws:iam::123456789012:role/SageMakerRole",
            "expected_args": {
                "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
                "hyperparameters": {
                    "distillation_data": "s3://bucket/distillation-data",
                    "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
                    "kms_key": "alias/my-kms-key",
                },
                "training_image": None,
                "source_code": None,
                "distributed": None,
            },
        },
    ],
)
def test_get_args_from_nova_recipe_with_distillation(test_case):
    recipe = OmegaConf.create(test_case["recipe"])
    args, _ = _get_args_from_nova_recipe(
        recipe=recipe, compute=test_case["compute"], role=test_case["role"]
    )
    assert args == test_case["expected_args"]


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "recipe": {
                "training_config": {
                    "distillation_data": "s3://bucket/distillation-data",
                    # Missing kms_key
                }
            },
            "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
            "role": "arn:aws:iam::123456789012:role/SageMakerRole",
        },
        {
            "recipe": {
                "training_config": {
                    "distillation_data": "s3://bucket/distillation-data",
                    "kms_key": "alias/my-kms-key",
                }
            },
            "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
            # Missing role
            "role": None,
        },
    ],
    ids=[
        "missing_kms_key",
        "missing_role",
    ],
)
def test_get_args_from_nova_recipe_with_distillation_errors(test_case):
    recipe = OmegaConf.create(test_case["recipe"])
    with pytest.raises(ValueError):
        _get_args_from_nova_recipe(
            recipe=recipe, compute=test_case["compute"], role=test_case.get("role")
        )


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "recipe": {
                "evaluation": {"task:": "gen_qa", "strategy": "gen_qa", "metric": "all"},
                "processor": {
                    "lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:MyLambdaFunction"
                },
            },
            "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
            "role": "arn:aws:iam::123456789012:role/SageMakerRole",
            "expected_args": {
                "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
                "hyperparameters": {
                    "eval_lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:MyLambdaFunction",
                },
                "training_image": None,
                "source_code": None,
                "distributed": None,
            },
        },
    ],
)
def test_get_args_from_nova_recipe_with_evaluation(test_case):
    recipe = OmegaConf.create(test_case["recipe"])
    args, _ = _get_args_from_nova_recipe(
        recipe=recipe, compute=test_case["compute"], role=test_case["role"]
    )
    assert args == test_case["expected_args"]


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "recipe": {
                "run": {
                    "name": "dummy-model",
                    "model_type": "llm_finetuning_aws",
                },
                "trainer": {"num_nodes": "12"},
                "training_config": {"model_save_name": "xyz"},
            },
            "is_llmft": True,
        },
        {
            "recipe": {
                "run": {
                    "name": "dummy-model",
                    "model_type": "llm_finetuning_aws",
                },
                "training_config": {"model_save_name": "xyz"},
            },
            "is_llmft": True,
        },
        {
            "recipe": {
                "run": {
                    "name": "dummy-model",
                    "model_type": "llm_finetuning_aws",
                },
            },
            "is_llmft": False,
        },
        {
            "recipe": {
                "run": {
                    "name": "dummy-model",
                    "model_type": "xyz",
                },
                "training_config": {"model_save_name": "xyz"},
            },
            "is_llmft": False,
        },
    ],
    ids=[
        "llmft_model",
        "llmft_model_subtype",
        "llmft_missing_training_config",
        "non_llmft_model",
    ],
)
def test_is_llmft_recipe(test_case):
    recipe = OmegaConf.create(test_case["recipe"])
    is_llmft = _is_llmft_recipe(recipe)
    assert is_llmft == test_case["is_llmft"]


@patch("sagemaker.modules.train.sm_recipes.utils._get_args_from_llmft_recipe")
def test_get_args_from_recipe_with_llmft_and_role(mock_get_args_from_llmft_recipe):
    # Set up mock return value
    mock_args = {}
    mock_dir = MagicMock()
    mock_get_args_from_llmft_recipe.return_value = (mock_args, mock_dir)

    recipe = {
        "run": {
            "name": "dummy-model",
            "model_type": "llm_finetuning_aws",
        },
        "trainer": {"num_nodes": "12"},
        "training_config": {"model_save_name": "xyz"},
    }
    compute = Compute(instance_type="ml.g5.xlarge")
    role = "arn:aws:iam::123456789012:role/SageMakerRole"

    # Mock the LLMFT recipe detection to return True
    with patch("sagemaker.modules.train.sm_recipes.utils._is_llmft_recipe", return_value=True):
        _get_args_from_recipe(
            training_recipe=recipe,
            compute=compute,
            region_name="us-west-2",
            recipe_overrides=None,
            requirements=None,
            role=role,
        )

        # Verify _get_args_from_llmft_recipe was called
        mock_get_args_from_llmft_recipe.assert_called_once_with(recipe, compute)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "recipe": {
                "run": {
                    "name": "dummy-model",
                    "model_type": "llm_finetuning_aws",
                },
                "trainer": {"num_nodes": "12"},
                "training_config": {"model_save_name": "xyz"},
            },
            "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
            "expected_args": {
                "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
                "training_image": None,
                "source_code": None,
                "distributed": None,
            },
        },
        {
            "recipe": {
                "run": {
                    "name": "dummy-model",
                    "model_type": "llm_finetuning_aws",
                },
                "training_config": {"model_save_name": "xyz"},
            },
            "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
            "expected_args": {
                "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
                "training_image": None,
                "source_code": None,
                "distributed": None,
            },
        },
    ],
)
def test_get_args_from_llmft_recipe(test_case):
    recipe = OmegaConf.create(test_case["recipe"])
    args, _ = _get_args_from_llmft_recipe(recipe=recipe, compute=test_case["compute"])
    assert args == test_case["expected_args"]


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "recipe": {
                "run": {
                    "model_type": "amazon.nova",
                    "model_name_or_path": "dummy-test",
                    "reward_lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:MyRewardLambdaFunction",
                },
            },
            "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
            "role": "arn:aws:iam::123456789012:role/SageMakerRole",
            "expected_args": {
                "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
                "hyperparameters": {
                    "base_model": "dummy-test",
                    "reward_lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:MyRewardLambdaFunction",
                },
                "training_image": None,
                "source_code": None,
                "distributed": None,
            },
        },
        {
            "recipe": {
                "run": {
                    "model_type": "amazon.nova",
                    "model_name_or_path": "dummy-test",
                    # No reward_lambda_arn - should not be in hyperparameters
                },
            },
            "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
            "role": "arn:aws:iam::123456789012:role/SageMakerRole",
            "expected_args": {
                "compute": Compute(instance_type="ml.m5.xlarge", instance_count=2),
                "hyperparameters": {
                    "base_model": "dummy-test",
                },
                "training_image": None,
                "source_code": None,
                "distributed": None,
            },
        },
    ],
)
def test_get_args_from_nova_recipe_with_reward_lambda(test_case):
    recipe = OmegaConf.create(test_case["recipe"])
    args, _ = _get_args_from_nova_recipe(
        recipe=recipe, compute=test_case["compute"], role=test_case["role"]
    )
    assert args == test_case["expected_args"]
