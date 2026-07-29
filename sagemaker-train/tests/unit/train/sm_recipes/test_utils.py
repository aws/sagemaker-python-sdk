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

from sagemaker.train.sm_recipes.utils import (
    _load_base_recipe,
    _drop_unknown_recipe_overrides,
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
from sagemaker.train.utils import _run_clone_command_silent
from sagemaker.train.configs import Compute


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


def test_load_base_recipe_drops_unknown_overrides(
    temporary_recipe, training_recipes_cfg, caplog
):
    """Override keys absent from the recipe are dropped (and warned), not injected.

    Regression test for the serverful SMTJ path (bug 3): overriding e.g.
    ``max_steps`` for a model whose recipe has no such field used to inject the
    bogus key unvalidated. It must now be dropped with a warning, while valid
    overrides still apply.
    """
    recipe_overrides = {
        "trainer": {"max_epochs": 20, "max_steps": 999},  # max_steps not in recipe
        "model": {"num_layers": 15},
        "totally_bogus_section": {"x": 1},  # whole section not in recipe
    }

    with caplog.at_level("WARNING"):
        load_recipe = _load_base_recipe(
            training_recipe=temporary_recipe,
            recipe_overrides=recipe_overrides,
            training_recipes_cfg=training_recipes_cfg,
        )

    # Valid overrides applied.
    assert load_recipe["trainer"]["max_epochs"] == 20
    assert load_recipe["model"]["num_layers"] == 15
    # Unknown keys dropped, not injected.
    assert "max_steps" not in load_recipe["trainer"]
    assert "totally_bogus_section" not in load_recipe
    # Original recipe values preserved where not overridden.
    assert load_recipe["trainer"]["num_nodes"] == 2
    # Each dropped key surfaced as a warning.
    messages = [r.getMessage() for r in caplog.records]
    assert any("trainer.max_steps" in m and "dropped" in m for m in messages)
    assert any("totally_bogus_section" in m and "dropped" in m for m in messages)


class TestDropUnknownRecipeOverrides:
    """Unit tests for the _drop_unknown_recipe_overrides helper."""

    def test_keeps_known_keys(self):
        base = OmegaConf.create({"a": 1, "b": {"c": 2}})
        result = _drop_unknown_recipe_overrides({"a": 10, "b": {"c": 20}}, base)
        assert result == {"a": 10, "b": {"c": 20}}

    def test_drops_unknown_top_level_key(self):
        base = OmegaConf.create({"a": 1})
        result = _drop_unknown_recipe_overrides({"a": 10, "bogus": 5}, base)
        assert result == {"a": 10}

    def test_drops_unknown_nested_key_keeps_sibling(self):
        base = OmegaConf.create({"trainer": {"max_epochs": 10}})
        result = _drop_unknown_recipe_overrides(
            {"trainer": {"max_epochs": 20, "max_steps": 99}}, base
        )
        assert result == {"trainer": {"max_epochs": 20}}

    def test_plain_dict_base_recipe(self):
        # Works with a plain dict base, not only OmegaConf mappings.
        result = _drop_unknown_recipe_overrides(
            {"a": 1, "bogus": 2}, {"a": 0}
        )
        assert result == {"a": 1}

    def test_empty_overrides(self):
        base = OmegaConf.create({"a": 1})
        assert _drop_unknown_recipe_overrides({}, base) == {}


@pytest.mark.parametrize(
    "test_case",
    [
        {"recipe_type": "local"},
        {"recipe_type": "sagemaker"},
        {"recipe_type": "url"},
        {"recipe_type": "not_found"},
    ],
)
@patch("sagemaker.train.sm_recipes.utils.urlretrieve")
@patch("sagemaker.train.sm_recipes.utils._run_clone_command_silent")
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
        # Mock the clone to do nothing and mock file operations
        mock_clone.return_value = None
        
        # Create a mock recipe in the expected structure
        import os
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the expected directory structure
            recipes_dir = os.path.join(temp_dir, "recipes_collection", "recipes", "training", "llama")
            os.makedirs(recipes_dir, exist_ok=True)
            
            # Create a mock recipe file
            recipe_path = os.path.join(recipes_dir, "p4_hf_llama3_70b_seq8k_gpu.yaml")
            with open(recipe_path, 'w') as f:
                yaml.dump({"trainer": {"num_nodes": 1}, "model": {"model_type": "llama"}}, f)
            
            # Patch the TemporaryDirectory to return our temp dir
            with patch('tempfile.TemporaryDirectory') as mock_temp:
                mock_temp_obj = MagicMock()
                mock_temp_obj.name = temp_dir
                mock_temp.return_value = mock_temp_obj
                
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
@patch("sagemaker.train.sm_recipes.utils._configure_gpu_args")
@patch("sagemaker.train.sm_recipes.utils._configure_trainium_args")
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

@pytest.mark.parametrize(
    "test_case",
    [
        {
            "model_type": "llama_v3",
            "model_base_name": "llama",
            "script": "llama_pretrain.py",
        },
        {
            "model_type": "mistral",
            "model_base_name": "mistral",
            "script": "mistral_pretrain.py",
        },
        {
            "model_type": "deepseek_llamav3",
            "model_base_name": "deepseek",
            "script": "deepseek_pretrain.py",
        },
        {
            "model_type": "deepseek_qwenv2",
            "model_base_name": "deepseek",
            "script": "deepseek_pretrain.py",
        },
    ],
)
def test_get_trainining_recipe_gpu_model_name_and_script(test_case):
    model_base_name, script = _get_trainining_recipe_gpu_model_name_and_script(
        test_case["model_type"]
    )
    assert model_base_name == test_case["model_base_name"]
    assert script == test_case["script"]


def test_get_args_from_recipe_with_evaluation(temporary_recipe):
    import tempfile
    import os
    from sagemaker.train.configs import SourceCode
    
    # Create a recipe with evaluation config
    recipe_data = {
        "trainer": {"num_nodes": 1},
        "model": {"model_type": "llama_v3"},
        "evaluation": {"task": "gen_qa"},
        "processor": {"lambda_arn": "arn:aws:lambda:us-east-1:123456789012:function:MyFunc"},
    }
    
    with NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        with open(f.name, "w") as file:
            yaml.dump(recipe_data, file)
        recipe_path = f.name
    
    try:
        compute = Compute(instance_type="ml.p4d.24xlarge", instance_count=1)
        with patch("sagemaker.train.sm_recipes.utils._configure_gpu_args") as mock_gpu:
            mock_source = SourceCode()
            mock_source.source_dir = "/tmp/test"
            mock_gpu.return_value = {"source_code": mock_source, "hyperparameters": {}}
            with patch("sagemaker.train.sm_recipes.utils.OmegaConf.save"):
                args, _ = _get_args_from_recipe(
                    training_recipe=recipe_path,
                    compute=compute,
                    region_name="us-west-2",
                    recipe_overrides=None,
                    requirements=None,
                )
                assert args["hyperparameters"]["lambda_arn"] == "arn:aws:lambda:us-east-1:123456789012:function:MyFunc"
    finally:
        os.unlink(recipe_path)

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
        {
            "recipe": {
                "run": {
                    "name": "verl-grpo-llama",
                    "model_type": "verl",
                },
                "trainer": {"num_nodes": "1"},
                "training_config": {"trainer": {"total_epochs": 2}},
            },
            "is_llmft": True,
        },
        {
            "recipe": {
                "run": {
                    "name": "verl-grpo-llama",
                    "model_type": "verl",
                },
            },
            "is_llmft": False,
        },
        {
            # OSS SMTJ evaluation recipe: no model_type/trainer/training_config,
            # but has an `evaluation` section -> routed via the LLMFT path.
            "recipe": {
                "run": {
                    "name": "dummy-eval",
                    "model_name_or_path": "{{model_name_or_path}}",
                },
                "evaluation": {"task": "{{task}}", "metric": "{{evaluation_metric}}"},
                "inference": {"max_new_tokens": "{{max_new_tokens}}"},
            },
            "is_llmft": True,
        },
        {
            # Nova evaluation recipe (run.model_type=amazon.nova) must NOT be
            # classified as LLMFT even though it has an `evaluation` section.
            "recipe": {
                "run": {
                    "name": "dummy-eval",
                    "model_type": "amazon.nova-lite-v1:0",
                    "model_name_or_path": "nova-lite/prod",
                },
                "evaluation": {"task": "{{task}}"},
            },
            "is_llmft": False,
        },
    ],
    ids=[
        "llmft_model",
        "llmft_model_subtype",
        "llmft_missing_training_config",
        "non_llmft_model",
        "verl_model",
        "verl_missing_training_config",
        "oss_eval_recipe",
        "nova_eval_recipe_excluded",
    ],
)
def test_is_llmft_recipe(test_case):
    recipe = OmegaConf.create(test_case["recipe"])
    is_llmft = _is_llmft_recipe(recipe)
    assert is_llmft == test_case["is_llmft"]


@patch("sagemaker.train.sm_recipes.utils._get_args_from_llmft_recipe")
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
    with patch("sagemaker.train.sm_recipes.utils._is_llmft_recipe", return_value=True):
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


class TestGetArgsFromNovaRecipeModelPackageConfig:
    """Tests for ModelPackageConfig support in _get_args_from_nova_recipe."""

    def test_mp_arn_routes_to_model_package_config(self):
        """MP ARN in model_name_or_path should go to ModelPackageConfig."""
        mp_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-mpg/1"
        recipe = OmegaConf.create({
            "run": {
                "name": "test",
                "model_type": "amazon.nova",
                "model_name_or_path": mp_arn,
                "replicas": 1,
            }
        })
        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=1)
        args, _ = _get_args_from_nova_recipe(recipe, compute)
        assert args["model_package_config"]["source_model_package_arn"] == mp_arn
        assert "base_model" not in args["hyperparameters"]
        assert "base_model_location" not in args["hyperparameters"]
        assert "source_model_package" not in args["hyperparameters"]

    def test_mpg_from_recipe(self):
        """model_package_group in recipe should map to ModelPackageGroupArn."""
        mpg_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/my-mpg"
        recipe = OmegaConf.create({
            "run": {
                "name": "test",
                "model_type": "amazon.nova",
                "model_name_or_path": "nova-pro",
                "model_package_group": mpg_arn,
                "replicas": 1,
            }
        })
        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=1)
        args, _ = _get_args_from_nova_recipe(recipe, compute)
        assert args["model_package_config"]["model_package_group_arn"] == mpg_arn
        assert args["hyperparameters"]["base_model"] == "nova-pro"

    def test_mp_arn_and_mpg_together(self):
        """Both MP ARN and MPG should be in model_package_config."""
        mp_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-mpg/1"
        mpg_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/my-mpg"
        recipe = OmegaConf.create({
            "run": {
                "name": "test",
                "model_type": "amazon.nova",
                "model_name_or_path": mp_arn,
                "model_package_group": mpg_arn,
                "replicas": 1,
            }
        })
        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=1)
        args, _ = _get_args_from_nova_recipe(recipe, compute)
        assert args["model_package_config"] == {
            "source_model_package_arn": mp_arn,
            "model_package_group_arn": mpg_arn,
        }

    def test_s3_path_still_goes_to_hyperparameters(self):
        """S3 path should still go to base_model_location HP (legacy path)."""
        recipe = OmegaConf.create({
            "run": {
                "name": "test",
                "model_type": "amazon.nova",
                "model_name_or_path": "s3://escrow-bucket/job-1/checkpoints/step_5",
                "replicas": 1,
            }
        })
        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=1)
        args, _ = _get_args_from_nova_recipe(recipe, compute)
        assert args["hyperparameters"]["base_model_location"] == "s3://escrow-bucket/job-1/checkpoints/step_5"
        assert "model_package_config" not in args

    def test_model_name_still_goes_to_hyperparameters(self):
        """Model name should still go to base_model HP."""
        recipe = OmegaConf.create({
            "run": {
                "name": "test",
                "model_type": "amazon.nova",
                "model_name_or_path": "nova-pro",
                "replicas": 1,
            }
        })
        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=1)
        args, _ = _get_args_from_nova_recipe(recipe, compute)
        assert args["hyperparameters"]["base_model"] == "nova-pro"
        assert "model_package_config" not in args

    def test_invalid_arn_treated_as_model_name(self):
        """Invalid ARN (wrong account ID format) should be treated as model name."""
        recipe = OmegaConf.create({
            "run": {
                "name": "test",
                "model_type": "amazon.nova",
                "model_name_or_path": "arn:aws:sagemaker:us-east-1:123:model-package/x",
                "replicas": 1,
            }
        })
        compute = Compute(instance_type="ml.p5.48xlarge", instance_count=1)
        args, _ = _get_args_from_nova_recipe(recipe, compute)
        assert args["hyperparameters"]["base_model"] == "arn:aws:sagemaker:us-east-1:123:model-package/x"
        assert "model_package_config" not in args
