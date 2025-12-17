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
"""Utility functions for SageMaker training recipes."""
from __future__ import absolute_import

import math
import os
import json
import shutil
import tempfile
from urllib.request import urlretrieve
from typing import Dict, Any, Optional, Tuple

import omegaconf
from omegaconf import OmegaConf, dictconfig

from sagemaker.core.image_uris import retrieve

from sagemaker.core.modules import logger
from sagemaker.core.modules.utils import _run_clone_command_silent
from sagemaker.core.modules.configs import Compute, SourceCode
from sagemaker.core.modules.distributed import Torchrun, SMP


def _try_resolve_recipe(recipe, key=None):
    """Try to resolve recipe and return resolved recipe."""
    if key is not None:
        recipe = dictconfig.DictConfig({key: recipe})
    try:
        OmegaConf.resolve(recipe)
    except omegaconf.errors.OmegaConfBaseException:
        return None
    if key is None:
        return recipe
    return recipe[key]


def _determine_device_type(instance_type: str) -> str:
    """Determine device type (gpu, cpu, trainium) based on instance type."""
    instance_family = instance_type.split(".")[1]
    if instance_family.startswith(("p", "g")):
        return "gpu"
    if instance_family.startswith("trn"):
        return "trainium"
    return "cpu"


def _load_recipes_cfg() -> str:
    """Load training recipes configuration json."""
    training_recipes_cfg_filename = os.path.join(os.path.dirname(__file__), "training_recipes.json")
    with open(training_recipes_cfg_filename) as training_recipes_cfg_file:
        training_recipes_cfg = json.load(training_recipes_cfg_file)
    return training_recipes_cfg


def _load_base_recipe(
    training_recipe: str,
    recipe_overrides: Optional[Dict[str, Any]] = None,
    training_recipes_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load recipe and apply overrides."""
    if recipe_overrides is None:
        recipe_overrides = dict()

    temp_local_recipe = tempfile.NamedTemporaryFile(prefix="recipe_original", suffix=".yaml").name

    if training_recipe.endswith(".yaml"):
        if os.path.isfile(training_recipe):
            shutil.copy(training_recipe, temp_local_recipe)
        else:
            try:
                urlretrieve(training_recipe, temp_local_recipe)
            except Exception as e:
                raise ValueError(
                    f"Could not fetch the provided recipe {training_recipe}: exception {str(e)}"
                )
    else:
        recipe_launcher_dir = tempfile.TemporaryDirectory(prefix="launcher_")

        launcher_repo = os.environ.get("TRAINING_LAUNCHER_GIT", None) or training_recipes_cfg.get(
            "launcher_repo"
        )
        _run_clone_command_silent(launcher_repo, recipe_launcher_dir.name)

        recipe = os.path.join(
            recipe_launcher_dir.name,
            "recipes_collection",
            "recipes",
            training_recipe + ".yaml",
        )
        if os.path.isfile(recipe):
            shutil.copy(recipe, temp_local_recipe)
        else:
            raise ValueError(f"Recipe {training_recipe} not found.")

    recipe = OmegaConf.load(temp_local_recipe)
    os.unlink(temp_local_recipe)
    recipe = OmegaConf.merge(recipe, recipe_overrides)
    return recipe


def _register_custom_resolvers():
    """Register custom resolvers for OmegaConf."""
    if not OmegaConf.has_resolver("multiply"):
        OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)
    if not OmegaConf.has_resolver("divide_ceil"):
        OmegaConf.register_new_resolver(
            "divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True
        )
    if not OmegaConf.has_resolver("divide_floor"):
        OmegaConf.register_new_resolver(
            "divide_floor", lambda x, y: int(math.floor(x / y)), replace=True
        )
    if not OmegaConf.has_resolver("add"):
        OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))


def _get_trainining_recipe_gpu_model_name_and_script(model_type: str):
    """Get the model base name and script for the training recipe."""

    model_type_to_script = {
        "llama": ("llama", "llama_pretrain.py"),
        "mistral": ("mistral", "mistral_pretrain.py"),
        "mixtral": ("mixtral", "mixtral_pretrain.py"),
        "deepseek": ("deepseek", "deepseek_pretrain.py"),
    }

    for key in model_type_to_script:
        if model_type.startswith(key):
            model_type = key
            break

    if model_type not in model_type_to_script:
        raise ValueError(f"Model type {model_type} not supported")

    return model_type_to_script[model_type][0], model_type_to_script[model_type][1]


def _configure_gpu_args(
    training_recipes_cfg: Dict[str, Any],
    region_name: str,
    recipe: OmegaConf,
    recipe_train_dir: tempfile.TemporaryDirectory,
) -> Dict[str, Any]:
    """Configure arguments specific to GPU."""
    source_code = SourceCode()
    args = dict()

    adapter_repo = os.environ.get("TRAINING_ADAPTER_GIT", None) or training_recipes_cfg.get(
        "adapter_repo"
    )
    _run_clone_command_silent(adapter_repo, recipe_train_dir.name)

    if "model" not in recipe:
        raise ValueError("Supplied recipe does not contain required field model.")
    if "model_type" not in recipe["model"]:
        raise ValueError("Supplied recipe does not contain required field model_type.")
    model_type = recipe["model"]["model_type"]

    model_base_name, script = _get_trainining_recipe_gpu_model_name_and_script(model_type)

    source_code.source_dir = os.path.join(recipe_train_dir.name, "examples", model_base_name)
    source_code.entry_script = script

    gpu_image_cfg = training_recipes_cfg.get("gpu_image")
    if isinstance(gpu_image_cfg, str):
        training_image = gpu_image_cfg
    else:
        training_image = retrieve(
            gpu_image_cfg.get("framework"),
            region=region_name,
            version=gpu_image_cfg.get("version"),
            image_scope="training",
            **gpu_image_cfg.get("additional_args"),
        )

    # Setting dummy parameters for now
    torch_distributed = Torchrun(smp=SMP(random_seed="123456"))
    args.update(
        {
            "source_code": source_code,
            "training_image": training_image,
            "distributed": torch_distributed,
        }
    )
    return args


def _configure_trainium_args(
    training_recipes_cfg: Dict[str, Any],
    region_name: str,
    recipe_train_dir: tempfile.TemporaryDirectory,
) -> Dict[str, Any]:
    """Configure arguments specific to Trainium."""
    source_code = SourceCode()
    args = dict()

    _run_clone_command_silent(training_recipes_cfg.get("neuron_dist_repo"), recipe_train_dir.name)

    source_code.source_dir = os.path.join(recipe_train_dir.name, "examples")
    source_code.entry_script = "training_orchestrator.py"
    neuron_image_cfg = training_recipes_cfg.get("neuron_image")
    if isinstance(neuron_image_cfg, str):
        training_image = neuron_image_cfg
    else:
        training_image = retrieve(
            neuron_image_cfg.get("framework"),
            region=region_name,
            version=neuron_image_cfg.get("version"),
            image_scope="training",
            **neuron_image_cfg.get("additional_args"),
        )

    args.update(
        {
            "source_code": source_code,
            "training_image": training_image,
            "distributed": Torchrun(),
        }
    )
    return args


def _get_args_from_recipe(
    training_recipe: str,
    compute: Compute,
    region_name: str,
    recipe_overrides: Optional[Dict[str, Any]],
    requirements: Optional[str],
) -> Tuple[Dict[str, Any], tempfile.TemporaryDirectory]:
    """Get arguments for ModelTrainer from a training recipe.

    Returns a dictionary of arguments to be used with ModelTrainer like:
    ```python
    {
        "source_code": SourceCode,
        "training_image": str,
        "distributed": DistributedConfig,
        "compute": Compute,
        "hyperparameters": Dict[str, Any],
    }
    ```

    Args:
        training_recipe (str):
            Name of the training recipe or path to the recipe file.
        compute (Compute):
            Compute configuration for training.
        region_name (str):
            Name of the AWS region.
        recipe_overrides (Optional[Dict[str, Any]]):
            Overrides for the training recipe.
        requirements (Optional[str]):
            Path to the requirements file.
    """
    if compute.instance_type is None:
        raise ValueError("Must set `instance_type` in compute when using training recipes.")

    training_recipes_cfg = _load_recipes_cfg()
    recipe = _load_base_recipe(training_recipe, recipe_overrides, training_recipes_cfg)

    if "trainer" not in recipe:
        raise ValueError("Supplied recipe does not contain required field trainer.")

    # Set instance_count
    if compute.instance_count and "num_nodes" in recipe["trainer"]:
        logger.warning(
            f"Using Compute to set instance_count:\n{compute}."
            "\nIgnoring trainer -> num_nodes in recipe."
        )
    if compute.instance_count is None:
        if "num_nodes" not in recipe["trainer"]:
            raise ValueError(
                "Must provide Compute with instance_count or" " set trainer -> num_nodes in recipe."
            )
        compute.instance_count = recipe["trainer"]["num_nodes"]

    if requirements and not os.path.isfile(requirements):
        raise ValueError(f"Recipe requirements file {requirements} not found.")

    # Get Training Image, SourceCode, and distributed args
    device_type = _determine_device_type(compute.instance_type)
    recipe_train_dir = tempfile.TemporaryDirectory(prefix="training_")
    if device_type == "gpu":
        args = _configure_gpu_args(training_recipes_cfg, region_name, recipe, recipe_train_dir)
    elif device_type == "trainium":
        args = _configure_trainium_args(training_recipes_cfg, region_name, recipe_train_dir)
    else:
        raise ValueError(f"Devices of type {device_type} are not supported with training recipes.")

    _register_custom_resolvers()

    # Resolve Final Recipe
    final_recipe = _try_resolve_recipe(recipe)
    if final_recipe is None:
        final_recipe = _try_resolve_recipe(recipe, "recipes")
    if final_recipe is None:
        final_recipe = _try_resolve_recipe(recipe, "training")
    if final_recipe is None:
        raise RuntimeError("Could not resolve provided recipe.")

    # Save Final Recipe to source_dir
    OmegaConf.save(
        config=final_recipe, f=os.path.join(args["source_code"].source_dir, "recipe.yaml")
    )

    # If recipe_requirements is provided, copy it to source_dir
    if requirements:
        shutil.copy(requirements, args["source_code"].source_dir)
        args["source_code"].requirements = os.path.basename(requirements)

    # Update args with compute and hyperparameters
    args.update(
        {
            "compute": compute,
            "hyperparameters": {"config-path": ".", "config-name": "recipe.yaml"},
        }
    )

    return args, recipe_train_dir
