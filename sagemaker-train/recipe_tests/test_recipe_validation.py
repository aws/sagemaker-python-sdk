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
"""Recipe validation integ test for the HP-ModelCustomization-RecipeValidator pipeline.

Iterates through every model in the private hub referenced by the ``HYPERPOD_HUB_NAME``
env var and validates that each fine-tuning recipe can be used to instantiate the
appropriate ``sagemaker.train`` Trainer class (SFT/DPO/RLAIF/RLVR).
"""
from __future__ import absolute_import

import json
import os

import boto3

from sagemaker.train.common import TrainingType
from sagemaker.train.dpo_trainer import DPOTrainer
from sagemaker.train.rlaif_trainer import RLAIFTrainer
from sagemaker.train.rlvr_trainer import RLVRTrainer
from sagemaker.train.sft_trainer import SFTTrainer

TRAINER_MAPPING = {
    "sft": SFTTrainer,
    "dpo": DPOTrainer,
    "rlaif": RLAIFTrainer,
    "rlvr": RLVRTrainer,
}

DUMMY_DATASET = "s3://placeholder/validation-data"
DUMMY_MODEL_PACKAGE_GROUP = "recipe-validation-test"


def detect_training_type(recipe_path: str) -> str:
    """Detect SFT/DPO/RLAIF/RLVR from the recipe name; default to SFT."""
    if not recipe_path:
        return "sft"
    lower = recipe_path.lower()
    if "rlvr" in lower:
        return "rlvr"
    if "rlaif" in lower:
        return "rlaif"
    if "dpo" in lower:
        return "dpo"
    return "sft"


def detect_lora_or_full(recipe_path: str) -> TrainingType:
    """Detect LoRA vs full fine-tuning from the recipe name; default to LoRA."""
    if not recipe_path:
        return TrainingType.LORA
    lower = recipe_path.lower()
    if "_fft" in lower or "full_fine_tuning" in lower:
        return TrainingType.FULL
    return TrainingType.LORA


def test_new_recipes_create_valid_trainers():
    """Validate every new/modified recipe in the private hub yields a valid Trainer."""
    hub_name = os.environ.get("HYPERPOD_HUB_NAME")
    assert hub_name, "HYPERPOD_HUB_NAME environment variable must be set"

    sm = boto3.client("sagemaker", region_name="us-west-2")

    models = []
    kwargs = {"HubName": hub_name, "HubContentType": "Model"}
    while True:
        response = sm.list_hub_contents(**kwargs)
        models.extend([item["HubContentName"] for item in response["HubContentSummaries"]])
        next_token = response.get("NextToken")
        if not next_token:
            break
        kwargs["NextToken"] = next_token

    if not models:
        return

    errors = []
    for model_name in models:
        try:
            response = sm.describe_hub_content(
                HubName=hub_name,
                HubContentType="Model",
                HubContentName=model_name,
            )
            doc = json.loads(response.get("HubContentDocument", "{}"))
            recipes = doc.get("RecipeCollection", [])

            ft_recipes = [r for r in recipes if r.get("Type") == "FineTuning"]
            for i, recipe in enumerate(ft_recipes):
                recipe_name = recipe.get("Name", f"recipe-{i}")
                training_type = detect_training_type(recipe_name)
                training_type_enum = detect_lora_or_full(recipe_name)
                trainer_class = TRAINER_MAPPING[training_type]

                trainer = trainer_class(
                    model=model_name,
                    training_type=training_type_enum,
                    training_dataset=DUMMY_DATASET,
                    model_package_group=DUMMY_MODEL_PACKAGE_GROUP,
                    accept_eula=True,
                )
                assert trainer is not None, (
                    f"{model_name}: {trainer_class.__name__} returned None"
                )
        except Exception as e:  # noqa: BLE001 - collect all errors across all models
            errors.append(f"{model_name}: {e}")

    assert not errors, "Recipe validation failures:\n" + "\n".join(errors)
