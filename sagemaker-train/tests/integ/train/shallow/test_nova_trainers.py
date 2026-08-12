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
"""Shallow submission tests for Nova models (SFT and RLVR).

Shallow counterparts of ``test_sft_trainer_integration.py::test_sft_trainer_nova_workflow``
and ``test_rlvr_trainer_integration.py::test_rlvr_trainer_nova_workflow``.

Nova is a distinct path: a different recipe family, a different region
(us-east-1), and a different test account, so these cannot share
``RecipeTrainerCases`` -- its ``MODEL_ID``, dataset fixtures and default session
are all us-west-2. Marked ``us_east_1`` so they run in that region's integ job.

Datasets and reward functions are the same pre-provisioned ones the deep suite
uses, in account 784379639078. If those move, both suites break together, which
is preferable to this suite silently drifting onto its own copies.
"""

from __future__ import absolute_import

import pytest
from sagemaker.core import shapes
from sagemaker.core.training.configs import TrainingJobCompute
from sagemaker.train.common import TrainingType
from sagemaker.train.rlvr_trainer import RLVRTrainer
from sagemaker.train.sft_trainer import SFTTrainer

from .harness import MAX_RUNTIME_IN_SECONDS, assert_submitted, submitted, unique_name

NOVA_MODEL = "nova-textgeneration-lite-v2"
MODEL_PACKAGE_GROUP = "sdk-test-finetuned-models"

# Pre-provisioned in the us-east-1 test account, shared with the deep suite.
_NOVA_BUCKET = "s3://sagemaker-us-east-1-784379639078"
SFT_DATASET = f"{_NOVA_BUCKET}/input_data/sft-nova/sft_200_samples.jsonl"
RLVR_DATASET = f"{_NOVA_BUCKET}/input_data/rlvr-nova/grpo-64-sample.jsonl"
OUTPUT_PATH = f"{_NOVA_BUCKET}/output/"
RLVR_REWARD_FUNCTION = (
    "arn:aws:sagemaker:us-east-1:784379639078:hub-content/sdktest/JsonDoc/rlvr-nova-test-rf/0.0.1"
)


def _stopping_condition():
    return shapes.StoppingCondition(max_runtime_in_seconds=MAX_RUNTIME_IN_SECONDS)


@pytest.mark.us_east_1
class TestNovaSFTSubmission:
    """Nova SFT selects a Nova-specific recipe family."""

    def test_nova_sft_is_accepted(self, sagemaker_session_us_east_1):
        trainer = SFTTrainer(
            model=NOVA_MODEL,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=SFT_DATASET,
            s3_output_path=OUTPUT_PATH,
            accept_eula=True,
            sagemaker_session=sagemaker_session_us_east_1,
            base_job_name=unique_name("shallow-nova-sft"),
            stopping_condition=_stopping_condition(),
        )

        with submitted(trainer) as job:
            assert_submitted(job)


@pytest.mark.us_east_1
class TestNovaRLVRSubmission:
    """Nova RLVR additionally carries a Nova-specific reward function."""

    def test_nova_rlvr_is_accepted(self, sagemaker_session_us_east_1):
        trainer = RLVRTrainer(
            model=NOVA_MODEL,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=RLVR_DATASET,
            validation_dataset=RLVR_DATASET,
            s3_output_path=OUTPUT_PATH,
            custom_reward_function=RLVR_REWARD_FUNCTION,
            accept_eula=True,
            sagemaker_session=sagemaker_session_us_east_1,
            base_job_name=unique_name("shallow-nova-rlvr"),
            stopping_condition=_stopping_condition(),
        )

        with submitted(trainer) as job:
            assert_submitted(job)


@pytest.mark.us_east_1
class TestNovaServerfulSubmission:
    """Nova on explicit TrainingJobCompute (serverful SMTJ).

    Shallow counterpart of ``test_sft_trainer_serverful_smtj.py``. Distinct from
    ``RecipeTrainerCases::test_explicit_compute_is_accepted``, which covers the
    serverful path for an OSS model in us-west-2: this is a Nova model, a Nova
    recipe family, a Nova-only instance type, and a different region, so the
    payload differs throughout.

    Also carries recipe overrides, as the deep test does, since Nova recipes nest
    epoch control differently from OSS ones.
    """

    SERVERFUL_INSTANCE_TYPE = "ml.g6.12xlarge"
    NOVA_MICRO = "amazon.nova-micro-v1"

    def test_nova_serverful_with_overrides_is_accepted(self, sagemaker_session_us_east_1):
        trainer = SFTTrainer(
            model=self.NOVA_MICRO,
            training_type=TrainingType.LORA,
            training_dataset=SFT_DATASET,
            s3_output_path=OUTPUT_PATH,
            compute=TrainingJobCompute(
                instance_type=self.SERVERFUL_INSTANCE_TYPE, instance_count=1
            ),
            sagemaker_session=sagemaker_session_us_east_1,
            overrides={"training_config": {"max_epochs": 1}},
            base_job_name=unique_name("shallow-nova-smtj"),
            stopping_condition=_stopping_condition(),
        )

        # The deep test asserts the override reached the resolved recipe; keep that,
        # since it is client-side and exact.
        #
        # Recipe families nest epoch control differently: Nova puts it under
        # ``trainer`` (which is what test_sft_trainer_serverful_smtj.py asserts),
        # while the OSS Llama recipes use ``training_args`` -- verified against AWS
        # by probing the resolver. Accept whichever this family uses rather than
        # hard-coding one shape, so the test fails on a lost override rather than
        # on a recipe-layout difference.
        training_config = trainer.get_resolved_recipe()["training_config"]
        epochs = training_config.get("trainer", {}).get(
            "max_epochs", training_config.get("training_args", {}).get("max_epochs")
        )
        assert epochs == 1, f"override did not reach the resolved recipe: {training_config}"

        with submitted(trainer) as job:
            assert_submitted(job)
