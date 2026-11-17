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
"""Shallow submission tests for DataMixingConfig (Nova only).

Shallow counterpart of test_sft_trainer_data_mixing_integration.py and
test_sft_data_mixing_hyperpod.py.

DataMixingConfig is serialized into flat per-category hyperparameters. It is
Nova-only, and Nova is exercised in us-east-1 in this repo, so these use
sagemaker_session_us_east_1 and carry the us_east_1 marker -- the PR-gate
job holds us-west-2 credentials only, so they run in the us-east-1 integ job.

Kept in its own file rather than folded into test_sft_trainer.py because the region
and model differ from every other case there.
"""

from __future__ import absolute_import

import pytest
from sagemaker.train.data_mixing_config import DataMixingConfig
from sagemaker.train.sft_trainer import SFTTrainer

from .harness import assert_submitted, submitted, unique_name
from .recipe_cases import NOVA_MODEL_PACKAGE_GROUP, stopping_condition

NOVA_MODEL = "nova-textgeneration-lite-v2"


def _nova_sft(session, dataset, name, config):
    return SFTTrainer(
        model=NOVA_MODEL,
        model_package_group=NOVA_MODEL_PACKAGE_GROUP,
        training_dataset=dataset,
        accept_eula=True,
        sagemaker_session=session,
        data_mixing_config=config,
        base_job_name=name,
        stopping_condition=stopping_condition(),
        # The existing data-mixing test sets the recipe name explicitly; keep that
        # so the rendered recipe matches what the service expects.
        overrides={"name": name},
    )


@pytest.mark.us_east_1
class TestNovaDataMixingSubmission:
    """DataMixingConfig serialization must be accepted by the service."""

    def test_explicit_percentages(self, sagemaker_session_us_east_1, nova_train_data_uri):
        """Per-category percentages must sum to 100 client-side and serialize into
        hyperparameters the service accepts."""
        config = DataMixingConfig(
            customer_data_percent=70.0,
            nova_data_percentages={
                "code": 30.0,
                "math": 20.0,
                "planning": 10.0,
                "instruction-following": 10.0,
                "reasoning-instruction-following": 20.0,
                "reasoning-math": 10.0,
            },
        )
        name = unique_name("shallow-nova-datamix")
        trainer = _nova_sft(sagemaker_session_us_east_1, nova_train_data_uri, name, config)

        with submitted(trainer) as job:
            assert_submitted(job)

    def test_recipe_defaults(self, sagemaker_session_us_east_1, nova_train_data_uri):
        """With nova_data_percentages=None the recipe template's defaults are
        used at submission time -- a different serialization path."""
        config = DataMixingConfig(customer_data_percent=80.0)
        name = unique_name("shallow-nova-datamix-default")
        trainer = _nova_sft(sagemaker_session_us_east_1, nova_train_data_uri, name, config)

        with submitted(trainer) as job:
            assert_submitted(job)
