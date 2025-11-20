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
"""This module contains the Integ Tests for JumpStart Training."""
from __future__ import absolute_import

import pytest

from sagemaker.core.jumpstart import JumpStartConfig
from sagemaker.train import ModelTrainer


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "model_id": "huggingface-spc-bert-base-cased",
            "hyperparameters": {
                "epochs": 1,  # Set to 1 for testing purposes
            },
        },
        {"model_id": "xgboost-classification-model"},
        {"model_id": "catboost-regression-model"},
    ],
    ids=[
        "huggingface-spc-bert-base-cased",
        "xgboost-classification-model",
        "catboost-regression-model",
    ],
)
def test_jumpstart_train(test_case):
    """Test JumpStart training."""
    jumpstart = JumpStartConfig(
        model_id=test_case["model_id"],
        accept_eula=test_case.get("accept_eula", False),
    )
    model_trainer = ModelTrainer.from_jumpstart_config(
        jumpstart,
        base_job_name=test_case["model_id"],
        hyperparameters=test_case.get("hyperparameters", {}),
    )
    model_trainer.train()
