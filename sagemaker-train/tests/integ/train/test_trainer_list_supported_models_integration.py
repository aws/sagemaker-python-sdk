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
"""Integration tests for ``BaseTrainer.list_supported_models`` across the
fine-tuning trainers (SFT / RLVR / RLAIF / DPO / CPT).

These tests run against real SageMaker services in prod us-west-2 and query
SageMakerPublicHub. Requires valid AWS credentials with appropriate permissions.

They verify the contract that unit tests (which mock the hub) cannot: that each
trainer's ``_customization_technique`` string matches how the live hub tags its
FineTuning recipes (``@recipe:finetuning_{technique}_...``). A drifted or
mistyped technique would still pass unit tests but return an empty list here.
"""
from __future__ import annotations

import boto3
import pytest
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.rlvr_trainer import RLVRTrainer
from sagemaker.train.rlaif_trainer import RLAIFTrainer
from sagemaker.train.dpo_trainer import DPOTrainer
from sagemaker.train.cpt_trainer import CPTTrainer

_REGION = "us-west-2"

# (trainer class, expected customization technique string)
_TRAINER_CASES = [
    pytest.param(SFTTrainer, "SFT", id="SFT"),
    pytest.param(RLVRTrainer, "RLVR", id="RLVR"),
    pytest.param(RLAIFTrainer, "RLAIF", id="RLAIF"),
    pytest.param(DPOTrainer, "DPO", id="DPO"),
    pytest.param(CPTTrainer, "CPT", id="CPT"),
]


@pytest.fixture(scope="module")
def sagemaker_session():
    boto_session = boto3.Session(region_name=_REGION)
    session = Session(boto_session=boto_session)
    yield session


class TestTrainerListSupportedModels:
    """List supported models per fine-tuning technique (requires API access)."""

    @pytest.mark.parametrize("trainer_cls,expected_technique", _TRAINER_CASES)
    def test_list_supported_models(self, trainer_cls, expected_technique, sagemaker_session):
        """Each trainer resolves its technique and returns hub models for it."""
        # Sanity: the class attribute the inherited method keys off is set.
        assert trainer_cls._customization_technique == expected_technique

        result = trainer_cls.list_supported_models(
            session=sagemaker_session.boto_session
        )

        assert isinstance(result, list)
        assert all(isinstance(name, str) and name for name in result)
        # The live public hub is expected to tag at least one model per technique.
        # An empty list here signals a technique-string / hub-keyword mismatch
        # (or that the technique has no publicly-tagged models yet).
        assert len(result) > 0, (
            f"No hub models returned for technique '{expected_technique}'; "
            "check the technique string matches the hub recipe keywords."
        )
        # Result is returned sorted by the underlying helper.
        assert result == sorted(result)
