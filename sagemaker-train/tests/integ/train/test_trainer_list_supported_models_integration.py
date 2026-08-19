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

These tests run against real SageMaker services in prod us-west-2 and query the
active SageMaker hub (the integ harness pins a private ``SAGEMAKER_HUB_NAME`` in
``conftest.py``; falls back to ``SageMakerPublicHub``).

They verify the contract that unit tests (which mock the hub) cannot: that each
trainer's ``_customization_technique`` string matches how the live hub tags its
FineTuning recipes. Rather than assert a hard non-empty count -- which is brittle
because the pinned hub may not carry every technique -- the test independently
scans the hub once and asserts ``list_supported_models`` returns exactly the set
of models tagged for that technique (whether that is zero or many). A helper
regression that (for example) required a ``_{strategy}`` suffix -- and so dropped
suffix-less techniques like CPT (``@recipe:finetuning_cpt``) -- would surface
here as a mismatch against the oracle.
"""
from __future__ import annotations

import collections
import os

import boto3
import pytest
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.rlvr_trainer import RLVRTrainer
from sagemaker.train.rlaif_trainer import RLAIFTrainer
from sagemaker.train.dpo_trainer import DPOTrainer
from sagemaker.train.cpt_trainer import CPTTrainer

_REGION = "us-west-2"
_FINETUNING_PREFIX = "@recipe:finetuning_"

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
    yield Session(boto_session=boto_session)


@pytest.fixture(scope="module")
def hub_finetuning_models(sagemaker_session):
    """Independent oracle: scan the active hub once and map technique token ->
    sorted list of model names tagged with a matching FineTuning recipe.

    Built independently of the SDK helper (groups by the token immediately after
    ``@recipe:finetuning_``), so it can catch a regression in that helper rather
    than merely re-deriving it.
    """
    client = sagemaker_session.boto_session.client("sagemaker", region_name=_REGION)
    hub_name = os.environ.get("SAGEMAKER_HUB_NAME", "SageMakerPublicHub")
    mapping: dict[str, set] = collections.defaultdict(set)
    next_token = None
    while True:
        kwargs = {"HubName": hub_name, "HubContentType": "Model"}
        if next_token:
            kwargs["NextToken"] = next_token
        response = client.list_hub_contents(**kwargs)
        for summary in response.get("HubContentSummaries", []):
            name = summary.get("HubContentName")
            if not name:
                continue
            for keyword in summary.get("HubContentSearchKeywords", []):
                kwl = keyword.lower()
                if kwl.startswith(_FINETUNING_PREFIX):
                    token = kwl[len(_FINETUNING_PREFIX):].split("_")[0]
                    mapping[token].add(name)
        next_token = response.get("NextToken")
        if not next_token:
            break
    return {tech: sorted(names) for tech, names in mapping.items()}


class TestTrainerListSupportedModels:
    """List supported models per fine-tuning technique (requires API access)."""

    @pytest.mark.parametrize("trainer_cls,expected_technique", _TRAINER_CASES)
    def test_list_supported_models(
        self, trainer_cls, expected_technique, sagemaker_session, hub_finetuning_models
    ):
        """Each trainer resolves its technique and returns exactly the hub models
        tagged for it."""
        # Sanity: the class attribute the inherited method keys off is set.
        assert trainer_cls._customization_technique == expected_technique

        result = trainer_cls.list_supported_models(
            session=sagemaker_session.boto_session
        )

        # Structural contract: a sorted, de-duplicated list of non-empty strings.
        assert isinstance(result, list)
        assert all(isinstance(name, str) and name for name in result)
        assert result == sorted(result)
        assert len(set(result)) == len(result)

        # Correctness contract: exactly the models the active hub tags for this
        # technique (may legitimately be empty if the pinned hub carries none).
        expected = hub_finetuning_models.get(expected_technique.lower(), [])
        assert result == expected

    def test_public_hub_has_models_for_core_techniques(self, hub_finetuning_models):
        """Guard against a silent all-empty hub / broken scan: only meaningful
        against the public hub, where these techniques are known to be tagged.
        Skipped when a private test hub is pinned."""
        if os.environ.get("SAGEMAKER_HUB_NAME", "SageMakerPublicHub") != "SageMakerPublicHub":
            pytest.skip("private hub pinned; model population is environment-specific")
        for technique in ("sft", "dpo", "rlvr", "rlaif", "cpt"):
            assert hub_finetuning_models.get(technique), (
                f"public hub returned no models for technique '{technique}'"
            )
