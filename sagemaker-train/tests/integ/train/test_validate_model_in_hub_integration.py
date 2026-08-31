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
"""Integration tests for the base-model-name Hub availability check.

Covers what the unit tests (which mock the hub) structurally cannot: the
behavior of ``_validate_model_in_hub`` / ``_resolve_model_and_name`` against a
real ``DescribeHubContent`` call in prod us-west-2, querying the active
SageMaker hub (the integ harness may pin a private ``SAGEMAKER_HUB_NAME`` in
``conftest.py``; falls back to ``SageMakerPublicHub``).

The critical case is the negative one: a genuine miss must surface as an error
that ``_is_hub_content_not_found`` classifies as not-found, so the check
fail-closes with a clear ``ValueError`` rather than fail-opening (silently
skipping) on an unexpected error code. A mocked unit test cannot confirm the
real service's error shape; this test can.
"""
from __future__ import annotations

import os

import boto3
import pytest
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.common_utils.finetune_utils import (
    _resolve_model_and_name,
    _validate_model_in_hub,
)

_REGION = "us-west-2"
_FINETUNING_PREFIX = "@recipe:finetuning_"
# A name that cannot exist as hub content, to exercise the not-found path.
_BOGUS_MODEL_NAME = "this-model-does-not-exist-in-hub-prepare-pr-integ-000"


@pytest.fixture(scope="module")
def sagemaker_session():
    boto_session = boto3.Session(region_name=_REGION)
    yield Session(boto_session=boto_session)


@pytest.fixture(scope="module")
def a_real_hub_model(sagemaker_session):
    """Pick one real FineTuning-tagged model name from the active hub.

    Scans the hub directly (independent of the SDK) so the positive case uses a
    name the service actually resolves. Skips when the hub carries none.
    """
    client = sagemaker_session.boto_session.client("sagemaker", region_name=_REGION)
    hub_name = os.environ.get("SAGEMAKER_HUB_NAME", "SageMakerPublicHub")
    names: set = set()
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
            if any(
                kw.lower().startswith(_FINETUNING_PREFIX)
                for kw in summary.get("HubContentSearchKeywords", [])
            ):
                names.add(name)
        next_token = response.get("NextToken")
        if not next_token:
            break
    if not names:
        pytest.skip("active hub carries no FineTuning-tagged models to validate against")
    # Deterministic pick.
    return sorted(names)[0]


class TestValidateModelInHubIntegration:
    """Requires real SageMaker API access (prod us-west-2)."""

    def test_real_model_passes_validation(self, a_real_hub_model, sagemaker_session):
        """A model present in the hub validates without error and resolves to
        its own name."""
        # Direct check: does not raise for a present model.
        _validate_model_in_hub(a_real_hub_model, sagemaker_session)

        # Through the resolve path used by the trainers.
        resolved, name = _resolve_model_and_name(a_real_hub_model, sagemaker_session)
        assert resolved == a_real_hub_model
        assert name == a_real_hub_model

    def test_bogus_model_name_raises(self, sagemaker_session):
        """A name that does not exist in the hub fails closed with a clear error.

        This confirms the live not-found error is classified as not-found (rather
        than mistaken for a transient error and skipped)."""
        with pytest.raises(ValueError, match="is not available in SageMaker Hub"):
            _validate_model_in_hub(_BOGUS_MODEL_NAME, sagemaker_session)

    def test_bogus_model_name_raises_through_resolve(self, sagemaker_session):
        """The same not-found guard fires through the shared resolve path."""
        with pytest.raises(ValueError, match="is not available in SageMaker Hub"):
            _resolve_model_and_name(_BOGUS_MODEL_NAME, sagemaker_session)
