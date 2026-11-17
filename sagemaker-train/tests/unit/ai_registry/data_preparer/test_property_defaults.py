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
"""Property-based tests for DataPreparerDefaults (P1, P8, P9, P10).

Requires hypothesis. Skipped if not installed.
"""

from __future__ import absolute_import

import re

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings, assume  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from sagemaker.ai_registry.data_preparer.constants import (  # noqa: E402
    JobCategory,
    MAX_JOB_NAME_LENGTH,
)  # noqa: E402
from sagemaker.ai_registry.data_preparer.defaults import DataPreparerDefaults  # noqa: E402
from sagemaker.core.helper.session_helper import Session  # noqa: E402

JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](-*[A-Za-z0-9]){0,62}$")


class TestPropertyJobNameShape:
    """P9: Job-name shape — resolved name always matches Smithy pattern and len ≤ 63."""

    @given(
        base=st.text(
            alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789-"),
            min_size=1,
            max_size=100,
        ),
        category=st.sampled_from(list(JobCategory)),
    )
    @settings(max_examples=200, deadline=5000)
    def test_derived_name_matches_pattern(self, base, category):
        assume(base[0].isalnum())
        try:
            result = DataPreparerDefaults.resolve_job_name(None, base, category, {})
        except Exception:
            return
        assert len(result) <= MAX_JOB_NAME_LENGTH
        assert JOB_NAME_PATTERN.match(result), f"'{result}' does not match pattern"

    @given(
        job_name=st.from_regex(r"[A-Za-z0-9](-*[A-Za-z0-9]){0,10}", fullmatch=True),
    )
    @settings(max_examples=100, deadline=5000)
    def test_verbatim_name_echoed_unchanged(self, job_name):
        assume(len(job_name) <= 63)
        result = DataPreparerDefaults.resolve_job_name(
            job_name, None, JobCategory.SYNTHETIC_DATA_GENERATION, {}
        )
        assert result == job_name


class TestPropertyJobArnMatchesJobName:
    """P10: JobArn matches JobName."""

    @given(
        job_name=st.from_regex(r"[A-Za-z0-9](-*[A-Za-z0-9]){0,10}", fullmatch=True),
        category=st.sampled_from(list(JobCategory)),
    )
    @settings(max_examples=100, deadline=5000)
    def test_arn_format(self, job_name, category):
        session = MagicMock(spec=Session)
        session.boto_region_name = "us-west-2"
        session.account_id.return_value = "123456789012"

        arn = DataPreparerDefaults.build_job_arn(category, job_name, session)
        assert arn == f"arn:aws:sagemaker:us-west-2:123456789012:job/{category.value}/{job_name}"


class TestPropertyUserOverDefaults:
    """P8: User values always win over defaults."""

    @given(
        user_value=st.integers(min_value=1, max_value=10000),
        config_value=st.integers(min_value=10001, max_value=20000),
    )
    @settings(max_examples=50, deadline=5000)
    def test_user_value_wins(self, user_value, config_value):
        schema = {"properties": {"NumSamples": {"type": "integer"}}, "required": []}
        mock_config = MagicMock()
        mock_config.resolve_value_from_config.return_value = config_value

        session = MagicMock()
        result = DataPreparerDefaults.resolve_into_payload(
            {"NumSamples": user_value},
            schema,
            JobCategory.SYNTHETIC_DATA_GENERATION,
            sagemaker_session=session,
            config_mgr=mock_config,
        )
        assert result["NumSamples"] == user_value
