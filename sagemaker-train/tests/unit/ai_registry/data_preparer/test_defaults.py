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
"""Unit tests for sagemaker.ai_registry.data_preparer.defaults."""

from __future__ import absolute_import

from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.defaults import (
    DataPreparerDefaults,
    _fallback_role_arn,
)
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerConfigurationError,
)


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.boto_region_name = "us-west-2"
    session.account_id.return_value = "123456789012"
    return session


@pytest.fixture
def mock_config_mgr():
    mgr = MagicMock()
    mgr.resolve_value_from_config.return_value = None
    return mgr


@pytest.fixture(autouse=True)
def _mock_execution_role():
    """Prevent the RoleArn fallback from calling real IAM (per Review 7 B2 fix,
    the fallback now raises DataPreparerConfigurationError on failure instead of
    returning None, so tests must provide a mock here or avoid triggering it).
    """
    with patch(
        "sagemaker.core.helper.session_helper.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/TestRole",
    ):
        yield


SIMPLE_SCHEMA = {
    "properties": {
        "NumSamples": {"type": "integer"},
        "SpecS3Uri": {"type": "string"},
        "RoleArn": {"type": "string"},
        "OutputDataConfig": {
            "type": "object",
            "properties": {
                "OutputS3Uri": {"type": "string"},
                "HubName": {"type": "string"},
            },
        },
    },
    "required": ["NumSamples"],
}

# Nested all-optional objects, shaped like DataQualityEvaluation's EvaluationConfig.
# Deliberately NOT reusing SIMPLE_SCHEMA for the empty-object tests below:
# `OutputDataConfig.OutputS3Uri` has a live well-known fallback, so that subtree is
# always filled and can never exercise the empty-object path. No property here is
# registered in _WELL_KNOWN_FALLBACKS (lookup is by exact path), so these objects stay
# empty unless the config mock fills them — which is what makes the path reachable.
EVAL_CONFIG_SCHEMA = {
    "properties": {
        "EvaluationName": {"type": "string"},
        "EvaluationConfig": {
            "type": "object",
            "properties": {
                "Deterministic": {
                    "type": "object",
                    "properties": {"Metrics": {"type": "array"}},
                },
                "Specification": {
                    "type": "object",
                    "properties": {"SpecS3Uri": {"type": "string"}},
                },
            },
        },
    },
}


class TestGetSagemakerSession:
    def test_returns_provided_session(self, mock_session):
        result = DataPreparerDefaults.get_sagemaker_session(mock_session)
        assert result is mock_session

    @patch("sagemaker.ai_registry.data_preparer.defaults.Session")
    def test_creates_default_when_none(self, mock_session_cls):
        mock_session_cls.return_value = MagicMock()
        result = DataPreparerDefaults.get_sagemaker_session(None)
        mock_session_cls.assert_called_once()
        assert result is mock_session_cls.return_value


class TestGetBaseJobName:
    def test_from_dataset_name(self):
        payload = {"DatasetName": "my-dataset"}
        result = DataPreparerDefaults.get_base_job_name(
            payload, JobCategory.SYNTHETIC_DATA_GENERATION
        )
        assert result == "my-dataset"

    def test_from_evaluation_name(self):
        payload = {"EvaluationName": "my-eval"}
        result = DataPreparerDefaults.get_base_job_name(
            payload, JobCategory.DATA_QUALITY_EVALUATION
        )
        assert result == "my-eval"

    def test_fallback_to_category(self):
        payload = {}
        result = DataPreparerDefaults.get_base_job_name(
            payload, JobCategory.SYNTHETIC_DATA_GENERATION
        )
        assert result == "SyntheticDataGeneration-job"


class TestResolveJobName:
    def test_verbatim_job_name(self):
        result = DataPreparerDefaults.resolve_job_name(
            "my-exact-name", None, JobCategory.SYNTHETIC_DATA_GENERATION, {}
        )
        assert result == "my-exact-name"

    def test_invalid_job_name_raises(self):
        with pytest.raises(DataPreparerConfigurationError):
            DataPreparerDefaults.resolve_job_name(
                "invalid name with spaces", None, JobCategory.SYNTHETIC_DATA_GENERATION, {}
            )

    def test_base_job_name_with_timestamp(self):
        result = DataPreparerDefaults.resolve_job_name(
            None, "my-base", JobCategory.SYNTHETIC_DATA_GENERATION, {}
        )
        assert result.startswith("my-base-")
        assert len(result) <= 63

    def test_truncation_to_63_chars(self):
        long_base = "a" * 100
        result = DataPreparerDefaults.resolve_job_name(
            None, long_base, JobCategory.SYNTHETIC_DATA_GENERATION, {}
        )
        assert len(result) <= 63

    def test_no_trailing_hyphen_after_truncation(self):
        # Base that ends with hyphen after truncation
        base = "a" * 38 + "-"
        result = DataPreparerDefaults.resolve_job_name(
            None, base, JobCategory.SYNTHETIC_DATA_GENERATION, {}
        )
        # Should not have double hyphens from truncation
        assert "--" not in result or result.count("--") == 0

    def test_category_fallback(self):
        result = DataPreparerDefaults.resolve_job_name(
            None, None, JobCategory.SYNTHETIC_DATA_GENERATION, {}
        )
        assert "SyntheticDataGeneration" in result


class TestBuildJobArn:
    def test_format(self, mock_session):
        result = DataPreparerDefaults.build_job_arn(
            JobCategory.SYNTHETIC_DATA_GENERATION, "my-job-2026", mock_session
        )
        assert result == (
            "arn:aws:sagemaker:us-west-2:123456789012:job/" "SyntheticDataGeneration/my-job-2026"
        )


class TestGetCustomerDetails:
    def test_only_account_id(self, mock_session):
        result = DataPreparerDefaults.get_customer_details(mock_session)
        assert result == {"AccountId": "123456789012"}
        assert "OrganizationId" not in result
        assert "UserContext" not in result


class TestResolveIntoPayload:
    def test_user_values_preserved(self, mock_session, mock_config_mgr):
        user_payload = {"NumSamples": 100, "SpecS3Uri": "s3://my-bucket/spec.txt"}
        result = DataPreparerDefaults.resolve_into_payload(
            user_payload,
            SIMPLE_SCHEMA,
            JobCategory.SYNTHETIC_DATA_GENERATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )
        assert result["NumSamples"] == 100
        assert result["SpecS3Uri"] == "s3://my-bucket/spec.txt"

    def test_does_not_mutate_user_payload(self, mock_session, mock_config_mgr):
        user_payload = {"NumSamples": 100}
        original = user_payload.copy()
        DataPreparerDefaults.resolve_into_payload(
            user_payload,
            SIMPLE_SCHEMA,
            JobCategory.SYNTHETIC_DATA_GENERATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )
        assert user_payload == original

    def test_config_value_fills_absent_field(self, mock_session, mock_config_mgr):
        def config_side_effect(config_path=None, sagemaker_session=None):
            if "SpecS3Uri" in config_path:
                return "s3://default/spec.txt"
            return None

        mock_config_mgr.resolve_value_from_config.side_effect = config_side_effect

        user_payload = {"NumSamples": 100}
        result = DataPreparerDefaults.resolve_into_payload(
            user_payload,
            SIMPLE_SCHEMA,
            JobCategory.SYNTHETIC_DATA_GENERATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )
        assert result["SpecS3Uri"] == "s3://default/spec.txt"

    def test_user_value_wins_over_config(self, mock_session, mock_config_mgr):
        def config_side_effect(config_path=None, sagemaker_session=None):
            if "SpecS3Uri" in config_path:
                return "s3://config/spec.txt"
            return None

        mock_config_mgr.resolve_value_from_config.side_effect = config_side_effect

        user_payload = {"NumSamples": 100, "SpecS3Uri": "s3://user/spec.txt"}
        result = DataPreparerDefaults.resolve_into_payload(
            user_payload,
            SIMPLE_SCHEMA,
            JobCategory.SYNTHETIC_DATA_GENERATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )
        assert result["SpecS3Uri"] == "s3://user/spec.txt"

    def test_nested_object_walk(self, mock_session, mock_config_mgr):
        def config_side_effect(config_path=None, sagemaker_session=None):
            if "OutputS3Uri" in config_path:
                return "s3://config/output/"
            return None

        mock_config_mgr.resolve_value_from_config.side_effect = config_side_effect

        user_payload = {"NumSamples": 100}
        result = DataPreparerDefaults.resolve_into_payload(
            user_payload,
            SIMPLE_SCHEMA,
            JobCategory.SYNTHETIC_DATA_GENERATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )
        assert result["OutputDataConfig"]["OutputS3Uri"] == "s3://config/output/"

    def test_unfilled_scaffolding_is_still_pruned(self, mock_session, mock_config_mgr):
        """The walk creates a `{}` for every schema object as it descends. Ones nothing
        defaults into are scaffolding, not requests, and must not reach the payload."""
        result = DataPreparerDefaults.resolve_into_payload(
            {"EvaluationName": "x"},
            EVAL_CONFIG_SCHEMA,
            JobCategory.DATA_QUALITY_EVALUATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )
        assert result == {"EvaluationName": "x"}

    def test_caller_supplied_empty_object_is_preserved(self, mock_session, mock_config_mgr):
        """An explicit `{}` is a request, not an absent value.

        `EvaluationConfig.Deterministic: {}` asks a DataQualityEvaluation job for the
        default metric set — the service selects defaults precisely when `Metrics` is
        omitted. Pruning it alongside our own unfilled scaffolding made that request a
        silent no-op: the job ran no deterministic metrics and reported no error.

        Note the cascade this also covers: deleting `Deterministic` left
        `EvaluationConfig` empty, so that was pruned too and the request vanished
        entirely rather than merely losing one group.
        """
        result = DataPreparerDefaults.resolve_into_payload(
            {"EvaluationName": "x", "EvaluationConfig": {"Deterministic": {}}},
            EVAL_CONFIG_SCHEMA,
            JobCategory.DATA_QUALITY_EVALUATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )
        assert result["EvaluationConfig"] == {"Deterministic": {}}

    def test_caller_supplied_empty_object_still_receives_defaults(
        self, mock_session, mock_config_mgr
    ):
        """Preserving `{}` must gate only the deletion, never the recursion — a
        caller's empty object still gets filled from config."""

        def config_side_effect(config_path=None, sagemaker_session=None):
            if "SpecS3Uri" in config_path:
                return "s3://config/spec.md"
            return None

        mock_config_mgr.resolve_value_from_config.side_effect = config_side_effect

        result = DataPreparerDefaults.resolve_into_payload(
            {"EvaluationName": "x", "EvaluationConfig": {"Specification": {}}},
            EVAL_CONFIG_SCHEMA,
            JobCategory.DATA_QUALITY_EVALUATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )
        assert result["EvaluationConfig"]["Specification"]["SpecS3Uri"] == "s3://config/spec.md"

    @patch(
        "sagemaker.ai_registry.data_preparer.defaults._WELL_KNOWN_FALLBACKS",
        {
            "RoleArn": MagicMock(return_value="arn:aws:iam::123:role/DefaultRole"),
            "OutputDataConfig.OutputS3Uri": MagicMock(return_value=None),
        },
    )
    def test_well_known_fallback_fires_for_role(self, mock_session, mock_config_mgr):
        user_payload = {"NumSamples": 100}
        result = DataPreparerDefaults.resolve_into_payload(
            user_payload,
            SIMPLE_SCHEMA,
            JobCategory.SYNTHETIC_DATA_GENERATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )
        assert result["RoleArn"] == "arn:aws:iam::123:role/DefaultRole"

    @patch(
        "sagemaker.ai_registry.data_preparer.defaults._WELL_KNOWN_FALLBACKS",
        {
            "RoleArn": MagicMock(return_value="should-not-appear"),
            "OutputDataConfig.OutputS3Uri": MagicMock(return_value=None),
        },
    )
    def test_well_known_fallback_does_not_fire_for_other_paths(self, mock_session, mock_config_mgr):
        user_payload = {"NumSamples": 100, "RoleArn": "user-role"}
        result = DataPreparerDefaults.resolve_into_payload(
            user_payload,
            SIMPLE_SCHEMA,
            JobCategory.SYNTHETIC_DATA_GENERATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )
        # User value wins
        assert result["RoleArn"] == "user-role"


# DataTransformation and DQ v2 spell the output location S3OutputPath, not
# OutputS3Uri. Tier-3 lookup is exact-path match, so the registry needs both
# spellings or those categories get no default at all.
DT_SCHEMA = {
    "properties": {
        "InputDataConfig": {"type": "array"},
        "OutputDataConfig": {
            "type": "object",
            "properties": {
                "S3OutputPath": {"type": "string"},
            },
        },
        "TransformationConfig": {"type": "object", "properties": {}},
    },
    "required": ["InputDataConfig", "OutputDataConfig", "TransformationConfig"],
}


class TestS3OutputPathFallback:
    """The output-location fallback must fire under both schema spellings.

    These deliberately exercise the real ``_WELL_KNOWN_FALLBACKS`` registry
    instead of patching it — patching is what let the missing S3OutputPath entry
    go unnoticed, since a stubbed registry proves nothing about the real one.
    """

    @patch("sagemaker.train.utils._default_s3_uri")
    def test_fallback_fills_s3_output_path_for_dt(
        self, mock_default_uri, mock_session, mock_config_mgr
    ):
        mock_default_uri.return_value = "s3://bucket/DataPreparer/DataTransformation/x"

        result = DataPreparerDefaults.resolve_into_payload(
            {
                "InputDataConfig": [
                    {"DatasetArn": "arn:aws:sagemaker:us-west-2:1:hub-content/H/DataSet/in/1.0.0"}
                ]
            },
            DT_SCHEMA,
            JobCategory.DATA_TRANSFORMATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )

        assert (
            result["OutputDataConfig"]["S3OutputPath"]
            == "s3://bucket/DataPreparer/DataTransformation/x"
        )

    @patch("sagemaker.train.utils._default_s3_uri")
    def test_user_supplied_s3_output_path_wins(
        self, mock_default_uri, mock_session, mock_config_mgr
    ):
        mock_default_uri.return_value = "s3://bucket/should-not-appear"

        result = DataPreparerDefaults.resolve_into_payload(
            {"OutputDataConfig": {"S3OutputPath": "s3://mine/out/"}},
            DT_SCHEMA,
            JobCategory.DATA_TRANSFORMATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )

        assert result["OutputDataConfig"]["S3OutputPath"] == "s3://mine/out/"

    @patch("sagemaker.train.utils._default_s3_uri")
    def test_config_value_outranks_fallback(self, mock_default_uri, mock_session, mock_config_mgr):
        """Tier 1/2 config must still take precedence over the new tier-3 entry."""
        mock_default_uri.return_value = "s3://bucket/fallback/"

        def config_side_effect(config_path=None, sagemaker_session=None):
            if "S3OutputPath" in config_path:
                return "s3://from-config/out/"
            return None

        mock_config_mgr.resolve_value_from_config.side_effect = config_side_effect

        result = DataPreparerDefaults.resolve_into_payload(
            {},
            DT_SCHEMA,
            JobCategory.DATA_TRANSFORMATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )

        assert result["OutputDataConfig"]["S3OutputPath"] == "s3://from-config/out/"

    @patch("sagemaker.train.utils._default_s3_uri")
    def test_output_s3_uri_spelling_still_works(
        self, mock_default_uri, mock_session, mock_config_mgr
    ):
        """Adding the DT spelling must not regress SDG/DQ v1."""
        mock_default_uri.return_value = "s3://bucket/sdg/"

        result = DataPreparerDefaults.resolve_into_payload(
            {"NumSamples": 100},
            SIMPLE_SCHEMA,
            JobCategory.SYNTHETIC_DATA_GENERATION,
            sagemaker_session=mock_session,
            config_mgr=mock_config_mgr,
        )

        assert result["OutputDataConfig"]["OutputS3Uri"] == "s3://bucket/sdg/"


class TestFallbackRoleArnA1:
    """A1 (P0): _fallback_role_arn must not leak internal exception text.

    The customer-facing message must contain only corrective guidance.
    The original exception must be preserved as __cause__ for tracebacks.
    """

    def test_internal_error_not_in_customer_message(self, mock_session, mock_config_mgr):
        """FAILS on original code (message contained '{e}'); passes after A1 fix."""
        sentinel = "INTERNAL_SECRET_XyzNotFoundException_DEADBEEF"
        with patch(
            "sagemaker.core.helper.session_helper.get_execution_role",
            side_effect=Exception(sentinel),
        ):
            with pytest.raises(DataPreparerConfigurationError) as exc_info:
                _fallback_role_arn(None, mock_session, {}, mock_config_mgr)

            msg = str(exc_info.value)
            assert sentinel not in msg, "Internal error must not leak into customer message"
            assert "role=" in msg, "Corrective guidance for role= kwarg must be present"
            assert (
                "SageMaker.TrainingJob.RoleArn" in msg
            ), "Corrective guidance for config path must be present"
            assert (
                "get_execution_role" in msg
            ), "Corrective guidance for get_execution_role must be present"
            assert (
                exc_info.value.__cause__ is not None
            ), "__cause__ must be set so tracebacks still show the root failure"

    def test_cause_is_original_exception(self, mock_session, mock_config_mgr):
        """__cause__ is the exact original exception instance."""
        original = Exception("original cause")
        with patch(
            "sagemaker.core.helper.session_helper.get_execution_role",
            side_effect=original,
        ):
            with pytest.raises(DataPreparerConfigurationError) as exc_info:
                _fallback_role_arn(None, mock_session, {}, mock_config_mgr)

            assert exc_info.value.__cause__ is original
