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
"""Unit tests for sagemaker.ai_registry.data_preparer.preparer."""

from __future__ import absolute_import

import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer.preparer import (
    DataPreparer,
    _parse_dataset_arn,
)
from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig
from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerConfigurationError,
    DataPreparerJobError,
    DataPreparerJobFailedError,
    DataPreparerJobStoppedError,
    DataPreparerJobTimeoutError,
)
from sagemaker.core.helper.session_helper import Session


@pytest.fixture
def mock_session():
    session = MagicMock(spec=Session)
    session.boto_region_name = "us-west-2"
    session.account_id.return_value = "123456789012"
    return session


@pytest.fixture
def resolved_dataset(mock_session):
    """A DatasetJobConfig that has been pre-resolved."""
    cfg = DatasetJobConfig(
        job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
        num_samples=500,
    )
    # Manually set resolved state
    cfg._resolved_schema_version = "1.0.0"
    cfg._resolved_schema = {"properties": {"NumSamples": {"type": "integer"}}}
    cfg._dynamic_model = MagicMock()
    cfg._validated_payload = {"NumSamples": 500}
    cfg._serialized_document = json.dumps({"NumSamples": 500})
    return cfg


class TestDataPreparerConstruction:
    def test_basic_construction(self, mock_session):
        dc = DataPreparer(
            sagemaker_session=mock_session,
            role="arn:aws:iam::123:role/MyRole",
            base_job_name="my-base",
        )
        assert dc.role == "arn:aws:iam::123:role/MyRole"
        assert dc.base_job_name == "my-base"

    def test_extra_fields_forbidden(self, mock_session):
        with pytest.raises(Exception):
            DataPreparer(sagemaker_session=mock_session, unknown_field="bad")


class TestDualCallable:
    """generate / evaluate must work both bound (instance) and unbound (class).

    Validates design §2.2.6 / Req 1.4, 1.5 — the classmethod shortcut builds a
    transient ``DataPreparer(sagemaker_session=..., role=..., base_job_name=...)``
    and delegates to the instance method, so both call shapes flow through the
    same ``_run`` pipeline (and therefore the same intelligent-defaults
    resolution).
    """

    def test_instance_method_call_shape(self, mock_session, resolved_dataset):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")
        captured = {}

        def recording_run(self, dataset, category, **kwargs):
            captured["self"] = self
            captured["dataset"] = dataset
            captured["category"] = category
            captured["kwargs"] = kwargs
            return MagicMock()

        with patch.object(DataPreparer, "_run", recording_run):
            dc.generate(resolved_dataset, job_name="my-job", base_job_name="my-base")

        # self must be the original instance (not a transient one)
        assert captured["self"] is dc
        assert captured["dataset"] is resolved_dataset
        assert captured["category"] == JobCategory.SYNTHETIC_DATA_GENERATION
        assert captured["kwargs"] == {
            "job_name": "my-job",
            "base_job_name": "my-base",
            "wait": True,
            "logs": False,
            "tags": None,
            "timeout": None,
        }

    def test_classmethod_call_builds_transient_instance(self, mock_session, resolved_dataset):
        captured = {}

        def recording_run(self, dataset, category, **kwargs):
            captured["self"] = self
            captured["dataset"] = dataset
            captured["category"] = category
            captured["kwargs"] = kwargs
            return MagicMock()

        with patch.object(DataPreparer, "_run", recording_run):
            DataPreparer.generate(
                resolved_dataset,
                sagemaker_session=mock_session,
                role="arn:role:from-classmethod",
                base_job_name="classmethod-base",
                job_name="explicit-job",
            )

        inst = captured["self"]
        assert isinstance(inst, DataPreparer)
        assert inst.sagemaker_session is mock_session
        assert inst.role == "arn:role:from-classmethod"
        assert inst.base_job_name == "classmethod-base"
        assert captured["dataset"] is resolved_dataset
        assert captured["category"] == JobCategory.SYNTHETIC_DATA_GENERATION
        # job_name flows through; base_job_name is None as a per-call kwarg
        # because it's already pinned on the transient instance.
        assert captured["kwargs"]["job_name"] == "explicit-job"
        assert captured["kwargs"]["base_job_name"] is None

    def test_classmethod_with_no_args_uses_intelligent_defaults(self, resolved_dataset):
        """DataPreparer.generate(dataset) — no constructor kwargs, no per-call
        kwargs beyond the dataset. Intelligent defaults flow through _run at
        runtime; we only verify the transient instance is constructed with
        all-None fields so the downstream fallbacks fire.
        """
        captured = {}

        def recording_run(self, dataset, category, **kwargs):
            captured["self"] = self
            return MagicMock()

        with patch.object(DataPreparer, "_run", recording_run):
            DataPreparer.generate(resolved_dataset)

        inst = captured["self"]
        assert inst.sagemaker_session is None
        assert inst.role is None
        assert inst.base_job_name is None

    def test_evaluate_dual_callable_bound_to_dq(self, mock_session, resolved_dataset):
        """Same dispatch shape for evaluate, bound to DataQualityEvaluation."""
        resolved_dataset.job_category = JobCategory.DATA_QUALITY_EVALUATION
        captured = {}

        def recording_run(self, dataset, category, **kwargs):
            captured["category"] = category
            return MagicMock()

        with patch.object(DataPreparer, "_run", recording_run):
            DataPreparer.evaluate(resolved_dataset)
        assert captured["category"] == JobCategory.DATA_QUALITY_EVALUATION

    def test_transform_dual_callable_bound_to_dt(self, mock_session, resolved_dataset):
        """Same dispatch shape for transform, bound to DataTransformation."""
        resolved_dataset.job_category = JobCategory.DATA_TRANSFORMATION
        captured = {}

        def recording_run(self, dataset, category, **kwargs):
            captured["category"] = category
            return MagicMock()

        with patch.object(DataPreparer, "_run", recording_run):
            DataPreparer.transform(resolved_dataset)
        assert captured["category"] == JobCategory.DATA_TRANSFORMATION


class TestCategoryInference:
    def test_category_inferred_for_generate(self, mock_session, resolved_dataset):
        resolved_dataset.job_category = None
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_config_document = json.dumps({"DatasetArn": "arn:ds"})

        with patch.object(dc, "_create_job", return_value=mock_job):
            with patch.object(dc, "_rehydrate_datasets", return_value=[MagicMock()]):
                dc.generate(resolved_dataset, wait=True, logs=False)

        assert resolved_dataset.job_category == JobCategory.SYNTHETIC_DATA_GENERATION

    def test_category_mismatch_raises(self, mock_session, resolved_dataset):
        resolved_dataset.job_category = JobCategory.DATA_QUALITY_EVALUATION
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        with pytest.raises(DataPreparerConfigurationError) as exc_info:
            dc.generate(resolved_dataset)

        assert "does not match" in str(exc_info.value)


class TestWaitFalseReturnsJobHandle:
    """wait=False submits and returns the Job handle (not None), so the caller
    can poll/wait later or persist job_name and re-attach."""

    def test_wait_false_returns_job_handle(self, mock_session, resolved_dataset):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")
        mock_job = MagicMock()

        with patch.object(dc, "_create_job", return_value=mock_job) as create:
            result = dc.generate(resolved_dataset, wait=False)

        assert result is mock_job
        create.assert_called_once()

    def test_wait_false_does_not_wait_or_rehydrate(self, mock_session, resolved_dataset):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")
        mock_job = MagicMock()

        with patch.object(dc, "_create_job", return_value=mock_job):
            with patch.object(dc, "_rehydrate_datasets") as rehydrate:
                dc.generate(resolved_dataset, wait=False)

        mock_job.wait.assert_not_called()
        rehydrate.assert_not_called()


class TestCreateJob:
    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    def test_request_has_required_fields(self, mock_get_job, mock_session, resolved_dataset):
        mock_job_cls = MagicMock()
        mock_get_job.return_value = mock_job_cls
        mock_job_cls.create.return_value = MagicMock()
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        dc._create_job(resolved_dataset, JobCategory.SYNTHETIC_DATA_GENERATION)

        call_kwargs = mock_job_cls.create.call_args[1]
        assert "job_name" in call_kwargs
        assert "role_arn" in call_kwargs
        assert "job_category" in call_kwargs
        assert "job_config_schema_version" in call_kwargs
        assert "job_config_document" in call_kwargs
        assert "session" in call_kwargs

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    def test_no_forbidden_fields(self, mock_get_job, mock_session, resolved_dataset):
        mock_job_cls = MagicMock()
        mock_get_job.return_value = mock_job_cls
        mock_job_cls.create.return_value = MagicMock()
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        dc._create_job(resolved_dataset, JobCategory.SYNTHETIC_DATA_GENERATION)

        call_kwargs = mock_job_cls.create.call_args[1]
        assert "SourceIdentity" not in call_kwargs
        assert "source_identity" not in call_kwargs
        assert "CreationTime" not in call_kwargs
        assert "creation_time" not in call_kwargs
        assert "IgnoredBySweeper" not in call_kwargs
        assert "ignored_by_sweeper" not in call_kwargs
        assert "UseMockAlgo" not in call_kwargs
        assert "use_mock_algo" not in call_kwargs


class TestTags:
    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    def test_tags_passed_to_job_create(self, mock_get_job, mock_session, resolved_dataset):
        from sagemaker.core.shapes import Tag

        mock_job_cls = MagicMock()
        mock_get_job.return_value = mock_job_cls
        mock_job_cls.create.return_value = MagicMock()
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        tags = [Tag(key="team", value="data-science"), Tag(key="project", value="sdg")]
        dc._create_job(resolved_dataset, JobCategory.SYNTHETIC_DATA_GENERATION, tags=tags)

        call_kwargs = mock_job_cls.create.call_args[1]
        assert call_kwargs["tags"] == tags

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    def test_tags_omitted_when_none(self, mock_get_job, mock_session, resolved_dataset):
        mock_job_cls = MagicMock()
        mock_get_job.return_value = mock_job_cls
        mock_job_cls.create.return_value = MagicMock()
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        dc._create_job(resolved_dataset, JobCategory.SYNTHETIC_DATA_GENERATION, tags=None)

        call_kwargs = mock_job_cls.create.call_args[1]
        assert "tags" not in call_kwargs

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    def test_generate_threads_tags_to_create_job(
        self, mock_get_job, mock_session, resolved_dataset
    ):
        from sagemaker.core.shapes import Tag

        mock_job_cls = MagicMock()
        mock_get_job.return_value = mock_job_cls
        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_config_document = json.dumps(
            {"DatasetArn": "arn:aws:sagemaker:us-west-2:123:dataset/test"}
        )
        mock_job_cls.create.return_value = mock_job
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        tags = [Tag(key="env", value="beta")]
        dc.generate(resolved_dataset, tags=tags, wait=False)

        call_kwargs = mock_job_cls.create.call_args[1]
        assert call_kwargs["tags"] == tags

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    def test_evaluate_threads_tags_to_create_job(self, mock_get_job, mock_session):
        from sagemaker.core.shapes import Tag

        cfg = DatasetJobConfig(
            job_category=JobCategory.DATA_QUALITY_EVALUATION,
            dataset_arn="arn:aws:sagemaker:us-west-2:123:dataset/test",
        )
        cfg._resolved_schema_version = "0.2.0"
        cfg._resolved_schema = {"properties": {}}
        cfg._dynamic_model = MagicMock()
        cfg._validated_payload = {"DatasetArn": "arn:aws:sagemaker:us-west-2:123:dataset/test"}
        cfg._serialized_document = json.dumps(cfg._validated_payload)

        mock_job_cls = MagicMock()
        mock_get_job.return_value = mock_job_cls
        mock_job_cls.create.return_value = MagicMock()
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        tags = [Tag(key="cost-center", value="12345")]
        dc.evaluate(cfg, tags=tags, wait=False)

        call_kwargs = mock_job_cls.create.call_args[1]
        assert call_kwargs["tags"] == tags

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    def test_transform_threads_tags_to_create_job(self, mock_get_job, mock_session):
        from sagemaker.core.shapes import Tag

        cfg = DatasetJobConfig(
            job_category=JobCategory.DATA_TRANSFORMATION,
        )
        cfg._resolved_schema_version = "1.0.0"
        cfg._resolved_schema = {"properties": {}}
        cfg._dynamic_model = MagicMock()
        cfg._validated_payload = {}
        cfg._serialized_document = json.dumps({})

        mock_job_cls = MagicMock()
        mock_get_job.return_value = mock_job_cls
        mock_job_cls.create.return_value = MagicMock()
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        tags = [Tag(key="team", value="ml-ops")]
        dc.transform(cfg, tags=tags, wait=False)

        call_kwargs = mock_job_cls.create.call_args[1]
        assert call_kwargs["tags"] == tags

    def test_tags_reach_boto_client_create_job(self, mock_session):
        """Deeper test: verify tags are serialized and reach client.create_job (boto layer).

        This catches bugs where _job_resource.py swallows tags via **kwargs
        without forwarding them to the API request dict.
        """
        from sagemaker.ai_registry.data_preparer._job_resource import Job
        from sagemaker.core.shapes import Tag

        mock_client = MagicMock()
        mock_client.create_job.return_value = {
            "JobArn": "arn:aws:sagemaker:us-west-2:123:job/SDG/test-1"
        }
        # Make the service-model introspection succeed with Tags as accepted
        mock_op_model = MagicMock()
        mock_op_model.input_shape.members.keys.return_value = [
            "JobName",
            "RoleArn",
            "JobCategory",
            "JobConfigSchemaVersion",
            "JobConfigDocument",
            "JobArn",
            "CustomerDetails",
            "Tags",
        ]
        mock_client.meta.service_model.operation_model.return_value = mock_op_model
        mock_session.sagemaker_client = mock_client

        tags = [Tag(key="team", value="data-science"), Tag(key="env", value="beta")]

        Job.create(
            job_name="test-job",
            role_arn="arn:role",
            job_category="SyntheticDataGeneration",
            job_config_schema_version="1.0.0",
            job_config_document="{}",
            session=mock_session,
            tags=tags,
        )

        call_kwargs = mock_client.create_job.call_args[1]
        assert (
            "Tags" in call_kwargs
        ), f"Tags missing from boto request. Keys: {list(call_kwargs.keys())}"
        assert call_kwargs["Tags"] == [
            {"Key": "team", "Value": "data-science"},
            {"Key": "env", "Value": "beta"},
        ]

    def test_tags_absent_from_boto_when_none(self, mock_session):
        """Verify Tags key is NOT in the boto request when tags=None."""
        from sagemaker.ai_registry.data_preparer._job_resource import Job

        mock_client = MagicMock()
        mock_client.create_job.return_value = {
            "JobArn": "arn:aws:sagemaker:us-west-2:123:job/SDG/test-1"
        }
        mock_op_model = MagicMock()
        mock_op_model.input_shape.members.keys.return_value = [
            "JobName",
            "RoleArn",
            "JobCategory",
            "JobConfigSchemaVersion",
            "JobConfigDocument",
            "JobArn",
            "CustomerDetails",
            "Tags",
        ]
        mock_client.meta.service_model.operation_model.return_value = mock_op_model
        mock_session.sagemaker_client = mock_client

        Job.create(
            job_name="test-job",
            role_arn="arn:role",
            job_category="SyntheticDataGeneration",
            job_config_schema_version="1.0.0",
            job_config_document="{}",
            session=mock_session,
            tags=None,
        )

        call_kwargs = mock_client.create_job.call_args[1]
        assert (
            "Tags" not in call_kwargs
        ), f"Tags should not be in request when None. Keys: {list(call_kwargs.keys())}"


class TestJobLifecycle:
    def test_wait_false_logs_true_emits_warning(self, mock_session, resolved_dataset, caplog):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        with patch.object(dc, "_create_job") as mock_create:
            mock_create.return_value = MagicMock()
            import logging

            with caplog.at_level(logging.WARNING):
                result = dc.generate(resolved_dataset, wait=False, logs=True)

        # wait=False returns the Job handle; logs=True still warns that logs
        # are not shown in fire-and-forget mode.
        assert result is mock_create.return_value
        assert "Not displaying the job logs" in caplog.text

    def test_failed_job_raises(self, mock_session, resolved_dataset):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Failed"
        mock_job.failure_reason = "OOM"
        mock_job.secondary_status = "GENERATING"
        mock_job.job_arn = "arn:job"
        mock_job.job_name = "my-job"

        with patch.object(dc, "_create_job", return_value=mock_job):
            with pytest.raises(DataPreparerJobFailedError) as exc_info:
                dc.generate(resolved_dataset)

        assert exc_info.value.failure_reason == "OOM"

    def test_hub_finalization_failed_raises_not_success(self, mock_session, resolved_dataset):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Failed"
        mock_job.failure_reason = "HUB_FINALIZATION_FAILED"
        mock_job.job_arn = "arn:job"
        mock_job.job_name = "my-job"

        with patch.object(dc, "_create_job", return_value=mock_job):
            with pytest.raises(DataPreparerJobFailedError) as exc_info:
                dc.generate(resolved_dataset)

        assert exc_info.value.failure_reason == "HUB_FINALIZATION_FAILED"

    def test_stopped_job_raises(self, mock_session, resolved_dataset):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Stopped"
        mock_job.job_arn = "arn:job"
        mock_job.job_name = "my-job"

        with patch.object(dc, "_create_job", return_value=mock_job):
            with pytest.raises(DataPreparerJobStoppedError):
                dc.generate(resolved_dataset)

    def test_waiter_timeout_raises(self, mock_session, resolved_dataset):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()

        class WaiterError(Exception):
            pass

        mock_job.wait.side_effect = WaiterError("timed out")
        mock_job.job_arn = "arn:job"

        with patch.object(dc, "_create_job", return_value=mock_job):
            with pytest.raises(DataPreparerJobTimeoutError):
                dc.generate(resolved_dataset)

    def test_completed_missing_dataset_arn_raises(self, mock_session, resolved_dataset):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_config_document = json.dumps({})  # No DatasetArn
        mock_job.job_arn = "arn:job"
        mock_job.job_name = "my-job"

        with patch.object(dc, "_create_job", return_value=mock_job):
            with pytest.raises(DataPreparerJobError) as exc_info:
                dc.generate(resolved_dataset)

        assert "DatasetArn" in str(exc_info.value)


class TestTelemetry:
    """Telemetry decorator emits DATA_PREPARATION feature on generate/evaluate."""

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_generate_emits_telemetry_on_success(
        self, mock_resolve_config, mock_send_telemetry, mock_session, resolved_dataset
    ):
        mock_resolve_config.return_value = False
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_config_document = json.dumps({"DatasetArn": "arn:ds"})

        with patch.object(dc, "_create_job", return_value=mock_job):
            with patch.object(dc, "_rehydrate_datasets", return_value=[MagicMock()]):
                dc.generate(resolved_dataset, wait=True, logs=False)

        mock_send_telemetry.assert_called_once()
        args = mock_send_telemetry.call_args.args
        assert args[0] == 1  # SUCCESS status
        assert 22 in args[1]  # DATA_PREPARATION feature code
        extra = args[5]
        assert "DataPreparer.generate" in extra

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_evaluate_emits_telemetry_on_success(
        self, mock_resolve_config, mock_send_telemetry, mock_session, resolved_dataset
    ):
        mock_resolve_config.return_value = False
        resolved_dataset.job_category = JobCategory.DATA_QUALITY_EVALUATION
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_config_document = json.dumps({"DatasetArn": "arn:ds"})

        with patch.object(dc, "_create_job", return_value=mock_job):
            with patch.object(dc, "_rehydrate_datasets", return_value=[MagicMock()]):
                dc.evaluate(resolved_dataset, wait=True, logs=False)

        mock_send_telemetry.assert_called_once()
        args = mock_send_telemetry.call_args.args
        assert args[0] == 1  # SUCCESS
        assert 22 in args[1]  # DATA_PREPARATION feature code
        extra = args[5]
        assert "DataPreparer.evaluate" in extra

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_transform_emits_telemetry_on_success(
        self, mock_resolve_config, mock_send_telemetry, mock_session, resolved_dataset
    ):
        mock_resolve_config.return_value = False
        resolved_dataset.job_category = JobCategory.DATA_TRANSFORMATION
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_config_document = json.dumps(
            {
                "OutputDatasetArns": [
                    "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/out-ds/1.0.0"
                ]
            }
        )

        with patch.object(dc, "_create_job", return_value=mock_job):
            with patch.object(dc, "_rehydrate_datasets", return_value=[MagicMock()]):
                dc.transform(resolved_dataset, wait=True, logs=False)

        mock_send_telemetry.assert_called_once()
        args = mock_send_telemetry.call_args.args
        assert args[0] == 1  # SUCCESS
        assert 22 in args[1]  # DATA_PREPARATION feature code
        extra = args[5]
        assert "DataPreparer.transform" in extra

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_generate_emits_telemetry_on_failure(
        self, mock_resolve_config, mock_send_telemetry, mock_session, resolved_dataset
    ):
        mock_resolve_config.return_value = False
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Failed"
        mock_job.job_arn = "arn:job"
        mock_job.job_name = "my-job"
        mock_job.failure_reason = "OOM"
        mock_job.secondary_status = None

        with patch.object(dc, "_create_job", return_value=mock_job):
            with pytest.raises(DataPreparerJobFailedError):
                dc.generate(resolved_dataset)

        mock_send_telemetry.assert_called_once()
        args = mock_send_telemetry.call_args.args
        assert args[0] == 0  # FAILURE status
        assert 22 in args[1]  # DATA_PREPARATION feature code

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_generate_emits_resource_arn_on_success(
        self, mock_resolve_config, mock_send_telemetry, mock_session, resolved_dataset
    ):
        """Telemetry emits x-resourceArn from DataSet.job_arn on success."""
        mock_resolve_config.return_value = False
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/sdg/my-job"
        mock_job.job_config_document = json.dumps({"DatasetArn": "arn:ds"})

        mock_ds = MagicMock()
        mock_ds.__class__.__name__ = "DataSet"

        with patch.object(dc, "_create_job", return_value=mock_job):
            with patch.object(dc, "_rehydrate_datasets", return_value=[mock_ds]):
                result = dc.generate(resolved_dataset, wait=True, logs=False)

        # Verify job_arn was stashed on the returned DataSet
        assert result.job_arn == "arn:aws:sagemaker:us-west-2:123:job/sdg/my-job"

        # Verify telemetry includes x-resourceArn
        mock_send_telemetry.assert_called_once()
        extra = mock_send_telemetry.call_args.args[5]
        assert "x-resourceArn=arn:aws:sagemaker:us-west-2:123:job/sdg/my-job" in extra

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    @patch("sagemaker.core.telemetry.telemetry_logging.resolve_value_from_config")
    def test_generate_emits_resource_arn_on_failure(
        self, mock_resolve_config, mock_send_telemetry, mock_session, resolved_dataset
    ):
        """Telemetry emits x-resourceArn from exception.job_arn on failure."""
        mock_resolve_config.return_value = False
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Failed"
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/sdg/failed-job"
        mock_job.job_name = "failed-job"
        mock_job.failure_reason = "OOM"
        mock_job.secondary_status = None

        with patch.object(dc, "_create_job", return_value=mock_job):
            with pytest.raises(DataPreparerJobFailedError):
                dc.generate(resolved_dataset)

        mock_send_telemetry.assert_called_once()
        extra = mock_send_telemetry.call_args.args[5]
        assert "x-resourceArn=arn:aws:sagemaker:us-west-2:123:job/sdg/failed-job" in extra


# v0.2 DQ wire-shape tests. Cover the two places in preparer.py where the
# JobConfigDocument shape flips between v1 and v2: _create_job's DQ progress
# tracking and _rehydrate_datasets's DescribeJob rehydration. Standalone shape
# extractors are pinned in test_api_shape.py.

_V2_DQ_DATASET_ARN = (
    "arn:aws:sagemaker:us-west-2:123456789012:hub-content/AIRegistry/DataSet/my-input/1.0.0"
)
_V2_DQ_OUTPUT_S3 = "s3://bucket/dq-output/v2/"


@pytest.fixture
def resolved_v0_2_dq_dataset():
    """A DatasetJobConfig pre-resolved to the v0.2 wire shape.

    Bypasses ``_resolve_and_validate`` by populating the same private attrs the
    real path would set. Enough for ``_create_job`` / ``_rehydrate_datasets`` to
    read the version + validated payload.
    """
    cfg = DatasetJobConfig(
        job_category=JobCategory.DATA_QUALITY_EVALUATION,
        schema_version="0.2.0",
    )
    cfg._resolved_schema_version = "0.2.0"
    cfg._resolved_schema = {"properties": {"InputDataConfig": {"type": "array"}}}
    cfg._dynamic_model = MagicMock()
    v2_payload = {
        "InputDataConfig": [{"DatasetArn": _V2_DQ_DATASET_ARN}],
        "EvaluationName": "my-eval",
        "OutputDataConfig": {"S3OutputPath": _V2_DQ_OUTPUT_S3},
        "EvaluationConfig": {"Deterministic": {"Metrics": [{"Metric": "compression_ratio"}]}},
    }
    cfg._validated_payload = v2_payload
    cfg._serialized_document = json.dumps(v2_payload)
    return cfg


class TestCreateJobDqV0_2:
    """The CreateJob envelope forwards jobConfigSchemaVersion='0.2.0' + the v2
    JobConfigDocument verbatim. No client-side shape rewriting — the service
    speaks the wire shape."""

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    def test_envelope_carries_v0_2_version_and_v2_document(
        self, mock_get_job, mock_session, resolved_v0_2_dq_dataset
    ):
        mock_job_cls = MagicMock()
        mock_get_job.return_value = mock_job_cls
        mock_job_cls.create.return_value = MagicMock()
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        dc._create_job(resolved_v0_2_dq_dataset, JobCategory.DATA_QUALITY_EVALUATION)

        call_kwargs = mock_job_cls.create.call_args[1]
        assert call_kwargs["job_config_schema_version"] == "0.2.0"

        doc = json.loads(call_kwargs["job_config_document"])
        assert doc["InputDataConfig"][0]["DatasetArn"] == _V2_DQ_DATASET_ARN
        assert doc["OutputDataConfig"]["S3OutputPath"] == _V2_DQ_OUTPUT_S3
        # v2 wire shape must NOT carry the legacy fields — mixing shapes is a
        # validation error.
        assert "DatasetArn" not in doc
        assert "OutputS3Uri" not in doc["OutputDataConfig"]

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    def test_dq_progress_extraction_reads_v2_fields(
        self, mock_get_job, mock_session, resolved_v0_2_dq_dataset
    ):
        """Progress-panel plumbing must find S3OutputPath (not OutputS3Uri) and
        InputDataConfig[0].DatasetArn (not scalar DatasetArn) for v0.2 jobs.
        Otherwise the Rich status panel goes blank and record-counting silently
        disables."""
        mock_job_cls = MagicMock()
        mock_get_job.return_value = mock_job_cls
        mock_job_cls.create.return_value = MagicMock()
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        # Stub the S3 record-counting helper so the test doesn't touch the network.
        with patch.object(dc, "_count_input_dataset_records", return_value=42) as counter:
            dc._create_job(resolved_v0_2_dq_dataset, JobCategory.DATA_QUALITY_EVALUATION)

        # Counter is called with the ARN extracted from InputDataConfig[0], not
        # the missing scalar DatasetArn.
        counter.assert_called_once_with(_V2_DQ_DATASET_ARN)

        # progress_output_s3_uri is threaded through to Job.create from the v2
        # S3OutputPath field.
        call_kwargs = mock_job_cls.create.call_args[1]
        assert call_kwargs["progress_output_s3_uri"] == _V2_DQ_OUTPUT_S3
        assert call_kwargs["expected_samples"] == 42


class TestRehydrateDatasetDqV3:
    """A completed 3.0.0 job rehydrates from ``InputDataConfig[0]``, as v2 does."""

    def test_rehydrates_from_input_data_config_when_version_is_v3(self, mock_session):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/DataQualityEvaluation/dq-v3"
        mock_job.job_name = "dq-v3"
        mock_job.job_config_schema_version = "3.0.0"
        mock_job.job_config_document = json.dumps(
            {
                "InputDataConfig": [{"DatasetArn": _V2_DQ_DATASET_ARN}],
                "OutputDataConfig": {"S3OutputPath": _V2_DQ_OUTPUT_S3},
                "EvaluationConfig": {"Operations": [{"Operator": "default_metrics"}]},
            }
        )

        with patch(
            "sagemaker.ai_registry.dataset.DataSet.get",
            return_value=MagicMock(),
        ) as mock_get:
            dc._rehydrate_datasets(mock_job, JobCategory.DATA_QUALITY_EVALUATION)

        mock_get.assert_called_once()
        assert mock_get.call_args.kwargs["name"] == "my-input"


class TestRehydrateDatasetDqV0_2:
    """DescribeJob echoes the canonical v2 wire shape. ``_rehydrate_datasets``
    must find the DatasetArn under ``InputDataConfig[0]``, not the legacy scalar."""

    def test_rehydrates_from_input_data_config_when_version_is_v0_2(self, mock_session):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/DataQualityEvaluation/dq-v2"
        mock_job.job_name = "dq-v2"
        mock_job.job_config_schema_version = "0.2.0"
        mock_job.job_config_document = json.dumps(
            {
                "InputDataConfig": [{"DatasetArn": _V2_DQ_DATASET_ARN}],
                "EvaluationName": "my-eval",
                "OutputDataConfig": {"S3OutputPath": _V2_DQ_OUTPUT_S3},
            }
        )

        with patch(
            "sagemaker.ai_registry.dataset.DataSet.get",
            return_value=MagicMock(),
        ) as mock_get:
            dc._rehydrate_datasets(mock_job, JobCategory.DATA_QUALITY_EVALUATION)

        # DataSet.get is called with the content name parsed from the v2-shape
        # ARN. Without version-aware extraction this test would raise
        # DataPreparerJobError("missing DatasetArn") — regressing today's SDK.
        mock_get.assert_called_once()
        assert mock_get.call_args.kwargs["name"] == "my-input"

    def test_rehydrates_with_real_job_resource_after_refresh(self, mock_session):
        """Regression: rehydration must work with the REAL ``Job`` resource.

        The sibling tests hand-set ``job_config_schema_version`` on a
        ``MagicMock`` — masking the bug where the real ``Job`` never populated
        the attribute. ``_rehydrate_datasets`` then read ``None`` via getattr,
        fell back to the v1 parser, and raised
        'Completed job missing DatasetArn in JobConfigDocument' for every
        completed v2 (0.2.0) job. This test drives the real refresh() path.
        """
        from sagemaker.ai_registry.data_preparer._job_resource import Job

        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        job_session = MagicMock()
        job_session.sagemaker_client.describe_job.return_value = {
            "JobArn": "arn:aws:sagemaker:us-west-2:123:job/DataQualityEvaluation/dq-v2",
            "JobStatus": "Completed",
            "JobConfigSchemaVersion": "0.2.0",
            "JobConfigDocument": json.dumps(
                {
                    "InputDataConfig": [{"DatasetArn": _V2_DQ_DATASET_ARN}],
                    "EvaluationName": "my-eval",
                    "OutputDataConfig": {"S3OutputPath": _V2_DQ_OUTPUT_S3},
                }
            ),
        }
        job = Job(
            job_name="dq-v2",
            job_category="DataQualityEvaluation",
            session=job_session,
        )
        job.refresh()

        with patch(
            "sagemaker.ai_registry.dataset.DataSet.get",
            return_value=MagicMock(),
        ) as mock_get:
            dc._rehydrate_datasets(job, JobCategory.DATA_QUALITY_EVALUATION)

        mock_get.assert_called_once()
        assert mock_get.call_args.kwargs["name"] == "my-input"

    def test_rehydrates_from_scalar_when_version_is_v1(self, mock_session):
        """Regression guard: the v1 code path must be byte-identical to today."""
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/DataQualityEvaluation/dq-v1"
        mock_job.job_name = "dq-v1"
        mock_job.job_config_schema_version = "1.0.0"
        mock_job.job_config_document = json.dumps(
            {
                "DatasetArn": _V2_DQ_DATASET_ARN,
                "EvaluationName": "my-eval",
                "OutputDataConfig": {"OutputS3Uri": _V2_DQ_OUTPUT_S3},
            }
        )

        with patch(
            "sagemaker.ai_registry.dataset.DataSet.get",
            return_value=MagicMock(),
        ) as mock_get:
            dc._rehydrate_datasets(mock_job, JobCategory.DATA_QUALITY_EVALUATION)

        mock_get.assert_called_once()
        assert mock_get.call_args.kwargs["name"] == "my-input"


class TestParseJobArn:
    """Tests for _parse_job_arn helper."""

    def test_valid_sdg_arn(self):
        from sagemaker.ai_registry.data_preparer.preparer import _parse_job_arn

        result = _parse_job_arn(
            "arn:aws:sagemaker:us-west-2:123456789012:job/SyntheticDataGeneration/my-job-2026"
        )
        assert result["region"] == "us-west-2"
        assert result["account"] == "123456789012"
        assert result["category"] == "SyntheticDataGeneration"
        assert result["job_name"] == "my-job-2026"

    def test_valid_dq_arn(self):
        from sagemaker.ai_registry.data_preparer.preparer import _parse_job_arn

        result = _parse_job_arn(
            "arn:aws:sagemaker:eu-west-1:999999999999:job/DataQualityEvaluation/eval-job-001"
        )
        assert result["category"] == "DataQualityEvaluation"
        assert result["job_name"] == "eval-job-001"

    def test_valid_transform_arn(self):
        from sagemaker.ai_registry.data_preparer.preparer import _parse_job_arn

        result = _parse_job_arn(
            "arn:aws:sagemaker:us-east-1:111111111111:job/DataTransformation/split-job"
        )
        assert result["category"] == "DataTransformation"
        assert result["job_name"] == "split-job"

    def test_invalid_arn_raises(self):
        from sagemaker.ai_registry.data_preparer.preparer import _parse_job_arn

        with pytest.raises(ValueError, match="Invalid job ARN format"):
            _parse_job_arn("not-an-arn")

    def test_missing_category_raises(self):
        from sagemaker.ai_registry.data_preparer.preparer import _parse_job_arn

        with pytest.raises(ValueError, match="Invalid job ARN resource segment"):
            _parse_job_arn("arn:aws:sagemaker:us-west-2:123:job/missing-job-name")


class TestAttach:
    """Tests for DataPreparer.attach()."""

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_attach_completed_job_returns_dataset(self, mock_get_session, mock_get_job_cls):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_job = MagicMock()
        mock_job.job_name = "my-sdg-job"
        mock_job.job_status = "Completed"
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/my-sdg-job"
        mock_job.job_config_document = json.dumps(
            {"DatasetArn": "arn:aws:sagemaker:us-west-2:123:hub-content/Dataset/my-ds/1"}
        )
        mock_job.job_config_schema_version = "1.0.0"

        MockJob = MagicMock()
        MockJob.get.return_value = mock_job
        mock_get_job_cls.return_value = MockJob

        mock_ds = MagicMock()
        with patch.object(DataPreparer, "_rehydrate_datasets", return_value=[mock_ds]):
            with patch.object(DataPreparer, "_preview_output"):
                result = DataPreparer.attach(
                    "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/my-sdg-job"
                )

        assert result == mock_ds
        assert (
            result.job_arn
            == "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/my-sdg-job"
        )
        MockJob.get.assert_called_once_with(
            job_name="my-sdg-job",
            job_category="SyntheticDataGeneration",
            session=mock_session,
        )

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_attach_inprogress_job_waits(self, mock_get_session, mock_get_job_cls):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_job = MagicMock()
        mock_job.job_name = "my-sdg-job"
        mock_job.job_status = "InProgress"
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/my-sdg-job"
        mock_job.job_config_document = json.dumps({"DatasetArn": "arn:ds"})
        mock_job.job_config_schema_version = "1.0.0"

        # After wait, status changes to Completed
        def fake_wait(logs=False, timeout=None):
            pass

        mock_job.wait = fake_wait

        def fake_refresh():
            mock_job.job_status = "Completed"
            return mock_job

        mock_job.refresh = fake_refresh

        MockJob = MagicMock()
        MockJob.get.return_value = mock_job
        mock_get_job_cls.return_value = MockJob

        mock_ds = MagicMock()
        with patch.object(DataPreparer, "_rehydrate_datasets", return_value=[mock_ds]):
            with patch.object(DataPreparer, "_preview_output"):
                result = DataPreparer.attach(
                    "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/my-sdg-job"
                )

        assert result == mock_ds

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_attach_wait_false_returns_job(self, mock_get_session, mock_get_job_cls):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_job = MagicMock()
        mock_job.job_name = "my-sdg-job"
        mock_job.job_status = "InProgress"

        MockJob = MagicMock()
        MockJob.get.return_value = mock_job
        mock_get_job_cls.return_value = MockJob

        result = DataPreparer.attach(
            "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/my-sdg-job",
            wait=False,
        )

        assert result == mock_job

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_attach_failed_job_raises(self, mock_get_session, mock_get_job_cls):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_job = MagicMock()
        mock_job.job_name = "my-sdg-job"
        mock_job.job_status = "Failed"
        mock_job.failure_reason = "OutOfMemory"
        mock_job.secondary_status = "Executing"

        MockJob = MagicMock()
        MockJob.get.return_value = mock_job
        mock_get_job_cls.return_value = MockJob

        with pytest.raises(DataPreparerJobFailedError):
            DataPreparer.attach(
                "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/my-sdg-job"
            )

    def test_attach_invalid_arn_raises(self):
        with pytest.raises(ValueError, match="Invalid job ARN"):
            DataPreparer.attach("garbage")

    @pytest.mark.parametrize("category", ["SomeOtherCategory", "AgentRFT"])
    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    def test_attach_unsupported_category_raises_before_any_call(self, mock_get_job_cls, category):
        """A job of a category DataPreparer does not handle is rejected up front,
        with the supported categories listed, and no DescribeJob call is made."""
        with pytest.raises(ValueError, match="Unsupported job category") as exc_info:
            DataPreparer.attach(
                f"arn:aws:sagemaker:us-west-2:123456789012:job/{category}/some-job",
                sagemaker_session=MagicMock(),
            )
        message = str(exc_info.value)
        for supported in ("SyntheticDataGeneration", "DataQualityEvaluation", "DataTransformation"):
            assert supported in message
        mock_get_job_cls.assert_not_called()


class _JobStub:
    """Test-double for the ``Job`` resource used by TestStop.

    ``DataPreparer.stop`` uses ``isinstance(job_or_arn, Job)`` to distinguish
    the handle path from the ARN path. ``isinstance`` requires a real class
    as the second arg — a MagicMock doesn't satisfy that — so the mocked
    ``_get_job_resource()`` returns this stub class instead.
    """

    get = MagicMock()


class TestStop:
    """Tests for DataPreparer.stop().

    Structurally identical to DataPreparer.attach(): classmethod, defaults
    ``wait=True`` + ``logs=False``, resolves the session via
    ``DataPreparerDefaults`` when hydrating from an ARN, and on ``wait=True``
    uses ``job.wait(logs=logs)`` re-mapping timeout-shaped exceptions to
    ``DataPreparerJobTimeoutError``. Accepts both a live Job handle
    (in-session) and an ARN string (cross-session).
    """

    _ARN = "arn:aws:sagemaker:us-west-2:123:job/SyntheticDataGeneration/my-sdg-job"

    def _job_stub(self, status="InProgress"):
        job = _JobStub()
        job.job_name = "my-sdg-job"
        job.job_arn = self._ARN
        job.job_status = status
        job.refresh = MagicMock()
        job.stop = MagicMock()
        job.wait = MagicMock()
        return job

    def _wire_arn_path(self, mock_get_job_cls, job):
        """Wire _get_job_resource so the ARN path returns ``job`` via Job.get."""
        _JobStub.get.reset_mock()
        _JobStub.get.return_value = job
        mock_get_job_cls.return_value = _JobStub

    # ------------------------------------------------------------------
    # ARN path
    # ------------------------------------------------------------------

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_stop_by_arn_hydrates_via_job_get(self, mock_get_session, mock_get_job_cls):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        job = self._job_stub()
        self._wire_arn_path(mock_get_job_cls, job)

        result = DataPreparer.stop(self._ARN, wait=False)

        _JobStub.get.assert_called_once_with(
            job_name="my-sdg-job",
            job_category="SyntheticDataGeneration",
            session=mock_session,
        )
        job.stop.assert_called_once_with()
        job.wait.assert_not_called()
        assert result is job

    # ------------------------------------------------------------------
    # Handle path
    # ------------------------------------------------------------------

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_stop_by_handle_uses_handle_directly(self, mock_get_session, mock_get_job_cls):
        mock_get_job_cls.return_value = _JobStub
        _JobStub.get.reset_mock()
        job = self._job_stub()

        result = DataPreparer.stop(job, wait=False)

        # Handle path must NOT re-resolve via Job.get or pull a session default.
        _JobStub.get.assert_not_called()
        mock_get_session.assert_not_called()
        # But we DO refresh the handle to pick up latest state.
        job.refresh.assert_called_once_with()
        job.stop.assert_called_once_with()
        assert result is job

    # ------------------------------------------------------------------
    # Terminal-state, wait behavior, invalid input
    # ------------------------------------------------------------------

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_stop_noop_when_already_terminal(self, mock_get_session, mock_get_job_cls):
        mock_get_session.return_value = MagicMock()
        job = self._job_stub(status="Stopped")
        self._wire_arn_path(mock_get_job_cls, job)

        result = DataPreparer.stop(self._ARN)

        job.stop.assert_not_called()
        job.wait.assert_not_called()
        assert result is job

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_stop_wait_true_calls_job_wait_and_refreshes(self, mock_get_session, mock_get_job_cls):
        mock_get_session.return_value = MagicMock()
        job = self._job_stub()
        self._wire_arn_path(mock_get_job_cls, job)

        # After wait, status flips to Stopped via the built-in waiter.
        def _wait(logs=False, timeout=None):
            job.job_status = "Stopped"

        job.wait.side_effect = _wait

        result = DataPreparer.stop(self._ARN, wait=True, logs=True)

        job.stop.assert_called_once_with()
        job.wait.assert_called_once_with(logs=True, timeout=None)
        # attach()-style: refresh() after wait to pick up final state.
        assert job.refresh.call_count >= 1
        assert result.job_status == "Stopped"

    def test_stop_invalid_arn_raises(self):
        with pytest.raises(ValueError, match="Invalid job ARN"):
            DataPreparer.stop("garbage")

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_stop_noop_when_completed(self, mock_get_session, mock_get_job_cls):
        """DataPreparer.stop is a no-op for Completed jobs (not just Stopped)."""
        mock_get_session.return_value = MagicMock()
        job = self._job_stub(status="Completed")
        self._wire_arn_path(mock_get_job_cls, job)

        result = DataPreparer.stop(self._ARN)

        job.stop.assert_not_called()
        job.wait.assert_not_called()
        assert result is job

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_stop_noop_when_failed(self, mock_get_session, mock_get_job_cls):
        """DataPreparer.stop is a no-op for Failed jobs (not just Stopped)."""
        mock_get_session.return_value = MagicMock()
        job = self._job_stub(status="Failed")
        self._wire_arn_path(mock_get_job_cls, job)

        result = DataPreparer.stop(self._ARN)

        job.stop.assert_not_called()
        job.wait.assert_not_called()
        assert result is job

    def test_stop_non_job_non_string_raises_TypeError(self):
        """Passing something that is neither a Job handle nor a str ARN raises TypeError."""
        with pytest.raises(TypeError, match="stop\\(\\) expects a Job handle"):
            DataPreparer.stop(42)

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_stop_service_error_propagates(self, mock_get_session, mock_get_job_cls):
        """A non-timeout exception from job.wait() is re-raised without wrapping."""
        from botocore.exceptions import ClientError

        mock_get_session.return_value = MagicMock()
        job = self._job_stub()
        self._wire_arn_path(mock_get_job_cls, job)

        service_error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
            "DescribeJob",
        )
        job.wait.side_effect = service_error

        with pytest.raises(ClientError, match="AccessDeniedException"):
            DataPreparer.stop(self._ARN, wait=True)

        job.stop.assert_called_once_with()

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_stop_wait_timeout_raises_DataPreparerJobTimeoutError(
        self, mock_get_session, mock_get_job_cls
    ):
        """A timeout-shaped exception from job.wait() is wrapped as DataPreparerJobTimeoutError."""
        mock_get_session.return_value = MagicMock()
        job = self._job_stub()
        self._wire_arn_path(mock_get_job_cls, job)

        job.wait.side_effect = DataPreparerJobTimeoutError(
            job_arn=self._ARN,
            elapsed_seconds=120,
            job_name="my-sdg-job",
        )

        with pytest.raises(DataPreparerJobTimeoutError):
            DataPreparer.stop(self._ARN, wait=True)

        job.stop.assert_called_once_with()


class TestJobStop:
    """Tests for Job.stop() instance method.

    The DataPreparer.stop() tests mock job.stop, so they never exercise the
    actual Job.stop() implementation. These tests drive Job.stop() directly
    to cover the client call and verify the correct API args are passed.
    """

    @pytest.fixture
    def session_with_region(self):
        s = MagicMock()
        s.boto_region_name = "us-west-2"
        s.account_id.return_value = "123456789012"
        return s

    def _make_job(self, session):
        from sagemaker.ai_registry.data_preparer._job_resource import Job

        return Job(
            job_name="my-stop-job",
            job_category="SyntheticDataGeneration",
            session=session,
        )

    def test_stop_calls_stop_job_api(self, session_with_region):
        """Job.stop() issues stop_job with the correct JobName and JobCategory."""
        job = self._make_job(session_with_region)
        job.stop()

        session_with_region.sagemaker_client.stop_job.assert_called_once_with(
            JobName="my-stop-job",
            JobCategory="SyntheticDataGeneration",
        )

    def test_stop_service_error_propagates(self, session_with_region):
        """A ClientError from stop_job propagates without wrapping."""
        from botocore.exceptions import ClientError

        job = self._make_job(session_with_region)
        session_with_region.sagemaker_client.stop_job.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFound", "Message": "job not found"}},
            "StopJob",
        )

        with pytest.raises(ClientError, match="ResourceNotFound"):
            job.stop()


class TestParseDatasetArn:
    """_parse_dataset_arn delegates to sagemaker-core's shared hub-ARN parser.

    The hub and version are only trusted for a well-formed hub-content ARN;
    every other shape falls back to a best-effort name with no hub and no
    version, so the lookup lands on the derived AI Registry hub and the latest
    version rather than mis-reading some other segment as a hub name.
    """

    def test_well_formed_arn_yields_name_hub_and_version(self):
        assert _parse_dataset_arn(
            "arn:aws:sagemaker:us-west-2:123456789012:hub-content/MyHub/DataSet/my-ds/3.0.0"
        ) == (
            "my-ds",
            "arn:aws:sagemaker:us-west-2:123456789012:hub/MyHub",
            "3.0.0",
        )

    @pytest.mark.parametrize(
        "arn,expected_name",
        [
            # Trailing segments past the version: the parser's regex would
            # otherwise absorb "1.0/extra" into the version.
            (
                "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/n/1.0/extra",
                "n",
            ),
            # A hub ARN, not a hub-content ARN.
            ("arn:aws:sagemaker:us-west-2:123:hub/H/DataSet/n/1", "n"),
            ("some-name/1.0.0", "some-name"),
            ("garbage", "garbage"),
        ],
    )
    def test_unexpected_shape_falls_back_to_name_only(self, arn, expected_name):
        assert _parse_dataset_arn(arn) == (expected_name, None, None)

    def test_non_aws_partition_round_trips(self):
        """A non-aws partition (e.g. aws-cn) in the source ARN must be preserved
        verbatim in the returned hub ARN; the partition must not be re-derived
        from the region."""
        cn_arn = (
            "arn:aws-cn:sagemaker:cn-north-1:123456789012:hub-content/HubCN/DataSet/cn-ds/1.0.0"
        )
        assert _parse_dataset_arn(cn_arn) == (
            "cn-ds",
            "arn:aws-cn:sagemaker:cn-north-1:123456789012:hub/HubCN",
            "1.0.0",
        )


class TestCountInputDatasetRecords:
    """The DQ record count resolves its input by the ARN's own hub and version.

    That ARN is the exact input the job was given. A bare-name lookup against
    the derived hub returns nothing for an input registered in any other hub,
    which surfaced as a missing record total in the progress display.
    """

    def test_resolves_dataset_by_hub_and_version_from_arn(self, mock_session):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        with patch("sagemaker.ai_registry.dataset.DataSet.get") as mock_get:
            mock_get.return_value = MagicMock(source=None)  # short-circuits before S3
            dc._count_input_dataset_records(
                "arn:aws:sagemaker:us-west-2:123:hub-content/OtherHub/DataSet/in-ds/2.0.0"
            )

        mock_get.assert_called_once_with(
            name="in-ds",
            sagemaker_session=mock_session,
            hub_name="arn:aws:sagemaker:us-west-2:123:hub/OtherHub",
            version="2.0.0",
        )

    def test_returns_none_when_lookup_fails(self, mock_session):
        """Best-effort and non-blocking: a failed lookup must not raise."""
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        with patch(
            "sagemaker.ai_registry.dataset.DataSet.get",
            side_effect=RuntimeError("ResourceNotFound"),
        ):
            assert dc._count_input_dataset_records(_V2_DQ_DATASET_ARN) is None


class TestPreviewOutput:
    """_preview_output uses ContentRange to detect truncation, not len >= cap.

    A ranged get_object returns ContentRange "bytes first-last/total"; truncated
    iff len(raw) < total.  This correctly handles an object whose size equals the
    Range cap: total == len(raw) → not truncated, so the final record (even with
    no trailing newline) is never dropped.  When ContentRange is absent, falls back
    to the original heuristic (len >= cap) so behaviour is never worse.
    """

    _SDG = JobCategory.SYNTHETIC_DATA_GENERATION

    # Four complete JSONL records; the last has NO trailing newline.
    # 32 bytes total — small enough to serve as a patched _PREVIEW_MAX_BYTES cap.
    _BODY = b'{"n":1}\n{"n":2}\n{"n":3}\n{"n":99}'

    def _make_dc(self, session):
        return DataPreparer(sagemaker_session=session, role="arn:role")

    @staticmethod
    def _mock_s3(body: bytes, content_range=None):
        import io

        resp = {"Body": io.BytesIO(body)}
        if content_range is not None:
            resp["ContentRange"] = content_range
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = resp
        return mock_s3

    def test_exact_cap_with_content_range_preserves_final_record(self, mock_session):
        """Bug: len(raw) == _cap classified as truncated by old heuristic.

        Fix: ContentRange total == len(raw) → truncated=False → decode as-is →
        the final record (no trailing newline) is included in the content passed
        to pd.read_json.
        """
        cap = len(self._BODY)  # 32: body == cap triggers old heuristic
        cr = f"bytes 0-{cap - 1}/{cap}"  # total == cap → NOT truncated

        content_seen = []

        # Save the real read_json before patching so we can re-use it.
        import io as _io
        import pandas as _pd

        _real = _pd.read_json

        def _capture(sio, **kwargs):
            content_seen.append(sio.getvalue())
            return _real(_io.StringIO(content_seen[-1]), **kwargs)

        dc = self._make_dc(mock_session)
        with patch("sagemaker.ai_registry.data_preparer.preparer._PREVIEW_MAX_BYTES", cap):
            with patch("boto3.client", return_value=self._mock_s3(self._BODY, cr)):
                with patch("pandas.read_json", side_effect=_capture):
                    with patch.object(
                        dc,
                        "_resolve_output_records_uri",
                        return_value="s3://bucket/key.jsonl",
                    ):
                        with patch("builtins.print"):
                            dc._preview_output(MagicMock(), self._SDG)

        assert content_seen, "pd.read_json was never called — preview was incorrectly skipped"
        # The final record {"n":99}, which has no trailing newline, must be present.
        assert (
            '"n":99' in content_seen[0] or '"n": 99' in content_seen[0]
        ), f"Final record missing from content: {content_seen[0]!r}"

    def test_truncated_object_drops_partial_tail_record(self, mock_session):
        """ContentRange total > len(raw) → truncated=True → tail after last \\n dropped."""
        cap = len(self._BODY)  # 32
        total = cap + 50  # object is larger than the Range read (genuinely truncated)
        cr = f"bytes 0-{cap - 1}/{total}"

        content_seen = []
        import io as _io
        import pandas as _pd

        _real = _pd.read_json

        def _capture(sio, **kwargs):
            content_seen.append(sio.getvalue())
            return _real(_io.StringIO(content_seen[-1]), **kwargs)

        dc = self._make_dc(mock_session)
        with patch("sagemaker.ai_registry.data_preparer.preparer._PREVIEW_MAX_BYTES", cap):
            with patch("boto3.client", return_value=self._mock_s3(self._BODY, cr)):
                with patch("pandas.read_json", side_effect=_capture):
                    with patch.object(
                        dc,
                        "_resolve_output_records_uri",
                        return_value="s3://bucket/key.jsonl",
                    ):
                        with patch("builtins.print"):
                            dc._preview_output(MagicMock(), self._SDG)  # must not raise

        assert content_seen, "pd.read_json was never called"
        # The partial tail {"n":99} (after the last \n) must be absent.
        assert (
            '"n":99' not in content_seen[0] and '"n": 99' not in content_seen[0]
        ), f"Partial tail unexpectedly present in content: {content_seen[0]!r}"
        # Complete records before the tail must be present.
        assert '"n":1' in content_seen[0] or '"n": 1' in content_seen[0]

    def test_no_content_range_fallback_heuristic(self, mock_session):
        """No ContentRange → fallback: truncated = len(raw) >= cap.

        Body == cap, no newlines (single oversized record): heuristic fires →
        truncated=True, no newline found → preview skipped, no exception raised.
        """
        body = b"x" * 50  # 50 bytes, no valid JSONL and no newlines
        cap = 50  # len(body) == cap → heuristic: truncated = True

        read_json_called = []

        dc = self._make_dc(mock_session)
        with patch("sagemaker.ai_registry.data_preparer.preparer._PREVIEW_MAX_BYTES", cap):
            with patch("boto3.client", return_value=self._mock_s3(body, None)):
                with patch(
                    "pandas.read_json", side_effect=lambda *a, **k: read_json_called.append(1)
                ):
                    with patch.object(
                        dc,
                        "_resolve_output_records_uri",
                        return_value="s3://bucket/key.jsonl",
                    ):
                        dc._preview_output(MagicMock(), self._SDG)  # must not raise

        # Heuristic fires (len == cap), no newline → preview skipped before read_json.
        assert not read_json_called, "read_json should not be called when preview is skipped"
