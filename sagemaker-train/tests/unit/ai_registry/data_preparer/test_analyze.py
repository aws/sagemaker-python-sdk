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
"""Unit tests for DataPreparer.analyze() — GapAnalysis."""
from __future__ import absolute_import

import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer.preparer import DataPreparer, _DualCallable
from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig
from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.core.helper.session_helper import Session


@pytest.fixture
def mock_session():
    session = MagicMock(spec=Session)
    session.boto_region_name = "us-west-2"
    session.account_id.return_value = "123456789012"
    return session


@pytest.fixture
def ga_dataset():
    """A DatasetJobConfig pre-resolved for GapAnalysis."""
    cfg = DatasetJobConfig(
        job_category=JobCategory.GAP_ANALYSIS,
    )
    cfg._resolved_schema_version = "1.0.0"
    cfg._resolved_schema = {"properties": {}}
    cfg._dynamic_model = MagicMock()
    cfg._validated_payload = {
        "ModelEvaluationJobArns": [
            "arn:aws:sagemaker:us-west-2:123456789012:job/GenAIEvaluation/my-eval-run"
        ],
        "OutputDataConfig": {"S3OutputPath": "s3://bucket/ga-out/"},
    }
    cfg._serialized_document = json.dumps(cfg._validated_payload)
    return cfg


class TestAnalyzeReturnsJob:
    def test_analyze_returns_job_on_completion(self, mock_session, ga_dataset):
        """GapAnalysis completed job returns the Job object, not a DataSet."""
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/GapAnalysis/my-job"
        # The service overlays MetadataS3Uri (readOnly) onto the returned
        # JobConfigDocument at finalization. It is the only pointer to the report.
        mock_job.job_config_document = json.dumps(
            {"MetadataS3Uri": "s3://bucket/ga-out/metadata.json"}
        )

        with patch.object(dc, "_create_job", return_value=mock_job):
            result = dc.analyze(ga_dataset, wait=True, logs=False)

        assert result is mock_job

    def test_analyze_does_not_rehydrate_datasets(self, mock_session, ga_dataset):
        """GapAnalysis path must NOT call _rehydrate_datasets."""
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_arn = "arn:job"

        with patch.object(dc, "_create_job", return_value=mock_job):
            with patch.object(dc, "_rehydrate_datasets") as mock_rehydrate:
                dc.analyze(ga_dataset, wait=True, logs=False)

        mock_rehydrate.assert_not_called()


class TestAnalyzeCategory:
    def test_analyze_uses_correct_category(self, mock_session, ga_dataset):
        """analyze() passes GAP_ANALYSIS to _run."""
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")
        captured = {}

        def recording_run(self, dataset, category, **kwargs):
            captured["category"] = category
            return MagicMock()

        with patch.object(DataPreparer, "_run", recording_run):
            dc.analyze(ga_dataset)

        assert captured["category"] == JobCategory.GAP_ANALYSIS

    def test_gap_analysis_wire_literal(self):
        """The wire value is the bare token the service publishes."""
        assert JobCategory.GAP_ANALYSIS.value == "GapAnalysis"


class TestAnalyzeDualCallable:
    def test_analyze_is_dual_callable(self):
        """DataPreparer.analyze is a _DualCallable descriptor."""
        assert isinstance(DataPreparer.__dict__["analyze"], _DualCallable)

    def test_analyze_classmethod_shape(self, mock_session, ga_dataset):
        """Class-level call builds transient instance and delegates."""
        captured = {}

        def recording_run(self, dataset, category, **kwargs):
            captured["self"] = self
            captured["category"] = category
            return MagicMock()

        with patch.object(DataPreparer, "_run", recording_run):
            DataPreparer.analyze(
                ga_dataset,
                sagemaker_session=mock_session,
                role="arn:role:class",
            )

        inst = captured["self"]
        assert isinstance(inst, DataPreparer)
        assert inst.role == "arn:role:class"
        assert captured["category"] == JobCategory.GAP_ANALYSIS


class TestAnalyzeSchemaResolution:
    def test_analyze_schema_resolves(self, mock_session, ga_dataset):
        """_resolve_and_validate is called for the GapAnalysis category."""
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"

        with patch.object(ga_dataset, "_resolve_and_validate") as mock_resolve:
            with patch.object(dc, "_create_job", return_value=mock_job):
                dc.analyze(ga_dataset, wait=True, logs=False)

        mock_resolve.assert_called_once_with(sagemaker_session=mock_session)
