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
"""Unit tests for DataPreparer.transform() — DataTransformation."""
from __future__ import absolute_import

import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer.preparer import DataPreparer, _DualCallable
from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig
from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerJobError
from sagemaker.core.helper.session_helper import Session


@pytest.fixture
def mock_session():
    session = MagicMock(spec=Session)
    session.boto_region_name = "us-west-2"
    session.account_id.return_value = "123456789012"
    return session


@pytest.fixture
def dt_dataset():
    """A DatasetJobConfig pre-resolved for DataTransformation."""
    payload = {
        "InputDataConfig": [
            {"DatasetArn": "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/in-ds/1.0.0"}
        ],
        "OutputDataConfig": {"S3OutputPath": "s3://bucket/dt-output/"},
        # Operator Config keys must match what the operator itself expects. The
        # Hub-published DataTransformation schema types Config as
        # ``Dict[str, Any]`` with additionalProperties allowed, and Operator as a
        # bare string with no enum — so neither the SDK nor service-side schema
        # validation rejects a misspelled key or operator name. Nothing catches a
        # mistake here until the operator runs, which is why these are spelled
        # out from the real FuzzyDedup contract rather than invented.
        "TransformationConfig": {
            "Operations": [
                {
                    "Operator": "FuzzyDedup",
                    "Config": {
                        "TextField": "text",
                        "JaccardThreshold": 0.8,
                        "NumPerm": 128,
                        "NgramSize": 5,
                    },
                }
            ]
        },
    }
    cfg = DatasetJobConfig(job_category=JobCategory.DATA_TRANSFORMATION)
    cfg._resolved_schema_version = "1.0.0"
    cfg._resolved_schema = {"properties": {}}
    cfg._dynamic_model = MagicMock()
    cfg._validated_payload = payload
    cfg._serialized_document = json.dumps(payload)
    return cfg


class TestTransformCategory:
    def test_transform_uses_correct_category(self, mock_session, dt_dataset):
        """transform() passes DATA_TRANSFORMATION to _run."""
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")
        captured = {}

        def recording_run(self, dataset, category, **kwargs):
            captured["category"] = category
            return MagicMock()

        with patch.object(DataPreparer, "_run", recording_run):
            dc.transform(dt_dataset)

        assert captured["category"] == JobCategory.DATA_TRANSFORMATION


class TestTransformDualCallable:
    def test_transform_is_dual_callable(self):
        """DataPreparer.transform is a _DualCallable descriptor."""
        assert isinstance(DataPreparer.__dict__["transform"], _DualCallable)

    def test_transform_classmethod_shape(self, mock_session, dt_dataset):
        """Class-level call builds transient instance and delegates."""
        captured = {}

        def recording_run(self, dataset, category, **kwargs):
            captured["self"] = self
            captured["category"] = category
            return MagicMock()

        with patch.object(DataPreparer, "_run", recording_run):
            DataPreparer.transform(
                dt_dataset,
                sagemaker_session=mock_session,
                role="arn:role:class",
            )

        inst = captured["self"]
        assert isinstance(inst, DataPreparer)
        assert inst.role == "arn:role:class"
        assert captured["category"] == JobCategory.DATA_TRANSFORMATION


class TestTransformSchemaResolution:
    def test_transform_schema_resolves(self, mock_session, dt_dataset):
        """_resolve_and_validate is called for DT category."""
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_config_document = json.dumps(
            {"OutputDatasetArns": ["arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/out-ds/1.0.0"]}
        )

        with patch.object(dt_dataset, "_resolve_and_validate") as mock_resolve:
            with patch.object(dc, "_create_job", return_value=mock_job):
                with patch.object(dc, "_rehydrate_datasets", return_value=[MagicMock()]):
                    dc.transform(dt_dataset, wait=True, logs=False)

        mock_resolve.assert_called_once_with(sagemaker_session=mock_session)


class TestTransformReturnsList:
    """transform() returns List[DataSet] — the DT wire contract carries a
    list of OutputDatasetArns and callers must handle >1 output (e.g. Split)."""

    def test_transform_returns_list_of_datasets(self, mock_session, dt_dataset):
        """Single-output DT (FuzzyDedup): result is a length-1 list of DataSet."""
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/dt/my-job"
        mock_job.job_config_document = json.dumps(
            {"OutputDatasetArns": ["arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/out-ds/1.0.0"]}
        )

        mock_ds = MagicMock()
        with patch.object(dc, "_create_job", return_value=mock_job):
            with patch.object(dc, "_rehydrate_datasets", return_value=[mock_ds]) as mock_rehydrate:
                result = dc.transform(dt_dataset, wait=True, logs=False)

        assert result == [mock_ds]
        mock_rehydrate.assert_called_once_with(mock_job, JobCategory.DATA_TRANSFORMATION)

    def test_transform_returns_all_datasets_for_split(self, mock_session, dt_dataset):
        """Split operator produces multiple outputs — the caller sees all of them."""
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/dt/split-job"
        mock_job.job_config_document = json.dumps({
            "OutputDatasetArns": [
                "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/train/1.0.0",
                "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/val/1.0.0",
            ]
        })

        mock_train, mock_val = MagicMock(), MagicMock()
        with patch.object(dc, "_create_job", return_value=mock_job):
            with patch.object(dc, "_rehydrate_datasets", return_value=[mock_train, mock_val]):
                result = dc.transform(dt_dataset, wait=True, logs=False)

        assert result == [mock_train, mock_val]

    def test_transform_stamps_job_arn_on_every_output(self, mock_session, dt_dataset):
        """Every returned DataSet carries the job's ARN for telemetry chaining."""
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/dt/split-job"
        mock_job.job_config_document = json.dumps({
            "OutputDatasetArns": ["arn:.../DataSet/a/1.0.0", "arn:.../DataSet/b/1.0.0"]
        })

        mock_a, mock_b = MagicMock(), MagicMock()
        with patch.object(dc, "_create_job", return_value=mock_job):
            with patch.object(dc, "_rehydrate_datasets", return_value=[mock_a, mock_b]):
                dc.transform(dt_dataset, wait=True, logs=False)

        assert mock_a.job_arn == "arn:aws:sagemaker:us-west-2:123:job/dt/split-job"
        assert mock_b.job_arn == "arn:aws:sagemaker:us-west-2:123:job/dt/split-job"


class TestRehydrateDatasetsForDT:
    """_rehydrate_datasets returns the full list of OutputDatasetArns for DT."""

    def test_rehydrate_returns_dataset_per_output_arn(self, mock_session):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_config_document = json.dumps({
            "OutputDatasetArns": [
                "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/train/1.0.0",
                "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/val/1.0.0",
            ]
        })

        with patch("sagemaker.ai_registry.dataset.DataSet.get") as mock_get:
            mock_get.side_effect = lambda name, sagemaker_session, hub_name, version: (
                MagicMock(_name=name)
            )
            result = dc._rehydrate_datasets(mock_job, JobCategory.DATA_TRANSFORMATION)

        assert len(result) == 2
        assert [r._name for r in result] == ["train", "val"]

    def test_rehydrate_resolves_hub_and_version_from_arn(self, mock_session):
        """The output ARN's hub and version must be honored, not re-derived.

        A job publishes into the caller's hub, which need not be the derived AI
        Registry hub, and publishes one specific version. Looking the content up
        by bare name raised ResourceNotFound for any other hub, and otherwise
        returned whichever version was latest at call time.
        """
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_config_document = json.dumps({
            "OutputDatasetArns": [
                "arn:aws:sagemaker:us-west-2:123:hub-content/DTBugBashHub/DataSet/out/0.0.2",
            ]
        })

        with patch("sagemaker.ai_registry.dataset.DataSet.get") as mock_get:
            dc._rehydrate_datasets(mock_job, JobCategory.DATA_TRANSFORMATION)

        mock_get.assert_called_once_with(
            name="out",
            sagemaker_session=mock_session,
            hub_name="DTBugBashHub",
            version="0.0.2",
        )

    @pytest.mark.parametrize(
        "arn",
        [
            "some-name/1.0.0",                                   # too few segments
            "arn:aws:sagemaker:us-west-2:123:hub/H/DataSet/n/1",  # not a hub-content ARN
            "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/n/1/extra",  # too many
        ],
    )
    def test_rehydrate_falls_back_when_arn_shape_unexpected(self, mock_session, arn):
        """An unexpected ARN shape must fall back, not mis-read a segment as the hub.

        Passing ``hub_name=None`` restores the previous behavior exactly: the
        derived AI Registry hub and the latest version.
        """
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_config_document = json.dumps({"OutputDatasetArns": [arn]})

        with patch("sagemaker.ai_registry.dataset.DataSet.get") as mock_get:
            dc._rehydrate_datasets(mock_job, JobCategory.DATA_TRANSFORMATION)

        assert mock_get.call_args.kwargs["hub_name"] is None
        assert mock_get.call_args.kwargs["version"] is None


    def test_rehydrate_raises_when_output_dataset_arns_missing(self, mock_session):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_config_document = json.dumps({})  # no OutputDatasetArns
        mock_job.job_arn = "arn:job"
        mock_job.job_name = "my-job"

        with pytest.raises(DataPreparerJobError) as exc_info:
            dc._rehydrate_datasets(mock_job, JobCategory.DATA_TRANSFORMATION)

        assert "OutputDatasetArns" in str(exc_info.value)

    def test_rehydrate_raises_when_output_dataset_arns_empty(self, mock_session):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_config_document = json.dumps({"OutputDatasetArns": []})
        mock_job.job_arn = "arn:job"
        mock_job.job_name = "my-job"

        with pytest.raises(DataPreparerJobError) as exc_info:
            dc._rehydrate_datasets(mock_job, JobCategory.DATA_TRANSFORMATION)

        assert "OutputDatasetArns" in str(exc_info.value)


class TestRehydrateDatasetsForSdgDq:
    """SDG/DQ still return a length-1 list via the scalar DatasetArn path."""

    def test_sdg_returns_single_dataset_list(self, mock_session):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_config_document = json.dumps(
            {"DatasetArn": "arn:aws:sagemaker:us-west-2:123:hub-content/H/DataSet/sdg-out/1.0.0"}
        )

        with patch("sagemaker.ai_registry.dataset.DataSet.get") as mock_get:
            mock_get.return_value = MagicMock()
            result = dc._rehydrate_datasets(mock_job, JobCategory.SYNTHETIC_DATA_GENERATION)

        assert len(result) == 1
        # Content name parsed from ARN path is the second-to-last segment.
        assert mock_get.call_args.kwargs["name"] == "sdg-out"

    def test_dq_raises_when_dataset_arn_missing(self, mock_session):
        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")

        mock_job = MagicMock()
        mock_job.job_config_document = json.dumps({})
        mock_job.job_arn = "arn:job"
        mock_job.job_name = "my-job"

        with pytest.raises(DataPreparerJobError) as exc_info:
            dc._rehydrate_datasets(mock_job, JobCategory.DATA_QUALITY_EVALUATION)

        assert "DatasetArn" in str(exc_info.value)


class TestTransformCreateJobDocumentShape:
    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    def test_transform_submits_document_verbatim(self, mock_get_job, mock_session, dt_dataset):
        """The resolved JobConfigDocument reaches Job.create unchanged for DT.

        The SDK is a thin passthrough — the JobConfigDocument is customer-supplied
        and validated against the Hub-published DataTransformation schema. Ensure
        transform() doesn't mutate the document en route.
        """
        mock_job_cls = MagicMock()
        mock_get_job.return_value = mock_job_cls
        mock_job_cls.create.return_value = MagicMock()

        dc = DataPreparer(sagemaker_session=mock_session, role="arn:role")
        dc._create_job(dt_dataset, JobCategory.DATA_TRANSFORMATION)

        call_kwargs = mock_job_cls.create.call_args[1]
        assert call_kwargs["job_category"] == "DataTransformation"
        submitted = json.loads(call_kwargs["job_config_document"])
        assert submitted == dt_dataset._validated_payload
