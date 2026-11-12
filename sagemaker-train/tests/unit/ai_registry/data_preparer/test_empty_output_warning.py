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
"""Unit tests for the empty-output warning on completed DataPreparer jobs.

A job can reach ``Completed`` while producing no records (for example a filter
that drops every row). The SDK keeps returning the dataset — zero records can be
legitimate — but logs a warning so the result is not mistaken for real output.
"""

from __future__ import absolute_import

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer import preparer as preparer_module
from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.preparer import DataPreparer

_LOGGER = "sagemaker.ai_registry.data_preparer.preparer"


@pytest.fixture
def preparer():
    session = MagicMock()
    session.boto_region_name = "us-west-2"
    session._get_account_id_if_default_bucket.return_value = None
    return DataPreparer(sagemaker_session=session)


def _dataset(source):
    ds = MagicMock()
    ds.source = source
    return ds


def _job(name="my-job", document=None, schema_version="1.0.0"):
    job = MagicMock()
    job.job_name = name
    job.job_config_document = json.dumps(document or {})
    job.job_config_schema_version = schema_version
    return job


def _s3_with(objects):
    """Mock S3 client serving ``objects``: {key: bytes}. Unknown keys raise."""
    s3 = MagicMock()

    def head_object(Bucket, Key):  # pylint: disable=invalid-name,unused-argument
        if Key not in objects:
            raise RuntimeError("NoSuchKey")
        return {"ContentLength": len(objects[Key])}

    def get_object(Bucket, Key, **kwargs):  # pylint: disable=invalid-name,unused-argument
        body = MagicMock()
        body.read.return_value = objects[Key]
        return {"Body": body}

    s3.head_object.side_effect = head_object
    s3.get_object.side_effect = get_object
    return s3


class TestOutputHasNoRecords:
    @patch("boto3.client")
    def test_zero_byte_object_is_empty_without_download(self, mock_boto3, preparer):
        s3 = _s3_with({"out/data.jsonl": b""})
        mock_boto3.return_value = s3
        assert preparer._output_has_no_records("s3://b/out/data.jsonl") is True
        s3.get_object.assert_not_called()

    @patch("boto3.client")
    def test_blank_lines_only_is_empty(self, mock_boto3, preparer):
        mock_boto3.return_value = _s3_with({"out/data.jsonl": b"\n  \n\r\n"})
        assert preparer._output_has_no_records("s3://b/out/data.jsonl") is True

    @patch("boto3.client")
    def test_records_present_is_not_empty(self, mock_boto3, preparer):
        mock_boto3.return_value = _s3_with({"out/data.jsonl": b'{"a": 1}'})
        assert preparer._output_has_no_records("s3://b/out/data.jsonl") is False

    @patch("boto3.client")
    def test_large_object_is_not_downloaded(self, mock_boto3, preparer):
        s3 = MagicMock()
        s3.head_object.return_value = {"ContentLength": preparer_module._EMPTY_CHECK_MAX_BYTES + 1}
        mock_boto3.return_value = s3
        assert preparer._output_has_no_records("s3://b/out/big.jsonl") is False
        s3.get_object.assert_not_called()

    @patch("boto3.client")
    def test_csv_header_only_is_empty(self, mock_boto3, preparer):
        mock_boto3.return_value = _s3_with({"out/t.csv": b"id,label\n"})
        assert preparer._output_has_no_records("s3://b/out/t.csv") is True

    @patch("boto3.client")
    def test_csv_with_a_row_is_not_empty(self, mock_boto3, preparer):
        mock_boto3.return_value = _s3_with({"out/t.csv": b"id,label\n1,yes\n"})
        assert preparer._output_has_no_records("s3://b/out/t.csv") is False

    @patch("boto3.client")
    def test_error_is_unknown_not_raised(self, mock_boto3, preparer):
        s3 = MagicMock()
        s3.head_object.side_effect = RuntimeError("AccessDenied")
        mock_boto3.return_value = s3
        assert preparer._output_has_no_records("s3://b/out/data.jsonl") is None


class TestResolveOutputRecordsUri:
    def test_sdg_uses_output_samples_under_source(self, preparer):
        ds = _dataset("s3://b/jobs/sdg/output/")
        assert (
            preparer._resolve_output_records_uri(ds, JobCategory.SYNTHETIC_DATA_GENERATION)
            == "s3://b/jobs/sdg/output/output_samples.jsonl"
        )

    def test_dt_uses_the_output_object(self, preparer):
        ds = _dataset("s3://b/dt/my-job/dataset.jsonl")
        assert (
            preparer._resolve_output_records_uri(ds, JobCategory.DATA_TRANSFORMATION)
            == "s3://b/dt/my-job/dataset.jsonl"
        )

    def test_dt_non_text_output_is_not_checked(self, preparer):
        ds = _dataset("s3://b/dt/my-job/dataset.parquet")
        assert preparer._resolve_output_records_uri(ds, JobCategory.DATA_TRANSFORMATION) is None

    @pytest.mark.parametrize("source", [None, "", "/local/path", 42])
    def test_unusable_source_is_none(self, preparer, source):
        ds = _dataset(source)
        assert (
            preparer._resolve_output_records_uri(ds, JobCategory.SYNTHETIC_DATA_GENERATION) is None
        )

    def test_dq_resolves_in_the_folder_of_metadata_s3_uri(self, preparer):
        job = _job(
            name="my-job",
            document={
                "InputDataConfig": [{"DatasetArn": "arn:in"}],
                "OutputDataConfig": {"S3OutputPath": "s3://b/dq-out/"},
                "MetadataS3Uri": "s3://b/dq-out/my-job-2026-10-27T192836.911Z/metadata.json",
            },
            schema_version="2.0.0",
        )
        uri = preparer._resolve_output_records_uri(
            _dataset("s3://b/input/ds.jsonl"), JobCategory.DATA_QUALITY_EVALUATION, job
        )
        assert (
            uri == "s3://b/dq-out/my-job-2026-10-27T192836.911Z/record_level/evaluated_data.jsonl"
        )


class TestWarnIfEmptyOutput:
    @patch("boto3.client")
    def test_warns_for_empty_dt_output(self, mock_boto3, preparer, caplog):
        mock_boto3.return_value = _s3_with({"dt/my-job/dataset.jsonl": b""})
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            preparer._warn_if_empty_output(
                [_dataset("s3://b/dt/my-job/dataset.jsonl")],
                JobCategory.DATA_TRANSFORMATION,
                _job("my-job"),
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "my-job" in message
        assert "no records" in message
        assert "s3://b/dt/my-job/dataset.jsonl" in message

    @patch("boto3.client")
    def test_warns_only_for_the_empty_split_output(self, mock_boto3, preparer, caplog):
        mock_boto3.return_value = _s3_with({"dt/train.jsonl": b'{"a": 1}\n', "dt/test.jsonl": b""})
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            preparer._warn_if_empty_output(
                [_dataset("s3://b/dt/train.jsonl"), _dataset("s3://b/dt/test.jsonl")],
                JobCategory.DATA_TRANSFORMATION,
                _job("split-job"),
            )
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "dt/test.jsonl" in warnings[0]
        assert "dt/train.jsonl" not in warnings[0]

    @patch("boto3.client")
    def test_no_warning_when_output_has_records(self, mock_boto3, preparer, caplog):
        mock_boto3.return_value = _s3_with({"sdg/output/output_samples.jsonl": b'{"q": 1}\n'})
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            preparer._warn_if_empty_output(
                [_dataset("s3://b/sdg/output")],
                JobCategory.SYNTHETIC_DATA_GENERATION,
                _job(),
            )
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    @patch("boto3.client")
    def test_sdg_wording(self, mock_boto3, preparer, caplog):
        mock_boto3.return_value = _s3_with({"sdg/output/output_samples.jsonl": b""})
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            preparer._warn_if_empty_output(
                [_dataset("s3://b/sdg/output")],
                JobCategory.SYNTHETIC_DATA_GENERATION,
                _job("gen"),
            )
        assert any("generated no records" in r.getMessage() for r in caplog.records)

    @patch("boto3.client")
    def test_unknown_emptiness_does_not_warn(self, mock_boto3, preparer, caplog):
        s3 = MagicMock()
        s3.head_object.side_effect = RuntimeError("AccessDenied")
        mock_boto3.return_value = s3
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            preparer._warn_if_empty_output(
                [_dataset("s3://b/dt/x.jsonl")], JobCategory.DATA_TRANSFORMATION, _job()
            )
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_never_raises(self, preparer):
        with patch.object(
            preparer, "_resolve_output_records_uri", side_effect=RuntimeError("boom")
        ):
            preparer._warn_if_empty_output(
                [_dataset("s3://b/dt/x.jsonl")], JobCategory.DATA_TRANSFORMATION, _job()
            )

    def test_other_categories_are_ignored(self, preparer):
        with patch.object(preparer, "_resolve_output_records_uri") as resolve:
            preparer._warn_if_empty_output(
                [_dataset("s3://b/x.jsonl")], "SomeOtherCategory", _job()
            )
        resolve.assert_not_called()


class TestCompletedPathsInvokeWarning:
    def test_run_completed_dt_checks_outputs(self):
        session = MagicMock()
        session.boto_region_name = "us-west-2"
        session.account_id.return_value = "123456789012"
        dc = DataPreparer(sagemaker_session=session, role="arn:aws:iam::123456789012:role/r")
        job = MagicMock()
        job.job_status = "Completed"
        job.job_arn = "arn:aws:sagemaker:us-west-2:123456789012:job/DataTransformation/j"
        outputs = [MagicMock(), MagicMock()]
        config = MagicMock()
        config.job_category = JobCategory.DATA_TRANSFORMATION
        with (
            patch.object(dc, "_create_job", return_value=job),
            patch.object(dc, "_rehydrate_datasets", return_value=outputs),
            patch.object(dc, "_warn_if_empty_output") as warn,
        ):
            result = dc._run(config, JobCategory.DATA_TRANSFORMATION, wait=True, logs=False)
        assert result == outputs
        warn.assert_called_once_with(outputs, JobCategory.DATA_TRANSFORMATION, job)

    @patch("sagemaker.ai_registry.data_preparer.preparer._get_job_resource")
    @patch(
        "sagemaker.ai_registry.data_preparer.preparer.DataPreparerDefaults.get_sagemaker_session"
    )
    def test_attach_completed_checks_outputs(self, mock_get_session, mock_get_job_cls):
        mock_get_session.return_value = MagicMock()
        job = MagicMock()
        job.job_name = "j"
        job.job_status = "Completed"
        job.job_arn = "arn:aws:sagemaker:us-west-2:123456789012:job/DataTransformation/j"
        job_cls = MagicMock()
        job_cls.get.return_value = job
        mock_get_job_cls.return_value = job_cls
        outputs = [MagicMock()]
        with (
            patch.object(DataPreparer, "_rehydrate_datasets", return_value=outputs),
            patch.object(DataPreparer, "_warn_if_empty_output") as warn,
        ):
            result = DataPreparer.attach(job.job_arn)
        assert result == outputs
        warn.assert_called_once_with(outputs, JobCategory.DATA_TRANSFORMATION, job)
