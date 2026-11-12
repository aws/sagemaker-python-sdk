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
"""Unit tests for sagemaker.ai_registry.data_preparer.auto_labeling.summary."""

from __future__ import absolute_import

import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer._job_resource import Job
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerError
from sagemaker.ai_registry.data_preparer.auto_labeling.summary import (
    LabelingSummary,
    _newest_labeling_entry,
)

DATASET_ARN = (
    "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/DataSet/tickets-labeled/3.0.0"
)
INDEX_URI = "s3://out-bucket/tickets-sdg-2026-10-26/labeling/index.json"
SUMMARY_URI = "s3://out-bucket/tickets-sdg-2026-10-26/labeling/labeling_summary.json"

ARTIFACTS = {
    "LabelingSummary": SUMMARY_URI,
    "Labels": "s3://out-bucket/tickets-sdg-2026-10-26/output/labels.jsonl",
    "ReviewRows": "s3://out-bucket/tickets-sdg-2026-10-26/review/review_rows.jsonl",
    "Failures": None,
}
INDEX = {
    "CreatedAt": "2026-10-26T12:00:00Z",
    "Status": "Certified",
    "Artifacts": ARTIFACTS,
}
SUMMARY = {
    "Status": "Certified",
    # The container writes a code here, not prose.
    "StatusReason": "spot_check_pending",
    "Type": "labeling.classification",
    "RowsLabeled": 1000,
    "RowsAccepted": 940,
    "RowsRouted": 60,
    "RowsFailed": 0,
    "RowsMissing": 0,
    "HeldOutAccuracy": 0.93,
    "HeldOutLowerBound95": 0.905,
    "AcceptedHeldOutAccuracy": 0.96,
    "AcceptedLowerBound95": 0.94,
    "Target": 0.9,
    "AcceptThreshold": 0.82,
    "ReviewDatasetName": "tickets-review",
    "Warnings": ["12 rows exceeded the token budget and were truncated."],
    # Keys the model does not declare, kept as extras.
    "SpecSha256": "ab12cd34",
    "JurorModelIds": ["juror-one", "juror-two"],
    "HeldOutRows": 200,
    "RowsHumanLabeled": 60,
    "SpotCheckRows": 29,
    "GradedRows": 0,
    # The container writes both as null: Hub mints the ARNs after it exits.
    "LabeledDatasetArn": None,
    "ReviewDatasetArn": None,
}

REVIEW_ARN = (
    "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/DataSet/tickets-review/1.0.0"
)


def _labeling_entry(created_at="2026-10-26T12:00:00Z", uri=INDEX_URI):
    return {
        "MetadataS3Uri": uri,
        "CreatedAt": created_at,
        "JobName": "tickets-sdg",
        "JobType": "SyntheticDataGeneration",
    }


def _fake_session(files):
    """A session whose ``read_s3_file`` serves *files*, a map of S3 URI to JSON object."""
    session = MagicMock()
    session.boto_region_name = "us-west-2"

    def read_s3_file(bucket, key):
        uri = f"s3://{bucket}/{key}"
        if uri not in files:
            raise AssertionError(f"unexpected read of {uri}")
        return json.dumps(files[uri])

    session.read_s3_file.side_effect = read_s3_file
    return session


def _describe_response(document, keywords=None):
    response = {"HubContentDocument": json.dumps(document)}
    if keywords is not None:
        response["HubContentSearchKeywords"] = keywords
    return response


class TestFromDataset:
    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_happy_path(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [_labeling_entry()]}
        )
        session = _fake_session({INDEX_URI: INDEX, SUMMARY_URI: SUMMARY})

        summary = LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)

        assert summary.status == "Certified"
        assert summary.status_reason == "spot_check_pending"
        assert summary.type == "labeling.classification"
        assert summary.rows_labeled == 1000
        assert summary.rows_accepted == 940
        assert summary.rows_routed == 60
        assert summary.held_out_accuracy == 0.93
        assert summary.held_out_lower_bound95 == 0.905
        assert summary.accepted_held_out_accuracy == 0.96
        assert summary.accepted_lower_bound95 == 0.94
        assert summary.target == 0.9
        assert summary.accept_threshold == 0.82
        assert summary.review_dataset_name == "tickets-review"
        assert summary.warnings == ["12 rows exceeded the token budget and were truncated."]
        # The index's artifact map comes along, so the labels are one hop away.
        assert summary.artifacts == ARTIFACTS

        # The DataSet is looked up by the name, version and hub its ARN names.
        args, kwargs = mock_hub.describe_hub_content.call_args
        assert args[:3] == ("DataSet", "tickets-labeled", "3.0.0")
        assert kwargs["hub_name"] == "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub"
        assert kwargs["session"] is session

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_relative_paths_resolve_from_where_the_index_sits(self, mock_hub):
        # The service can place the job's output one folder level deeper than the
        # container's absolute Artifacts URIs say; RelativePaths, resolved against the
        # index's own URI, find the files where they are.
        root = "s3://out-bucket/tickets/synthetic-data-generation/tickets-sdg/2026-11-10T18Z"
        index_uri = f"{root}/labeling/index.json"
        summary_uri = f"{root}/labeling/labeling_summary.json"
        index = {
            "Status": "Certified",
            "Artifacts": {
                "LabelingSummary": "s3://out-bucket/tickets/labeling/labeling_summary.json",
                "Labels": "s3://out-bucket/tickets/output/labels.jsonl",
                "Extra": "s3://elsewhere/extra.json",
            },
            "RelativePaths": {
                "LabelingSummary": "labeling/labeling_summary.json",
                "Labels": "output/labels.jsonl",
            },
        }
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [_labeling_entry(uri=index_uri)]}
        )
        session = _fake_session({index_uri: index, summary_uri: SUMMARY})

        summary = LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)

        assert summary.artifacts == {
            "LabelingSummary": summary_uri,
            "Labels": f"{root}/output/labels.jsonl",
            "Extra": "s3://elsewhere/extra.json",  # no relative path: kept as written
        }
        assert summary.rows_labeled == 1000

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_undeclared_keys_are_kept_as_extras(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [_labeling_entry()]}
        )
        session = _fake_session({INDEX_URI: INDEX, SUMMARY_URI: SUMMARY})

        summary = LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)

        assert summary.GradedRows == 0

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_dataset_arns_are_filled_in_over_the_documents_nulls(self, mock_hub):
        """Hub mints both ARNs after the container exits, so the summary has nulls."""
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [_labeling_entry()]},
            keywords=["auto-labeled", f"review-dataset:{REVIEW_ARN}", "spec-sha256:ab12cd34"],
        )
        session = _fake_session({INDEX_URI: INDEX, SUMMARY_URI: SUMMARY})

        summary = LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)

        assert summary.labeled_dataset_arn == DATASET_ARN
        assert summary.review_dataset_arn == REVIEW_ARN

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_review_arn_is_none_for_a_job_that_does_not_route(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [_labeling_entry()]}, keywords=["auto-labeled"]
        )
        session = _fake_session({INDEX_URI: INDEX, SUMMARY_URI: SUMMARY})

        summary = LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)

        assert summary.review_dataset_arn is None
        assert summary.labeled_dataset_arn == DATASET_ARN

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_typed_fields_cover_the_keys_callers_reach_for(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [_labeling_entry()]}
        )
        session = _fake_session({INDEX_URI: INDEX, SUMMARY_URI: SUMMARY})

        summary = LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)

        assert summary.spec_sha256 == "ab12cd34"
        assert summary.juror_model_ids == ["juror-one", "juror-two"]
        assert summary.held_out_rows == 200
        assert summary.rows_human_labeled == 60
        assert summary.spot_check_rows == 29

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_accepts_a_dataset_object(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [_labeling_entry()]}
        )
        session = _fake_session({INDEX_URI: INDEX, SUMMARY_URI: SUMMARY})
        dataset = MagicMock(arn=DATASET_ARN, sagemaker_session=session)

        summary = LabelingSummary.from_dataset(dataset)

        assert summary.status == "Certified"
        assert mock_hub.describe_hub_content.call_args.kwargs["session"] is session

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_newest_labeling_entry_wins(self, mock_hub):
        stale_uri = "s3://out-bucket/tickets-sdg-2026-01-01/labeling/index.json"
        mock_hub.describe_hub_content.return_value = _describe_response(
            {
                "JobMetadata": [
                    _labeling_entry(created_at="2026-01-01T00:00:00Z", uri=stale_uri),
                    _labeling_entry(created_at="2026-10-26T12:00:00Z"),
                    # A different job type on the same DataSet is not a labeling run.
                    {
                        "MetadataS3Uri": "s3://out-bucket/dq/index.json",
                        "CreatedAt": "2026-12-31T00:00:00Z",
                        "JobName": "tickets-dq",
                        "JobType": "DataQualityEvaluation",
                    },
                ]
            }
        )
        session = _fake_session({INDEX_URI: INDEX, SUMMARY_URI: SUMMARY})

        summary = LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)

        assert summary.status == "Certified"
        # The stale and the DQ entries were never read.
        read_uris = [
            f"s3://{call.args[0]}/{call.args[1]}" for call in session.read_s3_file.call_args_list
        ]
        assert read_uris == [INDEX_URI, SUMMARY_URI]

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_entry_without_a_metadata_uri_is_skipped(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [{"JobType": "SyntheticDataGeneration", "JobName": "tickets-sdg"}]}
        )
        session = _fake_session({})

        with pytest.raises(DataPreparerError, match="not produced by an auto-labeling job"):
            LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_no_job_metadata_raises(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"DatasetName": "tickets-labeled"}
        )
        session = _fake_session({})

        with pytest.raises(DataPreparerError, match="not produced by an auto-labeling job"):
            LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_index_without_a_summary_artifact_raises(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [_labeling_entry()]}
        )
        session = _fake_session({INDEX_URI: {"Status": "Failed", "Artifacts": {"Labels": None}}})

        with pytest.raises(DataPreparerError, match="names no LabelingSummary artifact"):
            LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_malformed_metadata_uri_raises(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [_labeling_entry(uri="/out-bucket/labeling/index.json")]}
        )
        session = _fake_session({})

        with pytest.raises(DataPreparerError, match="Expected an s3:// URI"):
            LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)

    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_s3_uri_without_a_key_raises(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [_labeling_entry(uri="s3://out-bucket")]}
        )
        session = _fake_session({})

        with pytest.raises(DataPreparerError, match="Malformed S3 URI"):
            LabelingSummary.from_dataset(DATASET_ARN, sagemaker_session=session)


class TestJobLabelingSummary:
    @patch("sagemaker.ai_registry.data_preparer.auto_labeling.summary.AIRHub")
    def test_job_reads_the_summary_through_its_dataset(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response(
            {"JobMetadata": [_labeling_entry()]}
        )
        session = _fake_session({INDEX_URI: INDEX, SUMMARY_URI: SUMMARY})
        job = Job(
            job_name="tickets-sdg",
            job_category="SyntheticDataGeneration",
            session=session,
            job_config_document=json.dumps({"DatasetArn": DATASET_ARN}),
        )

        assert job.labeling_summary.status == "Certified"

    def test_job_without_a_dataset_arn_raises(self):
        session = _fake_session({})
        job = Job(
            job_name="tickets-sdg",
            job_category="SyntheticDataGeneration",
            session=session,
            job_config_document=json.dumps({"NumSamples": 10}),
        )

        with pytest.raises(DataPreparerError, match="has no DataSet yet"):
            job.labeling_summary


class TestCreatedAtOrdering:
    """CreatedAt is parsed, so ordering survives a change of timestamp form."""

    def test_a_non_utc_offset_orders_by_instant(self):
        """A non-zero offset shifts the instant, which text ordering cannot see."""
        entries = [
            # 07:00 UTC, despite reading "12:00".
            _labeling_entry(created_at="2026-10-26T12:00:00+05:00", uri="s3://b/earlier.json"),
            _labeling_entry(created_at="2026-10-26T08:00:00Z", uri="s3://b/later.json"),
        ]

        newest = _newest_labeling_entry({"JobMetadata": entries})

        # As text "2026-10-26T12:00:00+05:00" sorts above "2026-10-26T08:00:00Z",
        # so a string comparison returns the earlier run here.
        assert newest["MetadataS3Uri"] == "s3://b/later.json"

    def test_fractional_seconds_order_after_a_bare_second(self):
        entries = [
            _labeling_entry(created_at="2026-10-26T12:00:00Z", uri="s3://b/earlier.json"),
            _labeling_entry(created_at="2026-10-26T12:00:00.100000Z", uri="s3://b/later.json"),
        ]

        newest = _newest_labeling_entry({"JobMetadata": entries})

        # As text "Z" sorts above ".", so a string comparison returns the bare
        # second — the earlier of the two — instead.
        assert newest["MetadataS3Uri"] == "s3://b/later.json"

    def test_naive_timestamp_is_read_as_utc(self):
        entries = [
            _labeling_entry(created_at="2026-10-26T12:00:00", uri="s3://b/naive.json"),
            _labeling_entry(created_at="2026-10-26T11:00:00Z", uri="s3://b/aware.json"),
        ]

        newest = _newest_labeling_entry({"JobMetadata": entries})

        assert newest["MetadataS3Uri"] == "s3://b/naive.json"

    @pytest.mark.parametrize("created_at", [None, "", "not-a-timestamp", 1761566400])
    def test_unusable_timestamp_sorts_oldest(self, created_at):
        """An entry with no readable CreatedAt never beats one that has it."""
        good = _labeling_entry(created_at="2020-01-01T00:00:00Z", uri="s3://b/good.json")
        bad = {
            "MetadataS3Uri": "s3://b/bad.json",
            "CreatedAt": created_at,
            "JobName": "tickets-sdg",
            "JobType": "SyntheticDataGeneration",
        }

        assert _newest_labeling_entry({"JobMetadata": [bad, good]}) is good
        assert _newest_labeling_entry({"JobMetadata": [good, bad]}) is good

    def test_a_lone_entry_without_a_timestamp_is_still_returned(self):
        entry = {
            "MetadataS3Uri": "s3://b/only.json",
            "JobName": "tickets-sdg",
            "JobType": "SyntheticDataGeneration",
        }

        assert _newest_labeling_entry({"JobMetadata": [entry]}) is entry
