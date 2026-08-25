# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Integration tests for BatchWriteRecord via ingest_dataframe(use_batch_write_record=True)."""
import time

import pytest
import pandas as pd

from sagemaker.core.utils import unique_name_from_base
from sagemaker.core.resources import FeatureGroup
from sagemaker.mlops.feature_store import OnlineStoreConfig
from sagemaker.mlops.feature_store.feature_utils import (
    ingest_dataframe,
    load_feature_definitions_from_dataframe,
)
from sagemaker.mlops.feature_store.ingestion_manager_pandas import IngestionError


@pytest.fixture(scope="module")
def feature_group_name():
    return unique_name_from_base("integ-test-fg")


@pytest.fixture(scope="module")
def sample_dataframe():
    """Create sample DataFrame matching the FG schema."""
    current_time = int(time.time())
    return pd.DataFrame(
        {
            "RecordIdentifier": [f"id-{i}" for i in range(5)],
            "EventTime": [float(current_time + i) for i in range(5)],
            "Feature1": [f"val-{i}" for i in range(5)],
        }
    )


@pytest.fixture(scope="module")
def feature_group(feature_group_name, sample_dataframe, role):
    """Create a FG with online store, wait for it, yield it, and tear down."""
    fg = None
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)

        fg = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="RecordIdentifier",
            event_time_feature_name="EventTime",
            feature_definitions=feature_definitions,
            role_arn=role,
            online_store_config=OnlineStoreConfig(enable_online_store=True),
        )

        fg.wait_for_status("Created")

        yield fg
    finally:
        if fg is not None:
            try:
                FeatureGroup.get(feature_group_name=feature_group_name).delete()
            except Exception:
                pass


@pytest.fixture
def timestamp():
    return float(time.time())


class TestBatchWriteRecordIntegration:
    """Integration tests calling the real BatchWriteRecord API."""

    def test_batch_write_basic(self, feature_group, feature_group_name, timestamp):
        """Write 5 records via BatchWriteRecord, verify no failures."""
        df = pd.DataFrame(
            {
                "RecordIdentifier": [f"integ-bwr-{i}-{int(time.time())}" for i in range(5)],
                "EventTime": [timestamp] * 5,
                "Feature1": [f"val-{i}" for i in range(5)],
            }
        )

        mgr = ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=df,
            max_workers=1,
            max_processes=1,
            use_batch_write_record=True,
        )
        assert mgr.failed_rows == []

    def test_batch_write_25_boundary(self, feature_group, feature_group_name, timestamp):
        """Exactly 25 records — single batch boundary."""
        df = pd.DataFrame(
            {
                "RecordIdentifier": [
                    f"integ-b25-{i}-{int(time.time())}" for i in range(25)
                ],
                "EventTime": [timestamp] * 25,
                "Feature1": [f"val-{i}" for i in range(25)],
            }
        )

        mgr = ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=df,
            max_workers=1,
            max_processes=1,
            use_batch_write_record=True,
        )
        assert mgr.failed_rows == []

    def test_batch_write_26_two_batches(self, feature_group, feature_group_name, timestamp):
        """26 records — splits into 2 batches (25+1)."""
        df = pd.DataFrame(
            {
                "RecordIdentifier": [
                    f"integ-b26-{i}-{int(time.time())}" for i in range(26)
                ],
                "EventTime": [timestamp] * 26,
                "Feature1": [f"val-{i}" for i in range(26)],
            }
        )

        mgr = ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=df,
            max_workers=1,
            max_processes=1,
            use_batch_write_record=True,
        )
        assert mgr.failed_rows == []

    def test_batch_write_verify_with_get_record(
        self, feature_group, feature_group_name, timestamp
    ):
        """Write via BatchWriteRecord, verify with GetRecord."""
        rid = f"integ-bwr-verify-{int(time.time())}"
        df = pd.DataFrame(
            {
                "RecordIdentifier": [rid],
                "EventTime": [timestamp],
                "Feature1": ["verify-value"],
            }
        )

        mgr = ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=df,
            max_workers=1,
            max_processes=1,
            use_batch_write_record=True,
        )
        assert mgr.failed_rows == []

        time.sleep(2)
        record = feature_group.get_record(record_identifier_value_as_string=rid)
        assert record is not None
        val = next(
            fv.value_as_string for fv in record.record if fv.feature_name == "Feature1"
        )
        assert val == "verify-value"

    def test_batch_write_null_skipped(self, feature_group, feature_group_name, timestamp):
        """None values excluded from record, record still written."""
        rid = f"integ-bwr-null-{int(time.time())}"
        df = pd.DataFrame(
            {
                "RecordIdentifier": [rid],
                "EventTime": [timestamp],
                "Feature1": [None],
            }
        )

        mgr = ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=df,
            max_workers=1,
            max_processes=1,
            use_batch_write_record=True,
        )
        assert mgr.failed_rows == []

        time.sleep(2)
        record = feature_group.get_record(record_identifier_value_as_string=rid)
        assert record is not None
        names = [fv.feature_name for fv in record.record]
        assert "Feature1" not in names

    def test_batch_write_partial_failure(
        self, feature_group, feature_group_name, timestamp
    ):
        """10 records, row 5 missing RecordIdentifier — only row 5 fails."""
        records = []
        for i in range(10):
            if i == 5:
                records.append(
                    {"RecordIdentifier": None, "EventTime": timestamp, "Feature1": f"v-{i}"}
                )
            else:
                records.append(
                    {
                        "RecordIdentifier": f"integ-partial-{i}-{int(time.time())}",
                        "EventTime": timestamp,
                        "Feature1": f"v-{i}",
                    }
                )
        df = pd.DataFrame(records)

        try:
            mgr = ingest_dataframe(
                feature_group_name=feature_group_name,
                data_frame=df,
                max_workers=1,
                max_processes=1,
                use_batch_write_record=True,
            )
            failed = mgr.failed_rows
        except IngestionError as e:
            failed = e.failed_rows

        assert 5 in failed
        # Valid rows should have been written
        time.sleep(2)
        rec = feature_group.get_record(
            record_identifier_value_as_string=records[0]["RecordIdentifier"]
        )
        assert rec is not None

    def test_putrecord_same_result(self, feature_group, feature_group_name, timestamp):
        """PutRecord path produces same outcome for comparison."""
        df = pd.DataFrame(
            {
                "RecordIdentifier": [
                    f"integ-put-{i}-{int(time.time())}" for i in range(5)
                ],
                "EventTime": [timestamp] * 5,
                "Feature1": [f"val-{i}" for i in range(5)],
            }
        )

        mgr = ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=df,
            max_workers=1,
            max_processes=1,
            use_batch_write_record=False,
        )
        assert mgr.failed_rows == []

    def test_putrecord_partial_failure_same(
        self, feature_group, feature_group_name, timestamp
    ):
        """PutRecord partial failure — same row 5 fails."""
        records = []
        for i in range(10):
            if i == 5:
                records.append(
                    {"RecordIdentifier": None, "EventTime": timestamp, "Feature1": f"v-{i}"}
                )
            else:
                records.append(
                    {
                        "RecordIdentifier": f"integ-put-partial-{i}-{int(time.time())}",
                        "EventTime": timestamp,
                        "Feature1": f"v-{i}",
                    }
                )
        df = pd.DataFrame(records)

        try:
            mgr = ingest_dataframe(
                feature_group_name=feature_group_name,
                data_frame=df,
                max_workers=1,
                max_processes=1,
                use_batch_write_record=False,
            )
            failed = mgr.failed_rows
        except IngestionError as e:
            failed = e.failed_rows

        assert 5 in failed

    def test_batch_write_default_flag_uses_putrecord(
        self, feature_group, feature_group_name, timestamp
    ):
        """No flag → defaults to PutRecord."""
        df = pd.DataFrame(
            {
                "RecordIdentifier": [f"integ-default-{int(time.time())}"],
                "EventTime": [timestamp],
                "Feature1": ["default-test"],
            }
        )

        mgr = ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=df,
            max_workers=1,
            max_processes=1,
        )
        assert mgr.use_batch_write_record is False
        assert mgr.failed_rows == []

    def test_batch_write_invalid_fg(self, timestamp):
        """Invalid feature group fails for both BWR and PutRecord."""
        df = pd.DataFrame(
            {
                "RecordIdentifier": ["x"],
                "EventTime": [timestamp],
                "Feature1": ["v"],
            }
        )

        with pytest.raises(Exception):
            ingest_dataframe(
                feature_group_name="non-existent-fg-xyz",
                data_frame=df,
                max_workers=1,
                max_processes=1,
                use_batch_write_record=True,
            )
