# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Integration tests for ingest_dataframe(enable_write_audit=True)."""

import json
import logging
import time

import pandas as pd
import pytest

from sagemaker.core.resources import FeatureGroup
from sagemaker.core.utils import unique_name_from_base
from sagemaker.mlops.feature_store import OnlineStoreConfig
from sagemaker.mlops.feature_store.feature_utils import (
    ingest_dataframe,
    load_feature_definitions_from_dataframe,
)
from sagemaker.mlops.feature_store.ingestion_manager_pandas import IngestionError
from sagemaker.mlops.feature_store.write_audit import WRITE_AUDIT_LOGGER_NAME


@pytest.fixture(scope="module")
def feature_group_name():
    return unique_name_from_base("integ-test-fg-audit")


@pytest.fixture(scope="module")
def feature_group(feature_group_name, role):
    """Create a FG with an online store, wait for it, yield it, and tear down."""
    fg = None
    schema = pd.DataFrame(
        {"RecordIdentifier": ["id-0"], "EventTime": [float(time.time())], "Feature1": ["v"]}
    )
    try:
        fg = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="RecordIdentifier",
            event_time_feature_name="EventTime",
            feature_definitions=load_feature_definitions_from_dataframe(schema),
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
def audit_events(caplog):
    caplog.set_level(logging.INFO, logger=WRITE_AUDIT_LOGGER_NAME)

    def events():
        return [
            json.loads(r.getMessage()) for r in caplog.records if r.name == WRITE_AUDIT_LOGGER_NAME
        ]

    return events


def _frame(prefix, count, missing_id_at=None):
    now = float(time.time())
    ids = [f"{prefix}-{i}-{int(now)}" for i in range(count)]
    if missing_id_at is not None:
        ids[missing_id_at] = None
    return pd.DataFrame(
        {"RecordIdentifier": ids, "EventTime": [now] * count, "Feature1": ["secret"] * count}
    )


class TestWriteAuditIntegration:
    """Audit lines carry the real request ID of the call that wrote each record."""

    @pytest.mark.parametrize("use_batch_write_record", [False, True])
    def test_every_record_logged_with_request_id(
        self, feature_group, feature_group_name, audit_events, use_batch_write_record
    ):
        df = _frame("integ-audit", 30)

        mgr = ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=df,
            use_batch_write_record=use_batch_write_record,
            enable_write_audit=True,
        )

        assert mgr.failed_rows == []
        accepted = {e["record_id"]: e for e in audit_events() if e["entry_status"] == "accepted"}
        assert set(accepted) == set(df["RecordIdentifier"])
        assert all(e["request_id"] and e["http_status"] == 200 for e in accepted.values())
        assert all(e["event_time"] == str(df["EventTime"][0]) for e in accepted.values())
        expected_requests = 2 if use_batch_write_record else 30  # 25 + 5, or one per record
        assert len({e["request_id"] for e in accepted.values()}) == expected_requests
        assert all("secret" not in json.dumps(e) for e in audit_events())

    def test_rejected_record_is_logged_with_its_request_id(
        self, feature_group, feature_group_name, audit_events
    ):
        df = _frame("integ-audit-fail", 3, missing_id_at=1)

        with pytest.raises(IngestionError):
            ingest_dataframe(
                feature_group_name=feature_group_name,
                data_frame=df,
                enable_write_audit=True,
            )

        rejected = [e for e in audit_events() if e["entry_status"] == "request_failed"]
        assert len(rejected) == 1
        assert rejected[0]["request_id"] and rejected[0]["http_status"] == 400

    def test_disabled_by_default(self, feature_group, feature_group_name, audit_events):
        ingest_dataframe(feature_group_name=feature_group_name, data_frame=_frame("integ-off", 2))
        assert audit_events() == []
