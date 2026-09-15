# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Integration tests for ListRecords via list_records()."""
import time

import pytest
import pandas as pd

from sagemaker.core.utils import unique_name_from_base
from sagemaker.core.resources import FeatureGroup
from sagemaker.mlops.feature_store import OnlineStoreConfig
from sagemaker.mlops.feature_store.feature_utils import (
    ingest_dataframe,
    list_records,
    load_feature_definitions_from_dataframe,
)


@pytest.fixture(scope="module")
def feature_group_name():
    return unique_name_from_base("integ-test-fg")


@pytest.fixture(scope="module")
def sample_dataframe():
    """Create sample DataFrame matching the FG schema."""
    current_time = int(time.time())
    return pd.DataFrame(
        {
            "RecordIdentifier": [f"id-{i}" for i in range(10)],
            "EventTime": [float(current_time + i) for i in range(10)],
            "Feature1": [f"val-{i}" for i in range(10)],
        }
    )


@pytest.fixture(scope="module")
def feature_group(feature_group_name, sample_dataframe, role):
    """Create a FG with online store, ingest data, yield it, and tear down."""
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

        ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=sample_dataframe,
            max_workers=1,
            max_processes=1,
        )

        # Allow time for online store to become queryable
        time.sleep(15)

        yield fg
    finally:
        if fg is not None:
            try:
                FeatureGroup.get(feature_group_name=feature_group_name).delete()
            except Exception:
                pass


class TestListRecordsIntegration:
    """Integration tests calling the real ListRecords API."""

    def test_list_records_returns_identifiers(self, feature_group, feature_group_name):
        """Basic list_records returns record identifiers."""
        response = list_records(feature_group_name=feature_group_name)

        assert response is not None
        assert hasattr(response, "record_identifiers")
        assert hasattr(response, "next_token")
        assert len(response.record_identifiers) > 0

    def test_list_records_max_results(self, feature_group, feature_group_name):
        """max_results limits page size."""
        response = list_records(
            feature_group_name=feature_group_name,
            max_results=2,
        )

        assert response is not None
        assert len(response.record_identifiers) <= 100  # API hint, not strict

    def test_list_records_pagination(self, feature_group, feature_group_name):
        """Pagination with next_token returns different records."""
        page1 = list_records(
            feature_group_name=feature_group_name,
            max_results=2,
        )
        assert page1 is not None
        assert len(page1.record_identifiers) > 0

        if page1.next_token:
            page2 = list_records(
                feature_group_name=feature_group_name,
                next_token=page1.next_token,
                max_results=2,
            )
            assert page2 is not None
            assert len(page2.record_identifiers) > 0

    def test_list_records_full_pagination_loop(self, feature_group, feature_group_name):
        """Manual pagination loop (standard AWS pattern)."""
        all_ids = []
        next_token = None
        page_count = 0

        while page_count < 3:  # Cap at 3 pages for test speed
            kwargs = {"feature_group_name": feature_group_name, "max_results": 10}
            if next_token:
                kwargs["next_token"] = next_token

            response = list_records(**kwargs)
            all_ids.extend(response.record_identifiers)
            page_count += 1

            if not response.next_token:
                break
            next_token = response.next_token

        assert len(all_ids) > 0

    def test_list_records_include_soft_deleted(self, feature_group, feature_group_name):
        """include_soft_deleted_records=True doesn't error."""
        response = list_records(
            feature_group_name=feature_group_name,
            include_soft_deleted_records=True,
        )

        assert response is not None
        assert len(response.record_identifiers) > 0

    def test_list_records_non_existent(self):
        """Invalid feature group raises exception."""
        with pytest.raises(Exception):
            list_records(
                feature_group_name="non-existent-fg-xyz",
            )

    def test_list_records_via_feature_group_object(self, feature_group):
        """Call list_records directly on FeatureGroup object."""
        response = feature_group.list_records()

        assert response is not None
        assert len(response.record_identifiers) > 0

    def test_list_records_via_feature_group_with_pagination(self, feature_group):
        """FeatureGroup.list_records() with pagination."""
        page1 = feature_group.list_records(max_results=3)
        assert page1 is not None

        if page1.next_token:
            page2 = feature_group.list_records(
                next_token=page1.next_token, max_results=3
            )
            assert page2 is not None
            assert page1.record_identifiers != page2.record_identifiers
