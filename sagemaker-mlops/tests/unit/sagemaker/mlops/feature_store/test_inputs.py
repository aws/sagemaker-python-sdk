# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests for inputs.py (enums)."""
import pytest

from sagemaker.mlops.feature_store.inputs import (
    TargetStoreEnum,
    OnlineStoreStorageTypeEnum,
    TableFormatEnum,
    ResourceEnum,
    SearchOperatorEnum,
    SortOrderEnum,
    FilterOperatorEnum,
    DeletionModeEnum,
    ExpirationTimeResponseEnum,
    ThroughputModeEnum,
)


class TestTargetStoreEnum:
    def test_online_store(self):
        assert TargetStoreEnum.ONLINE_STORE.value == "OnlineStore"

    def test_offline_store(self):
        assert TargetStoreEnum.OFFLINE_STORE.value == "OfflineStore"


class TestOnlineStoreStorageTypeEnum:
    def test_standard(self):
        assert OnlineStoreStorageTypeEnum.STANDARD.value == "Standard"

    def test_in_memory(self):
        assert OnlineStoreStorageTypeEnum.IN_MEMORY.value == "InMemory"


class TestTableFormatEnum:
    def test_glue(self):
        assert TableFormatEnum.GLUE.value == "Glue"

    def test_iceberg(self):
        assert TableFormatEnum.ICEBERG.value == "Iceberg"


class TestResourceEnum:
    def test_feature_group(self):
        assert ResourceEnum.FEATURE_GROUP.value == "FeatureGroup"

    def test_feature_metadata(self):
        assert ResourceEnum.FEATURE_METADATA.value == "FeatureMetadata"


class TestSearchOperatorEnum:
    def test_and(self):
        assert SearchOperatorEnum.AND.value == "And"

    def test_or(self):
        assert SearchOperatorEnum.OR.value == "Or"


class TestSortOrderEnum:
    def test_ascending(self):
        assert SortOrderEnum.ASCENDING.value == "Ascending"

    def test_descending(self):
        assert SortOrderEnum.DESCENDING.value == "Descending"


class TestFilterOperatorEnum:
    def test_equals(self):
        assert FilterOperatorEnum.EQUALS.value == "Equals"

    def test_not_equals(self):
        assert FilterOperatorEnum.NOT_EQUALS.value == "NotEquals"

    def test_greater_than(self):
        assert FilterOperatorEnum.GREATER_THAN.value == "GreaterThan"

    def test_contains(self):
        assert FilterOperatorEnum.CONTAINS.value == "Contains"

    def test_exists(self):
        assert FilterOperatorEnum.EXISTS.value == "Exists"

    def test_in(self):
        assert FilterOperatorEnum.IN.value == "In"


class TestDeletionModeEnum:
    def test_soft_delete(self):
        assert DeletionModeEnum.SOFT_DELETE.value == "SoftDelete"

    def test_hard_delete(self):
        assert DeletionModeEnum.HARD_DELETE.value == "HardDelete"


class TestExpirationTimeResponseEnum:
    def test_disabled(self):
        assert ExpirationTimeResponseEnum.DISABLED.value == "Disabled"

    def test_enabled(self):
        assert ExpirationTimeResponseEnum.ENABLED.value == "Enabled"


class TestThroughputModeEnum:
    def test_on_demand(self):
        assert ThroughputModeEnum.ON_DEMAND.value == "OnDemand"

    def test_provisioned(self):
        assert ThroughputModeEnum.PROVISIONED.value == "Provisioned"
