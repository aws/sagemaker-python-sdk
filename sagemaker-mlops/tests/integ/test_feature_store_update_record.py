"""Integration tests for UpdateRecord (feature-level writes) on Standard_V2 feature groups.

These tests require an updated boto3/botocore that ships the UpdateRecord operation and a
region where Feature Store Standard_V2 storage is available.
"""
import time
import pytest
import pandas as pd

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.mlops.feature_store import (
    FeatureGroup,
    OnlineStoreConfig,
    OnlineStoreStorageTypeEnum,
)
from sagemaker.mlops.feature_store.feature_utils import (
    load_feature_definitions_from_dataframe,
    ingest_dataframe,
    update_record,
)
from sagemaker.core.utils import unique_name_from_base


@pytest.fixture(scope="module")
def sagemaker_session():
    return Session()


@pytest.fixture(scope="module")
def role():
    return get_execution_role()


@pytest.fixture
def feature_group_name():
    return unique_name_from_base("integ-test-updaterecord-fg")


@pytest.fixture
def sample_dataframe():
    current_time = int(time.time())
    return pd.DataFrame(
        {
            "record_id": [f"id-{i}" for i in range(3)],
            "city": ["seattle", "portland", "denver"],
            "temperature": [float(i) for i in range(3)],
            "event_time": [float(current_time + i) for i in range(3)],
        }
    )


def cleanup_feature_group(feature_group_name):
    try:
        fg = FeatureGroup.get(feature_group_name=feature_group_name)
        fg.delete()
        time.sleep(2)
    except Exception:
        pass


def _create_standard_v2_group(feature_group_name, sample_dataframe, role):
    feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
    fg = FeatureGroup.create(
        feature_group_name=feature_group_name,
        record_identifier_feature_name="record_id",
        event_time_feature_name="event_time",
        feature_definitions=feature_definitions,
        role_arn=role,
        online_store_config=OnlineStoreConfig(
            enable_online_store=True,
            storage_type=OnlineStoreStorageTypeEnum.STANDARD_V2.value,
        ),
    )
    fg.wait_for_status("Created")
    return fg


def test_update_record_preserves_unlisted_features(
    feature_group_name, sample_dataframe, role
):
    """UpdateRecord writes only the supplied features; others are preserved."""
    try:
        fg = _create_standard_v2_group(feature_group_name, sample_dataframe, role)
        ingest_dataframe(feature_group_name=feature_group_name, data_frame=sample_dataframe)
        time.sleep(15)

        new_event_time = float(int(time.time()) + 100)
        update_record(
            feature_group_name=feature_group_name,
            record_identifier_value_as_string="id-0",
            features=[
                {"feature_name": "city", "value_as_string": "tacoma"},
                {"feature_name": "event_time", "value_as_string": str(new_event_time)},
            ],
        )
        time.sleep(10)

        record = fg.get_record(record_identifier_value_as_string="id-0")
        values = {fv.feature_name: fv.value_as_string for fv in record.record}
        assert values["city"] == "tacoma"  # updated
        assert values["temperature"] == "0.0"  # preserved (not in the update)
    finally:
        cleanup_feature_group(feature_group_name)


def test_update_record_stale_event_time_conflict(
    feature_group_name, sample_dataframe, role
):
    """An EventTime not greater than the current one is rejected with a conflict."""
    from botocore.exceptions import ClientError

    try:
        _create_standard_v2_group(feature_group_name, sample_dataframe, role)
        ingest_dataframe(feature_group_name=feature_group_name, data_frame=sample_dataframe)
        time.sleep(15)

        stale_event_time = "1.0"
        with pytest.raises(ClientError) as exc:
            update_record(
                feature_group_name=feature_group_name,
                record_identifier_value_as_string="id-0",
                features=[
                    {"feature_name": "city", "value_as_string": "tacoma"},
                    {"feature_name": "event_time", "value_as_string": stale_event_time},
                ],
            )
        assert exc.value.response["Error"]["Code"] in ("ConflictException", "ValidationError")
    finally:
        cleanup_feature_group(feature_group_name)
