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
"""Unit tests for model_reuse.py."""

import hashlib
import pytest
from unittest.mock import Mock, patch, call

from sagemaker.serve.model_reuse import (
    MODEL_SOURCE_TAG_KEY,
    normalize_tag_value,
    find_existing_bedrock_model,
    find_existing_sagemaker_endpoint,
    build_source_tag,
    check_bedrock_model_status,
    check_sagemaker_endpoint_status,
    _arn_to_name,
)


@pytest.fixture
def boto_session():
    return Mock()


@pytest.fixture
def bedrock_client(boto_session):
    client = Mock()
    boto_session.client.return_value = client
    return client


@pytest.fixture
def sagemaker_client(boto_session):
    client = Mock()
    boto_session.client.return_value = client
    return client


SAMPLE_ARN = "arn:aws:bedrock:us-east-1:123456789012:custom-model/my-model"
ENDPOINT_ARN = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-endpoint"


def _bedrock_with_tagged_model(bedrock_client, arn, tag_value):
    """Configure a bedrock client mock to return one model carrying the source tag."""
    bedrock_client.list_custom_models.return_value = {"modelSummaries": [{"modelArn": arn}]}
    bedrock_client.list_tags_for_resource.return_value = {
        "tags": [{"key": MODEL_SOURCE_TAG_KEY, "value": tag_value}]
    }


def _sagemaker_with_tagged_endpoint(sagemaker_client, arn, tag_value):
    """Configure a sagemaker client mock to return one endpoint carrying the source tag."""
    sagemaker_client.list_endpoints.return_value = {"Endpoints": [{"EndpointArn": arn}]}
    sagemaker_client.list_tags.return_value = {
        "Tags": [{"Key": MODEL_SOURCE_TAG_KEY, "Value": tag_value}]
    }


@pytest.mark.parametrize(
    "length",
    [0, 100, 256],
    ids=["empty", "short", "at_limit"],
)
def test_normalize_tag_value_within_limit(length):
    value = "a" * length
    assert normalize_tag_value(value) == value


@pytest.mark.parametrize(
    "length",
    [257, 512],
    ids=["one_over", "long"],
)
def test_normalize_tag_value_exceeds_limit(length):
    value = "x" * length
    result = normalize_tag_value(value)
    assert len(result) == 256
    assert result[:224] == value[:224]
    assert result[224] == "-"


def test_normalize_tag_value_sha256_suffix():
    value = "s3://my-bucket/" + "a" * 300
    result = normalize_tag_value(value)
    expected_hash = hashlib.sha256(value.encode()).hexdigest()[:31]
    assert result.endswith(expected_hash)
    assert result == f"{value[:224]}-{expected_hash}"


def test_find_existing_bedrock_model_returns_arn_when_active(boto_session, bedrock_client):
    _bedrock_with_tagged_model(bedrock_client, SAMPLE_ARN, "source-id")
    bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}

    result = find_existing_bedrock_model(bedrock_client, "source-id")

    assert result == SAMPLE_ARN
    bedrock_client.list_tags_for_resource.assert_called_once_with(resourceARN=SAMPLE_ARN)


def test_find_existing_bedrock_model_paginates(boto_session, bedrock_client):
    other_arn = "arn:aws:bedrock:us-east-1:123456789012:custom-model/other"
    bedrock_client.list_custom_models.side_effect = [
        {"modelSummaries": [{"modelArn": other_arn}], "nextToken": "page-2"},
        {"modelSummaries": [{"modelArn": SAMPLE_ARN}]},
    ]
    bedrock_client.list_tags_for_resource.side_effect = [
        {"tags": [{"key": MODEL_SOURCE_TAG_KEY, "value": "different"}]},
        {"tags": [{"key": MODEL_SOURCE_TAG_KEY, "value": "source-id"}]},
    ]
    bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}

    result = find_existing_bedrock_model(bedrock_client, "source-id")

    assert result == SAMPLE_ARN
    assert bedrock_client.list_custom_models.call_count == 2


@patch("sagemaker.serve.model_reuse.time.sleep")
def test_find_existing_bedrock_model_polls_creating_until_ready(mock_sleep, boto_session, bedrock_client):
    _bedrock_with_tagged_model(bedrock_client, SAMPLE_ARN, "source-id")
    bedrock_client.get_custom_model.side_effect = [
        {"modelStatus": "Creating"},
        {"modelStatus": "Creating"},
        {"modelStatus": "Active"},
    ]

    result = find_existing_bedrock_model(
        bedrock_client, "source-id", poll_interval=5, max_wait=900
    )

    assert result == SAMPLE_ARN
    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(5)


@patch("sagemaker.serve.model_reuse.time.sleep")
def test_find_existing_bedrock_model_raises_timeout_on_creating(mock_sleep, boto_session, bedrock_client):
    _bedrock_with_tagged_model(bedrock_client, SAMPLE_ARN, "source-id")
    bedrock_client.get_custom_model.return_value = {"modelStatus": "Creating"}

    with pytest.raises(TimeoutError, match="did not become ready"):
        find_existing_bedrock_model(
            bedrock_client, "source-id", poll_interval=5, max_wait=10
        )


def test_find_existing_bedrock_model_returns_none_on_failed(boto_session, bedrock_client):
    _bedrock_with_tagged_model(bedrock_client, SAMPLE_ARN, "source-id")
    bedrock_client.get_custom_model.return_value = {"modelStatus": "Failed"}

    result = find_existing_bedrock_model(bedrock_client, "source-id")

    assert result is None


def test_find_existing_bedrock_model_returns_none_on_list_failure(boto_session, bedrock_client):
    bedrock_client.list_custom_models.side_effect = Exception("Access denied")

    result = find_existing_bedrock_model(bedrock_client, "source-id")

    assert result is None


def test_find_existing_bedrock_model_returns_none_when_no_match(boto_session, bedrock_client):
    bedrock_client.list_custom_models.return_value = {
        "modelSummaries": [{"modelArn": SAMPLE_ARN}]
    }
    bedrock_client.list_tags_for_resource.return_value = {
        "tags": [{"key": MODEL_SOURCE_TAG_KEY, "value": "different"}]
    }

    result = find_existing_bedrock_model(bedrock_client, "source-id")

    assert result is None


def test_find_existing_bedrock_model_returns_none_when_no_models(boto_session, bedrock_client):
    bedrock_client.list_custom_models.return_value = {"modelSummaries": []}

    result = find_existing_bedrock_model(bedrock_client, "source-id")

    assert result is None


def test_find_existing_sagemaker_endpoint_returns_arn_when_in_service(boto_session, sagemaker_client):
    _sagemaker_with_tagged_endpoint(sagemaker_client, ENDPOINT_ARN, "source-id")
    sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}

    result = find_existing_sagemaker_endpoint(sagemaker_client, "source-id")

    assert result == ENDPOINT_ARN
    sagemaker_client.list_tags.assert_called_once_with(ResourceArn=ENDPOINT_ARN)


def test_find_existing_sagemaker_endpoint_paginates(boto_session, sagemaker_client):
    other_arn = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/other"
    sagemaker_client.list_endpoints.side_effect = [
        {"Endpoints": [{"EndpointArn": other_arn}], "NextToken": "page-2"},
        {"Endpoints": [{"EndpointArn": ENDPOINT_ARN}]},
    ]
    sagemaker_client.list_tags.side_effect = [
        {"Tags": [{"Key": MODEL_SOURCE_TAG_KEY, "Value": "different"}]},
        {"Tags": [{"Key": MODEL_SOURCE_TAG_KEY, "Value": "source-id"}]},
    ]
    sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}

    result = find_existing_sagemaker_endpoint(sagemaker_client, "source-id")

    assert result == ENDPOINT_ARN
    assert sagemaker_client.list_endpoints.call_count == 2


def test_find_existing_sagemaker_endpoint_returns_none_on_failed(boto_session, sagemaker_client):
    _sagemaker_with_tagged_endpoint(sagemaker_client, ENDPOINT_ARN, "source-id")
    sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "Failed"}

    result = find_existing_sagemaker_endpoint(sagemaker_client, "source-id")

    assert result is None


def test_find_existing_sagemaker_endpoint_returns_none_on_list_failure(boto_session, sagemaker_client):
    sagemaker_client.list_endpoints.side_effect = Exception("Access denied")

    result = find_existing_sagemaker_endpoint(sagemaker_client, "source-id")

    assert result is None


def test_find_existing_sagemaker_endpoint_returns_none_when_no_endpoints(boto_session, sagemaker_client):
    sagemaker_client.list_endpoints.return_value = {"Endpoints": []}

    result = find_existing_sagemaker_endpoint(sagemaker_client, "source-id")

    assert result is None


def test_build_source_tag_returns_correct_dict():
    source_id = "s3://bucket/path/to/model"
    tag = build_source_tag(source_id)

    assert tag == {"key": MODEL_SOURCE_TAG_KEY, "value": source_id}


def test_build_source_tag_normalizes_long_value():
    source_id = "s3://bucket/" + "a" * 300
    tag = build_source_tag(source_id)

    assert tag["key"] == MODEL_SOURCE_TAG_KEY
    assert len(tag["value"]) == 256


def test_check_bedrock_model_status_returns_model_status():
    bedrock_client = Mock()
    bedrock_client.get_custom_model.return_value = {"modelStatus": "Active"}

    result = check_bedrock_model_status(bedrock_client, SAMPLE_ARN)

    assert result == "Active"
    bedrock_client.get_custom_model.assert_called_once_with(modelIdentifier=SAMPLE_ARN)


def test_check_bedrock_model_status_raises_on_failure():
    bedrock_client = Mock()
    bedrock_client.get_custom_model.side_effect = Exception("Not found")

    with pytest.raises(Exception, match="Not found"):
        check_bedrock_model_status(bedrock_client, SAMPLE_ARN)


def test_check_sagemaker_endpoint_status_returns_endpoint_status():
    sm_client = Mock()
    sm_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}

    result = check_sagemaker_endpoint_status(sm_client, ENDPOINT_ARN)

    assert result == "InService"
    sm_client.describe_endpoint.assert_called_once_with(EndpointName="my-endpoint")


def test_check_sagemaker_endpoint_status_raises_on_failure():
    sm_client = Mock()
    sm_client.describe_endpoint.side_effect = Exception("Endpoint not found")

    with pytest.raises(Exception, match="Endpoint not found"):
        check_sagemaker_endpoint_status(sm_client, ENDPOINT_ARN)
