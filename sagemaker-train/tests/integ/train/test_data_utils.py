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
"""Integration tests for data_utils (S3 file loading and multimodal detection)."""

from __future__ import absolute_import

import json
import boto3
import pytest
from botocore.exceptions import ClientError

from sagemaker.train.common_utils.data_utils import (
    load_file_content,
    is_multimodal_data,
    FileLoadError,
)
from sagemaker.ai_registry.dataset import DataSet

TEST_BUCKET = "sagemaker-train-data-utils-integ-test"
TEST_PREFIX = "test-data"
DEFAULT_REGION = "us-west-2"

# --- Sample dataset content ---

SAMPLE_TEXT_JSONL = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    },
]

SAMPLE_MULTIMODAL_JSONL = [
    {
        "messages": [
            {"role": "user", "content": [{"text": "Describe this image"}, {"image": "image_data"}]},
            {"role": "assistant", "content": [{"text": "It shows a cat."}]},
        ]
    },
]

SAMPLE_TEXT_JSON = [
    {
        "messages": [
            {"role": "user", "content": "Summarize this."},
            {"role": "assistant", "content": "Done."},
        ]
    }
]

SAMPLE_MULTIMODAL_JSON = [
    {
        "messages": [
            {
                "role": "user",
                "content": [{"text": "What is in this video?"}, {"video": "video_data"}],
            },
            {"role": "assistant", "content": [{"text": "A dog playing fetch."}]},
        ]
    }
]

# --- Test dataset keys ---

DATASETS = {
    "text_only.jsonl": "\n".join(json.dumps(r) for r in SAMPLE_TEXT_JSONL) + "\n",
    "multimodal.jsonl": "\n".join(json.dumps(r) for r in SAMPLE_MULTIMODAL_JSONL) + "\n",
    "text_only.json": json.dumps(SAMPLE_TEXT_JSON),
    "multimodal.json": json.dumps(SAMPLE_MULTIMODAL_JSON),
}


@pytest.fixture(scope="module")
def s3_client():
    return boto3.client("s3", region_name=DEFAULT_REGION)


@pytest.fixture(scope="module", autouse=True)
def setup_test_bucket(s3_client):
    """Create the test bucket and upload sample datasets if they don't already exist."""
    # Create bucket if it doesn't exist
    try:
        s3_client.head_bucket(Bucket=TEST_BUCKET)
    except ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:
            s3_client.create_bucket(
                Bucket=TEST_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": DEFAULT_REGION},
            )
        else:
            raise

    # Upload each sample dataset if it doesn't already exist
    for filename, content in DATASETS.items():
        key = f"{TEST_PREFIX}/{filename}"
        try:
            s3_client.head_object(Bucket=TEST_BUCKET, Key=key)
        except ClientError as e:
            if int(e.response["Error"]["Code"]) == 404:
                s3_client.put_object(
                    Bucket=TEST_BUCKET,
                    Key=key,
                    Body=content.encode("utf-8"),
                )
            else:
                raise

    yield


def _s3_uri(filename: str) -> str:
    return f"s3://{TEST_BUCKET}/{TEST_PREFIX}/{filename}"


# --- load_file_content tests ---


class TestLoadFileContentS3:
    """Integration tests for load_file_content reading from S3."""

    def test_load_jsonl_from_s3(self):
        """Verify we can stream a .jsonl file from S3 line by line."""
        lines = list(load_file_content(_s3_uri("text_only.jsonl"), extension=".jsonl"))
        assert len(lines) == len(SAMPLE_TEXT_JSONL)
        for line in lines:
            record = json.loads(line)
            assert "messages" in record

    def test_load_json_from_s3(self):
        """Verify we can load a .json file from S3."""
        lines = list(load_file_content(_s3_uri("text_only.json"), extension=".json"))
        content = "\n".join(lines)
        data = json.loads(content)
        assert isinstance(data, list)
        assert len(data) == len(SAMPLE_TEXT_JSON)

    def test_load_file_wrong_extension_raises(self):
        """Verify extension validation triggers before S3 access."""
        with pytest.raises(FileLoadError, match="extension"):
            list(load_file_content(_s3_uri("text_only.jsonl"), extension=".json"))

    def test_load_nonexistent_s3_key_raises(self):
        """Verify a missing S3 key raises FileLoadError."""
        with pytest.raises(FileLoadError, match="Failed to load S3 file"):
            list(load_file_content(_s3_uri("does_not_exist.jsonl"), extension=".jsonl"))


# --- is_multimodal_data tests ---


class TestIsMultimodal:
    """Integration tests for is_multimodal_data reading datasets from S3."""

    def test_text_only_jsonl_returns_false(self):
        assert is_multimodal_data(_s3_uri("text_only.jsonl")) is False

    def test_multimodal_jsonl_returns_true(self):
        assert is_multimodal_data(_s3_uri("multimodal.jsonl")) is True

    def test_text_only_json_returns_false(self):
        assert is_multimodal_data(_s3_uri("text_only.json")) is False

    def test_multimodal_json_returns_true(self):
        assert is_multimodal_data(_s3_uri("multimodal.json")) is True

    def test_nonexistent_file_returns_false(self):
        """is_multimodal_data should gracefully return False for missing files."""
        result = is_multimodal_data(_s3_uri("nonexistent_dataset.jsonl"))
        assert result is False


def _make_dataset(source: str) -> DataSet:
    """Create a minimal DataSet object with the given source for testing."""
    return DataSet(
        name="test-dataset",
        arn="arn:aws:sagemaker:us-west-2:123456789012:hub-content/test-dataset/1",
        version="1",
        status="Available",
        source=source,
    )


class TestIsMultimodalWithDataSet:
    """Integration tests for is_multimodal_data accepting a DataSet object."""

    def test_dataset_text_only_jsonl_returns_false(self):
        """DataSet with text-only .jsonl source should return False."""
        ds = _make_dataset(_s3_uri("text_only.jsonl"))
        assert is_multimodal_data(ds) is False

    def test_dataset_multimodal_jsonl_returns_true(self):
        """DataSet with multimodal .jsonl source should return True."""
        ds = _make_dataset(_s3_uri("multimodal.jsonl"))
        assert is_multimodal_data(ds) is True

    def test_dataset_text_only_json_returns_false(self):
        """DataSet with text-only .json source should return False."""
        ds = _make_dataset(_s3_uri("text_only.json"))
        assert is_multimodal_data(ds) is False

    def test_dataset_multimodal_json_returns_true(self):
        """DataSet with multimodal .json source should return True."""
        ds = _make_dataset(_s3_uri("multimodal.json"))
        assert is_multimodal_data(ds) is True

    def test_dataset_nonexistent_source_returns_false(self):
        """DataSet pointing to a nonexistent S3 key should gracefully return False."""
        ds = _make_dataset(_s3_uri("nonexistent_dataset.jsonl"))
        assert is_multimodal_data(ds) is False
