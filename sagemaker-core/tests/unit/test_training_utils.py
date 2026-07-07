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
"""Unit tests for Nova manifest/checkpoint helpers in training/utils.py."""
import io
import json
import tarfile

import pytest
from unittest.mock import Mock

from sagemaker.core.training.utils import (
    build_nova_manifest_s3_uri,
    build_nova_output_tar_gz_s3_uri,
    read_nova_checkpoint_uri_from_manifest,
    resolve_nova_checkpoint_uri,
)

CHECKPOINT_URI = "s3://bucket/ckpt/step_100"


def test_build_nova_manifest_s3_uri():
    result = build_nova_manifest_s3_uri("s3://bucket/output/", "my-job")
    assert result == "s3://bucket/output/my-job/output/output/manifest.json"


def test_build_nova_manifest_s3_uri_strips_trailing_slash():
    assert build_nova_manifest_s3_uri(
        "s3://bucket/output//", "my-job"
    ) == "s3://bucket/output/my-job/output/output/manifest.json"


def test_build_nova_output_tar_gz_s3_uri():
    result = build_nova_output_tar_gz_s3_uri("s3://bucket/output", "my-job")
    assert result == "s3://bucket/output/my-job/output/output.tar.gz"


def _s3_client_returning(body_bytes):
    client = Mock()
    client.exceptions = Mock()
    client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
    body = Mock()
    body.read.return_value = body_bytes
    client.get_object.return_value = {"Body": body}
    return client


def test_read_manifest_returns_checkpoint_uri():
    client = _s3_client_returning(
        json.dumps({"checkpoint_s3_bucket": CHECKPOINT_URI}).encode("utf-8")
    )
    result = read_nova_checkpoint_uri_from_manifest(
        client, "s3://bucket/output/my-job/output/output/manifest.json"
    )
    assert result == CHECKPOINT_URI
    client.get_object.assert_called_once_with(
        Bucket="bucket", Key="output/my-job/output/output/manifest.json"
    )


def test_read_manifest_missing_key_raises():
    client = _s3_client_returning(json.dumps({"other": "value"}).encode("utf-8"))
    with pytest.raises(ValueError, match="checkpoint_s3_bucket"):
        read_nova_checkpoint_uri_from_manifest(client, "s3://bucket/manifest.json")


def test_read_manifest_not_found_raises():
    client = Mock()
    client.exceptions = Mock()
    client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
    client.get_object.side_effect = client.exceptions.NoSuchKey()
    with pytest.raises(ValueError, match="manifest.json not found"):
        read_nova_checkpoint_uri_from_manifest(client, "s3://bucket/manifest.json")


def test_read_manifest_invalid_json_raises():
    client = _s3_client_returning(b"not-json")
    with pytest.raises(ValueError, match="Failed to parse manifest.json"):
        read_nova_checkpoint_uri_from_manifest(client, "s3://bucket/manifest.json")


def _make_tar_gz_with_manifest(manifest_dict):
    content = json.dumps(manifest_dict).encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def test_resolve_checkpoint_uri_from_raw_manifest():
    client = _s3_client_returning(
        json.dumps({"checkpoint_s3_bucket": CHECKPOINT_URI}).encode("utf-8")
    )
    result = resolve_nova_checkpoint_uri(client, "s3://bucket/output/", "my-job")
    assert result == CHECKPOINT_URI


def test_resolve_checkpoint_uri_falls_back_to_tar_gz():
    client = Mock()
    client.exceptions = Mock()
    no_such_key = type("NoSuchKey", (Exception,), {})
    client.exceptions.NoSuchKey = no_such_key
    tar_bytes = _make_tar_gz_with_manifest({"checkpoint_s3_bucket": CHECKPOINT_URI})

    def get_object(Bucket, Key):
        if Key.endswith("manifest.json"):
            raise no_such_key()
        body = Mock()
        body.read.return_value = tar_bytes
        return {"Body": body}

    client.get_object.side_effect = get_object

    result = resolve_nova_checkpoint_uri(client, "s3://bucket/output/", "my-job")
    assert result == CHECKPOINT_URI


def test_resolve_checkpoint_uri_raises_when_both_sources_fail():
    client = Mock()
    client.exceptions = Mock()
    no_such_key = type("NoSuchKey", (Exception,), {})
    client.exceptions.NoSuchKey = no_such_key
    tar_bytes = _make_tar_gz_with_manifest({"other": "value"})

    def get_object(Bucket, Key):
        if Key.endswith("manifest.json"):
            raise no_such_key()
        body = Mock()
        body.read.return_value = tar_bytes
        return {"Body": body}

    client.get_object.side_effect = get_object

    with pytest.raises(ValueError, match="checkpoint_s3_bucket"):
        resolve_nova_checkpoint_uri(client, "s3://bucket/output/", "my-job")
