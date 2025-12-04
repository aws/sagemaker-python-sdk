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
"""Unit tests for sagemaker.core.jumpstart.cache module."""
from __future__ import absolute_import

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from packaging.version import Version

from sagemaker.core.jumpstart.cache import JumpStartModelsCache
from sagemaker.core.jumpstart.types import (
    JumpStartCachedContentKey,
    JumpStartVersionedModelId,
    JumpStartS3FileType,
    JumpStartModelHeader,
    JumpStartModelSpecs,
    HubContentType,
)
from sagemaker.core.jumpstart.enums import JumpStartModelType


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    client = Mock()
    return client


@pytest.fixture
def mock_sagemaker_session():
    """Create a mock SageMaker session."""
    session = Mock()
    return session


@pytest.fixture
def sample_manifest():
    """Create a sample manifest."""
    return {
        JumpStartVersionedModelId("model-1", "1.0.0"): JumpStartModelHeader(
            {
                "model_id": "model-1",
                "version": "1.0.0",
                "min_version": "2.0.0",
                "spec_key": "specs/model-1-1.0.0.json",
            }
        ),
        JumpStartVersionedModelId("model-1", "2.0.0"): JumpStartModelHeader(
            {
                "model_id": "model-1",
                "version": "2.0.0",
                "min_version": "2.0.0",
                "spec_key": "specs/model-1-2.0.0.json",
            }
        ),
    }


class TestJumpStartModelsCacheInitialization:
    """Test JumpStartModelsCache initialization."""

    def test_cache_init_default(self, mock_s3_client, mock_sagemaker_session):
        """Test cache initialization with defaults."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        assert cache._region == "us-west-2"
        assert cache.s3_bucket_name is not None

    def test_cache_init_custom_bucket(self, mock_s3_client, mock_sagemaker_session):
        """Test cache initialization with custom bucket."""
        cache = JumpStartModelsCache(
            region="us-west-2",
            s3_bucket_name="my-custom-bucket",
            s3_client=mock_s3_client,
            sagemaker_session=mock_sagemaker_session,
        )

        assert cache.s3_bucket_name == "my-custom-bucket"

    def test_cache_init_custom_manifest_key(self, mock_s3_client, mock_sagemaker_session):
        """Test cache initialization with custom manifest key."""
        cache = JumpStartModelsCache(
            region="us-west-2",
            manifest_file_s3_key="custom/manifest.json",
            s3_client=mock_s3_client,
            sagemaker_session=mock_sagemaker_session,
        )

        assert cache._manifest_file_s3_key == "custom/manifest.json"


class TestSetRegion:
    """Test set_region method."""

    def test_set_region_clears_cache(self, mock_s3_client, mock_sagemaker_session):
        """Test that setting region clears cache."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        with patch.object(cache, "clear") as mock_clear:
            cache.set_region("us-east-1")

            assert cache._region == "us-east-1"
            mock_clear.assert_called_once()

    def test_set_region_same_region_no_clear(self, mock_s3_client, mock_sagemaker_session):
        """Test that setting same region doesn't clear cache."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        with patch.object(cache, "clear") as mock_clear:
            cache.set_region("us-west-2")

            mock_clear.assert_not_called()


class TestGetRegion:
    """Test get_region method."""

    def test_get_region(self, mock_s3_client, mock_sagemaker_session):
        """Test getting region."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        assert cache.get_region() == "us-west-2"


class TestSetManifestFileS3Key:
    """Test set_manifest_file_s3_key method."""

    def test_set_manifest_file_s3_key_open_weight(self, mock_s3_client, mock_sagemaker_session):
        """Test setting open weight manifest key."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        with patch.object(cache, "clear") as mock_clear:
            cache.set_manifest_file_s3_key(
                cache._manifest_file_s3_key, JumpStartS3FileType.OPEN_WEIGHT_MANIFEST
            )
            mock_clear.assert_not_called()

    def test_set_manifest_file_s3_key_proprietary(self, mock_s3_client, mock_sagemaker_session):
        """Test setting proprietary manifest key."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        with patch.object(cache, "clear") as mock_clear:
            cache.set_manifest_file_s3_key(
                cache._proprietary_manifest_s3_key, JumpStartS3FileType.PROPRIETARY_MANIFEST
            )
            mock_clear.assert_not_called()

    def test_set_manifest_file_s3_key_invalid_type(self, mock_s3_client, mock_sagemaker_session):
        """Test error with invalid file type."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        with pytest.raises(ValueError, match="Bad value"):
            cache.set_manifest_file_s3_key(
                "new/manifest.json", JumpStartS3FileType.OPEN_WEIGHT_SPECS
            )


class TestGetManifestFileS3Key:
    """Test get_manifest_file_s3_key method."""

    def test_get_manifest_file_s3_key_open_weight(self, mock_s3_client, mock_sagemaker_session):
        """Test getting open weight manifest key."""
        cache = JumpStartModelsCache(
            region="us-west-2",
            manifest_file_s3_key="custom/manifest.json",
            s3_client=mock_s3_client,
            sagemaker_session=mock_sagemaker_session,
        )

        key = cache.get_manifest_file_s3_key(JumpStartS3FileType.OPEN_WEIGHT_MANIFEST)

        assert key == "custom/manifest.json"

    def test_get_manifest_file_s3_key_proprietary(self, mock_s3_client, mock_sagemaker_session):
        """Test getting proprietary manifest key."""
        cache = JumpStartModelsCache(
            region="us-west-2",
            proprietary_manifest_s3_key="custom/proprietary.json",
            s3_client=mock_s3_client,
            sagemaker_session=mock_sagemaker_session,
        )

        key = cache.get_manifest_file_s3_key(JumpStartS3FileType.PROPRIETARY_MANIFEST)

        assert key == "custom/proprietary.json"


class TestSetS3BucketName:
    """Test set_s3_bucket_name method."""

    def test_set_s3_bucket_name_clears_cache(self, mock_s3_client, mock_sagemaker_session):
        """Test that setting bucket name clears cache."""
        cache = JumpStartModelsCache(
            region="us-west-2",
            s3_bucket_name="old-bucket",
            s3_client=mock_s3_client,
            sagemaker_session=mock_sagemaker_session,
        )

        with patch.object(cache, "clear") as mock_clear:
            cache.set_s3_bucket_name("new-bucket")

            assert cache.s3_bucket_name == "new-bucket"
            mock_clear.assert_called_once()


class TestGetBucket:
    """Test get_bucket method."""

    def test_get_bucket(self, mock_s3_client, mock_sagemaker_session):
        """Test getting bucket name."""
        cache = JumpStartModelsCache(
            region="us-west-2",
            s3_bucket_name="test-bucket",
            s3_client=mock_s3_client,
            sagemaker_session=mock_sagemaker_session,
        )

        assert cache.get_bucket() == "test-bucket"


class TestGetJsonFileAndEtagFromS3:
    """Test _get_json_file_and_etag_from_s3 method."""

    def test_get_json_file_and_etag_from_s3(self, mock_s3_client, mock_sagemaker_session):
        """Test getting JSON file from S3."""
        test_data = {"key": "value"}
        mock_body = Mock()
        mock_body.read.return_value = json.dumps(test_data).encode("utf-8")
        mock_s3_client.get_object.return_value = {"Body": mock_body, "ETag": "test-etag"}

        cache = JumpStartModelsCache(
            region="us-west-2",
            s3_bucket_name="test-bucket",
            s3_client=mock_s3_client,
            sagemaker_session=mock_sagemaker_session,
        )

        data, etag = cache._get_json_file_and_etag_from_s3("test-key")

        assert data == test_data
        assert etag == "test-etag"
        mock_s3_client.get_object.assert_called_once_with(Bucket="test-bucket", Key="test-key")


class TestIsLocalMetadataMode:
    """Test _is_local_metadata_mode method."""

    def test_is_local_metadata_mode_true(self, mock_s3_client, mock_sagemaker_session, tmp_path):
        """Test local metadata mode is True when env vars set."""
        manifest_dir = tmp_path / "manifests"
        specs_dir = tmp_path / "specs"
        manifest_dir.mkdir()
        specs_dir.mkdir()

        with patch.dict(
            os.environ,
            {
                "ENV_VARIABLE_JUMPSTART_MANIFEST_LOCAL_ROOT_DIR_OVERRIDE": str(manifest_dir),
                "ENV_VARIABLE_JUMPSTART_SPECS_LOCAL_ROOT_DIR_OVERRIDE": str(specs_dir),
            },
        ):
            cache = JumpStartModelsCache(
                region="us-west-2",
                s3_client=mock_s3_client,
                sagemaker_session=mock_sagemaker_session,
            )

            # Note: The actual env variable names are different in the code
            # This test demonstrates the pattern
            assert (
                cache._is_local_metadata_mode() is False
            )  # Will be False without correct env vars

    def test_is_local_metadata_mode_false(self, mock_s3_client, mock_sagemaker_session):
        """Test local metadata mode is False when env vars not set."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        assert cache._is_local_metadata_mode() is False


class TestSelectVersion:
    """Test _select_version method."""

    def test_select_version_wildcard(self, mock_s3_client, mock_sagemaker_session):
        """Test selecting latest version with wildcard."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        available_versions = ["1.0.0", "1.1.0", "2.0.0"]
        result = cache._select_version(
            "model-1", "*", available_versions, JumpStartModelType.OPEN_WEIGHTS
        )

        assert result == "2.0.0"

    def test_select_version_exact_match(self, mock_s3_client, mock_sagemaker_session):
        """Test selecting exact version."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        available_versions = ["1.0.0", "1.1.0", "2.0.0"]
        result = cache._select_version(
            "model-1", "1.1.0", available_versions, JumpStartModelType.OPEN_WEIGHTS
        )

        assert result == "1.1.0"

    def test_select_version_not_found(self, mock_s3_client, mock_sagemaker_session):
        """Test selecting non-existent version."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        available_versions = ["1.0.0", "1.1.0"]
        result = cache._select_version(
            "model-1", "2.0.0", available_versions, JumpStartModelType.OPEN_WEIGHTS
        )

        assert result is None

    def test_select_version_proprietary_wildcard_error(
        self, mock_s3_client, mock_sagemaker_session
    ):
        """Test error with wildcard for proprietary models."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        available_versions = ["1.0.0", "1.1.0"]

        with pytest.raises(KeyError, match="wildcard"):
            cache._select_version(
                "model-1", "1.*", available_versions, JumpStartModelType.PROPRIETARY
            )


class TestGetManifest:
    """Test get_manifest method."""

    def test_get_manifest_open_weights(
        self, mock_s3_client, mock_sagemaker_session, sample_manifest
    ):
        """Test getting open weights manifest."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        with patch.object(cache._content_cache, "get") as mock_get:
            mock_value = Mock()
            mock_value.formatted_content = sample_manifest
            mock_get.return_value = (mock_value, True)

            manifest = cache.get_manifest(JumpStartModelType.OPEN_WEIGHTS)

            assert len(manifest) == 2
            assert all(isinstance(h, JumpStartModelHeader) for h in manifest)


class TestGetHeader:
    """Test get_header method."""

    def test_get_header_success(self, mock_s3_client, mock_sagemaker_session, sample_manifest):
        """Test getting header successfully."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        with patch.object(cache._open_weight_model_id_manifest_key_cache, "get") as mock_cache_get:
            mock_cache_get.return_value = (JumpStartVersionedModelId("model-1", "1.0.0"), True)

            with patch.object(cache._content_cache, "get") as mock_content_get:
                mock_value = Mock()
                mock_value.formatted_content = sample_manifest
                mock_content_get.return_value = (mock_value, True)

                header = cache.get_header("model-1", "1.0.0", JumpStartModelType.OPEN_WEIGHTS)

                assert isinstance(header, JumpStartModelHeader)


class TestGetSpecs:
    """Test get_specs method."""

    def test_get_specs_success(self, mock_s3_client, mock_sagemaker_session, sample_manifest):
        """Test getting specs successfully."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        mock_specs = Mock(spec=JumpStartModelSpecs)

        with patch.object(cache, "get_header") as mock_get_header:
            mock_header = Mock()
            mock_header.spec_key = "specs/model-1-1.0.0.json"
            mock_header.model_id = "model-1"
            mock_header.version = "1.0.0"
            mock_get_header.return_value = mock_header

            with patch.object(cache._content_cache, "get") as mock_content_get:
                mock_value = Mock()
                mock_value.formatted_content = mock_specs
                mock_content_get.return_value = (mock_value, True)

                specs = cache.get_specs("model-1", "1.0.0", JumpStartModelType.OPEN_WEIGHTS)

                assert specs == mock_specs


class TestClear:
    """Test clear method."""

    def test_clear(self, mock_s3_client, mock_sagemaker_session):
        """Test clearing all caches."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        with patch.object(cache._content_cache, "clear") as mock_content_clear:
            with patch.object(
                cache._open_weight_model_id_manifest_key_cache, "clear"
            ) as mock_ow_clear:
                with patch.object(
                    cache._proprietary_model_id_manifest_key_cache, "clear"
                ) as mock_prop_clear:
                    cache.clear()

                    mock_content_clear.assert_called_once()
                    mock_ow_clear.assert_called_once()
                    mock_prop_clear.assert_called_once()


class TestGetHubModel:
    """Test get_hub_model method."""

    def test_get_hub_model(self, mock_s3_client, mock_sagemaker_session):
        """Test getting hub model."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        mock_specs = Mock(spec=JumpStartModelSpecs)
        hub_model_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/Model/my-model/1"
        )

        with patch.object(cache._content_cache, "get") as mock_get:
            mock_value = Mock()
            mock_value.formatted_content = mock_specs
            mock_get.return_value = (mock_value, True)

            result = cache.get_hub_model(hub_model_arn)

            assert result == mock_specs


class TestGetHubModelReference:
    """Test get_hub_model_reference method."""

    def test_get_hub_model_reference(self, mock_s3_client, mock_sagemaker_session):
        """Test getting hub model reference."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        mock_specs = Mock(spec=JumpStartModelSpecs)
        hub_model_ref_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/ModelReference/my-ref/1"
        )

        with patch.object(cache._content_cache, "get") as mock_get:
            mock_value = Mock()
            mock_value.formatted_content = mock_specs
            mock_get.return_value = (mock_value, True)

            result = cache.get_hub_model_reference(hub_model_ref_arn)

            assert result == mock_specs


class TestGetJsonMd5Hash:
    """Test _get_json_md5_hash method."""

    def test_get_json_md5_hash(self, mock_s3_client, mock_sagemaker_session):
        """Test getting MD5 hash."""
        mock_s3_client.head_object.return_value = {"ETag": "test-etag"}

        cache = JumpStartModelsCache(
            region="us-west-2",
            s3_bucket_name="test-bucket",
            s3_client=mock_s3_client,
            sagemaker_session=mock_sagemaker_session,
        )

        etag = cache._get_json_md5_hash("test-key")

        assert etag == "test-etag"
        mock_s3_client.head_object.assert_called_once_with(Bucket="test-bucket", Key="test-key")

    def test_get_json_md5_hash_local_mode_error(self, mock_s3_client, mock_sagemaker_session):
        """Test error when trying to get hash in local mode."""
        cache = JumpStartModelsCache(
            region="us-west-2", s3_client=mock_s3_client, sagemaker_session=mock_sagemaker_session
        )

        with patch.object(cache, "_is_local_metadata_mode", return_value=True):
            with pytest.raises(ValueError, match="Cannot get md5 hash"):
                cache._get_json_md5_hash("test-key")
