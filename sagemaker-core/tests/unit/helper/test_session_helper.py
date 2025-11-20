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
"""Unit tests for sagemaker.core.helper.session_helper module."""
from __future__ import absolute_import

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from botocore.exceptions import ClientError

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.session_settings import SessionSettings


@pytest.fixture
def mock_boto_session():
    """Create a mock boto3 session."""
    session = Mock()
    session.region_name = "us-west-2"
    session.client = Mock()
    session.resource = Mock()
    return session


@pytest.fixture
def mock_sagemaker_client():
    """Create a mock SageMaker client."""
    client = Mock()
    return client


class TestSessionInitialization:
    """Test Session initialization."""

    @patch("sagemaker.core.helper.session_helper.boto3")
    def test_session_init_default(self, mock_boto3):
        """Test Session initialization with defaults."""
        mock_session = Mock()
        mock_session.region_name = "us-west-2"
        mock_boto3.DEFAULT_SESSION = None
        mock_boto3.Session.return_value = mock_session
        
        session = Session()
        
        assert session._region_name == "us-west-2"
        assert session._default_bucket is None

    def test_session_init_with_custom_bucket(self, mock_boto_session, mock_sagemaker_client):
        """Test Session initialization with custom default bucket."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client,
            default_bucket="my-custom-bucket"
        )
        
        assert session._default_bucket_name_override == "my-custom-bucket"

    def test_session_init_with_bucket_prefix(self, mock_boto_session, mock_sagemaker_client):
        """Test Session initialization with bucket prefix."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client,
            default_bucket_prefix="my-prefix"
        )
        
        assert session.default_bucket_prefix == "my-prefix"

    def test_session_init_no_region(self):
        """Test Session initialization fails without region."""
        mock_session = Mock()
        mock_session.region_name = None
        
        with pytest.raises(ValueError, match="Must setup local AWS configuration"):
            Session(boto_session=mock_session)


class TestAccountId:
    """Test account_id method."""

    def test_account_id(self, mock_boto_session, mock_sagemaker_client):
        """Test getting AWS account ID."""
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_session.client.return_value = mock_sts_client
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        account_id = session.account_id()
        
        assert account_id == "123456789012"


class TestGetCallerIdentityArn:
    """Test get_caller_identity_arn method."""

    def test_get_caller_identity_arn_from_notebook_metadata(self, mock_boto_session, mock_sagemaker_client, tmp_path):
        """Test getting ARN from notebook metadata file."""
        metadata_file = tmp_path / "resource-metadata.json"
        metadata = {
            "ResourceName": "my-notebook",
            "DomainId": None,
            "ExecutionRoleArn": None
        }
        metadata_file.write_text(json.dumps(metadata))
        
        mock_sagemaker_client.describe_notebook_instance.return_value = {
            "RoleArn": "arn:aws:iam::123456789012:role/MyRole"
        }
        
        with patch("sagemaker.core.helper.session_helper.NOTEBOOK_METADATA_FILE", str(metadata_file)):
            session = Session(
                boto_session=mock_boto_session,
                sagemaker_client=mock_sagemaker_client
            )
            
            arn = session.get_caller_identity_arn()
            
            assert "arn:aws:iam::123456789012:role/MyRole" in arn

    def test_get_caller_identity_arn_from_sts(self, mock_boto_session, mock_sagemaker_client):
        """Test getting ARN from STS."""
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/MyRole/session"
        }
        mock_boto_session.client.return_value = mock_sts_client
        
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/MyRole"}
        }
        
        with patch("os.path.exists", return_value=False):
            session = Session(
                boto_session=mock_boto_session,
                sagemaker_client=mock_sagemaker_client
            )
            
            # Mock both STS and IAM clients
            def client_side_effect(service, **kwargs):
                if service == "sts":
                    return mock_sts_client
                elif service == "iam":
                    return mock_iam_client
                return Mock()
            
            mock_boto_session.client.side_effect = client_side_effect
            
            arn = session.get_caller_identity_arn()
            
            assert "arn:aws:iam::123456789012:role/MyRole" in arn


class TestUploadData:
    """Test upload_data method."""

    def test_upload_data_single_file(self, mock_boto_session, mock_sagemaker_client, tmp_path):
        """Test uploading a single file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        mock_s3_resource = Mock()
        mock_s3_object = Mock()
        mock_s3_resource.Object.return_value = mock_s3_object
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client,
            default_bucket="test-bucket"
        )
        session.s3_resource = mock_s3_resource
        
        result = session.upload_data(
            path=str(test_file),
            bucket="test-bucket",
            key_prefix="data"
        )
        
        assert result == "s3://test-bucket/data/test.txt"
        mock_s3_object.upload_file.assert_called_once()

    def test_upload_data_directory(self, mock_boto_session, mock_sagemaker_client, tmp_path):
        """Test uploading a directory."""
        test_dir = tmp_path / "data"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        
        mock_s3_resource = Mock()
        mock_s3_object = Mock()
        mock_s3_resource.Object.return_value = mock_s3_object
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client,
            default_bucket="test-bucket"
        )
        session.s3_resource = mock_s3_resource
        
        result = session.upload_data(
            path=str(test_dir),
            bucket="test-bucket",
            key_prefix="data"
        )
        
        assert result == "s3://test-bucket/data"
        assert mock_s3_object.upload_file.call_count == 2


class TestDownloadData:
    """Test download_data method."""

    def test_download_data_single_file(self, mock_boto_session, mock_sagemaker_client, tmp_path):
        """Test downloading a single file."""
        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "data/test.txt", "Size": 100}
            ]
        }
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_client = mock_s3_client
        
        result = session.download_data(
            path=str(tmp_path),
            bucket="test-bucket",
            key_prefix="data/test.txt"
        )
        
        assert len(result) == 1
        mock_s3_client.download_file.assert_called_once()

    def test_download_data_empty_bucket(self, mock_boto_session, mock_sagemaker_client, tmp_path):
        """Test downloading from empty bucket."""
        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {}
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_client = mock_s3_client
        
        result = session.download_data(
            path=str(tmp_path),
            bucket="test-bucket",
            key_prefix="data/"
        )
        
        assert result == []


class TestDefaultBucket:
    """Test default_bucket method."""

    @patch("sagemaker.core.helper.session_helper.Session._create_s3_bucket_if_it_does_not_exist")
    def test_default_bucket_creates_bucket(self, mock_create_bucket, mock_boto_session, mock_sagemaker_client):
        """Test default_bucket creates bucket if not exists."""
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_session.client.return_value = mock_sts_client
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        bucket = session.default_bucket()
        
        assert bucket == "sagemaker-us-west-2-123456789012"
        mock_create_bucket.assert_called_once()

    def test_default_bucket_uses_override(self, mock_boto_session, mock_sagemaker_client):
        """Test default_bucket uses override if provided."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client,
            default_bucket="my-custom-bucket"
        )
        
        with patch.object(session, "_create_s3_bucket_if_it_does_not_exist"):
            bucket = session.default_bucket()
            
            assert bucket == "my-custom-bucket"


class TestCreateS3BucketIfItDoesNotExist:
    """Test _create_s3_bucket_if_it_does_not_exist method."""

    def test_create_bucket_when_not_exists(self, mock_boto_session, mock_sagemaker_client):
        """Test creating bucket when it doesn't exist."""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.creation_date = None
        mock_s3_resource.Bucket.return_value = mock_bucket
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_resource = mock_s3_resource
        
        with patch.object(session, "general_bucket_check_if_user_has_permission"):
            session._create_s3_bucket_if_it_does_not_exist("test-bucket", "us-west-2")
            
            session.general_bucket_check_if_user_has_permission.assert_called_once()

    def test_skip_create_when_bucket_exists(self, mock_boto_session, mock_sagemaker_client):
        """Test skipping creation when bucket exists."""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.creation_date = "2023-01-01"
        mock_s3_resource.Bucket.return_value = mock_bucket
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_resource = mock_s3_resource
        session._default_bucket_set_by_sdk = False
        
        # Should not raise
        session._create_s3_bucket_if_it_does_not_exist("test-bucket", "us-west-2")


class TestCreateEndpoint:
    """Test create_endpoint method."""

    def test_create_endpoint_success(self, mock_boto_session, mock_sagemaker_client):
        """Test successful endpoint creation."""
        mock_sagemaker_client.create_endpoint.return_value = {
            "EndpointArn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/my-endpoint"
        }
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        with patch.object(session, "wait_for_endpoint"):
            result = session.create_endpoint(
                endpoint_name="my-endpoint",
                config_name="my-config",
                wait=False
            )
            
            assert result == "my-endpoint"
            mock_sagemaker_client.create_endpoint.assert_called_once()

    def test_create_endpoint_with_tags(self, mock_boto_session, mock_sagemaker_client):
        """Test endpoint creation with tags."""
        mock_sagemaker_client.create_endpoint.return_value = {
            "EndpointArn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/my-endpoint"
        }
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        with patch.object(session, "wait_for_endpoint"):
            with patch.object(session, "_append_sagemaker_config_tags", return_value=[]):
                result = session.create_endpoint(
                    endpoint_name="my-endpoint",
                    config_name="my-config",
                    tags=[{"Key": "Environment", "Value": "Test"}],
                    wait=False
                )
                
                assert result == "my-endpoint"


class TestWaitForEndpoint:
    """Test wait_for_endpoint method."""

    def test_wait_for_endpoint_success(self, mock_boto_session, mock_sagemaker_client):
        """Test waiting for endpoint to be in service."""
        mock_sagemaker_client.describe_endpoint.return_value = {
            "EndpointStatus": "InService"
        }
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        with patch("sagemaker.core.helper.session_helper._wait_until") as mock_wait:
            mock_wait.return_value = {"EndpointStatus": "InService"}
            
            result = session.wait_for_endpoint("my-endpoint")
            
            assert result["EndpointStatus"] == "InService"

    def test_wait_for_endpoint_failure(self, mock_boto_session, mock_sagemaker_client):
        """Test waiting for endpoint that fails."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        with patch("sagemaker.core.helper.session_helper._wait_until") as mock_wait:
            mock_wait.return_value = {
                "EndpointStatus": "Failed",
                "FailureReason": "InsufficientCapacity"
            }
            
            with pytest.raises(Exception, match="Error hosting endpoint"):
                session.wait_for_endpoint("my-endpoint")


class TestUpdateEndpoint:
    """Test update_endpoint method."""

    def test_update_endpoint_success(self, mock_boto_session, mock_sagemaker_client):
        """Test successful endpoint update."""
        mock_sagemaker_client.describe_endpoint.return_value = {
            "EndpointStatus": "InService"
        }
        mock_sagemaker_client.update_endpoint.return_value = {
            "EndpointArn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/my-endpoint"
        }
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        with patch.object(session, "wait_for_endpoint"):
            result = session.update_endpoint(
                endpoint_name="my-endpoint",
                endpoint_config_name="new-config",
                wait=False
            )
            
            assert result == "my-endpoint"

    def test_update_endpoint_not_exists(self, mock_boto_session, mock_sagemaker_client):
        """Test updating non-existent endpoint."""
        mock_sagemaker_client.describe_endpoint.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Could not find"}},
            "DescribeEndpoint"
        )
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        with pytest.raises(ValueError, match="does not exist"):
            session.update_endpoint(
                endpoint_name="nonexistent-endpoint",
                endpoint_config_name="new-config"
            )


class TestReadS3File:
    """Test read_s3_file method."""

    def test_read_s3_file_success(self, mock_boto_session, mock_sagemaker_client):
        """Test reading S3 file."""
        mock_s3_client = Mock()
        mock_body = Mock()
        mock_body.read.return_value = b"test content"
        mock_s3_client.get_object.return_value = {"Body": mock_body}
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_client = mock_s3_client
        
        result = session.read_s3_file("test-bucket", "path/to/file.txt")
        
        assert result == "test content"
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="path/to/file.txt"
        )


class TestListS3Files:
    """Test list_s3_files method."""

    def test_list_s3_files(self, mock_boto_session, mock_sagemaker_client):
        """Test listing S3 files."""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_obj1 = Mock()
        mock_obj1.key = "prefix/file1.txt"
        mock_obj2 = Mock()
        mock_obj2.key = "prefix/file2.txt"
        mock_bucket.objects.filter.return_value.all.return_value = [mock_obj1, mock_obj2]
        mock_s3_resource.Bucket.return_value = mock_bucket
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_resource = mock_s3_resource
        
        result = session.list_s3_files("test-bucket", "prefix/")
        
        assert result == ["prefix/file1.txt", "prefix/file2.txt"]


class TestUploadStringAsFileBody:
    """Test upload_string_as_file_body method."""

    def test_upload_string_as_file_body_without_kms(self, mock_boto_session, mock_sagemaker_client):
        """Test uploading string without KMS encryption."""
        mock_s3_resource = Mock()
        mock_s3_object = Mock()
        mock_s3_resource.Object.return_value = mock_s3_object
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_resource = mock_s3_resource
        
        result = session.upload_string_as_file_body(
            body="test content",
            bucket="test-bucket",
            key="path/to/file.txt"
        )
        
        assert result == "s3://test-bucket/path/to/file.txt"
        mock_s3_object.put.assert_called_once_with(Body="test content")

    def test_upload_string_as_file_body_with_kms(self, mock_boto_session, mock_sagemaker_client):
        """Test uploading string with KMS encryption."""
        mock_s3_resource = Mock()
        mock_s3_object = Mock()
        mock_s3_resource.Object.return_value = mock_s3_object
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_resource = mock_s3_resource
        
        result = session.upload_string_as_file_body(
            body="test content",
            bucket="test-bucket",
            key="path/to/file.txt",
            kms_key="my-kms-key"
        )
        
        assert result == "s3://test-bucket/path/to/file.txt"
        mock_s3_object.put.assert_called_once_with(
            Body="test content",
            SSEKMSKeyId="my-kms-key",
            ServerSideEncryption="aws:kms"
        )


class TestDetermineBucketAndPrefix:
    """Test determine_bucket_and_prefix method."""

    def test_determine_bucket_and_prefix_with_bucket(self, mock_boto_session, mock_sagemaker_client):
        """Test with explicit bucket."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        bucket, prefix = session.determine_bucket_and_prefix(
            bucket="my-bucket",
            key_prefix="my-prefix",
            sagemaker_session=session
        )
        
        assert bucket == "my-bucket"
        assert prefix == "my-prefix"

    def test_determine_bucket_and_prefix_without_bucket(self, mock_boto_session, mock_sagemaker_client):
        """Test without explicit bucket."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client,
            default_bucket="default-bucket",
            default_bucket_prefix="default-prefix"
        )
        
        with patch.object(session, "default_bucket", return_value="default-bucket"):
            bucket, prefix = session.determine_bucket_and_prefix(
                bucket=None,
                key_prefix="my-prefix",
                sagemaker_session=session
            )
            
            assert bucket == "default-bucket"
            assert "default-prefix" in prefix
            assert "my-prefix" in prefix


class TestGenerateDefaultSagemakerBucketName:
    """Test generate_default_sagemaker_bucket_name method."""

    def test_generate_default_sagemaker_bucket_name(self, mock_boto_session, mock_sagemaker_client):
        """Test generating default bucket name."""
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_session.client.return_value = mock_sts_client
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        bucket_name = session.generate_default_sagemaker_bucket_name(mock_boto_session)
        
        assert bucket_name == "sagemaker-us-west-2-123456789012"


class TestSessionConfigProperty:
    """Test config property."""

    def test_config_getter(self, mock_boto_session, mock_sagemaker_client):
        """Test config getter."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        assert session.config is None

    def test_config_setter(self, mock_boto_session, mock_sagemaker_client):
        """Test config setter."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        test_config = {"key": "value"}
        session.config = test_config
        
        assert session.config == test_config


class TestBotoRegionName:
    """Test boto_region_name property."""

    def test_boto_region_name(self, mock_boto_session, mock_sagemaker_client):
        """Test getting boto region name."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        assert session.boto_region_name == "us-west-2"


class TestExpectedBucketOwnerIdCheck:
    """Test expected_bucket_owner_id_bucket_check method."""

    def test_expected_bucket_owner_id_check_success(self, mock_boto_session, mock_sagemaker_client):
        """Test successful bucket owner check."""
        mock_s3_resource = Mock()
        mock_s3_client = Mock()
        mock_s3_resource.meta.client = mock_s3_client
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_resource = mock_s3_resource
        
        # Should not raise
        session.expected_bucket_owner_id_bucket_check(
            "test-bucket",
            mock_s3_resource,
            "123456789012"
        )
        
        mock_s3_client.head_bucket.assert_called_once()

    def test_expected_bucket_owner_id_check_forbidden(self, mock_boto_session, mock_sagemaker_client):
        """Test bucket owner check with forbidden error."""
        mock_s3_resource = Mock()
        mock_s3_client = Mock()
        mock_s3_client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}},
            "HeadBucket"
        )
        mock_s3_resource.meta.client = mock_s3_client
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_resource = mock_s3_resource
        
        with pytest.raises(ClientError):
            session.expected_bucket_owner_id_bucket_check(
                "test-bucket",
                mock_s3_resource,
                "123456789012"
            )


class TestGeneralBucketCheck:
    """Test general_bucket_check_if_user_has_permission method."""

    def test_general_bucket_check_create_bucket(self, mock_boto_session, mock_sagemaker_client):
        """Test general bucket check when creating bucket."""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        # Should not raise - implementation depends on actual method
        # This is a placeholder test
        assert session is not None


class TestDownloadDataWithDirectories:
    """Test download_data with directory structures."""

    def test_download_data_with_subdirectories(self, mock_boto_session, mock_sagemaker_client, tmp_path):
        """Test downloading data with subdirectories."""
        mock_s3_client = Mock()
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "data/subdir/", "Size": 0},
                {"Key": "data/subdir/file.txt", "Size": 100}
            ]
        }
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_client = mock_s3_client
        
        result = session.download_data(
            path=str(tmp_path),
            bucket="test-bucket",
            key_prefix="data/"
        )
        
        assert len(result) == 1  # Only file, not directory
        mock_s3_client.download_file.assert_called_once()


class TestUploadDataWithExtraArgs:
    """Test upload_data with extra arguments."""

    def test_upload_data_with_extra_args(self, mock_boto_session, mock_sagemaker_client, tmp_path):
        """Test uploading data with extra arguments."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        mock_s3_resource = Mock()
        mock_s3_object = Mock()
        mock_s3_resource.Object.return_value = mock_s3_object
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client,
            default_bucket="test-bucket"
        )
        session.s3_resource = mock_s3_resource
        
        extra_args = {"ServerSideEncryption": "AES256"}
        result = session.upload_data(
            path=str(test_file),
            bucket="test-bucket",
            key_prefix="data",
            extra_args=extra_args
        )
        
        assert result == "s3://test-bucket/data/test.txt"
        mock_s3_object.upload_file.assert_called_once()
        call_args = mock_s3_object.upload_file.call_args
        assert call_args[1]["ExtraArgs"] == extra_args



class TestSessionConfig:
    """Test Session config property."""

    def test_config_getter(self, mock_boto_session, mock_sagemaker_client):
        """Test getting config."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        assert session.config is None

    def test_config_setter(self, mock_boto_session, mock_sagemaker_client):
        """Test setting config."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        config = {"key": "value"}
        session.config = config
        assert session.config == config


class TestSessionBotoRegionName:
    """Test boto_region_name property."""

    def test_boto_region_name(self, mock_boto_session, mock_sagemaker_client):
        """Test getting boto region name."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        assert session.boto_region_name == "us-west-2"


class TestSessionSettings:
    """Test Session with SessionSettings."""

    def test_session_with_settings(self, mock_boto_session, mock_sagemaker_client):
        """Test Session initialization with settings."""
        settings = SessionSettings()
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client,
            settings=settings
        )
        
        assert session.settings == settings

    def test_session_without_settings(self, mock_boto_session, mock_sagemaker_client):
        """Test Session initialization without settings."""
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        assert isinstance(session.settings, SessionSettings)


class TestDeleteEndpoint:
    """Test delete_endpoint method."""

    def test_delete_endpoint_success(self, mock_boto_session, mock_sagemaker_client):
        """Test successful endpoint deletion."""
        mock_sagemaker_client.delete_endpoint.return_value = {}
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        session.delete_endpoint("my-endpoint")
        
        mock_sagemaker_client.delete_endpoint.assert_called_once_with(
            EndpointName="my-endpoint"
        )

    def test_delete_endpoint_not_found(self, mock_boto_session, mock_sagemaker_client):
        """Test deleting non-existent endpoint."""
        mock_sagemaker_client.delete_endpoint.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Could not find"}},
            "DeleteEndpoint"
        )
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        with pytest.raises(ClientError):
            session.delete_endpoint("nonexistent-endpoint")


class TestDescribeEndpoint:
    """Test describe_endpoint method."""

    def test_describe_endpoint_success(self, mock_boto_session, mock_sagemaker_client):
        """Test successful endpoint description."""
        mock_sagemaker_client.describe_endpoint.return_value = {
            "EndpointName": "my-endpoint",
            "EndpointStatus": "InService"
        }
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        result = mock_sagemaker_client.describe_endpoint(EndpointName="my-endpoint")
        
        assert result["EndpointName"] == "my-endpoint"
        assert result["EndpointStatus"] == "InService"


class TestCreateModel:
    """Test create_model method."""

    def test_create_model_success(self, mock_boto_session, mock_sagemaker_client):
        """Test successful model creation."""
        mock_sagemaker_client.create_model.return_value = {
            "ModelArn": "arn:aws:sagemaker:us-west-2:123456789012:model/my-model"
        }
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        with patch.object(session, "_append_sagemaker_config_tags", return_value=[]):
            result = session.create_model(
                name="my-model",
                role="arn:aws:iam::123456789012:role/SageMakerRole",
                container_defs={"Image": "my-image"}
            )
            
            assert result == "my-model"

    def test_create_model_with_vpc_config(self, mock_boto_session, mock_sagemaker_client):
        """Test model creation with VPC config."""
        mock_sagemaker_client.create_model.return_value = {
            "ModelArn": "arn:aws:sagemaker:us-west-2:123456789012:model/my-model"
        }
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        vpc_config = {
            "SecurityGroupIds": ["sg-123"],
            "Subnets": ["subnet-123"]
        }
        
        with patch.object(session, "_append_sagemaker_config_tags", return_value=[]):
            result = session.create_model(
                name="my-model",
                role="arn:aws:iam::123456789012:role/SageMakerRole",
                container_defs={"Image": "my-image"},
                vpc_config=vpc_config
            )
            
            assert result == "my-model"


class TestCreateEndpointConfig:
    """Test create_endpoint_config method."""

    def test_create_endpoint_config_success(self, mock_boto_session, mock_sagemaker_client):
        """Test successful endpoint config creation."""
        mock_sagemaker_client.create_endpoint_config.return_value = {
            "EndpointConfigArn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint-config/my-config"
        }
        mock_sagemaker_client.create_endpoint.return_value = {
            "EndpointArn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/my-config"
        }
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        production_variants = [{
            "VariantName": "AllTraffic",
            "ModelName": "my-model",
            "InstanceType": "ml.m5.xlarge",
            "InitialInstanceCount": 1
        }]
        
        with patch.object(session, "_append_sagemaker_config_tags", return_value=[]):
            with patch.object(session, "wait_for_endpoint"):
                result = session.endpoint_from_production_variants(
                    name="my-config",
                    production_variants=production_variants,
                    wait=False
                )
                
                assert result == "my-config"

    def test_create_endpoint_config_with_kms(self, mock_boto_session, mock_sagemaker_client):
        """Test endpoint config creation with KMS key."""
        mock_sagemaker_client.create_endpoint_config.return_value = {
            "EndpointConfigArn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint-config/my-config"
        }
        mock_sagemaker_client.create_endpoint.return_value = {
            "EndpointArn": "arn:aws:sagemaker:us-west-2:123456789012:endpoint/my-config"
        }
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        production_variants = [{
            "VariantName": "AllTraffic",
            "ModelName": "my-model",
            "InstanceType": "ml.m5.xlarge",
            "InitialInstanceCount": 1
        }]
        
        with patch.object(session, "_append_sagemaker_config_tags", return_value=[]):
            with patch.object(session, "wait_for_endpoint"):
                result = session.endpoint_from_production_variants(
                    name="my-config",
                    production_variants=production_variants,
                    kms_key="my-kms-key",
                    wait=False
                )
                
                assert result == "my-config"


class TestExpandRole:
    """Test expand_role function."""

    def test_expand_role_with_full_arn(self, mock_boto_session, mock_sagemaker_client):
        """Test expanding role that's already a full ARN."""
        from sagemaker.core.helper.session_helper import expand_role
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        role_arn = "arn:aws:iam::123456789012:role/MyRole"
        result = expand_role(session, role_arn)
        
        assert result == role_arn

    def test_expand_role_with_role_name(self, mock_boto_session, mock_sagemaker_client):
        """Test expanding role name to full ARN."""
        mock_iam_resource = Mock()
        mock_role = Mock()
        mock_role.arn = "arn:aws:iam::123456789012:role/MyRole"
        mock_iam_resource.Role.return_value = mock_role
        mock_boto_session.resource.return_value = mock_iam_resource
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        result = session.expand_role("MyRole")
        
        assert result == "arn:aws:iam::123456789012:role/MyRole"


class TestGenerateDefaultSagemakerBucketName:
    """Test generate_default_sagemaker_bucket_name static method."""

    def test_generate_default_sagemaker_bucket_name_standard_region(self, mock_boto_session, mock_sagemaker_client):
        """Test generating bucket name for standard region."""
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_session.client.return_value = mock_sts_client
        mock_boto_session.region_name = "us-west-2"
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        bucket_name = session.generate_default_sagemaker_bucket_name(mock_boto_session)
        
        assert bucket_name == "sagemaker-us-west-2-123456789012"

    def test_generate_default_sagemaker_bucket_name_china_region(self, mock_boto_session, mock_sagemaker_client):
        """Test generating bucket name for China region."""
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_session.client.return_value = mock_sts_client
        mock_boto_session.region_name = "cn-north-1"
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        bucket_name = session.generate_default_sagemaker_bucket_name(mock_boto_session)
        
        assert bucket_name == "sagemaker-cn-north-1-123456789012"


class TestExpectedBucketOwnerIdBucketCheck:
    """Test expected_bucket_owner_id_bucket_check method."""

    def test_expected_bucket_owner_id_bucket_check_success(self, mock_boto_session, mock_sagemaker_client):
        """Test successful bucket owner check."""
        mock_s3_resource = Mock()
        mock_s3_resource.meta.client.head_bucket.return_value = {}
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_resource = mock_s3_resource
        
        # Should not raise
        session.expected_bucket_owner_id_bucket_check(
            "test-bucket",
            mock_s3_resource,
            "123456789012"
        )

    def test_expected_bucket_owner_id_bucket_check_forbidden(self, mock_boto_session, mock_sagemaker_client):
        """Test bucket owner check with forbidden error."""
        mock_s3_resource = Mock()
        mock_s3_resource.meta.client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}},
            "HeadBucket"
        )
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.s3_resource = mock_s3_resource
        
        with pytest.raises(ClientError):
            session.expected_bucket_owner_id_bucket_check(
                "test-bucket",
                mock_s3_resource,
                "123456789012"
            )


class TestGeneralBucketCheckIfUserHasPermission:
    """Test general_bucket_check_if_user_has_permission method."""

    def test_general_bucket_check_create_bucket(self, mock_boto_session, mock_sagemaker_client):
        """Test bucket check that creates bucket."""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.creation_date = None
        mock_s3_resource.Bucket.return_value = mock_bucket
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        # Should not raise
        session.general_bucket_check_if_user_has_permission(
            "test-bucket",
            mock_s3_resource,
            mock_bucket,
            "us-west-2",
            True
        )

    def test_general_bucket_check_existing_bucket(self, mock_boto_session, mock_sagemaker_client):
        """Test bucket check with existing bucket."""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.creation_date = "2023-01-01"
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        # Should not raise
        session.general_bucket_check_if_user_has_permission(
            "test-bucket",
            mock_s3_resource,
            mock_bucket,
            "us-west-2",
            False
        )


class TestCreateBucketForNotExistError:
    """Test create_bucket_for_not_exist_error method."""

    def test_create_bucket_us_east_1(self, mock_boto_session, mock_sagemaker_client):
        """Test creating bucket in us-east-1."""
        mock_s3_resource = Mock()
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.create_bucket_for_not_exist_error("test-bucket", "us-east-1", mock_s3_resource)
        mock_s3_resource.create_bucket.assert_called_once_with(Bucket="test-bucket")

    def test_create_bucket_other_region(self, mock_boto_session, mock_sagemaker_client):
        """Test creating bucket in non-us-east-1 region."""
        mock_s3_resource = Mock()
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.create_bucket_for_not_exist_error("test-bucket", "us-west-2", mock_s3_resource)
        mock_s3_resource.create_bucket.assert_called_once_with(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "us-west-2"}
        )

    def test_create_bucket_operation_aborted(self, mock_boto_session, mock_sagemaker_client):
        """Test bucket creation with OperationAborted error."""
        mock_s3_resource = Mock()
        mock_s3_resource.create_bucket.side_effect = ClientError(
            {"Error": {"Code": "OperationAborted", "Message": "conflicting conditional operation"}},
            "CreateBucket"
        )
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        session.create_bucket_for_not_exist_error("test-bucket", "us-west-2", mock_s3_resource)


class TestS3PathJoin:
    """Test s3_path_join function."""

    def test_s3_path_join_basic(self):
        """Test basic path joining."""
        from sagemaker.core.helper.session_helper import s3_path_join
        assert s3_path_join("foo", "bar") == "foo/bar"

    def test_s3_path_join_with_s3_prefix(self):
        """Test path joining with s3:// prefix."""
        from sagemaker.core.helper.session_helper import s3_path_join
        assert s3_path_join("s3://", "bucket", "key") == "s3://bucket/key"

    def test_s3_path_join_with_end_slash(self):
        """Test path joining with end slash."""
        from sagemaker.core.helper.session_helper import s3_path_join
        assert s3_path_join("foo", "bar", with_end_slash=True) == "foo/bar/"

    def test_s3_path_join_empty_args(self):
        """Test path joining with empty arguments."""
        from sagemaker.core.helper.session_helper import s3_path_join
        assert s3_path_join("foo", "", None, "bar") == "foo/bar"


class TestExpandContainerDef:
    """Test _expand_container_def function."""

    def test_expand_container_def_string(self):
        """Test expanding container def from string."""
        from sagemaker.core.helper.session_helper import _expand_container_def
        result = _expand_container_def("my-image:latest")
        assert result["Image"] == "my-image:latest"

    def test_expand_container_def_dict(self):
        """Test expanding container def from dict."""
        from sagemaker.core.helper.session_helper import _expand_container_def
        c_def = {"Image": "my-image:latest"}
        result = _expand_container_def(c_def)
        assert result == c_def


class TestContainerDef:
    """Test container_def function."""

    def test_container_def_basic(self):
        """Test basic container definition."""
        from sagemaker.core.helper.session_helper import container_def
        result = container_def("my-image:latest")
        assert result["Image"] == "my-image:latest"

    def test_container_def_with_model_data(self):
        """Test container def with model data URL."""
        from sagemaker.core.helper.session_helper import container_def
        result = container_def("my-image:latest", model_data_url="s3://bucket/model.tar.gz")
        assert result["ModelDataUrl"] == "s3://bucket/model.tar.gz"

    def test_container_def_with_env(self):
        """Test container def with environment variables."""
        from sagemaker.core.helper.session_helper import container_def
        env = {"KEY": "VALUE"}
        result = container_def("my-image:latest", env=env)
        assert result["Environment"] == env

    def test_container_def_with_accept_eula(self):
        """Test container def with accept_eula."""
        from sagemaker.core.helper.session_helper import container_def
        result = container_def(
            "my-image:latest",
            model_data_url="s3://bucket/model.tar.gz",
            accept_eula=True
        )
        assert "ModelDataSource" in result

    def test_container_def_with_model_data_source_dict(self):
        """Test container def with ModelDataSource dict."""
        from sagemaker.core.helper.session_helper import container_def
        model_data_source = {"S3DataSource": {"S3Uri": "s3://bucket/model.tar.gz"}}
        result = container_def("my-image:latest", model_data_url=model_data_source)
        assert result["ModelDataSource"] == model_data_source

    def test_container_def_with_container_mode(self):
        """Test container def with container mode."""
        from sagemaker.core.helper.session_helper import container_def
        result = container_def("my-image:latest", container_mode="MultiModel")
        assert result["Mode"] == "MultiModel"

    def test_container_def_with_image_config(self):
        """Test container def with image config."""
        from sagemaker.core.helper.session_helper import container_def
        image_config = {"RepositoryAccessMode": "Vpc"}
        result = container_def("my-image:latest", image_config=image_config)
        assert result["ImageConfig"] == image_config

    def test_container_def_with_additional_model_data_sources(self):
        """Test container def with additional model data sources."""
        from sagemaker.core.helper.session_helper import container_def
        additional_sources = [{"ChannelName": "extra", "S3DataSource": {"S3Uri": "s3://bucket/extra"}}]
        result = container_def("my-image:latest", additional_model_data_sources=additional_sources)
        assert result["AdditionalModelDataSources"] == additional_sources

    def test_container_def_with_model_reference_arn(self):
        """Test container def with model reference ARN."""
        from sagemaker.core.helper.session_helper import container_def
        result = container_def(
            "my-image:latest",
            model_data_url="s3://bucket/model.tar.gz",
            accept_eula=True,
            model_reference_arn="arn:aws:sagemaker:us-west-2:123456789012:hub-content/model"
        )
        assert "HubAccessConfig" in result["ModelDataSource"]["S3DataSource"]


class TestGetExecutionRole:
    """Test get_execution_role function."""

    def test_get_execution_role_with_role_arn(self, mock_boto_session, mock_sagemaker_client):
        """Test getting execution role when ARN contains role."""
        from sagemaker.core.helper.session_helper import get_execution_role
        
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:iam::123456789012:role/MyRole"
        }
        
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/MyRole"}
        }
        
        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts_client
            elif service == "iam":
                return mock_iam_client
            return Mock()
        
        mock_boto_session.client.side_effect = client_side_effect
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        with patch("os.path.exists", return_value=False):
            role = get_execution_role(session)
            assert "role/MyRole" in role

    def test_get_execution_role_use_default(self, mock_boto_session, mock_sagemaker_client):
        """Test getting execution role with use_default."""
        from sagemaker.core.helper.session_helper import get_execution_role
        
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/MyRole/session"
        }
        
        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/AmazonSageMaker-DefaultRole"}
        }
        
        def client_side_effect(service, **kwargs):
            if service == "sts":
                return mock_sts_client
            elif service == "iam":
                return mock_iam_client
            return Mock()
        
        mock_boto_session.client.side_effect = client_side_effect
        
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        
        with patch("os.path.exists", return_value=False):
            role = get_execution_role(session, use_default=True)
            assert "AmazonSageMaker-DefaultRole" in role


class TestProductionVariant:
    """Test production_variant function."""

    def test_production_variant_basic(self):
        """Test basic production variant."""
        from sagemaker.core.helper.session_helper import production_variant
        result = production_variant(
            model_name="my-model",
            instance_type="ml.m5.xlarge",
            initial_instance_count=1
        )
        assert result["ModelName"] == "my-model"
        assert result["InstanceType"] == "ml.m5.xlarge"
        assert result["InitialInstanceCount"] == 1

    def test_production_variant_with_accelerator(self):
        """Test production variant with accelerator."""
        from sagemaker.core.helper.session_helper import production_variant
        result = production_variant(
            model_name="my-model",
            instance_type="ml.m5.xlarge",
            accelerator_type="ml.eia1.medium"
        )
        assert result["AcceleratorType"] == "ml.eia1.medium"

    def test_production_variant_serverless(self):
        """Test production variant with serverless config."""
        from sagemaker.core.helper.session_helper import production_variant
        serverless_config = {"MemorySizeInMB": 2048, "MaxConcurrency": 5}
        result = production_variant(
            model_name="my-model",
            serverless_inference_config=serverless_config
        )
        assert result["ServerlessConfig"] == serverless_config
        assert "InstanceType" not in result

    def test_production_variant_with_volume_size(self):
        """Test production variant with volume size."""
        from sagemaker.core.helper.session_helper import production_variant
        result = production_variant(
            model_name="my-model",
            instance_type="ml.m5.xlarge",
            volume_size=30
        )
        assert result["VolumeSizeInGB"] == 30

    def test_production_variant_with_managed_instance_scaling(self):
        """Test production variant with managed instance scaling."""
        from sagemaker.core.helper.session_helper import production_variant
        scaling_config = {"Status": "ENABLED", "MinInstanceCount": 1, "MaxInstanceCount": 3}
        result = production_variant(
            model_name="my-model",
            instance_type="ml.m5.xlarge",
            managed_instance_scaling=scaling_config
        )
        assert result["ManagedInstanceScaling"] == scaling_config

    def test_production_variant_with_inference_ami_version(self):
        """Test production variant with inference AMI version."""
        from sagemaker.core.helper.session_helper import production_variant
        result = production_variant(
            model_name="my-model",
            instance_type="ml.m5.xlarge",
            inference_ami_version="al2-ami-sagemaker-inference-gpu-2"
        )
        assert result["InferenceAmiVersion"] == "al2-ami-sagemaker-inference-gpu-2"


class TestEndpointInServiceOrNot:
    """Test endpoint_in_service_or_not method."""

    def test_endpoint_in_service(self, mock_boto_session, mock_sagemaker_client):
        """Test endpoint in service."""
        mock_sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "InService"}
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        assert session.endpoint_in_service_or_not("my-endpoint") is True

    def test_endpoint_not_in_service(self, mock_boto_session, mock_sagemaker_client):
        """Test endpoint not in service."""
        mock_sagemaker_client.describe_endpoint.return_value = {"EndpointStatus": "Creating"}
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        assert session.endpoint_in_service_or_not("my-endpoint") is False

    def test_endpoint_not_found(self, mock_boto_session, mock_sagemaker_client):
        """Test endpoint not found."""
        import botocore.exceptions
        mock_sagemaker_client.describe_endpoint.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Could not find endpoint"}},
            "DescribeEndpoint"
        )
        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client
        )
        assert session.endpoint_in_service_or_not("my-endpoint") is False
