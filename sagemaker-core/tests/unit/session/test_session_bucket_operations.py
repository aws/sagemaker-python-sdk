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

import pytest
from unittest.mock import Mock, patch
from botocore.exceptions import ClientError

from sagemaker.core.helper.session_helper import Session


class TestSessionBucketOperations:
    """Test cases for Session bucket operations"""

    @pytest.fixture
    def mock_boto_session(self):
        """Mock boto3 session"""
        mock_session = Mock()
        mock_session.region_name = "us-west-2"
        mock_session.client.return_value = Mock()
        mock_session.resource.return_value = Mock()
        return mock_session

    def test_create_s3_bucket_if_it_does_not_exist_bucket_exists(self, mock_boto_session):
        """Test _create_s3_bucket_if_it_does_not_exist when bucket exists"""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.creation_date = "2023-01-01"  # Bucket exists
        mock_s3_resource.Bucket.return_value = mock_bucket
        mock_boto_session.resource.return_value = mock_s3_resource

        session = Session(boto_session=mock_boto_session)
        session._create_s3_bucket_if_it_does_not_exist("test-bucket", "us-west-2")

        # Should not attempt to create bucket
        mock_s3_resource.create_bucket.assert_not_called()

    def test_create_s3_bucket_if_it_does_not_exist_bucket_not_exists(self, mock_boto_session):
        """Test _create_s3_bucket_if_it_does_not_exist when bucket doesn't exist"""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.creation_date = None  # Bucket doesn't exist
        mock_s3_resource.Bucket.return_value = mock_bucket
        mock_s3_resource.meta.client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
        )
        mock_boto_session.resource.return_value = mock_s3_resource

        session = Session(boto_session=mock_boto_session)
        session._create_s3_bucket_if_it_does_not_exist("test-bucket", "us-west-2")

        # Should attempt to create bucket
        mock_s3_resource.create_bucket.assert_called_once()

    def test_create_s3_bucket_if_it_does_not_exist_us_east_1(self, mock_boto_session):
        """Test bucket creation in us-east-1 region"""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.creation_date = None
        mock_s3_resource.Bucket.return_value = mock_bucket
        mock_s3_resource.meta.client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
        )
        mock_boto_session.resource.return_value = mock_s3_resource

        session = Session(boto_session=mock_boto_session)
        session._create_s3_bucket_if_it_does_not_exist("test-bucket", "us-east-1")

        # Should create bucket without LocationConstraint for us-east-1
        mock_s3_resource.create_bucket.assert_called_with(Bucket="test-bucket")

    def test_create_s3_bucket_if_it_does_not_exist_other_region(self, mock_boto_session):
        """Test bucket creation in non-us-east-1 region"""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.creation_date = None
        mock_s3_resource.Bucket.return_value = mock_bucket
        mock_s3_resource.meta.client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
        )
        mock_boto_session.resource.return_value = mock_s3_resource

        session = Session(boto_session=mock_boto_session)
        session._create_s3_bucket_if_it_does_not_exist("test-bucket", "eu-west-1")

        # Should create bucket with LocationConstraint for non-us-east-1
        mock_s3_resource.create_bucket.assert_called_with(
            Bucket="test-bucket", CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
        )

    def test_expected_bucket_owner_id_bucket_check_success(self, mock_boto_session):
        """Test expected_bucket_owner_id_bucket_check with correct owner"""
        mock_s3_resource = Mock()
        mock_s3_resource.meta.client.head_bucket.return_value = None  # Success

        session = Session(boto_session=mock_boto_session)
        # Should not raise exception
        session.expected_bucket_owner_id_bucket_check(
            "test-bucket", mock_s3_resource, "123456789012"
        )

    def test_expected_bucket_owner_id_bucket_check_forbidden(self, mock_boto_session):
        """Test expected_bucket_owner_id_bucket_check with wrong owner"""
        mock_s3_resource = Mock()
        mock_s3_resource.meta.client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadBucket"
        )

        session = Session(boto_session=mock_boto_session)

        with pytest.raises(ClientError):
            session.expected_bucket_owner_id_bucket_check(
                "test-bucket", mock_s3_resource, "123456789012"
            )

    def test_general_bucket_check_if_user_has_permission_success(self, mock_boto_session):
        """Test general_bucket_check_if_user_has_permission with valid permissions"""
        mock_s3_resource = Mock()
        mock_s3_resource.meta.client.head_bucket.return_value = None  # Success

        session = Session(boto_session=mock_boto_session)
        # Should not raise exception
        session.general_bucket_check_if_user_has_permission(
            "test-bucket", mock_s3_resource, Mock(), "us-west-2", False
        )

    def test_general_bucket_check_if_user_has_permission_forbidden(self, mock_boto_session):
        """Test general_bucket_check_if_user_has_permission with forbidden access"""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.name = "test-bucket"
        mock_s3_resource.meta.client.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadBucket"
        )

        session = Session(boto_session=mock_boto_session)

        with pytest.raises(ClientError):
            session.general_bucket_check_if_user_has_permission(
                "test-bucket", mock_s3_resource, mock_bucket, "us-west-2", True
            )

    def test_create_bucket_for_not_exist_error_operation_aborted(self, mock_boto_session):
        """Test create_bucket_for_not_exist_error with OperationAborted error"""
        mock_s3_resource = Mock()
        mock_s3_resource.create_bucket.side_effect = ClientError(
            {"Error": {"Code": "OperationAborted", "Message": "conflicting conditional operation"}},
            "CreateBucket",
        )

        session = Session(boto_session=mock_boto_session)
        # Should not raise exception for OperationAborted with conflicting operation
        session.create_bucket_for_not_exist_error("test-bucket", "us-west-2", mock_s3_resource)

    def test_create_bucket_for_not_exist_error_other_error(self, mock_boto_session):
        """Test create_bucket_for_not_exist_error with other errors"""
        mock_s3_resource = Mock()
        mock_s3_resource.create_bucket.side_effect = ClientError(
            {"Error": {"Code": "InvalidBucketName", "Message": "Invalid bucket name"}},
            "CreateBucket",
        )

        session = Session(boto_session=mock_boto_session)

        with pytest.raises(ClientError):
            session.create_bucket_for_not_exist_error("test-bucket", "us-west-2", mock_s3_resource)

    def test_generate_default_sagemaker_bucket_name(self, mock_boto_session):
        """Test generate_default_sagemaker_bucket_name"""
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_session.client.return_value = mock_sts_client
        mock_boto_session.region_name = "us-west-2"

        with patch(
            "sagemaker.core.helper.session_helper.sts_regional_endpoint",
            return_value="https://sts.us-west-2.amazonaws.com",
        ):
            session = Session(boto_session=mock_boto_session)
            bucket_name = session.generate_default_sagemaker_bucket_name(mock_boto_session)

        assert bucket_name == "sagemaker-us-west-2-123456789012"

    def test_default_bucket_with_override(self, mock_boto_session):
        """Test default_bucket with name override"""
        session = Session(boto_session=mock_boto_session, default_bucket="custom-bucket")
        session._default_bucket = "custom-bucket"

        result = session.default_bucket()
        assert result == "custom-bucket"

    def test_default_bucket_sdk_generated_with_owner_check(self, mock_boto_session):
        """Test default_bucket with SDK-generated name and owner check"""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.creation_date = "2023-01-01"  # Bucket exists
        mock_s3_resource.Bucket.return_value = mock_bucket
        mock_s3_resource.meta.client.head_bucket.return_value = None  # Success
        mock_boto_session.resource.return_value = mock_s3_resource

        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_session.client.return_value = mock_sts_client

        with patch(
            "sagemaker.core.helper.session_helper.sts_regional_endpoint",
            return_value="https://sts.us-west-2.amazonaws.com",
        ):
            session = Session(boto_session=mock_boto_session)
            session._default_bucket_set_by_sdk = True
            result = session.default_bucket()

        assert result == "sagemaker-us-west-2-123456789012"
        # Should check bucket ownership
        mock_s3_resource.meta.client.head_bucket.assert_called()
