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

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from botocore.exceptions import ClientError

from sagemaker.core.helper.session_helper import (
    Session,
    s3_path_join,
    botocore_resolver,
    sts_regional_endpoint,
    get_execution_role,
    get_add_model_package_inference_args,
    get_update_model_package_inference_args,
    production_variant,
    update_args,
    NOTEBOOK_METADATA_FILE,
)
from sagemaker.core.config.config_schema import (
    FEATURE_GROUP_ROLE_ARN_PATH,
    FEATURE_GROUP_ONLINE_STORE_CONFIG_PATH,
    FEATURE_GROUP_OFFLINE_STORE_CONFIG_PATH,
)


class TestSession:
    """Test cases for Session class"""

    @pytest.fixture
    def mock_boto_session(self):
        """Mock boto3 session"""
        mock_session = Mock()
        mock_session.region_name = "us-west-2"
        mock_session.client.return_value = Mock()
        mock_session.resource.return_value = Mock()
        return mock_session

    def test_session_init_with_defaults(self, mock_boto_session):
        """Test Session initialization with default parameters"""
        session = Session(boto_session=mock_boto_session)
        assert session.boto_session == mock_boto_session
        assert session._region_name == "us-west-2"
        assert session.local_mode is False

    def test_session_init_with_custom_clients(self, mock_boto_session):
        """Test Session initialization with custom clients"""
        mock_sagemaker_client = Mock()
        mock_runtime_client = Mock()

        session = Session(
            boto_session=mock_boto_session,
            sagemaker_client=mock_sagemaker_client,
            sagemaker_runtime_client=mock_runtime_client,
        )

        assert session.sagemaker_client == mock_sagemaker_client
        assert session.sagemaker_runtime_client == mock_runtime_client

    def test_session_init_no_region_raises_error(self):
        """Test Session initialization fails without region"""
        mock_session = Mock()
        mock_session.region_name = None

        with pytest.raises(ValueError, match="Must setup local AWS configuration"):
            Session(boto_session=mock_session)

    def test_account_id(self, mock_boto_session):
        """Test account_id method"""
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_session.client.return_value = mock_sts_client

        with patch(
            "sagemaker.core.helper.session_helper.sts_regional_endpoint",
            return_value="https://sts.us-west-2.amazonaws.com",
        ):
            session = Session(boto_session=mock_boto_session)
            account_id = session.account_id()

        assert account_id == "123456789012"

    def test_boto_region_name_property(self, mock_boto_session):
        """Test boto_region_name property"""
        session = Session(boto_session=mock_boto_session)
        assert session.boto_region_name == "us-west-2"

    def test_config_property(self, mock_boto_session):
        """Test config property getter and setter"""
        session = Session(boto_session=mock_boto_session)

        # Test default value
        assert session.config is None

        # Test setter
        test_config = {"test": "value"}
        session.config = test_config
        assert session.config == test_config

    @pytest.mark.skip(reason="Complex mocking with config file loading - skipping")
    def test_get_caller_identity_arn_from_notebook_metadata(self, mock_boto_session):
        """Test get_caller_identity_arn from notebook metadata file"""
        pass

    @patch("os.path.exists")
    def test_get_caller_identity_arn_from_sts(self, mock_exists, mock_boto_session):
        """Test get_caller_identity_arn from STS when no metadata file"""
        mock_exists.return_value = False

        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {
            "Arn": "arn:aws:sts::123456789012:assumed-role/TestRole/session"
        }

        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/TestRole"}
        }

        def mock_client(service, **kwargs):
            if service == "sts":
                return mock_sts_client
            elif service == "iam":
                return mock_iam_client
            return Mock()

        mock_boto_session.client = mock_client

        with patch(
            "sagemaker.core.helper.session_helper.sts_regional_endpoint",
            return_value="https://sts.us-west-2.amazonaws.com",
        ):
            session = Session(boto_session=mock_boto_session)
            arn = session.get_caller_identity_arn()

        assert arn == "arn:aws:iam::123456789012:role/TestRole"

    def test_upload_data_single_file(self, mock_boto_session):
        """Test upload_data with single file"""
        mock_s3_resource = Mock()
        mock_s3_object = Mock()
        mock_s3_resource.Object.return_value = mock_s3_object
        mock_boto_session.resource.return_value = mock_s3_resource

        session = Session(boto_session=mock_boto_session)
        session._default_bucket = "test-bucket"

        with (
            patch("os.path.isdir", return_value=False),
            patch("os.path.split", return_value=("/path", "file.txt")),
        ):

            result = session.upload_data("/path/file.txt")

        assert result == "s3://test-bucket/data/file.txt"
        mock_s3_object.upload_file.assert_called_once()

    def test_read_s3_file(self, mock_boto_session):
        """Test read_s3_file method"""
        mock_s3_client = Mock()
        mock_body = Mock()
        mock_body.read.return_value = b"test content"
        mock_s3_client.get_object.return_value = {"Body": mock_body}
        mock_boto_session.client.return_value = mock_s3_client

        session = Session(boto_session=mock_boto_session)
        content = session.read_s3_file("test-bucket", "test-key")

        assert content == "test content"
        mock_s3_client.get_object.assert_called_with(Bucket="test-bucket", Key="test-key")

    def test_default_bucket_existing(self, mock_boto_session):
        """Test default_bucket when bucket already set"""
        session = Session(boto_session=mock_boto_session)
        session._default_bucket = "existing-bucket"

        result = session.default_bucket()
        assert result == "existing-bucket"

    def test_default_bucket_create_new(self, mock_boto_session):
        """Test default_bucket creates new bucket"""
        mock_s3_resource = Mock()
        mock_bucket = Mock()
        mock_bucket.creation_date = None
        mock_s3_resource.Bucket.return_value = mock_bucket
        mock_boto_session.resource.return_value = mock_s3_resource

        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_session.client.return_value = mock_sts_client

        with patch(
            "sagemaker.core.helper.session_helper.sts_regional_endpoint",
            return_value="https://sts.us-west-2.amazonaws.com",
        ):
            session = Session(boto_session=mock_boto_session)
            result = session.default_bucket()

        assert result == "sagemaker-us-west-2-123456789012"

    def test_determine_bucket_and_prefix_with_bucket(self, mock_boto_session):
        """Test determine_bucket_and_prefix with provided bucket"""
        session = Session(boto_session=mock_boto_session)

        bucket, prefix = session.determine_bucket_and_prefix(
            bucket="custom-bucket", key_prefix="custom-prefix"
        )

        assert bucket == "custom-bucket"
        assert prefix == "custom-prefix"

    def test_determine_bucket_and_prefix_default(self, mock_boto_session):
        """Test determine_bucket_and_prefix with defaults"""
        session = Session(boto_session=mock_boto_session)
        session._default_bucket = "default-bucket"
        session.default_bucket_prefix = "default-prefix"

        bucket, prefix = session.determine_bucket_and_prefix(sagemaker_session=session)

        assert bucket == "default-bucket"
        assert prefix == "default-prefix"


class TestS3PathJoin:
    """Test cases for s3_path_join function"""

    def test_basic_join(self):
        """Test basic path joining"""
        result = s3_path_join("foo", "bar", "baz")
        assert result == "foo/bar/baz"

    def test_with_s3_prefix(self):
        """Test joining with s3:// prefix"""
        result = s3_path_join("s3://", "bucket", "key")
        assert result == "s3://bucket/key"

    def test_with_slashes(self):
        """Test joining with existing slashes"""
        result = s3_path_join("/foo/", "/bar/", "/baz/")
        assert result == "foo/bar/baz"

    def test_with_end_slash(self):
        """Test joining with end slash option"""
        result = s3_path_join("foo", "bar", with_end_slash=True)
        assert result == "foo/bar/"

    def test_empty_args(self):
        """Test joining with empty arguments"""
        result = s3_path_join("foo", "", None, "bar")
        assert result == "foo/bar"

    def test_duplicate_slashes(self):
        """Test removal of duplicate slashes"""
        result = s3_path_join("s3://", "//bucket//", "///key///")
        assert result == "s3://bucket/key"


class TestHelperFunctions:
    """Test cases for helper functions"""

    def test_botocore_resolver(self):
        """Test botocore_resolver function"""
        with (
            patch("botocore.loaders.create_loader") as mock_loader,
            patch("botocore.regions.EndpointResolver") as mock_resolver,
        ):

            mock_loader_instance = Mock()
            mock_loader.return_value = mock_loader_instance

            result = botocore_resolver()

            mock_loader.assert_called_once()
            mock_resolver.assert_called_once_with(mock_loader_instance.load_data.return_value)

    def test_sts_regional_endpoint_normal_region(self):
        """Test sts_regional_endpoint for normal region"""
        with patch("sagemaker.core.helper.session_helper.botocore_resolver") as mock_resolver:
            mock_resolver_instance = Mock()
            mock_resolver_instance.construct_endpoint.return_value = {
                "hostname": "sts.us-west-2.amazonaws.com"
            }
            mock_resolver.return_value = mock_resolver_instance

            result = sts_regional_endpoint("us-west-2")

            assert result == "https://sts.us-west-2.amazonaws.com"

    def test_sts_regional_endpoint_il_central(self):
        """Test sts_regional_endpoint for il-central-1 region"""
        with patch("sagemaker.core.helper.session_helper.botocore_resolver") as mock_resolver:
            mock_resolver_instance = Mock()
            mock_resolver_instance.construct_endpoint.return_value = None
            mock_resolver.return_value = mock_resolver_instance

            result = sts_regional_endpoint("il-central-1")

            assert result == "https://sts.il-central-1.amazonaws.com"

    def test_get_execution_role_with_role_arn(self):
        """Test get_execution_role with valid role ARN"""
        mock_session = Mock()
        mock_session.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/TestRole"
        )

        result = get_execution_role(mock_session)
        assert result == "arn:aws:iam::123456789012:role/TestRole"

    def test_get_execution_role_with_user_arn_raises_error(self):
        """Test get_execution_role with user ARN raises ValueError"""
        mock_session = Mock()
        mock_session.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:user/TestUser"
        )

        with pytest.raises(ValueError, match="The current AWS identity is not a role"):
            get_execution_role(mock_session)

    def test_get_execution_role_use_default(self):
        """Test get_execution_role with use_default=True"""
        mock_session = Mock()
        mock_session.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:user/TestUser"
        )

        mock_iam_client = Mock()
        mock_iam_client.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123456789012:role/AmazonSageMaker-DefaultRole"}
        }
        mock_boto_session = Mock()
        mock_boto_session.client.return_value = mock_iam_client
        mock_session.boto_session = mock_boto_session

        result = get_execution_role(mock_session, use_default=True)
        assert result == "arn:aws:iam::123456789012:role/AmazonSageMaker-DefaultRole"


class TestModelPackageHelpers:
    """Test cases for model package helper functions"""

    def test_get_add_model_package_inference_args(self):
        """Test get_add_model_package_inference_args function"""
        result = get_add_model_package_inference_args(
            model_package_arn="arn:aws:sagemaker:us-west-2:123456789012:model-package/test",
            name="test-inference",
            containers=[{"Image": "test-image"}],
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            description="Test inference spec",
        )

        assert "AdditionalInferenceSpecificationsToAdd" in result
        assert (
            result["ModelPackageArn"]
            == "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        )

        inference_spec = result["AdditionalInferenceSpecificationsToAdd"][0]
        assert inference_spec["Name"] == "test-inference"
        assert inference_spec["Description"] == "Test inference spec"
        assert inference_spec["Containers"] == [{"Image": "test-image"}]

    def test_get_update_model_package_inference_args(self):
        """Test get_update_model_package_inference_args function"""
        result = get_update_model_package_inference_args(
            model_package_arn="arn:aws:sagemaker:us-west-2:123456789012:model-package/test",
            containers=[{"Image": "test-image"}],
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
        )

        assert "InferenceSpecification" in result
        assert (
            result["ModelPackageArn"]
            == "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        )

        inference_spec = result["InferenceSpecification"]
        assert inference_spec["Containers"] == [{"Image": "test-image"}]
        assert inference_spec["SupportedContentTypes"] == ["application/json"]


class TestProductionVariant:
    """Test cases for production_variant function"""

    def test_production_variant_basic(self):
        """Test basic production variant creation"""
        result = production_variant(
            model_name="test-model", instance_type="ml.m5.large", initial_instance_count=1
        )

        assert result["ModelName"] == "test-model"
        assert result["InstanceType"] == "ml.m5.large"
        assert result["InitialInstanceCount"] == 1
        assert result["VariantName"] == "AllTraffic"
        assert result["InitialVariantWeight"] == 1

    def test_production_variant_with_serverless(self):
        """Test production variant with serverless config"""
        serverless_config = {"MemorySizeInMB": 2048, "MaxConcurrency": 5}

        result = production_variant(
            model_name="test-model", serverless_inference_config=serverless_config
        )

        assert result["ModelName"] == "test-model"
        assert result["ServerlessConfig"] == serverless_config
        assert "InstanceType" not in result
        assert "InitialInstanceCount" not in result

    def test_production_variant_with_accelerator(self):
        """Test production variant with accelerator type"""
        result = production_variant(
            model_name="test-model", instance_type="ml.m5.large", accelerator_type="ml.eia1.medium"
        )

        assert result["AcceleratorType"] == "ml.eia1.medium"


class TestUpdateArgs:
    """Test cases for update_args function"""

    def test_update_args_with_values(self):
        """Test update_args with non-None values"""
        args = {"existing": "value"}

        update_args(args, new_key="new_value", another_key="another_value")

        assert args["existing"] == "value"
        assert args["new_key"] == "new_value"
        assert args["another_key"] == "another_value"

    def test_update_args_with_none_values(self):
        """Test update_args ignores None values"""
        args = {"existing": "value"}

        update_args(args, new_key=None, another_key="another_value")

        assert args["existing"] == "value"
        assert "new_key" not in args
        assert args["another_key"] == "another_value"

class TestFeatureGroupSessionMethods:
    """Test cases for Feature Group session methods"""

    @pytest.fixture
    def session_with_mock_client(self):
        """Create a Session with a mocked sagemaker_client."""
        mock_boto_session = Mock()
        mock_boto_session.region_name = "us-west-2"
        mock_boto_session.client.return_value = Mock()
        mock_boto_session.resource.return_value = Mock()
        session = Session(boto_session=mock_boto_session)
        session.sagemaker_client = Mock()
        return session

    # --- delete_feature_group ---

    def test_delete_feature_group(self, session_with_mock_client):
        """Test delete_feature_group delegates to sagemaker_client."""
        session = session_with_mock_client
        session.delete_feature_group("my-feature-group")

        session.sagemaker_client.delete_feature_group.assert_called_once_with(
            FeatureGroupName="my-feature-group"
        )

    # --- describe_feature_group ---

    def test_describe_feature_group(self, session_with_mock_client):
        """Test describe_feature_group delegates and returns response."""
        session = session_with_mock_client
        expected = {"FeatureGroupName": "my-fg", "CreationTime": "2024-01-01"}
        session.sagemaker_client.describe_feature_group.return_value = expected

        result = session.describe_feature_group("my-fg")

        session.sagemaker_client.describe_feature_group.assert_called_once_with(
            FeatureGroupName="my-fg"
        )
        assert result == expected

    def test_describe_feature_group_with_next_token(self, session_with_mock_client):
        """Test describe_feature_group includes NextToken when provided."""
        session = session_with_mock_client
        session.sagemaker_client.describe_feature_group.return_value = {}

        session.describe_feature_group("my-fg", next_token="abc123")

        session.sagemaker_client.describe_feature_group.assert_called_once_with(
            FeatureGroupName="my-fg", NextToken="abc123"
        )

    def test_describe_feature_group_omits_none_next_token(self, session_with_mock_client):
        """Test describe_feature_group omits NextToken when None."""
        session = session_with_mock_client
        session.sagemaker_client.describe_feature_group.return_value = {}

        session.describe_feature_group("my-fg", next_token=None)

        call_kwargs = session.sagemaker_client.describe_feature_group.call_args[1]
        assert "NextToken" not in call_kwargs

    # --- update_feature_group ---

    def test_update_feature_group_all_params(self, session_with_mock_client):
        """Test update_feature_group with all optional params provided."""
        session = session_with_mock_client
        expected = {"FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123:feature-group/my-fg"}
        session.sagemaker_client.update_feature_group.return_value = expected

        additions = [{"FeatureName": "new_feat", "FeatureType": "String"}]
        online_cfg = {"EnableOnlineStore": True}
        throughput_cfg = {"ThroughputMode": "OnDemand"}

        result = session.update_feature_group(
            "my-fg",
            feature_additions=additions,
            online_store_config=online_cfg,
            throughput_config=throughput_cfg,
        )

        session.sagemaker_client.update_feature_group.assert_called_once_with(
            FeatureGroupName="my-fg",
            FeatureAdditions=additions,
            OnlineStoreConfig=online_cfg,
            ThroughputConfig=throughput_cfg,
        )
        assert result == expected

    def test_update_feature_group_omits_none_params(self, session_with_mock_client):
        """Test update_feature_group omits None optional params."""
        session = session_with_mock_client
        session.sagemaker_client.update_feature_group.return_value = {}

        session.update_feature_group("my-fg")

        call_kwargs = session.sagemaker_client.update_feature_group.call_args[1]
        assert call_kwargs == {"FeatureGroupName": "my-fg"}

    def test_update_feature_group_partial_params(self, session_with_mock_client):
        """Test update_feature_group with only some optional params."""
        session = session_with_mock_client
        session.sagemaker_client.update_feature_group.return_value = {}

        throughput_cfg = {"ThroughputMode": "Provisioned"}
        session.update_feature_group("my-fg", throughput_config=throughput_cfg)

        call_kwargs = session.sagemaker_client.update_feature_group.call_args[1]
        assert call_kwargs == {
            "FeatureGroupName": "my-fg",
            "ThroughputConfig": throughput_cfg,
        }

    # --- list_feature_groups ---

    def test_list_feature_groups_no_params(self, session_with_mock_client):
        """Test list_feature_groups with no filters delegates with empty args."""
        session = session_with_mock_client
        expected = {"FeatureGroupSummaries": []}
        session.sagemaker_client.list_feature_groups.return_value = expected

        result = session.list_feature_groups()

        session.sagemaker_client.list_feature_groups.assert_called_once_with()
        assert result == expected

    def test_list_feature_groups_all_params(self, session_with_mock_client):
        """Test list_feature_groups with all params provided."""
        session = session_with_mock_client
        session.sagemaker_client.list_feature_groups.return_value = {}

        session.list_feature_groups(
            name_contains="test",
            feature_group_status_equals="Created",
            offline_store_status_equals="Active",
            creation_time_after="2024-01-01",
            creation_time_before="2024-12-31",
            sort_order="Ascending",
            sort_by="Name",
            max_results=10,
            next_token="token123",
        )

        session.sagemaker_client.list_feature_groups.assert_called_once_with(
            NameContains="test",
            FeatureGroupStatusEquals="Created",
            OfflineStoreStatusEquals="Active",
            CreationTimeAfter="2024-01-01",
            CreationTimeBefore="2024-12-31",
            SortOrder="Ascending",
            SortBy="Name",
            MaxResults=10,
            NextToken="token123",
        )

    def test_list_feature_groups_omits_none_params(self, session_with_mock_client):
        """Test list_feature_groups omits None params."""
        session = session_with_mock_client
        session.sagemaker_client.list_feature_groups.return_value = {}

        session.list_feature_groups(name_contains="test", max_results=5)

        call_kwargs = session.sagemaker_client.list_feature_groups.call_args[1]
        assert call_kwargs == {"NameContains": "test", "MaxResults": 5}

    # --- update_feature_metadata ---

    def test_update_feature_metadata_all_params(self, session_with_mock_client):
        """Test update_feature_metadata with all optional params."""
        session = session_with_mock_client
        session.sagemaker_client.update_feature_metadata.return_value = {}

        additions = [{"Key": "team", "Value": "ml"}]
        removals = [{"Key": "deprecated"}]

        result = session.update_feature_metadata(
            "my-fg",
            "my-feature",
            description="Updated desc",
            parameter_additions=additions,
            parameter_removals=removals,
        )

        session.sagemaker_client.update_feature_metadata.assert_called_once_with(
            FeatureGroupName="my-fg",
            FeatureName="my-feature",
            Description="Updated desc",
            ParameterAdditions=additions,
            ParameterRemovals=removals,
        )
        assert result == {}

    def test_update_feature_metadata_omits_none_params(self, session_with_mock_client):
        """Test update_feature_metadata omits None optional params."""
        session = session_with_mock_client
        session.sagemaker_client.update_feature_metadata.return_value = {}

        session.update_feature_metadata("my-fg", "my-feature")

        call_kwargs = session.sagemaker_client.update_feature_metadata.call_args[1]
        assert call_kwargs == {
            "FeatureGroupName": "my-fg",
            "FeatureName": "my-feature",
        }

    def test_update_feature_metadata_partial_params(self, session_with_mock_client):
        """Test update_feature_metadata with only description."""
        session = session_with_mock_client
        session.sagemaker_client.update_feature_metadata.return_value = {}

        session.update_feature_metadata("my-fg", "my-feature", description="New desc")

        call_kwargs = session.sagemaker_client.update_feature_metadata.call_args[1]
        assert call_kwargs == {
            "FeatureGroupName": "my-fg",
            "FeatureName": "my-feature",
            "Description": "New desc",
        }

    # --- describe_feature_metadata ---

    def test_describe_feature_metadata(self, session_with_mock_client):
        """Test describe_feature_metadata delegates and returns response."""
        session = session_with_mock_client
        expected = {"FeatureGroupName": "my-fg", "FeatureName": "my-feature"}
        session.sagemaker_client.describe_feature_metadata.return_value = expected

        result = session.describe_feature_metadata("my-fg", "my-feature")

        session.sagemaker_client.describe_feature_metadata.assert_called_once_with(
            FeatureGroupName="my-fg", FeatureName="my-feature"
        )
        assert result == expected

MODULE = "sagemaker.core.helper.session_helper"


class TestCreateFeatureGroup:
    """Test cases for create_feature_group session method."""

    @pytest.fixture
    def session(self):
        """Create a Session with a mocked sagemaker_client."""
        mock_boto_session = Mock()
        mock_boto_session.region_name = "us-west-2"
        mock_boto_session.client.return_value = Mock()
        mock_boto_session.resource.return_value = Mock()
        session = Session(boto_session=mock_boto_session)
        session.sagemaker_client = Mock()
        return session

    @pytest.fixture
    def base_args(self):
        """Minimal required arguments for create_feature_group."""
        return dict(
            feature_group_name="my-fg",
            record_identifier_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=[{"FeatureName": "f1", "FeatureType": "String"}],
        )

    # --- Full parameter pass-through ---

    def test_create_feature_group_all_params(self, session, base_args):
        """Test that all parameters are passed through to sagemaker_client."""
        role = "arn:aws:iam::123456789012:role/Role"
        online_cfg = {"SecurityConfig": {"KmsKeyId": "key-123"}}
        offline_cfg = {"S3StorageConfig": {"S3Uri": "s3://bucket"}}
        throughput_cfg = {"ThroughputMode": "ON_DEMAND"}
        description = "My feature group"
        tags = [{"Key": "team", "Value": "ml"}]

        expected_response = {"FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/my-fg"}
        session.sagemaker_client.create_feature_group.return_value = expected_response

        with patch(f"{MODULE}.format_tags", return_value=tags) as mock_format, \
             patch(f"{MODULE}._append_project_tags", return_value=tags) as mock_proj, \
             patch.object(session, "_append_sagemaker_config_tags", return_value=tags), \
             patch(f"{MODULE}.resolve_value_from_config", return_value=role), \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config", side_effect=[online_cfg, offline_cfg]):

            result = session.create_feature_group(
                **base_args,
                role_arn=role,
                online_store_config=online_cfg,
                offline_store_config=offline_cfg,
                throughput_config=throughput_cfg,
                description=description,
                tags=tags,
            )

        assert result == expected_response
        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert call_kwargs["FeatureGroupName"] == "my-fg"
        assert call_kwargs["RecordIdentifierFeatureName"] == "record_id"
        assert call_kwargs["EventTimeFeatureName"] == "event_time"
        assert call_kwargs["FeatureDefinitions"] == base_args["feature_definitions"]
        assert call_kwargs["RoleArn"] == role
        # EnableOnlineStore is set to True when online config is inferred
        assert call_kwargs["OnlineStoreConfig"]["EnableOnlineStore"] is True
        assert call_kwargs["OfflineStoreConfig"] == offline_cfg
        assert call_kwargs["ThroughputConfig"] == throughput_cfg
        assert call_kwargs["Description"] == description
        assert call_kwargs["Tags"] == tags

    # --- Tag processing pipeline ---

    def test_tag_processing_pipeline_order(self, session, base_args):
        """Test that tags go through format_tags -> _append_project_tags -> _append_sagemaker_config_tags."""
        raw_tags = {"team": "ml"}
        formatted = [{"Key": "team", "Value": "ml"}]
        with_project = [{"Key": "team", "Value": "ml"}, {"Key": "project", "Value": "p1"}]
        with_config = [{"Key": "team", "Value": "ml"}, {"Key": "project", "Value": "p1"}, {"Key": "cfg", "Value": "v"}]

        with patch(f"{MODULE}.format_tags", return_value=formatted) as mock_format, \
             patch(f"{MODULE}._append_project_tags", return_value=with_project) as mock_proj, \
             patch.object(session, "_append_sagemaker_config_tags", return_value=with_config) as mock_cfg, \
             patch(f"{MODULE}.resolve_value_from_config", return_value="arn:role"), \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config", return_value=None):

            session.create_feature_group(**base_args, tags=raw_tags)

        # format_tags is called with the raw input
        mock_format.assert_called_once_with(raw_tags)
        # _append_project_tags receives the formatted tags
        mock_proj.assert_called_once_with(formatted)
        # _append_sagemaker_config_tags receives the project-appended tags
        mock_cfg.assert_called_once_with(with_project, "SageMaker.FeatureGroup.Tags")

        # Final tags in the API call should be the config-appended tags
        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert call_kwargs["Tags"] == with_config

    def test_tags_none_still_processed(self, session, base_args):
        """Test that None tags still go through the pipeline (format_tags handles None)."""
        with patch(f"{MODULE}.format_tags", return_value=None) as mock_format, \
             patch(f"{MODULE}._append_project_tags", return_value=None) as mock_proj, \
             patch.object(session, "_append_sagemaker_config_tags", return_value=None), \
             patch(f"{MODULE}.resolve_value_from_config", return_value="arn:role"), \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config", return_value=None):

            session.create_feature_group(**base_args, tags=None)

        mock_format.assert_called_once_with(None)
        mock_proj.assert_called_once_with(None)
        # Tags=None should be omitted from the API call via update_args
        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert "Tags" not in call_kwargs

    # --- role_arn resolution from config ---

    def test_role_arn_resolved_from_config_when_none(self, session, base_args):
        """Test that role_arn is resolved from SageMaker Config when not provided."""
        config_role = "arn:aws:iam::123456789012:role/ConfigRole"

        with patch(f"{MODULE}.format_tags", return_value=None), \
             patch(f"{MODULE}._append_project_tags", return_value=None), \
             patch.object(session, "_append_sagemaker_config_tags", return_value=None), \
             patch(f"{MODULE}.resolve_value_from_config", return_value=config_role) as mock_resolve, \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config", return_value=None):

            session.create_feature_group(**base_args, role_arn=None)

        mock_resolve.assert_called_once_with(
            None, FEATURE_GROUP_ROLE_ARN_PATH, sagemaker_session=session
        )
        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert call_kwargs["RoleArn"] == config_role

    def test_role_arn_passed_through_when_provided(self, session, base_args):
        """Test that an explicit role_arn is passed to resolve_value_from_config (which returns it)."""
        explicit_role = "arn:aws:iam::123456789012:role/ExplicitRole"

        with patch(f"{MODULE}.format_tags", return_value=None), \
             patch(f"{MODULE}._append_project_tags", return_value=None), \
             patch.object(session, "_append_sagemaker_config_tags", return_value=None), \
             patch(f"{MODULE}.resolve_value_from_config", return_value=explicit_role) as mock_resolve, \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config", return_value=None):

            session.create_feature_group(**base_args, role_arn=explicit_role)

        mock_resolve.assert_called_once_with(
            explicit_role, FEATURE_GROUP_ROLE_ARN_PATH, sagemaker_session=session
        )
        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert call_kwargs["RoleArn"] == explicit_role

    # --- online_store_config merging and EnableOnlineStore ---

    def test_online_store_config_merged_and_enable_set(self, session, base_args):
        """Test that online_store_config is merged from config and EnableOnlineStore=True is set."""
        inferred_online = {"SecurityConfig": {"KmsKeyId": "config-key"}}

        with patch(f"{MODULE}.format_tags", return_value=None), \
             patch(f"{MODULE}._append_project_tags", return_value=None), \
             patch.object(session, "_append_sagemaker_config_tags", return_value=None), \
             patch(f"{MODULE}.resolve_value_from_config", return_value="arn:role"), \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config",
                   side_effect=[inferred_online, None]) as mock_update:

            session.create_feature_group(**base_args, online_store_config=None)

        # First call is for online store config
        mock_update.assert_any_call(
            None, FEATURE_GROUP_ONLINE_STORE_CONFIG_PATH, sagemaker_session=session
        )
        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert call_kwargs["OnlineStoreConfig"]["EnableOnlineStore"] is True
        assert call_kwargs["OnlineStoreConfig"]["SecurityConfig"]["KmsKeyId"] == "config-key"

    def test_online_store_config_none_when_no_config(self, session, base_args):
        """Test that OnlineStoreConfig is omitted when config returns None."""
        with patch(f"{MODULE}.format_tags", return_value=None), \
             patch(f"{MODULE}._append_project_tags", return_value=None), \
             patch.object(session, "_append_sagemaker_config_tags", return_value=None), \
             patch(f"{MODULE}.resolve_value_from_config", return_value="arn:role"), \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config", return_value=None):

            session.create_feature_group(**base_args)

        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert "OnlineStoreConfig" not in call_kwargs

    def test_online_store_config_explicit_gets_enable_set(self, session, base_args):
        """Test that explicitly provided online_store_config also gets EnableOnlineStore=True."""
        explicit_online = {"SecurityConfig": {"KmsKeyId": "my-key"}}
        # update_nested_dictionary returns the merged result
        merged_online = {"SecurityConfig": {"KmsKeyId": "my-key"}}

        with patch(f"{MODULE}.format_tags", return_value=None), \
             patch(f"{MODULE}._append_project_tags", return_value=None), \
             patch.object(session, "_append_sagemaker_config_tags", return_value=None), \
             patch(f"{MODULE}.resolve_value_from_config", return_value="arn:role"), \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config",
                   side_effect=[merged_online, None]):

            session.create_feature_group(**base_args, online_store_config=explicit_online)

        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert call_kwargs["OnlineStoreConfig"]["EnableOnlineStore"] is True

    # --- offline_store_config merging ---

    def test_offline_store_config_merged_from_config(self, session, base_args):
        """Test that offline_store_config is merged from SageMaker Config."""
        inferred_offline = {"S3StorageConfig": {"S3Uri": "s3://config-bucket"}}

        with patch(f"{MODULE}.format_tags", return_value=None), \
             patch(f"{MODULE}._append_project_tags", return_value=None), \
             patch.object(session, "_append_sagemaker_config_tags", return_value=None), \
             patch(f"{MODULE}.resolve_value_from_config", return_value="arn:role"), \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config",
                   side_effect=[None, inferred_offline]) as mock_update:

            session.create_feature_group(**base_args, offline_store_config=None)

        # Second call is for offline store config
        mock_update.assert_any_call(
            None, FEATURE_GROUP_OFFLINE_STORE_CONFIG_PATH, sagemaker_session=session
        )
        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert call_kwargs["OfflineStoreConfig"] == inferred_offline

    def test_offline_store_config_none_when_no_config(self, session, base_args):
        """Test that OfflineStoreConfig is omitted when config returns None."""
        with patch(f"{MODULE}.format_tags", return_value=None), \
             patch(f"{MODULE}._append_project_tags", return_value=None), \
             patch.object(session, "_append_sagemaker_config_tags", return_value=None), \
             patch(f"{MODULE}.resolve_value_from_config", return_value="arn:role"), \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config", return_value=None):

            session.create_feature_group(**base_args)

        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert "OfflineStoreConfig" not in call_kwargs

    # --- None optional parameters omitted ---

    def test_none_optional_params_omitted(self, session, base_args):
        """Test that None optional params (throughput, description, tags) are omitted from API call."""
        with patch(f"{MODULE}.format_tags", return_value=None), \
             patch(f"{MODULE}._append_project_tags", return_value=None), \
             patch.object(session, "_append_sagemaker_config_tags", return_value=None), \
             patch(f"{MODULE}.resolve_value_from_config", return_value="arn:role"), \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config", return_value=None):

            session.create_feature_group(**base_args)

        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert "ThroughputConfig" not in call_kwargs
        assert "Description" not in call_kwargs
        assert "Tags" not in call_kwargs
        assert "OnlineStoreConfig" not in call_kwargs
        assert "OfflineStoreConfig" not in call_kwargs
        # Required params should still be present
        assert "FeatureGroupName" in call_kwargs
        assert "RecordIdentifierFeatureName" in call_kwargs
        assert "EventTimeFeatureName" in call_kwargs
        assert "FeatureDefinitions" in call_kwargs
        assert "RoleArn" in call_kwargs

    def test_partial_optional_params(self, session, base_args):
        """Test that only provided optional params appear in the API call."""
        throughput = {"ThroughputMode": "ON_DEMAND"}

        with patch(f"{MODULE}.format_tags", return_value=None), \
             patch(f"{MODULE}._append_project_tags", return_value=None), \
             patch.object(session, "_append_sagemaker_config_tags", return_value=None), \
             patch(f"{MODULE}.resolve_value_from_config", return_value="arn:role"), \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config", return_value=None):

            session.create_feature_group(
                **base_args,
                throughput_config=throughput,
                description="test desc",
            )

        call_kwargs = session.sagemaker_client.create_feature_group.call_args[1]
        assert call_kwargs["ThroughputConfig"] == throughput
        assert call_kwargs["Description"] == "test desc"
        assert "Tags" not in call_kwargs
        assert "OnlineStoreConfig" not in call_kwargs
        assert "OfflineStoreConfig" not in call_kwargs

    # --- Return value ---

    def test_returns_api_response(self, session, base_args):
        """Test that the method returns the sagemaker_client response."""
        expected = {"FeatureGroupArn": "arn:fg"}
        session.sagemaker_client.create_feature_group.return_value = expected

        with patch(f"{MODULE}.format_tags", return_value=None), \
             patch(f"{MODULE}._append_project_tags", return_value=None), \
             patch.object(session, "_append_sagemaker_config_tags", return_value=None), \
             patch(f"{MODULE}.resolve_value_from_config", return_value="arn:role"), \
             patch(f"{MODULE}.update_nested_dictionary_with_values_from_config", return_value=None):

            result = session.create_feature_group(**base_args)

        assert result == expected
