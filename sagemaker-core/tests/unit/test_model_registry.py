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
from unittest.mock import Mock, patch, MagicMock
from sagemaker.core.model_registry import (
    get_model_package_args,
    get_create_model_package_request,
    create_model_package_from_containers,
)


@pytest.fixture
def mock_session():
    """Create a mock SageMaker session"""
    session = Mock()
    session.boto_session.region_name = "us-west-2"
    session.boto_region_name = "us-west-2"
    session.sagemaker_client = Mock()
    session.search = Mock()
    session.sagemaker_config = {}
    return session


class TestModelRegistry:
    """Test cases for model registry functions"""

    def test_get_model_package_args_minimal(self):
        """Test get_model_package_args with minimal parameters"""
        args = get_model_package_args(
            image_uri="test-image:latest",
            inference_instances=["ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"]
        )
        
        assert args["containers"][0]["Image"] == "test-image:latest"
        assert args["inference_instances"] == ["ml.m5.xlarge"]
        assert args["transform_instances"] == ["ml.m5.xlarge"]
        assert args["marketplace_cert"] is False

    def test_get_model_package_args_with_model_data(self):
        """Test get_model_package_args with model data"""
        args = get_model_package_args(
            image_uri="test-image:latest",
            model_data="s3://bucket/model.tar.gz",
            inference_instances=["ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"]
        )
        
        assert args["containers"][0]["ModelDataUrl"] == "s3://bucket/model.tar.gz"

    def test_get_model_package_args_with_container_list(self):
        """Test get_model_package_args with container definition list"""
        container_list = [
            {"Image": "image1:latest", "ModelDataUrl": "s3://bucket/model1.tar.gz"},
            {"Image": "image2:latest", "ModelDataUrl": "s3://bucket/model2.tar.gz"}
        ]
        
        args = get_model_package_args(
            container_def_list=container_list,
            inference_instances=["ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"]
        )
        
        assert len(args["containers"]) == 2
        assert args["containers"][0]["Image"] == "image1:latest"
        assert args["containers"][1]["Image"] == "image2:latest"

    def test_get_model_package_args_with_all_params(self):
        """Test get_model_package_args with all parameters"""
        model_metrics = Mock()
        model_metrics._to_request_dict = Mock(return_value={"Accuracy": 0.95})
        
        drift_check_baselines = Mock()
        drift_check_baselines._to_request_dict = Mock(return_value={"Constraints": {}})
        
        metadata_properties = Mock()
        metadata_properties._to_request_dict = Mock(return_value={"ProjectId": "123"})
        
        args = get_model_package_args(
            content_types=["text/csv"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_name="test-package",
            model_package_group_name="test-group",
            model_data="s3://bucket/model.tar.gz",
            image_uri="test-image:latest",
            model_metrics=model_metrics,
            metadata_properties=metadata_properties,
            marketplace_cert=True,
            approval_status="Approved",
            description="Test model package",
            tags=[{"Key": "test", "Value": "value"}],
            drift_check_baselines=drift_check_baselines,
            customer_metadata_properties={"custom": "value"},
            validation_specification={"ValidationRole": "arn:aws:iam::123:role/test"},
            domain="COMPUTER_VISION",
            sample_payload_url="s3://bucket/sample.json",
            task="IMAGE_CLASSIFICATION",
            skip_model_validation="All",
            source_uri="s3://bucket/source",
        )
        
        assert args["content_types"] == ["text/csv"]
        assert args["response_types"] == ["application/json"]
        assert args["model_package_name"] == "test-package"
        assert args["model_package_group_name"] == "test-group"
        assert args["model_metrics"] == {"Accuracy": 0.95}
        assert args["marketplace_cert"] is True
        assert args["approval_status"] == "Approved"
        assert args["description"] == "Test model package"
        assert args["customer_metadata_properties"] == {"custom": "value"}
        assert args["domain"] == "COMPUTER_VISION"
        assert args["task"] == "IMAGE_CLASSIFICATION"
        assert args["skip_model_validation"] == "All"
        assert args["source_uri"] == "s3://bucket/source"

    def test_get_create_model_package_request_minimal(self):
        """Test get_create_model_package_request with minimal parameters"""
        request = get_create_model_package_request(
            model_package_name="test-package",
            containers=[{"Image": "test-image:latest"}],
            inference_instances=["ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"]
        )
        
        assert request["ModelPackageName"] == "test-package"
        assert request["InferenceSpecification"]["Containers"][0]["Image"] == "test-image:latest"
        assert request["CertifyForMarketplace"] is False
        assert request["ModelApprovalStatus"] == "PendingManualApproval"

    def test_get_create_model_package_request_with_group(self):
        """Test get_create_model_package_request with model package group"""
        request = get_create_model_package_request(
            model_package_group_name="test-group",
            containers=[{"Image": "test-image:latest"}],
            content_types=["text/csv"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"]
        )
        
        assert request["ModelPackageGroupName"] == "test-group"
        assert "ModelPackageName" not in request
        assert request["InferenceSpecification"]["SupportedContentTypes"] == ["text/csv"]
        assert request["InferenceSpecification"]["SupportedResponseMIMETypes"] == ["application/json"]

    def test_get_create_model_package_request_validation_error_both_names(self):
        """Test get_create_model_package_request raises error with both names"""
        with pytest.raises(ValueError, match="cannot be present at the same time"):
            get_create_model_package_request(
                model_package_name="test-package",
                model_package_group_name="test-group",
                containers=[{"Image": "test-image:latest"}],
                inference_instances=["ml.m5.xlarge"],
                transform_instances=["ml.m5.xlarge"]
            )

    def test_get_create_model_package_request_validation_error_source_uri(self):
        """Test get_create_model_package_request raises error with unversioned and source_uri"""
        with pytest.raises(ValueError, match="cannot be created with source_uri"):
            get_create_model_package_request(
                model_package_name="test-package",
                source_uri="s3://bucket/source",
                containers=[{"Image": "test-image:latest"}],
                inference_instances=["ml.m5.xlarge"],
                transform_instances=["ml.m5.xlarge"]
            )

    def test_get_create_model_package_request_validation_error_model_data_source(self):
        """Test get_create_model_package_request raises error with ModelDataSource"""
        containers = [{"Image": "test-image:latest", "ModelDataSource": {"S3DataSource": {}}}]
        
        with pytest.raises(ValueError, match="cannot be created with ModelDataSource"):
            get_create_model_package_request(
                model_package_name="test-package",
                containers=containers,
                inference_instances=["ml.m5.xlarge"],
                transform_instances=["ml.m5.xlarge"]
            )

    def test_get_create_model_package_request_missing_instances(self):
        """Test get_create_model_package_request raises error without instances for unversioned"""
        with pytest.raises(ValueError, match="must be provided"):
            get_create_model_package_request(
                model_package_name="test-package",
                containers=[{"Image": "test-image:latest"}]
            )

    def test_get_create_model_package_request_with_metrics(self):
        """Test get_create_model_package_request with model metrics"""
        model_metrics = {"Accuracy": {"Value": 0.95}}
        
        request = get_create_model_package_request(
            model_package_group_name="test-group",
            containers=[{"Image": "test-image:latest"}],
            model_metrics=model_metrics,
            inference_instances=["ml.m5.xlarge"]
        )
        
        assert request["ModelMetrics"] == model_metrics

    def test_get_create_model_package_request_with_validation(self):
        """Test get_create_model_package_request with validation specification"""
        validation_spec = {
            "ValidationRole": "arn:aws:iam::123:role/test",
            "ValidationProfiles": [{"ProfileName": "test-profile"}]
        }
        
        request = get_create_model_package_request(
            model_package_group_name="test-group",
            containers=[{"Image": "test-image:latest"}],
            validation_specification=validation_spec,
            inference_instances=["ml.m5.xlarge"]
        )
        
        assert request["ValidationSpecification"] == validation_spec

    def test_get_create_model_package_request_with_domain_task(self):
        """Test get_create_model_package_request with domain and task"""
        request = get_create_model_package_request(
            model_package_group_name="test-group",
            containers=[{"Image": "test-image:latest"}],
            domain="NATURAL_LANGUAGE_PROCESSING",
            task="TEXT_GENERATION",
            sample_payload_url="s3://bucket/sample.json",
            inference_instances=["ml.m5.xlarge"]
        )
        
        assert request["Domain"] == "NATURAL_LANGUAGE_PROCESSING"
        assert request["Task"] == "TEXT_GENERATION"
        assert request["SamplePayloadUrl"] == "s3://bucket/sample.json"

    def test_get_create_model_package_request_skip_validation(self):
        """Test get_create_model_package_request with skip_model_validation"""
        request = get_create_model_package_request(
            model_package_group_name="test-group",
            containers=[{"Image": "test-image:latest"}],
            skip_model_validation="All",
            inference_instances=["ml.m5.xlarge"]
        )
        
        assert request["SkipModelValidation"] == "All"

    @patch("sagemaker.core.model_registry._create_resource")
    def test_create_model_package_from_containers_creates_group(self, mock_create_resource, mock_session):
        """Test create_model_package_from_containers creates model package group if needed"""
        mock_session.search.return_value = {"Results": []}
        mock_session.sagemaker_client.list_model_package_groups.return_value = {
            "ModelPackageGroupSummaryList": [],
            "NextToken": None
        }
        mock_session.sagemaker_client.create_model_package.return_value = {
            "ModelPackageArn": "arn:aws:sagemaker:us-west-2:123:model-package/test/1"
        }
        mock_session._intercept_create_request = Mock(side_effect=lambda req, submit, name: submit(req))
        
        create_model_package_from_containers(
            sagemaker_session=mock_session,
            model_package_group_name="new-group",
            containers=[{"Image": "test-image:latest"}],
            inference_instances=["ml.m5.xlarge"]
        )
        
        mock_create_resource.assert_called_once()

    def test_create_model_package_from_containers_with_source_uri_autopopulate(self, mock_session):
        """Test create_model_package_from_containers with autopopulate source_uri"""
        mock_session.sagemaker_client.create_model_package.return_value = {
            "ModelPackageArn": "arn:aws:sagemaker:us-west-2:123:model-package/test/1"
        }
        mock_session.search.return_value = {"Results": []}
        mock_session.sagemaker_client.list_model_package_groups.return_value = {
            "ModelPackageGroupSummaryList": [],
            "NextToken": None
        }
        mock_session._intercept_create_request = Mock(side_effect=lambda req, submit, name: submit(req))
        
        with patch("sagemaker.core.model_registry.can_model_package_source_uri_autopopulate", return_value=True):
            result = create_model_package_from_containers(
                sagemaker_session=mock_session,
                model_package_group_name="test-group",
                containers=[{"Image": "test-image:latest"}],
                source_uri="arn:aws:sagemaker:us-west-2:123:model-package/source/1",
                inference_instances=["ml.m5.xlarge"]
            )
        
        # Should call create_model_package once without InferenceSpecification
        assert mock_session.sagemaker_client.create_model_package.called

    def test_create_model_package_from_containers_with_source_uri_no_autopopulate(self, mock_session):
        """Test create_model_package_from_containers with non-autopopulate source_uri"""
        mock_session.sagemaker_client.create_model_package.return_value = {
            "ModelPackageArn": "arn:aws:sagemaker:us-west-2:123:model-package/test/1"
        }
        mock_session.sagemaker_client.update_model_package.return_value = {}
        mock_session.search.return_value = {"Results": []}
        mock_session.sagemaker_client.list_model_package_groups.return_value = {
            "ModelPackageGroupSummaryList": [],
            "NextToken": None
        }
        mock_session._intercept_create_request = Mock(side_effect=lambda req, submit, name: submit(req))
        
        with patch("sagemaker.core.model_registry.can_model_package_source_uri_autopopulate", return_value=False):
            result = create_model_package_from_containers(
                sagemaker_session=mock_session,
                model_package_group_name="test-group",
                containers=[{"Image": "test-image:latest"}],
                source_uri="s3://bucket/source",
                inference_instances=["ml.m5.xlarge"]
            )
        
        # Should call create_model_package and update_model_package
        assert mock_session.sagemaker_client.create_model_package.called
        assert mock_session.sagemaker_client.update_model_package.called

    def test_create_model_package_from_containers_with_validation_config(self, mock_session):
        """Test create_model_package_from_containers with validation specification config resolution"""
        validation_spec = {
            "ValidationRole": "arn:aws:iam::123:role/test",
            "ValidationProfiles": [{"ProfileName": "test-profile"}]
        }
        
        mock_session.sagemaker_client.create_model_package.return_value = {
            "ModelPackageArn": "arn:aws:sagemaker:us-west-2:123:model-package/test/1"
        }
        mock_session.search.return_value = {"Results": []}
        mock_session.sagemaker_client.list_model_package_groups.return_value = {
            "ModelPackageGroupSummaryList": [],
            "NextToken": None
        }
        mock_session._intercept_create_request = Mock(side_effect=lambda req, submit, name: submit(req))
        
        with patch("sagemaker.core.model_registry.resolve_value_from_config", side_effect=lambda x, *args, **kwargs: x):
            with patch("sagemaker.core.model_registry.update_list_of_dicts_with_values_from_config"):
                result = create_model_package_from_containers(
                    sagemaker_session=mock_session,
                    model_package_group_name="test-group",
                    containers=[{"Image": "test-image:latest"}],
                    validation_specification=validation_spec,
                    inference_instances=["ml.m5.xlarge"]
                )
        
        assert mock_session.sagemaker_client.create_model_package.called

    def test_create_model_package_from_containers_with_containers_config(self, mock_session):
        """Test create_model_package_from_containers with containers config resolution"""
        containers = [{"Image": "test-image:latest"}]
        
        mock_session.sagemaker_client.create_model_package.return_value = {
            "ModelPackageArn": "arn:aws:sagemaker:us-west-2:123:model-package/test/1"
        }
        mock_session.search.return_value = {"Results": []}
        mock_session.sagemaker_client.list_model_package_groups.return_value = {
            "ModelPackageGroupSummaryList": [],
            "NextToken": None
        }
        mock_session._intercept_create_request = Mock(side_effect=lambda req, submit, name: submit(req))
        
        with patch("sagemaker.core.model_registry.update_list_of_dicts_with_values_from_config") as mock_update:
            result = create_model_package_from_containers(
                sagemaker_session=mock_session,
                model_package_group_name="test-group",
                containers=containers,
                inference_instances=["ml.m5.xlarge"]
            )
        
        mock_update.assert_called_once()
        assert mock_session.sagemaker_client.create_model_package.called
