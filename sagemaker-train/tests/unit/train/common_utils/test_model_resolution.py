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
"""Tests for model_resolution module."""
from __future__ import absolute_import

import json
import pytest
from unittest.mock import patch, MagicMock, Mock
import os

from sagemaker.train.common_utils.model_resolution import (
    _ModelType,
    _ModelInfo,
    _ModelResolver,
    _resolve_base_model,
)


class TestModelType:
    """Tests for _ModelType enum."""
    
    def test_model_type_values(self):
        """Test ModelType enum values."""
        assert _ModelType.JUMPSTART.value == "jumpstart"
        assert _ModelType.FINE_TUNED.value == "fine_tuned"


class TestModelInfo:
    """Tests for _ModelInfo dataclass."""
    
    def test_model_info_creation(self):
        """Test creating ModelInfo instance."""
        info = _ModelInfo(
            base_model_name="test-model",
            base_model_arn="arn:aws:sagemaker:us-west-2:aws:hub-content/test",
            source_model_package_arn=None,
            model_type=_ModelType.JUMPSTART,
            hub_content_name="test-model",
            additional_metadata={}
        )
        
        assert info.base_model_name == "test-model"
        assert info.model_type == _ModelType.JUMPSTART
        assert info.source_model_package_arn is None


class TestModelResolver:
    """Tests for _ModelResolver class."""
    
    def test_resolver_initialization(self):
        """Test ModelResolver initialization."""
        resolver = _ModelResolver()
        assert resolver.sagemaker_session is None
        assert resolver.DEFAULT_HUB_NAME == "SageMakerPublicHub"
    
    def test_resolver_with_session(self):
        """Test ModelResolver with custom session."""
        mock_session = MagicMock()
        resolver = _ModelResolver(sagemaker_session=mock_session)
        assert resolver.sagemaker_session == mock_session
    
    @patch.dict(os.environ, {'SAGEMAKER_ENDPOINT': 'https://beta.endpoint'})
    def test_resolver_with_beta_endpoint(self):
        """Test ModelResolver detects beta endpoint."""
        resolver = _ModelResolver()
        assert resolver._endpoint == 'https://beta.endpoint'


class TestResolveModelInfo:
    """Tests for resolve_model_info method."""
    
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._resolve_jumpstart_model')
    def test_resolve_jumpstart_model_id(self, mock_resolve_js):
        """Test resolving JumpStart model ID."""
        resolver = _ModelResolver()
        mock_info = _ModelInfo(
            base_model_name="llama3-2-1b",
            base_model_arn="arn:test",
            source_model_package_arn=None,
            model_type=_ModelType.JUMPSTART,
            hub_content_name="llama3-2-1b",
            additional_metadata={}
        )
        mock_resolve_js.return_value = mock_info
        
        result = resolver.resolve_model_info("llama3-2-1b")
        
        assert result.base_model_name == "llama3-2-1b"
        assert result.model_type == _ModelType.JUMPSTART
        mock_resolve_js.assert_called_once_with("llama3-2-1b", "SageMakerPublicHub")
    
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._resolve_model_package_arn')
    def test_resolve_model_package_arn_string(self, mock_resolve_arn):
        """Test resolving ModelPackage ARN string."""
        resolver = _ModelResolver()
        arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-model/1"
        mock_info = _ModelInfo(
            base_model_name="base-model",
            base_model_arn="arn:base",
            source_model_package_arn=arn,
            model_type=_ModelType.FINE_TUNED,
            hub_content_name="base-model",
            additional_metadata={}
        )
        mock_resolve_arn.return_value = mock_info
        
        result = resolver.resolve_model_info(arn)
        
        assert result.source_model_package_arn == arn
        assert result.model_type == _ModelType.FINE_TUNED
        mock_resolve_arn.assert_called_once_with(arn)
    
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._resolve_model_package_object')
    def test_resolve_model_package_object(self, mock_resolve_obj):
        """Test resolving ModelPackage object."""
        resolver = _ModelResolver()
        mock_package = MagicMock()
        mock_package.model_package_arn = "arn:test"
        mock_info = _ModelInfo(
            base_model_name="base-model",
            base_model_arn="arn:base",
            source_model_package_arn="arn:test",
            model_type=_ModelType.FINE_TUNED,
            hub_content_name="base-model",
            additional_metadata={}
        )
        mock_resolve_obj.return_value = mock_info
        
        result = resolver.resolve_model_info(mock_package)
        
        assert result.model_type == _ModelType.FINE_TUNED
        mock_resolve_obj.assert_called_once_with(mock_package)
    
    def test_resolve_invalid_input(self):
        """Test error with invalid input type."""
        resolver = _ModelResolver()
        
        with pytest.raises(ValueError, match="base_model must be a string"):
            resolver.resolve_model_info(12345)
    
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._resolve_jumpstart_model')
    def test_resolve_with_custom_hub(self, mock_resolve_js):
        """Test resolving with custom hub name."""
        resolver = _ModelResolver()
        mock_resolve_js.return_value = MagicMock()
        
        resolver.resolve_model_info("test-model", hub_name="CustomHub")
        
        mock_resolve_js.assert_called_once_with("test-model", "CustomHub")


class TestResolveJumpStartModel:
    """Tests for _resolve_jumpstart_model method."""
    
    @patch('sagemaker.core.resources.HubContent')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._get_session')
    def test_resolve_jumpstart_success(self, mock_get_session, mock_hub_content_class):
        """Test successful JumpStart model resolution."""
        # Mock session
        mock_session = MagicMock()
        mock_session.boto_session.region_name = 'us-west-2'
        mock_get_session.return_value = mock_session
        
        # Mock HubContent
        mock_hub_content = MagicMock()
        mock_hub_content.hub_content_arn = "arn:aws:sagemaker:us-west-2:aws:hub-content/test"
        mock_hub_content.hub_content_document = '{"key": "value"}'
        mock_hub_content_class.get.return_value = mock_hub_content
        
        resolver = _ModelResolver()
        result = resolver._resolve_jumpstart_model("test-model", "SageMakerPublicHub")
        
        assert result.base_model_name == "test-model"
        assert result.hub_content_name == "test-model"
        assert result.model_type == _ModelType.JUMPSTART
        assert result.additional_metadata == {"key": "value"}
    
    @patch('sagemaker.core.resources.HubContent')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._get_session')
    def test_resolve_jumpstart_invalid_json(self, mock_get_session, mock_hub_content_class):
        """Test JumpStart resolution with invalid JSON document."""
        mock_session = MagicMock()
        mock_session.boto_session.region_name = 'us-west-2'
        mock_get_session.return_value = mock_session
        
        mock_hub_content = MagicMock()
        mock_hub_content.hub_content_arn = "arn:test"
        mock_hub_content.hub_content_document = "invalid json"
        mock_hub_content_class.get.return_value = mock_hub_content
        
        resolver = _ModelResolver()
        result = resolver._resolve_jumpstart_model("test-model", "SageMakerPublicHub")
        
        assert result.additional_metadata == {}
    
    @patch('sagemaker.core.resources.HubContent')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._get_session')
    def test_resolve_jumpstart_failure(self, mock_get_session, mock_hub_content_class):
        """Test JumpStart resolution failure."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_hub_content_class.get.side_effect = Exception("Hub error")
        
        resolver = _ModelResolver()
        
        with pytest.raises(ValueError, match="Failed to resolve JumpStart model"):
            resolver._resolve_jumpstart_model("test-model", "SageMakerPublicHub")


class TestResolveModelPackageObject:
    """Tests for _resolve_model_package_object method."""
    
    def test_resolve_package_object_success(self):
        """Test successful ModelPackage object resolution."""
        # Create mock ModelPackage
        mock_package = MagicMock()
        mock_package.model_package_arn = "arn:aws:sagemaker:us-west-2:123:model-package/test/1"
        
        # Mock inference specification
        mock_container = MagicMock()
        mock_base_model = MagicMock()
        mock_base_model.hub_content_name = "base-model"
        mock_base_model.hub_content_arn = "arn:aws:sagemaker:us-west-2:aws:hub-content/base"
        mock_container.base_model = mock_base_model
        
        mock_package.inference_specification = MagicMock()
        mock_package.inference_specification.containers = [mock_container]
        
        resolver = _ModelResolver()
        result = resolver._resolve_model_package_object(mock_package)
        
        assert result.base_model_name == "base-model"
        assert result.hub_content_name == "base-model"
        assert result.model_type == _ModelType.FINE_TUNED
        assert result.source_model_package_arn == mock_package.model_package_arn
    
    def test_resolve_package_no_inference_spec(self):
        """Test error when inference specification is missing."""
        mock_package = MagicMock()
        mock_package.model_package_arn = "arn:test"
        mock_package.inference_specification = None
        
        resolver = _ModelResolver()
        
        with pytest.raises(ValueError, match="NotSupported.*does not have an inference_specification"):
            resolver._resolve_model_package_object(mock_package)
    
    def test_resolve_package_no_containers(self):
        """Test error when containers are missing."""
        mock_package = MagicMock()
        mock_package.model_package_arn = "arn:test"
        mock_package.inference_specification = MagicMock()
        mock_package.inference_specification.containers = []
        
        resolver = _ModelResolver()
        
        with pytest.raises(ValueError, match="NotSupported.*does not have any containers"):
            resolver._resolve_model_package_object(mock_package)
    
    def test_resolve_package_no_base_model(self):
        """Test error when base_model metadata is missing."""
        mock_package = MagicMock()
        mock_package.model_package_arn = "arn:test"
        
        mock_container = MagicMock()
        mock_container.base_model = None
        
        mock_package.inference_specification = MagicMock()
        mock_package.inference_specification.containers = [mock_container]
        
        resolver = _ModelResolver()
        
        with pytest.raises(ValueError, match="NotSupported.*does not have base_model metadata"):
            resolver._resolve_model_package_object(mock_package)
    
    def test_resolve_package_fallback_name(self):
        """Test fallback to package name when hub_content_name is missing."""
        mock_package = MagicMock()
        mock_package.model_package_arn = "arn:aws:sagemaker:us-west-2:123:model-package/group-name/1"
        mock_package.model_package_name = "fallback-name"
        
        mock_container = MagicMock()
        mock_base_model = MagicMock()
        mock_base_model.hub_content_name = None
        mock_base_model.hub_content_arn = "arn:base"
        mock_container.base_model = mock_base_model
        
        mock_package.inference_specification = MagicMock()
        mock_package.inference_specification.containers = [mock_container]
        
        resolver = _ModelResolver()
        result = resolver._resolve_model_package_object(mock_package)
        
        assert result.base_model_name == "group-name"


class TestResolveModelPackageArn:
    """Tests for _resolve_model_package_arn method."""
    
    @patch('sagemaker.core.resources.ModelPackage')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._get_session')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._validate_model_package_arn')
    def test_resolve_arn_success(self, mock_validate, mock_get_session, mock_model_package_class):
        """Test successful ARN resolution using ModelPackage.get()."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-model/1"
        
        # Mock session
        mock_session = MagicMock()
        mock_session.boto_session.region_name = 'us-west-2'
        mock_get_session.return_value = mock_session
        
        # Mock ModelPackage.get() return value
        mock_package = MagicMock()
        mock_package.model_package_arn = arn
        
        # Mock inference specification with hub_content_arn
        mock_container = MagicMock()
        mock_base_model = MagicMock()
        mock_base_model.hub_content_name = 'base-model'
        mock_base_model.hub_content_version = '1.0'
        mock_base_model.hub_content_arn = 'arn:aws:sagemaker:us-west-2:aws:hub-content/base'
        mock_container.base_model = mock_base_model
        
        mock_package.inference_specification = MagicMock()
        mock_package.inference_specification.containers = [mock_container]
        
        mock_model_package_class.get.return_value = mock_package
        
        resolver = _ModelResolver()
        result = resolver._resolve_model_package_arn(arn)
        
        assert result.base_model_name == "base-model"
        assert result.hub_content_name == "base-model"
        assert result.source_model_package_arn == arn
        assert result.model_type == _ModelType.FINE_TUNED
        mock_model_package_class.get.assert_called_once_with(
            model_package_name=arn,
            session=mock_session.boto_session,
            region='us-west-2'
        )
    
    @patch('sagemaker.core.resources.ModelPackage')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._get_session')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._validate_model_package_arn')
    def test_resolve_arn_construct_hub_content_arn(self, mock_validate, mock_get_session, mock_model_package_class):
        """Test ARN resolution when HubContentArn needs to be constructed."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-model/1"
        
        # Mock session
        mock_session = MagicMock()
        mock_session.boto_session.region_name = 'us-west-2'
        mock_get_session.return_value = mock_session
        
        # Mock ModelPackage without hub_content_arn (needs to be constructed)
        mock_package = MagicMock()
        mock_package.model_package_arn = arn
        
        mock_container = MagicMock()
        mock_base_model = MagicMock()
        mock_base_model.hub_content_name = 'base-model'
        mock_base_model.hub_content_version = '1.0'
        mock_base_model.hub_content_arn = None  # Not provided, needs construction
        mock_container.base_model = mock_base_model
        
        mock_package.inference_specification = MagicMock()
        mock_package.inference_specification.containers = [mock_container]
        
        mock_model_package_class.get.return_value = mock_package
        
        resolver = _ModelResolver()
        result = resolver._resolve_model_package_arn(arn)
        
        # Should construct ARN from region and hub content name/version
        expected_arn = "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/base-model/1.0"
        assert result.base_model_arn == expected_arn
        assert result.base_model_name == "base-model"
        assert result.hub_content_name == "base-model"
    
    @patch('sagemaker.core.resources.ModelPackage')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._get_session')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._validate_model_package_arn')
    def test_resolve_arn_no_inference_spec(self, mock_validate, mock_get_session, mock_model_package_class):
        """Test error when InferenceSpecification is missing."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-model/1"
        
        # Mock session
        mock_session = MagicMock()
        mock_session.boto_session.region_name = 'us-west-2'
        mock_get_session.return_value = mock_session
        
        # Mock ModelPackage without inference_specification
        mock_package = MagicMock()
        mock_package.model_package_arn = arn
        mock_package.inference_specification = None
        
        mock_model_package_class.get.return_value = mock_package
        
        resolver = _ModelResolver()
        
        with pytest.raises(ValueError, match="NotSupported.*does not have an inference_specification"):
            resolver._resolve_model_package_arn(arn)
    
    @patch('sagemaker.core.resources.ModelPackage')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._get_session')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._validate_model_package_arn')
    def test_resolve_arn_no_base_model(self, mock_validate, mock_get_session, mock_model_package_class):
        """Test error when BaseModel is missing."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-model/1"
        
        # Mock session
        mock_session = MagicMock()
        mock_session.boto_session.region_name = 'us-west-2'
        mock_get_session.return_value = mock_session
        
        # Mock ModelPackage with container but no base_model
        mock_package = MagicMock()
        mock_package.model_package_arn = arn
        
        mock_container = MagicMock()
        mock_container.base_model = None
        
        mock_package.inference_specification = MagicMock()
        mock_package.inference_specification.containers = [mock_container]
        
        mock_model_package_class.get.return_value = mock_package
        
        resolver = _ModelResolver()
        
        with pytest.raises(ValueError, match="NotSupported.*does not have base_model metadata"):
            resolver._resolve_model_package_arn(arn)


class TestValidateModelPackageArn:
    """Tests for _validate_model_package_arn method."""
    
    def test_validate_valid_arn(self):
        """Test validation of valid ARN."""
        resolver = _ModelResolver()
        arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-model/1"
        
        result = resolver._validate_model_package_arn(arn)
        assert result is True
    
    def test_validate_invalid_arn_format(self):
        """Test validation of invalid ARN format."""
        resolver = _ModelResolver()
        
        with pytest.raises(ValueError, match="Invalid ModelPackage ARN format"):
            resolver._validate_model_package_arn("invalid-arn")
    
    def test_validate_wrong_service(self):
        """Test validation of ARN with wrong service."""
        resolver = _ModelResolver()
        
        with pytest.raises(ValueError, match="Invalid ModelPackage ARN format"):
            resolver._validate_model_package_arn("arn:aws:s3:us-west-2:123:bucket/key")


class TestGetSession:
    """Tests for _get_session method."""
    
    def test_get_existing_session(self):
        """Test returning existing session."""
        mock_session = MagicMock()
        resolver = _ModelResolver(sagemaker_session=mock_session)
        
        result = resolver._get_session()
        assert result == mock_session
    
    @patch('sagemaker.core.helper.session_helper.Session')
    def test_get_default_session(self, mock_session_class):
        """Test creating default session."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        resolver = _ModelResolver()
        result = resolver._get_session()
        
        assert result == mock_session
        mock_session_class.assert_called_once()
    
    @patch.dict(os.environ, {'SAGEMAKER_ENDPOINT': 'https://beta.endpoint'})
    @patch('boto3.client')
    @patch('sagemaker.core.helper.session_helper.Session')
    def test_get_session_with_beta_endpoint(self, mock_session_class, mock_boto_client):
        """Test creating session with beta endpoint."""
        mock_sm_client = MagicMock()
        mock_boto_client.return_value = mock_sm_client
        
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        resolver = _ModelResolver()
        result = resolver._get_session()
        
        mock_boto_client.assert_called_once_with(
            'sagemaker',
            endpoint_url='https://beta.endpoint'
        )
        mock_session_class.assert_called_once_with(sagemaker_client=mock_sm_client)


class TestResolveBaseModel:
    """Tests for _resolve_base_model convenience function."""
    
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver')
    def test_resolve_base_model_jumpstart(self, mock_resolver_class):
        """Test resolving JumpStart model."""
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver
        
        mock_info = _ModelInfo(
            base_model_name="test-model",
            base_model_arn="arn:test",
            source_model_package_arn=None,
            model_type=_ModelType.JUMPSTART,
            hub_content_name="test-model",
            additional_metadata={}
        )
        mock_resolver.resolve_model_info.return_value = mock_info
        
        result = _resolve_base_model("test-model")
        
        assert result.base_model_name == "test-model"
        assert result.model_type == _ModelType.JUMPSTART
        mock_resolver.resolve_model_info.assert_called_once_with("test-model", None)
    
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver')
    def test_resolve_base_model_with_session(self, mock_resolver_class):
        """Test resolving with custom session."""
        mock_session = MagicMock()
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver
        mock_resolver.resolve_model_info.return_value = MagicMock()
        
        _resolve_base_model("test-model", sagemaker_session=mock_session)
        
        mock_resolver_class.assert_called_once_with(mock_session)
    
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver')
    def test_resolve_base_model_with_hub_name(self, mock_resolver_class):
        """Test resolving with custom hub name."""
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver
        mock_resolver.resolve_model_info.return_value = MagicMock()
        
        _resolve_base_model("test-model", hub_name="CustomHub")
        
        mock_resolver.resolve_model_info.assert_called_once_with("test-model", "CustomHub")
