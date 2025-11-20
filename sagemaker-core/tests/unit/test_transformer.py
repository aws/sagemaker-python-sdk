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
from sagemaker.core.transformer import Transformer
from sagemaker.core.shapes import BatchDataCaptureConfig


@pytest.fixture
def mock_session():
    """Create a mock SageMaker session"""
    session = Mock()
    session.boto_session.region_name = "us-west-2"
    session.boto_region_name = "us-west-2"
    session.sagemaker_client = Mock()
    session.default_bucket.return_value = "test-bucket"
    session.default_bucket_prefix = "test-prefix"
    session.local_mode = False
    session.sagemaker_config = {}
    return session


class TestTransformer:
    """Test cases for Transformer class"""

    def test_init_with_minimal_params(self, mock_session):
        """Test initialization with minimal parameters"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        assert transformer.model_name == "test-model"
        assert transformer.instance_count == 1
        assert transformer.instance_type == "ml.m5.xlarge"
        assert transformer.strategy is None
        assert transformer.output_path is None

    def test_init_with_all_params(self, mock_session):
        """Test initialization with all parameters"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=2,
            instance_type="ml.m5.xlarge",
            strategy="MultiRecord",
            assemble_with="Line",
            output_path="s3://bucket/output",
            output_kms_key="output-key",
            accept="application/json",
            max_concurrent_transforms=4,
            max_payload=10,
            tags=[{"Key": "test", "Value": "value"}],
            env={"TEST_VAR": "value"},
            base_transform_job_name="test-job",
            sagemaker_session=mock_session,
            volume_kms_key="volume-key"
        )
        
        assert transformer.strategy == "MultiRecord"
        assert transformer.assemble_with == "Line"
        assert transformer.output_path == "s3://bucket/output"
        assert transformer.output_kms_key == "output-key"
        assert transformer.accept == "application/json"
        assert transformer.max_concurrent_transforms == 4
        assert transformer.max_payload == 10
        assert transformer.volume_kms_key == "volume-key"

    def test_format_inputs_to_input_config(self, mock_session):
        """Test _format_inputs_to_input_config method"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        config = transformer._format_inputs_to_input_config(
            data="s3://bucket/input",
            data_type="S3Prefix",
            content_type="text/csv",
            compression_type="Gzip",
            split_type="Line"
        )
        
        assert config["data_source"].s3_data_source.s3_uri == "s3://bucket/input"
        assert config["data_source"].s3_data_source.s3_data_type == "S3Prefix"
        assert config["content_type"] == "text/csv"
        assert config["compression_type"] == "Gzip"
        assert config["split_type"] == "Line"

    def test_format_inputs_to_input_config_minimal(self, mock_session):
        """Test _format_inputs_to_input_config with minimal params"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        config = transformer._format_inputs_to_input_config(
            data="s3://bucket/input",
            data_type="S3Prefix",
            content_type=None,
            compression_type=None,
            split_type=None
        )
        
        assert config["data_source"].s3_data_source.s3_uri == "s3://bucket/input"
        assert "content_type" not in config
        assert "compression_type" not in config
        assert "split_type" not in config

    def test_prepare_output_config(self, mock_session):
        """Test _prepare_output_config method"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        config = transformer._prepare_output_config(
            s3_path="s3://bucket/output",
            kms_key_id="kms-key",
            assemble_with="Line",
            accept="application/json"
        )
        
        assert config["s3_output_path"] == "s3://bucket/output"
        assert config["kms_key_id"] == "kms-key"
        assert config["assemble_with"] == "Line"
        assert config["accept"] == "application/json"

    def test_prepare_output_config_minimal(self, mock_session):
        """Test _prepare_output_config with minimal params"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        config = transformer._prepare_output_config(
            s3_path="s3://bucket/output",
            kms_key_id=None,
            assemble_with=None,
            accept=None
        )
        
        assert config["s3_output_path"] == "s3://bucket/output"
        assert "kms_key_id" not in config
        assert "assemble_with" not in config
        assert "accept" not in config

    def test_prepare_resource_config(self, mock_session):
        """Test _prepare_resource_config method"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        config = transformer._prepare_resource_config(
            instance_count=2,
            instance_type="ml.m5.xlarge",
            volume_kms_key="volume-key"
        )
        
        assert config["instance_count"] == 2
        assert config["instance_type"] == "ml.m5.xlarge"
        assert config["volume_kms_key_id"] == "volume-key"

    def test_prepare_resource_config_no_kms(self, mock_session):
        """Test _prepare_resource_config without KMS key"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        config = transformer._prepare_resource_config(
            instance_count=1,
            instance_type="ml.m5.xlarge",
            volume_kms_key=None
        )
        
        assert config["instance_count"] == 1
        assert config["instance_type"] == "ml.m5.xlarge"
        assert "volume_kms_key_id" not in config

    def test_prepare_data_processing_all_params(self, mock_session):
        """Test _prepare_data_processing with all parameters"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        data_processing = transformer._prepare_data_processing(
            input_filter="$.features",
            output_filter="$.prediction",
            join_source="Input"
        )
        
        assert data_processing is not None
        assert data_processing.input_filter == "$.features"
        assert data_processing.output_filter == "$.prediction"
        assert data_processing.join_source == "Input"

    def test_prepare_data_processing_none(self, mock_session):
        """Test _prepare_data_processing with no parameters"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        data_processing = transformer._prepare_data_processing(
            input_filter=None,
            output_filter=None,
            join_source=None
        )
        
        assert data_processing is None

    def test_prepare_data_processing_partial(self, mock_session):
        """Test _prepare_data_processing with partial parameters"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        data_processing = transformer._prepare_data_processing(
            input_filter="$.features",
            output_filter=None,
            join_source=None
        )
        
        assert data_processing is not None
        assert data_processing.input_filter == "$.features"

    @patch("sagemaker.core.transformer.Model")
    def test_retrieve_image_uri_success(self, mock_model_class, mock_session):
        """Test _retrieve_image_uri with successful model retrieval"""
        mock_primary_container = Mock()
        mock_primary_container.image = "test-image:latest"
        
        class DictWithAttrs(dict):
            """A dict that also supports attribute access"""
            def __getattr__(self, name):
                return self.get(name)
        
        class MockModel:
            def __init__(self):
                self.__dict__ = DictWithAttrs()
                self.__dict__['primary_container'] = mock_primary_container
                self.__dict__['containers'] = None
        
        mock_model = MockModel()
        mock_model_class.get.return_value = mock_model
        
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        image_uri = transformer._retrieve_image_uri()
        assert image_uri == "test-image:latest"

    @patch("sagemaker.core.transformer.Model")
    def test_retrieve_image_uri_with_containers(self, mock_model_class, mock_session):
        """Test _retrieve_image_uri with containers instead of primary_container"""
        mock_container = Mock()
        mock_container.image = "container-image:latest"
        
        class DictWithAttrs(dict):
            """A dict that also supports attribute access"""
            def __getattr__(self, name):
                return self.get(name)
        
        class MockModel:
            def __init__(self):
                self.__dict__ = DictWithAttrs()
                self.__dict__['primary_container'] = None
                self.__dict__['containers'] = [mock_container]
        
        mock_model = MockModel()
        mock_model_class.get.return_value = mock_model
        
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        image_uri = transformer._retrieve_image_uri()
        assert image_uri == "container-image:latest"

    @patch("sagemaker.core.transformer.Model")
    def test_retrieve_image_uri_no_model(self, mock_model_class, mock_session):
        """Test _retrieve_image_uri when model doesn't exist"""
        mock_model_class.get.return_value = None
        
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        image_uri = transformer._retrieve_image_uri()
        assert image_uri is None

    def test_retrieve_base_name_with_image(self, mock_session):
        """Test _retrieve_base_name when image URI is available"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        with patch.object(transformer, "_retrieve_image_uri", return_value="my-image:latest"):
            base_name = transformer._retrieve_base_name()
            assert base_name == "my-image"

    def test_retrieve_base_name_no_image(self, mock_session):
        """Test _retrieve_base_name when no image URI is available"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        with patch.object(transformer, "_retrieve_image_uri", return_value=None):
            base_name = transformer._retrieve_base_name()
            assert base_name == "test-model"

    def test_ensure_last_transform_job_raises_error(self, mock_session):
        """Test _ensure_last_transform_job raises error when no job exists"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        with pytest.raises(ValueError, match="No transform job available"):
            transformer._ensure_last_transform_job()

    @patch("sagemaker.core.transformer.TransformJob")
    def test_attach_success(self, mock_transform_job_class, mock_session):
        """Test attach method with successful job retrieval"""
        mock_resources = Mock()
        mock_resources.instance_count = 1
        mock_resources.instance_type = "ml.m5.xlarge"
        mock_resources.volume_kms_key_id = "volume-key"
        
        mock_output = Mock()
        mock_output.assemble_with = "Line"
        mock_output.s3_output_path = "s3://bucket/output"
        mock_output.kms_key_id = "output-key"
        mock_output.accept = "application/json"
        
        class MockJob:
            pass
        
        mock_job = MockJob()
        mock_job.__dict__ = {
            "model_name": "test-model",
            "transform_resources": mock_resources,
            "batch_strategy": "MultiRecord",
            "transform_output": mock_output,
            "max_concurrent_transforms": 4,
            "max_payload_in_mb": 10,
            "transform_job_name": "test-job-123"
        }
        mock_transform_job_class.get.return_value = mock_job
        
        transformer = Transformer.attach("test-job-123", mock_session)
        
        assert transformer.model_name == "test-model"
        assert transformer.instance_count == 1
        assert transformer.instance_type == "ml.m5.xlarge"
        assert transformer.strategy == "MultiRecord"

    @patch("sagemaker.core.transformer.TransformJob")
    def test_attach_job_not_found(self, mock_transform_job_class, mock_session):
        """Test attach method when job is not found"""
        mock_transform_job_class.get.return_value = None
        
        with pytest.raises(ValueError, match="Transform job .* not found"):
            Transformer.attach("nonexistent-job", mock_session)

    def test_prepare_init_params_from_job_description(self, mock_session):
        """Test _prepare_init_params_from_job_description method"""
        job_details = {
            "model_name": "test-model",
            "transform_resources": Mock(
                instance_count=2,
                instance_type="ml.m5.xlarge",
                volume_kms_key_id="volume-key"
            ),
            "batch_strategy": "SingleRecord",
            "transform_output": Mock(
                assemble_with="None",
                s3_output_path="s3://bucket/output",
                kms_key_id="output-key",
                accept="text/csv"
            ),
            "max_concurrent_transforms": 8,
            "max_payload_in_mb": 20,
            "transform_job_name": "test-job-456"
        }
        
        init_params = Transformer._prepare_init_params_from_job_description(job_details)
        
        assert init_params["model_name"] == "test-model"
        assert init_params["instance_count"] == 2
        assert init_params["instance_type"] == "ml.m5.xlarge"
        assert init_params["strategy"] == "SingleRecord"
        assert init_params["assemble_with"] == "None"
        assert init_params["output_path"] == "s3://bucket/output"
        assert init_params["output_kms_key"] == "output-key"
        assert init_params["accept"] == "text/csv"
        assert init_params["max_concurrent_transforms"] == 8
        assert init_params["max_payload"] == 20
        assert init_params["volume_kms_key"] == "volume-key"
        assert init_params["base_transform_job_name"] == "test-job-456"

    def test_delete_model(self, mock_session):
        """Test delete_model method"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        with patch("sagemaker.core.transformer.Model") as mock_model_class:
            mock_model = Mock()
            mock_model_class.get.return_value = mock_model
            
            transformer.delete_model()
            
            mock_model.delete.assert_called_once()

    def test_delete_model_no_model(self, mock_session):
        """Test delete_model when model doesn't exist"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session
        )
        
        with patch("sagemaker.core.transformer.Model") as mock_model_class:
            mock_model_class.get.return_value = None
            
            # Should not raise an error
            transformer.delete_model()

    def test_get_transform_args(self, mock_session):
        """Test _get_transform_args method"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            strategy="MultiRecord",
            max_concurrent_transforms=4,
            max_payload=10,
            env={"TEST": "value"},
            tags=[{"Key": "test", "Value": "value"}],
            sagemaker_session=mock_session
        )
        
        transformer._current_job_name = "test-job-123"
        
        args = transformer._get_transform_args(
            data="s3://bucket/input",
            data_type="S3Prefix",
            content_type="text/csv",
            compression_type=None,
            split_type="Line",
            input_filter=None,
            output_filter=None,
            join_source=None,
            experiment_config=None,
            model_client_config=None,
            batch_data_capture_config=None
        )
        
        assert args["job_name"] == "test-job-123"
        assert args["model_name"] == "test-model"
        assert args["strategy"] == "MultiRecord"
        assert args["max_concurrent_transforms"] == 4
        assert args["max_payload"] == 10
        assert args["env"] == {"TEST": "value"}

    def test_load_config(self, mock_session):
        """Test _load_config method"""
        transformer = Transformer(
            model_name="test-model",
            instance_count=2,
            instance_type="ml.m5.xlarge",
            output_path="s3://bucket/output",
            output_kms_key="output-key",
            assemble_with="Line",
            accept="application/json",
            volume_kms_key="volume-key",
            sagemaker_session=mock_session
        )
        
        config = transformer._load_config(
            data="s3://bucket/input",
            data_type="S3Prefix",
            content_type="text/csv",
            compression_type="Gzip",
            split_type="Line"
        )
        
        assert "input_config" in config
        assert "output_config" in config
        assert "resource_config" in config
        assert config["output_config"]["s3_output_path"] == "s3://bucket/output"
        assert config["resource_config"]["instance_count"] == 2
