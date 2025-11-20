"""Unit tests for ModelBuilder utility methods that don't require complex initialization."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import packaging.version


class TestModelBuilderMmsVersion(unittest.TestCase):
    """Test ModelBuilder._is_mms_version method."""

    def test_is_mms_version_with_valid_version(self):
        """Test _is_mms_version with version >= 1.2."""
        # Create a minimal mock object with just the attributes we need
        mock_builder = Mock()
        mock_builder.framework_version = "1.5.0"
        
        # Import the method we want to test
        from sagemaker.serve.model_builder import ModelBuilder, _LOWEST_MMS_VERSION
        
        # Call the method directly
        result = ModelBuilder._is_mms_version(mock_builder)
        
        self.assertTrue(result)

    def test_is_mms_version_with_exact_lowest_version(self):
        """Test _is_mms_version with exact lowest MMS version."""
        mock_builder = Mock()
        mock_builder.framework_version = "1.2"
        
        from sagemaker.serve.model_builder import ModelBuilder
        
        result = ModelBuilder._is_mms_version(mock_builder)
        
        self.assertTrue(result)

    def test_is_mms_version_with_lower_version(self):
        """Test _is_mms_version with version < 1.2."""
        mock_builder = Mock()
        mock_builder.framework_version = "1.1.0"
        
        from sagemaker.serve.model_builder import ModelBuilder
        
        result = ModelBuilder._is_mms_version(mock_builder)
        
        self.assertFalse(result)

    def test_is_mms_version_with_none(self):
        """Test _is_mms_version with None framework_version."""
        mock_builder = Mock()
        mock_builder.framework_version = None
        
        from sagemaker.serve.model_builder import ModelBuilder
        
        result = ModelBuilder._is_mms_version(mock_builder)
        
        self.assertFalse(result)

    def test_is_mms_version_with_higher_version(self):
        """Test _is_mms_version with much higher version."""
        mock_builder = Mock()
        mock_builder.framework_version = "2.0.0"
        
        from sagemaker.serve.model_builder import ModelBuilder
        
        result = ModelBuilder._is_mms_version(mock_builder)
        
        self.assertTrue(result)


class TestModelBuilderContainerEnv(unittest.TestCase):
    """Test ModelBuilder._get_container_env method."""

    def test_get_container_env_without_log_level(self):
        """Test _get_container_env when _container_log_level is not set."""
        mock_builder = Mock()
        mock_builder._container_log_level = None
        mock_builder.env = {"KEY1": "value1"}
        
        from sagemaker.serve.model_builder import ModelBuilder
        
        result = ModelBuilder._get_container_env(mock_builder)
        
        self.assertEqual(result, {"KEY1": "value1"})

    def test_get_container_env_with_valid_log_level(self):
        """Test _get_container_env with valid log level."""
        mock_builder = Mock()
        mock_builder._container_log_level = 20  # INFO level
        mock_builder.env = {"KEY1": "value1"}
        mock_builder.LOG_LEVEL_MAP = {
            10: "DEBUG",
            20: "INFO",
            30: "WARNING",
            40: "ERROR"
        }
        mock_builder.LOG_LEVEL_PARAM_NAME = "SAGEMAKER_CONTAINER_LOG_LEVEL"
        
        from sagemaker.serve.model_builder import ModelBuilder
        
        result = ModelBuilder._get_container_env(mock_builder)
        
        self.assertIn("SAGEMAKER_CONTAINER_LOG_LEVEL", result)
        self.assertEqual(result["SAGEMAKER_CONTAINER_LOG_LEVEL"], "INFO")
        self.assertEqual(result["KEY1"], "value1")

    @patch('logging.warning')
    def test_get_container_env_with_invalid_log_level(self, mock_warning):
        """Test _get_container_env with invalid log level."""
        mock_builder = Mock()
        mock_builder._container_log_level = 999  # Invalid level
        mock_builder.env = {"KEY1": "value1"}
        mock_builder.LOG_LEVEL_MAP = {20: "INFO", 30: "WARNING"}
        
        from sagemaker.serve.model_builder import ModelBuilder
        
        result = ModelBuilder._get_container_env(mock_builder)
        
        # Should return original env without modification
        self.assertEqual(result, {"KEY1": "value1"})
        # Should log a warning
        mock_warning.assert_called_once()


class TestModelBuilderPrepareContainerDef(unittest.TestCase):
    """Test ModelBuilder._prepare_container_def_base method."""

    def test_prepare_container_def_with_pipeline_models(self):
        """Test _prepare_container_def_base with list of Model objects."""
        from sagemaker.core.resources import Model
        
        mock_builder = Mock()
        mock_model1 = Mock(spec=Model)
        mock_model2 = Mock(spec=Model)
        mock_builder.model = [mock_model1, mock_model2]
        mock_builder._prepare_pipeline_container_defs = Mock(return_value=[{"Image": "img1"}, {"Image": "img2"}])
        
        from sagemaker.serve.model_builder import ModelBuilder
        
        result = ModelBuilder._prepare_container_def_base(mock_builder)
        
        mock_builder._prepare_pipeline_container_defs.assert_called_once()
        self.assertEqual(len(result), 2)

    def test_prepare_container_def_with_invalid_pipeline_models(self):
        """Test _prepare_container_def_base with invalid list elements."""
        mock_builder = Mock()
        mock_builder.model = ["not_a_model", "also_not_a_model"]
        
        from sagemaker.serve.model_builder import ModelBuilder
        
        with self.assertRaises(ValueError) as context:
            ModelBuilder._prepare_container_def_base(mock_builder)
        
        self.assertIn("must be sagemaker.core.resources.Model", str(context.exception))


class TestModelBuilderConstants(unittest.TestCase):
    """Test ModelBuilder module constants."""

    def test_lowest_mms_version_constant(self):
        """Test that _LOWEST_MMS_VERSION is defined correctly."""
        from sagemaker.serve.model_builder import _LOWEST_MMS_VERSION
        
        self.assertEqual(_LOWEST_MMS_VERSION, "1.2")

    def test_script_param_name_constant(self):
        """Test SCRIPT_PARAM_NAME constant."""
        from sagemaker.serve.model_builder import SCRIPT_PARAM_NAME
        
        self.assertEqual(SCRIPT_PARAM_NAME, "sagemaker_program")

    def test_dir_param_name_constant(self):
        """Test DIR_PARAM_NAME constant."""
        from sagemaker.serve.model_builder import DIR_PARAM_NAME
        
        self.assertEqual(DIR_PARAM_NAME, "sagemaker_submit_directory")

    def test_container_log_level_param_name_constant(self):
        """Test CONTAINER_LOG_LEVEL_PARAM_NAME constant."""
        from sagemaker.serve.model_builder import CONTAINER_LOG_LEVEL_PARAM_NAME
        
        self.assertEqual(CONTAINER_LOG_LEVEL_PARAM_NAME, "sagemaker_container_log_level")

    def test_job_name_param_name_constant(self):
        """Test JOB_NAME_PARAM_NAME constant."""
        from sagemaker.serve.model_builder import JOB_NAME_PARAM_NAME
        
        self.assertEqual(JOB_NAME_PARAM_NAME, "sagemaker_job_name")

    def test_model_server_workers_param_name_constant(self):
        """Test MODEL_SERVER_WORKERS_PARAM_NAME constant."""
        from sagemaker.serve.model_builder import MODEL_SERVER_WORKERS_PARAM_NAME
        
        self.assertEqual(MODEL_SERVER_WORKERS_PARAM_NAME, "sagemaker_model_server_workers")

    def test_sagemaker_region_param_name_constant(self):
        """Test SAGEMAKER_REGION_PARAM_NAME constant."""
        from sagemaker.serve.model_builder import SAGEMAKER_REGION_PARAM_NAME
        
        self.assertEqual(SAGEMAKER_REGION_PARAM_NAME, "sagemaker_region")

    def test_sagemaker_output_location_constant(self):
        """Test SAGEMAKER_OUTPUT_LOCATION constant."""
        from sagemaker.serve.model_builder import SAGEMAKER_OUTPUT_LOCATION
        
        self.assertEqual(SAGEMAKER_OUTPUT_LOCATION, "sagemaker_s3_output")


class TestModelBuilderDataclass(unittest.TestCase):
    """Test ModelBuilder dataclass structure."""

    def test_modelbuilder_is_dataclass(self):
        """Test that ModelBuilder is a dataclass."""
        from sagemaker.serve.model_builder import ModelBuilder
        from dataclasses import is_dataclass
        
        self.assertTrue(is_dataclass(ModelBuilder))

    def test_modelbuilder_has_model_field(self):
        """Test that ModelBuilder has model field."""
        from sagemaker.serve.model_builder import ModelBuilder
        from dataclasses import fields
        
        field_names = [f.name for f in fields(ModelBuilder)]
        self.assertIn('model', field_names)

    def test_modelbuilder_has_mode_field(self):
        """Test that ModelBuilder has mode field."""
        from sagemaker.serve.model_builder import ModelBuilder
        from dataclasses import fields
        
        field_names = [f.name for f in fields(ModelBuilder)]
        self.assertIn('mode', field_names)

    def test_modelbuilder_has_inference_spec_field(self):
        """Test that ModelBuilder has inference_spec field."""
        from sagemaker.serve.model_builder import ModelBuilder
        from dataclasses import fields
        
        field_names = [f.name for f in fields(ModelBuilder)]
        self.assertIn('inference_spec', field_names)

    def test_modelbuilder_has_schema_builder_field(self):
        """Test that ModelBuilder has schema_builder field."""
        from sagemaker.serve.model_builder import ModelBuilder
        from dataclasses import fields
        
        field_names = [f.name for f in fields(ModelBuilder)]
        self.assertIn('schema_builder', field_names)

    def test_modelbuilder_has_role_arn_field(self):
        """Test that ModelBuilder has role_arn field."""
        from sagemaker.serve.model_builder import ModelBuilder
        from dataclasses import fields
        
        field_names = [f.name for f in fields(ModelBuilder)]
        self.assertIn('role_arn', field_names)

    def test_modelbuilder_has_image_uri_field(self):
        """Test that ModelBuilder has image_uri field."""
        from sagemaker.serve.model_builder import ModelBuilder
        from dataclasses import fields
        
        field_names = [f.name for f in fields(ModelBuilder)]
        self.assertIn('image_uri', field_names)

    def test_modelbuilder_has_model_server_field(self):
        """Test that ModelBuilder has model_server field."""
        from sagemaker.serve.model_builder import ModelBuilder
        from dataclasses import fields
        
        field_names = [f.name for f in fields(ModelBuilder)]
        self.assertIn('model_server', field_names)

    def test_modelbuilder_has_env_vars_field(self):
        """Test that ModelBuilder has env_vars field."""
        from sagemaker.serve.model_builder import ModelBuilder
        from dataclasses import fields
        
        field_names = [f.name for f in fields(ModelBuilder)]
        self.assertIn('env_vars', field_names)


if __name__ == "__main__":
    unittest.main()
