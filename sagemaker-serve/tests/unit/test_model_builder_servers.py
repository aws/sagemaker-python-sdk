"""Unit tests for _ModelBuilderServers class methods."""
import unittest
from unittest.mock import Mock, patch, MagicMock

from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.constants import SUPPORTED_MODEL_SERVERS


class TestModelBuilderServersValidation(unittest.TestCase):
    """Test _ModelBuilderServers validation logic."""

    def test_build_for_model_server_unsupported_server(self):
        """Test that unsupported model server raises ValueError."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = "UNSUPPORTED_SERVER"
        
        with self.assertRaises(ValueError) as context:
            _ModelBuilderServers._build_for_model_server(mock_builder)
        
        self.assertIn("is not supported yet", str(context.exception))
        self.assertIn("UNSUPPORTED_SERVER", str(context.exception))

    def test_build_for_model_server_missing_required_params(self):
        """Test that missing required parameters raises ValueError."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.TORCHSERVE
        mock_builder.model = None
        mock_builder.model_metadata = None
        mock_builder.inference_spec = None
        
        with self.assertRaises(ValueError) as context:
            _ModelBuilderServers._build_for_model_server(mock_builder)
        
        self.assertIn("Missing required parameter", str(context.exception))

    def test_build_for_model_server_with_model(self):
        """Test that having model parameter passes validation."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.TORCHSERVE
        mock_builder.model = Mock()  # Has a model
        mock_builder.model_metadata = None
        mock_builder.inference_spec = None
        mock_builder._build_for_torchserve = Mock(return_value=Mock())
        
        result = _ModelBuilderServers._build_for_model_server(mock_builder)
        
        mock_builder._build_for_torchserve.assert_called_once()
        self.assertIsNotNone(result)

    def test_build_for_model_server_with_mlflow_path(self):
        """Test that having MLflow path passes validation."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        from sagemaker.serve.model_format.mlflow.constants import MLFLOW_MODEL_PATH
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.TORCHSERVE
        mock_builder.model = None
        mock_builder.model_metadata = {MLFLOW_MODEL_PATH: "s3://bucket/model"}
        mock_builder.inference_spec = None
        mock_builder._build_for_torchserve = Mock(return_value=Mock())
        
        result = _ModelBuilderServers._build_for_model_server(mock_builder)
        
        mock_builder._build_for_torchserve.assert_called_once()

    def test_build_for_model_server_with_inference_spec(self):
        """Test that having inference_spec passes validation."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.TORCHSERVE
        mock_builder.model = None
        mock_builder.model_metadata = None
        mock_builder.inference_spec = Mock()  # Has inference spec
        mock_builder._build_for_torchserve = Mock(return_value=Mock())
        
        result = _ModelBuilderServers._build_for_model_server(mock_builder)
        
        mock_builder._build_for_torchserve.assert_called_once()


class TestModelBuilderServersRouting(unittest.TestCase):
    """Test _ModelBuilderServers routing to correct builder methods."""

    def setUp(self):
        """Set up common test fixtures."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.mock_builder = Mock(spec=_ModelBuilderServers)
        self.mock_builder.model = Mock()
        self.mock_builder.model_metadata = None
        self.mock_builder.inference_spec = None

    def test_routes_to_torchserve(self):
        """Test routing to TorchServe builder."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.mock_builder.model_server = ModelServer.TORCHSERVE
        self.mock_builder._build_for_torchserve = Mock(return_value=Mock())
        
        _ModelBuilderServers._build_for_model_server(self.mock_builder)
        
        self.mock_builder._build_for_torchserve.assert_called_once()

    def test_routes_to_triton(self):
        """Test routing to Triton builder."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.mock_builder.model_server = ModelServer.TRITON
        self.mock_builder._build_for_triton = Mock(return_value=Mock())
        
        _ModelBuilderServers._build_for_model_server(self.mock_builder)
        
        self.mock_builder._build_for_triton.assert_called_once()

    def test_routes_to_tensorflow_serving(self):
        """Test routing to TensorFlow Serving builder."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.mock_builder.model_server = ModelServer.TENSORFLOW_SERVING
        self.mock_builder._build_for_tensorflow_serving = Mock(return_value=Mock())
        
        _ModelBuilderServers._build_for_model_server(self.mock_builder)
        
        self.mock_builder._build_for_tensorflow_serving.assert_called_once()

    def test_routes_to_djl_serving(self):
        """Test routing to DJL Serving builder."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.mock_builder.model_server = ModelServer.DJL_SERVING
        self.mock_builder._build_for_djl = Mock(return_value=Mock())
        
        _ModelBuilderServers._build_for_model_server(self.mock_builder)
        
        self.mock_builder._build_for_djl.assert_called_once()

    def test_routes_to_tei(self):
        """Test routing to TEI builder."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.mock_builder.model_server = ModelServer.TEI
        self.mock_builder._build_for_tei = Mock(return_value=Mock())
        
        _ModelBuilderServers._build_for_model_server(self.mock_builder)
        
        self.mock_builder._build_for_tei.assert_called_once()

    def test_routes_to_tgi(self):
        """Test routing to TGI builder."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.mock_builder.model_server = ModelServer.TGI
        self.mock_builder._build_for_tgi = Mock(return_value=Mock())
        
        _ModelBuilderServers._build_for_model_server(self.mock_builder)
        
        self.mock_builder._build_for_tgi.assert_called_once()

    def test_routes_to_mms(self):
        """Test routing to MMS builder."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.mock_builder.model_server = ModelServer.MMS
        self.mock_builder._build_for_transformers = Mock(return_value=Mock())
        
        _ModelBuilderServers._build_for_model_server(self.mock_builder)
        
        self.mock_builder._build_for_transformers.assert_called_once()

    def test_routes_to_smd(self):
        """Test routing to SMD builder."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.mock_builder.model_server = ModelServer.SMD
        self.mock_builder._build_for_smd = Mock(return_value=Mock())
        
        _ModelBuilderServers._build_for_model_server(self.mock_builder)
        
        self.mock_builder._build_for_smd.assert_called_once()


class TestModelBuilderServersConstants(unittest.TestCase):
    """Test constants defined in model_builder_servers module."""

    def test_script_param_name_constant(self):
        """Test SCRIPT_PARAM_NAME constant."""
        from sagemaker.serve.model_builder_servers import SCRIPT_PARAM_NAME
        
        self.assertEqual(SCRIPT_PARAM_NAME, "sagemaker_program")

    def test_dir_param_name_constant(self):
        """Test DIR_PARAM_NAME constant."""
        from sagemaker.serve.model_builder_servers import DIR_PARAM_NAME
        
        self.assertEqual(DIR_PARAM_NAME, "sagemaker_submit_directory")

    def test_container_log_level_param_name_constant(self):
        """Test CONTAINER_LOG_LEVEL_PARAM_NAME constant."""
        from sagemaker.serve.model_builder_servers import CONTAINER_LOG_LEVEL_PARAM_NAME
        
        self.assertEqual(CONTAINER_LOG_LEVEL_PARAM_NAME, "sagemaker_container_log_level")

    def test_job_name_param_name_constant(self):
        """Test JOB_NAME_PARAM_NAME constant."""
        from sagemaker.serve.model_builder_servers import JOB_NAME_PARAM_NAME
        
        self.assertEqual(JOB_NAME_PARAM_NAME, "sagemaker_job_name")

    def test_model_server_workers_param_name_constant(self):
        """Test MODEL_SERVER_WORKERS_PARAM_NAME constant."""
        from sagemaker.serve.model_builder_servers import MODEL_SERVER_WORKERS_PARAM_NAME
        
        self.assertEqual(MODEL_SERVER_WORKERS_PARAM_NAME, "sagemaker_model_server_workers")

    def test_sagemaker_region_param_name_constant(self):
        """Test SAGEMAKER_REGION_PARAM_NAME constant."""
        from sagemaker.serve.model_builder_servers import SAGEMAKER_REGION_PARAM_NAME
        
        self.assertEqual(SAGEMAKER_REGION_PARAM_NAME, "sagemaker_region")

    def test_sagemaker_output_location_constant(self):
        """Test SAGEMAKER_OUTPUT_LOCATION constant."""
        from sagemaker.serve.model_builder_servers import SAGEMAKER_OUTPUT_LOCATION
        
        self.assertEqual(SAGEMAKER_OUTPUT_LOCATION, "sagemaker_s3_output")


class TestModelBuilderServersClass(unittest.TestCase):
    """Test _ModelBuilderServers class structure."""

    def test_model_builder_servers_is_class(self):
        """Test that _ModelBuilderServers is a class."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.assertTrue(isinstance(_ModelBuilderServers, type))

    def test_model_builder_servers_has_build_method(self):
        """Test that _ModelBuilderServers has _build_for_model_server method."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.assertTrue(hasattr(_ModelBuilderServers, '_build_for_model_server'))
        self.assertTrue(callable(getattr(_ModelBuilderServers, '_build_for_model_server')))

    def test_model_builder_servers_has_torchserve_method(self):
        """Test that _ModelBuilderServers has _build_for_torchserve method."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.assertTrue(hasattr(_ModelBuilderServers, '_build_for_torchserve'))

    def test_model_builder_servers_has_tgi_method(self):
        """Test that _ModelBuilderServers has _build_for_tgi method."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.assertTrue(hasattr(_ModelBuilderServers, '_build_for_tgi'))

    def test_model_builder_servers_has_djl_method(self):
        """Test that _ModelBuilderServers has _build_for_djl method."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.assertTrue(hasattr(_ModelBuilderServers, '_build_for_djl'))

    def test_model_builder_servers_has_triton_method(self):
        """Test that _ModelBuilderServers has _build_for_triton method."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.assertTrue(hasattr(_ModelBuilderServers, '_build_for_triton'))

    def test_model_builder_servers_has_tensorflow_method(self):
        """Test that _ModelBuilderServers has _build_for_tensorflow_serving method."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.assertTrue(hasattr(_ModelBuilderServers, '_build_for_tensorflow_serving'))

    def test_model_builder_servers_has_tei_method(self):
        """Test that _ModelBuilderServers has _build_for_tei method."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.assertTrue(hasattr(_ModelBuilderServers, '_build_for_tei'))

    def test_model_builder_servers_has_smd_method(self):
        """Test that _ModelBuilderServers has _build_for_smd method."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.assertTrue(hasattr(_ModelBuilderServers, '_build_for_smd'))

    def test_model_builder_servers_has_transformers_method(self):
        """Test that _ModelBuilderServers has _build_for_transformers method."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.assertTrue(hasattr(_ModelBuilderServers, '_build_for_transformers'))


if __name__ == "__main__":
    unittest.main()


class TestModelBuilderServersEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in _ModelBuilderServers."""

    def test_build_for_model_server_with_all_params_none(self):
        """Test that all None parameters raises ValueError."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.TORCHSERVE
        mock_builder.model = None
        mock_builder.model_metadata = {}  # Empty dict, no MLFLOW_MODEL_PATH
        mock_builder.inference_spec = None
        
        with self.assertRaises(ValueError) as context:
            _ModelBuilderServers._build_for_model_server(mock_builder)
        
        self.assertIn("Missing required parameter", str(context.exception))

    def test_build_for_model_server_with_empty_mlflow_metadata(self):
        """Test that empty MLflow metadata raises ValueError."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.TORCHSERVE
        mock_builder.model = None
        mock_builder.model_metadata = {"other_key": "value"}  # No MLFLOW_MODEL_PATH
        mock_builder.inference_spec = None
        
        with self.assertRaises(ValueError) as context:
            _ModelBuilderServers._build_for_model_server(mock_builder)
        
        self.assertIn("Missing required parameter", str(context.exception))

    def test_build_for_model_server_unsupported_raises_correct_message(self):
        """Test that unsupported server error message includes supported servers."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = "CUSTOM_SERVER"
        
        with self.assertRaises(ValueError) as context:
            _ModelBuilderServers._build_for_model_server(mock_builder)
        
        error_msg = str(context.exception)
        self.assertIn("CUSTOM_SERVER", error_msg)
        self.assertIn("is not supported yet", error_msg)
        self.assertIn("Supported model servers", error_msg)

    def test_build_for_model_server_with_model_and_inference_spec(self):
        """Test that having both model and inference_spec works."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.TORCHSERVE
        mock_builder.model = Mock()
        mock_builder.model_metadata = None
        mock_builder.inference_spec = Mock()
        mock_builder._build_for_torchserve = Mock(return_value=Mock())
        
        result = _ModelBuilderServers._build_for_model_server(mock_builder)
        
        mock_builder._build_for_torchserve.assert_called_once()
        self.assertIsNotNone(result)

    def test_build_for_model_server_with_mlflow_and_inference_spec(self):
        """Test that having both MLflow path and inference_spec works."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        from sagemaker.serve.model_format.mlflow.constants import MLFLOW_MODEL_PATH
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.DJL_SERVING
        mock_builder.model = None
        mock_builder.model_metadata = {MLFLOW_MODEL_PATH: "s3://bucket/model"}
        mock_builder.inference_spec = Mock()
        mock_builder._build_for_djl = Mock(return_value=Mock())
        
        result = _ModelBuilderServers._build_for_model_server(mock_builder)
        
        mock_builder._build_for_djl.assert_called_once()

    def test_build_for_model_server_with_all_three_params(self):
        """Test that having model, MLflow path, and inference_spec all works."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        from sagemaker.serve.model_format.mlflow.constants import MLFLOW_MODEL_PATH
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.TRITON
        mock_builder.model = Mock()
        mock_builder.model_metadata = {MLFLOW_MODEL_PATH: "s3://bucket/model"}
        mock_builder.inference_spec = Mock()
        mock_builder._build_for_triton = Mock(return_value=Mock())
        
        result = _ModelBuilderServers._build_for_model_server(mock_builder)
        
        mock_builder._build_for_triton.assert_called_once()


class TestModelBuilderServersAllModelServers(unittest.TestCase):
    """Test all supported model servers can be routed correctly."""

    def setUp(self):
        """Set up common test fixtures."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        self.mock_builder = Mock(spec=_ModelBuilderServers)
        self.mock_builder.model = Mock()
        self.mock_builder.model_metadata = None
        self.mock_builder.inference_spec = None

    def test_all_supported_model_servers_have_routes(self):
        """Test that all supported model servers have corresponding build methods."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        # Map of model servers to their expected build methods
        server_method_map = {
            ModelServer.TORCHSERVE: '_build_for_torchserve',
            ModelServer.TRITON: '_build_for_triton',
            ModelServer.TENSORFLOW_SERVING: '_build_for_tensorflow_serving',
            ModelServer.DJL_SERVING: '_build_for_djl',
            ModelServer.TEI: '_build_for_tei',
            ModelServer.TGI: '_build_for_tgi',
            ModelServer.MMS: '_build_for_transformers',
            ModelServer.SMD: '_build_for_smd',
        }
        
        for model_server, method_name in server_method_map.items():
            with self.subTest(model_server=model_server):
                self.mock_builder.model_server = model_server
                
                # Mock the specific build method
                mock_method = Mock(return_value=Mock())
                setattr(self.mock_builder, method_name, mock_method)
                
                _ModelBuilderServers._build_for_model_server(self.mock_builder)
                
                mock_method.assert_called_once()

    def test_model_server_enum_values_exist(self):
        """Test that ModelServer enum values exist and are accessible."""
        # ModelServer is an enum, so values are enum members, not strings
        from enum import Enum
        
        # Verify ModelServer has the expected attributes
        self.assertTrue(hasattr(ModelServer, 'TORCHSERVE'))
        self.assertTrue(hasattr(ModelServer, 'TRITON'))
        self.assertTrue(hasattr(ModelServer, 'TGI'))
        self.assertTrue(hasattr(ModelServer, 'DJL_SERVING'))
        
        # Verify they are enum members
        self.assertIsNotNone(ModelServer.TORCHSERVE)
        self.assertIsNotNone(ModelServer.TRITON)


class TestModelBuilderServersParameterValidation(unittest.TestCase):
    """Test parameter validation in _ModelBuilderServers."""

    def test_model_metadata_with_none_mlflow_path(self):
        """Test that model_metadata with None MLFLOW_MODEL_PATH is treated as missing."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        from sagemaker.serve.model_format.mlflow.constants import MLFLOW_MODEL_PATH
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.TORCHSERVE
        mock_builder.model = None
        mock_builder.model_metadata = {MLFLOW_MODEL_PATH: None}  # Explicitly None
        mock_builder.inference_spec = None
        
        with self.assertRaises(ValueError) as context:
            _ModelBuilderServers._build_for_model_server(mock_builder)
        
        self.assertIn("Missing required parameter", str(context.exception))

    def test_model_metadata_with_empty_string_mlflow_path(self):
        """Test that model_metadata with empty string MLFLOW_MODEL_PATH is treated as missing."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        from sagemaker.serve.model_format.mlflow.constants import MLFLOW_MODEL_PATH
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.TORCHSERVE
        mock_builder.model = None
        mock_builder.model_metadata = {MLFLOW_MODEL_PATH: ""}  # Empty string
        mock_builder.inference_spec = None
        
        # Empty string is falsy, so should raise ValueError
        with self.assertRaises(ValueError) as context:
            _ModelBuilderServers._build_for_model_server(mock_builder)
        
        self.assertIn("Missing required parameter", str(context.exception))

    def test_model_as_empty_string_is_falsy(self):
        """Test that empty string model is treated as missing."""
        from sagemaker.serve.model_builder_servers import _ModelBuilderServers
        
        mock_builder = Mock(spec=_ModelBuilderServers)
        mock_builder.model_server = ModelServer.TORCHSERVE
        mock_builder.model = ""  # Empty string is falsy
        mock_builder.model_metadata = None
        mock_builder.inference_spec = None
        
        with self.assertRaises(ValueError) as context:
            _ModelBuilderServers._build_for_model_server(mock_builder)
        
        self.assertIn("Missing required parameter", str(context.exception))


class TestModelBuilderServersConstants(unittest.TestCase):
    """Test that constants are properly defined."""

    def test_all_constants_are_strings(self):
        """Test that all parameter name constants are strings."""
        from sagemaker.serve.model_builder_servers import (
            SCRIPT_PARAM_NAME,
            DIR_PARAM_NAME,
            CONTAINER_LOG_LEVEL_PARAM_NAME,
            JOB_NAME_PARAM_NAME,
            MODEL_SERVER_WORKERS_PARAM_NAME,
            SAGEMAKER_REGION_PARAM_NAME,
            SAGEMAKER_OUTPUT_LOCATION,
        )
        
        constants = [
            SCRIPT_PARAM_NAME,
            DIR_PARAM_NAME,
            CONTAINER_LOG_LEVEL_PARAM_NAME,
            JOB_NAME_PARAM_NAME,
            MODEL_SERVER_WORKERS_PARAM_NAME,
            SAGEMAKER_REGION_PARAM_NAME,
            SAGEMAKER_OUTPUT_LOCATION,
        ]
        
        for constant in constants:
            with self.subTest(constant=constant):
                self.assertIsInstance(constant, str)
                self.assertTrue(len(constant) > 0)

    def test_constants_follow_naming_convention(self):
        """Test that constants follow expected naming conventions."""
        from sagemaker.serve.model_builder_servers import (
            SCRIPT_PARAM_NAME,
            DIR_PARAM_NAME,
            CONTAINER_LOG_LEVEL_PARAM_NAME,
            JOB_NAME_PARAM_NAME,
            MODEL_SERVER_WORKERS_PARAM_NAME,
            SAGEMAKER_REGION_PARAM_NAME,
        )
        
        # All should start with "sagemaker_"
        sagemaker_constants = [
            SCRIPT_PARAM_NAME,
            DIR_PARAM_NAME,
            CONTAINER_LOG_LEVEL_PARAM_NAME,
            JOB_NAME_PARAM_NAME,
            MODEL_SERVER_WORKERS_PARAM_NAME,
            SAGEMAKER_REGION_PARAM_NAME,
        ]
        
        for constant in sagemaker_constants:
            with self.subTest(constant=constant):
                self.assertTrue(constant.startswith("sagemaker_"))
