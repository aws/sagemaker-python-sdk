"""
Unit tests for sagemaker.serve.local_resources module.

Tests local endpoint and endpoint configuration classes for V3 ModelBuilder local mode support.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import datetime
import io
import json

from sagemaker.serve.local_resources import (
    InvokeEndpointOutput,
    LocalEndpoint,
    LocalEndpointConfig,
    _get_container_config,
    DEFAULT_SERIALIZERS_BY_SERVER
)
from sagemaker.serve.utils.types import ModelServer


class TestInvokeEndpointOutput(unittest.TestCase):
    """Test InvokeEndpointOutput class."""

    def test_init_with_defaults(self):
        """Test initialization with default content type."""
        body = b'{"result": "success"}'
        
        output = InvokeEndpointOutput(body=body)
        
        self.assertEqual(output.body, body)
        self.assertEqual(output.content_type, "application/json")

    def test_init_with_custom_content_type(self):
        """Test initialization with custom content type."""
        body = b'binary data'
        content_type = "application/octet-stream"
        
        output = InvokeEndpointOutput(body=body, content_type=content_type)
        
        self.assertEqual(output.body, body)
        self.assertEqual(output.content_type, content_type)


class TestLocalEndpointInitialization(unittest.TestCase):
    """Test LocalEndpoint initialization."""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        mock_session = Mock()
        
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=mock_session,
            local_model=Mock(),
            in_process_mode=True,
            model_server=ModelServer.TORCHSERVE
        )
        
        self.assertEqual(endpoint.endpoint_name, "test-endpoint")
        self.assertEqual(endpoint.endpoint_config_name, "test-config")
        self.assertTrue(endpoint.in_process_mode)
        self.assertEqual(endpoint.model_server, ModelServer.TORCHSERVE)
        self.assertIsInstance(endpoint.creation_time, datetime.datetime)

    @patch('sagemaker.core.local.local_session.LocalSession')
    def test_init_creates_local_session_if_none(self, mock_local_session_class):
        """Test that LocalSession is created if not provided."""
        mock_session = Mock()
        mock_local_session_class.return_value = mock_session
        
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config"
        )
        
        mock_local_session_class.assert_called_once()
        self.assertEqual(endpoint._local_session, mock_session)


class TestLocalEndpointStatus(unittest.TestCase):
    """Test LocalEndpoint status property."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=self.mock_session
        )

    def test_endpoint_status_in_service(self):
        """Test endpoint status when endpoint is in service."""
        self.mock_session.sagemaker_client.describe_endpoint.return_value = {
            "EndpointStatus": "InService"
        }
        
        status = self.endpoint.endpoint_status
        
        self.assertEqual(status, "InService")
        self.mock_session.sagemaker_client.describe_endpoint.assert_called_once_with(
            EndpointName="test-endpoint"
        )

    def test_endpoint_status_creating(self):
        """Test endpoint status when endpoint is creating."""
        self.mock_session.sagemaker_client.describe_endpoint.return_value = {
            "EndpointStatus": "Creating"
        }
        
        status = self.endpoint.endpoint_status
        
        self.assertEqual(status, "Creating")

    def test_endpoint_status_failed_on_exception(self):
        """Test endpoint status returns Failed on exception."""
        self.mock_session.sagemaker_client.describe_endpoint.side_effect = Exception("Not found")
        
        status = self.endpoint.endpoint_status
        
        self.assertEqual(status, "Failed")


class TestLocalEndpointInvoke(unittest.TestCase):
    """Test LocalEndpoint invoke method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()

    @patch('sagemaker.core.deserializers.JSONDeserializer.deserialize')
    def test_invoke_in_process_mode(self, mock_deserialize):
        """Test invoke in in-process mode."""
        mock_in_process_obj = Mock()
        mock_in_process_obj._invoke_serving.return_value = b'{"result": "success"}'
        mock_deserialize.return_value = {"result": "success"}
        
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=self.mock_session,
            in_process_mode=True,
            in_process_mode_obj=mock_in_process_obj
        )
        
        result = endpoint.invoke(body={"input": "test"})
        
        self.assertIsInstance(result, InvokeEndpointOutput)
        mock_in_process_obj._invoke_serving.assert_called_once()

    def test_invoke_in_process_mode_without_obj_raises_error(self):
        """Test invoke in in-process mode without object raises error."""
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=self.mock_session,
            in_process_mode=True,
            in_process_mode_obj=None
        )
        
        with self.assertRaises(ValueError) as context:
            endpoint.invoke(body={"input": "test"})
        
        self.assertIn("In Process container mode not available", str(context.exception))

    def test_invoke_torchserve(self):
        """Test invoke with TorchServe model server."""
        mock_container_obj = Mock()
        mock_container_obj._invoke_torch_serve.return_value = b'{"predictions": [0.9]}'
        
        # Mock the deserializer to avoid the content_type issue
        mock_deserializer = Mock()
        mock_deserializer.deserialize.return_value = {"predictions": [0.9]}
        mock_deserializer.ACCEPT = "application/octet-stream"
        
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=self.mock_session,
            in_process_mode=False,
            local_container_mode_obj=mock_container_obj,
            model_server=ModelServer.TORCHSERVE,
            deserializer=mock_deserializer
        )
        
        result = endpoint.invoke(body={"input": "test"})
        
        self.assertIsInstance(result, InvokeEndpointOutput)
        mock_container_obj._invoke_torch_serve.assert_called_once()

    def test_invoke_djl_serving(self):
        """Test invoke with DJL Serving model server."""
        mock_container_obj = Mock()
        mock_container_obj._invoke_djl_serving.return_value = b'{"generated_text": "Hello"}'
        
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=self.mock_session,
            in_process_mode=False,
            local_container_mode_obj=mock_container_obj,
            model_server=ModelServer.DJL_SERVING
        )
        
        result = endpoint.invoke(body={"inputs": "test"})
        
        self.assertIsInstance(result, InvokeEndpointOutput)
        mock_container_obj._invoke_djl_serving.assert_called_once()

    def test_invoke_tgi(self):
        """Test invoke with TGI model server."""
        mock_container_obj = Mock()
        mock_container_obj._invoke_tgi_serving.return_value = b'{"generated_text": "Hello"}'
        
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=self.mock_session,
            in_process_mode=False,
            local_container_mode_obj=mock_container_obj,
            model_server=ModelServer.TGI
        )
        
        result = endpoint.invoke(body={"inputs": "test"})
        
        self.assertIsInstance(result, InvokeEndpointOutput)
        mock_container_obj._invoke_tgi_serving.assert_called_once()

    @patch('sagemaker.core.deserializers.JSONDeserializer.deserialize')
    def test_invoke_tensorflow_serving(self, mock_deserialize):
        """Test invoke with TensorFlow Serving model server."""
        mock_container_obj = Mock()
        mock_container_obj._invoke_tensorflow_serving.return_value = b'{"predictions": [[0.1, 0.9]]}'
        mock_deserialize.return_value = {"predictions": [[0.1, 0.9]]}
        
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=self.mock_session,
            in_process_mode=False,
            local_container_mode_obj=mock_container_obj,
            model_server=ModelServer.TENSORFLOW_SERVING
        )
        
        result = endpoint.invoke(body={"instances": [[1, 2, 3]]})
        
        self.assertIsInstance(result, InvokeEndpointOutput)
        mock_container_obj._invoke_tensorflow_serving.assert_called_once()

    def test_invoke_without_model_server_raises_error(self):
        """Test invoke without model server raises error."""
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=self.mock_session,
            in_process_mode=False,
            model_server=None
        )
        
        with self.assertRaises(ValueError) as context:
            endpoint.invoke(body={"input": "test"})
        
        self.assertIn("Model server or container mode not available", str(context.exception))

    def test_invoke_unsupported_model_server_raises_error(self):
        """Test invoke with unsupported model server raises error."""
        mock_container_obj = Mock()
        
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=self.mock_session,
            in_process_mode=False,
            local_container_mode_obj=mock_container_obj,
            model_server="UNSUPPORTED_SERVER"
        )
        
        with self.assertRaises(ValueError) as context:
            endpoint.invoke(body={"input": "test"})
        
        self.assertIn("Unsupported model server", str(context.exception))


if __name__ == '__main__':
    unittest.main()


class TestLocalEndpointCreate(unittest.TestCase):
    """Test LocalEndpoint create class method."""

    @patch('sagemaker.core.local.local_session.LocalSession')
    def test_create_in_process_mode(self, mock_local_session_class):
        """Test creating endpoint in in-process mode."""
        mock_session = Mock()
        mock_local_session_class.return_value = mock_session
        mock_in_process_obj = Mock()
        mock_model = Mock()
        mock_model.model_name = "test-model"
        
        endpoint = LocalEndpoint.create(
            endpoint_name="test-endpoint",
            local_model=mock_model,
            local_session=mock_session,
            in_process_mode=True,
            in_process_mode_obj=mock_in_process_obj
        )
        
        self.assertEqual(endpoint.endpoint_name, "test-endpoint")
        self.assertTrue(endpoint.in_process_mode)
        mock_in_process_obj.create_server.assert_called_once()

    @patch('sagemaker.serve.local_resources._get_container_config')
    @patch('sagemaker.core.local.local_session.LocalSession')
    def test_create_container_mode(self, mock_local_session_class, mock_get_config):
        """Test creating endpoint in container mode."""
        mock_session = Mock()
        mock_local_session_class.return_value = mock_session
        mock_container_obj = Mock()
        mock_container_obj.model_path = "/path/to/model"
        mock_model = Mock()
        mock_model.model_name = "test-model"
        mock_model.primary_container.image = "test-image:latest"
        mock_model.primary_container.environment = {"KEY": "value"}
        mock_get_config.return_value = {"network_mode": "host"}
        
        endpoint = LocalEndpoint.create(
            endpoint_name="test-endpoint",
            local_model=mock_model,
            local_session=mock_session,
            in_process_mode=False,
            local_container_mode_obj=mock_container_obj,
            model_server=ModelServer.TORCHSERVE
        )
        
        self.assertEqual(endpoint.endpoint_name, "test-endpoint")
        self.assertFalse(endpoint.in_process_mode)
        mock_container_obj.create_server.assert_called_once()
        mock_session.sagemaker_client.create_endpoint_config.assert_called_once()
        mock_session.sagemaker_client.create_endpoint.assert_called_once()

    @patch('sagemaker.core.local.local_session.LocalSession')
    def test_create_without_session_creates_one(self, mock_local_session_class):
        """Test that create creates LocalSession if not provided."""
        mock_session = Mock()
        mock_local_session_class.return_value = mock_session
        mock_in_process_obj = Mock()
        mock_model = Mock()
        
        endpoint = LocalEndpoint.create(
            endpoint_name="test-endpoint",
            local_model=mock_model,
            in_process_mode=True,
            in_process_mode_obj=mock_in_process_obj
        )
        
        mock_local_session_class.assert_called()


class TestLocalEndpointGet(unittest.TestCase):
    """Test LocalEndpoint get class method."""

    @patch('sagemaker.core.local.local_session.LocalSession')
    def test_get_existing_endpoint(self, mock_local_session_class):
        """Test getting an existing endpoint."""
        mock_session = Mock()
        mock_local_session_class.return_value = mock_session
        mock_session.sagemaker_client.describe_endpoint.return_value = {
            "EndpointName": "test-endpoint",
            "EndpointConfigName": "test-config",
            "EndpointStatus": "InService"
        }
        
        endpoint = LocalEndpoint.get("test-endpoint", local_session=mock_session)
        
        self.assertIsNotNone(endpoint)
        self.assertEqual(endpoint.endpoint_name, "test-endpoint")
        self.assertEqual(endpoint.endpoint_config_name, "test-config")

    @patch('sagemaker.core.local.local_session.LocalSession')
    def test_get_nonexistent_endpoint_returns_none(self, mock_local_session_class):
        """Test getting a non-existent endpoint returns None."""
        mock_session = Mock()
        mock_local_session_class.return_value = mock_session
        mock_session.sagemaker_client.describe_endpoint.side_effect = Exception("Not found")
        
        endpoint = LocalEndpoint.get("nonexistent-endpoint", local_session=mock_session)
        
        self.assertIsNone(endpoint)

    @patch('sagemaker.core.local.local_session.LocalSession')
    def test_get_without_session_creates_one(self, mock_local_session_class):
        """Test that get creates LocalSession if not provided."""
        mock_session = Mock()
        mock_local_session_class.return_value = mock_session
        mock_session.sagemaker_client.describe_endpoint.return_value = {
            "EndpointName": "test-endpoint",
            "EndpointConfigName": "test-config"
        }
        
        endpoint = LocalEndpoint.get("test-endpoint")
        
        mock_local_session_class.assert_called()


class TestLocalEndpointRefresh(unittest.TestCase):
    """Test LocalEndpoint refresh method."""

    def test_refresh_updates_attributes(self):
        """Test that refresh updates endpoint attributes."""
        mock_session = Mock()
        mock_session.sagemaker_client.describe_endpoint.return_value = {
            "EndpointName": "test-endpoint",
            "EndpointConfigName": "updated-config",
            "EndpointStatus": "InService"
        }
        
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="old-config",
            local_session=mock_session
        )
        
        refreshed = endpoint.refresh()
        
        self.assertEqual(refreshed.endpoint_config_name, "updated-config")
        self.assertIs(refreshed, endpoint)


class TestLocalEndpointDelete(unittest.TestCase):
    """Test LocalEndpoint delete method."""

    def test_delete_calls_session_delete(self):
        """Test that delete calls session's delete_endpoint."""
        mock_session = Mock()
        
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=mock_session
        )
        
        endpoint.delete()
        
        mock_session.sagemaker_client.delete_endpoint.assert_called_once_with(
            EndpointName="test-endpoint"
        )


class TestLocalEndpointUpdate(unittest.TestCase):
    """Test LocalEndpoint update method."""

    def test_update_raises_not_implemented(self):
        """Test that update raises NotImplementedError."""
        mock_session = Mock()
        
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=mock_session
        )
        
        with self.assertRaises(NotImplementedError) as context:
            endpoint.update("new-config")
        
        self.assertIn("not supported in local mode", str(context.exception))


class TestLocalEndpointUniversalDeepPing(unittest.TestCase):
    """Test LocalEndpoint _universal_deep_ping method."""

    def test_ping_in_process_mode_success(self):
        """Test successful ping in in-process mode."""
        mock_in_process_obj = Mock()
        mock_schema_builder = Mock()
        mock_schema_builder.sample_input = {"input": "test"}
        mock_in_process_obj.schema_builder = mock_schema_builder
        
        mock_session = Mock()
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=mock_session,
            in_process_mode=True,
            in_process_mode_obj=mock_in_process_obj
        )
        
        # Mock invoke to return successful response
        with patch.object(endpoint, 'invoke') as mock_invoke:
            mock_output = Mock()
            mock_output.body = {"result": "success"}
            mock_invoke.return_value = mock_output
            
            healthy, response = endpoint._universal_deep_ping()
            
            self.assertTrue(healthy)
            self.assertEqual(response, {"result": "success"})

    def test_ping_container_mode_success(self):
        """Test successful ping in container mode."""
        mock_container_obj = Mock()
        mock_schema_builder = Mock()
        mock_schema_builder.sample_input = {"input": "test"}
        mock_container_obj.schema_builder = mock_schema_builder
        
        mock_session = Mock()
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=mock_session,
            in_process_mode=False,
            local_container_mode_obj=mock_container_obj
        )
        
        # Mock invoke to return successful response
        with patch.object(endpoint, 'invoke') as mock_invoke:
            mock_output = Mock()
            mock_body = Mock()
            mock_body.read.return_value = b'{"result": "success"}'
            mock_output.body = mock_body
            mock_invoke.return_value = mock_output
            
            healthy, response = endpoint._universal_deep_ping()
            
            self.assertTrue(healthy)
            self.assertEqual(response, {"result": "success"})

    def test_ping_failure(self):
        """Test ping failure."""
        mock_in_process_obj = Mock()
        mock_schema_builder = Mock()
        mock_schema_builder.sample_input = {"input": "test"}
        mock_in_process_obj.schema_builder = mock_schema_builder
        
        mock_session = Mock()
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=mock_session,
            in_process_mode=True,
            in_process_mode_obj=mock_in_process_obj
        )
        
        # Mock invoke to raise exception
        with patch.object(endpoint, 'invoke') as mock_invoke:
            mock_invoke.side_effect = Exception("Connection failed")
            
            healthy, response = endpoint._universal_deep_ping()
            
            self.assertFalse(healthy)
            self.assertIsNone(response)

    def test_ping_422_error_raises_local_invocation_exception(self):
        """Test that 422 error raises LocalModelInvocationException."""
        mock_in_process_obj = Mock()
        mock_schema_builder = Mock()
        mock_schema_builder.sample_input = {"input": "test"}
        mock_in_process_obj.schema_builder = mock_schema_builder
        
        mock_session = Mock()
        endpoint = LocalEndpoint(
            endpoint_name="test-endpoint",
            endpoint_config_name="test-config",
            local_session=mock_session,
            in_process_mode=True,
            in_process_mode_obj=mock_in_process_obj
        )
        
        # Mock invoke to raise 422 error
        with patch.object(endpoint, 'invoke') as mock_invoke:
            mock_invoke.side_effect = Exception("422 Client Error: Unprocessable Entity for url")
            
            from sagemaker.serve.utils.exceptions import LocalModelInvocationException
            with self.assertRaises(LocalModelInvocationException):
                endpoint._universal_deep_ping()


class TestLocalEndpointConfig(unittest.TestCase):
    """Test LocalEndpointConfig class."""

    @patch('sagemaker.core.local.local_session.LocalSession')
    def test_init(self, mock_local_session_class):
        """Test LocalEndpointConfig initialization."""
        mock_session = Mock()
        mock_local_session_class.return_value = mock_session
        production_variants = [{"VariantName": "AllTraffic"}]
        
        config = LocalEndpointConfig(
            endpoint_config_name="test-config",
            production_variants=production_variants,
            local_session=mock_session
        )
        
        self.assertEqual(config.endpoint_config_name, "test-config")
        self.assertEqual(config.production_variants, production_variants)
        self.assertIsInstance(config.creation_time, datetime.datetime)

    @patch('sagemaker.core.local.local_session.LocalSession')
    def test_create(self, mock_local_session_class):
        """Test LocalEndpointConfig create method."""
        mock_session = Mock()
        mock_local_session_class.return_value = mock_session
        production_variants = [{"VariantName": "AllTraffic"}]
        
        config = LocalEndpointConfig.create(
            endpoint_config_name="test-config",
            production_variants=production_variants,
            local_session=mock_session
        )
        
        self.assertEqual(config.endpoint_config_name, "test-config")
        mock_session.sagemaker_client.create_endpoint_config.assert_called_once_with(
            EndpointConfigName="test-config",
            ProductionVariants=production_variants
        )

    @patch('sagemaker.core.local.local_session.LocalSession')
    def test_delete(self, mock_local_session_class):
        """Test LocalEndpointConfig delete method."""
        mock_session = Mock()
        mock_local_session_class.return_value = mock_session
        production_variants = [{"VariantName": "AllTraffic"}]
        
        config = LocalEndpointConfig(
            endpoint_config_name="test-config",
            production_variants=production_variants,
            local_session=mock_session
        )
        
        config.delete()
        
        mock_session.sagemaker_client.delete_endpoint_config.assert_called_once_with(
            EndpointConfigName="test-config"
        )


class TestGetContainerConfig(unittest.TestCase):
    """Test _get_container_config function."""

    def test_host_config(self):
        """Test host network configuration."""
        config = _get_container_config("host")
        
        self.assertEqual(config, {"network_mode": "host"})

    def test_bridge_config(self):
        """Test bridge network configuration."""
        config = _get_container_config("bridge")
        
        self.assertEqual(config, {"ports": {'8080/tcp': 8080}})

    @patch('platform.system')
    def test_auto_config_linux(self, mock_system):
        """Test auto configuration on Linux."""
        mock_system.return_value = "Linux"
        
        config = _get_container_config("auto")
        
        self.assertEqual(config, {"network_mode": "host"})

    @patch('platform.system')
    def test_auto_config_macos(self, mock_system):
        """Test auto configuration on macOS."""
        mock_system.return_value = "Darwin"
        
        config = _get_container_config("auto")
        
        self.assertEqual(config, {"ports": {'8080/tcp': 8080}})

    @patch('platform.system')
    def test_auto_config_windows(self, mock_system):
        """Test auto configuration on Windows."""
        mock_system.return_value = "Windows"
        
        config = _get_container_config("auto")
        
        self.assertEqual(config, {"ports": {'8080/tcp': 8080}})

    def test_invalid_config_raises_error(self):
        """Test that invalid config raises ValueError."""
        with self.assertRaises(ValueError) as context:
            _get_container_config("invalid")
        
        self.assertIn("container_config must be", str(context.exception))


class TestDefaultSerializersByServer(unittest.TestCase):
    """Test DEFAULT_SERIALIZERS_BY_SERVER constant."""

    def test_all_model_servers_have_serializers(self):
        """Test that all major model servers have default serializers."""
        expected_servers = [
            ModelServer.TORCHSERVE,
            ModelServer.TENSORFLOW_SERVING,
            ModelServer.DJL_SERVING,
            ModelServer.TEI,
            ModelServer.TGI,
            ModelServer.MMS,
            ModelServer.SMD
        ]
        
        for server in expected_servers:
            self.assertIn(server, DEFAULT_SERIALIZERS_BY_SERVER)
            serializer, deserializer = DEFAULT_SERIALIZERS_BY_SERVER[server]
            self.assertIsNotNone(serializer)
            self.assertIsNotNone(deserializer)

    def test_serializers_are_tuples(self):
        """Test that all serializers are returned as tuples."""
        for server, (serializer, deserializer) in DEFAULT_SERIALIZERS_BY_SERVER.items():
            self.assertIsNotNone(serializer)
            self.assertIsNotNone(deserializer)
