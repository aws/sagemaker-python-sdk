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
"""Local Model and Endpoint classes for V3 ModelBuilder local mode support.

These classes provide sagemaker-core compatible interfaces for local deployment,
wrapping V2 local mode functionality.
"""

from __future__ import absolute_import
import datetime
import json
import logging
from typing import Any, Dict, Optional, Tuple
import io
import json
from sagemaker.serve.utils.types import ModelServer
from sagemaker.core.serializers import JSONSerializer, IdentitySerializer
from sagemaker.core.deserializers import JSONDeserializer, BytesDeserializer
from sagemaker.core.resources import Model

logger = logging.getLogger(__name__)

APPLICATION_X_NPY = "application/x-npy"

# Triton gets serializer/deserializer from schema_builder
DEFAULT_SERIALIZERS_BY_SERVER: Dict[ModelServer, Tuple] = {
    ModelServer.TORCHSERVE: (IdentitySerializer(), BytesDeserializer()),
    ModelServer.TENSORFLOW_SERVING: (JSONSerializer(), JSONDeserializer()),  # TF Serving expects JSON
    ModelServer.DJL_SERVING: (JSONSerializer(), JSONDeserializer()),
    ModelServer.TEI: (JSONSerializer(), JSONDeserializer()),
    ModelServer.TGI: (JSONSerializer(), JSONDeserializer()),
    ModelServer.MMS: (JSONSerializer(), JSONDeserializer()),
    ModelServer.SMD: (JSONSerializer(), JSONDeserializer()),
}


class InvokeEndpointOutput:
    """Response wrapper to match sagemaker-core Endpoint.invoke() output format."""
    
    def __init__(self, body: bytes, content_type: str = "application/json"):
        self.body = body
        self.content_type = content_type

class LocalEndpoint:
    """Local endpoint that mimics sagemaker.core.Endpoint interface.
    
    This class wraps V2 LocalSession endpoint functionality to provide a unified
    interface compatible with sagemaker-core Endpoint resources.
    """
    
    def __init__(
        self,
        endpoint_name: str,
        endpoint_config_name: str,
        local_session=None,
        local_model=None,
        in_process_mode=False, 
        local_container_mode_obj=None,
        in_process_mode_obj=None,
        model_server=None,
        secret_key=None,
        serializer=None,
        deserializer=None,
        container_config="auto",
        **kwargs
    ):
        """Initialize local endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            endpoint_config_name: Name of the endpoint configuration
            local_session: V2 LocalSession instance
        """
        self.endpoint_name = endpoint_name
        self.endpoint_config_name = endpoint_config_name
        self.creation_time = datetime.datetime.now()
        self._local_model = local_model
        self.in_process_mode = in_process_mode
        self.local_container_mode_obj=local_container_mode_obj
        self.in_process_mode_obj=in_process_mode_obj
        self.model_server=model_server
        self.secret_key=secret_key
        self.serializer=serializer
        self.deserializer=deserializer
        self.container_config=container_config
        
        # Import V3 LocalSession
        if local_session is None:
            from sagemaker.core.local.local_session import LocalSession
            self._local_session = LocalSession()
        else:
            self._local_session = local_session
    
    # @property
    # def endpoint_arn(self) -> str:
    #     """Fake ARN for compatibility with sagemaker-core interface."""
    #     return f"arn:aws:sagemaker:local:000000000000:endpoint/{self.endpoint_name}"
    
    @property
    def endpoint_status(self) -> str:
        """Get endpoint status.
        
        Implementation based on V2 LocalSession.describe_endpoint()
        Reference: /sagemaker/local/local_session.py:describe_endpoint()
        """
        try:
            endpoint_info = self._local_session.sagemaker_client.describe_endpoint(
                EndpointName=self.endpoint_name
            )
            return endpoint_info["EndpointStatus"]
        except Exception:
            return "Failed"
    

    def _universal_deep_ping(self) -> tuple[bool, Any]:
        """Universal ping function that works for all model servers."""
        response = None
        logger.info("Pinging local endpoint...")
        try:
            # Get sample input from schema builder
            if self.in_process_mode:
                sample_input = self.in_process_mode_obj.schema_builder.sample_input
            else:
                sample_input = self.local_container_mode_obj.schema_builder.sample_input
            
            # Use unified invoke interface
            invoke_response = self.invoke(body=sample_input)
            
            if self.in_process_mode:
                # IN_PROCESS: Response is already deserialized
                response = invoke_response.body
                healthy = response is not None
            else:
                # LOCAL_CONTAINER: Response needs decoding
                response_body = invoke_response.body.read().decode('utf-8')
                response = json.loads(response_body)
                healthy = response is not None
            
            return (healthy, response)
            
        except Exception as e:
            if "422 Client Error: Unprocessable Entity for url" in str(e):
                from sagemaker.serve.utils.exceptions import LocalModelInvocationException
                raise LocalModelInvocationException(str(e))
            
            return (False, None)


    def invoke(
        self,
        body: Any,
        content_type: str = "application/json",
        accept: str = "application/json",
        **kwargs
    ) -> InvokeEndpointOutput:
        """Invoke the local endpoint using model server-specific logic."""
        if self.in_process_mode:
            if not self.in_process_mode_obj:
                raise ValueError("In Process container mode not available")
            
            serializer = self.serializer or JSONSerializer()
            deserializer = self.deserializer or JSONDeserializer()
            serialized_data = serializer.serialize(body)

            raw_response = self.in_process_mode_obj._invoke_serving(
                serialized_data, content_type, accept
            )
            return InvokeEndpointOutput(
                body=deserializer.deserialize(io.BytesIO(raw_response)),
                content_type=accept
            )
        
        else:
            if not self.model_server or not self.local_container_mode_obj:
                raise ValueError("Model server or container mode not available")
            
            # Get serializers (use defaults if not provided by model)
            serializer = self.serializer or JSONSerializer()
            deserializer = self.deserializer or JSONDeserializer()

            content_type = content_type if content_type != "application/json" else serializer.CONTENT_TYPE
            deserializer_accept = deserializer.ACCEPT
            if not isinstance(deserializer_accept, str):
                deserializer_accept = deserializer_accept[0]
            accept = accept if accept != "application/json" else deserializer_accept
            
            # Route to appropriate model server invoke method
            if self.model_server == ModelServer.TORCHSERVE:
                # TorchServe: Use serializer-derived content types (V2 pattern)
                serialized_data = serializer.serialize(body) if not isinstance(body, str) else body
                raw_response = self.local_container_mode_obj._invoke_torch_serve(
                    serialized_data,
                    content_type,
                    accept
                )
                response_data = deserializer.deserialize(io.BytesIO(raw_response))
                
            elif self.model_server == ModelServer.TRITON:
                # Triton: Direct data, no serialization, fixed content types (V2 pattern)
                from sagemaker.serve.utils.predictors import APPLICATION_X_NPY
                raw_response = self.local_container_mode_obj._invoke_triton_server(
                    body,  # ← Direct data, no serialization
                    APPLICATION_X_NPY,
                    APPLICATION_X_NPY
                )
                response_data = raw_response
                
            elif self.model_server == ModelServer.DJL_SERVING:
                # DJL: Use serializer-derived content types + deserialize with content_type
                serialized_data = serializer.serialize(body) if not isinstance(body, str) else body
                raw_response = self.local_container_mode_obj._invoke_djl_serving(
                    serialized_data,
                    content_type,
                    accept
                )
                response_data = deserializer.deserialize(
                    io.BytesIO(raw_response),
                    content_type 
                )
                
            elif self.model_server == ModelServer.TGI:
                # TGI: Use serializer-derived content types + list format
                serialized_data = serializer.serialize(body) if not isinstance(body, str) else body
                raw_response = self.local_container_mode_obj._invoke_tgi_serving(
                    serialized_data,
                    content_type,
                    accept
                )
                response_data = [deserializer.deserialize(
                    io.BytesIO(raw_response),
                    content_type
                )]
                
            elif self.model_server == ModelServer.MMS:
                # MMS: Use serializer-derived content types + list format
                serialized_data = serializer.serialize(body) if not isinstance(body, str) else body
                raw_response = self.local_container_mode_obj._invoke_multi_model_server_serving(
                    serialized_data,
                    content_type,
                    accept
                )
                response_data = [deserializer.deserialize(
                    io.BytesIO(raw_response),
                    content_type
                )]
                
            elif self.model_server == ModelServer.TENSORFLOW_SERVING:
                # TensorFlow: Use serializer-derived content types
                serialized_data = serializer.serialize(body) if not isinstance(body, str) else body
                raw_response = self.local_container_mode_obj._invoke_tensorflow_serving(
                    serialized_data,
                    content_type,
                    accept
                )
                response_data = deserializer.deserialize(io.BytesIO(raw_response))
                
            elif self.model_server == ModelServer.TEI:
                # TEI: Use serializer-derived content types
                serialized_data = serializer.serialize(body) if not isinstance(body, str) else body
                raw_response = self.local_container_mode_obj._invoke_serving(
                    serialized_data,
                    content_type,
                    accept
                )
                response_data = deserializer.deserialize(io.BytesIO(raw_response))
                
            else:
                raise ValueError(f"Unsupported model server: {self.model_server}")
            
            # Return in sagemaker-core compatible format
            return InvokeEndpointOutput(
                body=io.BytesIO(json.dumps(response_data).encode('utf-8')),
                content_type=accept
            )
        

    @classmethod
    def create(
        cls,
        endpoint_name: str,
        endpoint_config_name: Optional[str] = None,
        local_model: Optional[Model] = None,
        local_session=None,
        in_process_mode=False, 
        local_container_mode_obj=None,
        in_process_mode_obj=None,
        model_server=None,
        secret_key=None,
        serializer=None,
        deserializer=None,
        container_config="auto",
        **kwargs
    ) -> "LocalEndpoint":
        """Create and start local endpoint."""
        if local_session is None:
            from sagemaker.core.local.local_session import LocalSession
            local_session = LocalSession()
        

        if in_process_mode:
            endpoint = cls(
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_config_name or f"{endpoint_name}-config",
                local_session=local_session,
                local_model=local_model,
                in_process_mode=in_process_mode,
                local_container_mode_obj=local_container_mode_obj,
                in_process_mode_obj=in_process_mode_obj,
                model_server=model_server,
                secret_key=secret_key,
                serializer=serializer,
                deserializer=deserializer,
                container_config=container_config,
                **kwargs
            )

            endpoint.in_process_mode_obj.create_server(
                ping_fn=endpoint._universal_deep_ping
            )

            return endpoint
            

        else:
            # Create endpoint instance first so we can reference its ping method
            endpoint = cls(
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_config_name or f"{endpoint_name}-config",
                local_session=local_session,
                local_model=local_model,
                in_process_mode=in_process_mode,
                local_container_mode_obj=local_container_mode_obj,
                in_process_mode_obj=in_process_mode_obj,
                model_server=model_server,
                secret_key=secret_key,
                serializer=serializer,
                deserializer=deserializer,
                container_config=container_config,
                **kwargs
            )
            
            # Start container with ping function
            endpoint.local_container_mode_obj.create_server(
                image=local_model.primary_container.image,
                container_timeout_seconds=kwargs.get("container_timeout_seconds", 300),
                secret_key=endpoint.secret_key,
                ping_fn=endpoint._universal_deep_ping,
                env_vars=local_model.primary_container.environment or {},
                model_path=endpoint.local_container_mode_obj.model_path,
                container_config=_get_container_config(endpoint.container_config)
            )
            
            # Register endpoint with V2 LocalSession
            production_variants = [{
                "VariantName": "AllTraffic",
                "ModelName": local_model.model_name, 
                "InitialInstanceCount": 1,
                "InstanceType": "local"
            }]

            local_session.sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint.endpoint_config_name,
                ProductionVariants=production_variants
            )

            # Then create endpoint
            local_session.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint.endpoint_config_name  
            )
            
            return endpoint


    @classmethod
    def get(cls, endpoint_name: str, local_session=None) -> Optional["LocalEndpoint"]:
        """Get existing local endpoint.
        
        Implementation based on V2 LocalSession.describe_endpoint()
        Reference: /sagemaker/local/local_session.py:describe_endpoint()
        """
        if local_session is None:
            from sagemaker.core.local.local_session import LocalSession
            local_session = LocalSession()
        
        try:
            # Call V2 describe_endpoint to get endpoint info
            endpoint_info = local_session.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            
            return cls(
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_info["EndpointConfigName"],
                local_session=local_session
            )
        except Exception:
            # Endpoint not found
            return None
    
    def refresh(self) -> "LocalEndpoint":
        """Refresh endpoint state.
        
        Implementation based on V2 LocalSession.describe_endpoint()
        Reference: /sagemaker/local/local_session.py:describe_endpoint()
        """
        endpoint_info = self._local_session.sagemaker_client.describe_endpoint(
            EndpointName=self.endpoint_name
        )
        
        # Update attributes from V2 response
        self.endpoint_config_name = endpoint_info["EndpointConfigName"]
        
        return self
    
    def delete(self) -> None:
        """Delete local endpoint and cleanup container.
        
        Implementation based on V2 LocalSession.delete_endpoint()
        Reference: /sagemaker/local/local_session.py:delete_endpoint()
        This calls _LocalEndpoint.stop() which stops the Docker container
        """
        self._local_session.sagemaker_client.delete_endpoint(
            EndpointName=self.endpoint_name
        )
    
    def update(self, endpoint_config_name: str) -> None:
        """Update endpoint configuration.
        
        V2 Reference: /sagemaker/local/local_session.py:update_endpoint()
        Note: V2 raises NotImplementedError for update_endpoint
        """
        raise NotImplementedError("Update endpoint is not supported in local mode")


class LocalEndpointConfig:
    """Local endpoint configuration that mimics sagemaker.core.EndpointConfig interface."""
    
    def __init__(
        self,
        endpoint_config_name: str,
        production_variants: list,
        local_session=None,
        **kwargs
    ):
        """Initialize local endpoint config.
        
        Args:
            endpoint_config_name: Name of the endpoint configuration
            production_variants: List of production variant configurations
            local_session: V2 LocalSession instance
        """
        self.endpoint_config_name = endpoint_config_name
        self.production_variants = production_variants
        self.creation_time = datetime.datetime.now()
        
        if local_session is None:
            from sagemaker.core.local.local_session import LocalSession
            self._local_session = LocalSession()
        else:
            self._local_session = local_session
    
    @classmethod
    def create(
        cls,
        endpoint_config_name: str,
        production_variants: list,
        local_session=None,
        **kwargs
    ) -> "LocalEndpointConfig":
        """Create local endpoint configuration.
        
        Implementation based on V2 LocalSession.create_endpoint_config()
        Reference: /sagemaker/local/local_session.py:create_endpoint_config()
        """
        if local_session is None:
            from sagemaker.core.local.local_session import LocalSession
            local_session = LocalSession()
        
        # Create instance
        local_config = cls(
            endpoint_config_name=endpoint_config_name,
            production_variants=production_variants,
            local_session=local_session
        )
        
        # Call V2 LocalSession.create_endpoint_config()
        local_session.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=production_variants
        )
        
        return local_config
    
    def delete(self) -> None:
        """Delete local endpoint configuration.
        
        Implementation based on V2 LocalSession.delete_endpoint_config()
        Reference: /sagemaker/local/local_session.py:delete_endpoint_config()
        """
        self._local_session.sagemaker_client.delete_endpoint_config(
            EndpointConfigName=self.endpoint_config_name
        )



def _get_container_config(config: str) -> dict:
    """Get container configuration based on config type."""
    if config == "host":
        return {"network_mode": "host"}
    elif config == "bridge":
        return {"ports": {'8080/tcp': 8080}}
    elif config == "auto":
        import platform
        if platform.system().lower() == "linux":
            return {"network_mode": "host"}
        else:
            return {"ports": {'8080/tcp': 8080}}
    else:
        raise ValueError("container_config must be 'host', 'bridge', or 'auto'")