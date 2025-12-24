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
from __future__ import absolute_import

import os
import json
import uuid
import tempfile
import pytest
import logging

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.core.resources import EndpointConfig
from sagemaker.core.helper.session_helper import Session

# PyTorch Imports
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Configuration - easily customizable
MODEL_NAME_PREFIX = "triton-test-model"
ENDPOINT_NAME_PREFIX = "triton-test-endpoint"

sagemaker_session = Session()


# Create a simple PyTorch model 
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
    
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)


@pytest.mark.slow_test
def test_triton_build_deploy_invoke_cleanup():
    """Integration test for Triton model build, deploy, invoke, and cleanup workflow"""
    logger.info("Starting Triton integration test...")
    
    core_model = None
    core_endpoint = None
    
    try:
        # Build and deploy
        logger.info("Building and deploying Triton model...")
        core_model, core_endpoint = build_and_deploy()
        
        # Make prediction
        logger.info("Making prediction...")
        make_prediction(core_endpoint)
        
        # Test passed successfully
        logger.info("Triton integration test completed successfully")
        
    except Exception as e:
        logger.error(f"Triton integration test failed: {str(e)}")
        raise
    finally:
        # Cleanup resources
        if core_model and core_endpoint:
            logger.info("Cleaning up resources...")
            cleanup_resources(core_model, core_endpoint)


def create_schema_builder():
    """Create schema builder for SimpleModel"""
    from sagemaker.serve.builder.schema_builder import SchemaBuilder
    import torch
    
    # Use torch.tensor instead of np.array for PyTorch ONNX conversion
    sample_input = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)  # 4 features
    sample_output = torch.tensor([[0.9, 0.1]], dtype=torch.float32)  # 2 class probabilities
    
    return SchemaBuilder(sample_input, sample_output)


def build_and_deploy():
    """Build and deploy Triton model - preserving exact logic from manual test"""
    # Create model and save state dictionary (assume model is pre-trained!)
    pytorch_model = SimpleModel()
    model_path = tempfile.mkdtemp()
    torch.save(pytorch_model.state_dict(), os.path.join(model_path, "model.pth"))
    
    schema_builder = create_schema_builder()
    
    model_builder = ModelBuilder(
        model=pytorch_model,
        model_path=model_path,
        model_server=ModelServer.TRITON,
        schema_builder=schema_builder,
        sagemaker_session=sagemaker_session,
    )
      
    unique_id = str(uuid.uuid4())[:8]
    # Build and deploy your model. Returns SageMaker Core Model and Endpoint objects
    core_model = model_builder.build(model_name=f"{MODEL_NAME_PREFIX}-{unique_id}")
    logger.info(f"Model Successfully Created: {core_model.model_name}")

    core_endpoint = model_builder.deploy(endpoint_name=f"{ENDPOINT_NAME_PREFIX}-{unique_id}")
    logger.info(f"Endpoint Successfully Created: {core_endpoint.endpoint_name}")
    
    return core_model, core_endpoint


def make_prediction(core_endpoint):
    """Make prediction using the deployed endpoint - preserving exact logic from manual test"""
    # Invoke the endpoint on a sample query:
    test_data = {
        "inputs": [
            {
                "name": "input_1",
                "shape": [1, 4],
                "datatype": "FP32", 
                "data": [[0.1, 0.2, 0.3, 0.4]]
            }
        ]
    }
        
    result = core_endpoint.invoke(
        body=json.dumps(test_data),
        content_type="application/json"
     )

    # Decode the output of the invocation and print the result
    prediction = json.loads(result.body.read().decode('utf-8'))
    logger.info(f"Result of invoking endpoint: {prediction}")


def cleanup_resources(core_model, core_endpoint):
    """Fully clean up model and endpoint creation - preserving exact logic from manual test"""
    core_endpoint_config = EndpointConfig.get(
        endpoint_config_name=core_endpoint.endpoint_name,
    )
   
    core_model.delete()
    core_endpoint.delete()
    core_endpoint_config.delete()

    logger.info("Model and Endpoint Successfully Deleted!")