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

import uuid
import pytest
import logging

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.mode.function_pointers import Mode

logger = logging.getLogger(__name__)

# Configuration - easily customizable
MODEL_NAME_PREFIX = "inproces-test-model"
ENDPOINT_NAME_PREFIX = "inprocess-test-endpoint"


class MathInferenceSpec(InferenceSpec):
    """Simple math operations for IN_PROCESS testing."""
    
    def load(self, model_dir: str):
        """Load a simple math 'model'."""
        return {"operation": "multiply", "factor": 2.0}
    
    def invoke(self, input_object, model):
        """Perform math operation."""
        if isinstance(input_object, dict) and "numbers" in input_object:
            numbers = input_object["numbers"]
        elif isinstance(input_object, list):
            numbers = input_object
        else:
            numbers = [float(input_object)]
        
        factor = model["factor"]
        result = [num * factor for num in numbers]
        
        return {"result": result, "operation": f"multiply by {factor}"}


@pytest.mark.slow_test
def test_in_process_build_deploy_invoke_cleanup():
    """Integration test for In-Process mode build, deploy, invoke, and cleanup workflow"""
    logger.info("Starting In-Process integration test...")
    
    core_model = None
    local_endpoint = None
    
    try:
        # Build and deploy
        logger.info("Building and deploying In-Process model...")
        core_model, local_endpoint = build_and_deploy()
        
        # Make prediction
        logger.info("Making prediction...")
        make_prediction(local_endpoint)
        
        # Test passed successfully
        logger.info("In-Process integration test completed successfully")
        
    except Exception as e:
        logger.error(f"In-Process integration test failed: {str(e)}")
        raise
    finally:
        # Cleanup resources
        if core_model and local_endpoint:
            logger.info("Cleaning up resources...")
            cleanup_resources(core_model, local_endpoint)


def create_schema_builder():
    """Create a math schema builder for testing."""
    sample_input = {"numbers": [1.0, 2.0, 3.0]}
    sample_output = {"result": [2.0, 4.0, 6.0], "operation": "multiply by 2.0"}
    return SchemaBuilder(sample_input, sample_output)


def build_and_deploy():
    """Build and deploy In-Process model - preserving exact logic from manual test"""
    schema_builder = create_schema_builder()
    inference_spec = MathInferenceSpec()
    unique_id = str(uuid.uuid4())[:8]
    
    model_builder = ModelBuilder(
        inference_spec=inference_spec,
        schema_builder=schema_builder,
        mode=Mode.IN_PROCESS
    )
    
    core_model = model_builder.build(model_name=f"{MODEL_NAME_PREFIX}-{unique_id}")
    logger.info(f"Model Successfully Created: {core_model.model_name}")

    local_endpoint = model_builder.deploy_local(endpoint_name=f"{ENDPOINT_NAME_PREFIX}-{unique_id}")
    logger.info(f"Endpoint Successfully Created: {local_endpoint.endpoint_name}")
    
    return core_model, local_endpoint


def make_prediction(local_endpoint):
    """Make prediction using the deployed endpoint - preserving exact logic from manual test"""
    test_data = {"numbers": [1.0, 2.0, 3.0]}
    
    result = local_endpoint.invoke(
        body=test_data,
        content_type="application/json"
    )

    logger.info(f"Result of invoking endpoint: {result.body}")


def cleanup_resources(core_model, local_endpoint):
    """Clean up IN_PROCESS endpoint - preserving exact logic from manual test"""
    # Clean up IN_PROCESS endpoint
    if local_endpoint and hasattr(local_endpoint, 'in_process_mode_obj'):
        if local_endpoint.in_process_mode_obj:
            local_endpoint.in_process_mode_obj.destroy_server()
    
    logger.info("Model and Endpoint Successfully Deleted!")