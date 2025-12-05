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

import json
import uuid
import pytest
import logging
import boto3

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.train.configs import Compute
from sagemaker.core.resources import EndpointConfig
from sagemaker.core.helper.session_helper import Session

logger = logging.getLogger(__name__)

# Configuration - easily customizable
MODEL_ID = "BAAI/bge-m3"  # Small text generation model
MODEL_NAME_PREFIX = "tei-test-model"
ENDPOINT_NAME_PREFIX = "tei-test-endpoint"

# Configuration from backup file
AWS_REGION = "us-east-2"


@pytest.mark.slow_test
def test_tei_build_deploy_invoke_cleanup():
    """Integration test for TEI model build, deploy, invoke, and cleanup workflow"""
    logger.info("Starting TEI integration test...")
    
    core_model = None
    core_endpoint = None
    
    try:
        # Build and deploy
        logger.info("Building and deploying TEI model...")
        core_model, core_endpoint = build_and_deploy()
        
        # Make prediction
        logger.info("Making prediction...")
        make_prediction(core_endpoint)
        
        # Test passed successfully
        logger.info("TEI integration test completed successfully")
        
    except Exception as e:
        logger.error(f"TEI integration test failed: {str(e)}")
        raise
    finally:
        # Cleanup resources
        if core_model and core_endpoint:
            logger.info("Cleaning up resources...")
            cleanup_resources(core_model, core_endpoint)


def create_schema_builder():
    """Create schema builder for text generation - exact from backup file."""
    from sagemaker.serve.builder.schema_builder import SchemaBuilder
    
    sample_input = {"inputs": "What are falcons?", "parameters": {"max_new_tokens": 32}}
    sample_output = [{"generated_text": "Falcons are small to medium-sized birds of prey."}]
    
    return SchemaBuilder(sample_input, sample_output)


def build_and_deploy():
    """Build and deploy TEI model - exact logic from backup file."""
    # Use HuggingFace model string for TEI (text embeddings)
    hf_model_id = MODEL_ID
    
    schema_builder = create_schema_builder()
    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = Session(boto_session=boto_session)
    unique_id = str(uuid.uuid4())[:8]

    compute = Compute(
        instance_type="ml.g5.xlarge",
        instance_count=1,
    )
    
    model_builder = ModelBuilder(
        model=hf_model_id,  # Use HuggingFace model string
        model_server=ModelServer.TEI,
        schema_builder=schema_builder,
        sagemaker_session=sagemaker_session,
        compute=compute,
    )
    
    # Build and deploy your model. Returns SageMaker Core Model and Endpoint objects
    core_model = model_builder.build(model_name=f"{MODEL_NAME_PREFIX}-{unique_id}")
    logger.info(f"Model Successfully Created: {core_model.model_name}")

    core_endpoint = model_builder.deploy(
        endpoint_name=f"{ENDPOINT_NAME_PREFIX}-{unique_id}",
        initial_instance_count=1,
    )
    logger.info(f"Endpoint Successfully Created: {core_endpoint.endpoint_name}")
    
    return core_model, core_endpoint


def make_prediction(core_endpoint):
    """Test invoke - exact logic from backup file."""
    test_data = {"inputs": "This is a sample text for embeddings"}  # TEI text format
    
    result = core_endpoint.invoke(
        body=json.dumps(test_data),
        content_type="application/json"
    )

    # Decode the output of the invocation and print the result
    prediction = json.loads(result.body.read().decode('utf-8'))
    logger.info(f"Result of invoking endpoint: {prediction}")


def cleanup_resources(core_model, core_endpoint):
    """Fully clean up model and endpoint creation - preserving exact logic from manual test"""
    core_endpoint_config = EndpointConfig.get(endpoint_config_name=core_endpoint.endpoint_name)
   
    core_model.delete()
    core_endpoint.delete()
    core_endpoint_config.delete()

    logger.info("Model and Endpoint Successfully Deleted!")