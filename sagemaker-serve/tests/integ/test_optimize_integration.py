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
import time
import pytest
import logging
import boto3

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.resources import EndpointConfig
from sagemaker.core.helper.session_helper import Session

logger = logging.getLogger(__name__)

# Configuration - easily customizable
MODEL_ID = "meta-textgeneration-llama-3-8b-instruct"  # JumpStart model
MODEL_NAME_PREFIX = "jumpstart-optimize-test"
ENDPOINT_NAME_PREFIX = "jumpstart-optimize-test-endpoint"

# Configuration from optimize test
AWS_ACCOUNT_ID = "593793038179"
AWS_REGION = "us-east-2"


@pytest.mark.skip(reason="Test takes too long to run")
def test_optimize_build_deploy_invoke_cleanup():
    """Integration test for Optimize workflow"""
    logger.info("Starting Optimize integration test...")
    
    optimized_model = None
    core_endpoint = None
    
    try:
        # Build and deploy
        logger.info("Optimizing and deploying model...")
        optimized_model, core_endpoint = build_and_deploy()
        
        # Make prediction
        logger.info("Making prediction...")
        make_prediction(core_endpoint)
        
        # Test passed successfully
        logger.info("Optimize integration test completed successfully")
        
    except Exception as e:
        logger.error(f"Optimize integration test failed: {str(e)}")
        raise
    finally:
        # Cleanup resources
        if optimized_model and core_endpoint:
            logger.info("Cleaning up resources...")
            cleanup_resources(optimized_model, core_endpoint)


def create_schema_builder():
    """Create schema builder for text generation - exact from optimize test."""
    from sagemaker.serve.builder.schema_builder import SchemaBuilder
    
    sample_input = {"inputs": "What are falcons?", "parameters": {"max_new_tokens": 32}}
    sample_output = [{"generated_text": "Falcons are small to medium-sized birds of prey."}]
    
    return SchemaBuilder(sample_input, sample_output)


def build_and_deploy():
    """Optimize and deploy JumpStart model - exact logic from optimize test."""
    schema_builder = create_schema_builder()
    boto_session = boto3.Session(region_name=AWS_REGION)
    sagemaker_session = Session(boto_session=boto_session)
    unique_id = str(uuid.uuid4())[:8]
    
    model_builder = ModelBuilder(
        model=MODEL_ID,
        schema_builder=schema_builder,
        sagemaker_session=sagemaker_session,
    )
    
    # Optimize the model
    logger.info("Optimizing JumpStart model...")
    default_bucket = sagemaker_session.default_bucket()
    optimized_model = model_builder.optimize(
        model_name=f"{MODEL_NAME_PREFIX}-{unique_id}",
        instance_type="ml.g5.2xlarge",
        output_path=f"s3://{default_bucket}/optimize-output/jumpstart-{unique_id}/",
        quantization_config={"OverrideEnvironment": {"OPTION_QUANTIZE": "awq"}},
        accept_eula=True,
        job_name=f"js-optimize-{int(time.time())}",
        image_uri="763104351884.dkr.ecr.us-east-2.amazonaws.com/djl-inference:0.31.0-lmi13.0.0-cu124"
    )
    logger.info(f"Model Successfully Optimized: {optimized_model.model_name}")
    
    # Deploy the optimized model
    logger.info("Deploying optimized model to endpoint...")
    core_endpoint = model_builder.deploy(
        endpoint_name=f"{ENDPOINT_NAME_PREFIX}-{unique_id}",
        initial_instance_count=1,
        instance_type="ml.g5.2xlarge"
    )
    logger.info(f"Endpoint Successfully Created: {core_endpoint.endpoint_name}")
    
    return optimized_model, core_endpoint


def make_prediction(core_endpoint):
    """Test optimized model invocation - exact logic from optimize test."""
    test_data = {
        "inputs": "What are the benefits of machine learning?",
        "parameters": {"max_new_tokens": 50}
    }
    
    result = core_endpoint.invoke(
        body=json.dumps(test_data),
        content_type="application/json"
    )
    
    response_body = result.body.read().decode('utf-8')
    prediction = json.loads(response_body)
    logger.info(f"Result of invoking optimized endpoint: {prediction}")


def cleanup_resources(optimized_model, core_endpoint):
    """Clean up optimized model and endpoint - preserving exact logic from manual test"""
    core_endpoint_config = EndpointConfig.get(endpoint_config_name=core_endpoint.endpoint_name)
   
    optimized_model.delete()
    core_endpoint.delete()
    core_endpoint_config.delete()

    logger.info("Optimized model and endpoint successfully deleted!")