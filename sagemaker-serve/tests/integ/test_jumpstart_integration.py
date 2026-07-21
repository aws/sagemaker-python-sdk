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

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.train.configs import Compute

from cleanup_helpers import cleanup_by_name

logger = logging.getLogger(__name__)

# Configuration - easily customizable
MODEL_ID = "huggingface-llm-falcon-7b-bf16"
MODEL_NAME_PREFIX = "js-v3-test-model"
ENDPOINT_NAME_PREFIX = "js-v3-test-endpoint"

SERVE_SAGEMAKER_ENDPOINT_TIMEOUT = 15


@pytest.mark.slow_test
def test_jumpstart_build_deploy_invoke_cleanup():
    """Integration test for JumpStart model build, deploy, invoke, and cleanup workflow"""
    logger.info("Starting JumpStart integration test...")

    # Names are generated up front so cleanup can run by name even if build()
    # or deploy() raises or is killed before returning a resource handle.
    unique_id = str(uuid.uuid4())[:8]
    model_name = f"{MODEL_NAME_PREFIX}-{unique_id}"
    endpoint_name = f"{ENDPOINT_NAME_PREFIX}-{unique_id}"

    try:
        # Build and deploy
        logger.info("Building and deploying JumpStart model...")
        core_endpoint = build_and_deploy(model_name, endpoint_name)

        # Make prediction
        logger.info("Making prediction...")
        make_prediction(core_endpoint)

        # Test passed successfully
        logger.info("JumpStart integration test completed successfully")

    except Exception as e:
        logger.error(f"JumpStart integration test failed: {str(e)}")
        raise
    finally:
        # Best-effort cleanup by name; runs even if deploy() failed/hung so a
        # Failed or half-created endpoint is still torn down.
        logger.info("Cleaning up resources...")
        cleanup_by_name(endpoint_name=endpoint_name, model_name=model_name)


def build_and_deploy(model_name, endpoint_name):
    """Build and deploy JumpStart model - preserving exact logic from manual test"""
    # Initialize model_builder object with JumpStart configuration
    compute = Compute(instance_type="ml.g5.2xlarge")
    jumpstart_config = JumpStartConfig(model_id=MODEL_ID)
    model_builder = ModelBuilder.from_jumpstart_config(jumpstart_config=jumpstart_config, compute=compute)

    # Build and deploy your model. Returns SageMaker Core Model and Endpoint objects
    core_model = model_builder.build(model_name=model_name)
    logger.info(f"Model Successfully Created: {core_model.model_name}")

    core_endpoint = model_builder.deploy(endpoint_name=endpoint_name)
    logger.info(f"Endpoint Successfully Created: {core_endpoint.endpoint_name}")

    return core_endpoint


def make_prediction(core_endpoint):
    """Make prediction using the deployed endpoint - preserving exact logic from manual test"""
    # Invoke the endpoint on a sample query:
    test_data = {"inputs": "What are falcons?", "parameters": {"max_new_tokens": 32}}
    result = core_endpoint.invoke(
        body=json.dumps(test_data),
        content_type="application/json"
    )

    # Decode the output of the invocation and print the result
    prediction = json.loads(result.body.read().decode('utf-8'))
    logger.info(f"Result of invoking endpoint: {prediction}")
