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
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.train.configs import Compute

logger = logging.getLogger(__name__)

MODEL_ID = "huggingface-llm-falcon-7b-bf16"
INSTANCE_TYPE = "ml.g5.2xlarge"
MODEL_NAME_PREFIX = "js-netiso-test"


@pytest.mark.slow_test
def test_jumpstart_build_enables_network_isolation():
    """Integration test verifying JumpStart models are built with EnableNetworkIsolation.

    JumpStart model specs define inference_enable_network_isolation=True for most models.
    This test validates that ModelBuilder.build() propagates this setting to the
    SageMaker Model resource, matching v2 behavior.
    """
    logger.info("Starting JumpStart network isolation integration test...")

    compute = Compute(instance_type=INSTANCE_TYPE)
    jumpstart_config = JumpStartConfig(model_id=MODEL_ID)
    model_builder = ModelBuilder.from_jumpstart_config(
        jumpstart_config=jumpstart_config, compute=compute
    )
    unique_id = str(uuid.uuid4())[:8]

    core_model = model_builder.build(model_name=f"{MODEL_NAME_PREFIX}-{unique_id}")
    logger.info(f"Model created: {core_model.model_name}")

    try:
        # Verify ModelBuilder picked up network isolation from spec
        assert model_builder._enable_network_isolation, (
            f"ModelBuilder._enable_network_isolation should be True for {MODEL_ID}, "
            f"got {model_builder._enable_network_isolation}"
        )

        # Verify the actual SageMaker Model resource has EnableNetworkIsolation=True
        sm_client = model_builder.sagemaker_session.sagemaker_client
        desc = sm_client.describe_model(ModelName=core_model.model_name)
        assert desc.get("EnableNetworkIsolation") is True, (
            f"SageMaker Model should have EnableNetworkIsolation=True, "
            f"got {desc.get('EnableNetworkIsolation')}"
        )

        logger.info("✅ Network isolation correctly applied to SageMaker Model")
    finally:
        core_model.delete()
        logger.info("Model deleted.")
