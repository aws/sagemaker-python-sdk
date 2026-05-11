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


VOLUME_SIZE_MODEL_ID = "meta-textgenerationneuron-llama-2-7b"
VOLUME_SIZE_INSTANCE_TYPE = "ml.inf2.xlarge"


@pytest.mark.slow_test
def test_jumpstart_build_sets_volume_size():
    """Integration test verifying volume_size from model specs is propagated.

    JumpStart model specs define inference_volume_size for models that need
    large EBS volumes for model weights. This test validates that ModelBuilder
    propagates volume_size through both from_jumpstart_config() and build() paths,
    matching v2 behavior where VolumeSizeInGB appears in CreateEndpointConfig.
    """
    logger.info("Starting JumpStart volume_size integration test...")

    # Test from_jumpstart_config path
    compute = Compute(instance_type=VOLUME_SIZE_INSTANCE_TYPE)
    jumpstart_config = JumpStartConfig(model_id=VOLUME_SIZE_MODEL_ID)
    model_builder = ModelBuilder.from_jumpstart_config(
        jumpstart_config=jumpstart_config, compute=compute
    )

    assert getattr(model_builder, "volume_size", None) is not None, (
        f"ModelBuilder.volume_size should be set after from_jumpstart_config() "
        f"for model {VOLUME_SIZE_MODEL_ID} on {VOLUME_SIZE_INSTANCE_TYPE}, got None"
    )
    logger.info(f"from_jumpstart_config set volume_size={model_builder.volume_size}")

    # Test build path (also sets volume_size via _build_for_jumpstart)
    unique_id = str(uuid.uuid4())[:8]
    core_model = model_builder.build(model_name=f"js-volsize-test-{unique_id}")
    logger.info(f"Model created: {core_model.model_name}")

    try:
        assert getattr(model_builder, "volume_size", None) is not None, (
            f"ModelBuilder.volume_size should persist after build() "
            f"for model {VOLUME_SIZE_MODEL_ID}, got None"
        )
        assert model_builder.volume_size >= 256, (
            f"volume_size should be >= 256, "
            f"got {model_builder.volume_size}"
        )
        logger.info(
            f"✅ volume_size={model_builder.volume_size} correctly set"
        )
    finally:
        core_model.delete()
        logger.info("Model deleted.")
