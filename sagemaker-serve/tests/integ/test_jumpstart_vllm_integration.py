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
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.train.configs import Compute

logger = logging.getLogger(__name__)

MODEL_ID = "huggingface-vlm-qwen3-5-4b"
INSTANCE_TYPE = "ml.g6.12xlarge"
MODEL_NAME_PREFIX = "js-vllm-test-model"


@pytest.mark.slow_test
def test_jumpstart_vllm_build():
    """Integration test for JumpStart model using vLLM container image.

    Validates that ModelBuilder correctly handles vLLM container images
    which do not match the legacy djl-inference/tgi-inference/huggingface-pytorch-inference
    patterns in the endpoint mode routing.
    """
    logger.info("Starting JumpStart vLLM build test...")

    compute = Compute(instance_type=INSTANCE_TYPE)
    jumpstart_config = JumpStartConfig(model_id=MODEL_ID)
    model_builder = ModelBuilder.from_jumpstart_config(
        jumpstart_config=jumpstart_config, compute=compute
    )
    unique_id = str(uuid.uuid4())[:8]

    core_model = model_builder.build(model_name=f"{MODEL_NAME_PREFIX}-{unique_id}")
    logger.info(f"Model Successfully Created: {core_model.model_name}")

    try:
        assert (
            "vllm" in model_builder.image_uri
        ), f"Expected vLLM image URI, got: {model_builder.image_uri}"
        assert model_builder.s3_upload_path is not None, "s3_upload_path should be set"
        logger.info("JumpStart vLLM build test completed successfully")
    finally:
        core_model.delete()
        logger.info("Model Successfully Deleted!")
