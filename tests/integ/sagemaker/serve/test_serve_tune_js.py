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

import pytest

from sagemaker.serve import Mode
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from tests.integ import lock
from tests.integ.sagemaker.serve.constants import (
    SERVE_LOCAL_CONTAINER_TUNE_TIMEOUT,
    PYTHON_VERSION_IS_NOT_310,
)

from tests.integ.timeout import timeout
from tests.integ.utils import cleanup_model_resources
import logging

logger = logging.getLogger(__name__)

SAMPLE_PROMPT = {"inputs": "Hello, I'm a language model,", "parameters": {}}
SAMPLE_RESPONSE = [
    {"generated_text": "Hello, I'm a language model, and I'm here to help you with your English."}
]
JS_MODEL_ID = "huggingface-textgeneration-gpt2"


@pytest.fixture
def model_builder(sagemaker_local_session):
    return ModelBuilder(
        model=JS_MODEL_ID,
        schema_builder=SchemaBuilder(SAMPLE_PROMPT, SAMPLE_RESPONSE),
        mode=Mode.LOCAL_CONTAINER,
        sagemaker_session=sagemaker_local_session,
    )


# @pytest.mark.skipif(
#     PYTHON_VERSION_IS_NOT_310,
#     reason="The goal of these tests are to test the serving components of our feature",
# )
# @pytest.mark.local_mode
def test_happy_tgi_sagemaker_endpoint(model_builder, gpu_instance_type):
    logger.info("Running in LOCAL_CONTAINER mode...")
    caught_ex = None
    model = model_builder.build()

    # endpoint tests all use the same port, so we use this lock to prevent concurrent execution
    with lock.lock():
        try:
            print("************************************")
            logger.info("Deploying and predicting in LOCAL_CONTAINER mode...")
            predictor = predictor = model.deploy(instance_type=gpu_instance_type)
            logger.info("Endpoint successfully deployed.")
            print("************************************END")
        except Exception as e:
            caught_ex = e
        finally:
            predictor.delete_endpoint()
            if caught_ex:
                raise caught_ex


