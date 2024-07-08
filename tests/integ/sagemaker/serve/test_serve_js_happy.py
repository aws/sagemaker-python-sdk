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

import io
import sys

import pytest

from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from tests.integ.sagemaker.serve.constants import (
    SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
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
JS_MODEL_ID = "huggingface-textgeneration1-gpt-neo-125m-fp16"
ROLE_NAME = "SageMakerRole"

SAMPLE_MMS_PROMPT = [
    "How cute your dog is!",
    "Your dog is so cute.",
    "The mitochondria is the powerhouse of the cell.",
]
SAMPLE_MMS_RESPONSE = {"embedding": []}
JS_MMS_MODEL_ID = "huggingface-sentencesimilarity-bge-m3"


@pytest.fixture
def happy_model_builder(sagemaker_session):
    iam_client = sagemaker_session.boto_session.client("iam")
    return ModelBuilder(
        model=JS_MODEL_ID,
        schema_builder=SchemaBuilder(SAMPLE_PROMPT, SAMPLE_RESPONSE),
        role_arn=iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"],
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture
def meta_textgeneration_llama_2_7b_f_schema():
    prompt = "Hello, I'm a language model,"
    response = "Hello, I'm a language model, and I'm here to help you with your English."
    sample_input = {"inputs": prompt}
    sample_output = [{"generated_text": response}]

    return SchemaBuilder(
        sample_input=sample_input,
        sample_output=sample_output,
    )


@pytest.fixture
def happy_mms_model_builder(sagemaker_session):
    iam_client = sagemaker_session.boto_session.client("iam")
    return ModelBuilder(
        model=JS_MMS_MODEL_ID,
        schema_builder=SchemaBuilder(SAMPLE_MMS_PROMPT, SAMPLE_MMS_RESPONSE),
        role_arn=iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"],
        sagemaker_session=sagemaker_session,
    )


@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,
    reason="The goal of these test are to test the serving components of our feature",
)
@pytest.mark.slow_test
def test_happy_tgi_sagemaker_endpoint(happy_model_builder, gpu_instance_type):
    logger.info("Running in SAGEMAKER_ENDPOINT mode...")
    caught_ex = None
    model = happy_model_builder.build()

    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        try:
            logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
            predictor = model.deploy(instance_type=gpu_instance_type, endpoint_logging=False)
            logger.info("Endpoint successfully deployed.")

            updated_sample_input = happy_model_builder.schema_builder.sample_input

            predictor.predict(updated_sample_input)
        except Exception as e:
            caught_ex = e
        finally:
            cleanup_model_resources(
                sagemaker_session=happy_model_builder.sagemaker_session,
                model_name=model.name,
                endpoint_name=model.endpoint_name,
            )
            if caught_ex:
                raise caught_ex


@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,
    reason="The goal of these test are to test the serving components of our feature",
)
@pytest.mark.slow_test
def test_happy_mms_sagemaker_endpoint(happy_mms_model_builder, gpu_instance_type):
    logger.info("Running in SAGEMAKER_ENDPOINT mode...")
    caught_ex = None
    model = happy_mms_model_builder.build()

    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        try:
            logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
            predictor = model.deploy(instance_type=gpu_instance_type, endpoint_logging=False)
            logger.info("Endpoint successfully deployed.")

            updated_sample_input = happy_mms_model_builder.schema_builder.sample_input

            predictor.predict(updated_sample_input)
        except Exception as e:
            caught_ex = e
        finally:
            cleanup_model_resources(
                sagemaker_session=happy_mms_model_builder.sagemaker_session,
                model_name=model.name,
                endpoint_name=model.endpoint_name,
            )
            if caught_ex:
                raise caught_ex


@pytest.mark.skipif(
    True,
    reason="Only enable after metadata is fully deployed.",
)
def test_js_model_with_deployment_configs(
    meta_textgeneration_llama_2_7b_f_schema,
    sagemaker_session,
):
    logger.info("Running in SAGEMAKER_ENDPOINT mode...")
    caught_ex = None
    iam_client = sagemaker_session.boto_session.client("iam")
    role_arn = iam_client.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

    model_builder = ModelBuilder(
        model="meta-textgeneration-llama-2-13b",
        schema_builder=meta_textgeneration_llama_2_7b_f_schema,
    )
    configs = model_builder.list_deployment_configs()

    assert len(configs) > 0

    captured_output = io.StringIO()
    sys.stdout = captured_output
    model_builder.display_benchmark_metrics()
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue() is not None

    model_builder.set_deployment_config(
        configs[0]["ConfigName"],
        "ml.g5.2xlarge",
    )
    model = model_builder.build(role_arn=role_arn, sagemaker_session=sagemaker_session)
    assert model.config_name == configs[0]["ConfigName"]
    assert model_builder.get_deployment_config() is not None

    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        try:
            logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
            predictor = model.deploy(accept_eula=True)
            logger.info("Endpoint successfully deployed.")

            updated_sample_input = model_builder.schema_builder.sample_input

            predictor.predict(updated_sample_input)
        except Exception as e:
            caught_ex = e
        finally:
            cleanup_model_resources(
                sagemaker_session=sagemaker_session,
                model_name=model.name,
                endpoint_name=model.endpoint_name,
            )
            if caught_ex:
                raise caught_ex
