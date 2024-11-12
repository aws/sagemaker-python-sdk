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
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.builder.model_builder import ModelBuilder, Mode
import tests.integ
from tests.integ.sagemaker.serve.constants import (
    HF_DIR,
    PYTHON_VERSION_IS_NOT_310,
    SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
)
from tests.integ.timeout import timeout
from tests.integ.utils import cleanup_model_resources, gpu_list, retry_with_instance_list
import logging

logger = logging.getLogger(__name__)

model_id = "bert-base-uncased"

sample_input = {"inputs": "Hello I'm a [MASK] model."}

sample_output = [
    {
        "score": 0.10731109976768494,
        "token": 4827,
        "token_str": "fashion",
        "sequence": "hello i'm a fashion model.",
    },
    {
        "score": 0.08774465322494507,
        "token": 2535,
        "token_str": "role",
        "sequence": "hello i'm a role model.",
    },
    {
        "score": 0.05338414013385773,
        "token": 2047,
        "token_str": "new",
        "sequence": "hello i'm a new model.",
    },
    {
        "score": 0.04667224362492561,
        "token": 3565,
        "token_str": "super",
        "sequence": "hello i'm a super model.",
    },
    {
        "score": 0.027096163481473923,
        "token": 2986,
        "token_str": "fine",
        "sequence": "hello i'm a fine model.",
    },
]


@pytest.fixture
def model_input():
    return {"inputs": "The man worked as a [MASK]."}


@pytest.fixture
def model_builder_model_schema_builder():
    return ModelBuilder(
        model_path=HF_DIR, model=model_id, schema_builder=SchemaBuilder(sample_input, sample_output)
    )


@pytest.fixture
def model_builder(request):
    return request.getfixturevalue(request.param)


@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,
    tests.integ.test_region() in tests.integ.TRAINING_NO_P2_REGIONS
    and tests.integ.test_region() in tests.integ.TRAINING_NO_P3_REGIONS,
    reason="no ml.p2 or ml.p3 instances in this region",
)
@retry_with_instance_list(gpu_list(tests.integ.test_region()))
@pytest.mark.parametrize("model_builder", ["model_builder_model_schema_builder"], indirect=True)
def test_non_text_generation_model_single_GPU(
    sagemaker_session, model_builder, model_input, **kwargs
):
    iam_client = sagemaker_session.boto_session.client("iam")
    role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]
    model = model_builder.build(role_arn=role_arn, sagemaker_session=sagemaker_session)
    caught_ex = None
    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        try:
            logger.info("Running in SAGEMAKER_ENDPOINT mode")
            predictor = model.deploy(
                mode=Mode.SAGEMAKER_ENDPOINT,
                instance_type=kwargs["instance_type"],
                initial_instance_count=1,
            )
            logger.info("Endpoint successfully deployed.")
            prediction = predictor.predict(model_input)
            assert prediction is not None

            endpoint_name = predictor.endpoint_name
            sagemaker_client = sagemaker_session.boto_session.client("sagemaker")
            endpoint_config_name = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)[
                "EndpointConfigName"
            ]
            actual_instance_type = sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )["ProductionVariants"][0]["InstanceType"]
            assert kwargs["instance_type"] == actual_instance_type
        except Exception as e:
            caught_ex = e
        finally:
            cleanup_model_resources(
                sagemaker_session=model_builder.sagemaker_session,
                model_name=model.name,
                endpoint_name=model.endpoint_name,
            )
            if caught_ex:
                logger.exception(caught_ex)
                assert (
                    False
                ), f"Exception {caught_ex} was thrown when running model builder single GPU test"


@pytest.mark.skipif(
    PYTHON_VERSION_IS_NOT_310,
    tests.integ.test_region() in tests.integ.TRAINING_NO_P2_REGIONS
    and tests.integ.test_region() in tests.integ.TRAINING_NO_P3_REGIONS,
    reason="no ml.p2 or ml.p3 instances in this region",
)
@retry_with_instance_list(gpu_list(tests.integ.test_region()))
@pytest.mark.parametrize("model_builder", ["model_builder_model_schema_builder"], indirect=True)
def test_non_text_generation_model_multi_GPU(
    sagemaker_session, model_builder, model_input, **kwargs
):
    iam_client = sagemaker_session.boto_session.client("iam")
    role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]
    caught_ex = None
    model = model_builder.build(role_arn=role_arn, sagemaker_session=sagemaker_session)
    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        try:
            logger.info("Running in SAGEMAKER_ENDPOINT mode")
            predictor = model.deploy(
                mode=Mode.SAGEMAKER_ENDPOINT,
                instance_type=kwargs["instance_type"],
                initial_instance_count=1,
            )
            logger.info("Endpoint successfully deployed.")
            prediction = predictor.predict(model_input)
            assert prediction is not None

            endpoint_name = predictor.endpoint_name
            sagemaker_client = sagemaker_session.boto_session.client("sagemaker")
            endpoint_config_name = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)[
                "EndpointConfigName"
            ]
            actual_instance_type = sagemaker_client.describe_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )["ProductionVariants"][0]["InstanceType"]
            assert kwargs["instance_type"] == actual_instance_type
        except Exception as e:
            caught_ex = e
        finally:
            cleanup_model_resources(
                sagemaker_session=model_builder.sagemaker_session,
                model_name=model.name,
                endpoint_name=model.endpoint_name,
            )
            if caught_ex:
                logger.exception(caught_ex)
                assert (
                    False
                ), f"Exception {caught_ex} was thrown when running model builder multi GPU test"
