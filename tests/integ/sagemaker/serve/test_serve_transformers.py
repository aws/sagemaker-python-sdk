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

sample_input = {
    "inputs": "The man worked as a [MASK].",
}

loaded_response = [
    {
        "score": 0.0974755585193634,
        "token": 10533,
        "token_str": "carpenter",
        "sequence": "the man worked as a carpenter.",
    },
    {
        "score": 0.052383411675691605,
        "token": 15610,
        "token_str": "waiter",
        "sequence": "the man worked as a waiter.",
    },
    {
        "score": 0.04962712526321411,
        "token": 13362,
        "token_str": "barber",
        "sequence": "the man worked as a barber.",
    },
    {
        "score": 0.0378861166536808,
        "token": 15893,
        "token_str": "mechanic",
        "sequence": "the man worked as a mechanic.",
    },
    {
        "score": 0.037680838257074356,
        "token": 18968,
        "token_str": "salesman",
        "sequence": "the man worked as a salesman.",
    },
]


@pytest.fixture
def model_input():
    return {"inputs": "The man worked as a [MASK]."}


@pytest.fixture
def model_builder_model_schema_builder():
    return ModelBuilder(
        model_path=HF_DIR,
        model="bert-base-uncased",
        schema_builder=SchemaBuilder(sample_input, loaded_response),
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
def test_pytorch_transformers_sagemaker_endpoint(
    sagemaker_session, model_builder, model_input, **kwargs
):
    logger.info("Running in SAGEMAKER_ENDPOINT mode...")
    caught_ex = None

    iam_client = sagemaker_session.boto_session.client("iam")
    role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

    model = model_builder.build(
        mode=Mode.SAGEMAKER_ENDPOINT, role_arn=role_arn, sagemaker_session=sagemaker_session
    )

    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        try:
            logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
            predictor = model.deploy(
                instance_type=kwargs["instance_type"], initial_instance_count=2
            )
            logger.info("Endpoint successfully deployed.")
            predictor.predict(model_input)
            assert predictor is not None
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
                ), f"{caught_ex} was thrown when running pytorch transformers sagemaker endpoint test"
