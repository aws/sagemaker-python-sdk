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
import tests.integ
import uuid

from botocore.exceptions import ClientError
from sagemaker.predictor import Predictor
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
from sagemaker.utils import unique_name_from_base

from tests.integ.sagemaker.serve.constants import (
    SERVE_SAGEMAKER_ENDPOINT_TIMEOUT,
)
from tests.integ.timeout import timeout
import logging

logger = logging.getLogger(__name__)

sample_input = {"inputs": "What are falcons?", "parameters": {"max_new_tokens": 32}}

sample_output = [
    {
        "generated_text": "Falcons are small to medium-sized birds of prey related to hawks and eagles."
    }
]

LLAMA_2_7B_JS_ID = "meta-textgeneration-llama-2-7b"
LLAMA_IC_NAME = "llama2-mb-ic"
INSTANCE_TYPE = "ml.g5.24xlarge"


@pytest.fixture
def model_builder_llama_inference_component():
    return ModelBuilder(
        model=LLAMA_2_7B_JS_ID,
        schema_builder=SchemaBuilder(sample_input, sample_output),
        resource_requirements=ResourceRequirements(
            requests={"memory": 98304, "num_accelerators": 4, "copies": 1, "num_cpus": 40}
        ),
    )


@pytest.mark.skipif(
    tests.integ.test_region() not in "us-west-2",
    reason="G5 capacity available in PDX.",
)
def test_model_builder_ic_sagemaker_endpoint(
    sagemaker_session,
    model_builder_llama_inference_component,
):
    logger.info("Running in SAGEMAKER_ENDPOINT mode...")
    caught_ex = None

    model_builder_llama_inference_component.sagemaker_session = sagemaker_session
    model_builder_llama_inference_component.instance_type = INSTANCE_TYPE

    model_builder_llama_inference_component.inference_component_name = unique_name_from_base(
        LLAMA_IC_NAME
    )

    iam_client = sagemaker_session.boto_session.client("iam")
    role_arn = iam_client.get_role(RoleName="SageMakerRole")["Role"]["Arn"]

    chain = ModelBuilder(
        modelbuilder_list=[
            model_builder_llama_inference_component,
        ],
        role_arn=role_arn,
        sagemaker_session=sagemaker_session,
    )

    chain.build()

    with timeout(minutes=SERVE_SAGEMAKER_ENDPOINT_TIMEOUT):
        try:
            logger.info("Deploying and predicting in SAGEMAKER_ENDPOINT mode...")
            endpoint_name = f"llama-ic-endpoint-name-{uuid.uuid1().hex}"
            predictors = chain.deploy(
                instance_type=INSTANCE_TYPE,
                initial_instance_count=1,
                accept_eula=True,
                endpoint_name=endpoint_name,
            )
            logger.info("Inference components successfully deployed.")
            predictors[0].predict(sample_input)
            assert len(predictors) == 1
        except Exception as e:
            caught_ex = e
        finally:
            if caught_ex:
                logger.exception(caught_ex)
                cleanup_resources(sagemaker_session, [LLAMA_IC_NAME])
                assert False, f"{caught_ex} thrown when running mb-IC deployment test."

            cleanup_resources(sagemaker_session, [LLAMA_IC_NAME])


def cleanup_resources(sagemaker_session, ic_base_names):
    sm_client = sagemaker_session.sagemaker_client

    endpoint_names = set()
    for ic_base_name in ic_base_names:
        response = sm_client.list_inference_components(
            NameContains=ic_base_name, StatusEquals="InService"
        )
        ics = response["InferenceComponents"]

        logger.info(f"Cleaning up {len(ics)} ICs with base name {ic_base_name}.")
        for ic in ics:
            ic_name = ic["InferenceComponentName"]
            ep_name = ic["EndpointName"]

            try:
                logger.info(f"Deleting IC with name {ic_name}")
                Predictor(
                    endpoint_name=ep_name,
                    component_name=ic_name,
                    sagemaker_session=sagemaker_session,
                ).delete_predictor()
                sagemaker_session.wait_for_inference_component_deletion(
                    inference_component_name=ic_name,
                    poll=10,
                )
                endpoint_names.add(ep_name)
            except ClientError as e:
                logger.warning(e)

    for endpoint_name in endpoint_names:
        logger.info(f"Deleting endpoint with name {endpoint_name}")
        try:
            Predictor(
                endpoint_name=endpoint_name, sagemaker_session=sagemaker_session
            ).delete_endpoint()
        except ClientError as e:
            logger.warning(e)
