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
import os
import sys
import time
from unittest import mock
import logging

import pytest
from sagemaker.enums import EndpointType
from sagemaker.jumpstart.hub.hub import Hub
from sagemaker.jumpstart.hub.utils import generate_hub_arn_for_init_kwargs
from sagemaker.predictor import retrieve_default

from sagemaker.jumpstart.constants import JUMPSTART_LOGGER

import tests.integ

from sagemaker.jumpstart.model import JumpStartModel
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME,
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
    JUMPSTART_TAG,
)
from tests.integ.sagemaker.jumpstart.utils import (
    get_public_hub_model_arn,
    get_sm_session,
)

MAX_INIT_TIME_SECONDS = 5

TEST_MODEL_IDS = {
    "catboost-classification-model",
    "huggingface-txt2img-conflictx-complex-lineart",
    "meta-textgeneration-llama-2-7b",
    "meta-textgeneration-llama-3-2-1b",
    "catboost-regression-model",
}


@pytest.fixture(scope="module")
def add_models():
    # Create Model References to test in Hub
    hub_instance = Hub(
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME], sagemaker_session=get_sm_session()
    )
    for model in TEST_MODEL_IDS:
        hub_instance.create_model_reference(model_arn=get_public_hub_model_arn(hub_instance, model))


def test_jumpstart_hub_model(setup, add_models):

    JUMPSTART_LOGGER.info("starting test")
    JUMPSTART_LOGGER.info(f"get identity {get_sm_session().get_caller_identity_arn()}")

    model_id = "catboost-classification-model"

    model = JumpStartModel(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
    )

    # uses ml.m5.4xlarge instance
    model.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
    )


def test_jumpstart_hub_gated_model(setup, add_models):

    model_id = "meta-textgeneration-llama-3-2-1b"

    model = JumpStartModel(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
    )

    # uses ml.g6.xlarge instance
    predictor = model.deploy(
        accept_eula=True,
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
    )

    payload = {
        "inputs": "some-payload",
        "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.6},
    }

    response = predictor.predict(payload, custom_attributes="accept_eula=true")

    assert response is not None


def test_jumpstart_gated_model_inference_component_enabled(setup, add_models):

    model_id = "meta-textgeneration-llama-2-7b"

    hub_name = os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME]

    region = tests.integ.test_region()

    sagemaker_session = get_sm_session()

    hub_arn = generate_hub_arn_for_init_kwargs(
        hub_name=hub_name, region=region, session=sagemaker_session
    )

    model = JumpStartModel(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=sagemaker_session,
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
    )

    # uses ml.g5.2xlarge instance
    model.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        accept_eula=True,
        endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
    )

    predictor = retrieve_default(
        endpoint_name=model.endpoint_name,
        sagemaker_session=sagemaker_session,
        tolerate_vulnerable_model=True,
        hub_arn=hub_arn,
    )

    payload = {
        "inputs": "some-payload",
        "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.6},
    }

    response = predictor.predict(payload)

    assert response is not None

    model = JumpStartModel.attach(
        predictor.endpoint_name, sagemaker_session=sagemaker_session, hub_name=hub_name
    )
    assert model.model_id == model_id
    assert model.endpoint_name == predictor.endpoint_name
    assert model.inference_component_name == predictor.component_name


def test_instatiating_model(setup, add_models):

    model_id = "catboost-regression-model"

    start_time = time.perf_counter()

    JumpStartModel(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
    )

    elapsed_time = time.perf_counter() - start_time

    assert elapsed_time <= MAX_INIT_TIME_SECONDS
