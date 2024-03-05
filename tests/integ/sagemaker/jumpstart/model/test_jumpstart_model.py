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
import os
import time
from unittest import mock

import pytest
from sagemaker.enums import EndpointType
from sagemaker.predictor import retrieve_default

import tests.integ

from sagemaker.jumpstart.model import JumpStartModel
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
    JUMPSTART_TAG,
    InferenceTabularDataname,
)
from tests.integ.sagemaker.jumpstart.utils import (
    download_inference_assets,
    get_sm_session,
    get_tabular_data,
)

INF2_SUPPORTED_REGIONS = {
    "us-west-2",
    "us-east-1",
    "us-east-2",
}

MAX_INIT_TIME_SECONDS = 5

GATED_INFERENCE_MODEL_PACKAGE_SUPPORTED_REGIONS = {
    "us-west-2",
    "us-east-1",
    "eu-west-1",
    "ap-southeast-1",
    "us-east-2",
    "ap-southeast-2",
}


def test_non_prepacked_jumpstart_model(setup):

    model_id = "catboost-classification-model"

    model = JumpStartModel(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    # uses ml.m5.4xlarge instance
    model.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
    )

    predictor = retrieve_default(
        endpoint_name=model.endpoint_name,
        sagemaker_session=get_sm_session(),
        tolerate_vulnerable_model=True,
    )

    download_inference_assets()
    ground_truth_label, features = get_tabular_data(InferenceTabularDataname.MULTICLASS)

    response = predictor.predict(features)

    assert response is not None


def test_prepacked_jumpstart_model(setup):

    model_id = "huggingface-txt2img-conflictx-complex-lineart"

    model = JumpStartModel(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    # uses ml.p3.2xlarge instance
    predictor = model.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
    )

    response = predictor.predict("hello world!")

    assert response is not None


@pytest.mark.skipif(
    tests.integ.test_region() not in GATED_INFERENCE_MODEL_PACKAGE_SUPPORTED_REGIONS,
    reason=f"JumpStart model package inference models unavailable in {tests.integ.test_region()}.",
)
def test_model_package_arn_jumpstart_model(setup):

    model_id = "meta-textgeneration-llama-2-7b"

    model = JumpStartModel(
        model_id=model_id,
        model_version="2.*",  # version <3.0.0 uses model packages
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    # uses ml.g5.2xlarge instance
    predictor = model.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
    )

    payload = {
        "inputs": "some-payload",
        "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.6},
    }

    response = predictor.predict(payload, custom_attributes="accept_eula=true")

    assert response is not None


@pytest.mark.skipif(
    tests.integ.test_region() not in INF2_SUPPORTED_REGIONS,
    reason=f"INF2 instances unavailable in {tests.integ.test_region()}.",
)
def test_jumpstart_gated_model_neuron(setup):

    model_id = "meta-textgenerationneuron-llama-2-7b"

    model = JumpStartModel(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    # uses ml.inf2.xlarge instance
    predictor = model.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        accept_eula=True,
    )

    payload = {
        "inputs": "some-payload",
    }

    response = predictor.predict(payload)

    assert response is not None


def test_jumpstart_gated_model(setup):

    model_id = "meta-textgeneration-llama-2-7b"

    model = JumpStartModel(
        model_id=model_id,
        model_version="3.*",  # version >=3.0.0 stores artifacts in jumpstart-private-cache-* buckets
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    # uses ml.g5.2xlarge instance
    predictor = model.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        accept_eula=True,
    )

    payload = {
        "inputs": "some-payload",
        "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.6},
    }

    response = predictor.predict(payload)

    assert response is not None


def test_jumpstart_gated_model_inference_component_enabled(setup):

    model_id = "meta-textgeneration-llama-2-7b"

    model = JumpStartModel(
        model_id=model_id,
        model_version="3.*",  # version >=3.0.0 stores artifacts in jumpstart-private-cache-* buckets
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    # uses ml.g5.2xlarge instance
    model.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        accept_eula=True,
        endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED,
    )

    predictor = retrieve_default(
        endpoint_name=model.endpoint_name,
        sagemaker_session=get_sm_session(),
        tolerate_vulnerable_model=True,
    )

    payload = {
        "inputs": "some-payload",
        "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.6},
    }

    response = predictor.predict(payload)

    assert response is not None


@mock.patch("sagemaker.jumpstart.cache.JUMPSTART_LOGGER.warning")
def test_instatiating_model(mock_warning_logger, setup):

    model_id = "catboost-regression-model"

    start_time = time.perf_counter()

    JumpStartModel(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    elapsed_time = time.perf_counter() - start_time

    assert elapsed_time <= MAX_INIT_TIME_SECONDS

    mock_warning_logger.assert_called_once()


def test_jumpstart_model_register(setup):
    model_id = "huggingface-txt2img-conflictx-complex-lineart"

    model = JumpStartModel(
        model_id=model_id,
        model_version="1.1.0",
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    model_package = model.register()

    # uses  instance
    predictor = model_package.deploy(
        instance_type="ml.p3.2xlarge",
        initial_instance_count=1,
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
    )

    response = predictor.predict("hello world!")

    assert response is not None
