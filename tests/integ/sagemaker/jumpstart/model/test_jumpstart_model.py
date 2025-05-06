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

TEST_HUB_WITH_REFERENCE = "mock-hub-name"


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
        model_version="*",  # version >=3.0.0 stores artifacts in jumpstart-private-cache-* buckets
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
        model_version="*",  # version >=3.0.0 stores artifacts in jumpstart-private-cache-* buckets
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

    model = JumpStartModel.attach(predictor.endpoint_name, sagemaker_session=get_sm_session())
    assert model.model_id == model_id
    assert model.endpoint_name == predictor.endpoint_name
    assert model.inference_component_name == predictor.component_name


@mock.patch("sagemaker.jumpstart.cache.JUMPSTART_LOGGER.warning")
def test_instantiating_model(mock_warning_logger, setup):

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
    )

    response = predictor.predict("hello world!")

    predictor.delete_predictor()

    assert response is not None


@pytest.mark.skipif(
    True,
    reason="Only enable if test account is subscribed to the proprietary model",
)
def test_proprietary_jumpstart_model(setup):

    model_id = "ai21-jurassic-2-light"

    model = JumpStartModel(
        model_id=model_id,
        model_version="2.0.004",
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    predictor = model.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}]
    )
    payload = {"prompt": "To be, or", "maxTokens": 4, "temperature": 0, "numResults": 1}

    response = predictor.predict(payload)

    assert response is not None


@pytest.mark.skipif(
    True,
    reason="Only enable if test account is subscribed to the proprietary model",
)
def test_register_proprietary_jumpstart_model(setup):

    model_id = "ai21-jurassic-2-light"

    model = JumpStartModel(
        model_id=model_id,
        model_version="2.0.004",
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )
    model_package = model.register()

    predictor = model_package.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}]
    )
    payload = {"prompt": "To be, or", "maxTokens": 4, "temperature": 0, "numResults": 1}

    response = predictor.predict(payload)

    predictor.delete_predictor()

    assert response is not None


@pytest.mark.skipif(
    True,
    reason="Only enable if test account is subscribed to the proprietary model",
)
def test_register_gated_jumpstart_model(setup):

    model_id = "meta-textgenerationneuron-llama-2-7b"
    model = JumpStartModel(
        model_id=model_id,
        model_version="1.1.0",
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )
    model_package = model.register(accept_eula=True)

    predictor = model_package.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        accept_eula=True,
    )
    payload = {"prompt": "To be, or", "maxTokens": 4, "temperature": 0, "numResults": 1}

    response = predictor.predict(payload)

    predictor.delete_predictor()

    assert response is not None


@pytest.mark.skipif(
    True,
    reason="Only enable after metadata is fully deployed.",
)
def test_jumpstart_model_with_deployment_configs(setup):
    model_id = "meta-textgeneration-llama-2-13b"

    model = JumpStartModel(
        model_id=model_id,
        model_version="*",
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    captured_output = io.StringIO()
    sys.stdout = captured_output
    model.display_benchmark_metrics()
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue() is not None

    configs = model.list_deployment_configs()
    assert len(configs) > 0

    model.set_deployment_config(
        configs[0]["ConfigName"],
        "ml.g5.2xlarge",
    )
    assert model.config_name == configs[0]["ConfigName"]

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


def test_jumpstart_session_with_config_name():
    model = JumpStartModel(model_id="meta-textgeneration-llama-2-7b")
    assert model.config_name is not None
    session = model.sagemaker_session

    # we're mocking the http request, so it's expected to raise an Exception.
    # we're interested that the low-level request attaches the correct
    # jumpstart-related tags.
    with mock.patch("botocore.client.BaseClient._make_request") as mock_make_request:
        try:
            session.sagemaker_client.list_endpoints()
        except Exception:
            pass

    assert (
        "md/js_model_id#meta-textgeneration-llama-2-7b md/js_model_ver#* md/js_config#tgi"
        in mock_make_request.call_args[0][1]["headers"]["User-Agent"]
    )


def _setup_test_hub_with_reference(public_hub_model_id: str):
    session = get_sm_session()

    try:
        session.create_hub(
            hub_name=TEST_HUB_WITH_REFERENCE,
            hub_description="this is my sagemaker hub",
            hub_display_name="Mock Hub",
            hub_search_keywords=["mock", "hub", "123"],
            s3_storage_config={"S3OutputPath": "s3://my-hub-bucket/"},
            tags=[{"Key": "tag-key-1", "Value": "tag-value-1"}],
        )
    except Exception as e:
        if "ResourceInUse" in str(e):
            print("Hub already exists")
        else:
            raise e

    try:
        session.create_hub_content_reference(
            hub_name=TEST_HUB_WITH_REFERENCE,
            source_hub_content_arn=(
                f"arn:aws:sagemaker:{session.boto_region_name}:aws:"
                f"hub-content/SageMakerPublicHub/Model/{public_hub_model_id}"
            ),
        )
    except Exception as e:
        if "ResourceInUse" in str(e):
            print("Reference already exists")
        else:
            raise e


def _teardown_test_hub_with_reference(public_hub_model_id: str):
    session = get_sm_session()

    try:
        session.delete_hub_content_reference(
            hub_name=TEST_HUB_WITH_REFERENCE,
            hub_content_type="ModelReference",
            hub_content_name=public_hub_model_id,
        )
    except Exception as e:
        if "ResourceInUse" in str(e):
            print("Reference already exists")
        else:
            raise e

    try:
        session.delete_hub(hub_name=TEST_HUB_WITH_REFERENCE)
    except Exception as e:
        if "ResourceInUse" in str(e):
            print("Hub already exists")
        else:
            raise e


@pytest.mark.skip
# Currently JumpStartModel does not pull from HubService for the Public Hub.
def test_model_reference_marketplace_model(setup):
    session = get_sm_session()

    # TODO: hardcoded model ID is brittle - should be dynamic pull via ListHubContents
    public_hub_marketplace_model_id = "upstage-solar-mini-chat"
    _setup_test_hub_with_reference(public_hub_marketplace_model_id)

    JumpStartModel(  # Retrieving MP model None -> defaults to latest SemVer
        model_id=public_hub_marketplace_model_id,
        hub_name=TEST_HUB_WITH_REFERENCE,
        role=session.get_caller_identity_arn(),
        sagemaker_session=session,
    )

    model_semver = JumpStartModel(  # Retrieving MP model SemVer -> uses SemVer
        model_id=public_hub_marketplace_model_id,
        hub_name=TEST_HUB_WITH_REFERENCE,
        role=session.get_caller_identity_arn(),
        sagemaker_session=session,
        model_version="1.0.0",
    )

    model_marketplace_version = JumpStartModel(  # Retrieving MP model MP version -> uses MPver
        model_id=public_hub_marketplace_model_id,
        hub_name=TEST_HUB_WITH_REFERENCE,
        role=session.get_caller_identity_arn(),
        sagemaker_session=session,
        model_version="240612.5",
    )

    _teardown_test_hub_with_reference(public_hub_marketplace_model_id)  # Cleanup before assertions

    assert model_semver.model_version == model_marketplace_version.model_version


# TODO: PySDK test account not subscribed to this model
# def test_model_reference_marketplace_model_deployment(setup):
#     session = get_sm_session()
#     public_hub_marketplace_model_id = "upstage-solar-mini-chat"
#     _setup_test_hub_with_reference(public_hub_marketplace_model_id)

#     marketplace_model = JumpStartModel(  # Retrieving MP model MP version -> uses MPver
#         model_id=public_hub_marketplace_model_id,
#         hub_name=TEST_HUB_WITH_REFERENCE,
#         role=session.get_caller_identity_arn(),
#         sagemaker_session=session,
#         model_version="240612.5",
#     )
#     predictor = marketplace_model.deploy(
#         tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
#         accept_eula=True,
#     )

#     predictor.delete_predictor()
#     _teardown_test_hub_with_reference(public_hub_marketplace_model_id)


@pytest.mark.skip
def test_bedrock_store_model_tags_from_hub_service(setup):

    session = get_sm_session()
    brs_model_id = "huggingface-llm-gemma-2b-instruct"
    _setup_test_hub_with_reference(brs_model_id)

    brs_model = JumpStartModel(
        model_id=brs_model_id,
        hub_name=TEST_HUB_WITH_REFERENCE,
        role=session.get_caller_identity_arn(),
        sagemaker_session=session,
    )

    predictor = brs_model.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        accept_eula=True,
    )

    endpoint_arn = (
        f"arn:aws:sagemaker:{session.boto_region_name}:"
        f"{session.account_id()}:endpoint/{predictor.endpoint_name}"
    )
    tags = session.list_tags(endpoint_arn)

    predictor.delete_predictor()  # Cleanup before assertions
    _teardown_test_hub_with_reference(brs_model_id)

    expected_tag = {"Key": "sagemaker-sdk:bedrock", "Value": "compatible"}
    assert expected_tag in tags
