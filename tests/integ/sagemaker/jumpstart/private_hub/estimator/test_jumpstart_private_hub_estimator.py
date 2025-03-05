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

import pytest
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.jumpstart.hub.hub import Hub

from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket

from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME,
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
    JUMPSTART_TAG,
)
from tests.integ.sagemaker.jumpstart.utils import (
    get_public_hub_model_arn,
    get_sm_session,
    with_exponential_backoff,
    get_training_dataset_for_model_and_version,
)

MAX_INIT_TIME_SECONDS = 5

TEST_MODEL_IDS = {
    "huggingface-spc-bert-base-cased",
    "meta-textgeneration-llama-2-7b",
    "catboost-regression-model",
}


@with_exponential_backoff()
def create_model_reference(hub_instance, model_arn):
    hub_instance.create_model_reference(model_arn=model_arn)


@pytest.fixture(scope="session")
def add_model_references():
    # Create Model References to test in Hub
    hub_instance = Hub(
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME], sagemaker_session=get_sm_session()
    )
    for model in TEST_MODEL_IDS:
        model_arn = get_public_hub_model_arn(hub_instance, model)
        create_model_reference(hub_instance, model_arn)


def test_jumpstart_hub_estimator(setup, add_model_references):

    model_id, model_version = "huggingface-spc-bert-base-cased", "*"

    sagemaker_session = get_sm_session()

    estimator = JumpStartEstimator(
        model_id=model_id,
        role=sagemaker_session.get_caller_identity_arn(),
        sagemaker_session=sagemaker_session,
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
    )

    estimator.fit(
        inputs={
            "training": f"s3://{get_jumpstart_content_bucket(JUMPSTART_DEFAULT_REGION_NAME)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        }
    )

    # test that we can create a JumpStartEstimator from existing job with `attach`
    estimator = JumpStartEstimator.attach(
        training_job_name=estimator.latest_training_job.name,
        model_id=model_id,
        model_version=model_version,
        sagemaker_session=get_sm_session(),
    )

    # uses ml.p3.2xlarge instance
    predictor = estimator.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    response = predictor.predict(["hello", "world"])

    assert response is not None


def test_jumpstart_hub_estimator_with_default_session(setup, add_model_references):
    model_id, model_version = "huggingface-spc-bert-base-cased", "*"

    sagemaker_session = get_sm_session()

    estimator = JumpStartEstimator(
        model_id=model_id,
        role=sagemaker_session.get_caller_identity_arn(),
        sagemaker_session=sagemaker_session,
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
    )

    estimator.fit(
        inputs={
            "training": f"s3://{get_jumpstart_content_bucket(JUMPSTART_DEFAULT_REGION_NAME)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        }
    )

    # test that we can create a JumpStartEstimator from existing job with `attach`
    estimator = JumpStartEstimator.attach(
        training_job_name=estimator.latest_training_job.name,
        model_id=model_id,
        model_version=model_version,
    )

    # uses ml.p3.2xlarge instance
    predictor = estimator.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        role=get_sm_session().get_caller_identity_arn(),
    )

    response = predictor.predict(["hello", "world"])

    assert response is not None


def test_jumpstart_hub_gated_estimator_with_eula(setup, add_model_references):

    model_id, model_version = "meta-textgeneration-llama-2-7b", "*"

    estimator = JumpStartEstimator(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
    )

    estimator.fit(
        accept_eula=True,
        inputs={
            "training": f"s3://{get_jumpstart_content_bucket(JUMPSTART_DEFAULT_REGION_NAME)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        },
    )

    estimator = JumpStartEstimator.attach(
        training_job_name=estimator.latest_training_job.name,
        model_id=model_id,
        model_version=model_version,
        sagemaker_session=get_sm_session(),
    )

    # uses ml.p3.2xlarge instance
    predictor = estimator.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    response = predictor.predict(["hello", "world"])

    assert response is not None


def test_jumpstart_hub_gated_estimator_without_eula(setup, add_model_references):

    model_id, model_version = "meta-textgeneration-llama-2-7b", "*"

    estimator = JumpStartEstimator(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
    )
    with pytest.raises(Exception):
        estimator.fit(
            inputs={
                "training": f"s3://{get_jumpstart_content_bucket(JUMPSTART_DEFAULT_REGION_NAME)}/"
                f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
            }
        )


def test_instantiating_estimator(setup, add_model_references):

    model_id = "catboost-regression-model"

    start_time = time.perf_counter()

    JumpStartEstimator(
        model_id=model_id,
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
    )

    elapsed_time = time.perf_counter() - start_time

    assert elapsed_time <= MAX_INIT_TIME_SECONDS
