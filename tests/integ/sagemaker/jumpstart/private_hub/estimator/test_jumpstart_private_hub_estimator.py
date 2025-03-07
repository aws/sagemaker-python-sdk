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

from tests.integ.sagemaker.jumpstart.private_hub.setup import add_model_references


MAX_INIT_TIME_SECONDS = 5


def test_jumpstart_hub_estimator(setup, add_model_references):
    model_id, model_version = "huggingface-spc-bert-base-cased", "*"

    estimator = JumpStartEstimator(
        model_id=model_id,
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
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
    )

    response = predictor.predict(["hello", "world"])

    assert response is not None


def test_jumpstart_hub_estimator_with_session(setup, add_model_references):

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


def test_jumpstart_hub_gated_estimator_with_eula(setup, add_model_references):

    model_id, model_version = "meta-textgeneration-llama-2-7b", "*"

    estimator = JumpStartEstimator(
        model_id=model_id,
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
    )

    estimator.fit(
        accept_eula=True,
        inputs={
            "training": f"s3://{get_jumpstart_content_bucket(JUMPSTART_DEFAULT_REGION_NAME)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        },
    )

    predictor = estimator.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    payload = {
        "inputs": "some-payload",
        "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.6},
    }

    response = predictor.predict(payload, custom_attributes="accept_eula=true")

    assert response is not None


def test_jumpstart_hub_gated_estimator_without_eula(setup, add_model_references):

    model_id, model_version = "meta-textgeneration-llama-2-7b", "*"

    estimator = JumpStartEstimator(
        model_id=model_id,
        hub_name=os.environ[ENV_VAR_JUMPSTART_SDK_TEST_HUB_NAME],
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
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
