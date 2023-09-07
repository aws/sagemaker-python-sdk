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

from sagemaker.jumpstart.estimator import JumpStartEstimator
import tests
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
    JUMPSTART_TAG,
)
from tests.integ.sagemaker.jumpstart.utils import (
    get_sm_session,
    get_training_dataset_for_model_and_version,
    x_fail_if_ice,
)

from sagemaker.jumpstart.utils import get_jumpstart_content_bucket


MAX_INIT_TIME_SECONDS = 5

GATED_TRAINING_MODEL_SUPPORTED_REGIONS = {
    "us-west-2",
    "us-east-1",
    "eu-west-1",
    "ap-southeast-1",
    "us-east-2",
    "ap-southeast-2",
}


def test_jumpstart_estimator(setup):

    model_id, model_version = "huggingface-spc-bert-base-cased", "*"

    estimator = JumpStartEstimator(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        max_run=259200,  # avoid exceeding resource limits
    )

    # uses ml.p3.2xlarge instance
    estimator.fit(
        {
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


@x_fail_if_ice
@pytest.mark.skipif(
    tests.integ.test_region() not in GATED_TRAINING_MODEL_SUPPORTED_REGIONS,
    reason=f"JumpStart gated training models unavailable in {tests.integ.test_region()}.",
)
def test_gated_model_training(setup):

    model_id, model_version = "meta-textgeneration-llama-2-7b", "*"

    estimator = JumpStartEstimator(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        environment={"accept_eula": "true"},
        max_run=259200,  # avoid exceeding resource limits
    )

    # uses ml.g5.12xlarge instance
    estimator.fit(
        {
            "training": f"s3://{get_jumpstart_content_bucket(JUMPSTART_DEFAULT_REGION_NAME)}/"
            f"{get_training_dataset_for_model_and_version(model_id, model_version)}",
        }
    )

    # uses ml.g5.2xlarge instance
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


def test_instatiating_estimator_not_too_slow(setup):

    model_id = "xgboost-classification-model"

    start_time = time.perf_counter()

    JumpStartEstimator(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    elapsed_time = time.perf_counter() - start_time

    assert elapsed_time <= MAX_INIT_TIME_SECONDS
