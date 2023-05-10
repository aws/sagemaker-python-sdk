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
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME

from sagemaker.jumpstart.estimator import JumpStartEstimator
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
    JUMPSTART_TAG,
)
from tests.integ.sagemaker.jumpstart.utils import (
    get_sm_session,
    get_training_dataset_for_model_and_version,
)

from sagemaker.jumpstart.utils import get_jumpstart_content_bucket


MAX_INIT_TIME_SECONDS = 5


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

    # uses ml.p3.2xlarge instance
    predictor = estimator.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    response = predictor.predict(["hello", "world"])

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
