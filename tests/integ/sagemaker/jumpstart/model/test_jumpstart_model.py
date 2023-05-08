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


MAX_INIT_TIME_SECONDS = 5


def test_non_prepacked_jumpstart_model(setup):

    model_id = "catboost-classification-model"

    model = JumpStartModel(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    # uses ml.m5.4xlarge instance
    predictor = model.deploy(
        tags=[{"Key": JUMPSTART_TAG, "Value": os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]}],
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


def test_instatiating_model_not_too_slow(setup):

    model_id = "catboost-regression-model"

    start_time = time.perf_counter()

    JumpStartModel(
        model_id=model_id,
        role=get_sm_session().get_caller_identity_arn(),
        sagemaker_session=get_sm_session(),
    )

    elapsed_time = time.perf_counter() - start_time

    assert elapsed_time <= MAX_INIT_TIME_SECONDS
