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

from sagemaker import AutoML
from tests.integ import DATA_DIR, AUTO_ML_DEFAULT_TIMEMOUT_MINUTES
from tests.integ.timeout import timeout

ROLE = "SageMakerRole"
DATA_DIR = os.path.join(DATA_DIR, "automl", "data")
PREFIX = "sagemaker/beta-automl-xgboost"
TRAINING_DATA = os.path.join(DATA_DIR, "iris_training.csv")
TARGET_ATTRIBUTE_NAME = "virginica"


def create_auto_ml_job_if_not_exist(sagemaker_session, auto_ml_job_name):
    try:
        sagemaker_session.describe_auto_ml_job(job_name=auto_ml_job_name)
    except Exception as e:  # noqa: F841
        auto_ml = AutoML(
            role=ROLE,
            target_attribute_name=TARGET_ATTRIBUTE_NAME,
            sagemaker_session=sagemaker_session,
            max_candidates=3,
        )
        inputs = sagemaker_session.upload_data(path=TRAINING_DATA, key_prefix=PREFIX + "/input")
        with timeout(minutes=AUTO_ML_DEFAULT_TIMEMOUT_MINUTES):
            auto_ml.fit(inputs, job_name=auto_ml_job_name, wait=True)
