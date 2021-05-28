# Copyright 2018-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import pytest
from sagemaker.xgboost.processing import XGBoostProcessor
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout

ROLE = "SageMakerRole"


@pytest.mark.release
def test_framework_processing_job_with_deps(
    sagemaker_session,
    xgboost_latest_version,
    xgboost_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        code_path = os.path.join(DATA_DIR, "dummy_code_bundle_with_reqs")
        entry_point = "main_script.py"

        processor = XGBoostProcessor(
            framework_version=xgboost_latest_version,
            py_version=xgboost_latest_py_version,
            role=ROLE,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            base_job_name="test-xgboost",
        )

        processor.run(
            code=entry_point,
            source_dir=code_path,
            inputs=[],
            wait=True,
        )
