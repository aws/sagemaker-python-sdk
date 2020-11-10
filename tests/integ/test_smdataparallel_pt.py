# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import sagemaker.utils
import tests.integ as integ

from sagemaker.pytorch import PyTorch
from tests.integ import timeout


smdataparallel_dir = os.path.join(
    os.path.dirname(__file__), "..", "data", "smdistributed_dataparallel"
)


@pytest.mark.skip(
    reason="SMDistributedDataParallel-enabled DLC isn't publicly released hence not accessible for this test"
)
def test_smdataparallel_pt_mnist(sagemaker_session):
    job_name = sagemaker.utils.unique_name_from_base("pt-sm-distributed-dataparallel")
    estimator = PyTorch(
        entry_point="mnist_pt.py",
        role="SageMakerRole",
        image_uri="redacted",
        source_dir=smdataparallel_dir,
        instance_count=2,
        instance_type="ml.p3.16xlarge",
        sagemaker_session=sagemaker_session,
        distribution={"smdistributed": {"dataparallel": {"enabled": True}}},
    )

    with timeout.timeout(minutes=integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit(job_name=job_name)
