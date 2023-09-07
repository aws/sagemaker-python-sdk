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
import pytest
import sagemaker.utils
import tests.integ as integ
from sagemaker.pytorch import PyTorch
from tests.integ import timeout
from tests.integ.test_pytorch import _upload_training_data

pytorchddp_dir = os.path.join(os.path.dirname(__file__), "..", "data", "pytorch_ddp")


@pytest.mark.skip(
    reason="This test is skipped for now due ML capacity error."
    "This test should be re-enabled later."
)
@pytest.mark.skipif(
    integ.test_region() not in integ.DATA_PARALLEL_TESTING_REGIONS,
    reason="Only allow this test to run in IAD and CMH to limit usage of p3.16xlarge",
)
def test_pytorchddp_pt_mnist(
    sagemaker_session,
    pytorch_ddp_framework_version,
    pytorch_ddp_py_version,
):
    job_name = sagemaker.utils.unique_name_from_base("pt-pytorch-ddp")
    estimator = PyTorch(
        entry_point="mnist_pt.py",
        role="SageMakerRole",
        source_dir=pytorchddp_dir,
        instance_count=2,
        instance_type="ml.p3.16xlarge",
        sagemaker_session=sagemaker_session,
        framework_version=pytorch_ddp_framework_version,
        py_version=pytorch_ddp_py_version,
        distribution={"pytorchddp": {"enabled": True}},
    )

    with timeout.timeout(minutes=integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit({"training": _upload_training_data(estimator)}, job_name=job_name)
