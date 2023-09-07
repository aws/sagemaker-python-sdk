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

torch_distributed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "torch_distributed")


@pytest.mark.skip(
    reason="Disabling until the launch of SM Trainium containers"
    "This test should be re-enabled later."
)
def test_torch_distributed_trn1_pt_mnist(
    sagemaker_session,
    torch_distributed_framework_version,
    torch_distributed_py_version,
):
    job_name = sagemaker.utils.unique_name_from_base("pt-torch-distributed")
    estimator = PyTorch(
        entry_point="mnist_mlp_neuron.py",
        role="SageMakerRole",
        source_dir=torch_distributed_dir,
        instance_count=1,
        instance_type="ml.trn1.2xlarge",
        sagemaker_session=sagemaker_session,
        framework_version=torch_distributed_framework_version,
        py_version=torch_distributed_py_version,
        distribution={"torch_distributed": {"enabled": True}},
    )

    with timeout.timeout(minutes=integ.TRAINING_DEFAULT_TIMEOUT_MINUTES):
        estimator.fit({"training": _upload_training_data(estimator)}, job_name=job_name)
