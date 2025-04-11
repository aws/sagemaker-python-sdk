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

from sagemaker.mxnet.estimator import MXNet
from tests.integ import (
    DATA_DIR,
    TRAINING_DEFAULT_TIMEOUT_MINUTES,
)
from tests.integ.timeout import timeout


@pytest.fixture(scope="module")
def mxnet_training_job(
    sagemaker_session,
    cpu_instance_type,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_neo.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_training_latest_version,
            py_version=mxnet_training_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input})
        return mx.latest_training_job.name


@pytest.mark.skip(
    reason="Edge has been deprecated. Skipping until feature team deprecates functionality."
)
def test_edge_packaging_job(mxnet_training_job, sagemaker_session):
    estimator = MXNet.attach(mxnet_training_job, sagemaker_session=sagemaker_session)
    model = estimator.compile_model(
        target_instance_family="rasp3b",
        input_shape={"data": [1, 1, 28, 28], "softmax_label": [1]},
        output_path=estimator.output_path,
    )

    model.package_for_edge(
        output_path=estimator.output_path,
        role=estimator.role,
        model_name="sdk-test-model",
        model_version="1.0",
    )
