# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import numpy
import os
import pytest
from sagemaker.mxnet.estimator import MXNet
from sagemaker.mxnet.model import MXNetModel
from sagemaker.utils import sagemaker_timestamp
from tests.integ import DATA_DIR, PYTHON_VERSION, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name
import time


@pytest.fixture(scope="module")
def mxnet_training_job(sagemaker_session, mxnet_full_version):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_neo.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_full_version,
            py_version=PYTHON_VERSION,
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
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


@pytest.mark.canary_quick
@pytest.mark.regional_testing
@pytest.mark.skip(
    reason="This should be enabled along with the Boto SDK release for Neo API changes"
)
def test_attach_deploy(mxnet_training_job, sagemaker_session):
    endpoint_name = "test-mxnet-attach-deploy-{}".format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = MXNet.attach(mxnet_training_job, sagemaker_session=sagemaker_session)

        estimator.compile_model(
            target_instance_family="ml_m4",
            input_shape={"data": [1, 1, 28, 28]},
            output_path=estimator.output_path,
        )

        predictor = estimator.deploy(
            1, "ml.m4.xlarge", use_compiled_model=True, endpoint_name=endpoint_name
        )
        predictor.content_type = "application/vnd+python.numpy+binary"
        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)


@pytest.mark.skip(
    reason="This should be enabled along with the Boto SDK release for Neo API changes"
)
def test_deploy_model(mxnet_training_job, sagemaker_session):
    endpoint_name = "test-mxnet-deploy-model-{}".format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        desc = sagemaker_session.sagemaker_client.describe_training_job(
            TrainingJobName=mxnet_training_job
        )
        model_data = desc["ModelArtifacts"]["S3ModelArtifacts"]
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_neo.py")
        role = "SageMakerRole"
        model = MXNetModel(
            model_data,
            role,
            entry_point=script_path,
            py_version=PYTHON_VERSION,
            sagemaker_session=sagemaker_session,
        )

        model.compile(
            target_instance_family="ml_m4",
            input_shape={"data": [1, 1, 28, 28]},
            role=role,
            job_name="test-deploy-model-compilation-job-{}".format(int(time.time())),
            output_path="/".join(model_data.split("/")[:-1]),
        )
        predictor = model.deploy(1, "ml.m4.xlarge", endpoint_name=endpoint_name)

        predictor.content_type = "application/vnd+python.numpy+binary"
        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)
