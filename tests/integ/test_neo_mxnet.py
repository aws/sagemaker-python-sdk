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

import numpy
import pytest

from sagemaker.mxnet.estimator import MXNet
from sagemaker.mxnet.model import MXNetModel
from sagemaker.serializers import JSONSerializer
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name


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


@pytest.mark.release
@pytest.mark.skip(
    reason="This test is failing because the image uri and the training script format has changed."
)
def test_attach_deploy(
    mxnet_training_job, sagemaker_session, cpu_instance_type, cpu_instance_family
):
    endpoint_name = unique_name_from_base("test-neo-attach-deploy")

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        estimator = MXNet.attach(mxnet_training_job, sagemaker_session=sagemaker_session)

        estimator.compile_model(
            target_instance_family=cpu_instance_family,
            input_shape={"data": [1, 1, 28, 28]},
            output_path=estimator.output_path,
        )

        serializer = JSONSerializer(content_type="application/vnd+python.numpy+binary")

        predictor = estimator.deploy(
            1,
            cpu_instance_type,
            serializer=serializer,
            use_compiled_model=True,
            endpoint_name=endpoint_name,
        )
        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)


@pytest.mark.skip(
    reason="This test is failing because the image uri and the training script format has changed."
)
def test_deploy_model(
    mxnet_training_job,
    sagemaker_session,
    cpu_instance_type,
    cpu_instance_family,
    neo_mxnet_latest_version,
    neo_mxnet_latest_py_version,
):
    endpoint_name = unique_name_from_base("test-neo-deploy-model")

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
            py_version=neo_mxnet_latest_py_version,
            framework_version=neo_mxnet_latest_version,
            sagemaker_session=sagemaker_session,
        )

        serializer = JSONSerializer(content_type="application/vnd+python.numpy+binary")

        model.compile(
            target_instance_family=cpu_instance_family,
            input_shape={"data": [1, 1, 28, 28]},
            role=role,
            job_name=unique_name_from_base("test-deploy-model-compilation-job"),
            output_path="/".join(model_data.split("/")[:-1]),
        )
        predictor = model.deploy(
            1, cpu_instance_type, serializer=serializer, endpoint_name=endpoint_name
        )

        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)


@pytest.mark.skip(reason="Inferentia is not supported yet.")
def test_inferentia_deploy_model(
    mxnet_training_job,
    sagemaker_session,
    inf_instance_type,
    inf_instance_family,
    inferentia_mxnet_latest_version,
    inferentia_mxnet_latest_py_version,
):
    endpoint_name = unique_name_from_base("test-neo-deploy-model")

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
            framework_version=inferentia_mxnet_latest_version,
            py_version=inferentia_mxnet_latest_py_version,
            sagemaker_session=sagemaker_session,
        )

        model.compile(
            target_instance_family=inf_instance_family,
            input_shape={"data": [1, 1, 28, 28]},
            role=role,
            job_name=unique_name_from_base("test-deploy-model-compilation-job"),
            output_path="/".join(model_data.split("/")[:-1]),
        )

        serializer = JSONSerializer(content_type="application/vnd+python.numpy+binary")

        predictor = model.deploy(
            1, inf_instance_type, serializer=serializer, endpoint_name=endpoint_name
        )

        data = numpy.zeros(shape=(1, 1, 28, 28))
        predictor.predict(data)
