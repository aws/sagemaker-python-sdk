# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import tarfile

import boto3
import numpy
import pytest
import tempfile

import tests.integ.lock as lock
from tests.integ import DATA_DIR, PYTHON_VERSION
from tests.integ.timeout import timeout

from sagemaker.local import LocalSession, LocalSagemakerRuntimeClient, LocalSagemakerClient
from sagemaker.mxnet import MXNet
from sagemaker.tensorflow import TensorFlow

# endpoint tests all use the same port, so we use this lock to prevent concurrent execution
LOCK_PATH = os.path.join(tempfile.gettempdir(), "sagemaker_test_local_mode_lock")
DATA_PATH = os.path.join(DATA_DIR, "iris", "data")
DEFAULT_REGION = "us-west-2"


class LocalNoS3Session(LocalSession):
    """
    This Session sets  local_code: True regardless of any config file settings
    """

    def __init__(self):
        super(LocalSession, self).__init__()

    def _initialize(self, boto_session, sagemaker_client, sagemaker_runtime_client):
        self.boto_session = boto3.Session(region_name=DEFAULT_REGION)
        if self.config is None:
            self.config = {"local": {"local_code": True, "region_name": DEFAULT_REGION}}

        self._region_name = DEFAULT_REGION
        self.sagemaker_client = LocalSagemakerClient(self)
        self.sagemaker_runtime_client = LocalSagemakerRuntimeClient(self.config)
        self.local_mode = True


@pytest.fixture(scope="module")
def mxnet_model(sagemaker_local_session, mxnet_full_version):
    def _create_model(output_path):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            train_instance_count=1,
            train_instance_type="local",
            output_path=output_path,
            framework_version=mxnet_full_version,
            sagemaker_session=sagemaker_local_session,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input})
        model = mx.create_model(1)
        return model

    return _create_model


@pytest.mark.local_mode
@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def test_tf_local_mode(sagemaker_local_session):
    with timeout(minutes=5):
        script_path = os.path.join(DATA_DIR, "iris", "iris-dnn-classifier.py")

        estimator = TensorFlow(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version="1.12",
            training_steps=1,
            evaluation_steps=1,
            hyperparameters={"input_tensor_name": "inputs"},
            train_instance_count=1,
            train_instance_type="local",
            base_job_name="test-tf",
            sagemaker_session=sagemaker_local_session,
        )

        inputs = estimator.sagemaker_session.upload_data(
            path=DATA_PATH, key_prefix="integ-test-data/tf_iris"
        )
        estimator.fit(inputs)
        print("job succeeded: {}".format(estimator.latest_training_job.name))

    endpoint_name = estimator.latest_training_job.name
    with lock.lock(LOCK_PATH):
        try:
            json_predictor = estimator.deploy(
                initial_instance_count=1, instance_type="local", endpoint_name=endpoint_name
            )

            features = [6.4, 3.2, 4.5, 1.5]
            dict_result = json_predictor.predict({"inputs": features})
            print("predict result: {}".format(dict_result))
            list_result = json_predictor.predict(features)
            print("predict result: {}".format(list_result))

            assert dict_result == list_result
        finally:
            estimator.delete_endpoint()


@pytest.mark.local_mode
@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def test_tf_distributed_local_mode(sagemaker_local_session):
    with timeout(minutes=5):
        script_path = os.path.join(DATA_DIR, "iris", "iris-dnn-classifier.py")

        estimator = TensorFlow(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version="1.12",
            training_steps=1,
            evaluation_steps=1,
            hyperparameters={"input_tensor_name": "inputs"},
            train_instance_count=3,
            train_instance_type="local",
            base_job_name="test-tf",
            sagemaker_session=sagemaker_local_session,
        )

        inputs = "file://" + DATA_PATH
        estimator.fit(inputs)
        print("job succeeded: {}".format(estimator.latest_training_job.name))

    endpoint_name = estimator.latest_training_job.name

    with lock.lock(LOCK_PATH):
        try:
            json_predictor = estimator.deploy(
                initial_instance_count=1, instance_type="local", endpoint_name=endpoint_name
            )

            features = [6.4, 3.2, 4.5, 1.5]
            dict_result = json_predictor.predict({"inputs": features})
            print("predict result: {}".format(dict_result))
            list_result = json_predictor.predict(features)
            print("predict result: {}".format(list_result))

            assert dict_result == list_result
        finally:
            estimator.delete_endpoint()


@pytest.mark.local_mode
@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def test_tf_local_data(sagemaker_local_session):
    with timeout(minutes=5):
        script_path = os.path.join(DATA_DIR, "iris", "iris-dnn-classifier.py")

        estimator = TensorFlow(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version="1.12",
            training_steps=1,
            evaluation_steps=1,
            hyperparameters={"input_tensor_name": "inputs"},
            train_instance_count=1,
            train_instance_type="local",
            base_job_name="test-tf",
            sagemaker_session=sagemaker_local_session,
        )

        inputs = "file://" + DATA_PATH
        estimator.fit(inputs)
        print("job succeeded: {}".format(estimator.latest_training_job.name))

    endpoint_name = estimator.latest_training_job.name
    with lock.lock(LOCK_PATH):
        try:
            json_predictor = estimator.deploy(
                initial_instance_count=1, instance_type="local", endpoint_name=endpoint_name
            )

            features = [6.4, 3.2, 4.5, 1.5]
            dict_result = json_predictor.predict({"inputs": features})
            print("predict result: {}".format(dict_result))
            list_result = json_predictor.predict(features)
            print("predict result: {}".format(list_result))

            assert dict_result == list_result
        finally:
            estimator.delete_endpoint()


@pytest.mark.local_mode
@pytest.mark.skipif(PYTHON_VERSION != "py2", reason="TensorFlow image supports only python 2.")
def test_tf_local_data_local_script():
    with timeout(minutes=5):
        script_path = os.path.join(DATA_DIR, "iris", "iris-dnn-classifier.py")

        estimator = TensorFlow(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version="1.12",
            training_steps=1,
            evaluation_steps=1,
            hyperparameters={"input_tensor_name": "inputs"},
            train_instance_count=1,
            train_instance_type="local",
            base_job_name="test-tf",
            sagemaker_session=LocalNoS3Session(),
        )

        inputs = "file://" + DATA_PATH

        estimator.fit(inputs)
        print("job succeeded: {}".format(estimator.latest_training_job.name))

    endpoint_name = estimator.latest_training_job.name
    with lock.lock(LOCK_PATH):
        try:
            json_predictor = estimator.deploy(
                initial_instance_count=1, instance_type="local", endpoint_name=endpoint_name
            )

            features = [6.4, 3.2, 4.5, 1.5]
            dict_result = json_predictor.predict({"inputs": features})
            print("predict result: {}".format(dict_result))
            list_result = json_predictor.predict(features)
            print("predict result: {}".format(list_result))

            assert dict_result == list_result
        finally:
            estimator.delete_endpoint()


@pytest.mark.local_mode
def test_local_mode_serving_from_s3_model(sagemaker_local_session, mxnet_model, mxnet_full_version):
    path = "s3://%s" % sagemaker_local_session.default_bucket()
    s3_model = mxnet_model(path)
    s3_model.sagemaker_session = sagemaker_local_session

    predictor = None
    with lock.lock(LOCK_PATH):
        try:
            predictor = s3_model.deploy(initial_instance_count=1, instance_type="local")
            data = numpy.zeros(shape=(1, 1, 28, 28))
            predictor.predict(data)
        finally:
            if predictor:
                predictor.delete_endpoint()


@pytest.mark.local_mode
def test_local_mode_serving_from_local_model(tmpdir, sagemaker_local_session, mxnet_model):
    predictor = None

    with lock.lock(LOCK_PATH):
        try:
            path = "file://%s" % (str(tmpdir))
            model = mxnet_model(path)
            model.sagemaker_session = sagemaker_local_session
            predictor = model.deploy(initial_instance_count=1, instance_type="local")
            data = numpy.zeros(shape=(1, 1, 28, 28))
            predictor.predict(data)
        finally:
            if predictor:
                predictor.delete_endpoint()


@pytest.mark.local_mode
def test_mxnet_local_mode(sagemaker_local_session, mxnet_full_version):
    script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        py_version=PYTHON_VERSION,
        train_instance_count=1,
        train_instance_type="local",
        sagemaker_session=sagemaker_local_session,
        framework_version=mxnet_full_version,
    )

    train_input = mx.sagemaker_session.upload_data(
        path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
    )
    test_input = mx.sagemaker_session.upload_data(
        path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
    )

    mx.fit({"train": train_input, "test": test_input})
    endpoint_name = mx.latest_training_job.name

    with lock.lock(LOCK_PATH):
        try:
            predictor = mx.deploy(1, "local", endpoint_name=endpoint_name)
            data = numpy.zeros(shape=(1, 1, 28, 28))
            predictor.predict(data)
        finally:
            mx.delete_endpoint()


@pytest.mark.local_mode
def test_mxnet_local_data_local_script(mxnet_full_version):
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")
    script_path = os.path.join(data_path, "mnist.py")

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        train_instance_count=1,
        train_instance_type="local",
        framework_version=mxnet_full_version,
        sagemaker_session=LocalNoS3Session(),
    )

    train_input = "file://" + os.path.join(data_path, "train")
    test_input = "file://" + os.path.join(data_path, "test")

    mx.fit({"train": train_input, "test": test_input})
    endpoint_name = mx.latest_training_job.name

    with lock.lock(LOCK_PATH):
        try:
            predictor = mx.deploy(1, "local", endpoint_name=endpoint_name)
            data = numpy.zeros(shape=(1, 1, 28, 28))
            predictor.predict(data)
        finally:
            mx.delete_endpoint()


@pytest.mark.local_mode
def test_mxnet_training_failure(sagemaker_local_session, mxnet_full_version, tmpdir):
    script_path = os.path.join(DATA_DIR, "mxnet_mnist", "failure_script.py")

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        framework_version=mxnet_full_version,
        py_version=PYTHON_VERSION,
        train_instance_count=1,
        train_instance_type="local",
        sagemaker_session=sagemaker_local_session,
        output_path="file://{}".format(tmpdir),
    )

    with pytest.raises(RuntimeError):
        mx.fit()

    with tarfile.open(os.path.join(str(tmpdir), "output.tar.gz")) as tar:
        tar.getmember("failure")


@pytest.mark.local_mode
def test_local_transform_mxnet(sagemaker_local_session, tmpdir, mxnet_full_version):
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")
    script_path = os.path.join(data_path, "mnist.py")

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        train_instance_count=1,
        train_instance_type="ml.c4.xlarge",
        framework_version=mxnet_full_version,
        sagemaker_session=sagemaker_local_session,
    )

    train_input = mx.sagemaker_session.upload_data(
        path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
    )
    test_input = mx.sagemaker_session.upload_data(
        path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
    )

    with timeout(minutes=15):
        mx.fit({"train": train_input, "test": test_input})

    transform_input_path = os.path.join(data_path, "transform")
    transform_input_key_prefix = "integ-test-data/mxnet_mnist/transform"
    transform_input = mx.sagemaker_session.upload_data(
        path=transform_input_path, key_prefix=transform_input_key_prefix
    )

    output_path = "file://%s" % (str(tmpdir))
    transformer = mx.transformer(
        1,
        "local",
        assemble_with="Line",
        max_payload=1,
        strategy="SingleRecord",
        output_path=output_path,
    )

    with lock.lock(LOCK_PATH):
        transformer.transform(transform_input, content_type="text/csv", split_type="Line")
        transformer.wait()

    assert os.path.exists(os.path.join(str(tmpdir), "data.csv.out"))
