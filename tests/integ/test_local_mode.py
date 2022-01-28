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
import tarfile

import boto3
import numpy
import pytest
import tempfile

import stopit

import tests.integ.lock as lock
from tests.integ import DATA_DIR

from sagemaker import image_uris

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor

from sagemaker.local import LocalSession, LocalSagemakerRuntimeClient, LocalSagemakerClient
from sagemaker.mxnet import MXNet

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

    def _initialize(self, boto_session, sagemaker_client, sagemaker_runtime_client, **kwargs):
        self.boto_session = boto3.Session(region_name=DEFAULT_REGION)
        if self.config is None:
            self.config = {"local": {"local_code": True, "region_name": DEFAULT_REGION}}

        self._region_name = DEFAULT_REGION
        self.sagemaker_client = LocalSagemakerClient(self)
        self.sagemaker_runtime_client = LocalSagemakerRuntimeClient(self.config)
        self.local_mode = True


@pytest.fixture(scope="module")
def sagemaker_local_session_no_local_code(boto_session):
    return LocalSession(boto_session=boto_session, disable_local_code=True)


@pytest.fixture(scope="module")
def sklearn_image_uri(
    sklearn_latest_version,
    sklearn_latest_py_version,
    cpu_instance_type,
    sagemaker_session,
):
    return image_uris.retrieve(
        "sklearn",
        sagemaker_session.boto_region_name,
        version=sklearn_latest_version,
        py_version=sklearn_latest_py_version,
        instance_type=cpu_instance_type,
    )


@pytest.fixture(scope="module")
def mxnet_model(
    sagemaker_local_session, mxnet_inference_latest_version, mxnet_inference_latest_py_version
):
    def _create_model(output_path):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            instance_count=1,
            instance_type="local",
            output_path=output_path,
            framework_version=mxnet_inference_latest_version,
            py_version=mxnet_inference_latest_py_version,
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
def test_local_mode_serving_from_s3_model(
    sagemaker_local_session,
    mxnet_model,
    mxnet_inference_latest_version,
    mxnet_inference_latest_py_version,
):
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
def test_mxnet_local_mode(
    sagemaker_local_session, mxnet_training_latest_version, mxnet_training_latest_py_version
):
    script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        py_version=mxnet_training_latest_py_version,
        instance_count=1,
        instance_type="local",
        sagemaker_session=sagemaker_local_session,
        framework_version=mxnet_training_latest_version,
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
            predictor.delete_endpoint()


@pytest.mark.local_mode
def test_mxnet_distributed_local_mode(
    sagemaker_local_session, mxnet_training_latest_version, mxnet_training_latest_py_version
):
    script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        py_version=mxnet_training_latest_py_version,
        instance_count=2,
        instance_type="local",
        sagemaker_session=sagemaker_local_session,
        framework_version=mxnet_training_latest_version,
        distribution={"parameter_server": {"enabled": True}},
    )

    train_input = mx.sagemaker_session.upload_data(
        path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
    )
    test_input = mx.sagemaker_session.upload_data(
        path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
    )

    mx.fit({"train": train_input, "test": test_input})


@pytest.mark.local_mode
def test_mxnet_local_data_local_script(
    mxnet_training_latest_version, mxnet_training_latest_py_version
):
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")
    script_path = os.path.join(data_path, "mnist.py")

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        instance_count=1,
        instance_type="local",
        framework_version=mxnet_training_latest_version,
        py_version=mxnet_training_latest_py_version,
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
            predictor.delete_endpoint()


@pytest.mark.local_mode
def test_mxnet_training_failure(
    sagemaker_local_session, mxnet_training_latest_version, mxnet_training_latest_py_version, tmpdir
):
    script_path = os.path.join(DATA_DIR, "mxnet_mnist", "failure_script.py")

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        framework_version=mxnet_training_latest_version,
        py_version=mxnet_training_latest_py_version,
        instance_count=1,
        instance_type="local",
        sagemaker_session=sagemaker_local_session,
        code_location="s3://{}".format(sagemaker_local_session.default_bucket()),
        output_path="file://{}".format(tmpdir),
    )

    with pytest.raises(RuntimeError):
        mx.fit()

    with tarfile.open(os.path.join(str(tmpdir), "output.tar.gz")) as tar:
        tar.getmember("failure")


@pytest.mark.local_mode
def test_local_transform_mxnet(
    sagemaker_local_session,
    tmpdir,
    mxnet_inference_latest_version,
    mxnet_inference_latest_py_version,
    cpu_instance_type,
):
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")
    script_path = os.path.join(data_path, "mnist.py")

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        instance_count=1,
        instance_type="local",
        framework_version=mxnet_inference_latest_version,
        py_version=mxnet_inference_latest_py_version,
        sagemaker_session=sagemaker_local_session,
    )

    train_input = mx.sagemaker_session.upload_data(
        path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
    )
    test_input = mx.sagemaker_session.upload_data(
        path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
    )

    with stopit.ThreadingTimeout(5 * 60, swallow_exc=False):
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


@pytest.mark.local_mode
def test_local_processing_sklearn(sagemaker_local_session_no_local_code, sklearn_latest_version):
    script_path = os.path.join(DATA_DIR, "dummy_script.py")
    input_file_path = os.path.join(DATA_DIR, "dummy_input.txt")

    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_latest_version,
        role="SageMakerRole",
        instance_type="local",
        instance_count=1,
        command=["python3"],
        sagemaker_session=sagemaker_local_session_no_local_code,
    )

    sklearn_processor.run(
        code=script_path,
        inputs=[ProcessingInput(source=input_file_path, destination="/opt/ml/processing/inputs/")],
        wait=False,
        logs=False,
    )

    job_description = sklearn_processor.latest_job.describe()

    assert len(job_description["ProcessingInputs"]) == 2
    assert job_description["ProcessingResources"]["ClusterConfig"]["InstanceCount"] == 1
    assert job_description["ProcessingResources"]["ClusterConfig"]["InstanceType"] == "local"
    assert job_description["AppSpecification"]["ContainerEntrypoint"] == [
        "python3",
        "/opt/ml/processing/input/code/dummy_script.py",
    ]
    assert job_description["RoleArn"] == "<no_role>"


@pytest.mark.local_mode
def test_local_processing_script_processor(sagemaker_local_session, sklearn_image_uri):
    input_file_path = os.path.join(DATA_DIR, "dummy_input.txt")

    script_processor = ScriptProcessor(
        role="SageMakerRole",
        image_uri=sklearn_image_uri,
        command=["python3"],
        instance_count=1,
        instance_type="local",
        volume_size_in_gb=30,
        volume_kms_key=None,
        max_runtime_in_seconds=3600,
        base_job_name="test-script-processor",
        env={"DUMMY_ENVIRONMENT_VARIABLE": "dummy-value"},
        tags=[{"Key": "dummy-tag", "Value": "dummy-tag-value"}],
        sagemaker_session=sagemaker_local_session,
    )

    script_processor.run(
        code=os.path.join(DATA_DIR, "dummy_script.py"),
        inputs=[
            ProcessingInput(
                source=input_file_path,
                destination="/opt/ml/processing/input/container/path/",
                input_name="dummy_input",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/container/path/",
                output_name="dummy_output",
                s3_upload_mode="EndOfJob",
            )
        ],
        arguments=["-v"],
        wait=True,
        logs=True,
    )

    job_description = script_processor.latest_job.describe()

    assert job_description["ProcessingInputs"][0]["InputName"] == "dummy_input"

    assert job_description["ProcessingInputs"][1]["InputName"] == "code"

    assert job_description["ProcessingJobName"].startswith("test-script-processor")

    assert job_description["ProcessingJobStatus"] == "Completed"

    assert job_description["ProcessingOutputConfig"]["Outputs"][0]["OutputName"] == "dummy_output"

    assert job_description["ProcessingResources"]["ClusterConfig"]["InstanceCount"] == 1
    assert job_description["ProcessingResources"]["ClusterConfig"]["InstanceType"] == "local"
    assert job_description["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"] == 30

    assert job_description["AppSpecification"]["ContainerArguments"] == ["-v"]
    assert job_description["AppSpecification"]["ContainerEntrypoint"] == [
        "python3",
        "/opt/ml/processing/input/code/dummy_script.py",
    ]
    assert job_description["AppSpecification"]["ImageUri"] == sklearn_image_uri

    assert job_description["Environment"] == {"DUMMY_ENVIRONMENT_VARIABLE": "dummy-value"}
