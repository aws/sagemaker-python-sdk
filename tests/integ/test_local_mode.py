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
from sagemaker.config import SESSION_S3_BUCKET_PATH
from sagemaker.utils import resolve_value_from_config
from tests.integ import DATA_DIR
from mock import Mock, ANY

from sagemaker import image_uris

from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep, TransformStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.functions import JsonGet, PropertyFile, Join
from sagemaker.workflow.pipeline_context import LocalPipelineSession
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

        self.sagemaker_config = kwargs.get("sagemaker_config", None)

        # after sagemaker_config initialization, update self._default_bucket_name_override if needed
        self._default_bucket_name_override = resolve_value_from_config(
            direct_input=self._default_bucket_name_override,
            config_path=SESSION_S3_BUCKET_PATH,
            sagemaker_session=self,
        )


class LocalPipelineNoS3Session(LocalPipelineSession):
    """
    This Session sets  local_code: True regardless of any config file settings
    """

    def __init__(self):
        super(LocalPipelineSession, self).__init__()

    def _initialize(self, boto_session, sagemaker_client, sagemaker_runtime_client, **kwargs):
        self.boto_session = boto3.Session(region_name=DEFAULT_REGION)
        if self.config is None:
            self.config = {"local": {"local_code": True, "region_name": DEFAULT_REGION}}

        self._region_name = DEFAULT_REGION
        self.sagemaker_client = LocalSagemakerClient(self)
        self.sagemaker_runtime_client = LocalSagemakerRuntimeClient(self.config)
        self.local_mode = True

        self.sagemaker_config = kwargs.get("sagemaker_config", None)

        # after sagemaker_config initialization, update self._default_bucket_name_override if needed
        self._default_bucket_name_override = resolve_value_from_config(
            direct_input=self._default_bucket_name_override,
            config_path=SESSION_S3_BUCKET_PATH,
            sagemaker_session=self,
        )


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
    local_no_s3_session = LocalNoS3Session()
    local_no_s3_session.boto_session.resource = Mock(
        side_effect=local_no_s3_session.boto_session.resource
    )
    local_no_s3_session.boto_session.client = Mock(
        side_effect=local_no_s3_session.boto_session.client
    )

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        instance_count=1,
        instance_type="local",
        framework_version=mxnet_training_latest_version,
        py_version=mxnet_training_latest_py_version,
        sagemaker_session=local_no_s3_session,
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
            # check if no boto_session s3 calls were made
            with pytest.raises(AssertionError):
                local_no_s3_session.boto_session.resource.assert_called_with("s3", region_name=ANY)
            with pytest.raises(AssertionError):
                local_no_s3_session.boto_session.client.assert_called_with("s3", region_name=ANY)
        finally:
            predictor.delete_endpoint()


@pytest.mark.local_mode
def test_mxnet_local_training_env(mxnet_training_latest_version, mxnet_training_latest_py_version):
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")
    script_path = os.path.join(data_path, "check_env.py")

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        instance_count=1,
        instance_type="local",
        framework_version=mxnet_training_latest_version,
        py_version=mxnet_training_latest_py_version,
        sagemaker_session=LocalNoS3Session(),
        environment={"MYVAR": "HELLO_WORLD"},
    )

    train_input = "file://" + os.path.join(data_path, "train")
    test_input = "file://" + os.path.join(data_path, "test")

    mx.fit({"train": train_input, "test": test_input})


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
    script_path = os.path.join(data_path, "check_env.py")

    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        instance_count=1,
        instance_type="local",
        framework_version=mxnet_inference_latest_version,
        py_version=mxnet_inference_latest_py_version,
        sagemaker_session=sagemaker_local_session,
        environment={"MYVAR": "HELLO_WORLD"},
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


@pytest.mark.local_mode
def test_local_pipeline_with_processing_step(sklearn_latest_version, local_pipeline_session):
    string_container_arg = ParameterString(name="ProcessingContainerArg", default_value="foo")
    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_latest_version,
        role="SageMakerRole",
        instance_type="local",
        instance_count=1,
        command=["python3"],
        sagemaker_session=local_pipeline_session,
    )
    script_path = os.path.join(DATA_DIR, "dummy_script.py")
    input_file_path = os.path.join(DATA_DIR, "dummy_input.txt")
    processing_args = sklearn_processor.run(
        code=script_path,
        inputs=[ProcessingInput(source=input_file_path, destination="/opt/ml/processing/inputs/")],
        arguments=["--container_arg", string_container_arg],
    )
    processing_step = ProcessingStep(
        name="sklearn_processor_local_pipeline", step_args=processing_args
    )
    pipeline = Pipeline(
        name="local_pipeline_processing",
        steps=[processing_step],
        sagemaker_session=local_pipeline_session,
        parameters=[string_container_arg],
    )
    pipeline.create("SageMakerRole", "pipeline for sdk integ testing")

    with lock.lock(LOCK_PATH):
        execution = pipeline.start()

    pipeline_execution_describe_result = execution.describe()
    assert pipeline_execution_describe_result["PipelineArn"] == "local_pipeline_processing"
    assert pipeline_execution_describe_result["PipelineExecutionStatus"] == "Succeeded"

    pipeline_execution_list_steps_result = execution.list_steps()
    assert len(pipeline_execution_list_steps_result["PipelineExecutionSteps"]) == 1
    assert (
        pipeline_execution_list_steps_result["PipelineExecutionSteps"][0]["StepName"]
        == "sklearn_processor_local_pipeline"
    )
    assert (
        pipeline_execution_list_steps_result["PipelineExecutionSteps"][0]["StepStatus"]
        == "Succeeded"
    )


@pytest.mark.local_mode
def test_local_pipeline_with_training_and_transform_steps(
    mxnet_training_latest_version,
    mxnet_inference_latest_version,
    mxnet_training_latest_py_version,
    tmpdir,
):
    session = LocalPipelineNoS3Session()
    instance_count = ParameterInteger(name="InstanceCountParam")
    data_path = os.path.join(DATA_DIR, "mxnet_mnist")
    script_path = os.path.join(data_path, "check_env.py")
    output_path = "file://%s" % (str(tmpdir))

    # define Estimator
    mx = MXNet(
        entry_point=script_path,
        role="SageMakerRole",
        instance_count=instance_count,
        instance_type="local",
        framework_version=mxnet_training_latest_version,
        py_version=mxnet_training_latest_py_version,
        sagemaker_session=session,
        output_path=output_path,
        environment={"MYVAR": "HELLO_WORLD"},
    )

    # define training step
    train_input = "file://" + os.path.join(data_path, "train")
    test_input = "file://" + os.path.join(data_path, "test")
    training_args = mx.fit({"train": train_input, "test": test_input})
    training_step = TrainingStep(name="mxnet_mnist_training", step_args=training_args)

    # define model
    inference_image_uri = image_uris.retrieve(
        framework="mxnet",
        region=DEFAULT_REGION,
        version=mxnet_inference_latest_version,
        instance_type="local",
        image_scope="inference",
    )
    model = Model(
        image_uri=inference_image_uri,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=session,
        role="SageMakerRole",
    )

    # define create model step
    model_step_args = model.create(instance_type="local", accelerator_type="local")
    model_step = ModelStep(name="mxnet_mnist_model", step_args=model_step_args)

    # define transformer
    transformer = Transformer(
        model_name=model_step.properties.ModelName,
        instance_type="local",
        instance_count=instance_count,
        output_path=output_path,
        assemble_with="Line",
        max_payload=1,
        strategy="SingleRecord",
        sagemaker_session=session,
    )

    # define transform step
    transform_input = "file://" + os.path.join(data_path, "transform")
    transform_args = transformer.transform(
        transform_input, content_type="text/csv", split_type="Line"
    )
    transform_step = TransformStep(name="mxnet_mnist_transform", step_args=transform_args)

    pipeline = Pipeline(
        name="local_pipeline_training_transform",
        parameters=[instance_count],
        steps=[training_step, model_step, transform_step],
        sagemaker_session=session,
    )

    pipeline.create("SageMakerRole", "pipeline for sdk integ testing")

    with lock.lock(LOCK_PATH):
        execution = pipeline.start(parameters={"InstanceCountParam": 1})

    assert os.path.exists(os.path.join(str(tmpdir), "model.tar.gz"))
    assert os.path.exists(os.path.join(str(tmpdir), "data.csv.out"))

    pipeline_execution_describe_result = execution.describe()
    assert pipeline_execution_describe_result["PipelineArn"] == "local_pipeline_training_transform"
    assert pipeline_execution_describe_result["PipelineExecutionStatus"] == "Succeeded"

    pipeline_execution_list_steps_result = execution.list_steps()
    assert len(pipeline_execution_list_steps_result["PipelineExecutionSteps"]) == 3


@pytest.mark.local_mode
def test_local_pipeline_with_eval_cond_fail_steps(sklearn_image_uri, local_pipeline_session):
    processor = ScriptProcessor(
        image_uri=sklearn_image_uri,
        role="SageMakerRole",
        instance_count=1,
        instance_type="local",
        sagemaker_session=local_pipeline_session,
        command=["python3"],
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    base_dir = os.path.join(DATA_DIR, "mxnet_mnist")
    mx_mnist_model_data = os.path.join(base_dir, "model.tar.gz")
    test_input = os.path.join(base_dir, "test")

    eval_step = ProcessingStep(
        name="mxnet_mnist_eval",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=mx_mnist_model_data,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=test_input,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(base_dir, "code/evaluation.py"),
        property_files=[evaluation_report],
    )

    f1_score = JsonGet(
        step_name=eval_step.name,
        property_file=evaluation_report,
        json_path="metrics.f1.value",
    )

    fail_step = FailStep(
        name="mxnet_mnist_fail", error_message=Join(on=":", values=["F1 score too low", f1_score])
    )

    cond_lte = ConditionLessThanOrEqualTo(
        left=f1_score,
        right=0.8,
    )
    cond_step = ConditionStep(
        name="mxnet_mnist_condition",
        conditions=[cond_lte],
        if_steps=[fail_step],
        else_steps=[],
    )

    pipeline = Pipeline(
        name="local_pipeline_training_transform",
        steps=[eval_step, cond_step],
        sagemaker_session=local_pipeline_session,
    )

    pipeline.create("SageMakerRole", "pipeline for sdk integ testing")

    with lock.lock(LOCK_PATH):
        execution = pipeline.start()

    pipeline_execution_describe_result = execution.describe()
    assert pipeline_execution_describe_result["PipelineArn"] == "local_pipeline_training_transform"
    assert pipeline_execution_describe_result["PipelineExecutionStatus"] == "Failed"

    pipeline_execution_list_steps_result = execution.list_steps()
    assert len(pipeline_execution_list_steps_result["PipelineExecutionSteps"]) == 3
    for step in pipeline_execution_list_steps_result["PipelineExecutionSteps"]:
        if step["StepName"] == "mxnet_mnist_eval":
            assert step["StepStatus"] == "Succeeded"
        elif step["StepName"] == "mxnet_mnist_condition":
            assert step["StepStatus"] == "Succeeded"
            assert step["Metadata"]["Condition"]["Outcome"] is True
        else:
            assert step["StepStatus"] == "Failed"
            assert step["FailureReason"] == "F1 score too low:0.7"
