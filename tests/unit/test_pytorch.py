# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import logging

import json
import os
import pytest
import sys
from mock import ANY, MagicMock, Mock, patch

from sagemaker.pytorch import defaults
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch import PyTorchPredictor, PyTorchModel


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPT_PATH = os.path.join(DATA_DIR, "dummy_script.py")
SERVING_SCRIPT_FILE = "another_dummy_script.py"
MODEL_DATA = "s3://some/data.tar.gz"
TIMESTAMP = "2017-11-06-14:14:15.672"
TIME = 1507167947
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.4xlarge"
ACCELERATOR_TYPE = "ml.eia.medium"
PYTHON_VERSION = "py" + str(sys.version_info.major)
IMAGE_NAME = "sagemaker-pytorch"
JOB_NAME = "{}-{}".format(IMAGE_NAME, TIMESTAMP)
IMAGE_URI_FORMAT_STRING = "520713654638.dkr.ecr.{}.amazonaws.com/{}:{}-{}-{}"
ROLE = "Dummy"
REGION = "us-west-2"
GPU = "ml.p2.xlarge"
CPU = "ml.c4.xlarge"

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}

LIST_TAGS_RESULT = {"Tags": [{"Key": "TagtestKey", "Value": "TagtestValue"}]}

EXPERIMENT_CONFIG = {
    "ExperimentName": "exp",
    "TrialName": "trial",
    "TrialComponentDisplayName": "tc",
}


@pytest.fixture(name="sagemaker_session")
def fixture_sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
    )

    describe = {"ModelArtifacts": {"S3ModelArtifacts": "s3://m/m.tar.gz"}}
    session.sagemaker_client.describe_training_job = Mock(return_value=describe)
    session.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    session.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    session.sagemaker_client.list_tags = Mock(return_value=LIST_TAGS_RESULT)
    session.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    return session


def _get_full_cpu_image_uri(version, py_version=PYTHON_VERSION):
    return IMAGE_URI_FORMAT_STRING.format(REGION, IMAGE_NAME, version, "cpu", py_version)


def _get_full_gpu_image_uri(version, py_version=PYTHON_VERSION):
    return IMAGE_URI_FORMAT_STRING.format(REGION, IMAGE_NAME, version, "gpu", py_version)


def _get_full_cpu_image_uri_with_ei(version, py_version=PYTHON_VERSION):
    return _get_full_cpu_image_uri(version, py_version=py_version) + "-eia"


def _pytorch_estimator(
    sagemaker_session,
    framework_version=defaults.PYTORCH_VERSION,
    train_instance_type=None,
    base_job_name=None,
    **kwargs
):
    return PyTorch(
        entry_point=SCRIPT_PATH,
        framework_version=framework_version,
        py_version=PYTHON_VERSION,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=train_instance_type if train_instance_type else INSTANCE_TYPE,
        base_job_name=base_job_name,
        **kwargs
    )


def _create_train_job(version):
    return {
        "image": _get_full_cpu_image_uri(version),
        "input_mode": "File",
        "input_config": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataDistributionType": "FullyReplicated",
                        "S3DataType": "S3Prefix",
                    }
                },
            }
        ],
        "role": ROLE,
        "job_name": JOB_NAME,
        "output_config": {"S3OutputPath": "s3://{}/".format(BUCKET_NAME)},
        "resource_config": {
            "InstanceType": "ml.c4.4xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "hyperparameters": {
            "sagemaker_program": json.dumps("dummy_script.py"),
            "sagemaker_enable_cloudwatch_metrics": "false",
            "sagemaker_container_log_level": str(logging.INFO),
            "sagemaker_job_name": json.dumps(JOB_NAME),
            "sagemaker_submit_directory": json.dumps(
                "s3://{}/{}/source/sourcedir.tar.gz".format(BUCKET_NAME, JOB_NAME)
            ),
            "sagemaker_region": '"us-west-2"',
        },
        "stop_condition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
        "tags": None,
        "vpc_config": None,
        "metric_definitions": None,
        "experiment_config": None,
        "debugger_hook_config": {
            "CollectionConfigurations": [],
            "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        },
    }


def test_create_model(sagemaker_session, pytorch_version):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    pytorch = PyTorch(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        framework_version=pytorch_version,
        container_log_level=container_log_level,
        base_job_name="job",
        source_dir=source_dir,
    )

    job_name = "new_name"
    pytorch.fit(inputs="s3://mybucket/train", job_name="new_name")
    model = pytorch.create_model()

    assert model.sagemaker_session == sagemaker_session
    assert model.framework_version == pytorch_version
    assert model.py_version == pytorch.py_version
    assert model.entry_point == SCRIPT_PATH
    assert model.role == ROLE
    assert model.name == job_name
    assert model.container_log_level == container_log_level
    assert model.source_dir == source_dir
    assert model.vpc_config is None


def test_create_model_with_optional_params(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    enable_cloudwatch_metrics = "true"
    pytorch = PyTorch(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        container_log_level=container_log_level,
        base_job_name="job",
        source_dir=source_dir,
        enable_cloudwatch_metrics=enable_cloudwatch_metrics,
    )

    pytorch.fit(inputs="s3://mybucket/train", job_name="new_name")

    new_role = "role"
    model_server_workers = 2
    vpc_config = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    model = pytorch.create_model(
        role=new_role,
        model_server_workers=model_server_workers,
        vpc_config_override=vpc_config,
        entry_point=SERVING_SCRIPT_FILE,
    )

    assert model.role == new_role
    assert model.model_server_workers == model_server_workers
    assert model.vpc_config == vpc_config
    assert model.entry_point == SERVING_SCRIPT_FILE


def test_create_model_with_custom_image(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    image = "pytorch:9000"
    pytorch = PyTorch(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        container_log_level=container_log_level,
        image_name=image,
        base_job_name="job",
        source_dir=source_dir,
    )

    job_name = "new_name"
    pytorch.fit(inputs="s3://mybucket/train", job_name="new_name")
    model = pytorch.create_model()

    assert model.sagemaker_session == sagemaker_session
    assert model.image == image
    assert model.entry_point == SCRIPT_PATH
    assert model.role == ROLE
    assert model.name == job_name
    assert model.container_log_level == container_log_level
    assert model.source_dir == source_dir


@patch("sagemaker.utils.create_tar_file", MagicMock())
@patch("time.strftime", return_value=TIMESTAMP)
def test_pytorch(strftime, sagemaker_session, pytorch_version):
    pytorch = PyTorch(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        framework_version=pytorch_version,
        py_version=PYTHON_VERSION,
    )

    inputs = "s3://mybucket/train"

    pytorch.fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ["train", "logs_for_job"]
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ["resource"]

    expected_train_args = _create_train_job(pytorch_version)
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs
    expected_train_args["experiment_config"] = EXPERIMENT_CONFIG

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args

    model = pytorch.create_model()

    expected_image_base = "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-pytorch:{}-gpu-{}"
    assert {
        "Environment": {
            "SAGEMAKER_SUBMIT_DIRECTORY": "s3://mybucket/sagemaker-pytorch-{}/source/sourcedir.tar.gz".format(
                TIMESTAMP
            ),
            "SAGEMAKER_PROGRAM": "dummy_script.py",
            "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
            "SAGEMAKER_REGION": "us-west-2",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        },
        "Image": expected_image_base.format(pytorch_version, PYTHON_VERSION),
        "ModelDataUrl": "s3://m/m.tar.gz",
    } == model.prepare_container_def(GPU)

    assert "cpu" in model.prepare_container_def(CPU)["Image"]
    predictor = pytorch.deploy(1, GPU)
    assert isinstance(predictor, PyTorchPredictor)


@patch("sagemaker.utils.create_tar_file", MagicMock())
def test_model(sagemaker_session):
    model = PyTorchModel(
        MODEL_DATA, role=ROLE, entry_point=SCRIPT_PATH, sagemaker_session=sagemaker_session
    )
    predictor = model.deploy(1, GPU)
    assert isinstance(predictor, PyTorchPredictor)


@patch("sagemaker.utils.create_tar_file", MagicMock())
@patch("sagemaker.utils.repack_model")
def test_mms_model(repack_model, sagemaker_session):
    PyTorchModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        sagemaker_session=sagemaker_session,
        framework_version="1.2",
    ).deploy(1, GPU)

    repack_model.assert_called_with(
        dependencies=[],
        inference_script=SCRIPT_PATH,
        kms_key=None,
        model_uri="s3://some/data.tar.gz",
        repacked_model_uri=ANY,
        sagemaker_session=sagemaker_session,
        source_directory=None,
    )


@patch("sagemaker.utils.create_tar_file", MagicMock())
@patch("sagemaker.utils.repack_model")
def test_non_mms_model(repack_model, sagemaker_session):
    PyTorchModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        sagemaker_session=sagemaker_session,
        framework_version="1.1",
    ).deploy(1, GPU)

    repack_model.assert_not_called()


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_model_image_accelerator(sagemaker_session):
    model = PyTorchModel(
        MODEL_DATA, role=ROLE, entry_point=SCRIPT_PATH, sagemaker_session=sagemaker_session
    )
    with pytest.raises(ValueError):
        model.prepare_container_def(INSTANCE_TYPE, accelerator_type=ACCELERATOR_TYPE)


def test_train_image_default(sagemaker_session):
    pytorch = PyTorch(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
    )

    assert (
        _get_full_cpu_image_uri(defaults.PYTORCH_VERSION, defaults.PYTHON_VERSION)
        in pytorch.train_image()
    )


def test_train_image_cpu_instances(sagemaker_session, pytorch_version):
    pytorch = _pytorch_estimator(
        sagemaker_session, pytorch_version, train_instance_type="ml.c2.2xlarge"
    )
    assert pytorch.train_image() == _get_full_cpu_image_uri(pytorch_version)

    pytorch = _pytorch_estimator(
        sagemaker_session, pytorch_version, train_instance_type="ml.c4.2xlarge"
    )
    assert pytorch.train_image() == _get_full_cpu_image_uri(pytorch_version)

    pytorch = _pytorch_estimator(sagemaker_session, pytorch_version, train_instance_type="ml.m16")
    assert pytorch.train_image() == _get_full_cpu_image_uri(pytorch_version)


def test_train_image_gpu_instances(sagemaker_session, pytorch_version):
    pytorch = _pytorch_estimator(
        sagemaker_session, pytorch_version, train_instance_type="ml.g2.2xlarge"
    )
    assert pytorch.train_image() == _get_full_gpu_image_uri(pytorch_version)

    pytorch = _pytorch_estimator(
        sagemaker_session, pytorch_version, train_instance_type="ml.p2.2xlarge"
    )
    assert pytorch.train_image() == _get_full_gpu_image_uri(pytorch_version)


def test_attach(sagemaker_session, pytorch_version):
    training_image = "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-pytorch:{}-cpu-{}".format(
        pytorch_version, PYTHON_VERSION
    )
    returned_job_description = {
        "AlgorithmSpecification": {"TrainingInputMode": "File", "TrainingImage": training_image},
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
            "sagemaker_program": '"iris-dnn-classifier.py"',
            "sagemaker_s3_uri_training": '"sagemaker-3/integ-test-data/tf_iris"',
            "sagemaker_enable_cloudwatch_metrics": "false",
            "sagemaker_container_log_level": '"logging.INFO"',
            "sagemaker_job_name": '"neo"',
            "training_steps": "100",
            "sagemaker_region": '"us-west-2"',
        },
        "RoleArn": "arn:aws:iam::366:role/SageMakerRole",
        "ResourceConfig": {
            "VolumeSizeInGB": 30,
            "InstanceCount": 1,
            "InstanceType": "ml.c4.xlarge",
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
        "TrainingJobName": "neo",
        "TrainingJobStatus": "Completed",
        "TrainingJobArn": "arn:aws:sagemaker:us-west-2:336:training-job/neo",
        "OutputDataConfig": {"KmsKeyId": "", "S3OutputPath": "s3://place/output/neo"},
        "TrainingJobOutput": {"S3TrainingJobOutput": "s3://here/output.tar.gz"},
    }
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=returned_job_description
    )

    estimator = PyTorch.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == "neo"
    assert estimator.py_version == PYTHON_VERSION
    assert estimator.framework_version == pytorch_version
    assert estimator.role == "arn:aws:iam::366:role/SageMakerRole"
    assert estimator.train_instance_count == 1
    assert estimator.train_max_run == 24 * 60 * 60
    assert estimator.input_mode == "File"
    assert estimator.base_job_name == "neo"
    assert estimator.output_path == "s3://place/output/neo"
    assert estimator.output_kms_key == ""
    assert estimator.hyperparameters()["training_steps"] == "100"
    assert estimator.source_dir == "s3://some/sourcedir.tar.gz"
    assert estimator.entry_point == "iris-dnn-classifier.py"


def test_attach_wrong_framework(sagemaker_session):
    rjd = {
        "AlgorithmSpecification": {
            "TrainingInputMode": "File",
            "TrainingImage": "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py2-cpu:1.0.4",
        },
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
            "checkpoint_path": '"s3://other/1508872349"',
            "sagemaker_program": '"iris-dnn-classifier.py"',
            "sagemaker_enable_cloudwatch_metrics": "false",
            "sagemaker_container_log_level": '"logging.INFO"',
            "training_steps": "100",
            "sagemaker_region": '"us-west-2"',
        },
        "RoleArn": "arn:aws:iam::366:role/SageMakerRole",
        "ResourceConfig": {
            "VolumeSizeInGB": 30,
            "InstanceCount": 1,
            "InstanceType": "ml.c4.xlarge",
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
        "TrainingJobName": "neo",
        "TrainingJobStatus": "Completed",
        "TrainingJobArn": "arn:aws:sagemaker:us-west-2:336:training-job/neo",
        "OutputDataConfig": {"KmsKeyId": "", "S3OutputPath": "s3://place/output/neo"},
        "TrainingJobOutput": {"S3TrainingJobOutput": "s3://here/output.tar.gz"},
    }
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=rjd
    )

    with pytest.raises(ValueError) as error:
        PyTorch.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert "didn't use image for requested framework" in str(error)


def test_attach_custom_image(sagemaker_session):
    training_image = "pytorch:latest"
    returned_job_description = {
        "AlgorithmSpecification": {"TrainingInputMode": "File", "TrainingImage": training_image},
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
            "sagemaker_program": '"iris-dnn-classifier.py"',
            "sagemaker_s3_uri_training": '"sagemaker-3/integ-test-data/tf_iris"',
            "sagemaker_enable_cloudwatch_metrics": "false",
            "sagemaker_container_log_level": '"logging.INFO"',
            "sagemaker_job_name": '"neo"',
            "training_steps": "100",
            "sagemaker_region": '"us-west-2"',
        },
        "RoleArn": "arn:aws:iam::366:role/SageMakerRole",
        "ResourceConfig": {
            "VolumeSizeInGB": 30,
            "InstanceCount": 1,
            "InstanceType": "ml.c4.xlarge",
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
        "TrainingJobName": "neo",
        "TrainingJobStatus": "Completed",
        "TrainingJobArn": "arn:aws:sagemaker:us-west-2:336:training-job/neo",
        "OutputDataConfig": {"KmsKeyId": "", "S3OutputPath": "s3://place/output/neo"},
        "TrainingJobOutput": {"S3TrainingJobOutput": "s3://here/output.tar.gz"},
    }
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=returned_job_description
    )

    estimator = PyTorch.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == "neo"
    assert estimator.image_name == training_image
    assert estimator.train_image() == training_image


@patch("sagemaker.pytorch.estimator.python_deprecation_warning")
def test_estimator_py2_warning(warning, sagemaker_session):
    estimator = PyTorch(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        py_version="py2",
    )

    assert estimator.py_version == "py2"
    warning.assert_called_with(estimator.__framework_name__, defaults.LATEST_PY2_VERSION)


@patch("sagemaker.pytorch.model.python_deprecation_warning")
def test_model_py2_warning(warning, sagemaker_session):
    model = PyTorchModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        sagemaker_session=sagemaker_session,
        py_version="py2",
    )
    assert model.py_version == "py2"
    warning.assert_called_with(model.__framework_name__, defaults.LATEST_PY2_VERSION)


@patch("sagemaker.pytorch.estimator.empty_framework_version_warning")
def test_empty_framework_version(warning, sagemaker_session):
    estimator = PyTorch(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        framework_version=None,
    )

    assert estimator.framework_version == defaults.PYTORCH_VERSION
    warning.assert_called_with(defaults.PYTORCH_VERSION, estimator.LATEST_VERSION)


@patch("sagemaker.pytorch.model.empty_framework_version_warning")
def test_model_empty_framework_version(warning, sagemaker_session):
    model = PyTorchModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        sagemaker_session=sagemaker_session,
        framework_version=None,
    )

    assert model.framework_version == defaults.PYTORCH_VERSION
    warning.assert_called_with(defaults.PYTORCH_VERSION, defaults.LATEST_VERSION)


def test_pt_enable_sm_metrics(sagemaker_session):
    pytorch = _pytorch_estimator(sagemaker_session, enable_sagemaker_metrics=True)
    assert pytorch.enable_sagemaker_metrics


def test_pt_disable_sm_metrics(sagemaker_session):
    pytorch = _pytorch_estimator(sagemaker_session, enable_sagemaker_metrics=False)
    assert not pytorch.enable_sagemaker_metrics


def test_pt_disable_sm_metrics_if_pt_ver_is_less_than_1_15(sagemaker_session):
    for fw_version in ["1.1", "1.2"]:
        pytorch = _pytorch_estimator(sagemaker_session, framework_version=fw_version)
        assert pytorch.enable_sagemaker_metrics is None


def test_pt_enable_sm_metrics_if_fw_ver_is_at_least_1_15(sagemaker_session):
    for fw_version in ["1.3", "1.4", "2.0", "2.1"]:
        pytorch = _pytorch_estimator(sagemaker_session, framework_version=fw_version)
        assert pytorch.enable_sagemaker_metrics
