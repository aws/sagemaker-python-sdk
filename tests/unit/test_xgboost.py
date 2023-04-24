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

import logging
import json
import os
import pytest

from mock import Mock
from mock import patch
from packaging.version import Version


from sagemaker.fw_utils import UploadedCode
from sagemaker.session_settings import SessionSettings
from sagemaker.xgboost import XGBoost, XGBoostModel, XGBoostPredictor


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPT_PATH = os.path.join(DATA_DIR, "dummy_script.py")
SERVING_SCRIPT_FILE = "another_dummy_script.py"
TIMESTAMP = "2017-11-06-14:14:15.672"
TIME = 1510006209.073025
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
DIST_INSTANCE_COUNT = 2
INSTANCE_TYPE = "ml.c4.4xlarge"
GPU_INSTANCE_TYPE = "ml.p2.xlarge"
PYTHON_VERSION = "py3"
IMAGE_URI = "sagemaker-xgboost"
JOB_NAME = "{}-{}".format(IMAGE_URI, TIMESTAMP)
IMAGE_URI_FORMAT_STRING = "246618743249.dkr.ecr.{}.amazonaws.com/{}:{}"
ROLE = "Dummy"
REGION = "us-west-2"
CPU = "ml.c4.xlarge"

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}

LIST_TAGS_RESULT = {"Tags": [{"Key": "TagtestKey", "Value": "TagtestValue"}]}

EXPERIMENT_CONFIG = {
    "ExperimentName": "exp",
    "TrialName": "trial",
    "TrialComponentDisplayName": "tc",
    "RunName": "rn",
}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        s3_resource=None,
        s3_client=None,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )

    describe = {"ModelArtifacts": {"S3ModelArtifacts": "s3://m/m.tar.gz"}}
    session.sagemaker_client.describe_training_job = Mock(return_value=describe)
    session.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    session.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    session.sagemaker_client.list_tags = Mock(return_value=LIST_TAGS_RESULT)
    session.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    # For tests which doesn't verify config file injection, operate with empty config
    session.sagemaker_config = {}
    return session


def _get_full_image_uri(version):
    if Version(version) < Version("1.2-1"):
        image_tag = "-".join([version, "cpu", PYTHON_VERSION])
    else:
        image_tag = version
    return IMAGE_URI_FORMAT_STRING.format(REGION, IMAGE_URI, image_tag)


def _xgboost_estimator(
    sagemaker_session,
    framework_version,
    instance_type=None,
    instance_count=1,
    base_job_name=None,
    **kwargs,
):

    return XGBoost(
        entry_point=SCRIPT_PATH,
        framework_version=framework_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_type=instance_type if instance_type else INSTANCE_TYPE,
        instance_count=instance_count,
        base_job_name=base_job_name,
        py_version=PYTHON_VERSION,
        **kwargs,
    )


def _create_train_job(version, instance_count=1, instance_type="ml.c4.4xlarge"):
    return {
        "image_uri": _get_full_image_uri(version),
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
            "InstanceType": instance_type,
            "InstanceCount": instance_count,
            "VolumeSizeInGB": 30,
        },
        "hyperparameters": {
            "sagemaker_program": json.dumps("dummy_script.py"),
            "sagemaker_container_log_level": str(logging.INFO),
            "sagemaker_job_name": json.dumps(JOB_NAME),
            "sagemaker_submit_directory": json.dumps(
                "s3://{}/{}/source/sourcedir.tar.gz".format(BUCKET_NAME, JOB_NAME)
            ),
            "sagemaker_region": '"us-west-2"',
        },
        "stop_condition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
        "retry_strategy": None,
        "metric_definitions": None,
        "tags": None,
        "vpc_config": None,
        "environment": None,
        "experiment_config": None,
        "debugger_hook_config": {
            "CollectionConfigurations": [],
            "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        },
        "profiler_config": {
            "DisableProfiler": False,
            "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        },
    }


def test_create_model(sagemaker_session, xgboost_framework_version):
    source_dir = "s3://mybucket/source"

    xgboost_model = XGBoostModel(
        model_data=source_dir,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        entry_point=SCRIPT_PATH,
        framework_version=xgboost_framework_version,
    )
    default_image_uri = _get_full_image_uri(xgboost_framework_version)
    model_values = xgboost_model.prepare_container_def(CPU)
    assert model_values["Image"] == default_image_uri


@patch("sagemaker.model.FrameworkModel._upload_code")
def test_create_model_with_network_isolation(upload, sagemaker_session, xgboost_framework_version):
    source_dir = "s3://mybucket/source"
    repacked_model_data = "s3://mybucket/prefix/model.tar.gz"

    xgboost_model = XGBoostModel(
        model_data=source_dir,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        entry_point=SCRIPT_PATH,
        framework_version=xgboost_framework_version,
        enable_network_isolation=True,
    )
    xgboost_model.uploaded_code = UploadedCode(s3_prefix=repacked_model_data, script_name="script")
    xgboost_model.repacked_model_data = repacked_model_data
    model_values = xgboost_model.prepare_container_def(CPU)
    assert model_values["Environment"]["SAGEMAKER_SUBMIT_DIRECTORY"] == "/opt/ml/model/code"
    assert model_values["ModelDataUrl"] == repacked_model_data


@patch("sagemaker.estimator.name_from_base")
def test_create_model_from_estimator(name_from_base, sagemaker_session, xgboost_framework_version):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    base_job_name = "job"

    xgboost = XGBoost(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        framework_version=xgboost_framework_version,
        container_log_level=container_log_level,
        py_version=PYTHON_VERSION,
        base_job_name=base_job_name,
        source_dir=source_dir,
    )

    xgboost.fit(inputs="s3://mybucket/train", job_name="new_name")

    model_name = "model_name"
    name_from_base.return_value = model_name
    model = xgboost.create_model()

    assert model.sagemaker_session == sagemaker_session
    assert model.framework_version == xgboost_framework_version
    assert model.py_version == xgboost.py_version
    assert model.entry_point == SCRIPT_PATH
    assert model.role == ROLE
    assert model.name == model_name
    assert model.container_log_level == container_log_level
    assert model.source_dir == source_dir
    assert model.vpc_config is None

    name_from_base.assert_called_with(base_job_name)


def test_create_model_with_optional_params(sagemaker_session, xgboost_framework_version):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    xgboost = XGBoost(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        framework_version=xgboost_framework_version,
        sagemaker_session=sagemaker_session,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        container_log_level=container_log_level,
        py_version=PYTHON_VERSION,
        base_job_name="job",
        source_dir=source_dir,
    )

    xgboost.fit(inputs="s3://mybucket/train", job_name="new_name")

    custom_image = "ubuntu:latest"
    new_role = "role"
    model_server_workers = 2
    vpc_config = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    new_source_dir = "s3://myotherbucket/source"
    dependencies = ["/directory/a", "/directory/b"]
    model_name = "model-name"
    model = xgboost.create_model(
        image_uri=custom_image,
        role=new_role,
        model_server_workers=model_server_workers,
        vpc_config_override=vpc_config,
        entry_point=SERVING_SCRIPT_FILE,
        source_dir=new_source_dir,
        dependencies=dependencies,
        name=model_name,
    )

    assert model.image_uri == custom_image
    assert model.role == new_role
    assert model.model_server_workers == model_server_workers
    assert model.vpc_config == vpc_config
    assert model.entry_point == SERVING_SCRIPT_FILE
    assert model.source_dir == new_source_dir
    assert model.dependencies == dependencies
    assert model.name == model_name


def test_create_model_with_custom_image(sagemaker_session, xgboost_framework_version):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    custom_image = "ubuntu:latest"
    xgboost = XGBoost(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        framework_version=xgboost_framework_version,
        sagemaker_session=sagemaker_session,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        image_uri=custom_image,
        container_log_level=container_log_level,
        py_version=PYTHON_VERSION,
        base_job_name="job",
        source_dir=source_dir,
    )

    xgboost.fit(inputs="s3://mybucket/train", job_name="new_name")
    model = xgboost.create_model()

    assert model.image_uri == custom_image


@patch("time.strftime", return_value=TIMESTAMP)
@patch("time.time", return_value=TIME)
def test_xgboost_cpu(time, strftime, sagemaker_session, xgboost_framework_version):
    xgboost = XGBoost(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        py_version=PYTHON_VERSION,
        framework_version=xgboost_framework_version,
    )

    inputs = "s3://mybucket/train"

    xgboost.fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ["train", "logs_for_job"]
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ["resource"]

    expected_train_args = _create_train_job(xgboost_framework_version)
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs
    expected_train_args["experiment_config"] = EXPERIMENT_CONFIG

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args

    model = xgboost.create_model()

    assert {
        "Environment": {
            "SAGEMAKER_SUBMIT_DIRECTORY": "s3://mybucket/sagemaker-xgboost-{}/source/sourcedir.tar.gz".format(
                TIMESTAMP
            ),
            "SAGEMAKER_PROGRAM": "dummy_script.py",
            "SAGEMAKER_REGION": "us-west-2",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        },
        "Image": _get_full_image_uri(xgboost_framework_version),
        "ModelDataUrl": "s3://m/m.tar.gz",
    } == model.prepare_container_def(CPU)

    predictor = xgboost.deploy(1, CPU)
    assert isinstance(predictor, XGBoostPredictor)


@patch("time.strftime", return_value=TIMESTAMP)
@patch("time.time", return_value=TIME)
def test_xgboost_gpu(time, strftime, sagemaker_session, xgboost_gpu_framework_version):
    xgboost = XGBoost(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_type=GPU_INSTANCE_TYPE,
        instance_count=1,
        framework_version=xgboost_gpu_framework_version,
    )

    inputs = "s3://mybucket/train"

    xgboost.fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ["train", "logs_for_job"]
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ["resource"]

    expected_train_args = _create_train_job(
        xgboost_gpu_framework_version, instance_type=GPU_INSTANCE_TYPE
    )
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs
    expected_train_args["experiment_config"] = EXPERIMENT_CONFIG

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args

    model = xgboost.create_model()

    assert {
        "Environment": {
            "SAGEMAKER_SUBMIT_DIRECTORY": "s3://mybucket/sagemaker-xgboost-{}/source/sourcedir.tar.gz".format(
                TIMESTAMP
            ),
            "SAGEMAKER_PROGRAM": "dummy_script.py",
            "SAGEMAKER_REGION": "us-west-2",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        },
        "Image": _get_full_image_uri(xgboost_gpu_framework_version),
        "ModelDataUrl": "s3://m/m.tar.gz",
    } == model.prepare_container_def(GPU_INSTANCE_TYPE)

    predictor = xgboost.deploy(1, GPU_INSTANCE_TYPE)
    assert isinstance(predictor, XGBoostPredictor)


@patch("time.strftime", return_value=TIMESTAMP)
@patch("time.time", return_value=TIME)
def test_distributed_training(time, strftime, sagemaker_session, xgboost_framework_version):
    xgboost = XGBoost(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=DIST_INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        py_version=PYTHON_VERSION,
        framework_version=xgboost_framework_version,
    )

    inputs = "s3://mybucket/train"

    xgboost.fit(inputs=inputs)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ["train", "logs_for_job"]
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ["resource"]

    expected_train_args = _create_train_job(xgboost_framework_version, DIST_INSTANCE_COUNT)
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args

    model = xgboost.create_model()

    assert {
        "Environment": {
            "SAGEMAKER_SUBMIT_DIRECTORY": "s3://mybucket/sagemaker-xgboost-{}/source/sourcedir.tar.gz".format(
                TIMESTAMP
            ),
            "SAGEMAKER_PROGRAM": "dummy_script.py",
            "SAGEMAKER_REGION": "us-west-2",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        },
        "Image": _get_full_image_uri(xgboost_framework_version),
        "ModelDataUrl": "s3://m/m.tar.gz",
    } == model.prepare_container_def(CPU)

    predictor = xgboost.deploy(1, CPU)
    assert isinstance(predictor, XGBoostPredictor)


def test_model(sagemaker_session, xgboost_framework_version):
    model = XGBoostModel(
        "s3://some/data.tar.gz",
        role=ROLE,
        framework_version=xgboost_framework_version,
        entry_point=SCRIPT_PATH,
        sagemaker_session=sagemaker_session,
    )
    predictor = model.deploy(1, CPU)
    assert isinstance(predictor, XGBoostPredictor)


def test_model_custom_serialization(sagemaker_session, xgboost_framework_version):
    model = XGBoostModel(
        "s3://some/data.tar.gz",
        role=ROLE,
        framework_version=xgboost_framework_version,
        entry_point=SCRIPT_PATH,
        sagemaker_session=sagemaker_session,
    )
    custom_serializer = Mock()
    custom_deserializer = Mock()
    predictor = model.deploy(
        1,
        CPU,
        serializer=custom_serializer,
        deserializer=custom_deserializer,
    )
    assert isinstance(predictor, XGBoostPredictor)
    assert predictor.serializer is custom_serializer
    assert predictor.deserializer is custom_deserializer


def test_training_image_uri(sagemaker_session, xgboost_framework_version):
    xgboost = XGBoost(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        framework_version=xgboost_framework_version,
        sagemaker_session=sagemaker_session,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        py_version=PYTHON_VERSION,
    )

    assert _get_full_image_uri(xgboost_framework_version) in xgboost.training_image_uri()


def test_attach(sagemaker_session, xgboost_framework_version):
    training_image = "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:{}-cpu-{}".format(
        xgboost_framework_version, PYTHON_VERSION
    )
    returned_job_description = {
        "AlgorithmSpecification": {"TrainingInputMode": "File", "TrainingImage": training_image},
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
            "sagemaker_program": '"iris-dnn-classifier.py"',
            "sagemaker_s3_uri_training": '"sagemaker-3/integ-test-data/tf_iris"',
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

    estimator = XGBoost.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert estimator._current_job_name == "neo"
    assert estimator.latest_training_job.job_name == "neo"
    assert estimator.py_version == PYTHON_VERSION
    assert estimator.framework_version == xgboost_framework_version
    assert estimator.role == "arn:aws:iam::366:role/SageMakerRole"
    assert estimator.instance_count == 1
    assert estimator.max_run == 24 * 60 * 60
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
            "TrainingImage": "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py3-cpu:1.0.4",
        },
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
            "checkpoint_path": '"s3://other/1508872349"',
            "sagemaker_program": '"iris-dnn-classifier.py"',
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
        XGBoost.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert "didn't use image for requested framework" in str(error)


def test_attach_custom_image(sagemaker_session):
    training_image = "1.dkr.ecr.us-west-2.amazonaws.com/my_custom_xgboost_image:latest"
    returned_job_description = {
        "AlgorithmSpecification": {"TrainingInputMode": "File", "TrainingImage": training_image},
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
            "sagemaker_program": '"iris-dnn-classifier.py"',
            "sagemaker_s3_uri_training": '"sagemaker-3/integ-test-data/tf_iris"',
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

    with pytest.raises(TypeError) as error:
        XGBoost.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert "expected string" in str(error)


def test_py2_xgboost_error(sagemaker_session, xgboost_framework_version):
    with pytest.raises(ValueError) as error1:
        XGBoost(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            framework_version=xgboost_framework_version,
            sagemaker_session=sagemaker_session,
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            py_version="py2",
        )

    with pytest.raises(ValueError) as error2:
        model = XGBoostModel(
            model_data=DATA_DIR,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            entry_point=SCRIPT_PATH,
            framework_version=xgboost_framework_version,
            py_version="py2",
        )
        model.serving_image_uri(REGION, INSTANCE_TYPE)

    error_message = "Unsupported Python version: py2."
    assert error_message in str(error1)
    assert error_message in str(error2)


def test_unsupported_xgboost_version_error(sagemaker_session):
    with pytest.raises(ValueError) as error1:
        XGBoost(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            framework_version="1.1",
            sagemaker_session=sagemaker_session,
            instance_type=INSTANCE_TYPE,
            instance_count=1,
        )

    with pytest.raises(ValueError) as error2:
        XGBoost(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            framework_version="1.1-1",
            sagemaker_session=sagemaker_session,
            instance_type=INSTANCE_TYPE,
            instance_count=1,
        )

    error_message = "XGBoost 1.1 is not supported"
    assert error_message in str(error1)
    assert error_message in str(error2)
