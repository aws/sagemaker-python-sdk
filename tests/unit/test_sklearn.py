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

from mock import Mock
from mock import patch

from sagemaker.sklearn import defaults, SKLearn, SKLearnModel, SKLearnPredictor
from sagemaker.fw_utils import UploadedCode

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPT_PATH = os.path.join(DATA_DIR, "dummy_script.py")
SERVING_SCRIPT_FILE = "another_dummy_script.py"
TIMESTAMP = "2017-11-06-14:14:15.672"
TIME = 1507167947
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
DIST_INSTANCE_COUNT = 2
INSTANCE_TYPE = "ml.c4.4xlarge"
GPU_INSTANCE_TYPE = "ml.p2.xlarge"
PYTHON_VERSION = "py3"
IMAGE_NAME = "sagemaker-scikit-learn"
JOB_NAME = "{}-{}".format(IMAGE_NAME, TIMESTAMP)
IMAGE_URI_FORMAT_STRING = "246618743249.dkr.ecr.{}.amazonaws.com/{}:{}-{}-{}"
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
    )

    describe = {"ModelArtifacts": {"S3ModelArtifacts": "s3://m/m.tar.gz"}}
    session.sagemaker_client.describe_training_job = Mock(return_value=describe)
    session.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    session.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    session.sagemaker_client.list_tags = Mock(return_value=LIST_TAGS_RESULT)
    session.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    return session


def _get_full_cpu_image_uri(version):
    return IMAGE_URI_FORMAT_STRING.format(REGION, IMAGE_NAME, version, "cpu", PYTHON_VERSION)


def _sklearn_estimator(
    sagemaker_session,
    framework_version=defaults.SKLEARN_VERSION,
    train_instance_type=None,
    base_job_name=None,
    **kwargs
):
    return SKLearn(
        entry_point=SCRIPT_PATH,
        framework_version=framework_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_type=train_instance_type if train_instance_type else INSTANCE_TYPE,
        base_job_name=base_job_name,
        py_version=PYTHON_VERSION,
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
        "metric_definitions": None,
        "tags": None,
        "vpc_config": None,
        "experiment_config": None,
        "debugger_hook_config": {
            "CollectionConfigurations": [],
            "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        },
    }


def test_train_image(sagemaker_session, sklearn_version):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    sklearn = SKLearn(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_type=INSTANCE_TYPE,
        framework_version=sklearn_version,
        container_log_level=container_log_level,
        py_version=PYTHON_VERSION,
        base_job_name="job",
        source_dir=source_dir,
    )

    train_image = sklearn.train_image()
    assert (
        train_image
        == "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.20.0-cpu-py3"
    )


def test_create_model(sagemaker_session):
    source_dir = "s3://mybucket/source"

    sklearn_model = SKLearnModel(
        model_data=source_dir,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        entry_point=SCRIPT_PATH,
    )
    default_image_uri = _get_full_cpu_image_uri("0.20.0")
    model_values = sklearn_model.prepare_container_def(CPU)
    assert model_values["Image"] == default_image_uri


@patch("sagemaker.model.FrameworkModel._upload_code")
def test_create_model_with_network_isolation(upload, sagemaker_session):
    source_dir = "s3://mybucket/source"
    repacked_model_data = "s3://mybucket/prefix/model.tar.gz"

    sklearn_model = SKLearnModel(
        model_data=source_dir,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        entry_point=SCRIPT_PATH,
        enable_network_isolation=True,
    )
    sklearn_model.uploaded_code = UploadedCode(s3_prefix=repacked_model_data, script_name="script")
    sklearn_model.repacked_model_data = repacked_model_data
    model_values = sklearn_model.prepare_container_def(CPU)
    assert model_values["Environment"]["SAGEMAKER_SUBMIT_DIRECTORY"] == "/opt/ml/model/code"
    assert model_values["ModelDataUrl"] == repacked_model_data


def test_create_model_from_estimator(sagemaker_session, sklearn_version):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    sklearn = SKLearn(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_type=INSTANCE_TYPE,
        framework_version=sklearn_version,
        container_log_level=container_log_level,
        py_version=PYTHON_VERSION,
        base_job_name="job",
        source_dir=source_dir,
        enable_network_isolation=True,
    )

    job_name = "new_name"
    sklearn.fit(inputs="s3://mybucket/train", job_name=job_name)
    model = sklearn.create_model()

    assert model.sagemaker_session == sagemaker_session
    assert model.framework_version == sklearn_version
    assert model.py_version == sklearn.py_version
    assert model.entry_point == SCRIPT_PATH
    assert model.role == ROLE
    assert model.name == job_name
    assert model.container_log_level == container_log_level
    assert model.source_dir == source_dir
    assert model.vpc_config is None
    assert model.enable_network_isolation()


def test_create_model_with_optional_params(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    enable_cloudwatch_metrics = "true"
    sklearn = SKLearn(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_type=INSTANCE_TYPE,
        container_log_level=container_log_level,
        py_version=PYTHON_VERSION,
        base_job_name="job",
        source_dir=source_dir,
        enable_cloudwatch_metrics=enable_cloudwatch_metrics,
    )

    sklearn.fit(inputs="s3://mybucket/train", job_name="new_name")

    custom_image = "ubuntu:latest"
    new_role = "role"
    model_server_workers = 2
    vpc_config = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    new_source_dir = "s3://myotherbucket/source"
    dependencies = ["/directory/a", "/directory/b"]
    model_name = "model-name"
    model = sklearn.create_model(
        image=custom_image,
        role=new_role,
        model_server_workers=model_server_workers,
        vpc_config_override=vpc_config,
        entry_point=SERVING_SCRIPT_FILE,
        source_dir=new_source_dir,
        dependencies=dependencies,
        name=model_name,
    )

    assert model.image == custom_image
    assert model.role == new_role
    assert model.model_server_workers == model_server_workers
    assert model.vpc_config == vpc_config
    assert model.entry_point == SERVING_SCRIPT_FILE
    assert model.source_dir == new_source_dir
    assert model.dependencies == dependencies
    assert model.name == model_name


def test_create_model_with_custom_image(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    custom_image = "ubuntu:latest"
    sklearn = SKLearn(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_type=INSTANCE_TYPE,
        image_name=custom_image,
        container_log_level=container_log_level,
        py_version=PYTHON_VERSION,
        base_job_name="job",
        source_dir=source_dir,
    )

    sklearn.fit(inputs="s3://mybucket/train", job_name="new_name")
    model = sklearn.create_model()

    assert model.image == custom_image


@patch("time.strftime", return_value=TIMESTAMP)
def test_sklearn(strftime, sagemaker_session, sklearn_version):
    sklearn = SKLearn(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_type=INSTANCE_TYPE,
        py_version=PYTHON_VERSION,
        framework_version=sklearn_version,
    )

    inputs = "s3://mybucket/train"

    sklearn.fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ["train", "logs_for_job"]
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ["resource"]

    expected_train_args = _create_train_job(sklearn_version)
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs
    expected_train_args["experiment_config"] = EXPERIMENT_CONFIG

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args

    model = sklearn.create_model()

    expected_image_base = (
        "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:{}-cpu-{}"
    )
    assert {
        "Environment": {
            "SAGEMAKER_SUBMIT_DIRECTORY": "s3://mybucket/sagemaker-scikit-learn-{}/source/sourcedir.tar.gz".format(
                TIMESTAMP
            ),
            "SAGEMAKER_PROGRAM": "dummy_script.py",
            "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
            "SAGEMAKER_REGION": "us-west-2",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        },
        "Image": expected_image_base.format(sklearn_version, PYTHON_VERSION),
        "ModelDataUrl": "s3://m/m.tar.gz",
    } == model.prepare_container_def(CPU)

    assert "cpu" in model.prepare_container_def(CPU)["Image"]
    predictor = sklearn.deploy(1, CPU)
    assert isinstance(predictor, SKLearnPredictor)


def test_transform_multiple_values_for_entry_point_issue(sagemaker_session, sklearn_version):
    # https://github.com/aws/sagemaker-python-sdk/issues/974
    sklearn = SKLearn(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_type=INSTANCE_TYPE,
        py_version=PYTHON_VERSION,
        framework_version=sklearn_version,
    )

    inputs = "s3://mybucket/train"

    sklearn.fit(inputs=inputs)

    transformer = sklearn.transformer(instance_count=1, instance_type="ml.m4.xlarge")
    # if we got here, we didn't get a "multiple values" error
    assert transformer is not None


def test_fail_distributed_training(sagemaker_session, sklearn_version):
    with pytest.raises(AttributeError) as error:
        SKLearn(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            train_instance_count=DIST_INSTANCE_COUNT,
            train_instance_type=INSTANCE_TYPE,
            py_version=PYTHON_VERSION,
            framework_version=sklearn_version,
        )
    assert "Scikit-Learn does not support distributed training." in str(error)


def test_fail_GPU_training(sagemaker_session, sklearn_version):
    with pytest.raises(ValueError) as error:
        SKLearn(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            train_instance_type=GPU_INSTANCE_TYPE,
            py_version=PYTHON_VERSION,
            framework_version=sklearn_version,
        )
    assert "GPU training in not supported for Scikit-Learn." in str(error)


def test_model(sagemaker_session):
    model = SKLearnModel(
        "s3://some/data.tar.gz",
        role=ROLE,
        entry_point=SCRIPT_PATH,
        sagemaker_session=sagemaker_session,
    )
    predictor = model.deploy(1, CPU)
    assert isinstance(predictor, SKLearnPredictor)


def test_train_image_default(sagemaker_session):
    sklearn = SKLearn(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_type=INSTANCE_TYPE,
        py_version=PYTHON_VERSION,
    )

    assert _get_full_cpu_image_uri(defaults.SKLEARN_VERSION) in sklearn.train_image()


def test_train_image_cpu_instances(sagemaker_session, sklearn_version):
    sklearn = _sklearn_estimator(
        sagemaker_session, sklearn_version, train_instance_type="ml.c2.2xlarge"
    )
    assert sklearn.train_image() == _get_full_cpu_image_uri(sklearn_version)

    sklearn = _sklearn_estimator(
        sagemaker_session, sklearn_version, train_instance_type="ml.c4.2xlarge"
    )
    assert sklearn.train_image() == _get_full_cpu_image_uri(sklearn_version)

    sklearn = _sklearn_estimator(sagemaker_session, sklearn_version, train_instance_type="ml.m16")
    assert sklearn.train_image() == _get_full_cpu_image_uri(sklearn_version)


def test_attach(sagemaker_session, sklearn_version):
    training_image = "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:{}-cpu-{}".format(
        sklearn_version, PYTHON_VERSION
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

    estimator = SKLearn.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert estimator._current_job_name == "neo"
    assert estimator.latest_training_job.job_name == "neo"
    assert estimator.py_version == PYTHON_VERSION
    assert estimator.framework_version == sklearn_version
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
            "TrainingImage": "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py3-cpu:1.0.4",
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
        SKLearn.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert "didn't use image for requested framework" in str(error)


def test_attach_custom_image(sagemaker_session):
    training_image = "1.dkr.ecr.us-west-2.amazonaws.com/my_custom_sklearn_image:latest"
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

    estimator = SKLearn.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert estimator.image_name == training_image
    assert estimator.train_image() == training_image


@patch("sagemaker.sklearn.estimator.python_deprecation_warning")
def test_estimator_py2_warning(warning, sagemaker_session):
    estimator = SKLearn(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        py_version="py2",
    )

    assert estimator.py_version == "py2"
    warning.assert_called_with(estimator.__framework_name__, defaults.LATEST_PY2_VERSION)


@patch("sagemaker.sklearn.model.python_deprecation_warning")
def test_model_py2_warning(warning, sagemaker_session):
    source_dir = "s3://mybucket/source"

    model = SKLearnModel(
        model_data=source_dir,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        sagemaker_session=sagemaker_session,
        py_version="py2",
    )
    assert model.py_version == "py2"
    warning.assert_called_with(model.__framework_name__, defaults.LATEST_PY2_VERSION)


def test_custom_image_estimator_deploy(sagemaker_session):
    custom_image = "mycustomimage:latest"
    sklearn = _sklearn_estimator(sagemaker_session)
    sklearn.fit(inputs="s3://mybucket/train", job_name="new_name")
    model = sklearn.create_model(image=custom_image)
    assert model.image == custom_image
