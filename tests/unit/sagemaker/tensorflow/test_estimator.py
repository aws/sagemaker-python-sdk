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

import json
import logging
import os

from mock import patch, Mock, MagicMock
from packaging import version
import pytest

from sagemaker.estimator import _TrainingJob
from sagemaker.tensorflow import TensorFlow
from tests.unit import DATA_DIR

SCRIPT_FILE = "dummy_script.py"
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_FILE)
SERVING_SCRIPT_FILE = "another_dummy_script.py"
TIMESTAMP = "2017-11-06-14:14:15.673"
TIME = 1510006209.073025
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.4xlarge"
JOB_NAME = "sagemaker-tensorflow-scriptmode-{}".format(TIMESTAMP)
ROLE = "Dummy"
REGION = "us-west-2"
IMAGE_URI_FORMAT_STRING = (
    "520713654638.dkr.ecr.{}.amazonaws.com/sagemaker-tensorflow-scriptmode:{}-cpu-{}"
)
DISTRIBUTION_PS_ENABLED = {"parameter_server": {"enabled": True}}
DISTRIBUTION_MPI_ENABLED = {
    "mpi": {"enabled": True, "custom_mpi_options": "options", "processes_per_host": 2}
}

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
    session.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    describe = {"ModelArtifacts": {"S3ModelArtifacts": "s3://m/m.tar.gz"}}
    session.sagemaker_client.describe_training_job = Mock(return_value=describe)
    session.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    session.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    session.sagemaker_client.list_tags = Mock(return_value=LIST_TAGS_RESULT)
    return session


def _image_uri(tf_version, py_version):
    return IMAGE_URI_FORMAT_STRING.format(REGION, tf_version, py_version)


def _hyperparameters(horovod=False):
    hps = {
        "sagemaker_program": json.dumps("dummy_script.py"),
        "sagemaker_submit_directory": json.dumps(
            "s3://{}/{}/source/sourcedir.tar.gz".format(BUCKET_NAME, JOB_NAME)
        ),
        "sagemaker_enable_cloudwatch_metrics": "false",
        "sagemaker_container_log_level": str(logging.INFO),
        "sagemaker_job_name": json.dumps(JOB_NAME),
        "sagemaker_region": json.dumps("us-west-2"),
    }

    if horovod:
        hps["model_dir"] = json.dumps("/opt/ml/model")
    else:
        hps["model_dir"] = json.dumps("s3://{}/{}/model".format(BUCKET_NAME, JOB_NAME))

    return hps


def _create_train_job(tf_version, horovod=False, ps=False, py_version="py2"):
    conf = {
        "image_uri": _image_uri(tf_version, py_version),
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
        "hyperparameters": _hyperparameters(horovod),
        "stop_condition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
        "tags": None,
        "vpc_config": None,
        "metric_definitions": None,
        "experiment_config": None,
    }

    if not ps:
        conf["debugger_hook_config"] = {
            "CollectionConfigurations": [],
            "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        }

    return conf


def _build_tf(
    sagemaker_session,
    framework_version=None,
    py_version=None,
    train_instance_type=None,
    base_job_name=None,
    **kwargs
):
    return TensorFlow(
        entry_point=SCRIPT_PATH,
        framework_version=framework_version,
        py_version=py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=train_instance_type if train_instance_type else INSTANCE_TYPE,
        base_job_name=base_job_name,
        **kwargs
    )


@patch("sagemaker.estimator.name_from_base")
def test_create_model(name_from_base, sagemaker_session, tf_version, tf_py_version):
    if version.Version(tf_version) < version.Version("1.11"):
        pytest.skip(
            "Legacy TF version requires explicit image URI, and "
            "this logic is tested in test_create_model_with_custom_image."
        )

    container_log_level = '"logging.INFO"'
    base_job_name = "job"
    tf = TensorFlow(
        entry_point=SCRIPT_PATH,
        source_dir="s3://mybucket/source",
        framework_version=tf_version,
        py_version=tf_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        container_log_level=container_log_level,
        base_job_name=base_job_name,
        enable_network_isolation=True,
    )

    job_name = "doing something"
    tf.fit(inputs="s3://mybucket/train", job_name=job_name)

    model_name = "doing something else"
    name_from_base.return_value = model_name
    model = tf.create_model()

    name_from_base.assert_called_with("job")

    assert model.sagemaker_session == sagemaker_session
    assert model.framework_version == tf_version
    assert model.entry_point is None
    assert model.role == ROLE
    assert model.name == model_name
    assert model._container_log_level == container_log_level
    assert model.source_dir is None
    assert model.vpc_config is None
    assert model.enable_network_isolation()


def test_create_model_with_optional_params(sagemaker_session, tf_version, tf_py_version):
    if version.Version(tf_version) < version.Version("1.11"):
        pytest.skip(
            "Legacy TF version requires explicit image URI, and "
            "this logic is tested in test_create_model_with_custom_image."
        )

    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    enable_cloudwatch_metrics = "true"
    tf = TensorFlow(
        entry_point=SCRIPT_PATH,
        framework_version=tf_version,
        py_version=tf_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        container_log_level=container_log_level,
        base_job_name="job",
        source_dir=source_dir,
        enable_cloudwatch_metrics=enable_cloudwatch_metrics,
    )

    job_name = "doing something"
    tf.fit(inputs="s3://mybucket/train", job_name=job_name)

    new_role = "role"
    vpc_config = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    model_name = "model-name"
    model = tf.create_model(
        role=new_role,
        vpc_config_override=vpc_config,
        entry_point=SERVING_SCRIPT_FILE,
        name=model_name,
        enable_network_isolation=True,
    )

    assert model.role == new_role
    assert model.vpc_config == vpc_config
    assert model.entry_point == SERVING_SCRIPT_FILE
    assert model.name == model_name
    assert model.enable_network_isolation()


def test_create_model_with_custom_image(sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    custom_image = "tensorflow:1.0"
    tf = TensorFlow(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        image_uri=custom_image,
        container_log_level=container_log_level,
        base_job_name="job",
        source_dir=source_dir,
    )

    job_name = "doing something"
    tf.fit(inputs="s3://mybucket/train", job_name=job_name)
    model = tf.create_model()

    assert model.image_uri == custom_image


@patch("sagemaker.tensorflow.estimator.TensorFlow.create_model")
def test_transformer_creation_with_optional_args(
    create_model, sagemaker_session, tf_version, tf_py_version
):
    if version.Version(tf_version) < version.Version("1.11"):
        pytest.skip(
            "Legacy TF version requires explicit image URI, and "
            "this logic is tested in test_create_model_with_custom_image."
        )

    model = Mock()
    create_model.return_value = model

    tf = TensorFlow(
        entry_point=SCRIPT_PATH,
        framework_version=tf_version,
        py_version=tf_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
    )
    tf.latest_training_job = _TrainingJob(sagemaker_session, "some-job-name")

    strategy = "SingleRecord"
    assemble_with = "Line"
    output_path = "s3://{}/batch-output".format(BUCKET_NAME)
    kms_key = "kms"
    accept_type = "text/bytes"
    env = {"foo": "bar"}
    max_concurrent_transforms = 3
    max_payload = 100
    tags = {"Key": "foo", "Value": "bar"}
    new_role = "role"
    vpc_config = {"Subnets": ["1234"], "SecurityGroupIds": ["5678"]}
    model_name = "model-name"

    tf.transformer(
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        strategy=strategy,
        assemble_with=assemble_with,
        output_path=output_path,
        output_kms_key=kms_key,
        accept=accept_type,
        env=env,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        tags=tags,
        role=new_role,
        volume_kms_key=kms_key,
        entry_point=SERVING_SCRIPT_FILE,
        vpc_config_override=vpc_config,
        enable_network_isolation=True,
        model_name=model_name,
    )

    create_model.assert_called_with(
        role=new_role,
        vpc_config_override=vpc_config,
        entry_point=SERVING_SCRIPT_FILE,
        enable_network_isolation=True,
        name=model_name,
    )
    model.transformer.assert_called_with(
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        accept=accept_type,
        assemble_with=assemble_with,
        env=env,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        output_kms_key=kms_key,
        output_path=output_path,
        strategy=strategy,
        tags=tags,
        volume_kms_key=kms_key,
    )


@patch("sagemaker.tensorflow.estimator.TensorFlow.create_model")
@patch("sagemaker.estimator.name_from_base")
def test_transformer_creation_without_optional_args(
    name_from_base, create_model, sagemaker_session, tf_version, tf_py_version
):
    if version.Version(tf_version) < version.Version("1.11"):
        pytest.skip(
            "Legacy TF version requires explicit image URI, and "
            "this logic is tested in test_create_model_with_custom_image."
        )

    model_name = "generated-model-name"
    name_from_base.return_value = model_name

    model = Mock()
    create_model.return_value = model

    base_job_name = "tensorflow"
    tf = TensorFlow(
        entry_point=SCRIPT_PATH,
        framework_version=tf_version,
        py_version=tf_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        base_job_name=base_job_name,
    )
    tf.latest_training_job = _TrainingJob(sagemaker_session, "some-job-name")
    tf.transformer(INSTANCE_COUNT, INSTANCE_TYPE)

    name_from_base.assert_called_with(base_job_name)
    create_model.assert_called_with(
        role=ROLE,
        vpc_config_override="VPC_CONFIG_DEFAULT",
        entry_point=None,
        enable_network_isolation=False,
        name=model_name,
    )
    model.transformer.assert_called_with(
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        accept=None,
        assemble_with=None,
        env=None,
        max_concurrent_transforms=None,
        max_payload=None,
        output_kms_key=None,
        output_path=None,
        strategy=None,
        tags=None,
        volume_kms_key=None,
    )


@patch("time.strftime", return_value=TIMESTAMP)
@patch("time.time", return_value=TIME)
@patch("sagemaker.utils.create_tar_file", MagicMock())
def test_fit(time, strftime, sagemaker_session):
    tf = TensorFlow(
        entry_point=SCRIPT_FILE,
        framework_version="1.11",
        py_version="py2",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_type=INSTANCE_TYPE,
        train_instance_count=1,
        source_dir=DATA_DIR,
    )

    inputs = "s3://mybucket/train"
    tf.fit(inputs=inputs)

    call_names = [c[0] for c in sagemaker_session.method_calls]
    assert call_names == ["train", "logs_for_job"]

    expected_train_args = _create_train_job("1.11", py_version="py2")
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args


@patch("time.strftime", return_value=TIMESTAMP)
@patch("time.time", return_value=TIME)
@patch("sagemaker.utils.create_tar_file", MagicMock())
def test_fit_ps(time, strftime, sagemaker_session):
    tf = TensorFlow(
        entry_point=SCRIPT_FILE,
        framework_version="1.11",
        py_version="py2",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_type=INSTANCE_TYPE,
        train_instance_count=1,
        source_dir=DATA_DIR,
        distribution=DISTRIBUTION_PS_ENABLED,
    )

    inputs = "s3://mybucket/train"
    tf.fit(inputs=inputs)

    call_names = [c[0] for c in sagemaker_session.method_calls]
    assert call_names == ["train", "logs_for_job"]

    expected_train_args = _create_train_job("1.11", ps=True, py_version="py2")
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs
    expected_train_args["hyperparameters"][TensorFlow.LAUNCH_PS_ENV_NAME] = json.dumps(True)

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args


@patch("time.strftime", return_value=TIMESTAMP)
@patch("time.time", return_value=TIME)
@patch("sagemaker.utils.create_tar_file", MagicMock())
def test_fit_mpi(time, strftime, sagemaker_session):
    tf = TensorFlow(
        entry_point=SCRIPT_FILE,
        framework_version="1.11",
        py_version="py2",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_type=INSTANCE_TYPE,
        train_instance_count=1,
        source_dir=DATA_DIR,
        distribution=DISTRIBUTION_MPI_ENABLED,
    )

    inputs = "s3://mybucket/train"
    tf.fit(inputs=inputs)

    call_names = [c[0] for c in sagemaker_session.method_calls]
    assert call_names == ["train", "logs_for_job"]

    expected_train_args = _create_train_job("1.11", horovod=True, py_version="py2")
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs
    expected_train_args["hyperparameters"][TensorFlow.LAUNCH_MPI_ENV_NAME] = json.dumps(True)
    expected_train_args["hyperparameters"][TensorFlow.MPI_NUM_PROCESSES_PER_HOST] = json.dumps(2)
    expected_train_args["hyperparameters"][TensorFlow.MPI_CUSTOM_MPI_OPTIONS] = json.dumps(
        "options"
    )

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args


def test_hyperparameters_no_model_dir(sagemaker_session, tf_version, tf_py_version):
    if version.Version(tf_version) < version.Version("1.11"):
        pytest.skip(
            "Legacy TF version requires explicit image URI, and "
            "this logic is tested in test_create_model_with_custom_image."
        )

    tf = _build_tf(
        sagemaker_session, framework_version=tf_version, py_version=tf_py_version, model_dir=False
    )
    hyperparameters = tf.hyperparameters()
    assert "model_dir" not in hyperparameters


def test_train_image_custom_image(sagemaker_session):
    custom_image = "tensorflow:latest"
    tf = _build_tf(sagemaker_session, image_uri=custom_image)
    assert custom_image == tf.train_image()
