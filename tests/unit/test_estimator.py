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

import logging
import json
import os
import subprocess
from time import sleep

import pytest
from mock import ANY, MagicMock, Mock, patch

from sagemaker.amazon.amazon_estimator import registry
from sagemaker.algorithm import AlgorithmEstimator
from sagemaker.estimator import Estimator, EstimatorBase, Framework, _TrainingJob
from sagemaker.model import FrameworkModel
from sagemaker.predictor import RealTimePredictor
from sagemaker.session import s3_input, ShuffleConfig
from sagemaker.transformer import Transformer
from botocore.exceptions import ClientError

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
ENTRY_POINT = "blah.py"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPT_NAME = "dummy_script.py"
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_NAME)
TIMESTAMP = "2017-11-06-14:14:15.671"
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "c4.4xlarge"
ACCELERATOR_TYPE = "ml.eia.medium"
ROLE = "DummyRole"
IMAGE_NAME = "fakeimage"
REGION = "us-west-2"
JOB_NAME = "{}-{}".format(IMAGE_NAME, TIMESTAMP)
TAGS = [{"Name": "some-tag", "Value": "value-for-tag"}]
OUTPUT_PATH = "s3://bucket/prefix"
GIT_REPO = "https://github.com/aws/sagemaker-python-sdk.git"
BRANCH = "test-branch-git-config"
COMMIT = "ae15c9d7d5b97ea95ea451e4662ee43da3401d73"
PRIVATE_GIT_REPO_SSH = "git@github.com:testAccount/private-repo.git"
PRIVATE_GIT_REPO = "https://github.com/testAccount/private-repo.git"
PRIVATE_BRANCH = "test-branch"
PRIVATE_COMMIT = "329bfcf884482002c05ff7f44f62599ebc9f445a"
REPO_DIR = "/tmp/repo_dir"

DESCRIBE_TRAINING_JOB_RESULT = {"ModelArtifacts": {"S3ModelArtifacts": MODEL_DATA}}

RETURNED_JOB_DESCRIPTION = {
    "AlgorithmSpecification": {
        "TrainingInputMode": "File",
        "TrainingImage": "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-other-py2-cpu:1.0.4",
    },
    "HyperParameters": {
        "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
        "checkpoint_path": '"s3://other/1508872349"',
        "sagemaker_program": '"iris-dnn-classifier.py"',
        "sagemaker_enable_cloudwatch_metrics": "false",
        "sagemaker_container_log_level": '"logging.INFO"',
        "sagemaker_job_name": '"neo"',
        "training_steps": "100",
    },
    "RoleArn": "arn:aws:iam::366:role/SageMakerRole",
    "ResourceConfig": {"VolumeSizeInGB": 30, "InstanceCount": 1, "InstanceType": "ml.c4.xlarge"},
    "StoppingCondition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
    "TrainingJobName": "neo",
    "TrainingJobStatus": "Completed",
    "TrainingJobArn": "arn:aws:sagemaker:us-west-2:336:training-job/neo",
    "OutputDataConfig": {"KmsKeyId": "", "S3OutputPath": "s3://place/output/neo"},
    "TrainingJobOutput": {"S3TrainingJobOutput": "s3://here/output.tar.gz"},
    "EnableInterContainerTrafficEncryption": False,
}

MODEL_CONTAINER_DEF = {
    "Environment": {
        "SAGEMAKER_PROGRAM": ENTRY_POINT,
        "SAGEMAKER_SUBMIT_DIRECTORY": "s3://mybucket/mi-2017-10-10-14-14-15/sourcedir.tar.gz",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        "SAGEMAKER_REGION": REGION,
        "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
    },
    "Image": MODEL_IMAGE,
    "ModelDataUrl": MODEL_DATA,
}

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}

LIST_TAGS_RESULT = {"Tags": [{"Key": "TagtestKey", "Value": "TagtestValue"}]}


class DummyFramework(Framework):
    __framework_name__ = "dummy"

    def train_image(self):
        return IMAGE_NAME

    def create_model(self, role=None, model_server_workers=None):
        return DummyFrameworkModel(self.sagemaker_session, vpc_config=self.get_vpc_config())

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        init_params = super(DummyFramework, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )
        init_params.pop("image", None)
        return init_params


class DummyFrameworkModel(FrameworkModel):
    def __init__(self, sagemaker_session, **kwargs):
        super(DummyFrameworkModel, self).__init__(
            MODEL_DATA,
            MODEL_IMAGE,
            INSTANCE_TYPE,
            ROLE,
            ENTRY_POINT,
            sagemaker_session=sagemaker_session,
            **kwargs
        )

    def create_predictor(self, endpoint_name):
        return None

    def prepare_container_def(self, instance_type):
        return MODEL_CONTAINER_DEF


@pytest.fixture(autouse=True)
def mock_create_tar_file():
    with patch("sagemaker.utils.create_tar_file", MagicMock()) as create_tar_file:
        yield create_tar_file


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
    )
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    sms.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=DESCRIBE_TRAINING_JOB_RESULT
    )
    sms.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    sms.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    sms.sagemaker_client.list_tags = Mock(return_value=LIST_TAGS_RESULT)
    return sms


def test_framework_all_init_args(sagemaker_session):
    f = DummyFramework(
        "my_script.py",
        role="DummyRole",
        train_instance_count=3,
        train_instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        train_volume_size=123,
        train_volume_kms_key="volumekms",
        train_max_run=456,
        input_mode="inputmode",
        output_path="outputpath",
        output_kms_key="outputkms",
        base_job_name="basejobname",
        tags=[{"foo": "bar"}],
        subnets=["123", "456"],
        security_group_ids=["789", "012"],
        metric_definitions=[{"Name": "validation-rmse", "Regex": "validation-rmse=(\\d+)"}],
        encrypt_inter_container_traffic=True,
    )
    _TrainingJob.start_new(f, "s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args == {
        "input_mode": "inputmode",
        "tags": [{"foo": "bar"}],
        "hyperparameters": {},
        "image": "fakeimage",
        "input_config": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3DataDistributionType": "FullyReplicated",
                        "S3Uri": "s3://mydata",
                    }
                },
            }
        ],
        "output_config": {"KmsKeyId": "outputkms", "S3OutputPath": "outputpath"},
        "vpc_config": {"Subnets": ["123", "456"], "SecurityGroupIds": ["789", "012"]},
        "stop_condition": {"MaxRuntimeInSeconds": 456},
        "role": sagemaker_session.expand_role(),
        "job_name": None,
        "resource_config": {
            "VolumeSizeInGB": 123,
            "InstanceCount": 3,
            "VolumeKmsKeyId": "volumekms",
            "InstanceType": "ml.m4.xlarge",
        },
        "metric_definitions": [{"Name": "validation-rmse", "Regex": "validation-rmse=(\\d+)"}],
        "encrypt_inter_container_traffic": True,
    }


def test_framework_init_s3_entry_point_invalid(sagemaker_session):
    with pytest.raises(ValueError) as error:
        DummyFramework(
            "s3://remote-script-because-im-mistaken",
            role=ROLE,
            sagemaker_session=sagemaker_session,
            train_instance_count=INSTANCE_COUNT,
            train_instance_type=INSTANCE_TYPE,
        )
    assert "Must be a path to a local file" in str(error)


def test_sagemaker_s3_uri_invalid(sagemaker_session):
    with pytest.raises(ValueError) as error:
        t = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            train_instance_count=INSTANCE_COUNT,
            train_instance_type=INSTANCE_TYPE,
        )
        t.fit("thisdoesntstartwiths3")
    assert "must be a valid S3 or FILE URI" in str(error)


def test_sagemaker_model_s3_uri_invalid(sagemaker_session):
    with pytest.raises(ValueError) as error:
        t = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            train_instance_count=INSTANCE_COUNT,
            train_instance_type=INSTANCE_TYPE,
            model_uri="thisdoesntstartwiths3either.tar.gz",
        )
        t.fit("s3://mydata")
    assert "must be a valid S3 or FILE URI" in str(error)


def test_sagemaker_model_file_uri_invalid(sagemaker_session):
    with pytest.raises(ValueError) as error:
        t = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            train_instance_count=INSTANCE_COUNT,
            train_instance_type=INSTANCE_TYPE,
            model_uri="file://notins3.tar.gz",
        )
        t.fit("s3://mydata")
    assert "File URIs are supported in local mode only" in str(error)


def test_sagemaker_model_default_channel_name(sagemaker_session):
    f = DummyFramework(
        entry_point="my_script.py",
        role="DummyRole",
        train_instance_count=3,
        train_instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        model_uri="s3://model-bucket/prefix/model.tar.gz",
    )
    _TrainingJob.start_new(f, {})
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["input_config"] == [
        {
            "ChannelName": "model",
            "InputMode": "File",
            "ContentType": "application/x-sagemaker-model",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3Uri": "s3://model-bucket/prefix/model.tar.gz",
                }
            },
        }
    ]


def test_sagemaker_model_custom_channel_name(sagemaker_session):
    f = DummyFramework(
        entry_point="my_script.py",
        role="DummyRole",
        train_instance_count=3,
        train_instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        model_uri="s3://model-bucket/prefix/model.tar.gz",
        model_channel_name="testModelChannel",
    )
    _TrainingJob.start_new(f, {})
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["input_config"] == [
        {
            "ChannelName": "testModelChannel",
            "InputMode": "File",
            "ContentType": "application/x-sagemaker-model",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3Uri": "s3://model-bucket/prefix/model.tar.gz",
                }
            },
        }
    ]


@patch("time.strftime", return_value=TIMESTAMP)
def test_custom_code_bucket(time, sagemaker_session):
    code_bucket = "codebucket"
    prefix = "someprefix"
    code_location = "s3://{}/{}".format(code_bucket, prefix)
    t = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        code_location=code_location,
    )
    t.fit("s3://bucket/mydata")

    expected_key = "{}/{}/source/sourcedir.tar.gz".format(prefix, JOB_NAME)
    _, s3_args, _ = sagemaker_session.boto_session.resource("s3").Object.mock_calls[0]
    assert s3_args == (code_bucket, expected_key)

    expected_submit_dir = "s3://{}/{}".format(code_bucket, expected_key)
    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    assert train_kwargs["hyperparameters"]["sagemaker_submit_directory"] == json.dumps(
        expected_submit_dir
    )


@patch("time.strftime", return_value=TIMESTAMP)
def test_custom_code_bucket_without_prefix(time, sagemaker_session):
    code_bucket = "codebucket"
    code_location = "s3://{}".format(code_bucket)
    t = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        code_location=code_location,
    )
    t.fit("s3://bucket/mydata")

    expected_key = "{}/source/sourcedir.tar.gz".format(JOB_NAME)
    _, s3_args, _ = sagemaker_session.boto_session.resource("s3").Object.mock_calls[0]
    assert s3_args == (code_bucket, expected_key)

    expected_submit_dir = "s3://{}/{}".format(code_bucket, expected_key)
    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    assert train_kwargs["hyperparameters"]["sagemaker_submit_directory"] == json.dumps(
        expected_submit_dir
    )


def test_invalid_custom_code_bucket(sagemaker_session):
    code_location = "thisllworkright?"
    t = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        code_location=code_location,
    )

    with pytest.raises(ValueError) as error:
        t.fit("s3://bucket/mydata")
    assert "Expecting 's3' scheme" in str(error)


def test_augmented_manifest(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw.fit(
        inputs=s3_input(
            "s3://mybucket/train_manifest",
            s3_data_type="AugmentedManifestFile",
            attribute_names=["foo", "bar"],
        )
    )

    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    s3_data_source = train_kwargs["input_config"][0]["DataSource"]["S3DataSource"]
    assert s3_data_source["S3Uri"] == "s3://mybucket/train_manifest"
    assert s3_data_source["S3DataType"] == "AugmentedManifestFile"
    assert s3_data_source["AttributeNames"] == ["foo", "bar"]


def test_s3_input_mode(sagemaker_session):
    expected_input_mode = "Pipe"
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw.fit(inputs=s3_input("s3://mybucket/train_manifest", input_mode=expected_input_mode))

    actual_input_mode = sagemaker_session.method_calls[1][2]["input_mode"]
    assert actual_input_mode == expected_input_mode


def test_shuffle_config(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw.fit(inputs=s3_input("s3://mybucket/train_manifest", shuffle_config=ShuffleConfig(100)))
    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    channel = train_kwargs["input_config"][0]
    assert channel["ShuffleConfig"]["Seed"] == 100


BASE_HP = {
    "sagemaker_program": json.dumps(SCRIPT_NAME),
    "sagemaker_submit_directory": json.dumps(
        "s3://mybucket/{}/source/sourcedir.tar.gz".format(JOB_NAME)
    ),
    "sagemaker_job_name": json.dumps(JOB_NAME),
}


def test_local_code_location():
    config = {"local": {"local_code": True, "region": "us-west-2"}}
    sms = Mock(
        name="sagemaker_session",
        boto_session=None,
        boto_region_name=REGION,
        config=config,
        local_mode=True,
    )
    t = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sms,
        train_instance_count=1,
        train_instance_type="local",
        base_job_name=IMAGE_NAME,
        hyperparameters={123: [456], "learning_rate": 0.1},
    )

    t.fit("file:///data/file")
    assert t.source_dir == DATA_DIR
    assert t.entry_point == "dummy_script.py"


@patch("time.strftime", return_value=TIMESTAMP)
def test_start_new_convert_hyperparameters_to_str(strftime, sagemaker_session):
    uri = "bucket/mydata"

    t = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        base_job_name=IMAGE_NAME,
        hyperparameters={123: [456], "learning_rate": 0.1},
    )
    t.fit("s3://{}".format(uri))

    expected_hyperparameters = BASE_HP.copy()
    expected_hyperparameters["sagemaker_enable_cloudwatch_metrics"] = "false"
    expected_hyperparameters["sagemaker_container_log_level"] = str(logging.INFO)
    expected_hyperparameters["learning_rate"] = json.dumps(0.1)
    expected_hyperparameters["123"] = json.dumps([456])
    expected_hyperparameters["sagemaker_region"] = '"us-west-2"'

    actual_hyperparameter = sagemaker_session.method_calls[1][2]["hyperparameters"]
    assert actual_hyperparameter == expected_hyperparameters


@patch("time.strftime", return_value=TIMESTAMP)
def test_start_new_wait_called(strftime, sagemaker_session):
    uri = "bucket/mydata"

    t = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
    )

    t.fit("s3://{}".format(uri))

    expected_hyperparameters = BASE_HP.copy()
    expected_hyperparameters["sagemaker_enable_cloudwatch_metrics"] = "false"
    expected_hyperparameters["sagemaker_container_log_level"] = str(logging.INFO)
    expected_hyperparameters["sagemaker_region"] = '"us-west-2"'

    actual_hyperparameter = sagemaker_session.method_calls[1][2]["hyperparameters"]
    assert actual_hyperparameter == expected_hyperparameters
    assert sagemaker_session.wait_for_job.assert_called_once


def test_delete_endpoint(sagemaker_session):
    t = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        container_log_level=logging.INFO,
    )

    class tj(object):
        @property
        def name(self):
            return "myjob"

    t.latest_training_job = tj()

    t.delete_endpoint()

    sagemaker_session.delete_endpoint.assert_called_with("myjob")


def test_delete_endpoint_without_endpoint(sagemaker_session):
    t = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
    )

    with pytest.raises(ValueError) as error:
        t.delete_endpoint()
    assert "Endpoint was not created yet" in str(error)


def test_enable_cloudwatch_metrics(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw.fit(inputs=s3_input("s3://mybucket/train"))

    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    assert train_kwargs["hyperparameters"]["sagemaker_enable_cloudwatch_metrics"]


def test_attach_framework(sagemaker_session):
    returned_job_description = RETURNED_JOB_DESCRIPTION.copy()
    returned_job_description["VpcConfig"] = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=returned_job_description
    )

    framework_estimator = DummyFramework.attach(
        training_job_name="neo", sagemaker_session=sagemaker_session
    )
    assert framework_estimator._current_job_name == "neo"
    assert framework_estimator.latest_training_job.job_name == "neo"
    assert framework_estimator.role == "arn:aws:iam::366:role/SageMakerRole"
    assert framework_estimator.train_instance_count == 1
    assert framework_estimator.train_max_run == 24 * 60 * 60
    assert framework_estimator.input_mode == "File"
    assert framework_estimator.base_job_name == "neo"
    assert framework_estimator.output_path == "s3://place/output/neo"
    assert framework_estimator.output_kms_key == ""
    assert framework_estimator.hyperparameters()["training_steps"] == "100"
    assert framework_estimator.source_dir == "s3://some/sourcedir.tar.gz"
    assert framework_estimator.entry_point == "iris-dnn-classifier.py"
    assert framework_estimator.subnets == ["foo"]
    assert framework_estimator.security_group_ids == ["bar"]
    assert framework_estimator.encrypt_inter_container_traffic is False
    assert framework_estimator.tags == LIST_TAGS_RESULT["Tags"]


def test_attach_without_hyperparameters(sagemaker_session):
    returned_job_description = RETURNED_JOB_DESCRIPTION.copy()
    del returned_job_description["HyperParameters"]

    mock_describe_training_job = Mock(
        name="describe_training_job", return_value=returned_job_description
    )
    sagemaker_session.sagemaker_client.describe_training_job = mock_describe_training_job

    estimator = Estimator.attach(training_job_name="job", sagemaker_session=sagemaker_session)

    assert estimator.hyperparameters() == {}


def test_attach_framework_with_tuning(sagemaker_session):
    returned_job_description = RETURNED_JOB_DESCRIPTION.copy()
    returned_job_description["HyperParameters"]["_tuning_objective_metric"] = "Validation-accuracy"

    mock_describe_training_job = Mock(
        name="describe_training_job", return_value=returned_job_description
    )
    sagemaker_session.sagemaker_client.describe_training_job = mock_describe_training_job

    framework_estimator = DummyFramework.attach(
        training_job_name="neo", sagemaker_session=sagemaker_session
    )
    assert framework_estimator.latest_training_job.job_name == "neo"
    assert framework_estimator.role == "arn:aws:iam::366:role/SageMakerRole"
    assert framework_estimator.train_instance_count == 1
    assert framework_estimator.train_max_run == 24 * 60 * 60
    assert framework_estimator.input_mode == "File"
    assert framework_estimator.base_job_name == "neo"
    assert framework_estimator.output_path == "s3://place/output/neo"
    assert framework_estimator.output_kms_key == ""
    hyper_params = framework_estimator.hyperparameters()
    assert hyper_params["training_steps"] == "100"
    assert hyper_params["_tuning_objective_metric"] == '"Validation-accuracy"'
    assert framework_estimator.source_dir == "s3://some/sourcedir.tar.gz"
    assert framework_estimator.entry_point == "iris-dnn-classifier.py"
    assert framework_estimator.encrypt_inter_container_traffic is False


def test_attach_framework_with_model_channel(sagemaker_session):
    s3_uri = "s3://some/s3/path/model.tar.gz"
    returned_job_description = RETURNED_JOB_DESCRIPTION.copy()
    returned_job_description["InputDataConfig"] = [
        {
            "ChannelName": "model",
            "InputMode": "File",
            "DataSource": {"S3DataSource": {"S3Uri": s3_uri}},
        }
    ]

    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=returned_job_description
    )

    framework_estimator = DummyFramework.attach(
        training_job_name="neo", sagemaker_session=sagemaker_session
    )
    assert framework_estimator.model_uri is s3_uri
    assert framework_estimator.encrypt_inter_container_traffic is False


def test_attach_framework_with_inter_container_traffic_encryption_flag(sagemaker_session):
    returned_job_description = RETURNED_JOB_DESCRIPTION.copy()
    returned_job_description["EnableInterContainerTrafficEncryption"] = True

    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=returned_job_description
    )

    framework_estimator = DummyFramework.attach(
        training_job_name="neo", sagemaker_session=sagemaker_session
    )

    assert framework_estimator.encrypt_inter_container_traffic is True


@patch("time.strftime", return_value=TIMESTAMP)
def test_fit_verify_job_name(strftime, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
        tags=TAGS,
        encrypt_inter_container_traffic=True,
    )
    fw.fit(inputs=s3_input("s3://mybucket/train"))

    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]

    assert train_kwargs["hyperparameters"]["sagemaker_enable_cloudwatch_metrics"]
    assert train_kwargs["image"] == IMAGE_NAME
    assert train_kwargs["input_mode"] == "File"
    assert train_kwargs["tags"] == TAGS
    assert train_kwargs["job_name"] == JOB_NAME
    assert train_kwargs["encrypt_inter_container_traffic"] is True
    assert fw.latest_training_job.name == JOB_NAME


def test_prepare_for_training_unique_job_name_generation(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw._prepare_for_training()
    first_job_name = fw._current_job_name

    sleep(0.1)
    fw._prepare_for_training()
    second_job_name = fw._current_job_name

    assert first_job_name != second_job_name


def test_prepare_for_training_force_name(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        base_job_name="some",
        enable_cloudwatch_metrics=True,
    )
    fw._prepare_for_training(job_name="use_it")
    assert "use_it" == fw._current_job_name


@patch("time.strftime", return_value=TIMESTAMP)
def test_prepare_for_training_force_name_generation(strftime, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        base_job_name="some",
        enable_cloudwatch_metrics=True,
    )
    fw.base_job_name = None
    fw._prepare_for_training()
    assert JOB_NAME == fw._current_job_name


@patch("sagemaker.git_utils.git_clone_repo")
def test_git_support_with_branch_and_commit_succeed(git_clone_repo, sagemaker_session):
    git_clone_repo.side_effect = lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    }
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    entry_point = "entry_point"
    fw = DummyFramework(
        entry_point=entry_point,
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw.fit()
    git_clone_repo.assert_called_once_with(git_config, entry_point, None, [])


@patch("sagemaker.git_utils.git_clone_repo")
def test_git_support_with_branch_succeed(git_clone_repo, sagemaker_session):
    git_clone_repo.side_effect = lambda gitconfig, entrypoint, source_dir, dependencies=None: {
        "entry_point": "/tmp/repo_dir/source_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    }
    git_config = {"repo": GIT_REPO, "branch": BRANCH}
    entry_point = "entry_point"
    fw = DummyFramework(
        entry_point=entry_point,
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw.fit()
    git_clone_repo.assert_called_once_with(git_config, entry_point, None, [])


@patch("sagemaker.git_utils.git_clone_repo")
def test_git_support_with_dependencies_succeed(git_clone_repo, sagemaker_session):
    git_clone_repo.side_effect = lambda gitconfig, entrypoint, source_dir, dependencies: {
        "entry_point": "/tmp/repo_dir/source_dir/entry_point",
        "source_dir": None,
        "dependencies": ["/tmp/repo_dir/foo", "/tmp/repo_dir/foo/bar"],
    }
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    entry_point = "source_dir/entry_point"
    fw = DummyFramework(
        entry_point=entry_point,
        git_config=git_config,
        dependencies=["foo", "foo/bar"],
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw.fit()
    git_clone_repo.assert_called_once_with(git_config, entry_point, None, ["foo", "foo/bar"])


@patch("sagemaker.git_utils.git_clone_repo")
def test_git_support_without_branch_and_commit_succeed(git_clone_repo, sagemaker_session):
    git_clone_repo.side_effect = lambda gitconfig, entrypoint, source_dir, dependencies=None: {
        "entry_point": "/tmp/repo_dir/source_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    }
    git_config = {"repo": GIT_REPO}
    entry_point = "source_dir/entry_point"
    fw = DummyFramework(
        entry_point=entry_point,
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw.fit()
    git_clone_repo.assert_called_once_with(git_config, entry_point, None, [])


def test_git_support_repo_not_provided(sagemaker_session):
    git_config = {"branch": BRANCH, "commit": COMMIT}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        source_dir="source_dir",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    with pytest.raises(ValueError) as error:
        fw.fit()
    assert "Please provide a repo for git_config." in str(error)


def test_git_support_bad_repo_url_format(sagemaker_session):
    git_config = {"repo": "hhttps://github.com/user/repo.git", "branch": BRANCH}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        source_dir="source_dir",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    with pytest.raises(ValueError) as error:
        fw.fit()
    assert "Invalid Git url provided." in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git clone https://github.com/aws/no-such-repo.git /tmp/repo_dir"
    ),
)
def test_git_support_git_clone_fail(sagemaker_session):
    git_config = {"repo": "https://github.com/aws/no-such-repo.git", "branch": BRANCH}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        fw.fit()
    assert "returned non-zero exit status" in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git checkout branch-that-does-not-exist"
    ),
)
def test_git_support_branch_not_exist(sagemaker_session):
    git_config = {"repo": GIT_REPO, "branch": "branch-that-does-not-exist", "commit": COMMIT}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        fw.fit()
    assert "returned non-zero exit status" in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git checkout commit-sha-that-does-not-exist"
    ),
)
def test_git_support_commit_not_exist(sagemaker_session):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": "commit-sha-that-does-not-exist"}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        fw.fit()
    assert "returned non-zero exit status" in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=ValueError("Entry point does not exist in the repo."),
)
def test_git_support_entry_point_not_exist(sagemaker_session):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    fw = DummyFramework(
        entry_point="entry_point_that_does_not_exist",
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    with pytest.raises(ValueError) as error:
        fw.fit()
    assert "Entry point does not exist in the repo." in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=ValueError("Source directory does not exist in the repo."),
)
def test_git_support_source_dir_not_exist(sagemaker_session):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        source_dir="source_dir_that_does_not_exist",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    with pytest.raises(ValueError) as error:
        fw.fit()
    assert "Source directory does not exist in the repo." in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=ValueError("Dependency no-such-dir does not exist in the repo."),
)
def test_git_support_dependencies_not_exist(sagemaker_session):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        source_dir="source_dir",
        dependencies=["foo", "no-such-dir"],
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    with pytest.raises(ValueError) as error:
        fw.fit()
    assert "Dependency", "does not exist in the repo." in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    },
)
def test_git_support_with_username_password_no_2fa(git_clone_repo, sagemaker_session):
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "username": "username",
        "password": "passw0rd!",
    }
    entry_point = "entry_point"
    fw = DummyFramework(
        entry_point=entry_point,
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw.fit()
    git_clone_repo.assert_called_once_with(git_config, entry_point, None, [])
    assert fw.entry_point == "/tmp/repo_dir/entry_point"


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    },
)
def test_git_support_with_token_2fa(git_clone_repo, sagemaker_session):
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "token": "my-token",
        "2FA_enabled": True,
    }
    entry_point = "entry_point"
    fw = DummyFramework(
        entry_point=entry_point,
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw.fit()
    git_clone_repo.assert_called_once_with(git_config, entry_point, None, [])
    assert fw.entry_point == "/tmp/repo_dir/entry_point"


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    },
)
def test_git_support_ssh_no_passphrase_needed(git_clone_repo, sagemaker_session):
    git_config = {"repo": PRIVATE_GIT_REPO_SSH, "branch": PRIVATE_BRANCH, "commit": PRIVATE_COMMIT}
    entry_point = "entry_point"
    fw = DummyFramework(
        entry_point=entry_point,
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    fw.fit()
    git_clone_repo.assert_called_once_with(git_config, entry_point, None, [])
    assert fw.entry_point == "/tmp/repo_dir/entry_point"


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git clone {} {}".format(PRIVATE_GIT_REPO_SSH, REPO_DIR)
    ),
)
def test_git_support_ssh_passphrase_required(git_clone_repo, sagemaker_session):
    git_config = {"repo": PRIVATE_GIT_REPO_SSH, "branch": PRIVATE_BRANCH, "commit": PRIVATE_COMMIT}
    entry_point = "entry_point"
    fw = DummyFramework(
        entry_point=entry_point,
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=True,
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        fw.fit()
    assert "returned non-zero exit status" in str(error)


@patch("time.strftime", return_value=TIMESTAMP)
def test_init_with_source_dir_s3(strftime, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        source_dir="s3://location",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        enable_cloudwatch_metrics=False,
    )
    fw._prepare_for_training()

    expected_hyperparameters = {
        "sagemaker_program": SCRIPT_NAME,
        "sagemaker_job_name": JOB_NAME,
        "sagemaker_enable_cloudwatch_metrics": False,
        "sagemaker_container_log_level": logging.INFO,
        "sagemaker_submit_directory": "s3://location",
        "sagemaker_region": "us-west-2",
    }
    assert fw._hyperparameters == expected_hyperparameters


@patch("sagemaker.estimator.name_from_image", return_value=MODEL_IMAGE)
def test_framework_transformer_creation(name_from_image, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )
    fw.latest_training_job = _TrainingJob(sagemaker_session, JOB_NAME)

    transformer = fw.transformer(INSTANCE_COUNT, INSTANCE_TYPE)

    name_from_image.assert_called_with(MODEL_IMAGE)
    sagemaker_session.create_model.assert_called_with(
        MODEL_IMAGE, ROLE, MODEL_CONTAINER_DEF, None, tags=None
    )

    assert isinstance(transformer, Transformer)
    assert transformer.sagemaker_session == sagemaker_session
    assert transformer.instance_count == INSTANCE_COUNT
    assert transformer.instance_type == INSTANCE_TYPE
    assert transformer.model_name == MODEL_IMAGE
    assert transformer.tags is None
    assert transformer.env == {}


@patch("sagemaker.estimator.name_from_image", return_value=MODEL_IMAGE)
def test_framework_transformer_creation_with_optional_params(name_from_image, sagemaker_session):
    base_name = "foo"
    vpc_config = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        base_job_name=base_name,
        subnets=vpc_config["Subnets"],
        security_group_ids=vpc_config["SecurityGroupIds"],
    )
    fw.latest_training_job = _TrainingJob(sagemaker_session, JOB_NAME)

    strategy = "MultiRecord"
    assemble_with = "Line"
    kms_key = "key"
    accept = "text/csv"
    max_concurrent_transforms = 1
    max_payload = 6
    env = {"FOO": "BAR"}
    new_role = "dummy-model-role"

    transformer = fw.transformer(
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        strategy=strategy,
        assemble_with=assemble_with,
        output_path=OUTPUT_PATH,
        output_kms_key=kms_key,
        accept=accept,
        tags=TAGS,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        volume_kms_key=kms_key,
        env=env,
        role=new_role,
        model_server_workers=1,
    )

    sagemaker_session.create_model.assert_called_with(
        MODEL_IMAGE, new_role, MODEL_CONTAINER_DEF, vpc_config, tags=TAGS
    )
    assert transformer.strategy == strategy
    assert transformer.assemble_with == assemble_with
    assert transformer.output_path == OUTPUT_PATH
    assert transformer.output_kms_key == kms_key
    assert transformer.accept == accept
    assert transformer.max_concurrent_transforms == max_concurrent_transforms
    assert transformer.max_payload == max_payload
    assert transformer.env == env
    assert transformer.base_transform_job_name == base_name
    assert transformer.tags == TAGS
    assert transformer.volume_kms_key == kms_key


def test_ensure_latest_training_job(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )
    fw.latest_training_job = Mock(name="training_job")

    fw._ensure_latest_training_job()


def test_ensure_latest_training_job_failure(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )

    with pytest.raises(ValueError) as e:
        fw._ensure_latest_training_job()
    assert "Estimator is not associated with a training job" in str(e)


def test_estimator_transformer_creation(sagemaker_session):
    estimator = Estimator(
        image_name=IMAGE_NAME,
        role=ROLE,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )
    estimator.latest_training_job = _TrainingJob(sagemaker_session, JOB_NAME)
    sagemaker_session.create_model_from_job.return_value = JOB_NAME

    transformer = estimator.transformer(INSTANCE_COUNT, INSTANCE_TYPE)

    sagemaker_session.create_model_from_job.assert_called_with(JOB_NAME, role=None, tags=None)
    assert isinstance(transformer, Transformer)
    assert transformer.sagemaker_session == sagemaker_session
    assert transformer.instance_count == INSTANCE_COUNT
    assert transformer.instance_type == INSTANCE_TYPE
    assert transformer.model_name == JOB_NAME
    assert transformer.tags is None


def test_estimator_transformer_creation_with_optional_params(sagemaker_session):
    base_name = "foo"
    estimator = Estimator(
        image_name=IMAGE_NAME,
        role=ROLE,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        base_job_name=base_name,
    )
    estimator.latest_training_job = _TrainingJob(sagemaker_session, JOB_NAME)
    sagemaker_session.create_model_from_job.return_value = JOB_NAME

    strategy = "MultiRecord"
    assemble_with = "Line"
    kms_key = "key"
    accept = "text/csv"
    max_concurrent_transforms = 1
    max_payload = 6
    env = {"FOO": "BAR"}

    transformer = estimator.transformer(
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        strategy=strategy,
        assemble_with=assemble_with,
        output_path=OUTPUT_PATH,
        output_kms_key=kms_key,
        accept=accept,
        tags=TAGS,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        env=env,
        role=ROLE,
    )

    sagemaker_session.create_model_from_job.assert_called_with(JOB_NAME, role=ROLE, tags=TAGS)
    assert transformer.strategy == strategy
    assert transformer.assemble_with == assemble_with
    assert transformer.output_path == OUTPUT_PATH
    assert transformer.output_kms_key == kms_key
    assert transformer.accept == accept
    assert transformer.max_concurrent_transforms == max_concurrent_transforms
    assert transformer.max_payload == max_payload
    assert transformer.env == env
    assert transformer.base_transform_job_name == base_name
    assert transformer.tags == TAGS


# _TrainingJob 'utils'
def test_start_new(sagemaker_session):
    training_job = _TrainingJob(sagemaker_session, JOB_NAME)
    hyperparameters = {"mock": "hyperparameters"}
    inputs = "s3://mybucket/train"

    estimator = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters,
    )

    started_training_job = training_job.start_new(estimator, inputs)
    called_args = sagemaker_session.train.call_args

    assert started_training_job.sagemaker_session == sagemaker_session
    assert called_args[1]["hyperparameters"] == hyperparameters
    sagemaker_session.train.assert_called_once()


def test_start_new_not_local_mode_error(sagemaker_session):
    training_job = _TrainingJob(sagemaker_session, JOB_NAME)
    inputs = "file://mybucket/train"

    estimator = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )
    with pytest.raises(ValueError) as error:
        training_job.start_new(estimator, inputs)
        assert "File URIs are supported in local mode only. Please use a S3 URI instead." == str(
            error
        )


def test_container_log_level(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        container_log_level=logging.DEBUG,
    )
    fw.fit(inputs=s3_input("s3://mybucket/train"))

    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    assert train_kwargs["hyperparameters"]["sagemaker_container_log_level"] == "10"


@patch("sagemaker.utils")
def test_same_code_location_keeps_kms_key(utils, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        output_kms_key="kms-key",
    )

    fw.fit(wait=False)

    extra_args = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "kms-key"}
    obj = sagemaker_session.boto_session.resource("s3").Object

    obj.assert_called_with("mybucket", "%s/source/sourcedir.tar.gz" % fw._current_job_name)

    obj().upload_file.assert_called_with(utils.create_tar_file(), ExtraArgs=extra_args)


@patch("sagemaker.utils")
def test_different_code_location_kms_key(utils, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        code_location="s3://another-location",
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        output_kms_key="kms-key",
    )

    fw.fit(wait=False)

    obj = sagemaker_session.boto_session.resource("s3").Object

    obj.assert_called_with("another-location", "%s/source/sourcedir.tar.gz" % fw._current_job_name)

    obj().upload_file.assert_called_with(utils.create_tar_file(), ExtraArgs=None)


@patch("sagemaker.utils")
def test_default_code_location_uses_output_path(utils, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        output_path="s3://output_path",
        train_instance_count=INSTANCE_COUNT,
        train_instance_type=INSTANCE_TYPE,
        output_kms_key="kms-key",
    )

    fw.fit(wait=False)

    obj = sagemaker_session.boto_session.resource("s3").Object

    obj.assert_called_with("output_path", "%s/source/sourcedir.tar.gz" % fw._current_job_name)

    extra_args = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "kms-key"}
    obj().upload_file.assert_called_with(utils.create_tar_file(), ExtraArgs=extra_args)


def test_wait_without_logs(sagemaker_session):
    training_job = _TrainingJob(sagemaker_session, JOB_NAME)

    training_job.wait(False)

    sagemaker_session.wait_for_job.assert_called_once()
    assert not sagemaker_session.logs_for_job.called


def test_wait_with_logs(sagemaker_session):
    training_job = _TrainingJob(sagemaker_session, JOB_NAME)

    training_job.wait()

    sagemaker_session.logs_for_job.assert_called_once()
    assert not sagemaker_session.wait_for_job.called


def test_unsupported_type_in_dict():
    with pytest.raises(ValueError):
        _TrainingJob._format_inputs_to_input_config({"a": 66})


#################################################################################
# Tests for the generic Estimator class

NO_INPUT_TRAIN_CALL = {
    "hyperparameters": {},
    "image": IMAGE_NAME,
    "input_config": None,
    "input_mode": "File",
    "output_config": {"S3OutputPath": OUTPUT_PATH},
    "resource_config": {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": 30,
    },
    "stop_condition": {"MaxRuntimeInSeconds": 86400},
    "tags": None,
    "vpc_config": None,
    "metric_definitions": None,
}

INPUT_CONFIG = [
    {
        "DataSource": {
            "S3DataSource": {
                "S3DataDistributionType": "FullyReplicated",
                "S3DataType": "S3Prefix",
                "S3Uri": "s3://bucket/training-prefix",
            }
        },
        "ChannelName": "train",
    }
]

BASE_TRAIN_CALL = dict(NO_INPUT_TRAIN_CALL)
BASE_TRAIN_CALL.update({"input_config": INPUT_CONFIG})

HYPERPARAMS = {"x": 1, "y": "hello"}
STRINGIFIED_HYPERPARAMS = dict([(x, str(y)) for x, y in HYPERPARAMS.items()])
HP_TRAIN_CALL = dict(BASE_TRAIN_CALL)
HP_TRAIN_CALL.update({"hyperparameters": STRINGIFIED_HYPERPARAMS})


def test_fit_deploy_keep_tags(sagemaker_session):
    tags = [{"Key": "TagtestKey", "Value": "TagtestValue"}]
    estimator = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        tags=tags,
        sagemaker_session=sagemaker_session,
    )

    estimator.fit()

    estimator.deploy(INSTANCE_COUNT, INSTANCE_TYPE)

    variant = [
        {
            "InstanceType": "c4.4xlarge",
            "VariantName": "AllTraffic",
            "ModelName": ANY,
            "InitialVariantWeight": 1,
            "InitialInstanceCount": 1,
        }
    ]

    job_name = estimator._current_job_name
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        job_name, variant, tags, None, True
    )

    sagemaker_session.create_model.assert_called_with(
        ANY,
        "DummyRole",
        {"ModelDataUrl": "s3://bucket/model.tar.gz", "Environment": {}, "Image": "fakeimage"},
        enable_network_isolation=False,
        vpc_config=None,
        tags=tags,
    )


def test_generic_to_fit_no_input(sagemaker_session):
    e = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )

    e.fit()

    sagemaker_session.train.assert_called_once()
    assert len(sagemaker_session.train.call_args[0]) == 0
    args = sagemaker_session.train.call_args[1]
    assert args["job_name"].startswith(IMAGE_NAME)

    args.pop("job_name")
    args.pop("role")

    assert args == NO_INPUT_TRAIN_CALL


def test_generic_to_fit_no_hps(sagemaker_session):
    e = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )

    e.fit({"train": "s3://bucket/training-prefix"})

    sagemaker_session.train.assert_called_once()
    assert len(sagemaker_session.train.call_args[0]) == 0
    args = sagemaker_session.train.call_args[1]
    assert args["job_name"].startswith(IMAGE_NAME)

    args.pop("job_name")
    args.pop("role")

    assert args == BASE_TRAIN_CALL


def test_generic_to_fit_with_hps(sagemaker_session):
    e = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )

    e.set_hyperparameters(**HYPERPARAMS)

    e.fit({"train": "s3://bucket/training-prefix"})

    sagemaker_session.train.assert_called_once()
    assert len(sagemaker_session.train.call_args[0]) == 0
    args = sagemaker_session.train.call_args[1]
    assert args["job_name"].startswith(IMAGE_NAME)

    args.pop("job_name")
    args.pop("role")

    assert args == HP_TRAIN_CALL


def test_generic_to_fit_with_encrypt_inter_container_traffic_flag(sagemaker_session):
    e = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
        encrypt_inter_container_traffic=True,
    )

    e.fit()

    sagemaker_session.train.assert_called_once()
    args = sagemaker_session.train.call_args[1]
    assert args["encrypt_inter_container_traffic"] is True


def test_generic_to_deploy(sagemaker_session):
    e = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )

    e.set_hyperparameters(**HYPERPARAMS)

    e.fit({"train": "s3://bucket/training-prefix"})

    predictor = e.deploy(INSTANCE_COUNT, INSTANCE_TYPE)

    sagemaker_session.train.assert_called_once()
    assert len(sagemaker_session.train.call_args[0]) == 0
    args = sagemaker_session.train.call_args[1]
    assert args["job_name"].startswith(IMAGE_NAME)

    args.pop("job_name")
    args.pop("role")

    assert args == HP_TRAIN_CALL

    sagemaker_session.create_model.assert_called_once()
    args, kwargs = sagemaker_session.create_model.call_args
    assert args[0].startswith(IMAGE_NAME)
    assert args[1] == ROLE
    assert args[2]["Image"] == IMAGE_NAME
    assert args[2]["ModelDataUrl"] == MODEL_DATA
    assert kwargs["vpc_config"] is None

    assert isinstance(predictor, RealTimePredictor)
    assert predictor.endpoint.startswith(IMAGE_NAME)
    assert predictor.sagemaker_session == sagemaker_session


def test_generic_training_job_analytics(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job",
        return_value={
            "TuningJobArn": "arn:aws:sagemaker:us-west-2:968277160000:hyper-parameter-tuning-job/mock-tuner",
            "TrainingStartTime": 1530562991.299,
            "AlgorithmSpecification": {
                "TrainingImage": "some-image-url",
                "TrainingInputMode": "File",
                "MetricDefinitions": [
                    {"Name": "train:loss", "Regex": "train_loss=([0-9]+\\.[0-9]+)"},
                    {"Name": "validation:loss", "Regex": "valid_loss=([0-9]+\\.[0-9]+)"},
                ],
            },
        },
    )

    e = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )

    with pytest.raises(ValueError) as err:  # noqa: F841
        # No training job yet
        a = e.training_job_analytics
        assert a is not None  # This line is never reached

    e.set_hyperparameters(**HYPERPARAMS)
    e.fit({"train": "s3://bucket/training-prefix"})
    a = e.training_job_analytics
    assert a is not None


def test_generic_create_model_vpc_config_override(sagemaker_session):
    vpc_config_a = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    vpc_config_b = {"Subnets": ["foo", "bar"], "SecurityGroupIds": ["baz"]}

    e = Estimator(
        IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, sagemaker_session=sagemaker_session
    )
    e.fit({"train": "s3://bucket/training-prefix"})
    assert e.get_vpc_config() is None
    assert e.create_model().vpc_config is None
    assert e.create_model(vpc_config_override=vpc_config_a).vpc_config == vpc_config_a
    assert e.create_model(vpc_config_override=None).vpc_config is None

    e.subnets = vpc_config_a["Subnets"]
    e.security_group_ids = vpc_config_a["SecurityGroupIds"]
    assert e.get_vpc_config() == vpc_config_a
    assert e.create_model().vpc_config == vpc_config_a
    assert e.create_model(vpc_config_override=vpc_config_b).vpc_config == vpc_config_b
    assert e.create_model(vpc_config_override=None).vpc_config is None

    with pytest.raises(ValueError):
        e.get_vpc_config(vpc_config_override={"invalid"})
    with pytest.raises(ValueError):
        e.create_model(vpc_config_override={"invalid"})


def test_generic_deploy_vpc_config_override(sagemaker_session):
    vpc_config_a = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    vpc_config_b = {"Subnets": ["foo", "bar"], "SecurityGroupIds": ["baz"]}

    e = Estimator(
        IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, sagemaker_session=sagemaker_session
    )
    e.fit({"train": "s3://bucket/training-prefix"})
    e.deploy(INSTANCE_COUNT, INSTANCE_TYPE)
    assert sagemaker_session.create_model.call_args_list[0][1]["vpc_config"] is None

    e.subnets = vpc_config_a["Subnets"]
    e.security_group_ids = vpc_config_a["SecurityGroupIds"]
    e.deploy(INSTANCE_COUNT, INSTANCE_TYPE)
    assert sagemaker_session.create_model.call_args_list[1][1]["vpc_config"] == vpc_config_a

    e.deploy(INSTANCE_COUNT, INSTANCE_TYPE, vpc_config_override=vpc_config_b)
    assert sagemaker_session.create_model.call_args_list[2][1]["vpc_config"] == vpc_config_b

    e.deploy(INSTANCE_COUNT, INSTANCE_TYPE, vpc_config_override=None)
    assert sagemaker_session.create_model.call_args_list[3][1]["vpc_config"] is None


def test_generic_deploy_accelerator_type(sagemaker_session):
    e = Estimator(
        IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, sagemaker_session=sagemaker_session
    )
    e.fit({"train": "s3://bucket/training-prefix"})
    e.deploy(INSTANCE_COUNT, INSTANCE_TYPE, ACCELERATOR_TYPE)

    args = e.sagemaker_session.endpoint_from_production_variants.call_args[0]
    assert args[0].startswith(IMAGE_NAME)
    assert args[1][0]["AcceleratorType"] == ACCELERATOR_TYPE
    assert args[1][0]["InitialInstanceCount"] == INSTANCE_COUNT
    assert args[1][0]["InstanceType"] == INSTANCE_TYPE


def test_deploy_with_update_endpoint(sagemaker_session):
    estimator = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )
    estimator.set_hyperparameters(**HYPERPARAMS)
    estimator.fit({"train": "s3://bucket/training-prefix"})
    endpoint_name = "endpoint-name"
    estimator.deploy(
        INSTANCE_COUNT, INSTANCE_TYPE, endpoint_name=endpoint_name, update_endpoint=True
    )

    update_endpoint_args = sagemaker_session.update_endpoint.call_args[0]
    assert update_endpoint_args[0] == endpoint_name
    assert update_endpoint_args[1].startWith(IMAGE_NAME)

    sagemaker_session.create_endpoint.assert_not_called()


def test_deploy_with_model_name(sagemaker_session):
    estimator = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )
    estimator.set_hyperparameters(**HYPERPARAMS)
    estimator.fit({"train": "s3://bucket/training-prefix"})
    model_name = "model-name"
    estimator.deploy(INSTANCE_COUNT, INSTANCE_TYPE, model_name=model_name)

    sagemaker_session.create_model.assert_called_once()
    args, kwargs = sagemaker_session.create_model.call_args
    assert args[0] == model_name


def test_deploy_with_no_model_name(sagemaker_session):
    estimator = Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )
    estimator.set_hyperparameters(**HYPERPARAMS)
    estimator.fit({"train": "s3://bucket/training-prefix"})
    estimator.deploy(INSTANCE_COUNT, INSTANCE_TYPE)

    sagemaker_session.create_model.assert_called_once()
    args, kwargs = sagemaker_session.create_model.call_args
    assert args[0].startswith(IMAGE_NAME)


@patch("sagemaker.estimator.LocalSession")
@patch("sagemaker.estimator.Session")
def test_local_mode(session_class, local_session_class):
    local_session = Mock()
    local_session.local_mode = True

    session = Mock()
    session.local_mode = False

    local_session_class.return_value = local_session
    session_class.return_value = session

    e = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, "local")
    print(e.sagemaker_session.local_mode)
    assert e.sagemaker_session.local_mode is True

    e2 = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, "local_gpu")
    assert e2.sagemaker_session.local_mode is True

    e3 = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE)
    assert e3.sagemaker_session.local_mode is False


@patch("sagemaker.estimator.LocalSession")
def test_distributed_gpu_local_mode(LocalSession):
    with pytest.raises(RuntimeError):
        Estimator(IMAGE_NAME, ROLE, 3, "local_gpu", output_path=OUTPUT_PATH)


@patch("sagemaker.estimator.LocalSession")
def test_local_mode_file_output_path(local_session_class):
    local_session = Mock()
    local_session.local_mode = True
    local_session_class.return_value = local_session

    e = Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, "local", output_path="file:///tmp/model/")
    assert e.output_path == "file:///tmp/model/"


@patch("sagemaker.estimator.Session")
def test_file_output_path_not_supported_outside_local_mode(session_class):
    session = Mock()
    session.local_mode = False
    session_class.return_value = session

    with pytest.raises(RuntimeError):
        Estimator(IMAGE_NAME, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, output_path="file:///tmp/model")


def test_prepare_init_params_from_job_description_with_image_training_job():

    init_params = EstimatorBase._prepare_init_params_from_job_description(
        job_details=RETURNED_JOB_DESCRIPTION
    )

    assert init_params["role"] == "arn:aws:iam::366:role/SageMakerRole"
    assert init_params["train_instance_count"] == 1
    assert init_params["image"] == "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-other-py2-cpu:1.0.4"


def test_prepare_init_params_from_job_description_with_algorithm_training_job():

    algorithm_job_description = RETURNED_JOB_DESCRIPTION.copy()
    algorithm_job_description["AlgorithmSpecification"] = {
        "TrainingInputMode": "File",
        "AlgorithmName": "arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
    }

    init_params = EstimatorBase._prepare_init_params_from_job_description(
        job_details=algorithm_job_description
    )

    assert init_params["role"] == "arn:aws:iam::366:role/SageMakerRole"
    assert init_params["train_instance_count"] == 1
    assert (
        init_params["algorithm_arn"]
        == "arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees"
    )


def test_prepare_init_params_from_job_description_with_invalid_training_job():

    invalid_job_description = RETURNED_JOB_DESCRIPTION.copy()
    invalid_job_description["AlgorithmSpecification"] = {"TrainingInputMode": "File"}

    with pytest.raises(RuntimeError) as error:
        EstimatorBase._prepare_init_params_from_job_description(job_details=invalid_job_description)
        assert "Invalid AlgorithmSpecification" in str(error)


def test_prepare_for_training_with_base_name(sagemaker_session):
    estimator = Estimator(
        image_name="some-image",
        role="some_image",
        train_instance_count=1,
        train_instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
    )

    estimator._prepare_for_training()
    assert "base_job_name" in estimator._current_job_name


def test_prepare_for_training_with_name_based_on_image(sagemaker_session):
    estimator = Estimator(
        image_name="some-image",
        role="some_image",
        train_instance_count=1,
        train_instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    estimator._prepare_for_training()
    assert "some-image" in estimator._current_job_name


@patch("sagemaker.algorithm.AlgorithmEstimator.validate_train_spec", Mock())
@patch("sagemaker.algorithm.AlgorithmEstimator._parse_hyperparameters", Mock(return_value={}))
def test_prepare_for_training_with_name_based_on_algorithm(sagemaker_session):
    estimator = AlgorithmEstimator(
        algorithm_arn="arn:aws:sagemaker:us-west-2:1234:algorithm/scikit-decision-trees-1542410022",
        role="some_image",
        train_instance_count=1,
        train_instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    estimator._prepare_for_training()
    assert "scikit-decision-trees-1542410022" in estimator._current_job_name


@patch(
    "sagemaker.estimator.Estimator.fit",
    Mock(
        side_effect=ClientError(
            error_response={
                "Error": {
                    "Code": 403,
                    "Message": '"EnableInterContainerTrafficEncryption" and '
                    '"VpcConfig" must be provided together',
                }
            },
            operation_name="Unit Test",
        )
    ),
)
def test_encryption_flag_in_non_vpc_mode_invalid(sagemaker_session):
    image_name = registry("us-west-2") + "/factorization-machines:1"
    with pytest.raises(ClientError) as error:
        estimator = Estimator(
            image_name=image_name,
            role="SageMakerRole",
            train_instance_count=1,
            train_instance_type="ml.c4.xlarge",
            sagemaker_session=sagemaker_session,
            base_job_name="test-non-vpc-encryption",
            encrypt_inter_container_traffic=True,
        )
        estimator.fit()
    assert (
        '"EnableInterContainerTrafficEncryption" and "VpcConfig" must be provided together'
        in str(error)
    )
