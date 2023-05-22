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

import pytest
import os
from mock import Mock

from sagemaker import TrainingInput
from sagemaker.amazon.amazon_estimator import RecordSet, FileSystemRecordSet
from sagemaker.estimator import Estimator, Framework
from sagemaker.inputs import FileSystemInput
from sagemaker.instance_group import InstanceGroup
from sagemaker.job import _Job
from sagemaker.model import FrameworkModel
from sagemaker.workflow.parameters import ParameterString

BUCKET_NAME = "s3://mybucket/train"
S3_OUTPUT_PATH = "s3://bucket/prefix"
LOCAL_FILE_NAME = "file://local/file"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "c4.4xlarge"
KEEP_ALIVE_PERIOD = 1800
INSTANCE_GROUP = InstanceGroup("group", "ml.c4.xlarge", 1)
VOLUME_SIZE = 1
MAX_RUNTIME = 1
ROLE = "DummyRole"
REGION = "us-west-2"
IMAGE_NAME = "fakeimage"
SCRIPT_NAME = "script.py"
JOB_NAME = "fakejob"
VOLUME_KMS_KEY = "volkmskey"
MODEL_CHANNEL_NAME = "testModelChannel"
MODEL_URI = "s3://bucket/prefix/model.tar.gz"
LOCAL_MODEL_NAME = "file://local/file.tar.gz"
CODE_CHANNEL_NAME = "testCodeChannel"
CODE_URI = "s3://bucket/prefix/code.py"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_NAME)
MODEL_CONTAINER_DEF = {
    "Environment": {
        "SAGEMAKER_PROGRAM": SCRIPT_NAME,
        "SAGEMAKER_SUBMIT_DIRECTORY": "s3://mybucket/mi-2017-10-10-14-14-15/sourcedir.tar.gz",
        "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        "SAGEMAKER_REGION": REGION,
    },
    "Image": IMAGE_NAME,
    "ModelDataUrl": MODEL_URI,
}


@pytest.fixture()
def estimator(sagemaker_session):
    return Estimator(
        IMAGE_NAME,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        volume_size=VOLUME_SIZE,
        max_run=MAX_RUNTIME,
        output_path=S3_OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session")
    mock_session = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        s3_client=None,
        s3_resource=None,
        default_bucket_prefix=None,
    )
    mock_session.expand_role = Mock(name="expand_role", return_value=ROLE)
    # For tests which doesn't verify config file injection, operate with empty config
    mock_session.sagemaker_config = {}
    return mock_session


class DummyFramework(Framework):
    _framework_name = "dummy"

    def training_image_uri(self):
        return IMAGE_NAME

    def create_model(self, role=None, model_server_workers=None):
        return DummyFrameworkModel(self.sagemaker_session, vpc_config=self.get_vpc_config())

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        init_params = super(DummyFramework, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )
        init_params.pop("image_uri", None)
        return init_params


class DummyFrameworkModel(FrameworkModel):
    def __init__(self, sagemaker_session, **kwargs):
        super(DummyFrameworkModel, self).__init__(
            MODEL_URI,
            IMAGE_NAME,
            INSTANCE_TYPE,
            ROLE,
            SCRIPT_NAME,
            sagemaker_session=sagemaker_session,
            **kwargs,
        )

    def prepare_container_def(self, instance_type, accelerator_type=None):
        return MODEL_CONTAINER_DEF


@pytest.fixture()
def framework(sagemaker_session):
    return DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        output_path=S3_OUTPUT_PATH,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )


def test_load_config(estimator):
    inputs = TrainingInput(BUCKET_NAME)

    config = _Job._load_config(inputs, estimator)

    assert config["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] == BUCKET_NAME
    assert config["role"] == ROLE
    assert config["output_config"]["S3OutputPath"] == S3_OUTPUT_PATH
    assert "KmsKeyId" not in config["output_config"]
    assert config["resource_config"]["InstanceCount"] == INSTANCE_COUNT
    assert config["resource_config"]["InstanceType"] == INSTANCE_TYPE
    assert config["resource_config"]["VolumeSizeInGB"] == VOLUME_SIZE
    assert config["stop_condition"]["MaxRuntimeInSeconds"] == MAX_RUNTIME


def test_load_config_with_model_channel(estimator):
    inputs = TrainingInput(BUCKET_NAME)

    estimator.model_uri = MODEL_URI
    estimator.model_channel_name = MODEL_CHANNEL_NAME

    config = _Job._load_config(inputs, estimator)

    assert config["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] == BUCKET_NAME
    assert config["input_config"][1]["DataSource"]["S3DataSource"]["S3Uri"] == MODEL_URI
    assert config["input_config"][1]["ChannelName"] == MODEL_CHANNEL_NAME
    assert config["role"] == ROLE
    assert config["output_config"]["S3OutputPath"] == S3_OUTPUT_PATH
    assert "KmsKeyId" not in config["output_config"]
    assert config["resource_config"]["InstanceCount"] == INSTANCE_COUNT
    assert config["resource_config"]["InstanceType"] == INSTANCE_TYPE
    assert config["resource_config"]["VolumeSizeInGB"] == VOLUME_SIZE
    assert config["stop_condition"]["MaxRuntimeInSeconds"] == MAX_RUNTIME


def test_load_config_with_model_channel_no_inputs(estimator):
    estimator.model_uri = MODEL_URI
    estimator.model_channel_name = MODEL_CHANNEL_NAME

    config = _Job._load_config(inputs=None, estimator=estimator)

    assert config["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] == MODEL_URI
    assert config["input_config"][0]["ChannelName"] == MODEL_CHANNEL_NAME
    assert config["role"] == ROLE
    assert config["output_config"]["S3OutputPath"] == S3_OUTPUT_PATH
    assert "KmsKeyId" not in config["output_config"]
    assert config["resource_config"]["InstanceCount"] == INSTANCE_COUNT
    assert config["resource_config"]["InstanceType"] == INSTANCE_TYPE
    assert config["resource_config"]["VolumeSizeInGB"] == VOLUME_SIZE
    assert config["stop_condition"]["MaxRuntimeInSeconds"] == MAX_RUNTIME


def test_load_config_with_code_channel(framework):
    inputs = TrainingInput(BUCKET_NAME)

    framework.model_uri = MODEL_URI
    framework.model_channel_name = MODEL_CHANNEL_NAME
    framework.code_uri = CODE_URI
    framework._enable_network_isolation = True
    config = _Job._load_config(inputs, framework)

    assert len(config["input_config"]) == 3
    assert config["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] == BUCKET_NAME
    assert config["input_config"][2]["DataSource"]["S3DataSource"]["S3Uri"] == CODE_URI
    assert config["input_config"][2]["ChannelName"] == framework.code_channel_name
    assert config["role"] == ROLE
    assert config["output_config"]["S3OutputPath"] == S3_OUTPUT_PATH
    assert "KmsKeyId" not in config["output_config"]
    assert config["resource_config"]["InstanceCount"] == INSTANCE_COUNT
    assert config["resource_config"]["InstanceType"] == INSTANCE_TYPE


def test_load_config_with_code_channel_no_code_uri(framework):
    inputs = TrainingInput(BUCKET_NAME)

    framework.model_uri = MODEL_URI
    framework.model_channel_name = MODEL_CHANNEL_NAME
    framework._enable_network_isolation = True
    config = _Job._load_config(inputs, framework)

    assert len(config["input_config"]) == 2
    assert config["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] == BUCKET_NAME
    assert config["role"] == ROLE
    assert config["output_config"]["S3OutputPath"] == S3_OUTPUT_PATH
    assert "KmsKeyId" not in config["output_config"]
    assert config["resource_config"]["InstanceCount"] == INSTANCE_COUNT
    assert config["resource_config"]["InstanceType"] == INSTANCE_TYPE


def test_load_config_with_role_as_pipeline_parameter(estimator):
    inputs = TrainingInput(BUCKET_NAME)
    estimator.role = ParameterString(name="Role")

    config = _Job._load_config(inputs, estimator)

    assert config["role"] == estimator.role


def test_format_inputs_none():
    channels = _Job._format_inputs_to_input_config(inputs=None)

    assert channels is None


def test_format_inputs_to_input_config_string():
    inputs = BUCKET_NAME

    channels = _Job._format_inputs_to_input_config(inputs)

    assert channels[0]["DataSource"]["S3DataSource"]["S3Uri"] == inputs


def test_format_inputs_to_input_config_training_input():
    inputs = TrainingInput(BUCKET_NAME)

    channels = _Job._format_inputs_to_input_config(inputs)

    assert (
        channels[0]["DataSource"]["S3DataSource"]["S3Uri"]
        == inputs.config["DataSource"]["S3DataSource"]["S3Uri"]
    )


def test_format_inputs_to_input_config_dict():
    inputs = {"train": BUCKET_NAME}

    channels = _Job._format_inputs_to_input_config(inputs)

    assert channels[0]["DataSource"]["S3DataSource"]["S3Uri"] == inputs["train"]


def test_format_inputs_to_input_config_record_set():
    inputs = RecordSet(s3_data=BUCKET_NAME, num_records=1, feature_dim=1)

    channels = _Job._format_inputs_to_input_config(inputs)

    assert channels[0]["DataSource"]["S3DataSource"]["S3Uri"] == inputs.s3_data
    assert channels[0]["DataSource"]["S3DataSource"]["S3DataType"] == inputs.s3_data_type


def test_format_inputs_to_input_config_file_system_record_set():
    file_system_id = "fs-0a48d2a1"
    file_system_type = "EFS"
    directory_path = "ipinsights"
    num_records = 1
    feature_dim = 1
    records = FileSystemRecordSet(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=directory_path,
        num_records=num_records,
        feature_dim=feature_dim,
    )
    channels = _Job._format_inputs_to_input_config(records)
    assert channels[0]["DataSource"]["FileSystemDataSource"]["DirectoryPath"] == directory_path
    assert channels[0]["DataSource"]["FileSystemDataSource"]["FileSystemId"] == file_system_id
    assert channels[0]["DataSource"]["FileSystemDataSource"]["FileSystemType"] == file_system_type
    assert channels[0]["DataSource"]["FileSystemDataSource"]["FileSystemAccessMode"] == "ro"


def test_format_inputs_to_input_config_list():
    records = RecordSet(s3_data=BUCKET_NAME, num_records=1, feature_dim=1)
    inputs = [records]

    channels = _Job._format_inputs_to_input_config(inputs)

    assert channels[0]["DataSource"]["S3DataSource"]["S3Uri"] == records.s3_data
    assert channels[0]["DataSource"]["S3DataSource"]["S3DataType"] == records.s3_data_type


def test_format_record_set_list_input():
    records = FileSystemRecordSet(
        file_system_id="fs-fd85e556",
        file_system_type="EFS",
        directory_path="ipinsights",
        num_records=100,
        feature_dim=1,
    )
    test_records = FileSystemRecordSet(
        file_system_id="fs-fd85e556",
        file_system_type="EFS",
        directory_path="ipinsights",
        num_records=20,
        feature_dim=1,
        channel="validation",
    )
    inputs = [records, test_records]
    input_dict = _Job._format_record_set_list_input(inputs)
    assert isinstance(input_dict["train"], FileSystemInput)
    assert isinstance(input_dict["validation"], FileSystemInput)


@pytest.mark.parametrize(
    "channel_uri, channel_name, content_type, input_mode",
    [
        [MODEL_URI, MODEL_CHANNEL_NAME, "application/x-sagemaker-model", "File"],
        [CODE_URI, CODE_CHANNEL_NAME, None, None],
    ],
)
def test_prepare_channel(channel_uri, channel_name, content_type, input_mode):
    channel = _Job._prepare_channel(
        [], channel_uri, channel_name, content_type=content_type, input_mode=input_mode
    )

    assert channel["DataSource"]["S3DataSource"]["S3Uri"] == channel_uri
    assert channel["DataSource"]["S3DataSource"]["S3DataDistributionType"] == "FullyReplicated"
    assert channel["DataSource"]["S3DataSource"]["S3DataType"] == "S3Prefix"
    assert channel["ChannelName"] == channel_name
    assert "CompressionType" not in channel
    assert "RecordWrapperType" not in channel

    # The model channel should use all the defaults except InputMode and ContentType
    if channel_name == MODEL_CHANNEL_NAME:
        assert channel["ContentType"] == "application/x-sagemaker-model"
        assert channel["InputMode"] == "File"


def test_prepare_channel_duplicate():
    channels = [
        {
            "ChannelName": MODEL_CHANNEL_NAME,
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://blah/blah",
                }
            },
        }
    ]

    with pytest.raises(ValueError) as error:
        _Job._prepare_channel(channels, MODEL_URI, MODEL_CHANNEL_NAME)

    assert "Duplicate channel {} not allowed.".format(MODEL_CHANNEL_NAME) in str(error)


def test_prepare_channel_with_missing_name():
    with pytest.raises(ValueError) as ex:
        _Job._prepare_channel([], channel_uri=MODEL_URI, channel_name=None)

    assert "Expected a channel name if a channel URI {} is specified".format(MODEL_URI) in str(ex)


def test_prepare_channel_with_missing_uri():
    assert _Job._prepare_channel([], channel_uri=None, channel_name=None) is None


def test_format_inputs_to_input_config_list_not_all_records():
    records = RecordSet(s3_data=BUCKET_NAME, num_records=1, feature_dim=1)
    inputs = [records, "mock"]

    with pytest.raises(ValueError) as ex:
        _Job._format_inputs_to_input_config(inputs)

    assert "List compatible only with RecordSets or FileSystemRecordSets." in str(ex)


def test_format_inputs_to_input_config_list_duplicate_channel():
    record = RecordSet(s3_data=BUCKET_NAME, num_records=1, feature_dim=1)
    inputs = [record, record]

    with pytest.raises(ValueError) as ex:
        _Job._format_inputs_to_input_config(inputs)

    assert "Duplicate channels not allowed." in str(ex)


def test_format_input_single_unamed_channel():
    input_dict = _Job._format_inputs_to_input_config("s3://blah/blah")
    assert input_dict == [
        {
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://blah/blah",
                }
            },
        }
    ]


def test_format_input_multiple_channels():
    input_list = _Job._format_inputs_to_input_config({"a": "s3://blah/blah", "b": "s3://foo/bar"})
    expected = [
        {
            "ChannelName": "a",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://blah/blah",
                }
            },
        },
        {
            "ChannelName": "b",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://foo/bar",
                }
            },
        },
    ]

    # convert back into map for comparison so list order (which is arbitrary) is ignored
    assert {c["ChannelName"]: c for c in input_list} == {c["ChannelName"]: c for c in expected}


def test_format_input_training_input():
    input_dict = _Job._format_inputs_to_input_config(
        TrainingInput(
            "s3://foo/bar",
            distribution="ShardedByS3Key",
            compression="gzip",
            content_type="whizz",
            record_wrapping="bang",
        )
    )
    assert input_dict == [
        {
            "CompressionType": "gzip",
            "ChannelName": "training",
            "ContentType": "whizz",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3DataDistributionType": "ShardedByS3Key",
                    "S3Uri": "s3://foo/bar",
                }
            },
            "RecordWrapperType": "bang",
        }
    ]


def test_dict_of_mixed_input_types():
    input_list = _Job._format_inputs_to_input_config(
        {"a": "s3://foo/bar", "b": TrainingInput("s3://whizz/bang")}
    )

    expected = [
        {
            "ChannelName": "a",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://foo/bar",
                }
            },
        },
        {
            "ChannelName": "b",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://whizz/bang",
                }
            },
        },
    ]

    # convert back into map for comparison so list order (which is arbitrary) is ignored
    assert {c["ChannelName"]: c for c in input_list} == {c["ChannelName"]: c for c in expected}


def test_format_inputs_to_input_config_exception():
    inputs = 1

    with pytest.raises(ValueError):
        _Job._format_inputs_to_input_config(inputs)


def test_unsupported_type_in_dict():
    with pytest.raises(ValueError):
        _Job._format_inputs_to_input_config({"a": 66})


def test_format_string_uri_input_string():
    inputs = BUCKET_NAME

    s3_uri_input = _Job._format_string_uri_input(inputs)

    assert s3_uri_input.config["DataSource"]["S3DataSource"]["S3Uri"] == inputs


def test_format_string_uri_file_system_input():
    file_system_id = "fs-fd85e556"
    file_system_type = "EFS"
    directory_path = "ipinsights"

    file_system_input = FileSystemInput(
        file_system_id=file_system_id,
        file_system_type=file_system_type,
        directory_path=directory_path,
    )

    uri_input = _Job._format_string_uri_input(file_system_input)
    assert uri_input == file_system_input


def test_format_string_uri_input_string_exception():
    inputs = "mybucket/train"

    with pytest.raises(ValueError):
        _Job._format_string_uri_input(inputs)


def test_format_string_uri_input_local_file():
    file_uri_input = _Job._format_string_uri_input(LOCAL_FILE_NAME)

    assert file_uri_input.config["DataSource"]["FileDataSource"]["FileUri"] == LOCAL_FILE_NAME


def test_format_string_uri_input():
    inputs = TrainingInput(BUCKET_NAME)

    s3_uri_input = _Job._format_string_uri_input(inputs)

    assert (
        s3_uri_input.config["DataSource"]["S3DataSource"]["S3Uri"]
        == inputs.config["DataSource"]["S3DataSource"]["S3Uri"]
    )


def test_format_string_uri_input_exception():
    inputs = 1

    with pytest.raises(ValueError):
        _Job._format_string_uri_input(inputs)


def test_format_model_uri_input_string():
    model_uri = MODEL_URI

    model_uri_input = _Job._format_model_uri_input(model_uri)

    assert model_uri_input.config["DataSource"]["S3DataSource"]["S3Uri"] == model_uri


def test_format_model_uri_input_local_file():
    model_uri_input = _Job._format_model_uri_input(LOCAL_MODEL_NAME)

    assert model_uri_input.config["DataSource"]["FileDataSource"]["FileUri"] == LOCAL_MODEL_NAME


def test_format_model_uri_input_exception():
    model_uri = 1

    with pytest.raises(ValueError):
        _Job._format_model_uri_input(model_uri)


def test_prepare_output_config():
    kms_key_id = "kms_key"

    config = _Job._prepare_output_config(BUCKET_NAME, kms_key_id)

    assert config["S3OutputPath"] == BUCKET_NAME
    assert config["KmsKeyId"] == kms_key_id


def test_prepare_output_config_kms_key_none():
    s3_path = BUCKET_NAME
    kms_key_id = None

    config = _Job._prepare_output_config(s3_path, kms_key_id)

    assert config["S3OutputPath"] == s3_path
    assert "KmsKeyId" not in config


def test_prepare_resource_config():
    resource_config = _Job._prepare_resource_config(
        INSTANCE_COUNT, INSTANCE_TYPE, None, VOLUME_SIZE, None, None
    )

    assert resource_config == {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": VOLUME_SIZE,
    }


def test_prepare_resource_config_with_keep_alive_period():
    resource_config = _Job._prepare_resource_config(
        INSTANCE_COUNT, INSTANCE_TYPE, None, VOLUME_SIZE, VOLUME_KMS_KEY, KEEP_ALIVE_PERIOD
    )

    assert resource_config == {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": VOLUME_SIZE,
        "VolumeKmsKeyId": VOLUME_KMS_KEY,
        "KeepAlivePeriodInSeconds": KEEP_ALIVE_PERIOD,
    }


def test_prepare_resource_config_with_volume_kms():
    resource_config = _Job._prepare_resource_config(
        INSTANCE_COUNT, INSTANCE_TYPE, None, VOLUME_SIZE, VOLUME_KMS_KEY, None
    )

    assert resource_config == {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": VOLUME_SIZE,
        "VolumeKmsKeyId": VOLUME_KMS_KEY,
    }


def test_prepare_resource_config_with_heterogeneous_cluster():
    resource_config = _Job._prepare_resource_config(
        None,
        None,
        [InstanceGroup("group1", "ml.c4.xlarge", 1), InstanceGroup("group2", "ml.m4.xlarge", 2)],
        VOLUME_SIZE,
        None,
        None,
    )

    assert resource_config == {
        "InstanceGroups": [
            {"InstanceGroupName": "group1", "InstanceCount": 1, "InstanceType": "ml.c4.xlarge"},
            {"InstanceGroupName": "group2", "InstanceCount": 2, "InstanceType": "ml.m4.xlarge"},
        ],
        "VolumeSizeInGB": VOLUME_SIZE,
    }


def test_prepare_resource_config_with_instance_groups_instance_type_instance_count_set():
    with pytest.raises(ValueError) as error:
        _Job._prepare_resource_config(
            INSTANCE_COUNT,
            INSTANCE_TYPE,
            [INSTANCE_GROUP],
            VOLUME_SIZE,
            None,
            None,
        )
    assert "instance_count and instance_type cannot be set when instance_groups is set" in str(
        error
    )


def test_prepare_resource_config_with_instance_groups_instance_type_instance_count_not_set():
    with pytest.raises(ValueError) as error:
        _Job._prepare_resource_config(
            None,
            None,
            None,
            VOLUME_SIZE,
            None,
            None,
        )
    assert "instance_count and instance_type must be set if instance_groups is not set" in str(
        error
    )


def test_prepare_stop_condition():
    max_run = 1
    max_wait = 2

    stop_condition = _Job._prepare_stop_condition(max_run, max_wait)

    assert stop_condition["MaxRuntimeInSeconds"] == max_run
    assert stop_condition["MaxWaitTimeInSeconds"] == max_wait


def test_prepare_stop_condition_no_wait():
    max_run = 1
    max_wait = None

    stop_condition = _Job._prepare_stop_condition(max_run, max_wait)

    assert stop_condition["MaxRuntimeInSeconds"] == max_run
    assert "MaxWaitTimeInSeconds" not in stop_condition


def test_name(sagemaker_session):
    job = _Job(sagemaker_session, JOB_NAME)
    assert job.name == JOB_NAME
