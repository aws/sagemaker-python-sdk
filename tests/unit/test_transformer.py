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
from mock import MagicMock, Mock, patch, PropertyMock

from sagemaker.transformer import _TransformJob, Transformer
from sagemaker.workflow.pipeline_context import PipelineSession, _PipelineConfig
from sagemaker.inputs import BatchDataCaptureConfig

from tests.integ import test_local_mode
from tests.unit import SAGEMAKER_CONFIG_TRANSFORM_JOB

ROLE = "DummyRole"
REGION = "us-west-2"

MODEL_NAME = "model"
IMAGE_URI = "image-for-model"
JOB_NAME = "job"

INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.m4.xlarge"
KMS_KEY_ID = "kms-key-id"

S3_DATA_TYPE = "S3Prefix"
S3_BUCKET = "bucket"
DATA = "s3://{}/input-data".format(S3_BUCKET)
OUTPUT_PATH = "s3://{}/output".format(S3_BUCKET)

TIMESTAMP = "2018-07-12"

INIT_PARAMS = {
    "model_name": MODEL_NAME,
    "instance_count": INSTANCE_COUNT,
    "instance_type": INSTANCE_TYPE,
    "base_transform_job_name": JOB_NAME,
}

MODEL_DESC_PRIMARY_CONTAINER = {"PrimaryContainer": {"Image": IMAGE_URI}}

MODEL_DESC_CONTAINERS_ONLY = {"Containers": [{"Image": IMAGE_URI}]}

MOCKED_PIPELINE_CONFIG = _PipelineConfig(
    "test-pipeline", "test-training-step", "code-hash-0123456789", "config-hash-0123456789"
)


@pytest.fixture(autouse=True)
def mock_create_tar_file():
    with patch("sagemaker.utils.create_tar_file", MagicMock()) as create_tar_file:
        yield create_tar_file


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session")
    session = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        local_mode=False,
        default_bucket_prefix=None,
    )
    # For tests which doesn't verify config file injection, operate with empty config
    session.sagemaker_config = {}
    return session


@pytest.fixture()
def pipeline_session():
    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    client_mock.describe_model.return_value = {"PrimaryContainer": {"Image": IMAGE_URI}}
    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value=ROLE)
    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock
    session_mock = Mock(region_name=REGION)
    session_mock.resource.return_value = resource_mock
    session_mock.client.return_value = client_mock
    return PipelineSession(
        boto_session=session_mock, sagemaker_client=client_mock, default_bucket=S3_BUCKET
    )


@pytest.fixture()
def transformer(sagemaker_session):
    return Transformer(
        MODEL_NAME,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
        volume_kms_key=KMS_KEY_ID,
    )


@patch("sagemaker.transformer._TransformJob.start_new")
def test_transform_with_sagemaker_config_injection(start_new_job, sagemaker_session):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRANSFORM_JOB

    transformer = Transformer(
        MODEL_NAME,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )
    # volume kms key and output kms key are inserted from the config
    assert (
        transformer.volume_kms_key
        == SAGEMAKER_CONFIG_TRANSFORM_JOB["SageMaker"]["TransformJob"]["TransformResources"][
            "VolumeKmsKeyId"
        ]
    )
    assert (
        transformer.output_kms_key
        == SAGEMAKER_CONFIG_TRANSFORM_JOB["SageMaker"]["TransformJob"]["TransformOutput"][
            "KmsKeyId"
        ]
    )

    content_type = "text/csv"
    compression = "Gzip"
    split = "Line"
    input_filter = "$.feature"
    output_filter = "$['sagemaker_output', 'id']"
    join_source = "Input"
    experiment_config = {
        "ExperimentName": "exp",
        "TrialName": "t",
        "TrialComponentDisplayName": "tc",
    }
    model_client_config = {"InvocationsTimeoutInSeconds": 60, "InvocationsMaxRetries": 2}
    batch_data_capture_config = BatchDataCaptureConfig(
        destination_s3_uri=OUTPUT_PATH, generate_inference_id=False
    )
    transformer.transform(
        DATA,
        S3_DATA_TYPE,
        content_type=content_type,
        compression_type=compression,
        split_type=split,
        job_name=JOB_NAME,
        input_filter=input_filter,
        output_filter=output_filter,
        join_source=join_source,
        experiment_config=experiment_config,
        model_client_config=model_client_config,
        batch_data_capture_config=batch_data_capture_config,
    )

    assert transformer._current_job_name == JOB_NAME
    assert transformer.output_path == OUTPUT_PATH
    start_new_job.assert_called_once_with(
        transformer,
        DATA,
        S3_DATA_TYPE,
        content_type,
        compression,
        split,
        input_filter,
        output_filter,
        join_source,
        experiment_config,
        model_client_config,
        batch_data_capture_config,
    )
    # KmsKeyId in BatchDataCapture will be inserted from the config
    assert (
        batch_data_capture_config.kms_key_id
        == SAGEMAKER_CONFIG_TRANSFORM_JOB["SageMaker"]["TransformJob"]["DataCaptureConfig"][
            "KmsKeyId"
        ]
    )


def test_delete_model(sagemaker_session):
    transformer = Transformer(
        MODEL_NAME, INSTANCE_COUNT, INSTANCE_TYPE, sagemaker_session=sagemaker_session
    )
    transformer.delete_model()
    sagemaker_session.delete_model.assert_called_with(MODEL_NAME)


def test_transformer_fails_without_model():
    transformer = Transformer(
        model_name="remote-model",
        sagemaker_session=test_local_mode.LocalNoS3Session(),
        instance_type="local",
        instance_count=1,
    )

    with pytest.raises(ValueError) as error:

        transformer.transform("empty-data")

    assert (
        str(error.value) == "Failed to fetch model information for remote-model. "
        "Please ensure that the model exists. "
        "Local instance types require locally created models."
    )


def test_transformer_init(sagemaker_session):
    transformer = Transformer(
        MODEL_NAME, INSTANCE_COUNT, INSTANCE_TYPE, sagemaker_session=sagemaker_session
    )

    assert transformer.model_name == MODEL_NAME
    assert transformer.instance_count == INSTANCE_COUNT
    assert transformer.instance_type == INSTANCE_TYPE
    assert transformer.sagemaker_session == sagemaker_session

    assert transformer._current_job_name is None
    assert transformer.latest_transform_job is None
    assert transformer._reset_output_path is False


def test_transformer_init_optional_params(sagemaker_session):
    strategy = "MultiRecord"
    assemble_with = "Line"
    accept = "text/csv"
    max_concurrent_transforms = 100
    max_payload = 100
    tags = {"Key": "foo", "Value": "bar"}
    env = {"FOO": "BAR"}

    transformer = Transformer(
        MODEL_NAME,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        strategy=strategy,
        assemble_with=assemble_with,
        output_path=OUTPUT_PATH,
        output_kms_key=KMS_KEY_ID,
        accept=accept,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        tags=tags,
        env=env,
        base_transform_job_name=JOB_NAME,
        sagemaker_session=sagemaker_session,
        volume_kms_key=KMS_KEY_ID,
    )

    assert transformer.model_name == MODEL_NAME
    assert transformer.strategy == strategy
    assert transformer.env == env
    assert transformer.output_path == OUTPUT_PATH
    assert transformer.output_kms_key == KMS_KEY_ID
    assert transformer.accept == accept
    assert transformer.assemble_with == assemble_with
    assert transformer.instance_count == INSTANCE_COUNT
    assert transformer.instance_type == INSTANCE_TYPE
    assert transformer.volume_kms_key == KMS_KEY_ID
    assert transformer.max_concurrent_transforms == max_concurrent_transforms
    assert transformer.max_payload == max_payload
    assert transformer.tags == tags
    assert transformer.base_transform_job_name == JOB_NAME


@patch("sagemaker.transformer._TransformJob.start_new")
def test_transform_with_all_params(start_new_job, transformer):
    content_type = "text/csv"
    compression = "Gzip"
    split = "Line"
    input_filter = "$.feature"
    output_filter = "$['sagemaker_output', 'id']"
    join_source = "Input"
    experiment_config = {
        "ExperimentName": "exp",
        "TrialName": "t",
        "TrialComponentDisplayName": "tc",
    }
    model_client_config = {"InvocationsTimeoutInSeconds": 60, "InvocationsMaxRetries": 2}
    batch_data_capture_config = BatchDataCaptureConfig(
        destination_s3_uri=OUTPUT_PATH, kms_key_id=KMS_KEY_ID, generate_inference_id=False
    )

    transformer.transform(
        DATA,
        S3_DATA_TYPE,
        content_type=content_type,
        compression_type=compression,
        split_type=split,
        job_name=JOB_NAME,
        input_filter=input_filter,
        output_filter=output_filter,
        join_source=join_source,
        experiment_config=experiment_config,
        model_client_config=model_client_config,
        batch_data_capture_config=batch_data_capture_config,
    )

    assert transformer._current_job_name == JOB_NAME
    assert transformer.output_path == OUTPUT_PATH
    start_new_job.assert_called_once_with(
        transformer,
        DATA,
        S3_DATA_TYPE,
        content_type,
        compression,
        split,
        input_filter,
        output_filter,
        join_source,
        experiment_config,
        model_client_config,
        batch_data_capture_config,
    )


@patch("sagemaker.transformer.name_from_base")
@patch("sagemaker.transformer._TransformJob.start_new")
def test_transform_with_base_job_name_provided(start_new_job, name_from_base, transformer):
    base_name = "base-job-name"
    full_name = "{}-{}".format(base_name, TIMESTAMP)

    transformer.base_transform_job_name = base_name
    name_from_base.return_value = full_name

    transformer.transform(DATA)

    name_from_base.assert_called_once_with(base_name)
    assert transformer._current_job_name == full_name


@patch("sagemaker.transformer.Transformer._retrieve_base_name", return_value=IMAGE_URI)
@patch("sagemaker.transformer.name_from_base")
@patch("sagemaker.transformer._TransformJob.start_new")
def test_transform_with_base_name(start_new_job, name_from_base, retrieve_base_name, transformer):
    full_name = "{}-{}".format(IMAGE_URI, TIMESTAMP)
    name_from_base.return_value = full_name

    transformer.transform(DATA)

    retrieve_base_name.assert_called_once_with()
    name_from_base.assert_called_once_with(IMAGE_URI)
    assert transformer._current_job_name == full_name


@patch("sagemaker.transformer.Transformer._retrieve_image_uri", return_value=IMAGE_URI)
@patch("sagemaker.transformer.name_from_base")
@patch("sagemaker.transformer._TransformJob.start_new")
def test_transform_with_job_name_based_on_image(
    start_new_job, name_from_base, retrieve_image_uri, transformer
):
    full_name = "{}-{}".format(IMAGE_URI, TIMESTAMP)
    name_from_base.return_value = full_name

    transformer.transform(DATA)

    retrieve_image_uri.assert_called_once_with()
    name_from_base.assert_called_once_with(IMAGE_URI)
    assert transformer._current_job_name == full_name


@pytest.mark.parametrize("model_desc", [MODEL_DESC_PRIMARY_CONTAINER, MODEL_DESC_CONTAINERS_ONLY])
@patch("sagemaker.transformer.name_from_base")
@patch("sagemaker.transformer._TransformJob.start_new")
def test_transform_with_job_name_based_on_containers(
    start_new_job, name_from_base, model_desc, transformer
):
    transformer.sagemaker_session.sagemaker_client.describe_model.return_value = model_desc

    full_name = "{}-{}".format(IMAGE_URI, TIMESTAMP)
    name_from_base.return_value = full_name

    transformer.transform(DATA)

    transformer.sagemaker_session.sagemaker_client.describe_model.assert_called_once_with(
        ModelName=MODEL_NAME
    )
    name_from_base.assert_called_once_with(IMAGE_URI)
    assert transformer._current_job_name == full_name


@pytest.mark.parametrize(
    "model_desc", [{"PrimaryContainer": dict()}, {"Containers": [dict()]}, dict()]
)
@patch("sagemaker.transformer.name_from_base")
@patch("sagemaker.transformer._TransformJob.start_new")
def test_transform_with_job_name_based_on_model_name(
    start_new_job, name_from_base, model_desc, transformer
):
    transformer.sagemaker_session.sagemaker_client.describe_model.return_value = model_desc

    full_name = "{}-{}".format(MODEL_NAME, TIMESTAMP)
    name_from_base.return_value = full_name

    transformer.transform(DATA)

    transformer.sagemaker_session.sagemaker_client.describe_model.assert_called_once_with(
        ModelName=MODEL_NAME
    )
    name_from_base.assert_called_once_with(MODEL_NAME)
    assert transformer._current_job_name == full_name


@patch("sagemaker.transformer._TransformJob.start_new")
def test_transform_with_generated_output_path(start_new_job, transformer, sagemaker_session):
    transformer.output_path = None
    sagemaker_session.default_bucket.return_value = S3_BUCKET

    transformer.transform(DATA, job_name=JOB_NAME)
    assert transformer.output_path == "s3://{}/{}".format(S3_BUCKET, JOB_NAME)


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
def test_transform_with_generated_output_path_pipeline_config(pipeline_session):
    transformer = Transformer(
        MODEL_NAME,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        sagemaker_session=pipeline_session,
        volume_kms_key=KMS_KEY_ID,
    )
    step_args = transformer.transform(DATA)
    # execute transform.transform() and generate args, S3 paths
    step_args.func(*step_args.func_args, **step_args.func_kwargs)
    expected_path = {
        "Std:Join": {
            "On": "/",
            "Values": [
                "s3:/",
                S3_BUCKET,
                "test-pipeline",
                {"Get": "Execution.PipelineExecutionId"},
                "test-training-step",
            ],
        }
    }
    assert expected_path == transformer.output_path.expr


def test_transform_with_invalid_s3_uri(transformer):
    with pytest.raises(ValueError) as e:
        transformer.transform("not-an-s3-uri")

    assert "Invalid S3 URI" in str(e)


def test_retrieve_image_uri(sagemaker_session, transformer):
    sage_mock = Mock(name="sagemaker_client")
    sage_mock.describe_model.return_value = {"PrimaryContainer": {"Image": IMAGE_URI}}

    sagemaker_session.sagemaker_client = sage_mock

    assert transformer._retrieve_image_uri() == IMAGE_URI


@patch("sagemaker.transformer.Transformer._ensure_last_transform_job")
def test_wait(ensure_last_transform_job, transformer):
    transformer.latest_transform_job = Mock(name="latest_transform_job")

    transformer.wait()

    assert ensure_last_transform_job.called_once
    assert transformer.latest_transform_job.wait.called_once


def test_ensure_last_transform_job_exists(transformer, sagemaker_session):
    transformer.latest_transform_job = _TransformJob(sagemaker_session, "some-transform-job")
    transformer._ensure_last_transform_job()


def test_ensure_last_transform_job_none(transformer):
    transformer.latest_transform_job = None
    with pytest.raises(ValueError) as e:
        transformer._ensure_last_transform_job()

    assert "No transform job available" in str(e)


@patch(
    "sagemaker.transformer.Transformer._prepare_init_params_from_job_description",
    return_value=INIT_PARAMS,
)
def test_attach(prepare_init_params, transformer, sagemaker_session):
    sagemaker_session.sagemaker_client.describe_transform_job = Mock(name="describe_transform_job")
    attached = Transformer.attach(JOB_NAME, sagemaker_session)

    assert prepare_init_params.called_once
    assert attached.latest_transform_job.job_name == JOB_NAME
    assert attached.model_name == MODEL_NAME
    assert attached.instance_count == INSTANCE_COUNT
    assert attached.instance_type == INSTANCE_TYPE


def test_prepare_init_params_from_job_description_missing_keys(transformer):
    job_details = {
        "ModelName": MODEL_NAME,
        "TransformResources": {"InstanceCount": INSTANCE_COUNT, "InstanceType": INSTANCE_TYPE},
        "TransformOutput": {"S3OutputPath": None},
        "TransformJobName": JOB_NAME,
    }

    init_params = transformer._prepare_init_params_from_job_description(job_details)

    assert init_params["model_name"] == MODEL_NAME
    assert init_params["instance_count"] == INSTANCE_COUNT
    assert init_params["instance_type"] == INSTANCE_TYPE


def test_prepare_init_params_from_job_description_all_keys(transformer):
    job_details = {
        "ModelName": MODEL_NAME,
        "TransformResources": {
            "InstanceCount": INSTANCE_COUNT,
            "InstanceType": INSTANCE_TYPE,
            "VolumeKmsKeyId": KMS_KEY_ID,
        },
        "BatchStrategy": None,
        "TransformOutput": {
            "AssembleWith": None,
            "S3OutputPath": None,
            "KmsKeyId": None,
            "Accept": None,
        },
        "MaxConcurrentTransforms": None,
        "MaxPayloadInMB": None,
        "TransformJobName": JOB_NAME,
    }

    init_params = transformer._prepare_init_params_from_job_description(job_details)

    assert init_params["model_name"] == MODEL_NAME
    assert init_params["instance_count"] == INSTANCE_COUNT
    assert init_params["instance_type"] == INSTANCE_TYPE
    assert init_params["volume_kms_key"] == KMS_KEY_ID


# _TransformJob tests
@patch("sagemaker.transformer._TransformJob._load_config")
@patch("sagemaker.transformer._TransformJob._prepare_data_processing")
def test_start_new(prepare_data_processing, load_config, sagemaker_session):
    input_config = "input"
    output_config = "output"
    resource_config = "resource"
    load_config.return_value = {
        "input_config": input_config,
        "output_config": output_config,
        "resource_config": resource_config,
    }

    strategy = "MultiRecord"
    max_concurrent_transforms = 100
    max_payload = 100
    tags = {"Key": "foo", "Value": "bar"}
    env = {"FOO": "BAR"}

    transformer = Transformer(
        MODEL_NAME,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        strategy=strategy,
        output_path=OUTPUT_PATH,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        tags=tags,
        env=env,
        sagemaker_session=sagemaker_session,
    )
    transformer._current_job_name = JOB_NAME

    content_type = "text/csv"
    compression_type = "Gzip"
    split_type = "Line"
    io_filter = "$"
    join_source = "Input"
    model_client_config = {"InvocationsTimeoutInSeconds": 60, "InvocationsMaxRetries": 2}

    batch_data_capture_config = BatchDataCaptureConfig(
        destination_s3_uri=OUTPUT_PATH, kms_key_id=KMS_KEY_ID, generate_inference_id=False
    )

    job = _TransformJob.start_new(
        transformer=transformer,
        data=DATA,
        data_type=S3_DATA_TYPE,
        content_type=content_type,
        compression_type=compression_type,
        split_type=split_type,
        input_filter=io_filter,
        output_filter=io_filter,
        join_source=join_source,
        experiment_config={"ExperimentName": "exp"},
        model_client_config=model_client_config,
        batch_data_capture_config=batch_data_capture_config,
    )

    assert job.sagemaker_session == sagemaker_session
    assert job.job_name == JOB_NAME

    load_config.assert_called_with(
        DATA, S3_DATA_TYPE, content_type, compression_type, split_type, transformer
    )
    prepare_data_processing.assert_called_with(io_filter, io_filter, join_source)

    sagemaker_session.transform.assert_called_with(
        job_name=JOB_NAME,
        model_name=MODEL_NAME,
        strategy=strategy,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        env=env,
        input_config=input_config,
        output_config=output_config,
        resource_config=resource_config,
        experiment_config={"ExperimentName": "exp"},
        model_client_config=model_client_config,
        tags=tags,
        data_processing=prepare_data_processing.return_value,
        batch_data_capture_config=batch_data_capture_config,
    )


def test_load_config(transformer):
    expected_config = {
        "input_config": {
            "DataSource": {"S3DataSource": {"S3DataType": S3_DATA_TYPE, "S3Uri": DATA}}
        },
        "output_config": {"S3OutputPath": OUTPUT_PATH},
        "resource_config": {
            "InstanceCount": INSTANCE_COUNT,
            "InstanceType": INSTANCE_TYPE,
            "VolumeKmsKeyId": KMS_KEY_ID,
        },
    }

    actual_config = _TransformJob._load_config(DATA, S3_DATA_TYPE, None, None, None, transformer)
    assert actual_config == expected_config


def test_format_inputs_to_input_config():
    expected_config = {"DataSource": {"S3DataSource": {"S3DataType": S3_DATA_TYPE, "S3Uri": DATA}}}

    actual_config = _TransformJob._format_inputs_to_input_config(
        DATA, S3_DATA_TYPE, None, None, None
    )
    assert actual_config == expected_config


def test_format_inputs_to_input_config_with_optional_params():
    compression = "Gzip"
    content_type = "text/csv"
    split = "Line"

    expected_config = {
        "DataSource": {"S3DataSource": {"S3DataType": S3_DATA_TYPE, "S3Uri": DATA}},
        "CompressionType": compression,
        "ContentType": content_type,
        "SplitType": split,
    }

    actual_config = _TransformJob._format_inputs_to_input_config(
        DATA, S3_DATA_TYPE, content_type, compression, split
    )
    assert actual_config == expected_config


def test_prepare_output_config():
    config = _TransformJob._prepare_output_config(OUTPUT_PATH, None, None, None)

    assert config == {"S3OutputPath": OUTPUT_PATH}


def test_prepare_output_config_with_optional_params():
    kms_key = "key"
    assemble_with = "Line"
    accept = "text/csv"

    expected_config = {
        "S3OutputPath": OUTPUT_PATH,
        "KmsKeyId": kms_key,
        "AssembleWith": assemble_with,
        "Accept": accept,
    }

    actual_config = _TransformJob._prepare_output_config(
        OUTPUT_PATH, kms_key, assemble_with, accept
    )
    assert actual_config == expected_config


def test_prepare_resource_config():
    config = _TransformJob._prepare_resource_config(INSTANCE_COUNT, INSTANCE_TYPE, KMS_KEY_ID)
    assert config == {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeKmsKeyId": KMS_KEY_ID,
    }


def test_data_processing_config():
    actual_config = _TransformJob._prepare_data_processing("$", None, None)
    assert actual_config == {"InputFilter": "$"}

    actual_config = _TransformJob._prepare_data_processing(None, "$", None)
    assert actual_config == {"OutputFilter": "$"}

    actual_config = _TransformJob._prepare_data_processing(None, None, "Input")
    assert actual_config == {"JoinSource": "Input"}

    actual_config = _TransformJob._prepare_data_processing("$[0]", "$[1]", "Input")
    assert actual_config == {"InputFilter": "$[0]", "OutputFilter": "$[1]", "JoinSource": "Input"}

    actual_config = _TransformJob._prepare_data_processing(None, None, None)
    assert actual_config is None


def test_transform_job_wait(sagemaker_session):
    job = _TransformJob(sagemaker_session, JOB_NAME)
    job.wait()

    assert sagemaker_session.wait_for_transform_job.called_once


@patch("sagemaker.transformer._TransformJob.start_new")
def test_restart_output_path(start_new_job, transformer, sagemaker_session):
    transformer.output_path = None
    sagemaker_session.default_bucket.return_value = S3_BUCKET

    transformer.transform(DATA, job_name="job-1")
    assert transformer.output_path == "s3://{}/{}".format(S3_BUCKET, "job-1")

    transformer.transform(DATA, job_name="job-2")
    assert transformer.output_path == "s3://{}/{}".format(S3_BUCKET, "job-2")


def test_stop_transform_job(sagemaker_session, transformer):
    sagemaker_session.stop_transform_job = Mock(name="stop_transform_job")
    transformer.latest_transform_job = _TransformJob(sagemaker_session, JOB_NAME)

    transformer.stop_transform_job()

    sagemaker_session.stop_transform_job.assert_called_once_with(name=JOB_NAME)


def test_stop_transform_job_no_transform_job(transformer):
    with pytest.raises(ValueError) as e:
        transformer.stop_transform_job()
    assert "No transform job available" in str(e)
