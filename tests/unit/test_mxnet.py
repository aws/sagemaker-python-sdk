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
from mock import MagicMock, Mock, ANY
from mock import patch
from pkg_resources import parse_version

from sagemaker.fw_utils import UploadedCode
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.metadata_properties import MetadataProperties
from sagemaker.model_metrics import FileSource, MetricsSource, ModelMetrics
from sagemaker.mxnet import defaults
from sagemaker.mxnet import MXNet
from sagemaker.mxnet import MXNetPredictor, MXNetModel

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPT_NAME = "dummy_script.py"
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_NAME)
SERVING_SCRIPT_FILE = "another_dummy_script.py"
MODEL_DATA = "s3://mybucket/model"
ENV = {"DUMMY_ENV_VAR": "dummy_value"}
TIMESTAMP = "2017-11-06-14:14:15.672"
TIME = 1510006209.073025
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c4.4xlarge"
ACCELERATOR_TYPE = "ml.eia.medium"
IMAGE = "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:1.4.0-cpu-py3"
COMPILATION_JOB_NAME = "{}-{}".format("compilation-sagemaker-mxnet", TIMESTAMP)
EDGE_PACKAGING_JOB_NAME = "{}-{}".format("compilation-sagemaker-mxnet", TIMESTAMP)
FRAMEWORK = "mxnet"
ROLE = "Dummy"
REGION = "us-west-2"
GPU = "ml.p2.xlarge"
CPU = "ml.c4.xlarge"
CPU_C5 = "ml.c5.xlarge"
LAUNCH_PS_DISTRIBUTION_DICT = {"parameter_server": {"enabled": True}}

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}

LIST_TAGS_RESULT = {"Tags": [{"Key": "TagtestKey", "Value": "TagtestValue"}]}

EXPERIMENT_CONFIG = {
    "ExperimentName": "exp",
    "TrialName": "trial",
    "TrialComponentDisplayName": "tc",
}

MODEL_PKG_RESPONSE = {"ModelPackageArn": "arn:model-pkg-arn"}

ENV_INPUT = {"env_key1": "env_val1", "env_key2": "env_val2", "env_key3": "env_val3"}

INFERENCE_IMAGE_URI = "inference-uri"


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
    describe_compilation = {
        "ModelArtifacts": {"S3ModelArtifacts": "s3://m/model_c5.tar.gz"},
        "InferenceImage": INFERENCE_IMAGE_URI,
    }
    session.sagemaker_client.create_model_package.side_effect = MODEL_PKG_RESPONSE
    session.sagemaker_client.describe_training_job = Mock(return_value=describe)
    session.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    session.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    session.sagemaker_client.list_tags = Mock(return_value=LIST_TAGS_RESULT)
    session.wait_for_compilation_job = Mock(return_value=describe_compilation)
    session.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    return session


def _is_mms_version(mxnet_version):
    return parse_version(MXNetModel._LOWEST_MMS_VERSION) <= parse_version(mxnet_version)


@pytest.fixture()
def skip_if_mms_version(mxnet_inference_version):
    if _is_mms_version(mxnet_inference_version):
        pytest.skip("Skipping because this version uses MMS")


@pytest.fixture()
def skip_if_not_mms_version(mxnet_inference_version):
    if not _is_mms_version(mxnet_inference_version):
        pytest.skip("Skipping because this version does not use MMS")


def _get_train_args(job_name):
    return {
        "image_uri": IMAGE,
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
        "job_name": job_name,
        "output_config": {"S3OutputPath": "s3://{}/".format(BUCKET_NAME)},
        "resource_config": {
            "InstanceType": "ml.c4.4xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "hyperparameters": {
            "sagemaker_program": json.dumps("dummy_script.py"),
            "sagemaker_container_log_level": str(logging.INFO),
            "sagemaker_job_name": json.dumps(job_name),
            "sagemaker_submit_directory": json.dumps(
                "s3://{}/{}/source/sourcedir.tar.gz".format(BUCKET_NAME, job_name)
            ),
            "sagemaker_region": '"us-west-2"',
        },
        "stop_condition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
        "tags": None,
        "vpc_config": None,
        "metric_definitions": None,
        "environment": None,
        "retry_strategy": None,
        "experiment_config": None,
        "debugger_hook_config": {
            "CollectionConfigurations": [],
            "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        },
        "profiler_rule_configs": [
            {
                "RuleConfigurationName": "ProfilerReport-1510006209",
                "RuleEvaluatorImage": "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:1.4.0-cpu-py3",
                "RuleParameters": {"rule_to_invoke": "ProfilerReport"},
            }
        ],
        "profiler_config": {
            "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        },
    }


def _get_environment(submit_directory, model_url, image_uri):
    return {
        "Environment": {
            "SAGEMAKER_SUBMIT_DIRECTORY": submit_directory,
            "SAGEMAKER_PROGRAM": "dummy_script.py",
            "SAGEMAKER_REGION": "us-west-2",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
        },
        "Image": image_uri,
        "ModelDataUrl": model_url,
    }


def _create_compilation_job(input_shape, output_location):
    return {
        "input_model_config": {
            "DataInputConfig": input_shape,
            "Framework": FRAMEWORK.upper(),
            "S3Uri": "s3://m/m.tar.gz",
        },
        "job_name": COMPILATION_JOB_NAME,
        "output_model_config": {"S3OutputLocation": output_location, "TargetDevice": "ml_c4"},
        "role": ROLE,
        "stop_condition": {"MaxRuntimeInSeconds": 900},
        "tags": None,
    }


@patch("sagemaker.estimator.name_from_base")
@patch("sagemaker.utils.create_tar_file", MagicMock())
def test_create_model(
    name_from_base, sagemaker_session, mxnet_inference_version, mxnet_inference_py_version
):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    base_job_name = "job"

    mx = MXNet(
        entry_point=SCRIPT_NAME,
        source_dir=source_dir,
        framework_version=mxnet_inference_version,
        py_version=mxnet_inference_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        container_log_level=container_log_level,
        base_job_name=base_job_name,
    )

    mx.fit(inputs="s3://mybucket/train", job_name="new_name")

    model_name = "model_name"
    name_from_base.return_value = model_name
    model = mx.create_model()

    assert model.sagemaker_session == sagemaker_session
    assert model.framework_version == mxnet_inference_version
    assert model.py_version == mxnet_inference_py_version
    assert model.entry_point == SCRIPT_NAME
    assert model.role == ROLE
    assert model.name == model_name
    assert model.container_log_level == container_log_level
    assert model.source_dir == source_dir
    assert model.image_uri is None
    assert model.vpc_config is None

    name_from_base.assert_called_with(base_job_name)


def test_create_model_with_optional_params(
    sagemaker_session, mxnet_inference_version, mxnet_inference_py_version
):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    mx = MXNet(
        entry_point=SCRIPT_NAME,
        source_dir=source_dir,
        framework_version=mxnet_inference_version,
        py_version=mxnet_inference_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        container_log_level=container_log_level,
        base_job_name="job",
    )

    mx.fit(inputs="s3://mybucket/train", job_name="new_name")

    new_role = "role"
    model_server_workers = 2
    vpc_config = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    model_name = "model-name"
    model = mx.create_model(
        role=new_role,
        model_server_workers=model_server_workers,
        vpc_config_override=vpc_config,
        entry_point=SERVING_SCRIPT_FILE,
        env=ENV,
        name=model_name,
    )

    assert model.role == new_role
    assert model.model_server_workers == model_server_workers
    assert model.vpc_config == vpc_config
    assert model.entry_point == SERVING_SCRIPT_FILE
    assert model.env == ENV
    assert model.name == model_name


@patch("sagemaker.estimator.name_from_base")
def test_create_model_with_custom_image(name_from_base, sagemaker_session):
    container_log_level = '"logging.INFO"'
    source_dir = "s3://mybucket/source"
    custom_image = "mxnet:2.0"
    base_job_name = "job"

    mx = MXNet(
        entry_point=SCRIPT_NAME,
        source_dir=source_dir,
        framework_version="2.0",
        py_version="py3",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        image_uri=custom_image,
        container_log_level=container_log_level,
        base_job_name=base_job_name,
    )

    mx.fit(inputs="s3://mybucket/train", job_name="new_name")

    model_name = "model_name"
    name_from_base.return_value = model_name
    model = mx.create_model()

    assert model.sagemaker_session == sagemaker_session
    assert model.image_uri == custom_image
    assert model.entry_point == SCRIPT_NAME
    assert model.role == ROLE
    assert model.name == model_name
    assert model.container_log_level == container_log_level
    assert model.source_dir == source_dir

    name_from_base.assert_called_with(base_job_name)


@patch("sagemaker.utils.create_tar_file")
@patch("sagemaker.utils.repack_model")
@patch("time.strftime", return_value=TIMESTAMP)
@patch("time.time", return_value=TIME)
@patch("sagemaker.image_uris.retrieve", return_value=IMAGE)
def test_mxnet(
    retrieve_image_uri,
    time,
    strftime,
    repack_model,
    create_tar_file,
    sagemaker_session,
    mxnet_training_version,
    mxnet_training_py_version,
):
    mx = MXNet(
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_training_version,
        py_version=mxnet_training_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        enable_sagemaker_metrics=False,
    )
    inputs = "s3://mybucket/train"

    mx.fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ["train", "logs_for_job"]
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ["resource"]

    actual_train_args = sagemaker_session.method_calls[0][2]
    job_name = actual_train_args["job_name"]
    expected_train_args = _get_train_args(job_name)
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs
    expected_train_args["experiment_config"] = EXPERIMENT_CONFIG
    expected_train_args["enable_sagemaker_metrics"] = False

    assert actual_train_args == expected_train_args

    model = mx.create_model()

    actual_environment = model.prepare_container_def(GPU)
    submit_directory = actual_environment["Environment"]["SAGEMAKER_SUBMIT_DIRECTORY"]
    model_url = actual_environment["ModelDataUrl"]
    expected_environment = _get_environment(submit_directory, model_url, IMAGE)
    assert actual_environment == expected_environment

    assert "cpu" in model.prepare_container_def(CPU)["Image"]
    predictor = mx.deploy(1, GPU)
    assert isinstance(predictor, MXNetPredictor)
    assert _is_mms_version(mxnet_training_version) ^ (
        create_tar_file.called and not repack_model.called
    )


@patch("sagemaker.utils.repack_model", MagicMock())
@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
@patch("time.strftime", return_value=TIMESTAMP)
@patch("time.time", return_value=TIME)
def test_mxnet_neo(time, strftime, sagemaker_session, neo_mxnet_version):
    mx = MXNet(
        entry_point=SCRIPT_PATH,
        framework_version="1.6",
        py_version="py3",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        base_job_name="sagemaker-mxnet",
    )
    mx.fit()

    input_shape = {"data": [100, 1, 28, 28]}
    output_location = "s3://neo-sdk-test"

    compiled_model = mx.compile_model(
        target_instance_family="ml_c4",
        input_shape=input_shape,
        output_path=output_location,
        framework="mxnet",
        framework_version=neo_mxnet_version,
    )

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == [
        "train",
        "logs_for_job",
        "sagemaker_client.describe_training_job",
        "compile_model",
        "wait_for_compilation_job",
    ]

    expected_compile_model_args = _create_compilation_job(json.dumps(input_shape), output_location)
    actual_compile_model_args = sagemaker_session.method_calls[3][2]
    assert expected_compile_model_args == actual_compile_model_args

    assert compiled_model.image_uri == INFERENCE_IMAGE_URI

    predictor = mx.deploy(1, CPU, use_compiled_model=True)
    assert isinstance(predictor, MXNetPredictor)

    with pytest.raises(Exception) as wrong_target:
        mx.deploy(1, CPU_C5, use_compiled_model=True)
    assert str(wrong_target.value).startswith("No compiled model for")

    # deploy without sagemaker Neo should continue to work
    mx.deploy(1, CPU)


@patch("sagemaker.utils.create_tar_file", MagicMock())
def test_model(
    sagemaker_session, mxnet_inference_version, mxnet_inference_py_version, skip_if_mms_version
):
    model = MXNetModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_inference_version,
        py_version=mxnet_inference_py_version,
        sagemaker_session=sagemaker_session,
    )
    predictor = model.deploy(1, GPU)
    assert isinstance(predictor, MXNetPredictor)

    model_package_name = "test-mxnet-register-model"
    content_types = ["application/json"]
    response_types = ["application/json"]
    inference_instances = ["ml.m4.xlarge"]
    transform_instances = ["ml.m4.xlarget"]

    dummy_metrics_source = MetricsSource(
        content_type="a",
        s3_uri="s3://b/c",
        content_digest="d",
    )
    dummy_file_source = FileSource(
        content_type="a",
        s3_uri="s3://b/c",
        content_digest="d",
    )
    model_metrics = ModelMetrics(
        model_statistics=dummy_metrics_source,
        model_constraints=dummy_metrics_source,
        model_data_statistics=dummy_metrics_source,
        model_data_constraints=dummy_metrics_source,
        bias=dummy_metrics_source,
        bias_pre_training=dummy_metrics_source,
        bias_post_training=dummy_metrics_source,
        explainability=dummy_metrics_source,
    )
    drift_check_baselines = DriftCheckBaselines(
        model_statistics=dummy_metrics_source,
        model_constraints=dummy_metrics_source,
        model_data_statistics=dummy_metrics_source,
        model_data_constraints=dummy_metrics_source,
        bias_config_file=dummy_file_source,
        bias_pre_training_constraints=dummy_metrics_source,
        bias_post_training_constraints=dummy_metrics_source,
        explainability_constraints=dummy_metrics_source,
        explainability_config_file=dummy_file_source,
    )
    model.register(
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_package_name=model_package_name,
        model_metrics=model_metrics,
        marketplace_cert=True,
        approval_status="Approved",
        description="description",
        drift_check_baselines=drift_check_baselines,
    )
    expected_create_model_package_request = {
        "containers": ANY,
        "content_types": content_types,
        "response_types": response_types,
        "inference_instances": inference_instances,
        "transform_instances": transform_instances,
        "model_package_name": model_package_name,
        "model_metrics": model_metrics._to_request_dict(),
        "marketplace_cert": True,
        "approval_status": "Approved",
        "description": "description",
        "drift_check_baselines": drift_check_baselines._to_request_dict(),
    }
    sagemaker_session.create_model_package_from_containers.assert_called_with(
        **expected_create_model_package_request
    )


@patch("sagemaker.utils.create_tar_file", MagicMock())
def test_model_register(
    sagemaker_session, mxnet_inference_version, mxnet_inference_py_version, skip_if_mms_version
):
    model = MXNetModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_inference_version,
        py_version=mxnet_inference_py_version,
        sagemaker_session=sagemaker_session,
    )
    predictor = model.deploy(1, GPU)
    assert isinstance(predictor, MXNetPredictor)

    model_package_name = "test-mxnet-register-model"
    content_types = ["application/json"]
    response_types = ["application/json"]
    inference_instances = ["ml.m4.xlarge"]
    transform_instances = ["ml.m4.xlarget"]
    model.register(
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_package_name=model_package_name,
    )
    sagemaker_session.create_model_package_from_containers.assert_called()


@patch("sagemaker.utils.create_tar_file", MagicMock())
def test_model_register_all_args(
    sagemaker_session,
    mxnet_inference_version,
    mxnet_inference_py_version,
    skip_if_mms_version,
):
    model = MXNetModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_inference_version,
        py_version=mxnet_inference_py_version,
        sagemaker_session=sagemaker_session,
    )
    predictor = model.deploy(1, GPU)
    assert isinstance(predictor, MXNetPredictor)

    model_package_name = "test-mxnet-register-model"
    content_types = ["application/json"]
    response_types = ["application/json"]
    inference_instances = ["ml.m4.xlarge"]
    transform_instances = ["ml.m4.xlarget"]

    dummy_metrics_source = MetricsSource(
        content_type="a",
        s3_uri="s3://b/c",
        content_digest="d",
    )
    dummy_file_source = FileSource(
        content_type="a",
        s3_uri="s3://b/c",
        content_digest="d",
    )
    model_metrics = ModelMetrics(
        model_statistics=dummy_metrics_source,
        model_constraints=dummy_metrics_source,
        model_data_statistics=dummy_metrics_source,
        model_data_constraints=dummy_metrics_source,
        bias=dummy_metrics_source,
        bias_pre_training=dummy_metrics_source,
        bias_post_training=dummy_metrics_source,
        explainability=dummy_metrics_source,
    )
    drift_check_baselines = DriftCheckBaselines(
        model_statistics=dummy_metrics_source,
        model_constraints=dummy_metrics_source,
        model_data_statistics=dummy_metrics_source,
        model_data_constraints=dummy_metrics_source,
        bias_config_file=dummy_file_source,
        bias_pre_training_constraints=dummy_metrics_source,
        bias_post_training_constraints=dummy_metrics_source,
        explainability_constraints=dummy_metrics_source,
        explainability_config_file=dummy_file_source,
    )
    metadata_properties = MetadataProperties(
        commit_id="test-commit-id",
        repository="test-repository",
        generated_by="sagemaker-python-sdk-test",
        project_id="test-project-id",
    )
    model.register(
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_package_name=model_package_name,
        model_metrics=model_metrics,
        metadata_properties=metadata_properties,
        marketplace_cert=True,
        approval_status="Approved",
        description="description",
        drift_check_baselines=drift_check_baselines,
    )
    expected_create_model_package_request = {
        "containers": ANY,
        "content_types": content_types,
        "response_types": response_types,
        "inference_instances": inference_instances,
        "transform_instances": transform_instances,
        "model_package_name": model_package_name,
        "model_metrics": model_metrics._to_request_dict(),
        "metadata_properties": metadata_properties._to_request_dict(),
        "marketplace_cert": True,
        "approval_status": "Approved",
        "description": "description",
        "drift_check_baselines": drift_check_baselines._to_request_dict(),
    }
    sagemaker_session.create_model_package_from_containers.assert_called_with(
        **expected_create_model_package_request
    )


@patch("sagemaker.utils.create_tar_file", MagicMock())
def test_model_custom_serialization(
    sagemaker_session, mxnet_inference_version, mxnet_inference_py_version, skip_if_mms_version
):
    model = MXNetModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_inference_version,
        py_version=mxnet_inference_py_version,
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
    assert isinstance(predictor, MXNetPredictor)
    assert predictor.serializer is custom_serializer
    assert predictor.deserializer is custom_deserializer


@patch("sagemaker.utils.repack_model")
def test_model_mms_version(
    repack_model,
    sagemaker_session,
    mxnet_inference_version,
    mxnet_inference_py_version,
    skip_if_not_mms_version,
):
    model_kms_key = "kms-key"
    model = MXNetModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_inference_version,
        py_version=mxnet_inference_py_version,
        sagemaker_session=sagemaker_session,
        name="test-mxnet-model",
        model_kms_key=model_kms_key,
    )
    predictor = model.deploy(1, GPU)

    repack_model.assert_called_once_with(
        inference_script=SCRIPT_PATH,
        source_directory=None,
        dependencies=[],
        model_uri=MODEL_DATA,
        repacked_model_uri="s3://mybucket/test-mxnet-model/model.tar.gz",
        sagemaker_session=sagemaker_session,
        kms_key=model_kms_key,
    )

    assert model.model_data == MODEL_DATA
    assert model.repacked_model_data == "s3://mybucket/test-mxnet-model/model.tar.gz"
    assert model.uploaded_code == UploadedCode(
        s3_prefix="s3://mybucket/test-mxnet-model/model.tar.gz",
        script_name=os.path.basename(SCRIPT_PATH),
    )
    assert isinstance(predictor, MXNetPredictor)


@patch("sagemaker.fw_utils.tar_and_upload_dir")
@patch("sagemaker.utils.repack_model")
@patch("sagemaker.image_uris.retrieve", return_value=IMAGE)
def test_model_image_accelerator(
    retrieve_image_uri,
    repack_model,
    tar_and_upload,
    sagemaker_session,
    mxnet_eia_version,
    mxnet_eia_py_version,
):
    model = MXNetModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_eia_version,
        py_version=mxnet_eia_py_version,
        sagemaker_session=sagemaker_session,
    )
    container_def = model.prepare_container_def(INSTANCE_TYPE, accelerator_type=ACCELERATOR_TYPE)
    assert container_def["Image"] == IMAGE
    assert _is_mms_version(mxnet_eia_version) ^ (tar_and_upload.called and not repack_model.called)


def test_model_prepare_container_def_no_instance_type_or_image(
    mxnet_inference_version, mxnet_inference_py_version
):
    model = MXNetModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_inference_version,
        py_version=mxnet_inference_py_version,
    )

    with pytest.raises(ValueError) as e:
        model.prepare_container_def()

    expected_msg = "Must supply either an instance type (for choosing CPU vs GPU) or an image URI."
    assert expected_msg in str(e)


def test_attach(sagemaker_session, mxnet_training_version, mxnet_training_py_version):
    if mxnet_training_py_version == "py37":
        training_image = "1.dkr.ecr.us-west-2.amazonaws.com/mxnet-training:{1}-cpu-{0}".format(
            mxnet_training_py_version, mxnet_training_version
        )
    else:
        training_image = (
            "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-{0}-cpu:{1}-cpu-{0}".format(
                mxnet_training_py_version, mxnet_training_version
            )
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

    estimator = MXNet.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == "neo"
    assert estimator.py_version == mxnet_training_py_version
    assert estimator.framework_version == mxnet_training_version
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
    assert estimator.tags == LIST_TAGS_RESULT["Tags"]


def test_attach_old_container(sagemaker_session):
    returned_job_description = {
        "AlgorithmSpecification": {
            "TrainingInputMode": "File",
            "TrainingImage": "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py2-cpu:1.0",
        },
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

    estimator = MXNet.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == "neo"
    assert estimator.py_version == "py2"
    assert estimator.framework_version == "0.12"
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
            "TrainingImage": "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-py2-cpu:1.0.4",
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
        MXNet.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert "didn't use image for requested framework" in str(error)


def test_attach_custom_image(sagemaker_session):
    training_image = "ubuntu:latest"
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

    estimator = MXNet.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert estimator.image_uri == training_image
    assert estimator.training_image_uri() == training_image


def test_estimator_script_mode_dont_launch_parameter_server(sagemaker_session):
    mx = MXNet(
        entry_point=SCRIPT_PATH,
        framework_version="1.3.0",
        py_version="py2",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        distribution={"parameter_server": {"enabled": False}},
    )
    assert mx.hyperparameters().get(MXNet.LAUNCH_PS_ENV_NAME) == "false"


def test_estimator_wrong_version_launch_parameter_server(sagemaker_session):
    with pytest.raises(ValueError) as e:
        MXNet(
            entry_point=SCRIPT_PATH,
            framework_version="1.2.1",
            py_version="py2",
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            distribution=LAUNCH_PS_DISTRIBUTION_DICT,
        )
    assert "The distribution option is valid for only versions 1.3 and higher" in str(e)


@patch("sagemaker.mxnet.estimator.python_deprecation_warning")
def test_estimator_py2_warning(warning, sagemaker_session):
    estimator = MXNet(
        entry_point=SCRIPT_PATH,
        framework_version="1.2.1",
        py_version="py2",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )

    assert estimator.py_version == "py2"
    warning.assert_called_with(estimator._framework_name, defaults.LATEST_PY2_VERSION)


@patch("sagemaker.mxnet.model.python_deprecation_warning")
def test_model_py2_warning(warning, sagemaker_session):
    model = MXNetModel(
        MODEL_DATA,
        role=ROLE,
        entry_point=SCRIPT_PATH,
        framework_version="1.2.1",
        py_version="py2",
        sagemaker_session=sagemaker_session,
    )
    assert model.py_version == "py2"
    warning.assert_called_with(model._framework_name, defaults.LATEST_PY2_VERSION)


def test_create_model_with_custom_hosting_image(sagemaker_session):
    container_log_level = '"logging.INFO"'
    custom_image = "mxnet:2.0"
    custom_hosting_image = "mxnet_hosting:2.0"
    mx = MXNet(
        entry_point=SCRIPT_PATH,
        framework_version="2.0",
        py_version="py3",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        image_uri=custom_image,
        container_log_level=container_log_level,
        base_job_name="job",
    )

    mx.fit(inputs="s3://mybucket/train", job_name="new_name")
    model = mx.create_model(image_uri=custom_hosting_image)

    assert model.image_uri == custom_hosting_image


def test_mx_add_environment_variables(
    sagemaker_session, mxnet_training_version, mxnet_training_py_version
):
    mx = MXNet(
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_training_version,
        py_version=mxnet_training_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        environment=ENV_INPUT,
    )
    assert mx.environment == ENV_INPUT


def test_mx_missing_environment_variables(
    sagemaker_session, mxnet_training_version, mxnet_training_py_version
):
    mx = MXNet(
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_training_version,
        py_version=mxnet_training_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        environment=None,
    )
    assert not mx.environment


def test_mx_enable_sm_metrics(sagemaker_session, mxnet_training_version, mxnet_training_py_version):
    mx = MXNet(
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_training_version,
        py_version=mxnet_training_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        enable_sagemaker_metrics=True,
    )
    assert mx.enable_sagemaker_metrics


def test_mx_disable_sm_metrics(
    sagemaker_session, mxnet_training_version, mxnet_training_py_version
):
    mx = MXNet(
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_training_version,
        py_version=mxnet_training_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        enable_sagemaker_metrics=False,
    )
    assert not mx.enable_sagemaker_metrics


def test_mx_enable_sm_metrics_for_version(
    sagemaker_session, mxnet_training_version, mxnet_training_py_version
):
    mx = MXNet(
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_training_version,
        py_version=mxnet_training_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    version = tuple(int(s) for s in mxnet_training_version.split("."))
    lowest_version = (1, 6, 0)[: len(version)]
    if version >= lowest_version:
        assert mx.enable_sagemaker_metrics
    else:
        assert mx.enable_sagemaker_metrics is None


def test_custom_image_estimator_deploy(
    sagemaker_session, mxnet_training_version, mxnet_training_py_version
):
    custom_image = "mycustomimage:latest"
    mx = MXNet(
        entry_point=SCRIPT_PATH,
        framework_version=mxnet_training_version,
        py_version=mxnet_training_py_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    mx.fit(inputs="s3://mybucket/train", job_name="new_name")
    model = mx.create_model(image_uri=custom_image)
    assert model.image_uri == custom_image
