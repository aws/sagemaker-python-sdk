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
from mock import MagicMock, Mock, patch, ANY

from sagemaker import image_uris
from sagemaker.huggingface import HuggingFace, TrainingCompilerConfig
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.session_settings import SessionSettings

from tests.unit.sagemaker.training_compiler import EC2_GPU_INSTANCE_CLASSES


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPT_PATH = os.path.join(DATA_DIR, "dummy_script.py")
SERVING_SCRIPT_FILE = "another_dummy_script.py"
MODEL_DATA = "s3://some/data.tar.gz"
ENV = {"DUMMY_ENV_VAR": "dummy_value"}
TIMESTAMP = "2017-11-06-14:14:15.672"
TIME = 1510006209.073025
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.p3.2xlarge"
IMAGE_URI = "huggingface"
JOB_NAME = "{}-{}".format(IMAGE_URI, TIMESTAMP)
ROLE = "Dummy"
REGION = "us-east-1"
GPU = "ml.p3.2xlarge"
SUPPORTED_GPU_INSTANCE_CLASSES = {"p3", "p3dn", "g4dn", "p4d", "g5"}
UNSUPPORTED_GPU_INSTANCE_CLASSES = EC2_GPU_INSTANCE_CLASSES - SUPPORTED_GPU_INSTANCE_CLASSES

LIST_TAGS_RESULT = {"Tags": [{"Key": "TagtestKey", "Value": "TagtestValue"}]}

EXPERIMENT_CONFIG = {
    "ExperimentName": "exp",
    "TrialName": "trial",
    "TrialComponentDisplayName": "tc",
    "RunName": "rn",
}


@pytest.fixture(scope="module")
def cpu_instance_type():
    return "ml.m5.xlarge"


@pytest.fixture(name="sagemaker_session", scope="function")
def fixture_sagemaker_session():
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
    session.sagemaker_client.list_tags = Mock(return_value=LIST_TAGS_RESULT)
    session.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session.expand_role = Mock(name="expand_role", return_value=ROLE)
    # For tests which doesn't verify config file injection, operate with empty config
    session.sagemaker_config = {}
    return session


def _get_full_gpu_image_uri(
    version, base_framework_version, instance_type, training_compiler_config, py_version
):
    return image_uris.retrieve(
        "huggingface",
        REGION,
        version=version,
        py_version=py_version,
        instance_type=instance_type,
        image_scope="training",
        base_framework_version=base_framework_version,
        container_version="cu111-ubuntu20.04",
        training_compiler_config=training_compiler_config,
    )


def _create_train_job(
    version, base_framework_version, instance_type, training_compiler_config, py_version
):
    return {
        "image_uri": _get_full_gpu_image_uri(
            version, base_framework_version, instance_type, training_compiler_config, py_version
        ),
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
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "hyperparameters": {
            "sagemaker_program": json.dumps("dummy_script.py"),
            "sagemaker_container_log_level": str(logging.INFO),
            "sagemaker_job_name": json.dumps(JOB_NAME),
            "sagemaker_submit_directory": json.dumps(
                "s3://{}/{}/source/sourcedir.tar.gz".format(BUCKET_NAME, JOB_NAME)
            ),
            "sagemaker_region": '"us-east-1"',
        },
        "stop_condition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
        "tags": None,
        "vpc_config": None,
        "metric_definitions": None,
        "environment": None,
        "retry_strategy": None,
        "experiment_config": EXPERIMENT_CONFIG,
        "debugger_hook_config": {
            "CollectionConfigurations": [],
            "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        },
        "profiler_config": {
            "DisableProfiler": False,
            "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        },
    }


def test_unsupported_BYOC(
    huggingface_training_compiler_version,
    huggingface_training_compiler_tensorflow_version,
    huggingface_training_compiler_tensorflow_py_version,
):
    byoc = (
        f"1.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-trcomp-training:"
        f"2.6.3-"
        f"transformers4.17.0-gpu-"
        f"{huggingface_training_compiler_tensorflow_py_version}-cu112-ubuntu20.04"
    )
    with pytest.raises(ValueError):
        HuggingFace(
            image_uri=byoc,
            py_version=huggingface_training_compiler_tensorflow_py_version,
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            transformers_version=huggingface_training_compiler_version,
            tensorflow_version=huggingface_training_compiler_tensorflow_version,
            enable_sagemaker_metrics=False,
            compiler_config=TrainingCompilerConfig(),
        ).fit()


def test_unsupported_cpu_instance(
    cpu_instance_type,
    huggingface_training_compiler_version,
    huggingface_training_compiler_tensorflow_version,
    huggingface_training_compiler_tensorflow_py_version,
):
    with pytest.raises(ValueError):
        HuggingFace(
            py_version=huggingface_training_compiler_tensorflow_py_version,
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=INSTANCE_COUNT,
            instance_type=cpu_instance_type,
            transformers_version=huggingface_training_compiler_version,
            tensorflow_version=huggingface_training_compiler_tensorflow_version,
            enable_sagemaker_metrics=False,
            compiler_config=TrainingCompilerConfig(),
        ).fit()


@pytest.mark.parametrize("unsupported_gpu_instance_class", UNSUPPORTED_GPU_INSTANCE_CLASSES)
def test_unsupported_gpu_instance(
    unsupported_gpu_instance_class,
    huggingface_training_compiler_version,
    huggingface_training_compiler_tensorflow_version,
    huggingface_training_compiler_tensorflow_py_version,
):
    with pytest.raises(ValueError):
        HuggingFace(
            py_version=huggingface_training_compiler_tensorflow_py_version,
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=INSTANCE_COUNT,
            instance_type=f"ml.{unsupported_gpu_instance_class}.xlarge",
            transformers_version=huggingface_training_compiler_version,
            tensorflow_version=huggingface_training_compiler_tensorflow_version,
            enable_sagemaker_metrics=False,
            compiler_config=TrainingCompilerConfig(),
        ).fit()


def test_unsupported_framework_version(
    huggingface_training_compiler_version,
    huggingface_training_compiler_tensorflow_py_version,
):
    with pytest.raises(ValueError):
        HuggingFace(
            py_version=huggingface_training_compiler_tensorflow_py_version,
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            transformers_version=huggingface_training_compiler_version,
            tensorflow_version=".".join(
                ["99"] * len(huggingface_training_compiler_version.split("."))
            ),
            enable_sagemaker_metrics=False,
            compiler_config=TrainingCompilerConfig(),
        ).fit()


def test_unsupported_framework_mxnet(
    huggingface_training_compiler_version,
    huggingface_training_compiler_tensorflow_py_version,
):
    with pytest.raises(ValueError):
        HuggingFace(
            py_version=huggingface_training_compiler_tensorflow_py_version,
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            transformers_version=huggingface_training_compiler_version,
            mxnet_version=".".join(["99"] * len(huggingface_training_compiler_version.split("."))),
            enable_sagemaker_metrics=False,
            compiler_config=TrainingCompilerConfig(),
        ).fit()


def test_unsupported_python_2(
    huggingface_training_compiler_version,
    huggingface_training_compiler_tensorflow_version,
):
    with pytest.raises(ValueError):
        HuggingFace(
            py_version="py27",
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            transformers_version=huggingface_training_compiler_version,
            tensorflow_version=huggingface_training_compiler_tensorflow_version,
            enable_sagemaker_metrics=False,
            compiler_config=TrainingCompilerConfig(),
        ).fit()


def test_unsupported_distribution(
    huggingface_training_compiler_version,
    huggingface_training_compiler_tensorflow_version,
):
    with pytest.raises(ValueError):
        HuggingFace(
            py_version="py38",
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=2,
            instance_type=INSTANCE_TYPE,
            transformers_version=huggingface_training_compiler_version,
            tensorflow_version=huggingface_training_compiler_tensorflow_version,
            enable_sagemaker_metrics=False,
            compiler_config=TrainingCompilerConfig(),
            distribution={"smdistributed": {"dataparallel": {"enabled": True}}},
        ).fit()

    with pytest.raises(ValueError):
        HuggingFace(
            py_version="py38",
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=2,
            instance_type=INSTANCE_TYPE,
            transformers_version=huggingface_training_compiler_version,
            tensorflow_version=huggingface_training_compiler_tensorflow_version,
            enable_sagemaker_metrics=False,
            compiler_config=TrainingCompilerConfig(),
            distribution={"pytorchxla": {"enabled": True}},
        ).fit()


@patch("sagemaker.utils.repack_model", MagicMock())
@patch("sagemaker.utils.create_tar_file", MagicMock())
@patch("sagemaker.estimator.name_from_base", return_value=JOB_NAME)
@patch("time.time", return_value=TIME)
@pytest.mark.parametrize("instance_class", SUPPORTED_GPU_INSTANCE_CLASSES)
def test_default_compiler_config(
    time,
    name_from_base,
    sagemaker_session,
    huggingface_training_compiler_version,
    huggingface_training_compiler_tensorflow_version,
    instance_class,
    huggingface_training_compiler_tensorflow_py_version,
):
    compiler_config = TrainingCompilerConfig()
    instance_type = f"ml.{instance_class}.xlarge"

    hf = HuggingFace(
        py_version=huggingface_training_compiler_tensorflow_py_version,
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=instance_type,
        transformers_version=huggingface_training_compiler_version,
        tensorflow_version=huggingface_training_compiler_tensorflow_version,
        enable_sagemaker_metrics=False,
        compiler_config=compiler_config,
    )

    inputs = "s3://mybucket/train"

    hf.fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ["train", "logs_for_job"]
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ["resource"]

    expected_train_args = _create_train_job(
        huggingface_training_compiler_version,
        f"tensorflow{huggingface_training_compiler_tensorflow_version}",
        instance_type,
        compiler_config,
        huggingface_training_compiler_tensorflow_py_version,
    )
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs
    expected_train_args["enable_sagemaker_metrics"] = False
    expected_train_args["hyperparameters"][TrainingCompilerConfig.HP_ENABLE_COMPILER] = json.dumps(
        True
    )
    expected_train_args["hyperparameters"][TrainingCompilerConfig.HP_ENABLE_DEBUG] = json.dumps(
        False
    )

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert (
        actual_train_args == expected_train_args
    ), f"{json.dumps(actual_train_args, indent=2)} != {json.dumps(expected_train_args, indent=2)}"


@patch("sagemaker.utils.repack_model", MagicMock())
@patch("sagemaker.utils.create_tar_file", MagicMock())
@patch("sagemaker.estimator.name_from_base", return_value=JOB_NAME)
@patch("time.time", return_value=TIME)
def test_debug_compiler_config(
    time,
    name_from_base,
    sagemaker_session,
    huggingface_training_compiler_version,
    huggingface_training_compiler_tensorflow_version,
    huggingface_training_compiler_tensorflow_py_version,
):
    compiler_config = TrainingCompilerConfig(debug=True)

    hf = HuggingFace(
        py_version=huggingface_training_compiler_tensorflow_py_version,
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        transformers_version=huggingface_training_compiler_version,
        tensorflow_version=huggingface_training_compiler_tensorflow_version,
        enable_sagemaker_metrics=False,
        compiler_config=compiler_config,
    )

    inputs = "s3://mybucket/train"

    hf.fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ["train", "logs_for_job"]
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ["resource"]

    expected_train_args = _create_train_job(
        huggingface_training_compiler_version,
        f"tensorflow{huggingface_training_compiler_tensorflow_version}",
        INSTANCE_TYPE,
        compiler_config,
        huggingface_training_compiler_tensorflow_py_version,
    )
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs
    expected_train_args["enable_sagemaker_metrics"] = False
    expected_train_args["hyperparameters"][TrainingCompilerConfig.HP_ENABLE_COMPILER] = json.dumps(
        True
    )
    expected_train_args["hyperparameters"][TrainingCompilerConfig.HP_ENABLE_DEBUG] = json.dumps(
        True
    )

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert (
        actual_train_args == expected_train_args
    ), f"{json.dumps(actual_train_args, indent=2)} != {json.dumps(expected_train_args, indent=2)}"


@patch("sagemaker.utils.repack_model", MagicMock())
@patch("sagemaker.utils.create_tar_file", MagicMock())
@patch("sagemaker.estimator.name_from_base", return_value=JOB_NAME)
@patch("time.time", return_value=TIME)
def test_disable_compiler_config(
    time,
    name_from_base,
    sagemaker_session,
    huggingface_training_compiler_version,
    huggingface_training_compiler_tensorflow_version,
    huggingface_training_compiler_tensorflow_py_version,
):
    compiler_config = TrainingCompilerConfig(enabled=False)

    hf = HuggingFace(
        py_version=huggingface_training_compiler_tensorflow_py_version,
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        transformers_version=huggingface_training_compiler_version,
        tensorflow_version=huggingface_training_compiler_tensorflow_version,
        enable_sagemaker_metrics=False,
        compiler_config=TrainingCompilerConfig(enabled=False),
    )

    inputs = "s3://mybucket/train"

    hf.fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ["train", "logs_for_job"]
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ["resource"]

    expected_train_args = _create_train_job(
        huggingface_training_compiler_version,
        f"tensorflow{huggingface_training_compiler_tensorflow_version}",
        INSTANCE_TYPE,
        compiler_config,
        huggingface_training_compiler_tensorflow_py_version,
    )
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs
    expected_train_args["enable_sagemaker_metrics"] = False
    expected_train_args["hyperparameters"][TrainingCompilerConfig.HP_ENABLE_COMPILER] = json.dumps(
        False
    )
    expected_train_args["hyperparameters"][TrainingCompilerConfig.HP_ENABLE_DEBUG] = json.dumps(
        False
    )

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert (
        actual_train_args == expected_train_args
    ), f"{json.dumps(actual_train_args, indent=2)} != {json.dumps(expected_train_args, indent=2)}"


@pytest.mark.parametrize(
    ["compiler_enabled", "debug_enabled"], [(True, False), (True, True), (False, False)]
)
def test_attach(
    sagemaker_session,
    compiler_enabled,
    debug_enabled,
    huggingface_training_compiler_tensorflow_py_version,
):
    training_image = (
        f"1.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-trcomp-training:"
        f"2.6.3-"
        f"transformers4.17.0-gpu-"
        f"{huggingface_training_compiler_tensorflow_py_version}-cu112-ubuntu20.04"
    )
    returned_job_description = {
        "AlgorithmSpecification": {"TrainingInputMode": "File", "TrainingImage": training_image},
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
            "sagemaker_program": '"iris-dnn-classifier.py"',
            "sagemaker_s3_uri_training": '"sagemaker-3/integ-test-data/tf_iris"',
            "sagemaker_container_log_level": '"logging.INFO"',
            "sagemaker_job_name": '"trcomp"',
            "training_steps": "100",
            "sagemaker_region": '"us-east-1"',
            TrainingCompilerConfig.HP_ENABLE_COMPILER: json.dumps(compiler_enabled),
            TrainingCompilerConfig.HP_ENABLE_DEBUG: json.dumps(debug_enabled),
        },
        "RoleArn": "arn:aws:iam::366:role/SageMakerRole",
        "ResourceConfig": {
            "VolumeSizeInGB": 30,
            "InstanceCount": 1,
            "InstanceType": "ml.p3.2xlarge",
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 24 * 60 * 60},
        "TrainingJobName": "trcomp",
        "TrainingJobStatus": "Completed",
        "TrainingJobArn": "arn:aws:sagemaker:us-west-2:336:training-job/trcomp",
        "OutputDataConfig": {"KmsKeyId": "", "S3OutputPath": "s3://place/output/trcomp"},
        "TrainingJobOutput": {"S3TrainingJobOutput": "s3://here/output.tar.gz"},
    }
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=returned_job_description
    )

    estimator = HuggingFace.attach(training_job_name="trcomp", sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == "trcomp"
    assert estimator.py_version == huggingface_training_compiler_tensorflow_py_version
    assert estimator.framework_version == "4.17.0"
    assert estimator.tensorflow_version == "2.6.3"
    assert estimator.role == "arn:aws:iam::366:role/SageMakerRole"
    assert estimator.instance_count == 1
    assert estimator.max_run == 24 * 60 * 60
    assert estimator.input_mode == "File"
    assert estimator.base_job_name == "trcomp"
    assert estimator.output_path == "s3://place/output/trcomp"
    assert estimator.output_kms_key == ""
    assert estimator.hyperparameters()["training_steps"] == "100"
    assert estimator.hyperparameters()[TrainingCompilerConfig.HP_ENABLE_COMPILER] == json.dumps(
        compiler_enabled
    )
    assert estimator.hyperparameters()[TrainingCompilerConfig.HP_ENABLE_DEBUG] == json.dumps(
        debug_enabled
    )
    assert estimator.source_dir == "s3://some/sourcedir.tar.gz"
    assert estimator.entry_point == "iris-dnn-classifier.py"


def test_register_hf_tfs_model_auto_infer_framework(
    sagemaker_session,
    huggingface_training_compiler_version,
    huggingface_training_compiler_tensorflow_version,
    huggingface_training_compiler_tensorflow_py_version,
):

    model_package_group_name = "test-hf-tfs-register-model"
    content_types = ["application/json"]
    response_types = ["application/json"]
    inference_instances = ["ml.m4.xlarge"]
    transform_instances = ["ml.m4.xlarge"]
    image_uri = "fakeimage"

    hf_model = HuggingFaceModel(
        model_data="s3://some/data.tar.gz",
        role=ROLE,
        transformers_version=huggingface_training_compiler_version,
        tensorflow_version=huggingface_training_compiler_tensorflow_version,
        py_version=huggingface_training_compiler_tensorflow_py_version,
        sagemaker_session=sagemaker_session,
    )

    hf_model.register(
        content_types,
        response_types,
        inference_instances,
        transform_instances,
        model_package_group_name=model_package_group_name,
        marketplace_cert=True,
        image_uri=image_uri,
    )

    expected_create_model_package_request = {
        "containers": [
            {
                "Image": image_uri,
                "Environment": ANY,
                "ModelDataUrl": ANY,
                "Framework": "TENSORFLOW",
                "FrameworkVersion": huggingface_training_compiler_tensorflow_version,
            },
        ],
        "content_types": content_types,
        "response_types": response_types,
        "inference_instances": inference_instances,
        "transform_instances": transform_instances,
        "model_package_group_name": model_package_group_name,
        "marketplace_cert": True,
    }

    sagemaker_session.create_model_package_from_containers.assert_called_with(
        **expected_create_model_package_request
    )
