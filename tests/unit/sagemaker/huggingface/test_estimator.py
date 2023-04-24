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
from mock import MagicMock, Mock, patch

from sagemaker.huggingface import HuggingFace, HuggingFaceModel
from sagemaker.session_settings import SessionSettings


from .huggingface_utils import get_full_gpu_image_uri, GPU_INSTANCE_TYPE, REGION


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPT_PATH = os.path.join(DATA_DIR, "dummy_script.py")
SERVING_SCRIPT_FILE = "another_dummy_script.py"
MODEL_DATA = "s3://some/data.tar.gz"
ENV = {"DUMMY_ENV_VAR": "dummy_value"}
TIMESTAMP = "2017-11-06-14:14:15.672"
TIME = 1510006209.073025
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
ACCELERATOR_TYPE = "ml.eia.medium"
IMAGE_URI = "huggingface"
JOB_NAME = "{}-{}".format(IMAGE_URI, TIMESTAMP)
ROLE = "Dummy"

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}

LIST_TAGS_RESULT = {"Tags": [{"Key": "TagtestKey", "Value": "TagtestValue"}]}

EXPERIMENT_CONFIG = {
    "ExperimentName": "exp",
    "TrialName": "trial",
    "TrialComponentDisplayName": "tc",
    "RunName": "rn",
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


def _huggingface_estimator(
    sagemaker_session,
    framework_version,
    pytorch_version,
    tensorflow_version,
    py_version,
    instance_type=None,
    base_job_name=None,
    **kwargs,
):
    return HuggingFace(
        entry_point=SCRIPT_PATH,
        framework_version=framework_version,
        py_version=py_version,
        pytorch_version=pytorch_version,
        tensorflow_version=tensorflow_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=instance_type if instance_type else GPU_INSTANCE_TYPE,
        base_job_name=base_job_name,
        **kwargs,
    )


def _create_train_job(version, base_framework_version):
    return {
        "image_uri": get_full_gpu_image_uri(version, base_framework_version),
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
            "InstanceType": GPU_INSTANCE_TYPE,
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


def test_huggingface_invalid_args():
    with pytest.raises(ValueError) as error:
        HuggingFace(
            py_version="py36",
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=INSTANCE_COUNT,
            instance_type=GPU_INSTANCE_TYPE,
            transformers_version="4.2.1",
            pytorch_version="1.6",
            enable_sagemaker_metrics=False,
        )
    assert "use either full version or shortened version" in str(error)

    with pytest.raises(ValueError) as error:
        HuggingFace(
            py_version="py36",
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=INSTANCE_COUNT,
            instance_type=GPU_INSTANCE_TYPE,
            pytorch_version="1.6",
            enable_sagemaker_metrics=False,
        )
    assert "transformers_version, and image_uri are both None." in str(error)

    with pytest.raises(ValueError) as error:
        HuggingFace(
            py_version="py36",
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=INSTANCE_COUNT,
            instance_type=GPU_INSTANCE_TYPE,
            transformers_version="4.2.1",
            enable_sagemaker_metrics=False,
        )
    assert "tensorflow_version and pytorch_version are both None." in str(error)

    with pytest.raises(ValueError) as error:
        HuggingFace(
            py_version="py36",
            entry_point=SCRIPT_PATH,
            role=ROLE,
            instance_count=INSTANCE_COUNT,
            instance_type=GPU_INSTANCE_TYPE,
            transformers_version="4.2",
            pytorch_version="1.6",
            tensorflow_version="2.3",
            enable_sagemaker_metrics=False,
        )
    assert "tensorflow_version and pytorch_version are both not None." in str(error)


@patch("sagemaker.utils.repack_model", MagicMock())
@patch("sagemaker.utils.create_tar_file", MagicMock())
@patch("sagemaker.estimator.name_from_base", return_value=JOB_NAME)
@patch("time.time", return_value=TIME)
def test_huggingface(
    time,
    name_from_base,
    sagemaker_session,
    huggingface_training_version,
    huggingface_pytorch_training_version,
    huggingface_pytorch_training_py_version,
):
    hf = HuggingFace(
        py_version=huggingface_pytorch_training_py_version,
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=GPU_INSTANCE_TYPE,
        transformers_version=huggingface_training_version,
        pytorch_version=huggingface_pytorch_training_version,
        enable_sagemaker_metrics=False,
    )

    inputs = "s3://mybucket/train"

    hf.fit(inputs=inputs, experiment_config=EXPERIMENT_CONFIG)

    sagemaker_call_names = [c[0] for c in sagemaker_session.method_calls]
    assert sagemaker_call_names == ["train", "logs_for_job"]
    boto_call_names = [c[0] for c in sagemaker_session.boto_session.method_calls]
    assert boto_call_names == ["resource"]

    expected_train_args = _create_train_job(
        huggingface_training_version, f"pytorch{huggingface_pytorch_training_version}"
    )
    expected_train_args["input_config"][0]["DataSource"]["S3DataSource"]["S3Uri"] = inputs
    expected_train_args["experiment_config"] = EXPERIMENT_CONFIG
    expected_train_args["enable_sagemaker_metrics"] = False

    actual_train_args = sagemaker_session.method_calls[0][2]
    assert actual_train_args == expected_train_args


def test_huggingface_neuron(
    sagemaker_session,
    huggingface_neuron_latest_inference_pytorch_version,
    huggingface_neuron_latest_inference_transformer_version,
    huggingface_neuron_latest_inference_py_version,
):

    inputs = "s3://mybucket/train"
    huggingface_model = HuggingFaceModel(
        model_data=inputs,
        transformers_version=huggingface_neuron_latest_inference_transformer_version,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        pytorch_version=huggingface_neuron_latest_inference_pytorch_version,
        py_version=huggingface_neuron_latest_inference_py_version,
    )
    container = huggingface_model.prepare_container_def("ml.inf.xlarge")
    assert container["Image"]


def test_attach(
    sagemaker_session,
    huggingface_training_version,
    huggingface_pytorch_training_version,
    huggingface_pytorch_training_py_version,
):
    training_image = (
        f"1.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:"
        f"{huggingface_pytorch_training_version}-"
        f"transformers{huggingface_training_version}-gpu-"
        f"{huggingface_pytorch_training_py_version}-cu110-ubuntu20.04"
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
            "sagemaker_region": '"us-east-1"',
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

    estimator = HuggingFace.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == "neo"
    assert estimator.py_version == huggingface_pytorch_training_py_version
    assert estimator.framework_version == huggingface_training_version
    assert estimator.pytorch_version == huggingface_pytorch_training_version
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


def test_attach_custom_image(sagemaker_session):
    training_image = "pytorch:latest"
    returned_job_description = {
        "AlgorithmSpecification": {"TrainingInputMode": "File", "TrainingImage": training_image},
        "HyperParameters": {
            "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
            "sagemaker_program": '"iris-dnn-classifier.py"',
            "sagemaker_s3_uri_training": '"sagemaker-3/integ-test-data/tf_iris"',
            "sagemaker_container_log_level": '"logging.INFO"',
            "sagemaker_job_name": '"neo"',
            "training_steps": "100",
            "sagemaker_region": '"us-east-1"',
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

    estimator = HuggingFace.attach(training_job_name="neo", sagemaker_session=sagemaker_session)
    assert estimator.latest_training_job.job_name == "neo"
    assert estimator.image_uri == training_image
