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
from copy import deepcopy

import logging
import json
import os
import subprocess
from time import sleep

from sagemaker.fw_utils import UploadedCode


import pytest
from botocore.exceptions import ClientError
from mock import ANY, MagicMock, Mock, patch, PropertyMock
from sagemaker.huggingface.estimator import HuggingFace
from sagemaker.jumpstart.constants import (
    JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET,
    JUMPSTART_RESOURCE_BASE_NAME,
)
from sagemaker.jumpstart.enums import JumpStartTag

import sagemaker.local
from sagemaker import TrainingInput, utils, vpc_utils
from sagemaker.algorithm import AlgorithmEstimator
from sagemaker.debugger import (
    rule_configs,
    CollectionConfig,
    DebuggerHookConfig,
    FrameworkProfile,
    ProfilerConfig,
    ProfilerRule,
    Rule,
)
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.estimator import Estimator, EstimatorBase, Framework, _TrainingJob
from sagemaker.fw_utils import PROFILER_UNSUPPORTED_REGIONS
from sagemaker.inputs import ShuffleConfig
from sagemaker.instance_group import InstanceGroup
from sagemaker.interactive_apps import SupportedInteractiveAppTypes
from sagemaker.model import FrameworkModel
from sagemaker.mxnet.estimator import MXNet
from sagemaker.predictor import Predictor
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.session_settings import SessionSettings
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tensorflow.estimator import TensorFlow
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.transformer import Transformer
from sagemaker.workflow.execution_variables import ExecutionVariable
from sagemaker.workflow.parameters import ParameterString, ParameterBoolean
from sagemaker.workflow.pipeline_context import PipelineSession, _PipelineConfig
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.xgboost.estimator import XGBoost
from tests.unit import (
    SAGEMAKER_CONFIG_TRAINING_JOB,
    SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_FALSE,
    SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_TRUE,
    _test_default_bucket_and_prefix_combinations,
    DEFAULT_S3_BUCKET_NAME,
    DEFAULT_S3_OBJECT_KEY_PREFIX_NAME,
)

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
ENTRY_POINT = "blah.py"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPT_NAME = "dummy_script.py"
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_NAME)
TIMESTAMP = "2017-11-06-14:14:15.671"
TIME = 1510006209.073025
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "c4.4xlarge"
KEEP_ALIVE_PERIOD_IN_SECONDS = 1800
ACCELERATOR_TYPE = "ml.eia.medium"
ROLE = "DummyRole"
IMAGE_URI = "fakeimage"
REGION = "us-west-2"
JOB_NAME = "{}-{}".format(IMAGE_URI, TIMESTAMP)
TAGS = [{"Name": "some-tag", "Value": "value-for-tag"}]
OUTPUT_PATH = "s3://bucket/prefix"
GIT_REPO = "https://github.com/aws/sagemaker-python-sdk.git"
BRANCH = "test-branch-git-config"
COMMIT = "ae15c9d7d5b97ea95ea451e4662ee43da3401d73"
PRIVATE_GIT_REPO_SSH = "git@github.com:testAccount/private-repo.git"
PRIVATE_GIT_REPO = "https://github.com/testAccount/private-repo.git"
PRIVATE_BRANCH = "test-branch"
PRIVATE_COMMIT = "329bfcf884482002c05ff7f44f62599ebc9f445a"
CODECOMMIT_REPO = "https://git-codecommit.us-west-2.amazonaws.com/v1/repos/test-repo/"
CODECOMMIT_REPO_SSH = "ssh://git-codecommit.us-west-2.amazonaws.com/v1/repos/test-repo/"
CODECOMMIT_BRANCH = "master"
REPO_DIR = "/tmp/repo_dir"
ENV_INPUT = {"env_key1": "env_val1", "env_key2": "env_val2", "env_key3": "env_val3"}
TRAINING_REPOSITORY_ACCESS_MODE = "VPC"
ENABLE_INFRA_CHECK = True
TRAINING_REPOSITORY_CREDENTIALS_PROVIDER_ARN = "arn:aws:lambda:us-west-2:1234567890:function:test"
CONTAINER_ENTRY_POINT = ["entry_point1", "entry_point2"]
CONTAINER_ARGUMENTS = ["container_arg1", "container_arg2"]

DESCRIBE_TRAINING_JOB_RESULT = {"ModelArtifacts": {"S3ModelArtifacts": MODEL_DATA}}

DESCRIBE_TRAINING_JOB_RESULT_UNCOMPRESSED_S3_MODEL = {
    "ModelArtifacts": {
        "S3ModelArtifacts": "s3://bucket/model/prefix",
    },
    "OutputDataConfig": {
        "CompressionType": "NONE",
        "KmsKeyId": "outputkms",
        "S3OutputPath": "s3://path/to/model",
    },
}

RETURNED_JOB_DESCRIPTION = {
    "AlgorithmSpecification": {
        "TrainingInputMode": "File",
        "TrainingImage": "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-other:1.0.4",
    },
    "HyperParameters": {
        "sagemaker_submit_directory": '"s3://some/sourcedir.tar.gz"',
        "checkpoint_path": '"s3://other/1508872349"',
        "sagemaker_program": '"iris-dnn-classifier.py"',
        "sagemaker_container_log_level": '"logging.INFO"',
        "sagemaker_job_name": '"neo"',
        "training_steps": "100",
    },
    "RoleArn": "arn:aws:iam::366:role/SageMakerRole",
    "ResourceConfig": {"VolumeSizeInGB": 30, "InstanceCount": 1, "InstanceType": "ml.c4.xlarge"},
    "EnableNetworkIsolation": False,
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
    },
    "Image": MODEL_IMAGE,
    "ModelDataUrl": MODEL_DATA,
}

ENDPOINT_DESC = {"EndpointConfigName": "test-endpoint"}

ENDPOINT_CONFIG_DESC = {"ProductionVariants": [{"ModelName": "model-1"}, {"ModelName": "model-2"}]}

LIST_TAGS_RESULT = {"Tags": [{"Key": "TagtestKey", "Value": "TagtestValue"}]}

DISTRIBUTION_PS_ENABLED = {"parameter_server": {"enabled": True}}
DISTRIBUTION_MWMS_ENABLED = {"multi_worker_mirrored_strategy": {"enabled": True}}
DISTRIBUTION_MPI_ENABLED = {
    "mpi": {"enabled": True, "custom_mpi_options": "options", "processes_per_host": 2}
}
DISTRIBUTION_SM_DDP_ENABLED = {
    "smdistributed": {"dataparallel": {"enabled": True, "custom_mpi_options": "options"}},
    "torch_distributed": {"enabled": False},
}
DISTRIBUTION_SM_DDP_DISABLED = {
    "smdistributed": {"enabled": True},
    "torch_distributed": {"enabled": False},
}
DISTRIBUTION_SM_TORCH_DIST_AND_DDP_ENABLED = {
    "smdistributed": {"dataparallel": {"enabled": True, "custom_mpi_options": "options"}},
    "torch_distributed": {"enabled": True},
}
DISTRIBUTION_SM_TORCH_DIST_AND_DDP_DISABLED = {
    "smdistributed": {"enabled": True},
    "torch_distributed": {"enabled": True},
}
MOCKED_S3_URI = "s3://mocked_s3_uri_from_source_dir"
_DEFINITION_CONFIG = PipelineDefinitionConfig(use_custom_job_prefix=False)
MOCKED_PIPELINE_CONFIG = _PipelineConfig(
    "test-pipeline",
    "test-training-step",
    None,
    "code-hash-0123456789",
    "config-hash-0123456789",
    _DEFINITION_CONFIG,
)
HOOK_CONFIG_WITHOUT_S3_PATH = DebuggerHookConfig(
    hook_parameters={"save_interval": "1"},
)
HOOK_CONFIG = DebuggerHookConfig(
    hook_parameters={"save_interval": "1"},
    s3_output_path="s3://mytestbucket/testpath/",
)
S3_OUTPUT_PATH_FROM_SESSION_S3_DEFAULT_CONFIG = "s3://mybucket/"


class DummyFramework(Framework):
    _framework_name = "dummy"

    def training_image_uri(self):
        return IMAGE_URI

    def create_model(
        self,
        role=None,
        model_server_workers=None,
        entry_point=None,
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        enable_network_isolation=None,
        model_dir=None,
        **kwargs,
    ):
        if enable_network_isolation is None:
            enable_network_isolation = self.enable_network_isolation()

        return DummyFrameworkModel(
            self.sagemaker_session,
            vpc_config=self.get_vpc_config(vpc_config_override),
            entry_point=entry_point,
            enable_network_isolation=enable_network_isolation,
            role=role,
            **kwargs,
        )

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details, model_channel_name=None):
        init_params = super(DummyFramework, cls)._prepare_init_params_from_job_description(
            job_details, model_channel_name
        )
        init_params.pop("image_uri", None)
        return init_params


class DummyFrameworkModel(FrameworkModel):
    def __init__(self, sagemaker_session, entry_point=None, role=ROLE, **kwargs):
        super(DummyFrameworkModel, self).__init__(
            MODEL_DATA,
            MODEL_IMAGE,
            role,
            entry_point or ENTRY_POINT,
            sagemaker_session=sagemaker_session,
            **kwargs,
        )

    def create_predictor(self, endpoint_name):
        return None

    def prepare_container_def(
        self,
        instance_type,
        accelerator_type=None,
        serverless_inference_config=None,
        accept_eula=None,
    ):
        return MODEL_CONTAINER_DEF


@pytest.fixture(autouse=True)
def mock_create_tar_file():
    with patch("sagemaker.utils.create_tar_file", MagicMock()) as create_tar_file:
        yield create_tar_file


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        s3_client=None,
        s3_resource=None,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    sms.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job", return_value=DESCRIBE_TRAINING_JOB_RESULT
    )
    sms.sagemaker_client.describe_endpoint = Mock(return_value=ENDPOINT_DESC)
    sms.sagemaker_client.describe_endpoint_config = Mock(return_value=ENDPOINT_CONFIG_DESC)
    sms.sagemaker_client.list_tags = Mock(return_value=LIST_TAGS_RESULT)
    sms.upload_data = Mock(return_value=OUTPUT_PATH)

    # For tests which doesn't verify config file injection, operate with empty config
    sms.sagemaker_config = {}
    return sms


@pytest.fixture()
def pipeline_session():
    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value=ROLE)
    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock
    session_mock = Mock(region_name=REGION, settings=SessionSettings())
    session_mock.resource.return_value = resource_mock
    session_mock.client.return_value = client_mock
    return PipelineSession(
        boto_session=session_mock, sagemaker_client=client_mock, default_bucket=BUCKET_NAME
    )


@pytest.fixture()
def training_job_description(sagemaker_session):
    returned_job_description = RETURNED_JOB_DESCRIPTION.copy()
    mock_describe_training_job = Mock(
        name="describe_training_job", return_value=returned_job_description
    )
    sagemaker_session.sagemaker_client.describe_training_job = mock_describe_training_job
    sagemaker_session.describe_training_job = mock_describe_training_job
    return returned_job_description


def test_validate_smdistributed_unsupported_image_raises(sagemaker_session):
    # Test unsupported image raises error.
    for unsupported_image in DummyFramework.UNSUPPORTED_DLC_IMAGE_FOR_SM_PARALLELISM:
        # Fail due to unsupported CUDA12 DLC image.
        f = DummyFramework(
            "some_script.py",
            role="DummyRole",
            instance_type="ml.p4d.24xlarge",
            sagemaker_session=sagemaker_session,
            output_path="outputpath",
            image_uri=unsupported_image,
        )
        with pytest.raises(ValueError):
            f._distribution_configuration(DISTRIBUTION_SM_DDP_ENABLED)
        with pytest.raises(ValueError):
            f._distribution_configuration(DISTRIBUTION_SM_DDP_DISABLED)

    # Test unsupported image with suffix raises error.
    for unsupported_image in DummyFramework.UNSUPPORTED_DLC_IMAGE_FOR_SM_PARALLELISM:
        # Fail due to unsupported CUDA12 DLC image.
        f = DummyFramework(
            "some_script.py",
            role="DummyRole",
            instance_type="ml.p4d.24xlarge",
            sagemaker_session=sagemaker_session,
            output_path="outputpath",
            image_uri=unsupported_image + "-ubuntu20.04-sagemaker-pr-3303",
        )
        with pytest.raises(ValueError):
            f._distribution_configuration(DISTRIBUTION_SM_DDP_ENABLED)
        with pytest.raises(ValueError):
            f._distribution_configuration(DISTRIBUTION_SM_DDP_DISABLED)


def test_validate_smdistributed_p5_raises(sagemaker_session):
    # Supported DLC image.
    f = DummyFramework(
        "some_script.py",
        role="DummyRole",
        instance_type="ml.p5.48xlarge",
        sagemaker_session=sagemaker_session,
        output_path="outputpath",
        image_uri="some_acceptable_image",
    )
    # Both fail because instance type is p5 and torch_distributed is off.
    with pytest.raises(ValueError):
        f._distribution_configuration(DISTRIBUTION_SM_DDP_ENABLED)
    with pytest.raises(ValueError):
        f._distribution_configuration(DISTRIBUTION_SM_DDP_DISABLED)


def test_validate_smdistributed_p5_not_raises(sagemaker_session):
    f = DummyFramework(
        "some_script.py",
        role="DummyRole",
        instance_type="ml.p5.48xlarge",
        sagemaker_session=sagemaker_session,
        output_path="outputpath",
        image_uri="ecr-url/2.0.1-gpu-py310-cu121-ubuntu20.04-sagemaker-pr-3303",
    )
    # Testing with p5 instance and torch_distributed enabled.
    f._distribution_configuration(DISTRIBUTION_SM_TORCH_DIST_AND_DDP_ENABLED)
    f._distribution_configuration(DISTRIBUTION_SM_TORCH_DIST_AND_DDP_DISABLED)


def test_validate_smdistributed_backward_compat_p4_not_raises(sagemaker_session):
    f = DummyFramework(
        "some_script.py",
        role="DummyRole",
        instance_type="ml.p4d.24xlarge",
        sagemaker_session=sagemaker_session,
        output_path="outputpath",
        image_uri="some_acceptable_image",
    )
    # Testing backwards compatability with p4d instances.
    f._distribution_configuration(DISTRIBUTION_SM_TORCH_DIST_AND_DDP_ENABLED)
    f._distribution_configuration(DISTRIBUTION_SM_TORCH_DIST_AND_DDP_DISABLED)


def test_validate_smdistributed_instance_groups_raises(sagemaker_session):
    instance_group_1 = InstanceGroup("train_group", "ml.p4d.24xlarge", 2)
    instance_group_2 = InstanceGroup("train_group", "ml.p5.48xlarge", 2)
    f = DummyFramework(
        "some_script.py",
        role="DummyRole",
        instance_groups=[instance_group_1, instance_group_2],
        sagemaker_session=sagemaker_session,
        output_path="outputpath",
        image_uri="some_acceptable_image",
    )
    # Testing instance_group with p5 raises exception
    with pytest.raises(ValueError):
        f._distribution_configuration(DISTRIBUTION_SM_DDP_ENABLED)
    with pytest.raises(ValueError):
        f._distribution_configuration(DISTRIBUTION_SM_DDP_DISABLED)


def test_validate_smdistributed_instance_groups_not_raises(sagemaker_session):
    instance_group_1 = InstanceGroup("train_group", "ml.p4d.24xlarge", 2)
    f = DummyFramework(
        "some_script.py",
        role="DummyRole",
        instance_groups=[instance_group_1],
        sagemaker_session=sagemaker_session,
        output_path="outputpath",
        image_uri="some_acceptable_image",
    )
    # Testing instance_group without p5 does not raise exception
    f._distribution_configuration(DISTRIBUTION_SM_TORCH_DIST_AND_DDP_ENABLED)
    f._distribution_configuration(DISTRIBUTION_SM_TORCH_DIST_AND_DDP_DISABLED)


def test_framework_all_init_args(sagemaker_session):
    f = DummyFramework(
        "my_script.py",
        role="DummyRole",
        instance_count=3,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        volume_size=123,
        volume_kms_key="volumekms",
        max_run=456,
        input_mode="inputmode",
        output_path="outputpath",
        output_kms_key="outputkms",
        base_job_name="basejobname",
        tags=[{"foo": "bar"}],
        subnets=["123", "456"],
        security_group_ids=["789", "012"],
        metric_definitions=[{"Name": "validation-rmse", "Regex": "validation-rmse=(\\d+)"}],
        encrypt_inter_container_traffic=True,
        checkpoint_s3_uri="s3://bucket/checkpoint",
        checkpoint_local_path="file://local/checkpoint",
        enable_sagemaker_metrics=True,
        enable_network_isolation=True,
        environment=ENV_INPUT,
        max_retry_attempts=2,
    )
    _TrainingJob.start_new(f, "s3://mydata", None)
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args == {
        "input_mode": "inputmode",
        "tags": [{"foo": "bar"}],
        "hyperparameters": {},
        "image_uri": "fakeimage",
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
        "retry_strategy": {"MaximumRetryAttempts": 2},
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
        "environment": {"env_key1": "env_val1", "env_key2": "env_val2", "env_key3": "env_val3"},
        "experiment_config": None,
        "checkpoint_s3_uri": "s3://bucket/checkpoint",
        "checkpoint_local_path": "file://local/checkpoint",
        "enable_sagemaker_metrics": True,
        "enable_network_isolation": True,
    }


def test_subnets_without_security_groups(sagemaker_session):
    with pytest.raises(RuntimeError):
        DummyFramework(
            entry_point=SCRIPT_PATH,
            sagemaker_session=sagemaker_session,
            subnets=["123"],
        )


def test_security_groups_without_subnets(sagemaker_session):
    with pytest.raises(RuntimeError):
        DummyFramework(
            entry_point=SCRIPT_PATH,
            sagemaker_session=sagemaker_session,
            security_group_ids=["123"],
        )


def test_framework_without_role_parameter(sagemaker_session):
    with pytest.raises(ValueError):
        DummyFramework(
            entry_point=SCRIPT_PATH,
            sagemaker_session=sagemaker_session,
            instance_groups=[
                InstanceGroup("group1", "ml.c4.xlarge", 1),
                InstanceGroup("group2", "ml.m4.xlarge", 2),
            ],
        )


def test_default_value_of_enable_network_isolation(sagemaker_session):
    framework = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
    )
    assert framework.enable_network_isolation() is False


def test_framework_initialization_with_sagemaker_config_injection(sagemaker_session):

    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB

    framework = DummyFramework(
        entry_point=SCRIPT_PATH,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
    )
    expected_volume_kms_key_id = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ResourceConfig"
    ]["VolumeKmsKeyId"]
    expected_role_arn = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["RoleArn"]
    expected_kms_key_id = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "OutputDataConfig"
    ]["KmsKeyId"]
    expected_subnets = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["VpcConfig"][
        "Subnets"
    ]
    expected_security_groups = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "VpcConfig"
    ]["SecurityGroupIds"]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "EnableNetworkIsolation"
    ]
    expected_enable_inter_container_traffic_encryption = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"][
        "TrainingJob"
    ]["EnableInterContainerTrafficEncryption"]
    expected_environment = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["Environment"]
    expected_disable_profiler_attribute = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ProfilerConfig"
    ]["DisableProfiler"]
    expected_debugger_hook_config = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["PythonSDK"][
        "Modules"
    ]["Estimator"]["DebugHookConfig"]
    assert framework.role == expected_role_arn
    assert framework.enable_network_isolation() == expected_enable_network_isolation
    assert (
        framework.encrypt_inter_container_traffic
        == expected_enable_inter_container_traffic_encryption
    )
    assert framework.output_kms_key == expected_kms_key_id
    assert framework.volume_kms_key == expected_volume_kms_key_id
    assert framework.security_group_ids == expected_security_groups
    assert framework.subnets == expected_subnets
    assert framework.environment == expected_environment
    assert framework.disable_profiler == expected_disable_profiler_attribute
    assert framework.debugger_hook_config == expected_debugger_hook_config


def test_estimator_initialization_with_sagemaker_config_injection(sagemaker_session):
    """
    Tests that the estimator initialization works when all the supported defaults config params "
    are provided from the sagemaker_config
    """
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB

    estimator = Estimator(
        image_uri="some-image",
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.p3.16xlarge", 2),
        ],
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
    )
    expected_volume_kms_key_id = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ResourceConfig"
    ]["VolumeKmsKeyId"]
    expected_role_arn = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["RoleArn"]
    expected_kms_key_id = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "OutputDataConfig"
    ]["KmsKeyId"]
    expected_subnets = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["VpcConfig"][
        "Subnets"
    ]
    expected_security_groups = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "VpcConfig"
    ]["SecurityGroupIds"]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "EnableNetworkIsolation"
    ]
    expected_enable_inter_container_traffic_encryption = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"][
        "TrainingJob"
    ]["EnableInterContainerTrafficEncryption"]
    expected_environment = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["Environment"]
    expected_disable_profiler_attribute = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ProfilerConfig"
    ]["DisableProfiler"]
    expected_debugger_hook_config = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["PythonSDK"][
        "Modules"
    ]["Estimator"]["DebugHookConfig"]
    assert estimator.role == expected_role_arn
    assert estimator.enable_network_isolation() == expected_enable_network_isolation
    assert (
        estimator.encrypt_inter_container_traffic
        == expected_enable_inter_container_traffic_encryption
    )
    assert estimator.output_kms_key == expected_kms_key_id
    assert estimator.volume_kms_key == expected_volume_kms_key_id
    assert estimator.security_group_ids == expected_security_groups
    assert estimator.subnets == expected_subnets
    assert estimator.environment == expected_environment
    assert estimator.disable_profiler == expected_disable_profiler_attribute
    assert estimator.debugger_hook_config == expected_debugger_hook_config


def test_estimator_with_debugger_hook_config_provided_as_bool_from_direct_input(
    sagemaker_session,
):
    """
    Tests that the estimator initialization works correctly with sagemaker_config injection
    when debugger_hook_config is provided as True from direct input
    """
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB

    estimator = Estimator(
        image_uri="some-image",
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.p3.16xlarge", 2),
        ],
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
        debugger_hook_config=True,
    )
    assert estimator.debugger_hook_config == {}


def test_estimator_with_debugger_hook_config_provided_as_dict_from_direct_input(
    sagemaker_session,
):
    """
    Tests that the estimator initialization works correctly with sagemaker_config injection
    when debugger_hook_config is provided as dict from direct input
    """
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB
    estimator = Estimator(
        image_uri="some-image",
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.p3.16xlarge", 2),
        ],
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
        debugger_hook_config=HOOK_CONFIG,
    )
    assert estimator.debugger_hook_config == HOOK_CONFIG


def test_estimator_initialization_with_sagemaker_config_injection_no_kms_supported(
    sagemaker_session,
):

    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB

    estimator = Estimator(
        image_uri="some-image",
        instance_groups=[
            InstanceGroup("group1", "ml.g5.2xlarge", 1),
            InstanceGroup("group2", "ml.g5.2xlarge", 2),
        ],
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
    )

    expected_role_arn = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["RoleArn"]
    expected_kms_key_id = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "OutputDataConfig"
    ]["KmsKeyId"]
    expected_subnets = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["VpcConfig"][
        "Subnets"
    ]
    expected_security_groups = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "VpcConfig"
    ]["SecurityGroupIds"]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "EnableNetworkIsolation"
    ]
    expected_enable_inter_container_traffic_encryption = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"][
        "TrainingJob"
    ]["EnableInterContainerTrafficEncryption"]
    expected_disable_profiler_attribute = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ProfilerConfig"
    ]["DisableProfiler"]
    expected_debugger_hook_config = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["PythonSDK"][
        "Modules"
    ]["Estimator"]["DebugHookConfig"]
    assert estimator.role == expected_role_arn
    assert estimator.enable_network_isolation() == expected_enable_network_isolation
    assert (
        estimator.encrypt_inter_container_traffic
        == expected_enable_inter_container_traffic_encryption
    )
    assert estimator.output_kms_key == expected_kms_key_id
    assert estimator.volume_kms_key is None
    assert estimator.security_group_ids == expected_security_groups
    assert estimator.subnets == expected_subnets
    assert estimator.disable_profiler == expected_disable_profiler_attribute
    assert estimator.debugger_hook_config == expected_debugger_hook_config


def test_estimator_initialization_with_sagemaker_config_injection_partial_kms_support(
    sagemaker_session,
):

    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_TRAINING_JOB

    estimator = Estimator(
        image_uri="some-image",
        instance_groups=[
            InstanceGroup("group1", "ml.p2.xlarge", 1),
            InstanceGroup("group2", "ml.g5.2xlarge", 2),
        ],
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
    )

    expected_volume_kms_key_id = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ResourceConfig"
    ]["VolumeKmsKeyId"]
    expected_role_arn = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["RoleArn"]
    expected_kms_key_id = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "OutputDataConfig"
    ]["KmsKeyId"]
    expected_subnets = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"]["VpcConfig"][
        "Subnets"
    ]
    expected_security_groups = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "VpcConfig"
    ]["SecurityGroupIds"]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "EnableNetworkIsolation"
    ]
    expected_enable_inter_container_traffic_encryption = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"][
        "TrainingJob"
    ]["EnableInterContainerTrafficEncryption"]
    expected_disable_profiler_attribute = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["TrainingJob"][
        "ProfilerConfig"
    ]["DisableProfiler"]
    expected_debugger_hook_config = SAGEMAKER_CONFIG_TRAINING_JOB["SageMaker"]["PythonSDK"][
        "Modules"
    ]["Estimator"]["DebugHookConfig"]
    assert estimator.role == expected_role_arn
    assert estimator.enable_network_isolation() == expected_enable_network_isolation
    assert (
        estimator.encrypt_inter_container_traffic
        == expected_enable_inter_container_traffic_encryption
    )
    assert estimator.output_kms_key == expected_kms_key_id
    assert estimator.volume_kms_key == expected_volume_kms_key_id
    assert estimator.security_group_ids == expected_security_groups
    assert estimator.subnets == expected_subnets
    assert estimator.disable_profiler == expected_disable_profiler_attribute
    assert estimator.debugger_hook_config == expected_debugger_hook_config


def test_framework_with_heterogeneous_cluster(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["resource_config"]["InstanceGroups"][0] == {
        "InstanceGroupName": "group1",
        "InstanceCount": 1,
        "InstanceType": "ml.c4.xlarge",
    }
    assert args["resource_config"]["InstanceGroups"][1] == {
        "InstanceGroupName": "group2",
        "InstanceCount": 2,
        "InstanceType": "ml.m4.xlarge",
    }


def test_framework_with_keep_alive_period(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
        keep_alive_period_in_seconds=KEEP_ALIVE_PERIOD_IN_SECONDS,
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["resource_config"]["KeepAlivePeriodInSeconds"] == KEEP_ALIVE_PERIOD_IN_SECONDS


def test_framework_with_both_training_repository_config(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
        training_repository_access_mode=TRAINING_REPOSITORY_ACCESS_MODE,
        training_repository_credentials_provider_arn=TRAINING_REPOSITORY_CREDENTIALS_PROVIDER_ARN,
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert (
        args["training_image_config"]["TrainingRepositoryAccessMode"]
        == TRAINING_REPOSITORY_ACCESS_MODE
    )
    assert (
        args["training_image_config"]["TrainingRepositoryAuthConfig"][
            "TrainingRepositoryCredentialsProviderArn"
        ]
        == TRAINING_REPOSITORY_CREDENTIALS_PROVIDER_ARN
    )


def test_framework_with_training_repository_access_mode(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
        training_repository_access_mode=TRAINING_REPOSITORY_ACCESS_MODE,
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert (
        args["training_image_config"]["TrainingRepositoryAccessMode"]
        == TRAINING_REPOSITORY_ACCESS_MODE
    )
    assert "TrainingRepositoryAuthConfig" not in args["training_image_config"]


def test_framework_without_training_repository_config(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args.get("training_image_config") is None


def test_framework_without_infra_check_config(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args.get("health_check_config") is None


def test_framework_with_infra_check_config(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
        enable_infra_check=ENABLE_INFRA_CHECK,
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["infra_check_config"]["EnableInfraCheck"] == ENABLE_INFRA_CHECK


def test_framework_with_container_entry_point(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
        container_entry_point=CONTAINER_ENTRY_POINT,
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["container_entry_point"] == CONTAINER_ENTRY_POINT


def test_framework_with_container_arguments(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
        container_arguments=CONTAINER_ARGUMENTS,
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["container_arguments"] == CONTAINER_ARGUMENTS


def test_framework_with_debugger_and_built_in_rule(sagemaker_session):
    debugger_built_in_rule_with_custom_args = Rule.sagemaker(
        base_config=rule_configs.stalled_training_rule(),
        rule_parameters={"threshold": "120", "stop_training_on_fire": "True"},
        collections_to_save=[
            CollectionConfig(
                name="losses", parameters={"train.save_interval": "50", "eval.save_interval": "10"}
            )
        ],
    )
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        rules=[debugger_built_in_rule_with_custom_args],
        debugger_hook_config=DebuggerHookConfig(s3_output_path="s3://output"),
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["debugger_rule_configs"][0]["RuleParameters"] == {
        "rule_to_invoke": "StalledTrainingRule",
        "threshold": "120",
        "stop_training_on_fire": "True",
    }
    assert args["debugger_hook_config"] == {
        "S3OutputPath": "s3://output",
        "CollectionConfigurations": [
            {
                "CollectionName": "losses",
                "CollectionParameters": {"train.save_interval": "50", "eval.save_interval": "10"},
            }
        ],
    }
    assert args["profiler_config"] == {
        "DisableProfiler": False,
        "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
    }


def test_framework_with_debugger_and_custom_rule(sagemaker_session):
    hook_config = DebuggerHookConfig(
        s3_output_path="s3://output", collection_configs=[CollectionConfig(name="weights")]
    )
    debugger_custom_rule = Rule.custom(
        name="CustomRule",
        image_uri="RuleImageUri",
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=5,
        source="path/to/my_custom_rule.py",
        rule_to_invoke="CustomRule",
        other_trials_s3_input_paths=["s3://path/trial1", "s3://path/trial2"],
        rule_parameters={"threshold": "120"},
    )
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        rules=[debugger_custom_rule],
        debugger_hook_config=hook_config,
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["debugger_rule_configs"] == [
        {
            "RuleConfigurationName": "CustomRule",
            "RuleEvaluatorImage": "RuleImageUri",
            "InstanceType": INSTANCE_TYPE,
            "VolumeSizeInGB": 5,
            "RuleParameters": {
                "source_s3_uri": sagemaker_session.upload_data(),
                "rule_to_invoke": "CustomRule",
                "threshold": "120",
                "other_trial_0": "s3://path/trial1",
                "other_trial_1": "s3://path/trial2",
            },
        }
    ]
    assert args["debugger_hook_config"] == {
        "S3OutputPath": "s3://output",
        "CollectionConfigurations": [{"CollectionName": "weights"}],
    }


def test_framework_with_only_debugger_rule(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        rules=[Rule.sagemaker(rule_configs.stalled_training_rule())],
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["debugger_rule_configs"][0]["RuleParameters"] == {
        "rule_to_invoke": "StalledTrainingRule"
    }
    assert args["debugger_hook_config"] == {
        "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        "CollectionConfigurations": [],
    }


def test_framework_with_debugger_rule_and_single_action(sagemaker_session):
    stop_training_action = rule_configs.StopTraining()
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        rules=[Rule.sagemaker(rule_configs.stalled_training_rule(), actions=stop_training_action)],
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["debugger_rule_configs"][0]["RuleParameters"] == {
        "rule_to_invoke": "StalledTrainingRule",
        "action_json": stop_training_action.serialize(),
    }
    assert stop_training_action.action_parameters["training_job_prefix"] == f._current_job_name
    assert args["debugger_hook_config"] == {
        "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        "CollectionConfigurations": [],
    }


def test_framework_with_debugger_rule_and_multiple_actions(sagemaker_session):
    action_list = rule_configs.ActionList(
        rule_configs.StopTraining(),
        rule_configs.Email("abc@abc.com"),
        rule_configs.SMS("+1234567890"),
    )
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        rules=[Rule.sagemaker(rule_configs.stalled_training_rule(), actions=action_list)],
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["debugger_rule_configs"][0]["RuleParameters"] == {
        "rule_to_invoke": "StalledTrainingRule",
        "action_json": action_list.serialize(),
    }
    assert action_list.actions[0].action_parameters["training_job_prefix"] == f._current_job_name
    assert args["debugger_hook_config"] == {
        "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        "CollectionConfigurations": [],
    }


def test_framework_with_only_debugger_hook_config(sagemaker_session):
    hook_config = DebuggerHookConfig(
        s3_output_path="s3://output", collection_configs=[CollectionConfig(name="weights")]
    )
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        debugger_hook_config=hook_config,
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["debugger_hook_config"] == {
        "S3OutputPath": "s3://output",
        "CollectionConfigurations": [{"CollectionName": "weights"}],
    }
    assert "debugger_rule_configs" not in args


@patch("time.time", return_value=TIME)
def test_framework_without_debugger_and_profiler(time, sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["debugger_hook_config"] == {
        "CollectionConfigurations": [],
        "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
    }
    assert "debugger_rule_configs" not in args
    assert args["profiler_config"] == {
        "DisableProfiler": False,
        "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
    }


def test_framework_with_debugger_and_profiler_rules(sagemaker_session):
    debugger_built_in_rule_with_custom_args = Rule.sagemaker(
        base_config=rule_configs.stalled_training_rule(),
        rule_parameters={"threshold": "120", "stop_training_on_fire": "True"},
        collections_to_save=[
            CollectionConfig(
                name="losses", parameters={"train.save_interval": "50", "eval.save_interval": "10"}
            )
        ],
    )
    profiler_built_in_rule_with_custom_args = ProfilerRule.sagemaker(
        base_config=rule_configs.ProfilerReport(CPUBottleneck_threshold=90),
        name="CustomProfilerReportRule",
    )
    profiler_custom_rule = ProfilerRule.custom(
        name="CustomProfilerRule",
        image_uri="RuleImageUri",
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=5,
        source="path/to/my_custom_rule.py",
        rule_to_invoke="CustomProfilerRule",
        rule_parameters={"threshold": "10"},
    )
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        rules=[
            debugger_built_in_rule_with_custom_args,
            profiler_built_in_rule_with_custom_args,
            profiler_custom_rule,
        ],
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["debugger_rule_configs"] == [
        {
            "RuleConfigurationName": "StalledTrainingRule",
            "RuleEvaluatorImage": "895741380848.dkr.ecr.us-west-2.amazonaws.com/sagemaker-debugger-rules:latest",
            "RuleParameters": {
                "rule_to_invoke": "StalledTrainingRule",
                "threshold": "120",
                "stop_training_on_fire": "True",
            },
        }
    ]
    assert args["debugger_hook_config"] == {
        "S3OutputPath": "s3://mybucket/",
        "CollectionConfigurations": [
            {
                "CollectionName": "losses",
                "CollectionParameters": {"train.save_interval": "50", "eval.save_interval": "10"},
            }
        ],
    }
    assert args["profiler_config"] == {
        "DisableProfiler": False,
        "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
    }
    assert args["profiler_rule_configs"] == [
        {
            "RuleConfigurationName": "CustomProfilerReportRule",
            "RuleEvaluatorImage": "895741380848.dkr.ecr.us-west-2.amazonaws.com/sagemaker-debugger-rules:latest",
            "RuleParameters": {"rule_to_invoke": "ProfilerReport", "CPUBottleneck_threshold": "90"},
        },
        {
            "InstanceType": "c4.4xlarge",
            "RuleConfigurationName": "CustomProfilerRule",
            "RuleEvaluatorImage": "RuleImageUri",
            "RuleParameters": {
                "rule_to_invoke": "CustomProfilerRule",
                "source_s3_uri": OUTPUT_PATH,
                "threshold": "10",
            },
            "VolumeSizeInGB": 5,
        },
    ]


def test_framework_with_only_profiler_rule_specified(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        rules=[ProfilerRule.sagemaker(rule_configs.CPUBottleneck(gpu_threshold=60))],
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["profiler_config"] == {
        "DisableProfiler": False,
        "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
    }
    assert args["profiler_rule_configs"] == [
        {
            "RuleConfigurationName": "CPUBottleneck",
            "RuleEvaluatorImage": "895741380848.dkr.ecr.us-west-2.amazonaws.com/sagemaker-debugger-rules:latest",
            "RuleParameters": {
                "rule_to_invoke": "CPUBottleneck",
                "cpu_threshold": "90",
                "gpu_threshold": "60",
                "patience": "1000",
                "scan_interval_us": "60000000",
                "threshold": "50",
            },
        }
    ]


@patch("time.time", return_value=TIME)
def test_framework_with_profiler_config_without_s3_output_path(time, sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        profiler_config=ProfilerConfig(system_monitor_interval_millis=1000),
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["profiler_config"] == {
        "DisableProfiler": False,
        "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
        "ProfilingIntervalInMilliseconds": 1000,
    }


@pytest.mark.parametrize("region", PROFILER_UNSUPPORTED_REGIONS)
def test_framework_with_no_default_profiler_in_unsupported_region(region):
    boto_mock = Mock(name="boto_session", region_name=region)
    sms = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=region,
        config=None,
        local_mode=False,
        s3_client=None,
        s3_resource=None,
        settings=SessionSettings(),
    )
    sms.sagemaker_config = {}
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sms,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    f.fit("s3://mydata")
    sms.train.assert_called_once()
    _, args = sms.train.call_args
    # assert args.get("profiler_config") == {"DisableProfiler": True}
    # temporarily check if "DisableProfiler" flag is true until s3_output is changed to optional in service
    assert args.get("profiler_config")["DisableProfiler"] is True
    assert args.get("profiler_rule_configs") is None


@pytest.mark.parametrize("region", PROFILER_UNSUPPORTED_REGIONS)
def test_framework_with_debugger_config_set_up_in_unsupported_region(region):
    with pytest.raises(ValueError) as error:
        boto_mock = Mock(name="boto_session", region_name=region)
        sms = MagicMock(
            name="sagemaker_session",
            boto_session=boto_mock,
            boto_region_name=region,
            config=None,
            local_mode=False,
            s3_client=None,
            s3_resource=None,
            settings=SessionSettings(),
        )
        sms.sagemaker_config = {}
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sms,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            debugger_hook_config=DebuggerHookConfig(s3_output_path="s3://output"),
        )
        f.fit("s3://mydata")

    assert "Current region does not support debugger but debugger hook config is set!" in str(error)


@pytest.mark.parametrize("region", PROFILER_UNSUPPORTED_REGIONS)
def test_framework_enable_profiling_in_unsupported_region(region):
    with pytest.raises(ValueError) as error:
        boto_mock = Mock(name="boto_session", region_name=region)
        sms = MagicMock(
            name="sagemaker_session",
            boto_session=boto_mock,
            boto_region_name=region,
            config=None,
            local_mode=False,
            s3_client=None,
            s3_resource=None,
            settings=SessionSettings(),
        )
        sms.sagemaker_config = {}
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sms,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
        )
        f.fit("s3://mydata")
        f.enable_default_profiling()

    assert "Current region does not support profiler / debugger!" in str(error)


@pytest.mark.parametrize("region", PROFILER_UNSUPPORTED_REGIONS)
def test_framework_update_profiling_in_unsupported_region(region):
    with pytest.raises(ValueError) as error:
        boto_mock = Mock(name="boto_session", region_name=region)
        sms = MagicMock(
            name="sagemaker_session",
            boto_session=boto_mock,
            boto_region_name=region,
            config=None,
            local_mode=False,
            s3_client=None,
            s3_resource=None,
            settings=SessionSettings(),
        )
        sms.sagemaker_config = {}
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sms,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
        )
        f.fit("s3://mydata")
        f.update_profiler(system_monitor_interval_millis=1000)

    assert "Current region does not support profiler / debugger!" in str(error)


@pytest.mark.parametrize("region", PROFILER_UNSUPPORTED_REGIONS)
def test_framework_disable_profiling_in_unsupported_region(region):
    with pytest.raises(ValueError) as error:
        boto_mock = Mock(name="boto_session", region_name=region)
        sms = MagicMock(
            name="sagemaker_session",
            boto_session=boto_mock,
            boto_region_name=region,
            config=None,
            local_mode=False,
            s3_client=None,
            s3_resource=None,
            settings=SessionSettings(),
        )
        sms.sagemaker_config = {}
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sms,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
        )
        f.fit("s3://mydata")
        f.disable_profiling()

    assert "Current region does not support profiler / debugger!" in str(error)


def test_framework_with_profiler_config_and_profiler_disabled(sagemaker_session):
    with pytest.raises(RuntimeError) as error:
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            profiler_config=ProfilerConfig(),
            disable_profiler=True,
        )
        f.fit("s3://mydata")
    # assert "profiler_config cannot be set when disable_profiler is True." in str(error)
    assert "profiler_config.disable_profiler cannot be False when disable_profiler is True." in str(
        error
    )


def test_framework_with_profiler_rule_and_profiler_disabled(sagemaker_session):
    profiler_custom_rule = ProfilerRule.custom(
        name="CustomProfilerRule",
        image_uri="RuleImageUri",
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=5,
    )
    with pytest.raises(RuntimeError) as error:
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            rules=[profiler_custom_rule],
            disable_profiler=True,
        )
        f.fit("s3://mydata")
    assert "ProfilerRule cannot be set when disable_profiler is True." in str(error)


def test_framework_with_enabling_default_profiling_when_profiler_is_already_enabled(
    sagemaker_session, training_job_description
):
    with pytest.raises(ValueError) as error:
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
        )
        f.fit("s3://mydata")
        training_job_description["ProfilingStatus"] = "Enabled"
        f.enable_default_profiling()
    assert (
        "Debugger monitoring is already enabled. To update the profiler_config parameter "
        "and the Debugger profiling rules, please use the update_profiler function." in str(error)
    )


@patch("time.time", return_value=TIME)
def test_framework_with_enabling_default_profiling(
    time, sagemaker_session, training_job_description
):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        disable_profiler=True,
    )
    f.fit("s3://mydata")
    training_job_description["ProfilingStatus"] = "Disabled"
    f.enable_default_profiling()
    sagemaker_session.update_training_job.assert_called_once()
    _, args = sagemaker_session.update_training_job.call_args
    assert args["profiler_config"] == {
        "DisableProfiler": False,
        "S3OutputPath": "s3://{}/".format(BUCKET_NAME),
    }


@patch("time.time", return_value=TIME)
def test_framework_with_enabling_default_profiling_with_existed_s3_output_path(
    time, sagemaker_session, training_job_description
):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        disable_profiler=True,
    )
    f.fit("s3://mydata")
    training_job_description["ProfilingStatus"] = "Disabled"
    training_job_description["ProfilerConfig"] = {
        "S3OutputPath": "s3://custom/",
        "ProfilingIntervalInMilliseconds": 1000,
    }
    f.enable_default_profiling()
    sagemaker_session.update_training_job.assert_called_once()
    _, args = sagemaker_session.update_training_job.call_args
    assert args["profiler_config"] == {"DisableProfiler": False, "S3OutputPath": "s3://custom/"}


def test_framework_with_disabling_profiling_when_profiler_is_already_disabled(
    sagemaker_session, training_job_description
):
    with pytest.raises(ValueError) as error:
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
        )
        f.fit("s3://mydata")
        training_job_description["ProfilingStatus"] = "Disabled"
        f.disable_profiling()
    assert "Profiler is already disabled." in str(error)


def test_framework_with_disabling_profiling(sagemaker_session, training_job_description):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    f.fit("s3://mydata")
    training_job_description["ProfilingStatus"] = "Enabled"
    f.disable_profiling()
    sagemaker_session.update_training_job.assert_called_once()
    _, args = sagemaker_session.update_training_job.call_args
    # assert args["profiler_config"] == {"DisableProfiler": True}
    # temporarily check if "DisableProfiler" flag is true until s3_output is changed to optional in service
    assert args.get("profiler_config")["DisableProfiler"] is True


def test_framework_with_update_profiler_when_no_training_job(sagemaker_session):
    with pytest.raises(ValueError) as error:
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
        )
        f.update_profiler(system_monitor_interval_millis=1000)
    assert "Estimator is not associated with a training job" in str(error)


def test_framework_with_update_profiler_without_any_parameter(sagemaker_session):
    with pytest.raises(ValueError) as error:
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
        )
        f.fit("s3://mydata")
        f.update_profiler()
    assert "Please provide profiler config or profiler rule to be updated." in str(error)


def test_framework_with_update_profiler_with_debugger_rule(sagemaker_session):
    with pytest.raises(ValueError) as error:
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
        )
        f.fit("s3://mydata")
        f.update_profiler(rules=[Rule.sagemaker(rule_configs.stalled_training_rule())])
    assert "Please provide ProfilerRule to be updated." in str(error)


def test_framework_with_update_profiler_config(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    f.fit("s3://mydata")
    f.update_profiler(system_monitor_interval_millis=1000)
    sagemaker_session.update_training_job.assert_called_once()
    _, args = sagemaker_session.update_training_job.call_args
    assert args["profiler_config"] == {
        "DisableProfiler": False,
        "ProfilingIntervalInMilliseconds": 1000,
    }
    assert "profiler_rule_configs" not in args


def test_framework_with_update_profiler_report_rule(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    f.fit("s3://mydata")
    f.update_profiler(
        rules=[
            ProfilerRule.sagemaker(rule_configs.ProfilerReport(), name="CustomProfilerReportRule")
        ]
    )
    sagemaker_session.update_training_job.assert_called_once()
    _, args = sagemaker_session.update_training_job.call_args
    assert args["profiler_rule_configs"] == [
        {
            "RuleConfigurationName": "CustomProfilerReportRule",
            "RuleEvaluatorImage": "895741380848.dkr.ecr.us-west-2.amazonaws.com/sagemaker-debugger-rules:latest",
            "RuleParameters": {"rule_to_invoke": "ProfilerReport"},
        }
    ]
    assert args["profiler_config"]["DisableProfiler"] is False


def test_framework_with_disable_framework_metrics(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    f.fit("s3://mydata")
    f.update_profiler(disable_framework_metrics=True)
    sagemaker_session.update_training_job.assert_called_once()
    _, args = sagemaker_session.update_training_job.call_args
    assert args["profiler_config"] == {"DisableProfiler": False, "ProfilingParameters": {}}
    assert "profiler_rule_configs" not in args


def test_framework_with_disable_framework_metrics_and_update_system_metrics(
    sagemaker_session,
):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    f.fit("s3://mydata")
    f.update_profiler(system_monitor_interval_millis=1000, disable_framework_metrics=True)
    sagemaker_session.update_training_job.assert_called_once()
    _, args = sagemaker_session.update_training_job.call_args
    assert args["profiler_config"] == {
        "DisableProfiler": False,
        "ProfilingIntervalInMilliseconds": 1000,
        "ProfilingParameters": {},
    }
    assert "profiler_rule_configs" not in args


def test_framework_with_disable_framework_metrics_and_update_framework_params(
    sagemaker_session,
):
    with pytest.raises(ValueError) as error:
        f = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
        )
        f.fit("s3://mydata")
        f.update_profiler(
            framework_profile_params=FrameworkProfile(), disable_framework_metrics=True
        )
    assert "framework_profile_params cannot be set when disable_framework_metrics is True" in str(
        error
    )


def test_framework_with_update_profiler_config_and_profiler_rule(sagemaker_session):
    profiler_custom_rule = ProfilerRule.custom(
        name="CustomProfilerRule",
        image_uri="RuleImageUri",
        instance_type=INSTANCE_TYPE,
        volume_size_in_gb=5,
    )
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    f.fit("s3://mydata")
    f.update_profiler(rules=[profiler_custom_rule], system_monitor_interval_millis=1000)
    sagemaker_session.update_training_job.assert_called_once()
    _, args = sagemaker_session.update_training_job.call_args
    assert args["profiler_config"] == {
        "DisableProfiler": False,
        "ProfilingIntervalInMilliseconds": 1000,
    }
    assert args["profiler_rule_configs"] == [
        {
            "InstanceType": "c4.4xlarge",
            "RuleConfigurationName": "CustomProfilerRule",
            "RuleEvaluatorImage": "RuleImageUri",
            "VolumeSizeInGB": 5,
        }
    ]


def test_training_job_with_rule_job_summary(sagemaker_session, training_job_description):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    f.fit("s3://mydata")
    training_job_description["DebugRuleEvaluationStatuses"] = [
        {
            "RuleConfigurationName": "debugger_rule",
            "RuleEvaluationJobArn": "debugger_rule_job_arn",
            "RuleEvaluationStatus": "InProgress",
        }
    ]
    training_job_description["ProfilerRuleEvaluationStatuses"] = [
        {
            "RuleConfigurationName": "profiler_rule_1",
            "RuleEvaluationJobArn": "profiler_rule_job_arn_1",
            "RuleEvaluationStatus": "InProgress",
        },
        {
            "RuleConfigurationName": "profiler_rule_2",
            "RuleEvaluationJobArn": "profiler_rule_job_arn_2",
            "RuleEvaluationStatus": "ERROR",
        },
    ]
    job_summary = f.latest_training_job.rule_job_summary()
    assert job_summary == [
        {
            "RuleConfigurationName": "debugger_rule",
            "RuleEvaluationJobArn": "debugger_rule_job_arn",
            "RuleEvaluationStatus": "InProgress",
        },
        {
            "RuleConfigurationName": "profiler_rule_1",
            "RuleEvaluationJobArn": "profiler_rule_job_arn_1",
            "RuleEvaluationStatus": "InProgress",
        },
        {
            "RuleConfigurationName": "profiler_rule_2",
            "RuleEvaluationJobArn": "profiler_rule_job_arn_2",
            "RuleEvaluationStatus": "ERROR",
        },
    ]


def test_framework_with_spot_and_checkpoints(sagemaker_session):
    f = DummyFramework(
        "my_script.py",
        role="DummyRole",
        instance_count=3,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        volume_size=123,
        volume_kms_key="volumekms",
        max_run=456,
        input_mode="inputmode",
        output_path="outputpath",
        output_kms_key="outputkms",
        base_job_name="basejobname",
        tags=[{"foo": "bar"}],
        subnets=["123", "456"],
        security_group_ids=["789", "012"],
        metric_definitions=[{"Name": "validation-rmse", "Regex": "validation-rmse=(\\d+)"}],
        encrypt_inter_container_traffic=True,
        use_spot_instances=True,
        max_wait=500,
        checkpoint_s3_uri="s3://mybucket/checkpoints/",
        checkpoint_local_path="/tmp/checkpoints",
    )
    _TrainingJob.start_new(f, "s3://mydata", None)
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args == {
        "input_mode": "inputmode",
        "tags": [{"foo": "bar"}],
        "hyperparameters": {},
        "image_uri": "fakeimage",
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
        "stop_condition": {"MaxRuntimeInSeconds": 456, "MaxWaitTimeInSeconds": 500},
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
        "use_spot_instances": True,
        "checkpoint_s3_uri": "s3://mybucket/checkpoints/",
        "enable_network_isolation": False,
        "checkpoint_local_path": "/tmp/checkpoints",
        "environment": None,
        "experiment_config": None,
        "retry_strategy": None,
    }


def test_framework_init_s3_entry_point_invalid(sagemaker_session):
    with pytest.raises(ValueError) as error:
        DummyFramework(
            "s3://remote-script-because-im-mistaken",
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
        )
    assert "Must be a path to a local file" in str(error)


def test_sagemaker_s3_uri_invalid(sagemaker_session):
    with pytest.raises(ValueError) as error:
        t = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
        )
        t.fit("thisdoesntstartwiths3")
    assert "must be a valid S3 or FILE URI" in str(error)


def test_sagemaker_model_s3_uri_invalid(sagemaker_session):
    with pytest.raises(ValueError) as error:
        t = DummyFramework(
            entry_point=SCRIPT_PATH,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
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
            instance_count=INSTANCE_COUNT,
            instance_type=INSTANCE_TYPE,
            model_uri="file://notins3.tar.gz",
        )
        t.fit("s3://mydata")
    assert "File URIs are supported in local mode only" in str(error)


def test_sagemaker_model_default_channel_name(sagemaker_session):
    f = DummyFramework(
        entry_point="my_script.py",
        role="DummyRole",
        instance_count=3,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        model_uri="s3://model-bucket/prefix/model.tar.gz",
    )
    _TrainingJob.start_new(f, {}, None)
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
        instance_count=3,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        model_uri="s3://model-bucket/prefix/model.tar.gz",
        model_channel_name="testModelChannel",
    )
    _TrainingJob.start_new(f, {}, None)
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


def test_framework_with_remote_debug_config(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
        enable_remote_debug=True,
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["remote_debug_config"]["EnableRemoteDebug"]
    assert f.get_remote_debug_config()["EnableRemoteDebug"]


def test_framework_without_remote_debug_config(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args.get("remote_debug_config") is None
    assert f.get_remote_debug_config() is None


def test_framework_enable_remote_debug(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    f.fit("s3://mydata")
    f.enable_remote_debug()

    sagemaker_session.update_training_job.assert_called_once()
    _, args = sagemaker_session.update_training_job.call_args
    assert args["remote_debug_config"] == {
        "EnableRemoteDebug": True,
    }
    assert f.get_remote_debug_config()["EnableRemoteDebug"]
    assert len(args) == 2


def test_framework_disable_remote_debug(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        enable_remote_debug=True,
    )
    f.fit("s3://mydata")
    f.disable_remote_debug()

    sagemaker_session.update_training_job.assert_called_once()
    _, args = sagemaker_session.update_training_job.call_args
    assert args["remote_debug_config"] == {
        "EnableRemoteDebug": False,
    }
    assert not f.get_remote_debug_config()["EnableRemoteDebug"]
    assert len(args) == 2


def test_framework_with_session_chaining_config(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
        enable_session_tag_chaining=True,
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args["session_chaining_config"]["EnableSessionTagChaining"]
    assert f.get_session_chaining_config()["EnableSessionTagChaining"]


def test_framework_without_session_chaining_config(sagemaker_session):
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
    )
    f.fit("s3://mydata")
    sagemaker_session.train.assert_called_once()
    _, args = sagemaker_session.train.call_args
    assert args.get("SessionTagChaining") is None
    assert f.get_remote_debug_config() is None


@patch("time.strftime", return_value=TIMESTAMP)
def test_custom_code_bucket(time, sagemaker_session):
    code_bucket = "codebucket"
    prefix = "someprefix"
    code_location = "s3://{}/{}".format(code_bucket, prefix)
    t = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        code_location=code_location,
    )

    with pytest.raises(ValueError) as error:
        t.fit("s3://bucket/mydata")
    assert "Expecting 's3' scheme" in str(error)


def test_get_instance_type_gpu(sagemaker_session):
    estimator = Estimator(
        image_uri="some-image",
        role="some_image",
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.p3.16xlarge", 2),
        ],
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
    )

    assert "ml.p3.16xlarge" == estimator._get_instance_type()


def test_estimator_with_output_compression_disabled(sagemaker_session):
    estimator = Estimator(
        image_uri="some-image",
        role="some_image",
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
        disable_output_compression=True,
    )

    assert estimator.disable_output_compression


def test_estimator_with_output_compression_as_default(sagemaker_session):
    estimator = Estimator(
        image_uri="some-image",
        role="some_image",
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
    )

    assert not estimator.disable_output_compression


def test_get_instance_type_cpu(sagemaker_session):
    estimator = Estimator(
        image_uri="some-image",
        role="some_image",
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.c5.xlarge", 2),
        ],
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
    )

    assert "ml.c4.xlarge" == estimator._get_instance_type()


def test_get_instance_type_no_instance_groups(sagemaker_session):
    estimator = Estimator(
        image_uri="some-image",
        role="some_image",
        instance_type="ml.c4.xlarge",
        instance_count=1,
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
    )

    assert "ml.c4.xlarge" == estimator._get_instance_type()


def test_get_instance_type_no_instance_groups_or_instance_type(sagemaker_session):
    estimator = Estimator(
        image_uri="some-image",
        role="some_image",
        instance_type=None,
        instance_count=None,
        instance_groups=None,
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
    )
    with pytest.raises(ValueError) as error:
        estimator._get_instance_type()

    assert (
        "instance_groups must be set if instance_type is not set and instance_groups must be a list."
        in str(error)
    )


def test_augmented_manifest(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    fw.fit(
        inputs=TrainingInput(
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    fw.fit(inputs=TrainingInput("s3://mybucket/train_manifest", input_mode=expected_input_mode))

    actual_input_mode = sagemaker_session.method_calls[1][2]["input_mode"]
    assert actual_input_mode == expected_input_mode


def test_shuffle_config(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    fw.fit(inputs=TrainingInput("s3://mybucket/train_manifest", shuffle_config=ShuffleConfig(100)))
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
        spec=sagemaker.local.LocalSession,
        settings=SessionSettings(),
    )

    sms.sagemaker_config = {}
    t = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sms,
        instance_count=1,
        instance_type="local",
        base_job_name=IMAGE_URI,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        base_job_name=IMAGE_URI,
        hyperparameters={123: [456], "learning_rate": 0.1},
    )
    t.fit("s3://{}".format(uri))

    expected_hyperparameters = BASE_HP.copy()
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )

    t.fit("s3://{}".format(uri))

    expected_hyperparameters = BASE_HP.copy()
    expected_hyperparameters["sagemaker_container_log_level"] = str(logging.INFO)
    expected_hyperparameters["sagemaker_region"] = '"us-west-2"'

    actual_hyperparameter = sagemaker_session.method_calls[1][2]["hyperparameters"]
    assert actual_hyperparameter == expected_hyperparameters
    assert sagemaker_session.wait_for_job.assert_called_once


def test_attach_framework(sagemaker_session, training_job_description):
    training_job_description["VpcConfig"] = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    training_job_description["EnableNetworkIsolation"] = True

    framework_estimator = DummyFramework.attach(
        training_job_name="neo", sagemaker_session=sagemaker_session
    )
    assert framework_estimator._current_job_name == "neo"
    assert framework_estimator.latest_training_job.job_name == "neo"
    assert framework_estimator.role == "arn:aws:iam::366:role/SageMakerRole"
    assert framework_estimator.instance_count == 1
    assert framework_estimator.max_run == 24 * 60 * 60
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
    assert framework_estimator.enable_network_isolation() is True


def test_attach_no_logs(sagemaker_session, training_job_description):
    Estimator.attach(training_job_name="job", sagemaker_session=sagemaker_session)
    sagemaker_session.logs_for_job.assert_not_called()


def test_logs(sagemaker_session, training_job_description):
    estimator = Estimator.attach(training_job_name="job", sagemaker_session=sagemaker_session)
    estimator.logs()
    sagemaker_session.logs_for_job.assert_called_with(estimator.latest_training_job.name, wait=True)


def test_attach_without_hyperparameters(sagemaker_session, training_job_description):
    del training_job_description["HyperParameters"]
    estimator = Estimator.attach(training_job_name="job", sagemaker_session=sagemaker_session)
    assert estimator.hyperparameters() == {}


def test_attach_framework_with_tuning(sagemaker_session, training_job_description):
    training_job_description["HyperParameters"]["_tuning_objective_metric"] = "Validation-accuracy"
    framework_estimator = DummyFramework.attach(
        training_job_name="neo", sagemaker_session=sagemaker_session
    )
    assert framework_estimator.latest_training_job.job_name == "neo"
    assert framework_estimator.role == "arn:aws:iam::366:role/SageMakerRole"
    assert framework_estimator.instance_count == 1
    assert framework_estimator.max_run == 24 * 60 * 60
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


def test_attach_framework_with_model_channel(sagemaker_session, training_job_description):
    s3_uri = "s3://some/s3/path/model.tar.gz"
    training_job_description["InputDataConfig"] = [
        {
            "ChannelName": "model",
            "InputMode": "File",
            "DataSource": {"S3DataSource": {"S3Uri": s3_uri}},
        }
    ]

    framework_estimator = DummyFramework.attach(
        training_job_name="neo", sagemaker_session=sagemaker_session
    )
    assert framework_estimator.model_uri is s3_uri
    assert framework_estimator.encrypt_inter_container_traffic is False


def test_attach_framework_with_inter_container_traffic_encryption_flag(
    sagemaker_session, training_job_description
):
    training_job_description["EnableInterContainerTrafficEncryption"] = True
    framework_estimator = DummyFramework.attach(
        training_job_name="neo", sagemaker_session=sagemaker_session
    )

    assert framework_estimator.encrypt_inter_container_traffic is True


def test_attach_framework_base_from_generated_name(sagemaker_session, training_job_description):
    base_job_name = "neo"
    framework_estimator = DummyFramework.attach(
        training_job_name=utils.name_from_base("neo"), sagemaker_session=sagemaker_session
    )

    assert framework_estimator.base_job_name == base_job_name


@patch("time.strftime", return_value=TIMESTAMP)
def test_fit_verify_job_name(strftime, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        tags=TAGS,
        encrypt_inter_container_traffic=True,
    )
    fw.fit(inputs=TrainingInput("s3://mybucket/train"))

    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]

    assert train_kwargs["image_uri"] == IMAGE_URI
    assert train_kwargs["input_mode"] == "File"
    assert train_kwargs["tags"] == TAGS
    assert train_kwargs["job_name"] == JOB_NAME
    assert train_kwargs["encrypt_inter_container_traffic"] is True
    assert fw.latest_training_job.name == JOB_NAME


@pytest.mark.parametrize(
    "debugger_hook_config_direct_input, sagemaker_config, expected_debugger_hook_config_output",
    [
        (None, None, S3_OUTPUT_PATH_FROM_SESSION_S3_DEFAULT_CONFIG),
        (True, None, S3_OUTPUT_PATH_FROM_SESSION_S3_DEFAULT_CONFIG),
        (False, None, False),
        (HOOK_CONFIG, None, HOOK_CONFIG.s3_output_path),
        (HOOK_CONFIG_WITHOUT_S3_PATH, None, S3_OUTPUT_PATH_FROM_SESSION_S3_DEFAULT_CONFIG),
        (None, SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_FALSE, False),
        (
            True,
            SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_FALSE,
            S3_OUTPUT_PATH_FROM_SESSION_S3_DEFAULT_CONFIG,
        ),
        (False, SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_FALSE, False),
        (
            HOOK_CONFIG,
            SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_FALSE,
            HOOK_CONFIG.s3_output_path,
        ),
        (
            HOOK_CONFIG_WITHOUT_S3_PATH,
            SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_FALSE,
            S3_OUTPUT_PATH_FROM_SESSION_S3_DEFAULT_CONFIG,
        ),
        (
            None,
            SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_TRUE,
            S3_OUTPUT_PATH_FROM_SESSION_S3_DEFAULT_CONFIG,
        ),
        (
            True,
            SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_TRUE,
            S3_OUTPUT_PATH_FROM_SESSION_S3_DEFAULT_CONFIG,
        ),
        (False, SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_TRUE, False),
        (
            HOOK_CONFIG,
            SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_TRUE,
            HOOK_CONFIG.s3_output_path,
        ),
        (
            HOOK_CONFIG_WITHOUT_S3_PATH,
            SAGEMAKER_CONFIG_TRAINING_JOB_WITH_DEBUG_HOOK_CONFIG_AS_TRUE,
            S3_OUTPUT_PATH_FROM_SESSION_S3_DEFAULT_CONFIG,
        ),
    ],
)
def test_prepare_for_training_for_debugger_hook_config_value_combinations(
    sagemaker_session,
    sagemaker_config,
    debugger_hook_config_direct_input,
    expected_debugger_hook_config_output,
):
    sagemaker_session.sagemaker_config = sagemaker_config
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        debugger_hook_config=debugger_hook_config_direct_input,
    )
    fw._prepare_for_training()

    if expected_debugger_hook_config_output is False:
        assert fw.debugger_hook_config == expected_debugger_hook_config_output
    else:
        assert fw.debugger_hook_config.s3_output_path == expected_debugger_hook_config_output


def test_prepare_for_training_unique_job_name_generation(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        base_job_name="some",
    )
    fw._prepare_for_training(job_name="use_it")
    assert "use_it" == fw._current_job_name


@patch("time.strftime", return_value=TIMESTAMP)
def test_prepare_for_training_force_name_generation(strftime, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        base_job_name="some",
    )
    fw.base_job_name = None
    fw._prepare_for_training()
    assert JOB_NAME == fw._current_job_name


@patch("sagemaker.git_utils.git_clone_repo")
def test_git_support_with_branch_and_commit_succeed(git_clone_repo, sagemaker_session):
    git_clone_repo.side_effect = lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": [],
    }
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    entry_point = "entry_point"
    fw = DummyFramework(
        entry_point=entry_point,
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
def test_git_support_git_clone_fail(git_clone_repo, sagemaker_session):
    git_config = {"repo": "https://github.com/aws/no-such-repo.git", "branch": BRANCH}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        fw.fit()
    assert "returned non-zero exit status" in str(error.value)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git checkout branch-that-does-not-exist"
    ),
)
def test_git_support_branch_not_exist(git_clone_repo, sagemaker_session):
    git_config = {"repo": GIT_REPO, "branch": "branch-that-does-not-exist", "commit": COMMIT}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        fw.fit()
    assert "returned non-zero exit status" in str(error.value)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git checkout commit-sha-that-does-not-exist"
    ),
)
def test_git_support_commit_not_exist(git_clone_repo, sagemaker_session):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": "commit-sha-that-does-not-exist"}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        fw.fit()
    assert "returned non-zero exit status" in str(error.value)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=ValueError("Entry point does not exist in the repo."),
)
def test_git_support_entry_point_not_exist(git_clone_repo, sagemaker_session):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    fw = DummyFramework(
        entry_point="entry_point_that_does_not_exist",
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    with pytest.raises(ValueError) as error:
        fw.fit()
    assert "Entry point does not exist in the repo." in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=ValueError("Source directory does not exist in the repo."),
)
def test_git_support_source_dir_not_exist(git_clone_repo, sagemaker_session):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        source_dir="source_dir_that_does_not_exist",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    with pytest.raises(ValueError) as error:
        fw.fit()
    assert "Source directory does not exist in the repo." in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=ValueError("Dependency no-such-dir does not exist in the repo."),
)
def test_git_support_dependencies_not_exist(git_clone_repo, sagemaker_session):
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    fw = DummyFramework(
        entry_point="entry_point",
        git_config=git_config,
        source_dir="source_dir",
        dependencies=["foo", "no-such-dir"],
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        fw.fit()
    assert "returned non-zero exit status" in str(error.value)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    },
)
def test_git_support_codecommit_with_username_and_password_succeed(
    git_clone_repo, sagemaker_session
):
    git_config = {
        "repo": CODECOMMIT_REPO,
        "branch": CODECOMMIT_BRANCH,
        "username": "username",
        "password": "passw0rd!",
    }
    entry_point = "entry_point"
    fw = DummyFramework(
        entry_point=entry_point,
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
def test_git_support_codecommit_with_ssh_no_passphrase_needed(git_clone_repo, sagemaker_session):
    git_config = {"repo": CODECOMMIT_REPO_SSH, "branch": CODECOMMIT_BRANCH}
    entry_point = "entry_point"
    fw = DummyFramework(
        entry_point=entry_point,
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    fw.fit()
    git_clone_repo.assert_called_once_with(git_config, entry_point, None, [])
    assert fw.entry_point == "/tmp/repo_dir/entry_point"


@patch("time.strftime", return_value=TIMESTAMP)
def test_init_with_source_dir_s3(strftime, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_NAME,
        source_dir="s3://location",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    fw._prepare_for_training()

    expected_hyperparameters = {
        "sagemaker_program": SCRIPT_NAME,
        "sagemaker_job_name": JOB_NAME,
        "sagemaker_container_log_level": logging.INFO,
        "sagemaker_submit_directory": "s3://location",
        "sagemaker_region": "us-west-2",
    }
    assert fw._hyperparameters == expected_hyperparameters


@patch("sagemaker.estimator.name_from_base", return_value=MODEL_IMAGE)
def test_framework_transformer_creation(name_from_base, sagemaker_session):
    vpc_config = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        subnets=vpc_config["Subnets"],
        security_group_ids=vpc_config["SecurityGroupIds"],
    )
    fw.latest_training_job = _TrainingJob(sagemaker_session, JOB_NAME)

    transformer = fw.transformer(INSTANCE_COUNT, INSTANCE_TYPE)

    name_from_base.assert_called_with(IMAGE_URI)
    sagemaker_session.create_model.assert_called_with(
        name=MODEL_IMAGE,
        role=ROLE,
        container_defs=MODEL_CONTAINER_DEF,
        tags=None,
        vpc_config=vpc_config,
        enable_network_isolation=False,
    )

    assert isinstance(transformer, Transformer)
    assert transformer.sagemaker_session == sagemaker_session
    assert transformer.instance_count == INSTANCE_COUNT
    assert transformer.instance_type == INSTANCE_TYPE
    assert transformer.model_name == MODEL_IMAGE
    assert transformer.tags is None
    assert transformer.env == {}


@patch("sagemaker.model.utils.name_from_image", return_value=MODEL_IMAGE)
def test_framework_transformer_creation_with_optional_params(name_from_image, sagemaker_session):
    base_name = "foo"
    vpc_config = {"Subnets": ["foo"], "SecurityGroupIds": ["bar"]}
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        base_job_name=base_name,
        subnets=vpc_config["Subnets"],
        security_group_ids=vpc_config["SecurityGroupIds"],
        enable_network_isolation=False,
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
    new_vpc_config = {"Subnets": ["x"], "SecurityGroupIds": ["y"]}
    model_name = "model-name"

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
        vpc_config_override=new_vpc_config,
        enable_network_isolation=True,
        model_name=model_name,
    )

    sagemaker_session.create_model.assert_called_with(
        name=model_name,
        role=new_role,
        container_defs=MODEL_CONTAINER_DEF,
        vpc_config=new_vpc_config,
        tags=TAGS,
        enable_network_isolation=True,
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
    assert transformer.model_name == model_name


def test_ensure_latest_training_job(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )
    fw.latest_training_job = Mock(name="training_job")

    fw._ensure_latest_training_job()


def test_ensure_latest_training_job_failure(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )

    with pytest.raises(ValueError) as e:
        fw._ensure_latest_training_job()
    assert "Estimator is not associated with a training job" in str(e)


@patch("sagemaker.estimator.Estimator.create_model")
@patch("sagemaker.estimator.name_from_base")
def test_estimator_transformer_creation(name_from_base, create_model, sagemaker_session):
    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )
    estimator.latest_training_job = _TrainingJob(sagemaker_session, JOB_NAME)

    model_name = "model_name"
    name_from_base.return_value = model_name
    transformer = estimator.transformer(INSTANCE_COUNT, INSTANCE_TYPE)

    create_model.assert_called_with(
        vpc_config_override=vpc_utils.VPC_CONFIG_DEFAULT,
        model_kms_key=estimator.output_kms_key,
        enable_network_isolation=False,
    )

    assert isinstance(transformer, Transformer)
    assert transformer.sagemaker_session == sagemaker_session
    assert transformer.instance_count == INSTANCE_COUNT
    assert transformer.instance_type == INSTANCE_TYPE
    assert transformer.model_name == model_name
    assert transformer.tags is None


@patch("sagemaker.estimator.Estimator.create_model")
def test_estimator_transformer_creation_with_optional_params(create_model, sagemaker_session):
    base_name = "foo"
    kms_key = "key"

    estimator = Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        base_job_name=base_name,
        output_kms_key=kms_key,
    )
    estimator.latest_training_job = _TrainingJob(sagemaker_session, JOB_NAME)

    strategy = "MultiRecord"
    assemble_with = "Line"
    accept = "text/csv"
    max_concurrent_transforms = 1
    max_payload = 6
    env = {"FOO": "BAR"}
    new_vpc_config = {"Subnets": ["x"], "SecurityGroupIds": ["y"]}
    model_name = "model-name"

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
        vpc_config_override=new_vpc_config,
        enable_network_isolation=True,
        model_name=model_name,
    )

    create_model.assert_called_with(
        vpc_config_override=new_vpc_config, model_kms_key=kms_key, enable_network_isolation=True
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
    assert transformer.model_name == model_name


# _TrainingJob 'utils'
def test_start_new(sagemaker_session):
    training_job = _TrainingJob(sagemaker_session, JOB_NAME)
    hyperparameters = {"mock": "hyperparameters"}
    inputs = "s3://mybucket/train"

    estimator = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters,
    )

    exp_config = {
        "ExperimentName": "exp",
        "TrialName": "t",
        "TrialComponentDisplayName": "tc",
        "RunName": "rn",
    }

    started_training_job = training_job.start_new(estimator, inputs, experiment_config=exp_config)
    called_args = sagemaker_session.train.call_args

    assert started_training_job.sagemaker_session == sagemaker_session
    assert called_args[1]["hyperparameters"] == hyperparameters
    assert called_args[1]["experiment_config"] == exp_config
    sagemaker_session.train.assert_called_once()


def test_start_new_not_local_mode_error(sagemaker_session):
    training_job = _TrainingJob(sagemaker_session, JOB_NAME)
    inputs = "file://mybucket/train"

    estimator = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )
    with pytest.raises(ValueError) as error:
        training_job.start_new(estimator, inputs, None)
        assert "File URIs are supported in local mode only. Please use a S3 URI instead." == str(
            error
        )


def test_container_log_level(sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        container_log_level=logging.DEBUG,
    )
    fw.fit(inputs=TrainingInput("s3://mybucket/train"))

    _, _, train_kwargs = sagemaker_session.train.mock_calls[0]
    assert train_kwargs["hyperparameters"]["sagemaker_container_log_level"] == "10"


@patch("sagemaker.utils")
def test_same_code_location_keeps_kms_key(utils, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        output_kms_key="kms-key",
    )

    fw.fit(wait=False)

    obj = sagemaker_session.boto_session.resource("s3").Object

    obj.assert_called_with("another-location", "%s/source/sourcedir.tar.gz" % fw._current_job_name)
    extra_args = {"ServerSideEncryption": "aws:kms"}
    obj().upload_file.assert_called_with(utils.create_tar_file(), ExtraArgs=extra_args)


@patch("sagemaker.utils")
def test_default_code_location_uses_output_path(utils, sagemaker_session):
    fw = DummyFramework(
        entry_point=SCRIPT_PATH,
        role="DummyRole",
        sagemaker_session=sagemaker_session,
        output_path="s3://output_path",
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
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
    "image_uri": IMAGE_URI,
    "input_config": None,
    "input_mode": "File",
    "output_config": {"S3OutputPath": OUTPUT_PATH},
    "profiler_config": {"DisableProfiler": False, "S3OutputPath": OUTPUT_PATH},
    "resource_config": {
        "InstanceCount": INSTANCE_COUNT,
        "InstanceType": INSTANCE_TYPE,
        "VolumeSizeInGB": 30,
    },
    "stop_condition": {"MaxRuntimeInSeconds": 86400},
    "retry_strategy": None,
    "tags": None,
    "vpc_config": None,
    "metric_definitions": None,
    "environment": None,
    "enable_network_isolation": False,
    "experiment_config": None,
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

EXP_TRAIN_CALL = dict(BASE_TRAIN_CALL)
EXP_TRAIN_CALL.update(
    {
        "experiment_config": {
            "ExperimentName": "exp",
            "TrialName": "trial",
            "TrialComponentDisplayName": "tc",
            "RunName": "rn",
        }
    }
)


@patch("sagemaker.estimator.name_from_base")
def test_fit_deploy_tags_in_estimator(name_from_base, sagemaker_session):
    tags = [{"Key": "TagtestKey", "Value": "TagtestValue"}]
    estimator = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        tags=tags,
        sagemaker_session=sagemaker_session,
    )

    estimator.fit()

    model_name = "model_name"
    name_from_base.return_value = model_name

    estimator.deploy(INSTANCE_COUNT, INSTANCE_TYPE)

    variant = [
        {
            "InstanceType": "c4.4xlarge",
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialVariantWeight": 1,
            "InitialInstanceCount": 1,
        }
    ]

    name_from_base.assert_called_with(IMAGE_URI)

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=model_name,
        production_variants=variant,
        tags=tags,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        explainer_config_dict=None,
        live_logging=False,
    )

    sagemaker_session.create_model.assert_called_with(
        name=model_name,
        role="DummyRole",
        container_defs={
            "ModelDataUrl": "s3://bucket/model.tar.gz",
            "Environment": {},
            "Image": "fakeimage",
        },
        enable_network_isolation=False,
        vpc_config=None,
        tags=tags,
    )


@patch("sagemaker.estimator.name_from_base")
def test_fit_deploy_tags(name_from_base, sagemaker_session):
    estimator = Estimator(
        IMAGE_URI, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, sagemaker_session=sagemaker_session
    )

    estimator.fit()

    model_name = "model_name"
    name_from_base.return_value = model_name

    tags = [{"Key": "TagtestKey", "Value": "TagtestValue"}]
    estimator.deploy(INSTANCE_COUNT, INSTANCE_TYPE, tags=tags)

    variant = [
        {
            "InstanceType": "c4.4xlarge",
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialVariantWeight": 1,
            "InitialInstanceCount": 1,
        }
    ]

    name_from_base.assert_called_with(IMAGE_URI)

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=model_name,
        production_variants=variant,
        tags=tags,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        explainer_config_dict=None,
        live_logging=False,
    )

    sagemaker_session.create_model.assert_called_with(
        name=ANY,
        role="DummyRole",
        container_defs={
            "ModelDataUrl": "s3://bucket/model.tar.gz",
            "Environment": {},
            "Image": "fakeimage",
        },
        enable_network_isolation=False,
        vpc_config=None,
        tags=tags,
    )


@patch("sagemaker.estimator.name_from_base")
def test_fit_deploy_uncompressed_s3_model(name_from_base, sagemaker_session):
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job",
        return_value=DESCRIBE_TRAINING_JOB_RESULT_UNCOMPRESSED_S3_MODEL,
    )
    estimator = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )

    estimator.fit()

    model_name = "model_name"
    name_from_base.return_value = model_name

    estimator.deploy(INSTANCE_COUNT, INSTANCE_TYPE)

    variant = [
        {
            "InstanceType": "c4.4xlarge",
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialVariantWeight": 1,
            "InitialInstanceCount": 1,
        }
    ]

    name_from_base.assert_called_with(IMAGE_URI)

    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        name=model_name,
        production_variants=variant,
        kms_key=None,
        wait=True,
        data_capture_config_dict=None,
        async_inference_config_dict=None,
        explainer_config_dict=None,
        tags=None,
        live_logging=False,
    )

    sagemaker_session.create_model.assert_called_with(
        name=model_name,
        role="DummyRole",
        container_defs={
            "ModelDataSource": {
                "S3DataSource": {
                    # S3 URI passed to Createmodel API should have trailing forward slash appeneded
                    "S3Uri": "s3://bucket/model/prefix/",
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            "Environment": {},
            "Image": "fakeimage",
        },
        enable_network_isolation=False,
        vpc_config=None,
        tags=None,
    )


@patch("sagemaker.estimator.name_from_base")
def test_fit_deploy_uncompressed_s3_model_unrecognized_compression_type(
    name_from_base, sagemaker_session
):
    training_job_desc = deepcopy(DESCRIBE_TRAINING_JOB_RESULT_UNCOMPRESSED_S3_MODEL)
    training_job_desc["OutputDataConfig"]["CompressionType"] = "JUNK"
    sagemaker_session.sagemaker_client.describe_training_job = Mock(
        name="describe_training_job",
        return_value=training_job_desc,
    )
    estimator = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
    )

    estimator.fit()

    model_name = "model_name"
    name_from_base.return_value = model_name

    with pytest.raises(
        ValueError,
        match='Unrecognized training job output data compression type "JUNK"',
    ):
        estimator.deploy(INSTANCE_COUNT, INSTANCE_TYPE)

    name_from_base.assert_called_with(IMAGE_URI)

    sagemaker_session.endpoint_from_production_variants.assert_not_called()
    sagemaker_session.create_model.assert_not_called()


@patch("time.time", return_value=TIME)
def test_generic_to_fit_no_input(time, sagemaker_session):
    e = Estimator(
        IMAGE_URI,
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
    assert args["job_name"].startswith(IMAGE_URI)

    args.pop("job_name")
    args.pop("role")
    args.pop("debugger_hook_config")

    assert args == NO_INPUT_TRAIN_CALL


@patch("time.time", return_value=TIME)
def test_generic_to_fit_no_hps(time, sagemaker_session):
    e = Estimator(
        IMAGE_URI,
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
    assert args["job_name"].startswith(IMAGE_URI)

    args.pop("job_name")
    args.pop("role")
    args.pop("debugger_hook_config")

    assert args == BASE_TRAIN_CALL


@patch("time.time", return_value=TIME)
def test_generic_to_fit_with_hps(time, sagemaker_session):
    e = Estimator(
        IMAGE_URI,
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
    assert args["job_name"].startswith(IMAGE_URI)

    args.pop("job_name")
    args.pop("role")
    args.pop("debugger_hook_config")

    assert args == HP_TRAIN_CALL


@patch("time.time", return_value=TIME)
def test_generic_to_fit_with_experiment_config(time, sagemaker_session):
    e = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )

    e.fit(
        inputs={"train": "s3://bucket/training-prefix"},
        experiment_config={
            "ExperimentName": "exp",
            "TrialName": "trial",
            "TrialComponentDisplayName": "tc",
            "RunName": "rn",
        },
    )

    sagemaker_session.train.assert_called_once()
    assert len(sagemaker_session.train.call_args[0]) == 0
    args = sagemaker_session.train.call_args[1]
    assert args["job_name"].startswith(IMAGE_URI)

    args.pop("job_name")
    args.pop("role")
    args.pop("debugger_hook_config")

    assert args == EXP_TRAIN_CALL


def test_generic_to_fit_with_encrypt_inter_container_traffic_flag(sagemaker_session):
    e = Estimator(
        IMAGE_URI,
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


def test_generic_to_fit_with_network_isolation(sagemaker_session):
    e = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
        enable_network_isolation=True,
    )

    e.fit()

    sagemaker_session.train.assert_called_once()
    args = sagemaker_session.train.call_args[1]
    assert args["enable_network_isolation"] is True


def test_generic_to_fit_with_sagemaker_metrics_missing(sagemaker_session):
    e = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )

    e.fit()

    sagemaker_session.train.assert_called_once()
    args = sagemaker_session.train.call_args[1]
    assert "enable_sagemaker_metrics" not in args


def test_add_environment_variables_to_train_args(sagemaker_session):
    e = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
        environment=ENV_INPUT,
    )

    e.fit()

    sagemaker_session.train.assert_called_once()
    args = sagemaker_session.train.call_args[1]
    assert args["environment"] == ENV_INPUT


def test_add_retry_strategy_to_train_args(sagemaker_session):
    e = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
        max_retry_attempts=2,
    )

    e.fit()

    sagemaker_session.train.assert_called_once()
    args = sagemaker_session.train.call_args[1]
    assert args["retry_strategy"] == {"MaximumRetryAttempts": 2}


def test_generic_to_fit_with_sagemaker_metrics_enabled(sagemaker_session):
    e = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
        enable_sagemaker_metrics=True,
    )

    e.fit()

    sagemaker_session.train.assert_called_once()
    args = sagemaker_session.train.call_args[1]
    assert args["enable_sagemaker_metrics"]


def test_generic_to_fit_with_sagemaker_metrics_disabled(sagemaker_session):
    e = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
        enable_sagemaker_metrics=False,
    )

    e.fit()

    sagemaker_session.train.assert_called_once()
    args = sagemaker_session.train.call_args[1]
    assert not args["enable_sagemaker_metrics"]


@patch("time.time", return_value=TIME)
def test_generic_to_deploy(time, sagemaker_session):
    e = Estimator(
        IMAGE_URI,
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
    assert args["job_name"].startswith(IMAGE_URI)

    args.pop("job_name")
    args.pop("role")
    args.pop("debugger_hook_config")

    assert args == HP_TRAIN_CALL

    sagemaker_session.create_model.assert_called_once()
    args, kwargs = sagemaker_session.create_model.call_args
    assert kwargs["name"].startswith(IMAGE_URI)
    assert kwargs["role"] == ROLE
    assert kwargs["container_defs"]["Image"] == IMAGE_URI
    assert kwargs["container_defs"]["ModelDataUrl"] == MODEL_DATA
    assert kwargs["vpc_config"] is None

    assert isinstance(predictor, Predictor)
    assert predictor.endpoint_name.startswith(IMAGE_URI)
    assert predictor.sagemaker_session == sagemaker_session


def test_generic_to_deploy_async(sagemaker_session):
    e = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )

    e.fit()
    s3_output_path = "s3://some-s3-path"
    s3_failure_path = "s3://some-s3-failures-path"

    predictor_async = e.deploy(
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        async_inference_config=AsyncInferenceConfig(
            output_path=s3_output_path, failure_path=s3_failure_path
        ),
    )

    sagemaker_session.create_model.assert_called_once()
    _, kwargs = sagemaker_session.create_model.call_args
    assert isinstance(predictor_async, AsyncPredictor)
    assert predictor_async.endpoint_name.startswith(IMAGE_URI)
    assert predictor_async.sagemaker_session == sagemaker_session


def test_generic_to_deploy_bad_arguments_combination(sagemaker_session):
    e = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )

    e.fit()

    bad_args = (
        {"instance_type": INSTANCE_TYPE},
        {"initial_instance_count": INSTANCE_COUNT},
        {"instance_type": None, "initial_instance_count": None},
    )
    for args in bad_args:
        with pytest.raises(
            ValueError,
            match="Must specify instance type and instance count unless using serverless inference",
        ):
            e.deploy(args)

    with pytest.raises(
        ValueError,
        match="serverless_inference_config needs to be a ServerlessInferenceConfig object",
    ):
        e.deploy(serverless_inference_config={})

    with pytest.raises(
        ValueError,
        match="explainer_config needs to be a ExplainerConfig object",
    ):
        e.deploy(explainer_config={})


def test_generic_to_deploy_network_isolation(sagemaker_session):
    e = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        enable_network_isolation=True,
        sagemaker_session=sagemaker_session,
    )

    e.fit()
    e.deploy(INSTANCE_COUNT, INSTANCE_TYPE)

    sagemaker_session.create_model.assert_called_once()
    _, kwargs = sagemaker_session.create_model.call_args
    assert kwargs["enable_network_isolation"]


@patch("sagemaker.estimator.Estimator.create_model")
def test_generic_to_deploy_kms(create_model, sagemaker_session):
    e = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )
    e.fit()

    model = MagicMock()
    create_model.return_value = model

    endpoint_name = "foo"
    kms_key = "key"
    e.deploy(INSTANCE_COUNT, INSTANCE_TYPE, endpoint_name=endpoint_name, kms_key=kms_key)

    model.deploy.assert_called_with(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        serializer=None,
        deserializer=None,
        accelerator_type=None,
        endpoint_name=endpoint_name,
        tags=None,
        wait=True,
        kms_key=kms_key,
        data_capture_config=None,
        async_inference_config=None,
        serverless_inference_config=None,
        volume_size=None,
        model_data_download_timeout=None,
        container_startup_health_check_timeout=None,
        inference_recommendation_id=None,
        explainer_config=None,
    )


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
        IMAGE_URI,
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
        IMAGE_URI, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, sagemaker_session=sagemaker_session
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
        IMAGE_URI, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, sagemaker_session=sagemaker_session
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
        IMAGE_URI, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, sagemaker_session=sagemaker_session
    )
    e.fit({"train": "s3://bucket/training-prefix"})
    e.deploy(INSTANCE_COUNT, INSTANCE_TYPE, accelerator_type=ACCELERATOR_TYPE)

    args = e.sagemaker_session.endpoint_from_production_variants.call_args[1]
    assert args["name"].startswith(IMAGE_URI)
    assert args["production_variants"][0]["AcceleratorType"] == ACCELERATOR_TYPE
    assert args["production_variants"][0]["InitialInstanceCount"] == INSTANCE_COUNT
    assert args["production_variants"][0]["InstanceType"] == INSTANCE_TYPE


def test_deploy_with_model_name(sagemaker_session):
    estimator = Estimator(
        IMAGE_URI,
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
    assert kwargs["name"] == model_name


def test_deploy_with_no_model_name(sagemaker_session):
    estimator = Estimator(
        IMAGE_URI,
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
    assert kwargs["name"].startswith(IMAGE_URI)


@patch("sagemaker.estimator.Estimator.create_model")
def test_deploy_with_customized_volume_size_timeout(create_model, sagemaker_session):
    estimator = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )
    estimator.set_hyperparameters(**HYPERPARAMS)
    estimator.fit({"train": "s3://bucket/training-prefix"})
    endpoint_name = "endpoint-name"
    volume_size_gb = 256
    model_data_download_timeout_sec = 600
    startup_health_check_timeout_sec = 600

    model = MagicMock()
    create_model.return_value = model

    estimator.deploy(
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        endpoint_name=endpoint_name,
        volume_size=volume_size_gb,
        model_data_download_timeout=model_data_download_timeout_sec,
        container_startup_health_check_timeout=startup_health_check_timeout_sec,
    )

    model.deploy.assert_called_with(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INSTANCE_COUNT,
        serializer=None,
        deserializer=None,
        accelerator_type=None,
        endpoint_name=endpoint_name,
        tags=None,
        wait=True,
        kms_key=None,
        data_capture_config=None,
        async_inference_config=None,
        serverless_inference_config=None,
        volume_size=volume_size_gb,
        model_data_download_timeout=model_data_download_timeout_sec,
        container_startup_health_check_timeout=startup_health_check_timeout_sec,
        inference_recommendation_id=None,
        explainer_config=None,
    )


def test_register_default_image(sagemaker_session):
    estimator = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )
    estimator.set_hyperparameters(**HYPERPARAMS)
    estimator.fit({"train": "s3://bucket/training-prefix"})

    model_package_name = "test-estimator-register-model"
    content_types = ["application/json"]
    response_types = ["application/json"]
    inference_instances = ["ml.m4.xlarge"]
    transform_instances = ["ml.m4.xlarget"]
    sample_payload_url = "s3://test-bucket/model"
    task = "IMAGE_CLASSIFICATION"
    framework = "TENSORFLOW"
    framework_version = "2.9"
    nearest_model_name = "resnet50"
    data_input_config = '{"input_1":[1,224,224,3]}'

    estimator.register(
        content_types=content_types,
        response_types=response_types,
        inference_instances=inference_instances,
        transform_instances=transform_instances,
        model_package_name=model_package_name,
        sample_payload_url=sample_payload_url,
        task=task,
        framework=framework,
        framework_version=framework_version,
        nearest_model_name=nearest_model_name,
        data_input_configuration=data_input_config,
    )
    sagemaker_session.create_model.assert_not_called()

    expected_create_model_package_request = {
        "containers": [{"Image": estimator.image_uri, "ModelDataUrl": estimator.model_data}],
        "content_types": content_types,
        "response_types": response_types,
        "inference_instances": inference_instances,
        "transform_instances": transform_instances,
        "model_package_name": model_package_name,
        "marketplace_cert": False,
        "sample_payload_url": sample_payload_url,
        "task": task,
    }
    sagemaker_session.create_model_package_from_containers.assert_called_with(
        **expected_create_model_package_request
    )


def test_register_default_image_without_instance_type_args(sagemaker_session):
    estimator = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )
    estimator.set_hyperparameters(**HYPERPARAMS)
    estimator.fit({"train": "s3://bucket/training-prefix"})

    model_package_name = "test-estimator-register-model"
    content_types = ["application/json"]
    response_types = ["application/json"]
    sample_payload_url = "s3://test-bucket/model"
    task = "IMAGE_CLASSIFICATION"
    framework = "TENSORFLOW"
    framework_version = "2.9"
    nearest_model_name = "resnet50"

    estimator.register(
        content_types=content_types,
        response_types=response_types,
        model_package_name=model_package_name,
        sample_payload_url=sample_payload_url,
        task=task,
        framework=framework,
        framework_version=framework_version,
        nearest_model_name=nearest_model_name,
    )
    sagemaker_session.create_model.assert_not_called()

    expected_create_model_package_request = {
        "containers": [{"Image": estimator.image_uri, "ModelDataUrl": estimator.model_data}],
        "content_types": content_types,
        "response_types": response_types,
        "inference_instances": None,
        "transform_instances": None,
        "model_package_name": model_package_name,
        "marketplace_cert": False,
        "sample_payload_url": sample_payload_url,
        "task": task,
    }
    sagemaker_session.create_model_package_from_containers.assert_called_with(
        **expected_create_model_package_request
    )


def test_register_inference_image(sagemaker_session):
    estimator = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session,
    )
    estimator.set_hyperparameters(**HYPERPARAMS)
    estimator.fit({"train": "s3://bucket/training-prefix"})

    model_package_name = "test-estimator-register-model"
    content_types = ["application/json"]
    response_types = ["application/json"]
    inference_instances = ["ml.m4.xlarge"]
    transform_instances = ["ml.m4.xlarget"]
    inference_image = "fake-inference-image"
    sample_payload_url = "s3://test-bucket/model"
    task = "IMAGE_CLASSIFICATION"
    framework = "TENSORFLOW"
    framework_version = "2.9"
    nearest_model_name = "resnet50"

    estimator.register(
        content_types=content_types,
        response_types=response_types,
        inference_instances=inference_instances,
        transform_instances=transform_instances,
        model_package_name=model_package_name,
        sample_payload_url=sample_payload_url,
        task=task,
        image_uri=inference_image,
        framework=framework,
        framework_version=framework_version,
        nearest_model_name=nearest_model_name,
    )
    sagemaker_session.create_model.assert_not_called()

    expected_create_model_package_request = {
        "containers": [{"Image": inference_image, "ModelDataUrl": estimator.model_data}],
        "content_types": content_types,
        "response_types": response_types,
        "inference_instances": inference_instances,
        "transform_instances": transform_instances,
        "model_package_name": model_package_name,
        "marketplace_cert": False,
        "sample_payload_url": sample_payload_url,
        "task": task,
    }
    sagemaker_session.create_model_package_from_containers.assert_called_with(
        **expected_create_model_package_request
    )


def test_register_under_pipeline_session(pipeline_session):
    estimator = Estimator(
        IMAGE_URI,
        ROLE,
        INSTANCE_COUNT,
        INSTANCE_TYPE,
        output_path=OUTPUT_PATH,
        sagemaker_session=pipeline_session,
    )

    model_package_name = "test-estimator-register-model"
    content_types = ["application/json"]
    response_types = ["application/json"]
    inference_instances = ["ml.m4.xlarge"]
    transform_instances = ["ml.m4.xlarget"]

    with pytest.raises(TypeError) as error:
        estimator.register(
            content_types=content_types,
            response_types=response_types,
            inference_instances=inference_instances,
            transform_instances=transform_instances,
            model_package_name=model_package_name,
        )
    assert "estimator.register does not support PipelineSession" in str(error.value)


@patch("sagemaker.estimator.LocalSession")
@patch("sagemaker.estimator.Session")
def test_local_mode(session_class, local_session_class):
    local_session = Mock(spec=sagemaker.local.LocalSession)
    local_session.local_mode = True

    local_session.settings = SessionSettings()

    local_session.sagemaker_config = {}

    session = Mock()
    session.local_mode = False

    session.sagemaker_config = {}

    local_session_class.return_value = local_session
    session_class.return_value = session

    e = Estimator(IMAGE_URI, ROLE, INSTANCE_COUNT, "local")
    assert e.sagemaker_session.local_mode is True

    e2 = Estimator(IMAGE_URI, ROLE, INSTANCE_COUNT, "local_gpu")
    assert e2.sagemaker_session.local_mode is True

    e3 = Estimator(IMAGE_URI, ROLE, INSTANCE_COUNT, INSTANCE_TYPE)
    assert e3.sagemaker_session.local_mode is False


@patch("sagemaker.estimator.LocalSession")
def test_distributed_gpu_local_mode(LocalSession):
    with pytest.raises(RuntimeError):
        Estimator(IMAGE_URI, ROLE, 3, "local_gpu", output_path=OUTPUT_PATH)


@patch("sagemaker.estimator.LocalSession")
def test_local_mode_file_output_path(local_session_class):
    local_session = Mock(spec=sagemaker.local.LocalSession)
    local_session.local_mode = True
    local_session_class.return_value = local_session

    local_session.settings = SessionSettings()

    local_session.sagemaker_config = {}

    e = Estimator(IMAGE_URI, ROLE, INSTANCE_COUNT, "local", output_path="file:///tmp/model/")
    assert e.output_path == "file:///tmp/model/"


@patch("sagemaker.estimator.Session")
def test_file_output_path_not_supported_outside_local_mode(session_class):
    session = Mock()
    session.local_mode = False
    session_class.return_value = session

    with pytest.raises(RuntimeError):
        Estimator(IMAGE_URI, ROLE, INSTANCE_COUNT, INSTANCE_TYPE, output_path="file:///tmp/model")


def test_prepare_init_params_from_job_description_with_image_training_job():
    init_params = EstimatorBase._prepare_init_params_from_job_description(
        job_details=RETURNED_JOB_DESCRIPTION
    )

    assert init_params["role"] == "arn:aws:iam::366:role/SageMakerRole"
    assert init_params["instance_count"] == 1
    assert init_params["image_uri"] == "1.dkr.ecr.us-west-2.amazonaws.com/sagemaker-other:1.0.4"


def test_prepare_init_params_from_job_description_with_algorithm_training_job():
    algorithm_job_description = RETURNED_JOB_DESCRIPTION.copy()
    algorithm_job_description["AlgorithmSpecification"] = {
        "TrainingInputMode": "File",
        "AlgorithmName": "arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
        "TrainingImage": "",
    }

    init_params = EstimatorBase._prepare_init_params_from_job_description(
        job_details=algorithm_job_description
    )

    assert init_params["role"] == "arn:aws:iam::366:role/SageMakerRole"
    assert init_params["instance_count"] == 1
    assert (
        init_params["algorithm_arn"]
        == "arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees"
    )


def test_prepare_init_params_from_job_description_with_spot_training():
    job_description = RETURNED_JOB_DESCRIPTION.copy()
    job_description["EnableManagedSpotTraining"] = True
    job_description["StoppingCondition"] = {
        "MaxRuntimeInSeconds": 86400,
        "MaxWaitTimeInSeconds": 87000,
    }

    init_params = EstimatorBase._prepare_init_params_from_job_description(
        job_details=job_description
    )

    assert init_params["role"] == "arn:aws:iam::366:role/SageMakerRole"
    assert init_params["instance_count"] == 1
    assert init_params["use_spot_instances"]
    assert init_params["max_run"] == 86400
    assert init_params["max_wait"] == 87000


def test_prepare_init_params_from_job_description_with_retry_strategy():
    job_description = RETURNED_JOB_DESCRIPTION.copy()
    job_description["RetryStrategy"] = {"MaximumRetryAttempts": 2}
    job_description["StoppingCondition"] = {
        "MaxRuntimeInSeconds": 86400,
        "MaxWaitTimeInSeconds": 87000,
    }

    init_params = EstimatorBase._prepare_init_params_from_job_description(
        job_details=job_description
    )

    assert init_params["role"] == "arn:aws:iam::366:role/SageMakerRole"
    assert init_params["instance_count"] == 1
    assert init_params["max_run"] == 86400
    assert init_params["max_wait"] == 87000
    assert init_params["max_retry_attempts"] == 2


def test_prepare_init_params_from_job_description_with_training_image_config():
    job_description = RETURNED_JOB_DESCRIPTION.copy()
    job_description["AlgorithmSpecification"]["TrainingImageConfig"] = {
        "TrainingRepositoryAccessMode": "Vpc",
        "TrainingRepositoryAuthConfig": {
            "TrainingRepositoryCredentialsProviderArn": "arn:aws:lambda:us-west-2:1234567890:function:test"
        },
    }

    init_params = EstimatorBase._prepare_init_params_from_job_description(
        job_details=job_description
    )

    assert init_params["role"] == "arn:aws:iam::366:role/SageMakerRole"
    assert init_params["instance_count"] == 1
    assert init_params["training_repository_access_mode"] == "Vpc"
    assert (
        init_params["training_repository_credentials_provider_arn"]
        == "arn:aws:lambda:us-west-2:1234567890:function:test"
    )


def test_prepare_init_params_from_job_description_with_container_entry_point_and_args():
    job_description = RETURNED_JOB_DESCRIPTION.copy()
    job_description["AlgorithmSpecification"]["ContainerEntrypoint"] = CONTAINER_ENTRY_POINT
    job_description["AlgorithmSpecification"]["ContainerArguments"] = CONTAINER_ARGUMENTS

    init_params = EstimatorBase._prepare_init_params_from_job_description(
        job_details=job_description
    )

    assert init_params["role"] == "arn:aws:iam::366:role/SageMakerRole"
    assert init_params["instance_count"] == 1
    assert init_params["container_entry_point"] == CONTAINER_ENTRY_POINT
    assert init_params["container_arguments"] == CONTAINER_ARGUMENTS


def test_prepare_init_params_from_job_description_with_invalid_training_job():
    invalid_job_description = RETURNED_JOB_DESCRIPTION.copy()
    invalid_job_description["AlgorithmSpecification"] = {"TrainingInputMode": "File"}

    with pytest.raises(RuntimeError) as error:
        EstimatorBase._prepare_init_params_from_job_description(job_details=invalid_job_description)
        assert "Invalid AlgorithmSpecification" in str(error)


def test_prepare_for_training_with_base_name(sagemaker_session):
    estimator = Estimator(
        image_uri="some-image",
        role="some_image",
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        base_job_name="base_job_name",
    )

    estimator._prepare_for_training()
    assert "base_job_name" in estimator._current_job_name


def test_prepare_for_training_with_name_based_on_image(sagemaker_session):
    estimator = Estimator(
        image_uri="some-image",
        role="some_image",
        instance_count=1,
        instance_type="ml.m4.xlarge",
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
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    estimator._prepare_for_training()
    assert "scikit-decision-trees-1542410022" in estimator._current_job_name


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
def test_prepare_for_training_with_pipeline_name_in_s3_path_no_source_dir(
    pipeline_session,
):
    # script_uri is NOT provided -> use new cache key behavior that builds path using pipeline name + code_hash
    image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38"
    model_uri = "s3://someprefix2/models/model.tar.gz"
    estimator = Estimator(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=pipeline_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        image_uri=image_uri,
        model_uri=model_uri,
    )
    step_args = estimator.fit()
    # execute estimator.fit() and generate args, S3 paths
    step_args.func(*step_args.func_args, **step_args.func_kwargs)
    expected_path = "/".join(["test-pipeline", "code", "code-hash-0123456789"])
    assert expected_path in estimator.uploaded_code.s3_prefix


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
    with pytest.raises(ClientError) as error:
        estimator = Estimator(
            image_uri="some-image",
            role="SageMakerRole",
            instance_count=1,
            instance_type="ml.c4.xlarge",
            sagemaker_session=sagemaker_session,
            base_job_name="test-non-vpc-encryption",
            encrypt_inter_container_traffic=True,
        )
        estimator.fit()
    assert (
        '"EnableInterContainerTrafficEncryption" and "VpcConfig" must be provided together'
        in str(error)
    )


def test_estimator_local_mode_error(sagemaker_session):
    # When using instance local with a session which is not LocalSession we should error out
    with pytest.raises(RuntimeError):
        Estimator(
            image_uri="some-image",
            role="some_image",
            instance_count=1,
            instance_type="local",
            sagemaker_session=sagemaker_session,
            base_job_name="base_job_name",
        )


def test_estimator_local_mode_ok(sagemaker_local_session):

    sagemaker_local_session.sagemaker_config = {}
    # When using instance local with a session which is not LocalSession we should error out
    Estimator(
        image_uri="some-image",
        role="some_image",
        instance_count=1,
        instance_type="local",
        sagemaker_session=sagemaker_local_session,
        base_job_name="base_job_name",
    )


def test_framework_distribution_configuration(sagemaker_session):
    framework = DummyFramework(
        entry_point="script",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    actual_ps = framework._distribution_configuration(distribution=DISTRIBUTION_PS_ENABLED)
    expected_ps = {"sagemaker_parameter_server_enabled": True}
    assert actual_ps == expected_ps

    actual_mpi = framework._distribution_configuration(distribution=DISTRIBUTION_MPI_ENABLED)
    expected_mpi = {
        "sagemaker_mpi_enabled": True,
        "sagemaker_mpi_num_of_processes_per_host": 2,
        "sagemaker_mpi_custom_mpi_options": "options",
    }
    assert actual_mpi == expected_mpi

    actual_ddp = framework._distribution_configuration(distribution=DISTRIBUTION_SM_DDP_ENABLED)
    expected_ddp = {
        "sagemaker_distributed_dataparallel_enabled": True,
        "sagemaker_distributed_dataparallel_custom_mpi_options": "options",
        "sagemaker_instance_type": INSTANCE_TYPE,
    }
    assert actual_ddp == expected_ddp


def test_mwms_distribution_configuration(sagemaker_session):
    framework = DummyFramework(
        entry_point="script",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )
    with pytest.raises(ValueError) as error:
        framework._distribution_configuration(distribution=DISTRIBUTION_MWMS_ENABLED)

    assert "only supported with" in str(error)
    assert "but received" in str(error)


def test_image_name_map(sagemaker_session):
    e = DummyFramework(
        "my_script.py",
        image_name=IMAGE_URI,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
    )

    assert e.image_uri == IMAGE_URI


@patch("sagemaker.git_utils.git_clone_repo")
def test_git_support_with_branch_and_commit_succeed_estimator_class(
    git_clone_repo, sagemaker_session
):
    git_clone_repo.side_effect = lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    }
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    entry_point = "entry_point"
    fw = Estimator(
        entry_point=entry_point,
        git_config=git_config,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        image_uri=IMAGE_URI,
    )
    fw.fit()
    git_clone_repo.assert_called_once_with(git_config, entry_point, None, [])


@patch("sagemaker.estimator.Estimator._stage_user_code_in_s3")
def test_script_mode_estimator(patched_stage_user_code, sagemaker_session):
    patched_stage_user_code.return_value = UploadedCode(
        s3_prefix="s3://bucket/key", script_name="script_name"
    )
    script_uri = "s3://codebucket/someprefix/sourcedir.tar.gz"
    image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38"
    model_uri = "s3://someprefix2/models/model.tar.gz"
    t = Estimator(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        source_dir=script_uri,
        image_uri=image_uri,
        model_uri=model_uri,
    )
    t.fit("s3://bucket/mydata")

    patched_stage_user_code.assert_called_once()
    sagemaker_session.train.assert_called_once()


@patch("time.time", return_value=TIME)
@patch("sagemaker.estimator.tar_and_upload_dir")
def test_script_mode_estimator_same_calls_as_framework(
    patched_tar_and_upload_dir, sagemaker_session
):
    patched_tar_and_upload_dir.return_value = UploadedCode(
        s3_prefix="s3://%s/%s" % ("bucket", "key"), script_name="script_name"
    )
    sagemaker_session.boto_region_name = REGION
    sagemaker_session.sagemaker_config = {}

    script_uri = "s3://codebucket/someprefix/sourcedir.tar.gz"

    instance_type = "ml.p2.xlarge"
    instance_count = 1

    model_uri = "s3://someprefix2/models/model.tar.gz"
    training_data_uri = "s3://bucket/mydata"
    hyperparameters = {
        "int_hyperparam": 1,
        "string_hyperparam": "hello",
        "stringified_numeric_hyperparam": "44",
        "float_hyperparam": 1.234,
    }

    generic_estimator = Estimator(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        region=REGION,
        sagemaker_session=sagemaker_session,
        instance_count=instance_count,
        instance_type=instance_type,
        source_dir=script_uri,
        image_uri=IMAGE_URI,
        model_uri=model_uri,
        dependencies=[],
        debugger_hook_config={},
        hyperparameters=deepcopy(hyperparameters),
    )
    generic_estimator.fit(training_data_uri)

    generic_estimator_tar_and_upload_dir_args = patched_tar_and_upload_dir.call_args_list
    generic_estimator_train_args = sagemaker_session.train.call_args_list

    patched_tar_and_upload_dir.reset_mock()
    sagemaker_session.train.reset_mock()

    framework_estimator = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        region=REGION,
        source_dir=script_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        model_uri=model_uri,
        dependencies=[],
        debugger_hook_config={},
        hyperparameters=deepcopy(hyperparameters),
    )
    framework_estimator.fit(training_data_uri)

    assert len(generic_estimator_tar_and_upload_dir_args) == 1
    assert len(generic_estimator_train_args) == 1
    assert generic_estimator_tar_and_upload_dir_args == patched_tar_and_upload_dir.call_args_list
    assert generic_estimator_train_args == sagemaker_session.train.call_args_list


@patch("time.time", return_value=TIME)
@patch("sagemaker.estimator.tar_and_upload_dir")
@patch("sagemaker.model.Model._upload_code")
def test_script_mode_estimator_tags_jumpstart_estimators_and_models(
    patched_upload_code, patched_tar_and_upload_dir, sagemaker_session
):
    patched_tar_and_upload_dir.return_value = UploadedCode(
        s3_prefix="s3://%s/%s" % ("bucket", "key"), script_name="script_name"
    )
    sagemaker_session.boto_region_name = REGION
    sagemaker_session.sagemaker_config = {}

    instance_type = "ml.p2.xlarge"
    instance_count = 1

    training_data_uri = "s3://bucket/mydata"

    jumpstart_source_dir = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[0]}/source_dirs/source.tar.gz"
    )
    jumpstart_source_dir_2 = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[1]}/source_dirs/source.tar.gz"
    )

    generic_estimator = Estimator(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        region=REGION,
        sagemaker_session=sagemaker_session,
        instance_count=instance_count,
        instance_type=instance_type,
        source_dir=jumpstart_source_dir,
        image_uri=IMAGE_URI,
        model_uri=jumpstart_source_dir_2,
        tags=[{"Key": "some", "Value": "tag"}],
    )
    generic_estimator.fit(training_data_uri)

    assert [
        {"Key": "some", "Value": "tag"},
        {"Key": JumpStartTag.TRAINING_MODEL_URI.value, "Value": jumpstart_source_dir_2},
        {"Key": JumpStartTag.TRAINING_SCRIPT_URI.value, "Value": jumpstart_source_dir},
    ] == sagemaker_session.train.call_args_list[0][1]["tags"]

    sagemaker_session.reset_mock()
    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "ModelArtifacts": {"S3ModelArtifacts": "some-uri"}
    }

    inference_jumpstart_source_dir = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[0]}"
        "/source_dirs/inference/source.tar.gz"
    )

    generic_estimator.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        image_uri=IMAGE_URI,
        source_dir=inference_jumpstart_source_dir,
        entry_point="inference.py",
        role=ROLE,
        tags=[{"Key": "deploys", "Value": "tag"}],
    )

    assert sagemaker_session.create_model.call_args_list[0][1]["tags"] == [
        {"Key": "deploys", "Value": "tag"},
        {"Key": JumpStartTag.TRAINING_MODEL_URI.value, "Value": jumpstart_source_dir_2},
        {"Key": JumpStartTag.TRAINING_SCRIPT_URI.value, "Value": jumpstart_source_dir},
        {"Key": JumpStartTag.INFERENCE_SCRIPT_URI.value, "Value": inference_jumpstart_source_dir},
    ]
    assert sagemaker_session.endpoint_from_production_variants.call_args_list[0][1]["tags"] == [
        {"Key": "deploys", "Value": "tag"},
        {"Key": JumpStartTag.TRAINING_MODEL_URI.value, "Value": jumpstart_source_dir_2},
        {"Key": JumpStartTag.TRAINING_SCRIPT_URI.value, "Value": jumpstart_source_dir},
        {"Key": JumpStartTag.INFERENCE_SCRIPT_URI.value, "Value": inference_jumpstart_source_dir},
    ]


@patch("time.time", return_value=TIME)
@patch("sagemaker.estimator.tar_and_upload_dir")
@patch("sagemaker.model.Model._upload_code")
def test_script_mode_estimator_tags_jumpstart_models(
    patched_upload_code, patched_tar_and_upload_dir, sagemaker_session
):
    patched_tar_and_upload_dir.return_value = UploadedCode(
        s3_prefix="s3://%s/%s" % ("bucket", "key"), script_name="script_name"
    )
    sagemaker_session.boto_region_name = REGION
    sagemaker_session.sagemaker_config = {}

    instance_type = "ml.p2.xlarge"
    instance_count = 1

    training_data_uri = "s3://bucket/mydata"

    jumpstart_source_dir = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[0]}/source_dirs/source.tar.gz"
    )

    generic_estimator = Estimator(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        region=REGION,
        sagemaker_session=sagemaker_session,
        instance_count=instance_count,
        instance_type=instance_type,
        source_dir=jumpstart_source_dir,
        image_uri=IMAGE_URI,
        model_uri=MODEL_DATA,
    )
    generic_estimator.fit(training_data_uri)

    assert [
        {"Key": JumpStartTag.TRAINING_SCRIPT_URI.value, "Value": jumpstart_source_dir}
    ] == sagemaker_session.train.call_args_list[0][1]["tags"]

    sagemaker_session.reset_mock()
    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "ModelArtifacts": {"S3ModelArtifacts": "some-uri"}
    }

    inference_source_dir = "s3://dsfsdfsd/sdfsdfs/sdfsd"

    generic_estimator.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        image_uri=IMAGE_URI,
        source_dir=inference_source_dir,
        entry_point="inference.py",
        role=ROLE,
    )

    assert sagemaker_session.create_model.call_args_list[0][1]["tags"] == [
        {"Key": JumpStartTag.TRAINING_SCRIPT_URI.value, "Value": jumpstart_source_dir}
    ]
    assert sagemaker_session.endpoint_from_production_variants.call_args_list[0][1]["tags"] == [
        {"Key": JumpStartTag.TRAINING_SCRIPT_URI.value, "Value": jumpstart_source_dir}
    ]


@patch("time.time", return_value=TIME)
@patch("sagemaker.estimator.tar_and_upload_dir")
@patch("sagemaker.model.Model._upload_code")
def test_script_mode_estimator_tags_jumpstart_models_with_no_estimator_js_tags(
    patched_upload_code, patched_tar_and_upload_dir, sagemaker_session
):
    patched_tar_and_upload_dir.return_value = UploadedCode(
        s3_prefix="s3://%s/%s" % ("bucket", "key"), script_name="script_name"
    )
    sagemaker_session.boto_region_name = REGION
    sagemaker_session.sagemaker_config = {}

    instance_type = "ml.p2.xlarge"
    instance_count = 1

    training_data_uri = "s3://bucket/mydata"

    source_dir = "s3://dsfsdfsd/sdfsdfs/sdfsd"

    generic_estimator = Estimator(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        region=REGION,
        sagemaker_session=sagemaker_session,
        instance_count=instance_count,
        instance_type=instance_type,
        source_dir=source_dir,
        image_uri=IMAGE_URI,
        model_uri=MODEL_DATA,
    )
    generic_estimator.fit(training_data_uri)

    assert None is sagemaker_session.train.call_args_list[0][1]["tags"]

    sagemaker_session.reset_mock()
    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "ModelArtifacts": {"S3ModelArtifacts": "some-uri"}
    }

    inference_jumpstart_source_dir = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[0]}"
        "/source_dirs/inference/source.tar.gz"
    )

    generic_estimator.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        image_uri=IMAGE_URI,
        source_dir=inference_jumpstart_source_dir,
        entry_point="inference.py",
        role=ROLE,
    )

    assert sagemaker_session.create_model.call_args_list[0][1]["tags"] == [
        {"Key": JumpStartTag.INFERENCE_SCRIPT_URI.value, "Value": inference_jumpstart_source_dir}
    ]
    assert sagemaker_session.endpoint_from_production_variants.call_args_list[0][1]["tags"] == [
        {"Key": JumpStartTag.INFERENCE_SCRIPT_URI.value, "Value": inference_jumpstart_source_dir}
    ]


@patch("sagemaker.estimator.tar_and_upload_dir")
@patch("sagemaker.model.Model._upload_code")
@patch("sagemaker.utils.repack_model")
def test_all_framework_estimators_add_jumpstart_uri_tags(
    patched_repack_model, patched_upload_code, patched_tar_and_upload_dir, sagemaker_session
):
    sagemaker_session.boto_region_name = REGION
    sagemaker_session.sagemaker_config = {}
    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "ModelArtifacts": {"S3ModelArtifacts": "some-uri"}
    }

    patched_tar_and_upload_dir.return_value = UploadedCode(
        s3_prefix="s3://%s/%s" % ("bucket", "key"), script_name="script_name"
    )

    framework_estimator_classes_to_kwargs = {
        PyTorch: {
            "framework_version": "1.5.0",
            "py_version": "py3",
            "instance_type": "ml.p2.xlarge",
        },
        TensorFlow: {
            "framework_version": "2.3",
            "py_version": "py37",
            "instance_type": "ml.p2.xlarge",
        },
        HuggingFace: {
            "pytorch_version": "1.7.1",
            "py_version": "py36",
            "transformers_version": "4.6.1",
            "instance_type": "ml.p2.xlarge",
        },
        MXNet: {"framework_version": "1.7.0", "py_version": "py3", "instance_type": "ml.p2.xlarge"},
        SKLearn: {"framework_version": "0.23-1", "instance_type": "ml.m2.xlarge"},
        XGBoost: {"framework_version": "1.3-1", "instance_type": "ml.m2.xlarge"},
    }
    jumpstart_model_uri = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[0]}/model_dirs/model.tar.gz"
    )
    jumpstart_model_uri_2 = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[1]}/model_dirs/model.tar.gz"
    )
    for framework_estimator_class, kwargs in framework_estimator_classes_to_kwargs.items():
        estimator = framework_estimator_class(
            entry_point=ENTRY_POINT,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            model_uri=jumpstart_model_uri,
            instance_count=INSTANCE_COUNT,
            tags=[{"Key": "blah", "Value": "yoyoma"}],
            **kwargs,
        )

        estimator.fit()

        assert {
            "Key": JumpStartTag.TRAINING_MODEL_URI.value,
            "Value": jumpstart_model_uri,
        } in sagemaker_session.train.call_args_list[0][1]["tags"]

        assert {"Key": "blah", "Value": "yoyoma"} in sagemaker_session.train.call_args_list[0][1][
            "tags"
        ]

        estimator.deploy(
            initial_instance_count=INSTANCE_COUNT,
            instance_type=kwargs["instance_type"],
            image_uri=IMAGE_URI,
            source_dir=jumpstart_model_uri_2,
            entry_point="inference.py",
            role=ROLE,
            tags=[{"Key": "blah", "Value": "yoyoma"}],
        )

        assert sagemaker_session.create_model.call_args_list[0][1]["tags"] == [
            {"Key": "blah", "Value": "yoyoma"}
        ] + [
            {"Key": JumpStartTag.TRAINING_MODEL_URI.value, "Value": jumpstart_model_uri},
            {"Key": JumpStartTag.INFERENCE_SCRIPT_URI.value, "Value": jumpstart_model_uri_2},
        ]
        assert sagemaker_session.endpoint_from_production_variants.call_args_list[0][1]["tags"] == [
            {"Key": "blah", "Value": "yoyoma"}
        ] + [
            {"Key": JumpStartTag.TRAINING_MODEL_URI.value, "Value": jumpstart_model_uri},
            {"Key": JumpStartTag.INFERENCE_SCRIPT_URI.value, "Value": jumpstart_model_uri_2},
        ]

        sagemaker_session.train.reset_mock()


@patch("sagemaker.estimator.tar_and_upload_dir")
@patch("sagemaker.model.Model._upload_code")
@patch("sagemaker.utils.repack_model")
def test_all_framework_estimators_support_disabling_jumpstart_uri_tags(
    patched_repack_model, patched_upload_code, patched_tar_and_upload_dir, sagemaker_session
):
    sagemaker_session.boto_region_name = REGION
    sagemaker_session.sagemaker_config = {}
    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "ModelArtifacts": {"S3ModelArtifacts": "some-uri"}
    }

    patched_tar_and_upload_dir.return_value = UploadedCode(
        s3_prefix="s3://%s/%s" % ("bucket", "key"), script_name="script_name"
    )

    sagemaker_session.settings = SessionSettings(include_jumpstart_tags=False)

    framework_estimator_classes_to_kwargs = {
        PyTorch: {
            "framework_version": "1.5.0",
            "py_version": "py3",
            "instance_type": "ml.p2.xlarge",
        },
        TensorFlow: {
            "framework_version": "2.3",
            "py_version": "py37",
            "instance_type": "ml.p2.xlarge",
        },
        HuggingFace: {
            "pytorch_version": "1.7.1",
            "py_version": "py36",
            "transformers_version": "4.6.1",
            "instance_type": "ml.p2.xlarge",
        },
        MXNet: {"framework_version": "1.7.0", "py_version": "py3", "instance_type": "ml.p2.xlarge"},
        SKLearn: {"framework_version": "0.23-1", "instance_type": "ml.m2.xlarge"},
        XGBoost: {"framework_version": "1.3-1", "instance_type": "ml.m2.xlarge"},
    }
    jumpstart_model_uri = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[0]}/model_dirs/model.tar.gz"
    )
    jumpstart_model_uri_2 = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[1]}/model_dirs/model.tar.gz"
    )
    for framework_estimator_class, kwargs in framework_estimator_classes_to_kwargs.items():
        estimator = framework_estimator_class(
            entry_point=ENTRY_POINT,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            model_uri=jumpstart_model_uri,
            instance_count=INSTANCE_COUNT,
            tags=[{"Key": "blah", "Value": "yoyoma"}],
            **kwargs,
        )

        estimator.fit()

        assert [{"Key": "blah", "Value": "yoyoma"}] == sagemaker_session.train.call_args_list[0][1][
            "tags"
        ]

        estimator.deploy(
            initial_instance_count=INSTANCE_COUNT,
            instance_type=kwargs["instance_type"],
            image_uri=IMAGE_URI,
            source_dir=jumpstart_model_uri_2,
            entry_point="inference.py",
            role=ROLE,
        )

        assert sagemaker_session.create_model.call_args_list[0][1]["tags"] == [
            {"Key": "blah", "Value": "yoyoma"}
        ]
        assert sagemaker_session.endpoint_from_production_variants.call_args_list[0][1]["tags"] == [
            {"Key": "blah", "Value": "yoyoma"}
        ]
        sagemaker_session.train.reset_mock()


@patch("time.time", return_value=TIME)
@patch("sagemaker.estimator.tar_and_upload_dir")
@patch("sagemaker.model.Model._upload_code")
def test_script_mode_estimator_uses_jumpstart_base_name_with_js_models(
    patched_upload_code, patched_tar_and_upload_dir, sagemaker_session
):
    patched_tar_and_upload_dir.return_value = UploadedCode(
        s3_prefix="s3://%s/%s" % ("bucket", "key"), script_name="script_name"
    )
    sagemaker_session.boto_region_name = REGION
    sagemaker_session.sagemaker_config = {}

    instance_type = "ml.p2.xlarge"
    instance_count = 1

    training_data_uri = "s3://bucket/mydata"

    source_dir = "s3://dsfsdfsd/sdfsdfs/sdfsd"

    generic_estimator = Estimator(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        region=REGION,
        sagemaker_session=sagemaker_session,
        instance_count=instance_count,
        instance_type=instance_type,
        source_dir=source_dir,
        image_uri=IMAGE_URI,
        model_uri=MODEL_DATA,
    )
    generic_estimator.fit(training_data_uri)

    assert not sagemaker_session.train.call_args_list[0][1]["job_name"].startswith(
        JUMPSTART_RESOURCE_BASE_NAME
    )
    sagemaker_session.reset_mock()
    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "ModelArtifacts": {"S3ModelArtifacts": "some-uri"}
    }

    inference_jumpstart_source_dir = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[0]}"
        "/source_dirs/inference/source.tar.gz"
    )

    generic_estimator.deploy(
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        image_uri=IMAGE_URI,
        source_dir=inference_jumpstart_source_dir,
        entry_point="inference.py",
        role=ROLE,
    )

    assert sagemaker_session.create_model.call_args_list[0][1]["name"].startswith(
        JUMPSTART_RESOURCE_BASE_NAME
    )

    assert sagemaker_session.endpoint_from_production_variants.call_args_list[0].startswith(
        JUMPSTART_RESOURCE_BASE_NAME
    )


@patch("sagemaker.estimator.tar_and_upload_dir")
@patch("sagemaker.model.Model._upload_code")
@patch("sagemaker.utils.repack_model")
def test_all_framework_estimators_add_jumpstart_base_name(
    patched_repack_model, patched_upload_code, patched_tar_and_upload_dir, sagemaker_session
):
    sagemaker_session.boto_region_name = REGION
    sagemaker_session.sagemaker_config = {}
    sagemaker_session.sagemaker_client.describe_training_job.return_value = {
        "ModelArtifacts": {"S3ModelArtifacts": "some-uri"}
    }

    patched_tar_and_upload_dir.return_value = UploadedCode(
        s3_prefix="s3://%s/%s" % ("bucket", "key"), script_name="script_name"
    )

    framework_estimator_classes_to_kwargs = {
        PyTorch: {
            "framework_version": "1.5.0",
            "py_version": "py3",
            "instance_type": "ml.p2.xlarge",
        },
        TensorFlow: {
            "framework_version": "2.3",
            "py_version": "py37",
            "instance_type": "ml.p2.xlarge",
        },
        HuggingFace: {
            "pytorch_version": "1.7.1",
            "py_version": "py36",
            "transformers_version": "4.6.1",
            "instance_type": "ml.p2.xlarge",
        },
        MXNet: {"framework_version": "1.7.0", "py_version": "py3", "instance_type": "ml.p2.xlarge"},
        SKLearn: {"framework_version": "0.23-1", "instance_type": "ml.m2.xlarge"},
        XGBoost: {"framework_version": "1.3-1", "instance_type": "ml.m2.xlarge"},
    }
    jumpstart_model_uri = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[0]}/model_dirs/model.tar.gz"
    )
    jumpstart_model_uri_2 = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[1]}/model_dirs/model.tar.gz"
    )
    for framework_estimator_class, kwargs in framework_estimator_classes_to_kwargs.items():
        estimator = framework_estimator_class(
            entry_point=ENTRY_POINT,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            model_uri=jumpstart_model_uri,
            instance_count=INSTANCE_COUNT,
            **kwargs,
        )

        estimator.fit()

        assert sagemaker_session.train.call_args_list[0][1]["job_name"].startswith(
            JUMPSTART_RESOURCE_BASE_NAME
        )

        estimator.deploy(
            initial_instance_count=INSTANCE_COUNT,
            instance_type=kwargs["instance_type"],
            image_uri=IMAGE_URI,
            source_dir=jumpstart_model_uri_2,
            entry_point="inference.py",
            role=ROLE,
        )

        assert sagemaker_session.create_model.call_args_list[0][1]["name"].startswith(
            JUMPSTART_RESOURCE_BASE_NAME
        )

        assert sagemaker_session.endpoint_from_production_variants.call_args_list[0].startswith(
            JUMPSTART_RESOURCE_BASE_NAME
        )

        sagemaker_session.endpoint_from_production_variants.reset_mock()
        sagemaker_session.create_model.reset_mock()
        sagemaker_session.train.reset_mock()


def test_insert_invalid_source_code_args():
    with pytest.raises(TypeError) as err:
        Estimator(
            image_uri="IMAGE_URI",
            role=ROLE,
            entry_point=ParameterString(name="EntryPoint"),
            instance_type="ml.m5.xlarge",
            instance_count=1,
            enable_network_isolation=True,
        )
    assert (
        "entry_point, source_dir should not be pipeline variables "
        "when enable_network_isolation is a pipeline variable or it is set to True."
    ) in str(err.value)

    with pytest.raises(TypeError) as err:
        Estimator(
            image_uri="IMAGE_URI",
            role=ROLE,
            entry_point="dummy.py",
            source_dir=ParameterString(name="SourceDir"),
            instance_type="ml.m5.xlarge",
            instance_count=1,
            enable_network_isolation=ParameterBoolean(name="EnableNetworkIsolation"),
        )
    assert (
        "entry_point, source_dir should not be pipeline variables "
        "when enable_network_isolation is a pipeline variable or it is set to True."
    ) in str(err.value)

    with pytest.raises(TypeError) as err:
        Estimator(
            image_uri=IMAGE_URI,
            role=ROLE,
            git_config={"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT},
            source_dir=ParameterString(name="SourceDir"),
            entry_point=ParameterString(name="EntryPoint"),
            instance_type="ml.m5.xlarge",
            instance_count=1,
        )
    assert (
        "entry_point, source_dir should not be pipeline variables when git_config is given"
        in str(err.value)
    )

    with pytest.raises(TypeError) as err:
        Estimator(
            image_uri=IMAGE_URI,
            role=ROLE,
            entry_point=ParameterString(name="EntryPoint"),
            instance_type="ml.m5.xlarge",
            instance_count=1,
        )
    assert "The entry_point should not be a pipeline variable when source_dir is missing" in str(
        err.value
    )

    with pytest.raises(TypeError) as err:
        Estimator(
            image_uri="IMAGE_URI",
            role=ROLE,
            entry_point=ParameterString(name="EntryPoint"),
            source_dir="file://my-file/",
            instance_type="ml.m5.xlarge",
            instance_count=1,
        )
    assert (
        "The entry_point should not be a pipeline variable " "when source_dir is a local path"
    ) in str(err.value)


@patch("time.time", return_value=TIME)
@patch("sagemaker.estimator.tar_and_upload_dir")
@patch("sagemaker.model.Model._upload_code")
def test_script_mode_estimator_escapes_hyperparameters_as_json(
    patched_upload_code, patched_tar_and_upload_dir, sagemaker_session
):
    patched_tar_and_upload_dir.return_value = UploadedCode(
        s3_prefix="s3://%s/%s" % ("bucket", "key"), script_name="script_name"
    )
    sagemaker_session.boto_region_name = REGION
    sagemaker_session.sagemaker_config = {}

    instance_type = "ml.p2.xlarge"
    instance_count = 1

    training_data_uri = "s3://bucket/mydata"

    jumpstart_source_dir = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[0]}/source_dirs/source.tar.gz"
    )

    hyperparameters = {
        "int_hyperparam": 1,
        "string_hyperparam": "hello",
        "stringified_numeric_hyperparam": "44",
        "float_hyperparam": 1.234,
    }

    generic_estimator = Estimator(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        region=REGION,
        sagemaker_session=sagemaker_session,
        instance_count=instance_count,
        instance_type=instance_type,
        source_dir=jumpstart_source_dir,
        image_uri=IMAGE_URI,
        model_uri=MODEL_DATA,
        hyperparameters=hyperparameters,
    )
    generic_estimator.fit(training_data_uri)

    formatted_hyperparams = EstimatorBase._json_encode_hyperparameters(hyperparameters)

    assert (
        set(formatted_hyperparams.items())
        - set(sagemaker_session.train.call_args_list[0][1]["hyperparameters"].items())
        == set()
    )


@patch("time.time", return_value=TIME)
@patch("sagemaker.estimator.tar_and_upload_dir")
@patch("sagemaker.model.Model._upload_code")
def test_estimator_local_download_dir(
    patched_upload_code, patched_tar_and_upload_dir, sagemaker_session
):
    patched_tar_and_upload_dir.return_value = UploadedCode(
        s3_prefix="s3://%s/%s" % ("bucket", "key"), script_name="script_name"
    )
    sagemaker_session.boto_region_name = REGION
    sagemaker_session.sagemaker_config = {}

    local_download_dir = "some/download/dir"

    sagemaker_session.settings.local_download_dir = local_download_dir

    instance_type = "ml.p2.xlarge"
    instance_count = 1

    training_data_uri = "s3://bucket/mydata"

    jumpstart_source_dir = (
        f"s3://{list(JUMPSTART_GATED_AND_PUBLIC_BUCKET_NAME_SET)[0]}/source_dirs/source.tar.gz"
    )

    generic_estimator = Estimator(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        region=REGION,
        sagemaker_session=sagemaker_session,
        instance_count=instance_count,
        instance_type=instance_type,
        source_dir=jumpstart_source_dir,
        image_uri=IMAGE_URI,
        model_uri=MODEL_DATA,
    )
    generic_estimator.fit(training_data_uri)

    assert (
        patched_tar_and_upload_dir.call_args_list[0][1]["settings"].local_download_dir
        == local_download_dir
    )


@pytest.mark.parametrize(
    "input_key_prefix, input_current_job_name, input_pipeline_config, output_code_s3_prefix",
    [
        (
            "my/prefix",
            "job-name",
            MOCKED_PIPELINE_CONFIG,
            "my/prefix/test-pipeline/code/code-hash-0123456789",
        ),
        ("my/prefix", "job-name", None, "my/prefix/job-name/source"),
        ("", "job-name", MOCKED_PIPELINE_CONFIG, "test-pipeline/code/code-hash-0123456789"),
        ("", "job-name", None, "job-name/source"),
        (None, "job-name", MOCKED_PIPELINE_CONFIG, "test-pipeline/code/code-hash-0123456789"),
        (None, "job-name", None, "job-name/source"),
        (None, None, MOCKED_PIPELINE_CONFIG, "test-pipeline/code/code-hash-0123456789"),
        (None, None, None, "source"),
    ],
)
def test_assign_s3_prefix(
    sagemaker_session,
    input_key_prefix,
    input_current_job_name,
    input_pipeline_config,
    output_code_s3_prefix,
):

    with patch("sagemaker.workflow.utilities._pipeline_config", input_pipeline_config):
        framework = DummyFramework(
            "my_script.py",
            role="DummyRole",
            sagemaker_session=sagemaker_session,
        )
        framework._current_job_name = input_current_job_name
        assert framework._assign_s3_prefix(input_key_prefix) == output_code_s3_prefix


@patch("sagemaker.estimator._TrainingJob.start_new")
@patch("sagemaker.estimator.tar_and_upload_dir")
def test_output_path_default_bucket_and_prefix_combinations(start_new, tar_and_upload_dir):
    def with_user_input(sess):
        framework = DummyFramework(
            "my_script.py",
            role="DummyRole",
            sagemaker_session=sess,
            output_path="s3://test",
        )
        framework.fit(None, job_name=JOB_NAME, wait=False, logs=True)
        start_new.assert_called()  # just to make sure this is patched with a mock
        tar_and_upload_dir.assert_called()  # just to make sure this is patched with a mock
        return framework.output_path

    def without_user_input(sess):
        framework = DummyFramework(
            "my_script.py",
            role="DummyRole",
            sagemaker_session=sess,
        )
        framework.fit(None, job_name=JOB_NAME, wait=False, logs=True)
        start_new.assert_called()  # just to make sure this is patched with a mock
        tar_and_upload_dir.assert_called()  # just to make sure this is patched with a mock
        return framework.output_path

    actual, expected = _test_default_bucket_and_prefix_combinations(
        function_with_user_input=with_user_input,
        function_without_user_input=without_user_input,
        expected__without_user_input__with_default_bucket_and_default_prefix=(
            f"s3://{DEFAULT_S3_BUCKET_NAME}/{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/"
        ),
        expected__without_user_input__with_default_bucket_only=f"s3://{DEFAULT_S3_BUCKET_NAME}/",
        expected__with_user_input__with_default_bucket_and_prefix="s3://test",
        expected__with_user_input__with_default_bucket_only="s3://test",
    )
    assert actual == expected


@patch("sagemaker.estimator.tar_and_upload_dir")
@pytest.mark.parametrize(
    (
        "output_path, code_location,"
        "expected__without_user_input__with_default_bucket_and_default_prefix, "
        "expected__without_user_input__with_default_bucket_only, "
        "expected__with_user_input__with_default_bucket_and_prefix, "
        "expected__with_user_input__with_default_bucket_only"
    ),
    [
        # Group of not-None output_bucket
        (
            "s3://output-bucket/output-prefix/output-prefix2",
            "s3://code-bucket/code-prefix/code-prefix2",
            (DEFAULT_S3_BUCKET_NAME, f"{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{JOB_NAME}/source"),
            ("code-bucket", f"code-prefix/code-prefix2/{JOB_NAME}/source"),
            ("code-bucket", f"code-prefix/code-prefix2/{JOB_NAME}/source"),
        ),
        (
            "s3://output-bucket/output-prefix/output-prefix2",
            None,
            (DEFAULT_S3_BUCKET_NAME, f"{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{JOB_NAME}/source"),
            ("output-bucket", f"{JOB_NAME}/source"),
            ("output-bucket", f"{JOB_NAME}/source"),
        ),
        # Group of None output_bucket
        (
            None,
            "s3://code-bucket/code-prefix/code-prefix2",
            (DEFAULT_S3_BUCKET_NAME, f"{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{JOB_NAME}/source"),
            ("code-bucket", f"code-prefix/code-prefix2/{JOB_NAME}/source"),
            ("code-bucket", f"code-prefix/code-prefix2/{JOB_NAME}/source"),
        ),
        (
            None,
            None,
            (DEFAULT_S3_BUCKET_NAME, f"{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{JOB_NAME}/source"),
        ),
        # Group of PipelineVariable output_bucket
        (
            ExecutionVariable("output_path"),
            "s3://code-bucket/code-prefix/code-prefix2",
            (DEFAULT_S3_BUCKET_NAME, f"{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{JOB_NAME}/source"),
            ("code-bucket", f"code-prefix/code-prefix2/{JOB_NAME}/source"),
            ("code-bucket", f"code-prefix/code-prefix2/{JOB_NAME}/source"),
        ),
        (
            ExecutionVariable("output_path"),
            None,
            (DEFAULT_S3_BUCKET_NAME, f"{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{JOB_NAME}/source"),
        ),
        # Group of file output_bucket
        (
            "file://output-bucket/output-prefix/output-prefix2",
            "s3://code-bucket/code-prefix/code-prefix2",
            (DEFAULT_S3_BUCKET_NAME, f"{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{JOB_NAME}/source"),
            ("code-bucket", f"code-prefix/code-prefix2/{JOB_NAME}/source"),
            ("code-bucket", f"code-prefix/code-prefix2/{JOB_NAME}/source"),
        ),
        (
            "file://output-bucket/output-prefix/output-prefix2",
            None,
            (DEFAULT_S3_BUCKET_NAME, f"{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{DEFAULT_S3_OBJECT_KEY_PREFIX_NAME}/{JOB_NAME}/source"),
            (DEFAULT_S3_BUCKET_NAME, f"{JOB_NAME}/source"),
        ),
    ],
)
def test_stage_user_code_in_s3_default_bucket_and_prefix_combinations(
    tar_and_upload_dir,
    output_path,
    code_location,
    expected__without_user_input__with_default_bucket_and_default_prefix,
    expected__without_user_input__with_default_bucket_only,
    expected__with_user_input__with_default_bucket_and_prefix,
    expected__with_user_input__with_default_bucket_only,
):
    def with_user_input(sess):
        framework = DummyFramework(
            "my_script.py",
            role="DummyRole",
            sagemaker_session=sess,
        )

        if output_path is not None:
            framework.output_path = output_path
        if code_location is not None:
            framework.code_location = code_location

        # this method calls _stage_user_code_in_s3()
        framework._prepare_for_training(job_name=JOB_NAME)
        kwargs = tar_and_upload_dir.call_args.kwargs
        return kwargs["bucket"], kwargs["s3_key_prefix"]

    def without_user_input(sess):
        framework = DummyFramework(
            "my_script.py",
            role="DummyRole",
            sagemaker_session=sess,
        )

        # this method calls _stage_user_code_in_s3()
        framework._prepare_for_training(job_name=JOB_NAME)
        kwargs = tar_and_upload_dir.call_args.kwargs
        return kwargs["bucket"], kwargs["s3_key_prefix"]

    actual, expected = _test_default_bucket_and_prefix_combinations(
        function_with_user_input=with_user_input,
        function_without_user_input=without_user_input,
        expected__without_user_input__with_default_bucket_and_default_prefix=(
            expected__without_user_input__with_default_bucket_and_default_prefix
        ),
        expected__without_user_input__with_default_bucket_only=(
            expected__without_user_input__with_default_bucket_only
        ),
        expected__with_user_input__with_default_bucket_and_prefix=(
            expected__with_user_input__with_default_bucket_and_prefix
        ),
        expected__with_user_input__with_default_bucket_only=(
            expected__with_user_input__with_default_bucket_only
        ),
    )
    assert actual == expected


def test_estimator_get_app_url_success(sagemaker_session):
    job_name = "get-app-url-test-job-name"
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        base_job_name=job_name,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
    )
    f.fit("s3://mydata")

    url = f.get_app_url("TensorBoard", open_in_default_web_browser=False)

    assert url and job_name in url

    app_type = SupportedInteractiveAppTypes.TENSORBOARD
    url = f.get_app_url(app_type, open_in_default_web_browser=False)

    assert url and job_name in url


def test_estimator_get_app_url_fail(sagemaker_session):
    job_name = "get-app-url-test-job-name"
    f = DummyFramework(
        entry_point=SCRIPT_PATH,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        base_job_name=job_name,
        instance_groups=[
            InstanceGroup("group1", "ml.c4.xlarge", 1),
            InstanceGroup("group2", "ml.m4.xlarge", 2),
        ],
    )
    f.fit("s3://mydata")
    with pytest.raises(ValueError) as error:
        f.get_app_url("fake-app")

    assert "does not support URL retrieval." in str(error)
