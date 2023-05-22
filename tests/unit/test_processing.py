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

import copy

import pytest
from mock import Mock, patch, MagicMock
from packaging import version

from sagemaker import LocalSession
from sagemaker.dataset_definition.inputs import (
    S3Input,
    DatasetDefinition,
    AthenaDatasetDefinition,
    RedshiftDatasetDefinition,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    Processor,
    ScriptProcessor,
    ProcessingJob,
)
from sagemaker.session_settings import SessionSettings
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.tensorflow.processing import TensorFlowProcessor
from sagemaker.xgboost.processing import XGBoostProcessor
from sagemaker.mxnet.processing import MXNetProcessor
from sagemaker.network import NetworkConfig
from sagemaker.processing import FeatureStoreOutput
from sagemaker.fw_utils import UploadedCode
from sagemaker.workflow.pipeline_context import PipelineSession, _PipelineConfig
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables
from tests.unit import SAGEMAKER_CONFIG_PROCESSING_JOB

BUCKET_NAME = "mybucket"
REGION = "us-west-2"
ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
ECR_HOSTNAME = "ecr.us-west-2.amazonaws.com"
CUSTOM_IMAGE_URI = "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri"
MOCKED_S3_URI = "s3://mocked_s3_uri_from_upload_data"
MOCKED_PIPELINE_CONFIG = _PipelineConfig(
    "test-pipeline", "test-processing-step", "code-hash-abcdefg", "config-hash-abcdefg"
)


@pytest.fixture(autouse=True)
def mock_create_tar_file():
    with patch("sagemaker.utils.create_tar_file", MagicMock()) as create_tar_file:
        yield create_tar_file


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)

    session_mock.upload_data = Mock(name="upload_data", return_value=MOCKED_S3_URI)
    session_mock.download_data = Mock(name="download_data")
    session_mock.expand_role.return_value = ROLE
    session_mock.describe_processing_job = MagicMock(
        name="describe_processing_job", return_value=_get_describe_response_inputs_and_ouputs()
    )

    # For tests which doesn't verify config file injection, operate with empty config
    session_mock.sagemaker_config = {}
    return session_mock


@pytest.fixture()
def pipeline_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)

    session_mock.upload_data = Mock(name="upload_data", return_value=MOCKED_S3_URI)
    session_mock.download_data = Mock(name="download_data")
    session_mock.expand_role.return_value = ROLE
    session_mock.describe_processing_job = MagicMock(
        name="describe_processing_job", return_value=_get_describe_response_inputs_and_ouputs()
    )
    session_mock.__class__ = PipelineSession

    # For tests which doesn't verify config file injection, operate with empty config
    session_mock.sagemaker_config = {}

    return session_mock


@pytest.fixture()
def uploaded_code(
    s3_prefix="s3://mocked_s3_uri_from_upload_data/my_job_name/source/sourcedir.tar.gz",
    script_name="processing_code.py",
):
    return UploadedCode(s3_prefix=s3_prefix, script_name=script_name)


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_sklearn_processor_with_required_parameters(
    exists_mock, isfile_mock, botocore_resolver, sagemaker_session, sklearn_version
):
    botocore_resolver.return_value.construct_endpoint.return_value = {"hostname": ECR_HOSTNAME}
    processor = SKLearnProcessor(
        role=ROLE,
        instance_type="ml.m4.xlarge",
        framework_version=sklearn_version,
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )

    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args(processor._current_job_name)

    sklearn_image_uri = (
        "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:{}-cpu-py3"
    ).format(sklearn_version)
    expected_args["app_specification"]["ImageUri"] = sklearn_image_uri
    sagemaker_session.process.assert_called_with(**expected_args)


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_sklearn_with_all_parameters(
    exists_mock, isfile_mock, botocore_resolver, sklearn_version, sagemaker_session
):
    botocore_resolver.return_value.construct_endpoint.return_value = {"hostname": ECR_HOSTNAME}

    processor = SKLearnProcessor(
        role=ROLE,
        framework_version=sklearn_version,
        instance_type="ml.m4.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="my_sklearn_processor",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=NetworkConfig(
            subnets=["my_subnet_id"],
            security_group_ids=["my_security_group_id"],
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
        ),
        sagemaker_session=sagemaker_session,
    )

    processor.run(
        code="/local/path/to/processing_code.py",
        inputs=_get_data_inputs_all_parameters(),
        outputs=_get_data_outputs_all_parameters(),
        arguments=["--drop-columns", "'SelfEmployed'"],
        wait=True,
        logs=False,
        job_name="my_job_name",
        experiment_config={"ExperimentName": "AnExperiment"},
    )

    expected_args = _get_expected_args_all_parameters(processor._current_job_name)
    sklearn_image_uri = (
        "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:{}-cpu-py3"
    ).format(sklearn_version)
    expected_args["app_specification"]["ImageUri"] = sklearn_image_uri

    sagemaker_session.process.assert_called_with(**expected_args)


def test_local_mode_disables_local_code_by_default():
    processor = Processor(
        image_uri="",
        role=ROLE,
        instance_count=1,
        instance_type="local",
    )

    # Most tests use a fixture for sagemaker_session for consistent behaviour, so this unit test
    # checks that the default initialization disables unsupported 'local_code' mode:
    assert processor.sagemaker_session._disable_local_code
    assert isinstance(processor.sagemaker_session, LocalSession)


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_sklearn_with_all_parameters_via_run_args(
    exists_mock, isfile_mock, botocore_resolver, sklearn_version, sagemaker_session
):
    botocore_resolver.return_value.construct_endpoint.return_value = {"hostname": ECR_HOSTNAME}

    processor = SKLearnProcessor(
        role=ROLE,
        framework_version=sklearn_version,
        instance_type="ml.m4.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="my_sklearn_processor",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=NetworkConfig(
            subnets=["my_subnet_id"],
            security_group_ids=["my_security_group_id"],
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
        ),
        sagemaker_session=sagemaker_session,
    )

    run_args = processor.get_run_args(
        code="/local/path/to/processing_code.py",
        inputs=_get_data_inputs_all_parameters(),
        outputs=_get_data_outputs_all_parameters(),
        arguments=["--drop-columns", "'SelfEmployed'"],
    )

    processor.run(
        code=run_args.code,
        inputs=run_args.inputs,
        outputs=run_args.outputs,
        arguments=run_args.arguments,
        wait=True,
        logs=False,
        experiment_config={"ExperimentName": "AnExperiment"},
    )

    expected_args = _get_expected_args_all_parameters(processor._current_job_name)
    sklearn_image_uri = (
        "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:{}-cpu-py3"
    ).format(sklearn_version)
    expected_args["app_specification"]["ImageUri"] = sklearn_image_uri

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_sklearn_with_all_parameters_via_run_args_called_twice(
    exists_mock, isfile_mock, botocore_resolver, sklearn_version, sagemaker_session
):
    botocore_resolver.return_value.construct_endpoint.return_value = {"hostname": ECR_HOSTNAME}

    processor = SKLearnProcessor(
        role=ROLE,
        framework_version=sklearn_version,
        instance_type="ml.m4.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="my_sklearn_processor",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=NetworkConfig(
            subnets=["my_subnet_id"],
            security_group_ids=["my_security_group_id"],
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
        ),
        sagemaker_session=sagemaker_session,
    )

    run_args = processor.get_run_args(
        code="/local/path/to/processing_code.py",
        inputs=_get_data_inputs_all_parameters(),
        outputs=_get_data_outputs_all_parameters(),
        arguments=["--drop-columns", "'SelfEmployed'"],
    )
    processor.run(
        code=run_args.code,
        inputs=run_args.inputs,
        outputs=run_args.outputs,
        arguments=run_args.arguments,
        wait=True,
        logs=False,
        experiment_config={"ExperimentName": "AnExperiment"},
    )

    expected_args = _get_expected_args_all_parameters(processor._current_job_name)

    sklearn_image_uri = (
        "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:{}-cpu-py3"
    ).format(sklearn_version)
    expected_args["app_specification"]["ImageUri"] = sklearn_image_uri

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_pytorch_processor_with_required_parameters(
    exists_mock,
    isfile_mock,
    botocore_resolver,
    sagemaker_session,
    pytorch_training_version,
    pytorch_training_py_version,
):
    botocore_resolver.return_value.construct_endpoint.return_value = {"hostname": ECR_HOSTNAME}

    processor = PyTorchProcessor(
        role=ROLE,
        instance_type="ml.m4.xlarge",
        framework_version=pytorch_training_version,
        py_version=pytorch_training_py_version,
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )

    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args_modular_code(processor._current_job_name)

    if version.parse(pytorch_training_version) < version.parse("1.2"):
        pytorch_image_uri = (
            "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-pytorch:{}-cpu-{}".format(
                pytorch_training_version, pytorch_training_py_version
            )
        )
    else:
        pytorch_image_uri = (
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:{}-cpu-{}".format(
                pytorch_training_version, pytorch_training_py_version
            )
        )

    expected_args["app_specification"]["ImageUri"] = pytorch_image_uri

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_xgboost_processor_with_required_parameters(
    exists_mock, isfile_mock, botocore_resolver, sagemaker_session, xgboost_framework_version
):
    botocore_resolver.return_value.construct_endpoint.return_value = {"hostname": ECR_HOSTNAME}

    processor = XGBoostProcessor(
        role=ROLE,
        instance_type="ml.m4.xlarge",
        framework_version=xgboost_framework_version,
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )

    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args_modular_code(processor._current_job_name)

    if version.parse(xgboost_framework_version) < version.parse("1.2-1"):
        xgboost_image_uri = (
            "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:{}-cpu-py3"
        ).format(xgboost_framework_version)
    else:
        xgboost_image_uri = (
            "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:{}"
        ).format(xgboost_framework_version)

    expected_args["app_specification"]["ImageUri"] = xgboost_image_uri

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_mxnet_processor_with_required_parameters(
    exists_mock,
    isfile_mock,
    botocore_resolver,
    sagemaker_session,
    mxnet_training_version,
    mxnet_training_py_version,
):
    botocore_resolver.return_value.construct_endpoint.return_value = {"hostname": ECR_HOSTNAME}

    processor = MXNetProcessor(
        role=ROLE,
        instance_type="ml.m4.xlarge",
        framework_version=mxnet_training_version,
        py_version=mxnet_training_py_version,
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )

    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args_modular_code(processor._current_job_name)

    if (mxnet_training_py_version == "py3") & (
        mxnet_training_version == "1.4"
    ):  # probably there is a better way to handle this
        mxnet_image_uri = (
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-training:{}-cpu-{}"
        ).format(mxnet_training_version, mxnet_training_py_version)
    elif version.parse(mxnet_training_version) > version.parse(
        "1.4.1" if mxnet_training_py_version == "py2" else "1.4"
    ):
        mxnet_image_uri = (
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-training:{}-cpu-{}"
        ).format(mxnet_training_version, mxnet_training_py_version)
    else:
        mxnet_image_uri = (
            "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:{}-cpu-{}"
        ).format(mxnet_training_version, mxnet_training_py_version)

    expected_args["app_specification"]["ImageUri"] = mxnet_image_uri

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_tensorflow_processor_with_required_parameters(
    exists_mock,
    isfile_mock,
    botocore_resolver,
    sagemaker_session,
    tensorflow_training_version,
    tensorflow_training_py_version,
):

    botocore_resolver.return_value.construct_endpoint.return_value = {"hostname": ECR_HOSTNAME}

    if version.parse(tensorflow_training_version) <= version.parse("1.13.1"):

        processor = TensorFlowProcessor(
            role=ROLE,
            instance_type="ml.m4.xlarge",
            framework_version=tensorflow_training_version,
            py_version=tensorflow_training_py_version,
            instance_count=1,
            sagemaker_session=sagemaker_session,
            image_uri="520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:{}-cpu-{}".format(
                tensorflow_training_version, tensorflow_training_py_version
            ),
        )
    else:
        processor = TensorFlowProcessor(
            role=ROLE,
            instance_type="ml.m4.xlarge",
            framework_version=tensorflow_training_version,
            py_version=tensorflow_training_py_version,
            instance_count=1,
            sagemaker_session=sagemaker_session,
        )

    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args_modular_code(processor._current_job_name)

    if version.parse(tensorflow_training_version) <= version.parse("1.13.1"):
        tensorflow_image_uri = (
            "520713654638.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:{}-cpu-{}"
        ).format(tensorflow_training_version, tensorflow_training_py_version)
    else:
        tensorflow_image_uri = (
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:{}-cpu-{}"
        ).format(tensorflow_training_version, tensorflow_training_py_version)

    expected_args["app_specification"]["ImageUri"] = tensorflow_image_uri

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=False)
def test_script_processor_errors_with_nonexistent_local_code(exists_mock, sagemaker_session):
    processor = _get_script_processor(sagemaker_session)
    with pytest.raises(ValueError):
        processor.run(code="/local/path/to/processing_code.py")


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=False)
def test_script_processor_errors_with_code_directory(exists_mock, isfile_mock, sagemaker_session):
    processor = _get_script_processor(sagemaker_session)
    with pytest.raises(ValueError):
        processor.run(code="/local/path/to/code")


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_errors_with_invalid_code_url_scheme(
    exists_mock, isfile_mock, sagemaker_session
):
    processor = _get_script_processor(sagemaker_session)
    with pytest.raises(ValueError):
        processor.run(code="hdfs:///path/to/processing_code.py")


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_works_with_absolute_local_path(
    exists_mock, isfile_mock, sagemaker_session
):
    processor = _get_script_processor(sagemaker_session)
    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args(processor._current_job_name, code_s3_uri=MOCKED_S3_URI)

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_works_with_relative_local_path(
    exists_mock, isfile_mock, sagemaker_session
):
    processor = _get_script_processor(sagemaker_session)
    processor.run(code="processing_code.py")

    expected_args = _get_expected_args(processor._current_job_name, code_s3_uri=MOCKED_S3_URI)
    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_works_with_relative_local_path_with_directories(
    exists_mock, isfile_mock, sagemaker_session
):
    processor = _get_script_processor(sagemaker_session)
    processor.run(code="path/to/processing_code.py")
    expected_args = _get_expected_args(processor._current_job_name, code_s3_uri=MOCKED_S3_URI)
    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_works_with_file_code_url_scheme(
    exists_mock, isfile_mock, sagemaker_session
):
    processor = _get_script_processor(sagemaker_session)
    processor.run(code="file:///path/to/processing_code.py")

    expected_args = _get_expected_args(processor._current_job_name, code_s3_uri=MOCKED_S3_URI)
    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_works_with_s3_code_url(exists_mock, isfile_mock, sagemaker_session):
    processor = _get_script_processor(sagemaker_session)
    processor.run(code="s3://bucket/path/to/processing_code.py")

    expected_args = _get_expected_args(
        processor._current_job_name, "s3://bucket/path/to/processing_code.py"
    )
    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_with_one_input(exists_mock, isfile_mock, sagemaker_session):
    processor = _get_script_processor(sagemaker_session)
    processor.run(
        code="/local/path/to/processing_code.py",
        inputs=[
            ProcessingInput(source="/local/path/to/my/dataset/census.csv", destination="/data/")
        ],
    )

    expected_args = _get_expected_args(processor._current_job_name, code_s3_uri=MOCKED_S3_URI)
    expected_args["inputs"].insert(0, _get_data_input())

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_with_required_parameters(exists_mock, isfile_mock, sagemaker_session):
    processor = _get_script_processor(sagemaker_session)

    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args(processor._current_job_name, code_s3_uri=MOCKED_S3_URI)
    sagemaker_session.process.assert_called_with(**expected_args)


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_without_role(exists_mock, isfile_mock, sagemaker_session):
    with pytest.raises(ValueError):
        ScriptProcessor(
            image_uri=CUSTOM_IMAGE_URI,
            command=["python3"],
            instance_type="ml.m4.xlarge",
            instance_count=1,
            volume_size_in_gb=100,
            volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
            output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
            max_runtime_in_seconds=3600,
            base_job_name="my_sklearn_processor",
            env={"my_env_variable": "my_env_variable_value"},
            tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
            network_config=NetworkConfig(
                subnets=["my_subnet_id"],
                security_group_ids=["my_security_group_id"],
                enable_network_isolation=True,
                encrypt_inter_container_traffic=True,
            ),
            sagemaker_session=sagemaker_session,
        )


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_with_sagemaker_config_injection(
    exists_mock, isfile_mock, sagemaker_session
):
    sagemaker_session.sagemaker_config = SAGEMAKER_CONFIG_PROCESSING_JOB

    sagemaker_session.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    sagemaker_session.upload_data = Mock(name="upload_data", return_value=MOCKED_S3_URI)
    sagemaker_session.wait_for_processing_job = MagicMock(
        name="wait_for_processing_job", return_value=_get_describe_response_inputs_and_ouputs()
    )
    sagemaker_session.process = Mock()
    sagemaker_session.expand_role = Mock(name="expand_role", side_effect=lambda a: a)

    processor = ScriptProcessor(
        image_uri=CUSTOM_IMAGE_URI,
        command=["python3"],
        instance_type="ml.m4.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        max_runtime_in_seconds=3600,
        base_job_name="my_sklearn_processor",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        sagemaker_session=sagemaker_session,
    )
    processor.run(
        code="/local/path/to/processing_code.py",
        inputs=_get_data_inputs_all_parameters(),
        outputs=_get_data_outputs_all_parameters(),
        arguments=["--drop-columns", "'SelfEmployed'"],
        wait=True,
        logs=False,
        job_name="my_job_name",
        experiment_config={"ExperimentName": "AnExperiment"},
    )
    expected_args = copy.deepcopy(_get_expected_args_all_parameters(processor._current_job_name))
    expected_volume_kms_key_id = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"]["ProcessingJob"][
        "ProcessingResources"
    ]["ClusterConfig"]["VolumeKmsKeyId"]
    expected_output_kms_key_id = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"]["ProcessingJob"][
        "ProcessingOutputConfig"
    ]["KmsKeyId"]
    expected_role_arn = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"]["ProcessingJob"]["RoleArn"]
    expected_vpc_config = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"]["ProcessingJob"][
        "NetworkConfig"
    ]["VpcConfig"]
    expected_enable_network_isolation = SAGEMAKER_CONFIG_PROCESSING_JOB["SageMaker"][
        "ProcessingJob"
    ]["NetworkConfig"]["EnableNetworkIsolation"]
    expected_enable_inter_containter_traffic_encryption = SAGEMAKER_CONFIG_PROCESSING_JOB[
        "SageMaker"
    ]["ProcessingJob"]["NetworkConfig"]["EnableInterContainerTrafficEncryption"]

    expected_args["resources"]["ClusterConfig"]["VolumeKmsKeyId"] = expected_volume_kms_key_id
    expected_args["output_config"]["KmsKeyId"] = expected_output_kms_key_id
    expected_args["role_arn"] = expected_role_arn
    expected_args["network_config"]["VpcConfig"] = expected_vpc_config
    expected_args["network_config"]["EnableNetworkIsolation"] = expected_enable_network_isolation
    expected_args["network_config"][
        "EnableInterContainerTrafficEncryption"
    ] = expected_enable_inter_containter_traffic_encryption

    sagemaker_session.process.assert_called_with(**expected_args)
    assert "my_job_name" in processor._current_job_name


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_with_all_parameters(exists_mock, isfile_mock, sagemaker_session):
    processor = ScriptProcessor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        command=["python3"],
        instance_type="ml.m4.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="my_sklearn_processor",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=NetworkConfig(
            subnets=["my_subnet_id"],
            security_group_ids=["my_security_group_id"],
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
        ),
        sagemaker_session=sagemaker_session,
    )

    processor.run(
        code="/local/path/to/processing_code.py",
        inputs=_get_data_inputs_all_parameters(),
        outputs=_get_data_outputs_all_parameters(),
        arguments=["--drop-columns", "'SelfEmployed'"],
        wait=True,
        logs=False,
        job_name="my_job_name",
        experiment_config={"ExperimentName": "AnExperiment"},
    )

    expected_args = _get_expected_args_all_parameters(processor._current_job_name)

    sagemaker_session.process.assert_called_with(**expected_args)
    assert "my_job_name" in processor._current_job_name


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_script_processor_with_all_parameters_via_run_args(
    exists_mock, isfile_mock, sagemaker_session
):
    processor = ScriptProcessor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        command=["python3"],
        instance_type="ml.m4.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="my_sklearn_processor",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=NetworkConfig(
            subnets=["my_subnet_id"],
            security_group_ids=["my_security_group_id"],
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
        ),
        sagemaker_session=sagemaker_session,
    )

    run_args = processor.get_run_args(
        code="/local/path/to/processing_code.py",
        inputs=_get_data_inputs_all_parameters(),
        outputs=_get_data_outputs_all_parameters(),
        arguments=["--drop-columns", "'SelfEmployed'"],
    )

    processor.run(
        code=run_args.code,
        inputs=run_args.inputs,
        outputs=run_args.outputs,
        arguments=run_args.arguments,
        wait=True,
        logs=False,
        job_name="my_job_name",
        experiment_config={"ExperimentName": "AnExperiment"},
    )

    expected_args = _get_expected_args_all_parameters(processor._current_job_name)

    sagemaker_session.process.assert_called_with(**expected_args)
    assert "my_job_name" in processor._current_job_name


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
def test_script_processor_code_path_with_pipeline_config(
    exists_mock, isfile_mock, pipeline_session
):
    processor = _get_script_processor(pipeline_session)
    step_args = processor.run(
        code="/local/path/to/processing_code.py",
    )
    # execute process.run() and generate args, S3 paths
    step_args.func(*step_args.func_args, **step_args.func_kwargs)
    pipeline_session.upload_data.assert_called_with(
        path="/local/path/to/processing_code.py",
        bucket="mybucket",
        key_prefix="test-pipeline/code/code-hash-abcdefg",
        extra_args=None,
    )


def test_processor_with_required_parameters(sagemaker_session):
    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    processor.run()

    expected_args = _get_expected_args(processor._current_job_name)
    del expected_args["app_specification"]["ContainerEntrypoint"]
    expected_args["inputs"] = []

    sagemaker_session.process.assert_called_with(**expected_args)


def test_processor_with_missing_network_config_parameters(sagemaker_session):
    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        network_config=NetworkConfig(enable_network_isolation=True),
    )

    processor.run()

    expected_args = _get_expected_args(processor._current_job_name)
    del expected_args["app_specification"]["ContainerEntrypoint"]
    expected_args["inputs"] = []
    expected_args["network_config"] = {"EnableNetworkIsolation": True}

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
def test_processor_with_pipeline_s3_output_paths(pipeline_session):
    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=pipeline_session,
    )

    outputs = [
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
    ]

    step_args = processor.run(outputs=outputs)
    # execute process.run() and generate args, S3 paths
    step_args.func(*step_args.func_args, **step_args.func_kwargs)
    expected_output_config = {
        "Outputs": [
            {
                "OutputName": "train",
                "AppManaged": False,
                "S3Output": {
                    "S3Uri": Join(
                        on="/",
                        values=[
                            "s3:/",
                            "mybucket",
                            "test-pipeline",
                            ExecutionVariables.PIPELINE_EXECUTION_ID,
                            "test-processing-step",
                            "output",
                            "train",
                        ],
                    ),
                    "LocalPath": "/opt/ml/processing/train",
                    "S3UploadMode": "EndOfJob",
                },
            }
        ]
    }
    pipeline_session.process.assert_called_with(
        inputs=[],
        output_config=expected_output_config,
        experiment_config=None,
        job_name=processor._current_job_name,
        resources={
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        stopping_condition=None,
        app_specification={
            "ImageUri": "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri"
        },
        environment=None,
        network_config=None,
        role_arn="arn:aws:iam::012345678901:role/SageMakerRole",
        tags=None,
    )


def test_processor_with_encryption_parameter_in_network_config(sagemaker_session):
    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        network_config=NetworkConfig(encrypt_inter_container_traffic=False),
    )

    processor.run()

    expected_args = _get_expected_args(processor._current_job_name)
    del expected_args["app_specification"]["ContainerEntrypoint"]
    expected_args["inputs"] = []
    expected_args["network_config"] = {
        "EnableNetworkIsolation": False,
        "EnableInterContainerTrafficEncryption": False,
    }

    sagemaker_session.process.assert_called_with(**expected_args)


def test_processor_with_all_parameters(sagemaker_session):
    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        entrypoint=["python3", "/opt/ml/processing/input/code/processing_code.py"],
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="processor_base_name",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=NetworkConfig(
            subnets=["my_subnet_id"],
            security_group_ids=["my_security_group_id"],
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
        ),
    )

    processor.run(
        inputs=_get_data_inputs_all_parameters(),
        outputs=_get_data_outputs_all_parameters(),
        arguments=["--drop-columns", "'SelfEmployed'"],
        wait=True,
        logs=False,
        job_name="my_job_name",
        experiment_config={"ExperimentName": "AnExperiment"},
    )

    expected_args = _get_expected_args_all_parameters(processor._current_job_name)
    # Drop the "code" input from expected values.
    expected_args["inputs"] = expected_args["inputs"][:-1]

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
def test_processor_input_path_with_pipeline_config(pipeline_session):
    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=pipeline_session,
    )

    inputs = [
        ProcessingInput(
            input_name="s3_input",
            s3_input=S3Input(
                local_path="/container/path/",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            ),
        )
    ]

    step_args = processor.run(
        inputs=inputs,
    )
    # execute process.run() and generate args, S3 paths
    step_args.func(*step_args.func_args, **step_args.func_kwargs)
    pipeline_session.upload_data.assert_called_with(
        path=None,
        bucket="mybucket",
        key_prefix="test-pipeline/test-processing-step/input/s3_input",
        extra_args=None,
    )


def test_processing_job_from_processing_arn(sagemaker_session):
    processing_job = ProcessingJob.from_processing_arn(
        sagemaker_session=sagemaker_session,
        processing_job_arn="arn:aws:sagemaker:dummy-region:dummy-account-number:processing-job/dummy-job-name",
    )

    assert isinstance(processing_job, ProcessingJob)
    assert [
        processing_input._to_request_dict() for processing_input in processing_job.inputs
    ] == _get_describe_response_inputs_and_ouputs()["ProcessingInputs"]
    assert [
        processing_output._to_request_dict() for processing_output in processing_job.outputs
    ] == _get_describe_response_inputs_and_ouputs()["ProcessingOutputConfig"]["Outputs"]
    assert (
        processing_job.output_kms_key
        == _get_describe_response_inputs_and_ouputs()["ProcessingOutputConfig"]["KmsKeyId"]
    )


def test_extend_processing_args(sagemaker_session):
    inputs = []
    outputs = []

    processor = Processor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
        network_config=NetworkConfig(encrypt_inter_container_traffic=False),
    )

    extended_inputs, extended_outputs = processor._extend_processing_args([], [])

    assert extended_inputs == inputs
    assert extended_outputs == outputs


@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
@patch("sagemaker.workflow.utilities._pipeline_config", MOCKED_PIPELINE_CONFIG)
def test_pyspark_processor_configuration_path_pipeline_config(
    exists_mock, isfile_mock, pipeline_session
):
    processor = PySparkProcessor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=pipeline_session,
    )

    extended_inputs, extended_outputs = processor._extend_processing_args(
        inputs=[], outputs=[], configuration={"Classification": "hadoop-env", "Properties": {}}
    )

    s3_uri = extended_inputs[0].s3_input.s3_uri
    assert (
        s3_uri
        == "s3://mybucket/test-pipeline/test-processing-step/input/conf/config-hash-abcdefg/configuration.json"
    )


def _get_script_processor(sagemaker_session):
    return ScriptProcessor(
        role=ROLE,
        image_uri=CUSTOM_IMAGE_URI,
        command=["python3"],
        instance_type="ml.m4.xlarge",
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )


def _get_expected_args(job_name, code_s3_uri="s3://mocked_s3_uri_from_upload_data"):
    return {
        "inputs": [
            {
                "InputName": "code",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": code_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {"Outputs": []},
        "job_name": job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "stopping_condition": None,
        "app_specification": {
            "ImageUri": CUSTOM_IMAGE_URI,
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/processing_code.py"],
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
        "experiment_config": None,
    }


def _get_expected_args_modular_code(job_name, code_s3_uri=f"s3://{BUCKET_NAME}"):
    return {
        "inputs": [
            {
                "InputName": "code",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{code_s3_uri}/{job_name}/source/sourcedir.tar.gz",
                    "LocalPath": "/opt/ml/processing/input/code/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "entrypoint",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{code_s3_uri}/{job_name}/source/runproc.sh",
                    "LocalPath": "/opt/ml/processing/input/entrypoint",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {"Outputs": []},
        "experiment_config": None,
        "job_name": job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "stopping_condition": None,
        "app_specification": {
            "ImageUri": CUSTOM_IMAGE_URI,
            "ContainerEntrypoint": [
                "/bin/bash",
                "/opt/ml/processing/input/entrypoint/runproc.sh",
            ],
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
        "experiment_config": None,
    }


def _get_data_input():
    data_input = {
        "InputName": "input-1",
        "AppManaged": False,
        "S3Input": {
            "S3Uri": MOCKED_S3_URI,
            "LocalPath": "/data/",
            "S3DataType": "S3Prefix",
            "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated",
            "S3CompressionType": "None",
        },
    }
    return data_input


def _get_data_inputs_all_parameters():
    return [
        ProcessingInput(
            source="s3://path/to/my/dataset/census.csv",
            destination="/container/path/",
            input_name="my_dataset",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
            s3_data_distribution_type="FullyReplicated",
            s3_compression_type="None",
        ),
        ProcessingInput(
            input_name="s3_input",
            s3_input=S3Input(
                s3_uri="s3://path/to/my/dataset/census.csv",
                local_path="/container/path/",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            ),
        ),
        ProcessingInput(
            input_name="redshift_dataset_definition",
            app_managed=True,
            dataset_definition=DatasetDefinition(
                data_distribution_type="FullyReplicated",
                input_mode="File",
                local_path="/opt/ml/processing/input/dd",
                redshift_dataset_definition=RedshiftDatasetDefinition(
                    cluster_id="cluster_id",
                    database="database",
                    db_user="db_user",
                    query_string="query_string",
                    cluster_role_arn="cluster_role_arn",
                    output_s3_uri="output_s3_uri",
                    kms_key_id="kms_key_id",
                    output_format="CSV",
                    output_compression="SNAPPY",
                ),
            ),
        ),
        ProcessingInput(
            input_name="athena_dataset_definition",
            app_managed=True,
            dataset_definition=DatasetDefinition(
                data_distribution_type="FullyReplicated",
                input_mode="File",
                local_path="/opt/ml/processing/input/dd",
                athena_dataset_definition=AthenaDatasetDefinition(
                    catalog="catalog",
                    database="database",
                    query_string="query_string",
                    output_s3_uri="output_s3_uri",
                    work_group="workgroup",
                    kms_key_id="kms_key_id",
                    output_format="AVRO",
                    output_compression="ZLIB",
                ),
            ),
        ),
    ]


def _get_data_outputs_all_parameters():
    return [
        ProcessingOutput(
            source="/container/path/",
            destination="s3://uri/",
            output_name="my_output",
            s3_upload_mode="EndOfJob",
        ),
        ProcessingOutput(
            output_name="feature_store_output",
            app_managed=True,
            feature_store_output=FeatureStoreOutput(feature_group_name="FeatureGroupName"),
        ),
    ]


def _get_expected_args_all_parameters_modular_code(
    job_name,
    code_s3_uri=MOCKED_S3_URI,
    instance_count=1,
    code_s3_prefix=None,
):
    if code_s3_prefix is None:
        code_s3_prefix = f"{code_s3_uri}/{job_name}/source"

    return {
        "inputs": [
            {
                "InputName": "my_dataset",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": "s3://path/to/my/dataset/census.csv",
                    "LocalPath": "/container/path/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "s3_input",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": "s3://path/to/my/dataset/census.csv",
                    "LocalPath": "/container/path/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "redshift_dataset_definition",
                "AppManaged": True,
                "DatasetDefinition": {
                    "DataDistributionType": "FullyReplicated",
                    "InputMode": "File",
                    "LocalPath": "/opt/ml/processing/input/dd",
                    "RedshiftDatasetDefinition": {
                        "ClusterId": "cluster_id",
                        "Database": "database",
                        "DbUser": "db_user",
                        "QueryString": "query_string",
                        "ClusterRoleArn": "cluster_role_arn",
                        "OutputS3Uri": "output_s3_uri",
                        "KmsKeyId": "kms_key_id",
                        "OutputFormat": "CSV",
                        "OutputCompression": "SNAPPY",
                    },
                },
            },
            {
                "InputName": "athena_dataset_definition",
                "AppManaged": True,
                "DatasetDefinition": {
                    "DataDistributionType": "FullyReplicated",
                    "InputMode": "File",
                    "LocalPath": "/opt/ml/processing/input/dd",
                    "AthenaDatasetDefinition": {
                        "Catalog": "catalog",
                        "Database": "database",
                        "QueryString": "query_string",
                        "OutputS3Uri": "output_s3_uri",
                        "WorkGroup": "workgroup",
                        "KmsKeyId": "kms_key_id",
                        "OutputFormat": "AVRO",
                        "OutputCompression": "ZLIB",
                    },
                },
            },
            {
                "InputName": "code",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{code_s3_prefix}/sourcedir.tar.gz",
                    "LocalPath": "/opt/ml/processing/input/code/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "entrypoint",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{code_s3_prefix}/runproc.sh",
                    "LocalPath": "/opt/ml/processing/input/entrypoint",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {
            "Outputs": [
                {
                    "OutputName": "my_output",
                    "AppManaged": False,
                    "S3Output": {
                        "S3Uri": "s3://uri/",
                        "LocalPath": "/container/path/",
                        "S3UploadMode": "EndOfJob",
                    },
                },
                {
                    "OutputName": "feature_store_output",
                    "AppManaged": True,
                    "FeatureStoreOutput": {"FeatureGroupName": "FeatureGroupName"},
                },
            ],
            "KmsKeyId": "arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        },
        "experiment_config": {"ExperimentName": "AnExperiment"},
        "job_name": job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": instance_count,
                "VolumeSizeInGB": 100,
                "VolumeKmsKeyId": "arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
            }
        },
        "stopping_condition": {"MaxRuntimeInSeconds": 3600},
        "app_specification": {
            "ImageUri": "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri",
            "ContainerArguments": ["--drop-columns", "'SelfEmployed'"],
            "ContainerEntrypoint": [
                "/bin/bash",
                "/opt/ml/processing/input/entrypoint/runproc.sh",
            ],
        },
        "environment": {"my_env_variable": "my_env_variable_value"},
        "network_config": {
            "EnableNetworkIsolation": True,
            "EnableInterContainerTrafficEncryption": True,
            "VpcConfig": {
                "SecurityGroupIds": ["my_security_group_id"],
                "Subnets": ["my_subnet_id"],
            },
        },
        "role_arn": ROLE,
        "tags": [{"Key": "my-tag", "Value": "my-tag-value"}],
    }


def _get_expected_args_all_parameters(job_name):
    return {
        "inputs": [
            {
                "InputName": "my_dataset",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": "s3://path/to/my/dataset/census.csv",
                    "LocalPath": "/container/path/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "s3_input",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": "s3://path/to/my/dataset/census.csv",
                    "LocalPath": "/container/path/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "redshift_dataset_definition",
                "AppManaged": True,
                "DatasetDefinition": {
                    "DataDistributionType": "FullyReplicated",
                    "InputMode": "File",
                    "LocalPath": "/opt/ml/processing/input/dd",
                    "RedshiftDatasetDefinition": {
                        "ClusterId": "cluster_id",
                        "Database": "database",
                        "DbUser": "db_user",
                        "QueryString": "query_string",
                        "ClusterRoleArn": "cluster_role_arn",
                        "OutputS3Uri": "output_s3_uri",
                        "KmsKeyId": "kms_key_id",
                        "OutputFormat": "CSV",
                        "OutputCompression": "SNAPPY",
                    },
                },
            },
            {
                "InputName": "athena_dataset_definition",
                "AppManaged": True,
                "DatasetDefinition": {
                    "DataDistributionType": "FullyReplicated",
                    "InputMode": "File",
                    "LocalPath": "/opt/ml/processing/input/dd",
                    "AthenaDatasetDefinition": {
                        "Catalog": "catalog",
                        "Database": "database",
                        "QueryString": "query_string",
                        "OutputS3Uri": "output_s3_uri",
                        "WorkGroup": "workgroup",
                        "KmsKeyId": "kms_key_id",
                        "OutputFormat": "AVRO",
                        "OutputCompression": "ZLIB",
                    },
                },
            },
            {
                "InputName": "code",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": MOCKED_S3_URI,
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {
            "Outputs": [
                {
                    "OutputName": "my_output",
                    "AppManaged": False,
                    "S3Output": {
                        "S3Uri": "s3://uri/",
                        "LocalPath": "/container/path/",
                        "S3UploadMode": "EndOfJob",
                    },
                },
                {
                    "OutputName": "feature_store_output",
                    "AppManaged": True,
                    "FeatureStoreOutput": {"FeatureGroupName": "FeatureGroupName"},
                },
            ],
            "KmsKeyId": "arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        },
        "job_name": job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 100,
                "VolumeKmsKeyId": "arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
            }
        },
        "stopping_condition": {"MaxRuntimeInSeconds": 3600},
        "app_specification": {
            "ImageUri": "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri",
            "ContainerArguments": ["--drop-columns", "'SelfEmployed'"],
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/processing_code.py"],
        },
        "environment": {"my_env_variable": "my_env_variable_value"},
        "network_config": {
            "EnableNetworkIsolation": True,
            "EnableInterContainerTrafficEncryption": True,
            "VpcConfig": {
                "SecurityGroupIds": ["my_security_group_id"],
                "Subnets": ["my_subnet_id"],
            },
        },
        "role_arn": ROLE,
        "tags": [{"Key": "my-tag", "Value": "my-tag-value"}],
        "experiment_config": {"ExperimentName": "AnExperiment"},
    }


def _get_describe_response_inputs_and_ouputs():
    return {
        "ProcessingInputs": _get_expected_args_all_parameters(None)["inputs"],
        "ProcessingOutputConfig": _get_expected_args_all_parameters(None)["output_config"],
    }
