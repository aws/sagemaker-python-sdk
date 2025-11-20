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
"""Unit tests for workflow notebook_job_step."""
from __future__ import absolute_import

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.mlops.workflow.notebook_job_step import NotebookJobStep
from sagemaker.mlops.workflow.retry import RetryPolicy
from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.core.config.config_schema import (
    NOTEBOOK_JOB_ROLE_ARN,
    NOTEBOOK_JOB_S3_ROOT_URI,
)


@pytest.fixture
def mock_session():
    session = Mock()
    session.boto_region_name = "us-west-2"
    session.default_bucket.return_value = "test-bucket"
    session.default_bucket_prefix = "prefix"
    session.boto_session = Mock()
    session.sagemaker_config = {}
    return session


@pytest.fixture
def temp_notebook():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False) as f:
        f.write('{"cells":[]}')
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def temp_script():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write("#!/bin/bash\necho test")
        path = f.name
    yield path
    os.unlink(path)


def test_notebook_job_step_module_exists():
    """Test NotebookJobStep module can be imported"""
    from sagemaker.mlops.workflow import notebook_job_step
    assert notebook_job_step is not None


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_init_with_minimal_params(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    assert step.input_notebook == temp_notebook
    assert step.kernel_name == "python3"
    assert step._scheduler_container_entry_point == ["amazon_sagemaker_scheduler"]
    assert step._scheduler_container_arguments == []


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_init_with_all_params(mock_uploader, mock_context, temp_notebook, temp_script, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        name="test-step",
        display_name="Test Step",
        description="Test description",
        notebook_job_name="test-job",
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root",
        parameters={"param1": "value1"},
        environment_variables={"ENV1": "val1"},
        initialization_script=temp_script,
        s3_kms_key="kms-key",
        instance_type="ml.m5.xlarge",
        volume_size=50,
        volume_kms_key="vol-kms-key",
        encrypt_inter_container_traffic=False,
        security_group_ids=["sg-123"],
        subnets=["subnet-123"],
        max_retry_attempts=3,
        max_runtime_in_seconds=3600,
        tags={"key": "value"},
        additional_dependencies=[]
    )
    assert step.name == "test-step"
    assert step.display_name == "Test Step"


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_validate_invalid_notebook_job_name(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        notebook_job_name="123-invalid",
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    with pytest.raises(ValueError, match="Notebook Job Name.*is not valid"):
        step.arguments


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
def test_validate_missing_notebook(mock_context, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook="/nonexistent/notebook.ipynb",
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    with pytest.raises(ValueError, match="input notebook.*is not a valid file"):
        step.arguments


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_validate_invalid_init_script(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        initialization_script="/nonexistent/script.sh",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    with pytest.raises(ValueError, match="initialization script.*is not a valid file"):
        step.arguments


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_validate_invalid_additional_dependencies(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        additional_dependencies=["/nonexistent/path"],
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    with pytest.raises(ValueError, match="path.*does not exist"):
        step.arguments


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_validate_invalid_image_uri(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-east-1.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    with pytest.raises(ValueError, match="image uri.*should be hosted in same region"):
        step.arguments


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_validate_missing_kernel_name(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    with pytest.raises(ValueError, match="kernel name is required"):
        step.arguments


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_properties(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        name="test-step",
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    props = step.properties
    assert hasattr(props, "ComputingJobName")
    assert hasattr(props, "ComputingJobStatus")
    assert hasattr(props, "NotebookJobInputLocation")
    assert hasattr(props, "NotebookJobOutputLocationPrefix")
    assert hasattr(props, "InputNotebookName")
    assert hasattr(props, "OutputNotebookName")


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_depends_on_setter_raises_error(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    with pytest.raises(ValueError, match="Cannot set depends_on"):
        step.depends_on = []


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_arguments_generation(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        name="test-step",
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    args = step.arguments
    assert "TrainingJobName" in args
    assert "RoleArn" in args
    assert "AlgorithmSpecification" in args
    assert "InputDataConfig" in args
    assert "OutputDataConfig" in args
    assert "ResourceConfig" in args
    assert "Environment" in args
    assert "Tags" in args


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_prepare_tags(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        notebook_job_name="test-job",
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root",
        tags={"custom": "tag"}
    )
    step.arguments
    tags = step._prepare_tags()
    assert any(t["Key"] == "sagemaker:name" for t in tags)
    assert any(t["Key"] == "sagemaker:notebook-job-origin" for t in tags)


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_prepare_env_variables(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root",
        environment_variables={"CUSTOM": "value"}
    )
    step.arguments
    envs = step._prepare_env_variables()
    assert "SM_KERNEL_NAME" in envs
    assert "SM_INPUT_NOTEBOOK_NAME" in envs
    assert "CUSTOM" in envs


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_get_job_name_prefix(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    result = step._get_job_name_prefix("test_job@name#123")
    assert result == "test-job-name-123"


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_to_request(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    request = step.to_request()
    assert isinstance(request, dict)


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_init_derives_name_from_notebook(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    assert step.name is not None
    assert step.notebook_job_name is not None


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
@patch("sagemaker.mlops.workflow.notebook_job_step.resolve_value_from_config")
@patch("sagemaker.mlops.workflow.notebook_job_step.get_execution_role")
def test_resolve_defaults_no_role(mock_get_role, mock_resolve, mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    mock_resolve.side_effect = lambda direct_input, config_path, sagemaker_session, default_value=None: direct_input or default_value
    mock_get_role.return_value = "arn:aws:iam::123456789:role/DefaultRole"
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3"
    )
    step.arguments
    mock_get_role.assert_called_once()


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
@patch("sagemaker.mlops.workflow.notebook_job_step.resolve_value_from_config")
@patch("sagemaker.mlops.workflow.notebook_job_step.expand_role")
def test_resolve_defaults_with_role(mock_expand, mock_resolve, mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    def resolve_side_effect(direct_input, config_path, sagemaker_session, default_value=None):
        if config_path == NOTEBOOK_JOB_ROLE_ARN:
            return "role-from-config"
        if config_path == NOTEBOOK_JOB_S3_ROOT_URI:
            return "s3://test-bucket/root"
        return direct_input or default_value
    mock_resolve.side_effect = resolve_side_effect
    mock_expand.return_value = "arn:aws:iam::123456789:role/ExpandedRole"
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3"
    )
    step.arguments
    mock_expand.assert_called_once()


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_prepare_env_with_init_script(mock_uploader, mock_context, temp_notebook, temp_script, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        initialization_script=temp_script,
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    step.arguments
    envs = step._prepare_env_variables()
    assert "SM_INIT_SCRIPT" in envs


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_arguments_with_init_script(mock_uploader, mock_context, temp_notebook, temp_script, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        initialization_script=temp_script,
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    args = step.arguments
    mock_uploader.upload.assert_called_once()


@pytest.fixture
def temp_dir():
    import tempfile
    temp_path = tempfile.mkdtemp()
    yield temp_path
    import shutil
    shutil.rmtree(temp_path)


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_arguments_with_additional_dependencies(mock_uploader, mock_context, temp_notebook, temp_dir, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        additional_dependencies=[temp_dir],
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    args = step.arguments
    mock_uploader.upload.assert_called_once()


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_arguments_with_s3_kms_key(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        s3_kms_key="kms-key-123",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    args = step.arguments
    assert args["OutputDataConfig"]["KmsKeyId"] == "kms-key-123"


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_arguments_with_volume_kms_key(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        volume_kms_key="vol-kms-key-123",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    args = step.arguments
    assert args["ResourceConfig"]["VolumeKmsKeyId"] == "vol-kms-key-123"


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
@patch("sagemaker.mlops.workflow.notebook_job_step.vpc_utils")
def test_arguments_with_vpc_config(mock_vpc, mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    mock_vpc.to_dict.return_value = {"Subnets": ["subnet-123"], "SecurityGroupIds": ["sg-123"]}
    mock_vpc.sanitize.return_value = {"Subnets": ["subnet-123"], "SecurityGroupIds": ["sg-123"]}
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        subnets=["subnet-123"],
        security_group_ids=["sg-123"],
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    args = step.arguments
    assert "VpcConfig" in args


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_arguments_with_parameters(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        parameters={"param1": "value1"},
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    args = step.arguments
    assert "HyperParameters" in args
    assert args["HyperParameters"]["param1"] == "value1"


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_arguments_without_context(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = None
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    with pytest.raises(AttributeError):
        step.arguments


@patch("sagemaker.mlops.workflow.notebook_job_step._tmpdir")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_upload_job_files_with_file(mock_uploader, mock_tmpdir, temp_notebook, mock_session):
    import tempfile
    temp_folder = tempfile.mkdtemp()
    mock_tmpdir.return_value.__enter__ = Mock(return_value=temp_folder)
    mock_tmpdir.return_value.__exit__ = Mock(return_value=False)
    
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    step._upload_job_files("s3://bucket/path", [temp_notebook], None, mock_session)
    mock_uploader.upload.assert_called_once()
    import shutil
    shutil.rmtree(temp_folder)


@patch("sagemaker.mlops.workflow.notebook_job_step._tmpdir")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_upload_job_files_with_dir(mock_uploader, mock_tmpdir, temp_dir, mock_session):
    import tempfile
    temp_folder = tempfile.mkdtemp()
    mock_tmpdir.return_value.__enter__ = Mock(return_value=temp_folder)
    mock_tmpdir.return_value.__exit__ = Mock(return_value=False)
    
    step = NotebookJobStep(
        input_notebook=temp_dir + "/test.ipynb",
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    with open(temp_dir + "/test.ipynb", "w") as f:
        f.write('{"cells":[]}')
    step._upload_job_files("s3://bucket/path", [temp_dir], None, mock_session)
    mock_uploader.upload.assert_called_once()
    import shutil
    shutil.rmtree(temp_folder)


@patch("sagemaker.mlops.workflow.notebook_job_step.load_step_compilation_context")
@patch("sagemaker.mlops.workflow.notebook_job_step.S3Uploader")
def test_arguments_with_container_arguments(mock_uploader, mock_context, temp_notebook, mock_session):
    mock_context.return_value = Mock(sagemaker_session=mock_session, pipeline_name="test-pipeline")
    step = NotebookJobStep(
        input_notebook=temp_notebook,
        image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        kernel_name="python3",
        role="arn:aws:iam::123456789:role/TestRole",
        s3_root_uri="s3://test-bucket/root"
    )
    step._scheduler_container_arguments = ["arg1", "arg2"]
    args = step.arguments
    assert "ContainerArguments" in args["AlgorithmSpecification"]
