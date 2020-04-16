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

import os
import subprocess

from sagemaker.model import FrameworkModel
from sagemaker.predictor import RealTimePredictor

import pytest
from mock import MagicMock, Mock, patch

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
ENTRY_POINT = "blah.py"
ROLE = "some-role"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPT_NAME = "dummy_script.py"
SCRIPT_PATH = os.path.join(DATA_DIR, SCRIPT_NAME)
TIMESTAMP = "2017-10-10-14-14-15"
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "c4.4xlarge"
ACCELERATOR_TYPE = "ml.eia.medium"
IMAGE_NAME = "fakeimage"
REGION = "us-west-2"
NEO_REGION_ACCOUNT = "301217895009"
MODEL_NAME = "{}-{}".format(MODEL_IMAGE, TIMESTAMP)
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

DESCRIBE_COMPILATION_JOB_RESPONSE = {
    "CompilationJobStatus": "Completed",
    "ModelArtifacts": {"S3ModelArtifacts": "s3://output-path/model.tar.gz"},
}


class DummyFrameworkModel(FrameworkModel):
    def __init__(self, sagemaker_session, **kwargs):
        super(DummyFrameworkModel, self).__init__(
            MODEL_DATA,
            MODEL_IMAGE,
            ROLE,
            ENTRY_POINT,
            sagemaker_session=sagemaker_session,
            **kwargs
        )

    def create_predictor(self, endpoint_name):
        return RealTimePredictor(endpoint_name, sagemaker_session=self.sagemaker_session)


class DummyFrameworkModelForGit(FrameworkModel):
    def __init__(self, sagemaker_session, entry_point, **kwargs):
        super(DummyFrameworkModelForGit, self).__init__(
            MODEL_DATA,
            MODEL_IMAGE,
            ROLE,
            entry_point=entry_point,
            sagemaker_session=sagemaker_session,
            **kwargs
        )

    def create_predictor(self, endpoint_name):
        return RealTimePredictor(endpoint_name, sagemaker_session=self.sagemaker_session)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    sms = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        s3_client=None,
        s3_resource=None,
    )
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    return sms


@patch("shutil.rmtree", MagicMock())
@patch("tarfile.open", MagicMock())
@patch("os.listdir", MagicMock(return_value=["blah.py"]))
@patch("time.strftime", return_value=TIMESTAMP)
def test_prepare_container_def(time, sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session)
    assert model.prepare_container_def(INSTANCE_TYPE) == {
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


@patch("shutil.rmtree", MagicMock())
@patch("tarfile.open", MagicMock())
@patch("os.listdir", MagicMock(return_value=["blah.py"]))
@patch("time.strftime", return_value=TIMESTAMP)
def test_prepare_container_def_with_network_isolation(time, sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session, enable_network_isolation=True)
    assert model.prepare_container_def(INSTANCE_TYPE) == {
        "Environment": {
            "SAGEMAKER_PROGRAM": ENTRY_POINT,
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_REGION": REGION,
            "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "false",
        },
        "Image": MODEL_IMAGE,
        "ModelDataUrl": MODEL_DATA,
    }


@patch("shutil.rmtree", MagicMock())
@patch("tarfile.open", MagicMock())
@patch("os.path.exists", MagicMock(return_value=True))
@patch("os.path.isdir", MagicMock(return_value=True))
@patch("os.listdir", MagicMock(return_value=["blah.py"]))
@patch("time.strftime", MagicMock(return_value=TIMESTAMP))
def test_create_no_defaults(sagemaker_session, tmpdir):
    model = DummyFrameworkModel(
        sagemaker_session,
        source_dir="sd",
        env={"a": "a"},
        name="name",
        enable_cloudwatch_metrics=True,
        container_log_level=55,
        code_location="s3://cb/cp",
    )

    assert model.prepare_container_def(INSTANCE_TYPE) == {
        "Environment": {
            "SAGEMAKER_PROGRAM": ENTRY_POINT,
            "SAGEMAKER_SUBMIT_DIRECTORY": "s3://cb/cp/name/sourcedir.tar.gz",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "55",
            "SAGEMAKER_REGION": REGION,
            "SAGEMAKER_ENABLE_CLOUDWATCH_METRICS": "true",
            "a": "a",
        },
        "Image": MODEL_IMAGE,
        "ModelDataUrl": MODEL_DATA,
    }


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_deploy_update_endpoint(sagemaker_session, tmpdir):
    model = DummyFrameworkModel(sagemaker_session, source_dir=tmpdir)
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1, update_endpoint=True)
    sagemaker_session.create_endpoint_config.assert_called_with(
        name=model.name,
        model_name=model.name,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=None,
        tags=None,
        kms_key=None,
        data_capture_config_dict=None,
    )
    config_name = sagemaker_session.create_endpoint_config(
        name=model.name,
        model_name=model.name,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
    )
    sagemaker_session.update_endpoint.assert_called_with(model.name, config_name, wait=True)
    sagemaker_session.create_endpoint.assert_not_called()


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_deploy_update_endpoint_optional_args(sagemaker_session, tmpdir):
    endpoint_name = "endpoint-name"
    tags = [{"Key": "Value"}]
    kms_key = "foo"
    data_capture_config = MagicMock()

    model = DummyFrameworkModel(sagemaker_session, source_dir=tmpdir)
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=1,
        update_endpoint=True,
        endpoint_name=endpoint_name,
        accelerator_type=ACCELERATOR_TYPE,
        tags=tags,
        kms_key=kms_key,
        wait=False,
        data_capture_config=data_capture_config,
    )
    sagemaker_session.create_endpoint_config.assert_called_with(
        name=model.name,
        model_name=model.name,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        tags=tags,
        kms_key=kms_key,
        data_capture_config_dict=data_capture_config._to_request_dict(),
    )
    config_name = sagemaker_session.create_endpoint_config(
        name=model.name,
        model_name=model.name,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        wait=False,
    )
    sagemaker_session.update_endpoint.assert_called_with(endpoint_name, config_name, wait=False)
    sagemaker_session.create_endpoint.assert_not_called()


def test_model_enable_network_isolation(sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session=sagemaker_session)
    assert model.enable_network_isolation() is False


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
@patch("time.strftime", MagicMock(return_value=TIMESTAMP))
def test_model_delete_model(sagemaker_session, tmpdir):
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1)
    model.delete_model()

    sagemaker_session.delete_model.assert_called_with(model.name)


def test_delete_non_deployed_model(sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session)
    with pytest.raises(
        ValueError, match="The SageMaker model must be created first before attempting to delete."
    ):
        model.delete_model()


def test_compile_model_for_inferentia(sagemaker_session, tmpdir):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    model.compile(
        target_instance_family="ml_inf",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        framework_version="1.15.0",
        job_name="compile-model",
    )
    assert (
        "{}.dkr.ecr.{}.amazonaws.com/sagemaker-neo-tensorflow:1.15.0-inf-py3".format(
            NEO_REGION_ACCOUNT, REGION
        )
        == model.image
    )
    assert model._is_compiled_model is True


def test_compile_model_for_edge_device(sagemaker_session, tmpdir):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    model.compile(
        target_instance_family="deeplens",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        job_name="compile-model",
    )
    assert model._is_compiled_model is False


def test_compile_model_for_edge_device_tflite(sagemaker_session, tmpdir):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    model.compile(
        target_instance_family="deeplens",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tflite",
        job_name="tflite-compile-model",
    )
    assert model._is_compiled_model is False


def test_compile_model_for_cloud(sagemaker_session, tmpdir):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    model.compile(
        target_instance_family="ml_c4",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        job_name="compile-model",
    )
    assert model._is_compiled_model is True


def test_compile_model_for_cloud_tflite(sagemaker_session, tmpdir):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    model.compile(
        target_instance_family="ml_c4",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tflite",
        job_name="tflite-compile-model",
    )
    assert model._is_compiled_model is True


@patch("sagemaker.session.Session")
@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_compile_creates_session(session):
    session.return_value.boto_region_name = "us-west-2"

    model = DummyFrameworkModel(sagemaker_session=None)
    model.compile(
        target_instance_family="ml_c4",
        input_shape={"data": [1, 3, 1024, 1024]},
        output_path="s3://output",
        role="role",
        framework="tensorflow",
        job_name="compile-model",
    )

    assert model.sagemaker_session == session.return_value


def test_check_neo_region(sagemaker_session, tmpdir):
    sagemaker_session.wait_for_compilation_job = Mock(
        return_value=DESCRIBE_COMPILATION_JOB_RESPONSE
    )
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    ec2_region_list = [
        "us-east-2",
        "us-east-1",
        "us-west-1",
        "us-west-2",
        "ap-east-1",
        "ap-south-1",
        "ap-northeast-3",
        "ap-northeast-2",
        "ap-southeast-1",
        "ap-southeast-2",
        "ap-northeast-1",
        "ca-central-1",
        "cn-north-1",
        "cn-northwest-1",
        "eu-central-1",
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "eu-north-1",
        "sa-east-1",
        "us-gov-east-1",
        "us-gov-west-1",
    ]
    neo_support_region = [
        "us-west-1",
        "us-west-2",
        "us-east-1",
        "us-east-2",
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "eu-central-1",
        "eu-north-1",
        "ap-northeast-1",
        "ap-northeast-2",
        "ap-east-1",
        "ap-south-1",
        "ap-southeast-1",
        "ap-southeast-2",
        "sa-east-1",
        "ca-central-1",
        "me-south-1",
        "cn-north-1",
        "cn-northwest-1",
        "us-gov-west-1",
    ]
    for region_name in ec2_region_list:
        if region_name in neo_support_region:
            assert model.check_neo_region(region_name) is True
        else:
            assert model.check_neo_region(region_name) is False


@patch("sagemaker.git_utils.git_clone_repo")
@patch("sagemaker.model.fw_utils.tar_and_upload_dir")
def test_git_support_succeed(tar_and_upload_dir, git_clone_repo, sagemaker_session):
    git_clone_repo.side_effect = lambda gitconfig, entrypoint, sourcedir, dependency: {
        "entry_point": "entry_point",
        "source_dir": "/tmp/repo_dir/source_dir",
        "dependencies": ["/tmp/repo_dir/foo", "/tmp/repo_dir/bar"],
    }
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    model = DummyFrameworkModelForGit(
        sagemaker_session=sagemaker_session,
        entry_point=entry_point,
        source_dir=source_dir,
        dependencies=dependencies,
        git_config=git_config,
    )
    model.prepare_container_def(instance_type=INSTANCE_TYPE)
    git_clone_repo.assert_called_with(git_config, entry_point, source_dir, dependencies)
    assert model.entry_point == "entry_point"
    assert model.source_dir == "/tmp/repo_dir/source_dir"
    assert model.dependencies == ["/tmp/repo_dir/foo", "/tmp/repo_dir/bar"]


def test_git_support_repo_not_provided(sagemaker_session):
    entry_point = "source_dir/entry_point"
    git_config = {"branch": BRANCH, "commit": COMMIT}
    with pytest.raises(ValueError) as error:
        model = DummyFrameworkModelForGit(
            sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
        )
        model.prepare_container_def(instance_type=INSTANCE_TYPE)
    assert "Please provide a repo for git_config." in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git clone https://github.com/aws/no-such-repo.git /tmp/repo_dir"
    ),
)
def test_git_support_git_clone_fail(sagemaker_session):
    entry_point = "source_dir/entry_point"
    git_config = {"repo": "https://github.com/aws/no-such-repo.git", "branch": BRANCH}
    with pytest.raises(subprocess.CalledProcessError) as error:
        model = DummyFrameworkModelForGit(
            sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
        )
        model.prepare_container_def(instance_type=INSTANCE_TYPE)
    assert "returned non-zero exit status" in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git checkout branch-that-does-not-exist"
    ),
)
def test_git_support_branch_not_exist(git_clone_repo, sagemaker_session):
    entry_point = "source_dir/entry_point"
    git_config = {"repo": GIT_REPO, "branch": "branch-that-does-not-exist", "commit": COMMIT}
    with pytest.raises(subprocess.CalledProcessError) as error:
        model = DummyFrameworkModelForGit(
            sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
        )
        model.prepare_container_def(instance_type=INSTANCE_TYPE)
    assert "returned non-zero exit status" in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git checkout commit-sha-that-does-not-exist"
    ),
)
def test_git_support_commit_not_exist(git_clone_repo, sagemaker_session):
    entry_point = "source_dir/entry_point"
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": "commit-sha-that-does-not-exist"}
    with pytest.raises(subprocess.CalledProcessError) as error:
        model = DummyFrameworkModelForGit(
            sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
        )
        model.prepare_container_def(instance_type=INSTANCE_TYPE)
    assert "returned non-zero exit status" in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=ValueError("Entry point does not exist in the repo."),
)
def test_git_support_entry_point_not_exist(sagemaker_session):
    entry_point = "source_dir/entry_point"
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    with pytest.raises(ValueError) as error:
        model = DummyFrameworkModelForGit(
            sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
        )
        model.prepare_container_def(instance_type=INSTANCE_TYPE)
    assert "Entry point does not exist in the repo." in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=ValueError("Source directory does not exist in the repo."),
)
def test_git_support_source_dir_not_exist(sagemaker_session):
    entry_point = "entry_point"
    source_dir = "source_dir_that_does_not_exist"
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    with pytest.raises(ValueError) as error:
        model = DummyFrameworkModelForGit(
            sagemaker_session=sagemaker_session,
            entry_point=entry_point,
            source_dir=source_dir,
            git_config=git_config,
        )
        model.prepare_container_def(instance_type=INSTANCE_TYPE)
    assert "Source directory does not exist in the repo." in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=ValueError("Dependency no-such-dir does not exist in the repo."),
)
def test_git_support_dependencies_not_exist(sagemaker_session):
    entry_point = "entry_point"
    dependencies = ["foo", "no_such_dir"]
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    with pytest.raises(ValueError) as error:
        model = DummyFrameworkModelForGit(
            sagemaker_session=sagemaker_session,
            entry_point=entry_point,
            dependencies=dependencies,
            git_config=git_config,
        )
        model.prepare_container_def(instance_type=INSTANCE_TYPE)
    assert "Dependency", "does not exist in the repo." in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    },
)
@patch("sagemaker.model.fw_utils.tar_and_upload_dir")
def test_git_support_with_username_password_no_2fa(
    tar_and_upload_dir, git_clone_repo, sagemaker_session
):
    entry_point = "entry_point"
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "username": "username",
        "password": "passw0rd!",
    }
    model = DummyFrameworkModelForGit(
        sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
    )
    model.prepare_container_def(instance_type=INSTANCE_TYPE)
    git_clone_repo.assert_called_with(git_config, entry_point, None, [])
    assert model.entry_point == "/tmp/repo_dir/entry_point"


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    },
)
@patch("sagemaker.model.fw_utils.tar_and_upload_dir")
def test_git_support_with_token_2fa(tar_and_upload_dir, git_clone_repo, sagemaker_session):
    entry_point = "entry_point"
    git_config = {
        "repo": PRIVATE_GIT_REPO,
        "branch": PRIVATE_BRANCH,
        "commit": PRIVATE_COMMIT,
        "token": "my-token",
        "2FA_enabled": True,
    }
    model = DummyFrameworkModelForGit(
        sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
    )
    model.prepare_container_def(instance_type=INSTANCE_TYPE)
    git_clone_repo.assert_called_with(git_config, entry_point, None, [])
    assert model.entry_point == "/tmp/repo_dir/entry_point"


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    },
)
@patch("sagemaker.model.fw_utils.tar_and_upload_dir")
def test_git_support_ssh_no_passphrase_needed(
    tar_and_upload_dir, git_clone_repo, sagemaker_session
):
    entry_point = "entry_point"
    git_config = {"repo": PRIVATE_GIT_REPO_SSH, "branch": PRIVATE_BRANCH, "commit": PRIVATE_COMMIT}
    model = DummyFrameworkModelForGit(
        sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
    )
    model.prepare_container_def(instance_type=INSTANCE_TYPE)
    git_clone_repo.assert_called_with(git_config, entry_point, None, [])
    assert model.entry_point == "/tmp/repo_dir/entry_point"


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git clone {} {}".format(PRIVATE_GIT_REPO_SSH, REPO_DIR)
    ),
)
@patch("sagemaker.model.fw_utils.tar_and_upload_dir")
def test_git_support_ssh_passphrase_required(tar_and_upload_dir, git_clone_repo, sagemaker_session):
    entry_point = "entry_point"
    git_config = {"repo": PRIVATE_GIT_REPO_SSH, "branch": PRIVATE_BRANCH, "commit": PRIVATE_COMMIT}
    with pytest.raises(subprocess.CalledProcessError) as error:
        model = DummyFrameworkModelForGit(
            sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
        )
        model.prepare_container_def(instance_type=INSTANCE_TYPE)
    assert "returned non-zero exit status" in str(error)


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    },
)
@patch("sagemaker.model.fw_utils.tar_and_upload_dir")
def test_git_support_codecommit_with_username_and_password_succeed(
    tar_and_upload_dir, git_clone_repo, sagemaker_session
):
    entry_point = "entry_point"
    git_config = {
        "repo": CODECOMMIT_REPO,
        "branch": CODECOMMIT_BRANCH,
        "username": "username",
        "password": "passw0rd!",
    }
    model = DummyFrameworkModelForGit(
        sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
    )
    model.prepare_container_def(instance_type=INSTANCE_TYPE)
    git_clone_repo.assert_called_with(git_config, entry_point, None, [])
    assert model.entry_point == "/tmp/repo_dir/entry_point"


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=lambda gitconfig, entrypoint, source_dir=None, dependencies=None: {
        "entry_point": "/tmp/repo_dir/entry_point",
        "source_dir": None,
        "dependencies": None,
    },
)
@patch("sagemaker.model.fw_utils.tar_and_upload_dir")
def test_git_support_codecommit_ssh_no_passphrase_needed(
    tar_and_upload_dir, git_clone_repo, sagemaker_session
):
    entry_point = "entry_point"
    git_config = {"repo": CODECOMMIT_REPO_SSH, "branch": CODECOMMIT_BRANCH}
    model = DummyFrameworkModelForGit(
        sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
    )
    model.prepare_container_def(instance_type=INSTANCE_TYPE)
    git_clone_repo.assert_called_with(git_config, entry_point, None, [])
    assert model.entry_point == "/tmp/repo_dir/entry_point"


@patch(
    "sagemaker.git_utils.git_clone_repo",
    side_effect=subprocess.CalledProcessError(
        returncode=1, cmd="git clone {} {}".format(PRIVATE_GIT_REPO_SSH, REPO_DIR)
    ),
)
@patch("sagemaker.model.fw_utils.tar_and_upload_dir")
def test_git_support_codecommit_ssh_passphrase_required(
    tar_and_upload_dir, git_clone_repo, sagemaker_session
):
    entry_point = "entry_point"
    git_config = {"repo": CODECOMMIT_REPO_SSH, "branch": CODECOMMIT_BRANCH}
    with pytest.raises(subprocess.CalledProcessError) as error:
        model = DummyFrameworkModelForGit(
            sagemaker_session=sagemaker_session, entry_point=entry_point, git_config=git_config
        )
        model.prepare_container_def(instance_type=INSTANCE_TYPE)
    assert "returned non-zero exit status" in str(error)
