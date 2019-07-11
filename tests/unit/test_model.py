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

import copy
import os
import subprocess

import sagemaker
from sagemaker.model import FrameworkModel, ModelPackage
from sagemaker.predictor import RealTimePredictor

import pytest
from mock import MagicMock, Mock, patch

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
ENTRY_POINT = "blah.py"
INSTANCE_TYPE = "p2.xlarge"
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
MODEL_NAME = "{}-{}".format(MODEL_IMAGE, TIMESTAMP)
GIT_REPO = "https://github.com/aws/sagemaker-python-sdk.git"
BRANCH = "test-branch-git-config"
COMMIT = "ae15c9d7d5b97ea95ea451e4662ee43da3401d73"
PRIVATE_GIT_REPO_SSH = "git@github.com:testAccount/private-repo.git"
PRIVATE_GIT_REPO = "https://github.com/testAccount/private-repo.git"
PRIVATE_BRANCH = "test-branch"
PRIVATE_COMMIT = "329bfcf884482002c05ff7f44f62599ebc9f445a"
REPO_DIR = "/tmp/repo_dir"


DESCRIBE_MODEL_PACKAGE_RESPONSE = {
    "InferenceSpecification": {
        "SupportedResponseMIMETypes": ["text"],
        "SupportedContentTypes": ["text/csv"],
        "SupportedTransformInstanceTypes": ["ml.m4.xlarge", "ml.m4.2xlarge"],
        "Containers": [
            {
                "Image": "1.dkr.ecr.us-east-2.amazonaws.com/decision-trees-sample:latest",
                "ImageDigest": "sha256:1234556789",
                "ModelDataUrl": "s3://bucket/output/model.tar.gz",
            }
        ],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m4.xlarge", "ml.m4.2xlarge"],
    },
    "ModelPackageDescription": "Model Package created from training with "
    "arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
    "CreationTime": 1542752036.687,
    "ModelPackageArn": "arn:aws:sagemaker:us-east-2:123:model-package/mp-scikit-decision-trees",
    "ModelPackageStatusDetails": {"ValidationStatuses": [], "ImageScanStatuses": []},
    "SourceAlgorithmSpecification": {
        "SourceAlgorithms": [
            {
                "ModelDataUrl": "s3://bucket/output/model.tar.gz",
                "AlgorithmName": "arn:aws:sagemaker:us-east-2:1234:algorithm/scikit-decision-trees",
            }
        ]
    },
    "ModelPackageStatus": "Completed",
    "ModelPackageName": "mp-scikit-decision-trees-1542410022-2018-11-20-22-13-56-502",
    "CertifyForMarketplace": False,
}

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
@patch("time.strftime", MagicMock(return_value=TIMESTAMP))
def test_deploy(sagemaker_session, tmpdir):
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        MODEL_NAME,
        [
            {
                "InitialVariantWeight": 1,
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        None,
        None,
        True,
    )


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
@patch("time.strftime", MagicMock(return_value=TIMESTAMP))
def test_deploy_endpoint_name(sagemaker_session, tmpdir):
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    model.deploy(endpoint_name="blah", instance_type=INSTANCE_TYPE, initial_instance_count=55)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        "blah",
        [
            {
                "InitialVariantWeight": 1,
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 55,
                "VariantName": "AllTraffic",
            }
        ],
        None,
        None,
        True,
    )


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
@patch("time.strftime", MagicMock(return_value=TIMESTAMP))
def test_deploy_tags(sagemaker_session, tmpdir):
    model = DummyFrameworkModel(sagemaker_session, source_dir=str(tmpdir))
    tags = [{"ModelName": "TestModel"}]
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1, tags=tags)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        MODEL_NAME,
        [
            {
                "InitialVariantWeight": 1,
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        tags,
        None,
        True,
    )


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
@patch("tarfile.open")
@patch("time.strftime", return_value=TIMESTAMP)
def test_deploy_accelerator_type(tfo, time, sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session)
    model.deploy(
        instance_type=INSTANCE_TYPE, initial_instance_count=1, accelerator_type=ACCELERATOR_TYPE
    )
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        MODEL_NAME,
        [
            {
                "InitialVariantWeight": 1,
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
                "AcceleratorType": ACCELERATOR_TYPE,
            }
        ],
        None,
        None,
        True,
    )


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
@patch("tarfile.open")
@patch("time.strftime", return_value=TIMESTAMP)
def test_deploy_kms_key(tfo, time, sagemaker_session):
    key = "some-key-arn"
    model = DummyFrameworkModel(sagemaker_session)
    model.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=1, kms_key=key)
    sagemaker_session.endpoint_from_production_variants.assert_called_with(
        MODEL_NAME,
        [
            {
                "InitialVariantWeight": 1,
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "VariantName": "AllTraffic",
            }
        ],
        None,
        key,
        True,
    )


@patch("sagemaker.session.Session")
@patch("sagemaker.local.LocalSession")
@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_deploy_creates_correct_session(local_session, session, tmpdir):
    # We expect a LocalSession when deploying to instance_type = 'local'
    model = DummyFrameworkModel(sagemaker_session=None, source_dir=str(tmpdir))
    model.deploy(endpoint_name="blah", instance_type="local", initial_instance_count=1)
    assert model.sagemaker_session == local_session.return_value

    # We expect a real Session when deploying to instance_type != local/local_gpu
    model = DummyFrameworkModel(sagemaker_session=None, source_dir=str(tmpdir))
    model.deploy(
        endpoint_name="remote_endpoint", instance_type="ml.m4.4xlarge", initial_instance_count=2
    )
    assert model.sagemaker_session == session.return_value


@patch("sagemaker.fw_utils.tar_and_upload_dir", MagicMock())
def test_deploy_update_endpoint(sagemaker_session, tmpdir):
    model = DummyFrameworkModel(sagemaker_session, source_dir=tmpdir)
    endpoint_name = "endpoint-name"
    model.deploy(
        instance_type=INSTANCE_TYPE,
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        update_endpoint=True,
        accelerator_type=ACCELERATOR_TYPE,
    )
    sagemaker_session.create_endpoint_config.assert_called_with(
        name=model.name,
        model_name=model.name,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        tags=None,
        kms_key=None,
    )
    config_name = sagemaker_session.create_endpoint_config(
        name=model.name,
        model_name=model.name,
        initial_instance_count=INSTANCE_COUNT,
        instance_type=INSTANCE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
    )
    sagemaker_session.update_endpoint.assert_called_with(endpoint_name, config_name)
    sagemaker_session.create_endpoint.assert_not_called()


def test_model_enable_network_isolation(sagemaker_session):
    model = DummyFrameworkModel(sagemaker_session=sagemaker_session)
    assert model.enable_network_isolation() is False


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
def test_model_create_transformer(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=DESCRIBE_MODEL_PACKAGE_RESPONSE
    )

    model = DummyFrameworkModel(sagemaker_session=sagemaker_session)
    model.name = "auto-generated-model"
    transformer = model.transformer(
        instance_count=1, instance_type="ml.m4.xlarge", env={"test": True}
    )
    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.model_name == "auto-generated-model"
    assert transformer.instance_type == "ml.m4.xlarge"
    assert transformer.env == {"test": True}


def test_model_package_enable_network_isolation_with_no_product_id(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=DESCRIBE_MODEL_PACKAGE_RESPONSE
    )

    model_package = ModelPackage(
        role="role", model_package_arn="my-model-package", sagemaker_session=sagemaker_session
    )
    assert model_package.enable_network_isolation() is False


def test_model_package_enable_network_isolation_with_product_id(sagemaker_session):
    model_package_response = copy.deepcopy(DESCRIBE_MODEL_PACKAGE_RESPONSE)
    model_package_response["InferenceSpecification"]["Containers"].append(
        {
            "Image": "1.dkr.ecr.us-east-2.amazonaws.com/some-container:latest",
            "ModelDataUrl": "s3://bucket/output/model.tar.gz",
            "ProductId": "some-product-id",
        }
    )
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=model_package_response
    )

    model_package = ModelPackage(
        role="role", model_package_arn="my-model-package", sagemaker_session=sagemaker_session
    )
    assert model_package.enable_network_isolation() is True


@patch("sagemaker.model.ModelPackage._create_sagemaker_model", Mock())
def test_model_package_create_transformer(sagemaker_session):
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=DESCRIBE_MODEL_PACKAGE_RESPONSE
    )

    model_package = ModelPackage(
        role="role", model_package_arn="my-model-package", sagemaker_session=sagemaker_session
    )
    model_package.name = "auto-generated-model"
    transformer = model_package.transformer(
        instance_count=1, instance_type="ml.m4.xlarge", env={"test": True}
    )
    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.model_name == "auto-generated-model"
    assert transformer.instance_type == "ml.m4.xlarge"
    assert transformer.env == {"test": True}


@patch("sagemaker.model.ModelPackage._create_sagemaker_model", Mock())
def test_model_package_create_transformer_with_product_id(sagemaker_session):
    model_package_response = copy.deepcopy(DESCRIBE_MODEL_PACKAGE_RESPONSE)
    model_package_response["InferenceSpecification"]["Containers"].append(
        {
            "Image": "1.dkr.ecr.us-east-2.amazonaws.com/some-container:latest",
            "ModelDataUrl": "s3://bucket/output/model.tar.gz",
            "ProductId": "some-product-id",
        }
    )
    sagemaker_session.sagemaker_client.describe_model_package = Mock(
        return_value=model_package_response
    )

    model_package = ModelPackage(
        role="role", model_package_arn="my-model-package", sagemaker_session=sagemaker_session
    )
    model_package.name = "auto-generated-model"
    transformer = model_package.transformer(
        instance_count=1, instance_type="ml.m4.xlarge", env={"test": True}
    )
    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.model_name == "auto-generated-model"
    assert transformer.instance_type == "ml.m4.xlarge"
    assert transformer.env is None


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
        " eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "eu-north-1",
        "sa-east-1",
        "us-gov-east-1",
        "us-gov-west-1",
    ]
    neo_support_region = ["us-west-2", "eu-west-1", "us-east-1", "us-east-2", "ap-northeast-1"]
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
