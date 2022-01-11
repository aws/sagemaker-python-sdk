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
from mock import MagicMock, Mock, patch
from sagemaker.model import FrameworkModel, Model


ENTRY_POINT_INFERENCE = "inference.py"
REGION = "us-west-2"
TIMESTAMP = "2017-11-06-14:14:15.671"
BUCKET_NAME = "mybucket"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.p2.xlarge"
ROLE = "DummyRole"
SCRIPT_URI = "s3://codebucket/someprefix/sourcedir.tar.gz"
IMAGE_URI = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.9.0-gpu-py38"
MODEL_DATA = "s3://someprefix2/models/model.tar.gz"
GIT_REPO = "https://github.com/aws/sagemaker-python-sdk.git"
BRANCH = "test-branch-git-config"
COMMIT = "ae15c9d7d5b97ea95ea451e4662ee43da3401d73"


class DummyFrameworkModel(FrameworkModel):
    def __init__(self, **kwargs):
        super(DummyFrameworkModel, self).__init__(
            **kwargs,
        )


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
    )
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)

    return sms


@patch("time.strftime", MagicMock(return_value=TIMESTAMP))
@patch("sagemaker.utils.repack_model")
def test_script_mode_model_same_calls_as_framework(repack_model, sagemaker_session):
    t = Model(
        entry_point=ENTRY_POINT_INFERENCE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        source_dir=SCRIPT_URI,
        image_uri=IMAGE_URI,
        model_data=MODEL_DATA,
    )
    t.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)

    assert len(sagemaker_session.create_model.call_args_list) == 1
    assert len(sagemaker_session.endpoint_from_production_variants.call_args_list) == 1
    assert len(repack_model.call_args_list) == 1

    generic_model_create_model_args = sagemaker_session.create_model.call_args_list
    generic_model_endpoint_from_production_variants_args = (
        sagemaker_session.endpoint_from_production_variants.call_args_list
    )
    generic_model_repack_model_args = repack_model.call_args_list

    sagemaker_session.create_model.reset_mock()
    sagemaker_session.endpoint_from_production_variants.reset_mock()
    repack_model.reset_mock()

    t = DummyFrameworkModel(
        entry_point=ENTRY_POINT_INFERENCE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        source_dir=SCRIPT_URI,
        image_uri=IMAGE_URI,
        model_data=MODEL_DATA,
    )
    t.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)

    assert generic_model_create_model_args == sagemaker_session.create_model.call_args_list
    assert (
        generic_model_endpoint_from_production_variants_args
        == sagemaker_session.endpoint_from_production_variants.call_args_list
    )
    assert generic_model_repack_model_args == repack_model.call_args_list


@patch("sagemaker.git_utils.git_clone_repo")
@patch("sagemaker.model.fw_utils.tar_and_upload_dir")
def test_git_support_succeed_model_class(tar_and_upload_dir, git_clone_repo, sagemaker_session):
    git_clone_repo.side_effect = lambda gitconfig, entrypoint, sourcedir, dependency: {
        "entry_point": "entry_point",
        "source_dir": "/tmp/repo_dir/source_dir",
        "dependencies": ["/tmp/repo_dir/foo", "/tmp/repo_dir/bar"],
    }
    entry_point = "entry_point"
    source_dir = "source_dir"
    dependencies = ["foo", "bar"]
    git_config = {"repo": GIT_REPO, "branch": BRANCH, "commit": COMMIT}
    model = Model(
        sagemaker_session=sagemaker_session,
        entry_point=entry_point,
        source_dir=source_dir,
        dependencies=dependencies,
        git_config=git_config,
        image_uri=IMAGE_URI,
    )
    model.prepare_container_def(instance_type=INSTANCE_TYPE)
    git_clone_repo.assert_called_with(git_config, entry_point, source_dir, dependencies)
    assert model.entry_point == "entry_point"
    assert model.source_dir == "/tmp/repo_dir/source_dir"
    assert model.dependencies == ["/tmp/repo_dir/foo", "/tmp/repo_dir/bar"]
