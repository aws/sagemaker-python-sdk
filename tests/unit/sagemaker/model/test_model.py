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
from unittest.mock import MagicMock

import pytest
from mock import Mock, patch

import sagemaker
from sagemaker.model import FrameworkModel, Model
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.jumpstart.constants import JUMPSTART_BUCKET_NAME_SET, JUMPSTART_RESOURCE_BASE_NAME
from sagemaker.jumpstart.enums import JumpStartTag
from sagemaker.mxnet.model import MXNetModel
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.tensorflow.model import TensorFlowModel
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.workflow.properties import Properties


MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"
TIMESTAMP = "2017-10-10-14-14-15"
MODEL_NAME = "{}-{}".format(MODEL_IMAGE, TIMESTAMP)

INSTANCE_COUNT = 2
INSTANCE_TYPE = "ml.c4.4xlarge"
ROLE = "some-role"

REGION = "us-west-2"
BUCKET_NAME = "some-bucket-name"
GIT_REPO = "https://github.com/aws/sagemaker-python-sdk.git"
BRANCH = "test-branch-git-config"
COMMIT = "ae15c9d7d5b97ea95ea451e4662ee43da3401d73"
ENTRY_POINT_INFERENCE = "inference.py"
SCRIPT_URI = "s3://codebucket/someprefix/sourcedir.tar.gz"
IMAGE_URI = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.9.0-gpu-py38"


MODEL_DESCRIPTION = "a description"

SUPPORTED_REALTIME_INFERENCE_INSTANCE_TYPES = ["ml.m4.xlarge"]
SUPPORTED_BATCH_TRANSFORM_INSTANCE_TYPES = ["ml.m4.xlarge"]

SUPPORTED_CONTENT_TYPES = ["text/csv", "application/json", "application/jsonlines"]
SUPPORTED_RESPONSE_MIME_TYPES = ["application/json", "text/csv", "application/jsonlines"]

VALIDATION_FILE_NAME = "input.csv"
VALIDATION_INPUT_PATH = "s3://" + BUCKET_NAME + "/validation-input-csv/"
VALIDATION_OUTPUT_PATH = "s3://" + BUCKET_NAME + "/validation-output-csv/"

VALIDATION_SPECIFICATION = {
    "ValidationRole": "some_role",
    "ValidationProfiles": [
        {
            "ProfileName": "Validation-test",
            "TransformJobDefinition": {
                "BatchStrategy": "SingleRecord",
                "TransformInput": {
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": VALIDATION_INPUT_PATH,
                        }
                    },
                    "ContentType": SUPPORTED_CONTENT_TYPES[0],
                },
                "TransformOutput": {
                    "S3OutputPath": VALIDATION_OUTPUT_PATH,
                },
                "TransformResources": {
                    "InstanceType": SUPPORTED_BATCH_TRANSFORM_INSTANCE_TYPES[0],
                    "InstanceCount": 1,
                },
            },
        },
    ],
}


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
        default_bucket_prefix=None,
    )
    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    # For tests which doesn't verify config file injection, operate with empty config
    sms.sagemaker_config = {}
    return sms


@patch("shutil.rmtree", MagicMock())
@patch("tarfile.open", MagicMock())
@patch("os.listdir", MagicMock(return_value=[ENTRY_POINT_INFERENCE]))
def test_prepare_container_def_with_model_src_s3_returns_correct_url(sagemaker_session):
    model = Model(
        entry_point=ENTRY_POINT_INFERENCE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        source_dir=SCRIPT_URI,
        image_uri=MODEL_IMAGE,
        model_data=Properties("Steps.MyStep"),
    )
    container_def = model.prepare_container_def(INSTANCE_TYPE, "ml.eia.medium")

    assert container_def["Environment"]["SAGEMAKER_SUBMIT_DIRECTORY"] == SCRIPT_URI


def test_prepare_container_def_with_model_data():
    model = Model(MODEL_IMAGE)
    container_def = model.prepare_container_def(INSTANCE_TYPE, "ml.eia.medium")

    expected = {"Image": MODEL_IMAGE, "Environment": {}}
    assert expected == container_def


def test_prepare_container_def_with_model_data_and_env():
    env = {"FOO": "BAR"}
    model = Model(MODEL_IMAGE, MODEL_DATA, env=env)

    expected = {"Image": MODEL_IMAGE, "Environment": env, "ModelDataUrl": MODEL_DATA}

    container_def = model.prepare_container_def(INSTANCE_TYPE, "ml.eia.medium")
    assert expected == container_def

    container_def = model.prepare_container_def()
    assert expected == container_def


def test_prepare_container_def_with_image_config():
    image_config = {"RepositoryAccessMode": "Vpc"}
    model = Model(MODEL_IMAGE, image_config=image_config)

    expected = {
        "Image": MODEL_IMAGE,
        "ImageConfig": {"RepositoryAccessMode": "Vpc"},
        "Environment": {},
    }

    container_def = model.prepare_container_def()
    assert expected == container_def


def test_model_enable_network_isolation():
    model = Model(MODEL_IMAGE, MODEL_DATA)
    assert model.enable_network_isolation() is False

    model = Model(MODEL_IMAGE, MODEL_DATA, enable_network_isolation=True)
    assert model.enable_network_isolation()


@patch("sagemaker.model.Model.prepare_container_def")
def test_create_sagemaker_model(prepare_container_def, sagemaker_session):
    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(MODEL_DATA, MODEL_IMAGE, name=MODEL_NAME, sagemaker_session=sagemaker_session)
    model._create_sagemaker_model()

    prepare_container_def.assert_called_with(
        None, accelerator_type=None, serverless_inference_config=None
    )
    sagemaker_session.create_model.assert_called_with(
        name=MODEL_NAME,
        role=None,
        container_defs=container_def,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )


@patch("sagemaker.model.Model.prepare_container_def")
def test_create_sagemaker_model_instance_type(prepare_container_def, sagemaker_session):
    model = Model(MODEL_DATA, MODEL_IMAGE, name=MODEL_NAME, sagemaker_session=sagemaker_session)
    model._create_sagemaker_model(INSTANCE_TYPE)

    prepare_container_def.assert_called_with(
        INSTANCE_TYPE, accelerator_type=None, serverless_inference_config=None
    )


@patch("sagemaker.model.Model.prepare_container_def")
def test_create_sagemaker_model_accelerator_type(prepare_container_def, sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, name=MODEL_NAME, sagemaker_session=sagemaker_session)

    accelerator_type = "ml.eia.medium"
    model._create_sagemaker_model(INSTANCE_TYPE, accelerator_type=accelerator_type)

    prepare_container_def.assert_called_with(
        INSTANCE_TYPE, accelerator_type=accelerator_type, serverless_inference_config=None
    )


@patch("sagemaker.model.Model.prepare_container_def")
def test_create_sagemaker_model_tags(prepare_container_def, sagemaker_session):
    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(MODEL_IMAGE, MODEL_DATA, name=MODEL_NAME, sagemaker_session=sagemaker_session)

    tags = {"Key": "foo", "Value": "bar"}
    model._create_sagemaker_model(INSTANCE_TYPE, tags=tags)

    sagemaker_session.create_model.assert_called_with(
        name=MODEL_NAME,
        role=None,
        container_defs=container_def,
        vpc_config=None,
        enable_network_isolation=False,
        tags=tags,
    )


@patch("sagemaker.model.Model.prepare_container_def")
@patch("sagemaker.utils.name_from_base")
@patch("sagemaker.utils.base_name_from_image")
def test_create_sagemaker_model_optional_model_params(
    base_name_from_image, name_from_base, prepare_container_def, sagemaker_session
):
    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    vpc_config = {"Subnets": ["123"], "SecurityGroupIds": ["456", "789"]}

    model = Model(
        MODEL_IMAGE,
        MODEL_DATA,
        name=MODEL_NAME,
        role=ROLE,
        vpc_config=vpc_config,
        enable_network_isolation=True,
        sagemaker_session=sagemaker_session,
    )
    model._create_sagemaker_model(INSTANCE_TYPE)

    base_name_from_image.assert_not_called()
    name_from_base.assert_not_called()

    sagemaker_session.create_model.assert_called_with(
        name=MODEL_NAME,
        role=ROLE,
        container_defs=container_def,
        vpc_config=vpc_config,
        enable_network_isolation=True,
        tags=None,
    )


@patch("sagemaker.model.Model.prepare_container_def")
@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
@patch("sagemaker.utils.base_name_from_image")
def test_create_sagemaker_model_generates_model_name(
    base_name_from_image, name_from_base, prepare_container_def, sagemaker_session
):
    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(
        MODEL_IMAGE,
        MODEL_DATA,
        sagemaker_session=sagemaker_session,
    )
    model._create_sagemaker_model(INSTANCE_TYPE)

    base_name_from_image.assert_called_with(MODEL_IMAGE, default_base_name="Model")
    name_from_base.assert_called_with(base_name_from_image.return_value)

    sagemaker_session.create_model.assert_called_with(
        name=MODEL_NAME,
        role=None,
        container_defs=container_def,
        vpc_config=None,
        enable_network_isolation=False,
        tags=None,
    )


@patch("sagemaker.model.Model.prepare_container_def")
@patch("sagemaker.utils.name_from_base", return_value=MODEL_NAME)
@patch("sagemaker.utils.base_name_from_image")
def test_create_sagemaker_model_generates_model_name_each_time(
    base_name_from_image, name_from_base, prepare_container_def, sagemaker_session
):
    container_def = {"Image": MODEL_IMAGE, "Environment": {}, "ModelDataUrl": MODEL_DATA}
    prepare_container_def.return_value = container_def

    model = Model(
        MODEL_IMAGE,
        MODEL_DATA,
        sagemaker_session=sagemaker_session,
    )
    model._create_sagemaker_model(INSTANCE_TYPE)
    model._create_sagemaker_model(INSTANCE_TYPE)

    base_name_from_image.assert_called_once_with(MODEL_IMAGE, default_base_name="Model")
    name_from_base.assert_called_with(base_name_from_image.return_value)
    assert 2 == name_from_base.call_count


@patch("sagemaker.session.Session")
@patch("sagemaker.local.LocalSession")
def test_create_sagemaker_model_creates_correct_session(local_session, session):
    local_session.return_value.sagemaker_config = {}
    session.return_value.sagemaker_config = {}
    model = Model(MODEL_IMAGE, MODEL_DATA)
    model._create_sagemaker_model("local")
    assert model.sagemaker_session == local_session.return_value

    model = Model(MODEL_IMAGE, MODEL_DATA)
    model._create_sagemaker_model("ml.m5.xlarge")
    assert model.sagemaker_session == session.return_value


@patch("sagemaker.model.Model._create_sagemaker_model")
def test_model_create_transformer(create_sagemaker_model, sagemaker_session):
    model_name = "auto-generated-model"
    model = Model(MODEL_IMAGE, MODEL_DATA, name=model_name, sagemaker_session=sagemaker_session)

    instance_type = "ml.m4.xlarge"
    transformer = model.transformer(instance_count=1, instance_type=instance_type)

    create_sagemaker_model.assert_called_with(instance_type, tags=None)

    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.model_name == model_name
    assert transformer.instance_type == instance_type
    assert transformer.instance_count == 1
    assert transformer.sagemaker_session == sagemaker_session
    assert transformer.base_transform_job_name == model_name

    assert transformer.strategy is None
    assert transformer.env is None
    assert transformer.output_path is None
    assert transformer.output_kms_key is None
    assert transformer.accept is None
    assert transformer.assemble_with is None
    assert transformer.volume_kms_key is None
    assert transformer.max_concurrent_transforms is None
    assert transformer.max_payload is None
    assert transformer.tags is None


@patch("sagemaker.model.Model._create_sagemaker_model")
def test_model_create_transformer_optional_params(create_sagemaker_model, sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session)

    instance_type = "ml.m4.xlarge"
    strategy = "MultiRecord"
    assemble_with = "Line"
    output_path = "s3://bucket/path"
    kms_key = "key"
    accept = "text/csv"
    env = {"test": True}
    max_concurrent_transforms = 1
    max_payload = 6
    tags = [{"Key": "k", "Value": "v"}]

    transformer = model.transformer(
        instance_count=1,
        instance_type=instance_type,
        strategy=strategy,
        assemble_with=assemble_with,
        output_path=output_path,
        output_kms_key=kms_key,
        accept=accept,
        env=env,
        max_concurrent_transforms=max_concurrent_transforms,
        max_payload=max_payload,
        tags=tags,
        volume_kms_key=kms_key,
    )

    create_sagemaker_model.assert_called_with(instance_type, tags=tags)

    assert isinstance(transformer, sagemaker.transformer.Transformer)
    assert transformer.strategy == strategy
    assert transformer.assemble_with == assemble_with
    assert transformer.output_path == output_path
    assert transformer.output_kms_key == kms_key
    assert transformer.accept == accept
    assert transformer.max_concurrent_transforms == max_concurrent_transforms
    assert transformer.max_payload == max_payload
    assert transformer.env == env
    assert transformer.tags == tags
    assert transformer.volume_kms_key == kms_key


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
def test_model_create_transformer_network_isolation(sagemaker_session):
    model = Model(
        MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session, enable_network_isolation=True
    )

    transformer = model.transformer(1, "ml.m4.xlarge", env={"should_be": "overwritten"})
    assert transformer.env is None


@patch("sagemaker.model.Model._create_sagemaker_model", Mock())
def test_model_create_transformer_base_name(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session)

    base_name = "foo"
    model._base_name = base_name

    transformer = model.transformer(1, "ml.m4.xlarge")
    assert base_name == transformer.base_transform_job_name


@patch("sagemaker.session.Session")
@patch("sagemaker.local.LocalSession")
def test_transformer_creates_correct_session(local_session, session):
    local_session.return_value.sagemaker_config = {}
    session.return_value.sagemaker_config = {}
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=None)
    transformer = model.transformer(instance_count=1, instance_type="local")
    assert model.sagemaker_session == local_session.return_value
    assert transformer.sagemaker_session == local_session.return_value

    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=None)
    transformer = model.transformer(instance_count=1, instance_type="ml.m5.xlarge")
    assert model.sagemaker_session == session.return_value
    assert transformer.sagemaker_session == session.return_value


def test_delete_model(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, name=MODEL_NAME, sagemaker_session=sagemaker_session)

    model.delete_model()
    sagemaker_session.delete_model.assert_called_with(model.name)


def test_delete_model_no_name(sagemaker_session):
    model = Model(MODEL_IMAGE, MODEL_DATA, sagemaker_session=sagemaker_session)

    with pytest.raises(
        ValueError, match="The SageMaker model must be created first before attempting to delete."
    ):
        model.delete_model()
    sagemaker_session.delete_model.assert_not_called()


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


@patch("sagemaker.utils.repack_model")
def test_script_mode_model_tags_jumpstart_models(repack_model, sagemaker_session):

    jumpstart_source_dir = f"s3://{list(JUMPSTART_BUCKET_NAME_SET)[0]}/source_dirs/source.tar.gz"
    t = Model(
        entry_point=ENTRY_POINT_INFERENCE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        source_dir=jumpstart_source_dir,
        image_uri=IMAGE_URI,
        model_data=MODEL_DATA,
    )
    t.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)

    assert sagemaker_session.create_model.call_args_list[0][1]["tags"] == [
        {
            "Key": JumpStartTag.INFERENCE_SCRIPT_URI.value,
            "Value": jumpstart_source_dir,
        },
    ]
    assert sagemaker_session.endpoint_from_production_variants.call_args_list[0][1]["tags"] == [
        {
            "Key": JumpStartTag.INFERENCE_SCRIPT_URI.value,
            "Value": jumpstart_source_dir,
        },
    ]

    non_jumpstart_source_dir = "s3://blah/blah/blah"
    t = Model(
        entry_point=ENTRY_POINT_INFERENCE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        source_dir=non_jumpstart_source_dir,
        image_uri=IMAGE_URI,
        model_data=MODEL_DATA,
    )
    t.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)

    assert {
        "Key": JumpStartTag.INFERENCE_SCRIPT_URI.value,
        "Value": non_jumpstart_source_dir,
    } not in sagemaker_session.create_model.call_args_list[0][1]["tags"]

    assert {
        "Key": JumpStartTag.INFERENCE_SCRIPT_URI.value,
        "Value": non_jumpstart_source_dir,
    } not in sagemaker_session.create_model.call_args_list[0][1]["tags"]


@patch("sagemaker.utils.repack_model")
@patch("sagemaker.fw_utils.tar_and_upload_dir")
def test_all_framework_models_add_jumpstart_tags(
    repack_model, tar_and_uload_dir, sagemaker_session
):
    framework_model_classes_to_kwargs = {
        PyTorchModel: {"framework_version": "1.5.0", "py_version": "py3"},
        TensorFlowModel: {
            "framework_version": "2.3",
        },
        HuggingFaceModel: {
            "pytorch_version": "1.7.1",
            "py_version": "py36",
            "transformers_version": "4.6.1",
        },
        MXNetModel: {"framework_version": "1.7.0", "py_version": "py3"},
        SKLearnModel: {
            "framework_version": "0.23-1",
        },
        XGBoostModel: {
            "framework_version": "1.3-1",
        },
    }
    jumpstart_model_dir = f"s3://{list(JUMPSTART_BUCKET_NAME_SET)[0]}/model_dirs/model.tar.gz"
    for framework_model_class, kwargs in framework_model_classes_to_kwargs.items():
        framework_model_class(
            entry_point=ENTRY_POINT_INFERENCE,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            model_data=jumpstart_model_dir,
            **kwargs,
        ).deploy(instance_type="ml.m2.xlarge", initial_instance_count=INSTANCE_COUNT)

        assert {
            "Key": JumpStartTag.INFERENCE_MODEL_URI.value,
            "Value": jumpstart_model_dir,
        } in sagemaker_session.create_model.call_args_list[0][1]["tags"]

        assert {
            "Key": JumpStartTag.INFERENCE_MODEL_URI.value,
            "Value": jumpstart_model_dir,
        } in sagemaker_session.endpoint_from_production_variants.call_args_list[0][1]["tags"]

        sagemaker_session.create_model.reset_mock()
        sagemaker_session.endpoint_from_production_variants.reset_mock()


@patch("sagemaker.utils.repack_model")
def test_script_mode_model_uses_jumpstart_base_name(repack_model, sagemaker_session):

    jumpstart_source_dir = f"s3://{list(JUMPSTART_BUCKET_NAME_SET)[0]}/source_dirs/source.tar.gz"
    t = Model(
        entry_point=ENTRY_POINT_INFERENCE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        source_dir=jumpstart_source_dir,
        image_uri=IMAGE_URI,
        model_data=MODEL_DATA,
    )
    t.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)

    assert sagemaker_session.create_model.call_args_list[0][1]["name"].startswith(
        JUMPSTART_RESOURCE_BASE_NAME
    )

    assert sagemaker_session.endpoint_from_production_variants.call_args_list[0].startswith(
        JUMPSTART_RESOURCE_BASE_NAME
    )

    sagemaker_session.create_model.reset_mock()
    sagemaker_session.endpoint_from_production_variants.reset_mock()

    non_jumpstart_source_dir = "s3://blah/blah/blah"
    t = Model(
        entry_point=ENTRY_POINT_INFERENCE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        source_dir=non_jumpstart_source_dir,
        image_uri=IMAGE_URI,
        model_data=MODEL_DATA,
    )
    t.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)

    assert not sagemaker_session.create_model.call_args_list[0][1]["name"].startswith(
        JUMPSTART_RESOURCE_BASE_NAME
    )

    assert not sagemaker_session.endpoint_from_production_variants.call_args_list[0][1][
        "name"
    ].startswith(JUMPSTART_RESOURCE_BASE_NAME)


@patch("sagemaker.utils.repack_model")
@patch("sagemaker.fw_utils.tar_and_upload_dir")
def test_all_framework_models_add_jumpstart_base_name(
    repack_model, tar_and_uload_dir, sagemaker_session
):
    framework_model_classes_to_kwargs = {
        PyTorchModel: {"framework_version": "1.5.0", "py_version": "py3"},
        TensorFlowModel: {
            "framework_version": "2.3",
        },
        HuggingFaceModel: {
            "pytorch_version": "1.7.1",
            "py_version": "py36",
            "transformers_version": "4.6.1",
        },
        MXNetModel: {"framework_version": "1.7.0", "py_version": "py3"},
        SKLearnModel: {
            "framework_version": "0.23-1",
        },
        XGBoostModel: {
            "framework_version": "1.3-1",
        },
    }
    jumpstart_model_dir = f"s3://{list(JUMPSTART_BUCKET_NAME_SET)[0]}/model_dirs/model.tar.gz"
    for framework_model_class, kwargs in framework_model_classes_to_kwargs.items():
        framework_model_class(
            entry_point=ENTRY_POINT_INFERENCE,
            role=ROLE,
            sagemaker_session=sagemaker_session,
            model_data=jumpstart_model_dir,
            **kwargs,
        ).deploy(instance_type="ml.m2.xlarge", initial_instance_count=INSTANCE_COUNT)

        assert sagemaker_session.create_model.call_args_list[0][1]["name"].startswith(
            JUMPSTART_RESOURCE_BASE_NAME
        )

        assert sagemaker_session.endpoint_from_production_variants.call_args_list[0].startswith(
            JUMPSTART_RESOURCE_BASE_NAME
        )

        sagemaker_session.create_model.reset_mock()
        sagemaker_session.endpoint_from_production_variants.reset_mock()


@patch("sagemaker.utils.repack_model")
def test_script_mode_model_uses_proper_sagemaker_submit_dir(repack_model, sagemaker_session):

    source_dir = "s3://blah/blah/blah"
    t = Model(
        entry_point=ENTRY_POINT_INFERENCE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        source_dir=source_dir,
        image_uri=IMAGE_URI,
        model_data=MODEL_DATA,
    )
    t.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)

    assert (
        sagemaker_session.create_model.call_args_list[0][1]["container_defs"]["Environment"][
            "SAGEMAKER_SUBMIT_DIRECTORY"
        ]
        == "/opt/ml/model/code"
    )


@patch("sagemaker.get_model_package_args")
def test_register_calls_model_package_args(get_model_package_args, sagemaker_session):
    """model.register() should pass the ValidationSpecification to get_model_package_args()"""

    source_dir = "s3://blah/blah/blah"
    t = Model(
        entry_point=ENTRY_POINT_INFERENCE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        source_dir=source_dir,
        image_uri=IMAGE_URI,
        model_data=MODEL_DATA,
    )

    t.register(
        SUPPORTED_CONTENT_TYPES,
        SUPPORTED_RESPONSE_MIME_TYPES,
        SUPPORTED_REALTIME_INFERENCE_INSTANCE_TYPES,
        SUPPORTED_BATCH_TRANSFORM_INSTANCE_TYPES,
        marketplace_cert=True,
        description=MODEL_DESCRIPTION,
        model_package_name=MODEL_NAME,
        validation_specification=VALIDATION_SPECIFICATION,
    )

    # check that the kwarg validation_specification was passed to the internal method 'get_model_package_args'
    assert (
        "validation_specification" in get_model_package_args.call_args_list[0][1]
    ), "validation_specification kwarg was not passed to get_model_package_args"

    # check that the kwarg validation_specification is identical to the one passed into the method 'register'
    assert (
        VALIDATION_SPECIFICATION
        == get_model_package_args.call_args_list[0][1]["validation_specification"]
    ), """ValidationSpecification from model.register method is not identical to validation_spec from
         get_model_package_args"""


@patch("sagemaker.utils.repack_model")
def test_model_local_download_dir(repack_model, sagemaker_session):

    source_dir = "s3://blah/blah/blah"
    local_download_dir = "local download dir"

    sagemaker_session.settings.local_download_dir = local_download_dir

    t = Model(
        entry_point=ENTRY_POINT_INFERENCE,
        role=ROLE,
        sagemaker_session=sagemaker_session,
        source_dir=source_dir,
        image_uri=IMAGE_URI,
        model_data=MODEL_DATA,
    )
    t.deploy(instance_type=INSTANCE_TYPE, initial_instance_count=INSTANCE_COUNT)

    assert (
        repack_model.call_args_list[0][1]["sagemaker_session"].settings.local_download_dir
        == local_download_dir
    )
