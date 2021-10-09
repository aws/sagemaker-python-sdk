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

import inspect
import os
import tarfile
from contextlib import contextmanager
from itertools import product

import pytest

from mock import Mock, patch

from sagemaker import fw_utils
from sagemaker.utils import name_from_image

TIMESTAMP = "2017-10-10-14-14-15"


@contextmanager
def cd(path):
    old_dir = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(old_dir)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name="us-west-2")
    session_mock = Mock(
        name="sagemaker_session", boto_session=boto_mock, s3_client=None, s3_resource=None
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value="my-bucket")
    session_mock.expand_role = Mock(name="expand_role", return_value="my-role")
    session_mock.sagemaker_client.describe_training_job = Mock(
        return_value={"ModelArtifacts": {"S3ModelArtifacts": "s3://m/m.tar.gz"}}
    )
    return session_mock


def test_tar_and_upload_dir_s3(sagemaker_session):
    bucket = "mybucket"
    s3_key_prefix = "something/source"
    script = "mnist.py"
    directory = "s3://m"
    result = fw_utils.tar_and_upload_dir(
        sagemaker_session, bucket, s3_key_prefix, script, directory
    )

    assert result == fw_utils.UploadedCode("s3://m", "mnist.py")


def test_tar_and_upload_dir_s3_with_script_dir(sagemaker_session):
    bucket = "mybucket"
    s3_key_prefix = "something/source"
    script = "some/dir/mnist.py"
    directory = "s3://m"
    result = fw_utils.tar_and_upload_dir(
        sagemaker_session, bucket, s3_key_prefix, script, directory
    )

    assert result == fw_utils.UploadedCode("s3://m", "some/dir/mnist.py")


@patch("sagemaker.utils")
def test_tar_and_upload_dir_s3_with_kms(utils, sagemaker_session):
    bucket = "mybucket"
    s3_key_prefix = "something/source"
    script = "mnist.py"
    kms_key = "kms-key"
    result = fw_utils.tar_and_upload_dir(
        sagemaker_session, bucket, s3_key_prefix, script, kms_key=kms_key
    )

    assert result == fw_utils.UploadedCode(
        "s3://{}/{}/sourcedir.tar.gz".format(bucket, s3_key_prefix), script
    )

    extra_args = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": kms_key}
    obj = sagemaker_session.resource("s3").Object("", "")
    obj.upload_file.assert_called_with(utils.create_tar_file(), ExtraArgs=extra_args)


def test_mp_config_partition_exists():
    mp_parameters = {}
    with pytest.raises(ValueError):
        fw_utils.validate_mp_config(mp_parameters)


@pytest.mark.parametrize(
    "pipeline, placement_strategy, optimize, trace_device",
    [
        ("simple", "spread", "speed", "cpu"),
        ("interleaved", "cluster", "memory", "gpu"),
        ("_only_forward", "spread", "speed", "gpu"),
    ],
)
def test_mp_config_string_names(pipeline, placement_strategy, optimize, trace_device):
    mp_parameters = {
        "partitions": 2,
        "pipeline": pipeline,
        "placement_strategy": placement_strategy,
        "optimize": optimize,
        "trace_device": trace_device,
        "active_microbatches": 8,
        "deterministic_server": True,
    }
    fw_utils.validate_mp_config(mp_parameters)


def test_mp_config_auto_partition_arg():
    mp_parameters = {}
    mp_parameters["partitions"] = 2
    mp_parameters["auto_partition"] = False
    with pytest.raises(ValueError):
        fw_utils.validate_mp_config(mp_parameters)

    mp_parameters["default_partition"] = 1
    fw_utils.validate_mp_config(mp_parameters)

    mp_parameters["default_partition"] = 4
    with pytest.raises(ValueError):
        fw_utils.validate_mp_config(mp_parameters)


def test_validate_source_dir_does_not_exits(sagemaker_session):
    script = "mnist.py"
    directory = " !@#$%^&*()path probably in not there.!@#$%^&*()"
    with pytest.raises(ValueError):
        fw_utils.validate_source_dir(script, directory)


def test_validate_source_dir_is_not_directory(sagemaker_session):
    script = "mnist.py"
    directory = inspect.getfile(inspect.currentframe())
    with pytest.raises(ValueError):
        fw_utils.validate_source_dir(script, directory)


def test_validate_source_dir_file_not_in_dir():
    script = " !@#$%^&*() .myscript. !@#$%^&*() "
    directory = "."
    with pytest.raises(ValueError):
        fw_utils.validate_source_dir(script, directory)


def test_tar_and_upload_dir_not_s3(sagemaker_session):
    bucket = "mybucket"
    s3_key_prefix = "something/source"
    script = os.path.basename(__file__)
    directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    result = fw_utils.tar_and_upload_dir(
        sagemaker_session, bucket, s3_key_prefix, script, directory
    )
    assert result == fw_utils.UploadedCode(
        "s3://{}/{}/sourcedir.tar.gz".format(bucket, s3_key_prefix), script
    )


def file_tree(tmpdir, files=None, folders=None):
    files = files or []
    folders = folders or []
    for file in files:
        tmpdir.join(file).ensure(file=True)

    for folder in folders:
        tmpdir.join(folder).ensure(dir=True)

    return str(tmpdir)


def test_tar_and_upload_dir_no_directory(sagemaker_session, tmpdir):
    source_dir = file_tree(tmpdir, ["train.py"])
    entrypoint = os.path.join(source_dir, "train.py")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", entrypoint, None
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/train.py"} == list_source_dir_files(sagemaker_session, tmpdir)


def test_tar_and_upload_dir_no_directory_only_entrypoint(sagemaker_session, tmpdir):
    source_dir = file_tree(tmpdir, ["train.py", "not_me.py"])
    entrypoint = os.path.join(source_dir, "train.py")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", entrypoint, None
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/train.py"} == list_source_dir_files(sagemaker_session, tmpdir)


def test_tar_and_upload_dir_no_directory_bare_filename(sagemaker_session, tmpdir):
    source_dir = file_tree(tmpdir, ["train.py"])
    entrypoint = "train.py"

    with patch("shutil.rmtree"):
        with cd(source_dir):
            result = fw_utils.tar_and_upload_dir(
                sagemaker_session, "bucket", "prefix", entrypoint, None
            )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/train.py"} == list_source_dir_files(sagemaker_session, tmpdir)


def test_tar_and_upload_dir_with_directory(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["src-dir/train.py"])
    source_dir = os.path.join(str(tmpdir), "src-dir")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", "train.py", source_dir
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/train.py"} == list_source_dir_files(sagemaker_session, tmpdir)


def test_tar_and_upload_dir_with_subdirectory(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["src-dir/sub/train.py"])
    source_dir = os.path.join(str(tmpdir), "src-dir")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", "train.py", source_dir
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/sub/train.py"} == list_source_dir_files(sagemaker_session, tmpdir)


def test_tar_and_upload_dir_with_directory_and_files(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["src-dir/train.py", "src-dir/laucher", "src-dir/module/__init__.py"])
    source_dir = os.path.join(str(tmpdir), "src-dir")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", "train.py", source_dir
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="train.py"
    )

    assert {"/laucher", "/module/__init__.py", "/train.py"} == list_source_dir_files(
        sagemaker_session, tmpdir
    )


def test_tar_and_upload_dir_with_directories_and_files(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["src-dir/a/b", "src-dir/a/b2", "src-dir/x/y", "src-dir/x/y2", "src-dir/z"])
    source_dir = os.path.join(str(tmpdir), "src-dir")

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", "a/b", source_dir
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="a/b"
    )

    assert {"/a/b", "/a/b2", "/x/y", "/x/y2", "/z"} == list_source_dir_files(
        sagemaker_session, tmpdir
    )


def test_tar_and_upload_dir_with_many_folders(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["src-dir/a/b", "src-dir/a/b2", "common/x/y", "common/x/y2", "t/y/z"])
    source_dir = os.path.join(str(tmpdir), "src-dir")
    dependencies = [os.path.join(str(tmpdir), "common"), os.path.join(str(tmpdir), "t", "y", "z")]

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session, "bucket", "prefix", "pipeline.py", source_dir, dependencies
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="pipeline.py"
    )

    assert {"/a/b", "/a/b2", "/common/x/y", "/common/x/y2", "/z"} == list_source_dir_files(
        sagemaker_session, tmpdir
    )


def test_test_tar_and_upload_dir_with_subfolders(sagemaker_session, tmpdir):
    file_tree(tmpdir, ["a/b/c", "a/b/c2"])
    root = file_tree(tmpdir, ["x/y/z", "x/y/z2"])

    with patch("shutil.rmtree"):
        result = fw_utils.tar_and_upload_dir(
            sagemaker_session,
            "bucket",
            "prefix",
            "b/c",
            os.path.join(root, "a"),
            [os.path.join(root, "x")],
        )

    assert result == fw_utils.UploadedCode(
        s3_prefix="s3://bucket/prefix/sourcedir.tar.gz", script_name="b/c"
    )

    assert {"/b/c", "/b/c2", "/x/y/z", "/x/y/z2"} == list_source_dir_files(
        sagemaker_session, tmpdir
    )


def list_source_dir_files(sagemaker_session, tmpdir):
    source_dir_tar = sagemaker_session.resource("s3").Object().upload_file.call_args[0][0]

    source_dir_files = list_tar_files("/opt/ml/code/", source_dir_tar, tmpdir)
    return source_dir_files


def list_tar_files(folder, tar_ball, tmpdir):
    startpath = str(tmpdir.ensure(folder, dir=True))

    with tarfile.open(name=tar_ball, mode="r:gz") as t:
        t.extractall(path=startpath)

    def walk():
        for root, dirs, files in os.walk(startpath):
            path = root.replace(startpath, "")
            for f in files:
                yield "%s/%s" % (path, f)

    result = set(walk())
    return result if result else {}


def test_framework_name_from_image_mxnet():
    image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:1.1-gpu-py3"
    assert ("mxnet", "py3", "1.1-gpu-py3", None) == fw_utils.framework_name_from_image(image_uri)


def test_framework_name_from_image_mxnet_in_gov():
    image_uri = "123.dkr.ecr.region-name.c2s.ic.gov/sagemaker-mxnet:1.1-gpu-py3"
    assert ("mxnet", "py3", "1.1-gpu-py3", None) == fw_utils.framework_name_from_image(image_uri)


def test_framework_name_from_image_tf():
    image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:1.6-cpu-py2"
    assert ("tensorflow", "py2", "1.6-cpu-py2", None) == fw_utils.framework_name_from_image(
        image_uri
    )


def test_framework_name_from_image_tf_scriptmode():
    image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-scriptmode:1.12-cpu-py3"
    assert (
        "tensorflow",
        "py3",
        "1.12-cpu-py3",
        "scriptmode",
    ) == fw_utils.framework_name_from_image(image_uri)

    image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.13-cpu-py3"
    assert ("tensorflow", "py3", "1.13-cpu-py3", "training") == fw_utils.framework_name_from_image(
        image_uri
    )


def test_framework_name_from_image_rl():
    image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-rl-mxnet:toolkit1.1-gpu-py3"
    assert ("mxnet", "py3", "toolkit1.1-gpu-py3", None) == fw_utils.framework_name_from_image(
        image_uri
    )


def test_framework_name_from_image_python_versions():
    image_name = "123.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.2-cpu-py37"
    assert ("tensorflow", "py37", "2.2-cpu-py37", "training") == fw_utils.framework_name_from_image(
        image_name
    )

    image_name = "123.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.15.2-cpu-py36"
    expected_result = ("tensorflow", "py36", "1.15.2-cpu-py36", "training")
    assert expected_result == fw_utils.framework_name_from_image(image_name)


def test_legacy_name_from_framework_image():
    image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py3-gpu:2.5.6-gpu-py2"
    framework, py_ver, tag, _ = fw_utils.framework_name_from_image(image_uri)
    assert framework == "mxnet"
    assert py_ver == "py3"
    assert tag == "2.5.6-gpu-py2"


def test_legacy_name_from_wrong_framework():
    framework, py_ver, tag, _ = fw_utils.framework_name_from_image(
        "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py2-gpu:1"
    )
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_legacy_name_from_wrong_python():
    framework, py_ver, tag, _ = fw_utils.framework_name_from_image(
        "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py4-gpu:1"
    )
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_legacy_name_from_wrong_device():
    framework, py_ver, tag, _ = fw_utils.framework_name_from_image(
        "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py4-gpu:1"
    )
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_legacy_name_from_image_any_tag():
    image_uri = "123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-py2-cpu:any-tag"
    framework, py_ver, tag, _ = fw_utils.framework_name_from_image(image_uri)
    assert framework == "tensorflow"
    assert py_ver == "py2"
    assert tag == "any-tag"


def test_framework_version_from_tag():
    tags = (
        "1.5rc-keras-cpu-py2",
        "1.5rc-keras-gpu-py2",
        "1.5rc-keras-cpu-py3",
        "1.5rc-keras-gpu-py36",
        "1.5rc-keras-gpu-py37",
    )

    for tag in tags:
        version = fw_utils.framework_version_from_tag(tag)
        assert "1.5rc-keras" == version


def test_framework_version_from_tag_other():
    version = fw_utils.framework_version_from_tag("weird-tag-py2")
    assert version is None


def test_model_code_key_prefix_with_all_values_present():
    key_prefix = fw_utils.model_code_key_prefix("prefix", "model_name", "image_uri")
    assert key_prefix == "prefix/model_name"


def test_model_code_key_prefix_with_no_prefix_and_all_other_values_present():
    key_prefix = fw_utils.model_code_key_prefix(None, "model_name", "image_uri")
    assert key_prefix == "model_name"


@patch("time.strftime", return_value=TIMESTAMP)
def test_model_code_key_prefix_with_only_image_present(time):
    key_prefix = fw_utils.model_code_key_prefix(None, None, "image_uri")
    assert key_prefix == name_from_image("image_uri")


@patch("time.strftime", return_value=TIMESTAMP)
def test_model_code_key_prefix_and_image_present(time):
    key_prefix = fw_utils.model_code_key_prefix("prefix", None, "image_uri")
    assert key_prefix == "prefix/" + name_from_image("image_uri")


def test_model_code_key_prefix_with_prefix_present_and_others_none_fail():
    with pytest.raises(TypeError) as error:
        fw_utils.model_code_key_prefix("prefix", None, None)
    assert "expected string" in str(error.value)


def test_model_code_key_prefix_with_all_none_fail():
    with pytest.raises(TypeError) as error:
        fw_utils.model_code_key_prefix(None, None, None)
    assert "expected string" in str(error.value)


def test_region_supports_debugger_feature_returns_true_for_supported_regions():
    assert fw_utils._region_supports_debugger("us-west-2") is True
    assert fw_utils._region_supports_debugger("us-east-2") is True


def test_region_supports_debugger_feature_returns_false_for_unsupported_regions():
    assert fw_utils._region_supports_debugger("us-iso-east-1") is False


def test_warn_if_parameter_server_with_multi_gpu(caplog):
    instance_type = "ml.p2.8xlarge"
    distribution = {"parameter_server": {"enabled": True}}

    fw_utils.warn_if_parameter_server_with_multi_gpu(
        training_instance_type=instance_type, distribution=distribution
    )
    assert fw_utils.PARAMETER_SERVER_MULTI_GPU_WARNING in caplog.text


def test_warn_if_parameter_server_with_local_multi_gpu(caplog):
    instance_type = "local_gpu"
    distribution = {"parameter_server": {"enabled": True}}

    fw_utils.warn_if_parameter_server_with_multi_gpu(
        training_instance_type=instance_type, distribution=distribution
    )
    assert fw_utils.PARAMETER_SERVER_MULTI_GPU_WARNING in caplog.text


def test_validate_version_or_image_args_not_raises():
    good_args = [("1.0", "py3", None), (None, "py3", "my:uri"), ("1.0", None, "my:uri")]
    for framework_version, py_version, image_uri in good_args:
        fw_utils.validate_version_or_image_args(framework_version, py_version, image_uri)


def test_validate_version_or_image_args_raises():
    bad_args = [(None, None, None), (None, "py3", None), ("1.0", None, None)]
    for framework_version, py_version, image_uri in bad_args:
        with pytest.raises(ValueError):
            fw_utils.validate_version_or_image_args(framework_version, py_version, image_uri)


def test_validate_smdistributed_not_raises():
    smdataparallel_enabled = {"smdistributed": {"dataparallel": {"enabled": True}}}
    smdataparallel_enabled_custom_mpi = {
        "smdistributed": {"dataparallel": {"enabled": True, "custom_mpi_options": "--verbose"}}
    }
    smdataparallel_disabled = {"smdistributed": {"dataparallel": {"enabled": False}}}
    instance_types = list(fw_utils.SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES)

    good_args = [
        (smdataparallel_enabled, "custom-container"),
        (smdataparallel_enabled_custom_mpi, "custom-container"),
        (smdataparallel_disabled, "custom-container"),
    ]
    frameworks = ["tensorflow", "pytorch"]

    for framework, instance_type in product(frameworks, instance_types):
        for distribution, image_uri in good_args:
            fw_utils.validate_smdistributed(
                instance_type=instance_type,
                framework_name=framework,
                framework_version=None,
                py_version=None,
                distribution=distribution,
                image_uri=image_uri,
            )


def test_validate_smdistributed_raises():
    bad_args = [
        {"smdistributed": "dummy"},
        {"smdistributed": {"dummy"}},
        {"smdistributed": {"dummy": "val"}},
        {"smdistributed": {"dummy": {"enabled": True}}},
    ]
    instance_types = list(fw_utils.SM_DATAPARALLEL_SUPPORTED_INSTANCE_TYPES)
    frameworks = ["tensorflow", "pytorch"]
    for framework, distribution, instance_type in product(frameworks, bad_args, instance_types):
        with pytest.raises(ValueError):
            fw_utils.validate_smdistributed(
                instance_type=instance_type,
                framework_name=framework,
                framework_version=None,
                py_version=None,
                distribution=distribution,
                image_uri="custom-container",
            )


def test_validate_smdataparallel_args_raises():
    # TODO: add validation for dataparallel in mxnet
    smdataparallel_enabled = {"smdistributed": {"dataparallel": {"enabled": True}}}

    # Cases {PT|TF2}
    # 1. None instance type
    # 2. incorrect instance type
    # 3. incorrect python version
    # 4. incorrect framework version

    bad_args = [
        (None, "tensorflow", "2.3.1", "py3", smdataparallel_enabled),
        ("ml.p3.2xlarge", "tensorflow", "2.3.1", "py3", smdataparallel_enabled),
        ("ml.p3dn.24xlarge", "tensorflow", "2.3.1", "py2", smdataparallel_enabled),
        ("ml.p3.16xlarge", "tensorflow", "1.3.1", "py3", smdataparallel_enabled),
        (None, "pytorch", "1.6.0", "py3", smdataparallel_enabled),
        ("ml.p3.2xlarge", "pytorch", "1.6.0", "py3", smdataparallel_enabled),
        ("ml.p3dn.24xlarge", "pytorch", "1.6.0", "py2", smdataparallel_enabled),
        ("ml.p3.16xlarge", "pytorch", "1.5.0", "py3", smdataparallel_enabled),
    ]
    for instance_type, framework_name, framework_version, py_version, distribution in bad_args:
        with pytest.raises(ValueError):
            fw_utils._validate_smdataparallel_args(
                instance_type, framework_name, framework_version, py_version, distribution
            )


def test_validate_smdataparallel_args_not_raises():
    smdataparallel_enabled = {"smdistributed": {"dataparallel": {"enabled": True}}}
    smdataparallel_enabled_custom_mpi = {
        "smdistributed": {"dataparallel": {"enabled": True, "custom_mpi_options": "--verbose"}}
    }
    smdataparallel_disabled = {"smdistributed": {"dataparallel": {"enabled": False}}}

    # Cases {PT|TF2}
    # 1. SM Distributed dataparallel disabled
    # 2. SM Distributed dataparallel enabled with supported args

    good_args = [
        (None, None, None, None, smdataparallel_disabled),
        ("ml.p3.16xlarge", "tensorflow", "2.3.1", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "tensorflow", "2.3.2", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "tensorflow", "2.3", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "tensorflow", "2.4.1", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "tensorflow", "2.4", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "pytorch", "1.6.0", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "pytorch", "1.6", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "pytorch", "1.7.1", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "pytorch", "1.7", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "pytorch", "1.8.0", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "pytorch", "1.8.1", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "pytorch", "1.8", "py3", smdataparallel_enabled),
        ("ml.p3.16xlarge", "tensorflow", "2.4.1", "py3", smdataparallel_enabled_custom_mpi),
        ("ml.p3.16xlarge", "pytorch", "1.8.0", "py3", smdataparallel_enabled_custom_mpi),
        ("ml.p3.16xlarge", "pytorch", "1.9.0", "py3", smdataparallel_enabled_custom_mpi),
    ]
    for instance_type, framework_name, framework_version, py_version, distribution in good_args:
        fw_utils._validate_smdataparallel_args(
            instance_type, framework_name, framework_version, py_version, distribution
        )
