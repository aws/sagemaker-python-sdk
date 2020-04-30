# -*- coding: utf-8 -*-

# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import shutil
import tarfile
from datetime import datetime
import os
import re
import time

from boto3 import exceptions
import botocore
import pytest
from mock import call, patch, Mock, MagicMock

import sagemaker

BUCKET_WITHOUT_WRITING_PERMISSION = "s3://bucket-without-writing-permission"

NAME = "base_name"
BUCKET_NAME = "some_bucket"


def test_get_config_value():

    config = {"local": {"region_name": "us-west-2", "port": "123"}, "other": {"key": 1}}

    assert sagemaker.utils.get_config_value("local.region_name", config) == "us-west-2"
    assert sagemaker.utils.get_config_value("local", config) == {
        "region_name": "us-west-2",
        "port": "123",
    }

    assert sagemaker.utils.get_config_value("does_not.exist", config) is None
    assert sagemaker.utils.get_config_value("other.key", None) is None


def test_get_short_version():
    assert sagemaker.utils.get_short_version("1.13.1") == "1.13"
    assert sagemaker.utils.get_short_version("1.13") == "1.13"


def test_deferred_error():
    de = sagemaker.utils.DeferredError(ImportError("pretend the import failed"))
    with pytest.raises(ImportError) as _:  # noqa: F841
        de.something()


def test_bad_import():
    try:
        import pandas_is_not_installed as pd
    except ImportError as e:
        pd = sagemaker.utils.DeferredError(e)
    assert pd is not None
    with pytest.raises(ImportError) as _:  # noqa: F841
        pd.DataFrame()


@patch("sagemaker.utils.sagemaker_timestamp")
def test_name_from_base(sagemaker_timestamp):
    sagemaker.utils.name_from_base(NAME, short=False)
    assert sagemaker_timestamp.called_once


@patch("sagemaker.utils.sagemaker_short_timestamp")
def test_name_from_base_short(sagemaker_short_timestamp):
    sagemaker.utils.name_from_base(NAME, short=True)
    assert sagemaker_short_timestamp.called_once


def test_unique_name_from_base():
    assert re.match(r"base-\d{10}-[a-f0-9]{4}", sagemaker.utils.unique_name_from_base("base"))


def test_unique_name_from_base_truncated():
    assert re.match(
        r"real-\d{10}-[a-f0-9]{4}",
        sagemaker.utils.unique_name_from_base("really-long-name", max_length=20),
    )


def test_to_str_with_native_string():
    value = "some string"
    assert sagemaker.utils.to_str(value) == value


def test_to_str_with_unicode_string():
    value = u"åñøthér strîng"
    assert sagemaker.utils.to_str(value) == value


def test_name_from_tuning_arn():
    arn = "arn:aws:sagemaker:us-west-2:968277160000:hyper-parameter-tuning-job/resnet-sgd-tuningjob-11-07-34-11"
    name = sagemaker.utils.extract_name_from_job_arn(arn)
    assert name == "resnet-sgd-tuningjob-11-07-34-11"


def test_name_from_training_arn():
    arn = "arn:aws:sagemaker:us-west-2:968277160000:training-job/resnet-sgd-tuningjob-11-22-38-46-002-2927640b"
    name = sagemaker.utils.extract_name_from_job_arn(arn)
    assert name == "resnet-sgd-tuningjob-11-22-38-46-002-2927640b"


MESSAGE = "message"
STATUS = "status"
TRAINING_JOB_DESCRIPTION_1 = {
    "SecondaryStatusTransitions": [{"StatusMessage": MESSAGE, "Status": STATUS}]
}
TRAINING_JOB_DESCRIPTION_2 = {
    "SecondaryStatusTransitions": [{"StatusMessage": "different message", "Status": STATUS}]
}

TRAINING_JOB_DESCRIPTION_EMPTY = {"SecondaryStatusTransitions": []}


def test_secondary_training_status_changed_true():
    changed = sagemaker.utils.secondary_training_status_changed(
        TRAINING_JOB_DESCRIPTION_1, TRAINING_JOB_DESCRIPTION_2
    )
    assert changed is True


def test_secondary_training_status_changed_false():
    changed = sagemaker.utils.secondary_training_status_changed(
        TRAINING_JOB_DESCRIPTION_1, TRAINING_JOB_DESCRIPTION_1
    )
    assert changed is False


def test_secondary_training_status_changed_prev_missing():
    changed = sagemaker.utils.secondary_training_status_changed(TRAINING_JOB_DESCRIPTION_1, {})
    assert changed is True


def test_secondary_training_status_changed_prev_none():
    changed = sagemaker.utils.secondary_training_status_changed(TRAINING_JOB_DESCRIPTION_1, None)
    assert changed is True


def test_secondary_training_status_changed_current_missing():
    changed = sagemaker.utils.secondary_training_status_changed({}, TRAINING_JOB_DESCRIPTION_1)
    assert changed is False


def test_secondary_training_status_changed_empty():
    changed = sagemaker.utils.secondary_training_status_changed(
        TRAINING_JOB_DESCRIPTION_EMPTY, TRAINING_JOB_DESCRIPTION_1
    )
    assert changed is False


def test_secondary_training_status_message_status_changed():
    now = datetime.now()
    TRAINING_JOB_DESCRIPTION_1["LastModifiedTime"] = now
    expected = "{} {} - {}".format(
        datetime.utcfromtimestamp(time.mktime(now.timetuple())).strftime("%Y-%m-%d %H:%M:%S"),
        STATUS,
        MESSAGE,
    )
    assert (
        sagemaker.utils.secondary_training_status_message(
            TRAINING_JOB_DESCRIPTION_1, TRAINING_JOB_DESCRIPTION_EMPTY
        )
        == expected
    )


def test_secondary_training_status_message_status_not_changed():
    now = datetime.now()
    TRAINING_JOB_DESCRIPTION_1["LastModifiedTime"] = now
    expected = "{} {} - {}".format(
        datetime.utcfromtimestamp(time.mktime(now.timetuple())).strftime("%Y-%m-%d %H:%M:%S"),
        STATUS,
        MESSAGE,
    )
    assert (
        sagemaker.utils.secondary_training_status_message(
            TRAINING_JOB_DESCRIPTION_1, TRAINING_JOB_DESCRIPTION_2
        )
        == expected
    )


def test_secondary_training_status_message_prev_missing():
    now = datetime.now()
    TRAINING_JOB_DESCRIPTION_1["LastModifiedTime"] = now
    expected = "{} {} - {}".format(
        datetime.utcfromtimestamp(time.mktime(now.timetuple())).strftime("%Y-%m-%d %H:%M:%S"),
        STATUS,
        MESSAGE,
    )
    assert (
        sagemaker.utils.secondary_training_status_message(TRAINING_JOB_DESCRIPTION_1, {})
        == expected
    )


def test_generate_tensorboard_url_valid_domain_and_bucket_paths():
    domain = "jupyterlab.us-east-2.abcdefgh.com"
    bucket_paths = ["bucket1/path1", "bucket2/path2"]
    expected = "https://{}/tensorboard/default?{}".format(
        domain, "s3urls=s3%3A%2F%2Fbucket1%2Fpath1,s3%3A%2F%2Fbucket2%2Fpath2"
    )
    assert sagemaker.utils.generate_tensorboard_url(domain, bucket_paths) == expected


def test_generate_tensorboard_url_valid_domain_and_bucket_paths_with_s3_prefixes():
    domain = "jupyterlab.us-east-2.abcdefgh.com"
    bucket_paths = ["s3://bucket1/path1", "s3://bucket2/path2"]
    expected = "https://{}/tensorboard/default?{}".format(
        domain, "s3urls=s3%3A%2F%2Fbucket1%2Fpath1,s3%3A%2F%2Fbucket2%2Fpath2"
    )
    assert sagemaker.utils.generate_tensorboard_url(domain, bucket_paths) == expected


def test_generate_tensorboard_url_valid_domain_and_bucket_paths_single():
    domain = "jupyterlab.us-east-2.abcdefgh.com"
    bucket_paths = ["bucket1/path1"]
    expected = "https://{}/tensorboard/default?{}".format(
        domain, "s3urls=s3%3A%2F%2Fbucket1%2Fpath1"
    )
    assert sagemaker.utils.generate_tensorboard_url(domain, bucket_paths) == expected


def test_generate_tensorboard_url_valid_domain_and_bucket_paths_string():
    domain = "jupyterlab.us-east-2.abcdefgh.com"
    bucket_paths = "bucket1/path1"
    expected = "https://{}/tensorboard/default?{}".format(
        domain, "s3urls=s3%3A%2F%2Fbucket1%2Fpath1"
    )
    assert sagemaker.utils.generate_tensorboard_url(domain, bucket_paths) == expected


def test_generate_tensorboard_url_valid_domain_with_http_prefix_and_bucket_paths():
    domain = "http://jupyterlab.us-east-2.abcdefgh.com"
    bucket_paths = ["bucket1/path1"]
    expected = "https://{}/tensorboard/default?{}".format(
        "jupyterlab.us-east-2.abcdefgh.com", "s3urls=s3%3A%2F%2Fbucket1%2Fpath1"
    )
    assert sagemaker.utils.generate_tensorboard_url(domain, bucket_paths) == expected


def test_generate_tensorboard_url_valid_domain_with_https_prefix_and_bucket_paths():
    domain = "https://jupyterlab.us-east-2.abcdefgh.com"
    bucket_paths = ["bucket1/path1"]
    expected = "https://{}/tensorboard/default?{}".format(
        "jupyterlab.us-east-2.abcdefgh.com", "s3urls=s3%3A%2F%2Fbucket1%2Fpath1"
    )
    assert sagemaker.utils.generate_tensorboard_url(domain, bucket_paths) == expected


def test_generate_tensorboard_url_bucket_path_neither_string_nor_list():
    domain = "jupyterlab.us-east-2.abcdefgh.com"
    bucket_paths = None
    try:
        sagemaker.utils.generate_tensorboard_url(domain, bucket_paths)
    except AttributeError as error:
        assert str(error) == "bucket paths should be a list or a string"


def test_generate_tensorboard_url_empty_domain():
    domain = ""
    bucket_paths = ["bucket1/path1"]
    try:
        sagemaker.utils.generate_tensorboard_url(domain, bucket_paths)
    except AttributeError as error:
        assert str(error) == "domain parameter should not be empty"


def test_generate_tensorboard_url_empty_bucket_paths():
    domain = "jupyterlab.us-east-2.abcdefgh.com"
    bucket_paths = []
    try:
        sagemaker.utils.generate_tensorboard_url(domain, bucket_paths)
    except AttributeError as error:
        assert str(error) == "bucket_paths parameter should not be empty list"


def test_generate_tensorboard_url_bucket_paths_with_empty_string():
    domain = "jupyterlab.us-east-2.abcdefgh.com"
    bucket_paths = [""]
    try:
        sagemaker.utils.generate_tensorboard_url(domain, bucket_paths)
    except AttributeError as error:
        assert str(error) == "bucket_paths element should not be empty"


def test_generate_tensorboard_url_domain_non_string():
    domain = None
    bucket_paths = ["bucket1/path1"]
    try:
        sagemaker.utils.generate_tensorboard_url(domain, bucket_paths)
    except AttributeError as error:
        assert str(error) == "domain parameter should be string"


@patch("os.makedirs")
def test_download_folder(makedirs):
    boto_mock = Mock(name="boto_session")
    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())
    s3_mock = boto_mock.resource("s3")

    obj_mock = Mock()
    s3_mock.Object.return_value = obj_mock

    def obj_mock_download(path):
        # Mock the S3 object to raise an error when the input to download_file
        # is a "folder"
        if path in ("/tmp/", os.path.join("/tmp", "prefix")):
            raise botocore.exceptions.ClientError(
                error_response={"Error": {"Code": "404", "Message": "Not Found"}},
                operation_name="HeadObject",
            )
        else:
            return Mock()

    obj_mock.download_file.side_effect = obj_mock_download

    train_data = Mock()
    validation_data = Mock()

    train_data.bucket_name.return_value = BUCKET_NAME
    train_data.key = "prefix/train/train_data.csv"
    validation_data.bucket_name.return_value = BUCKET_NAME
    validation_data.key = "prefix/train/validation_data.csv"

    s3_files = [train_data, validation_data]
    s3_mock.Bucket(BUCKET_NAME).objects.filter.return_value = s3_files

    # all the S3 mocks are set, the test itself begins now.
    sagemaker.utils.download_folder(BUCKET_NAME, "/prefix", "/tmp", session)

    obj_mock.download_file.assert_called()
    calls = [
        call(os.path.join("/tmp", "train", "train_data.csv")),
        call(os.path.join("/tmp", "train", "validation_data.csv")),
    ]
    obj_mock.download_file.assert_has_calls(calls)
    assert s3_mock.Object.call_count == 3

    s3_mock.reset_mock()
    obj_mock.reset_mock()

    # Test with a trailing slash for the prefix.
    sagemaker.utils.download_folder(BUCKET_NAME, "/prefix/", "/tmp", session)
    obj_mock.download_file.assert_called()
    obj_mock.download_file.assert_has_calls(calls)
    assert s3_mock.Object.call_count == 2


@patch("os.makedirs")
def test_download_folder_points_to_single_file(makedirs):
    boto_mock = Mock(name="boto_session")
    boto_mock.client("sts").get_caller_identity.return_value = {"Account": "123"}

    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())

    train_data = Mock()

    train_data.bucket_name.return_value = BUCKET_NAME
    train_data.key = "prefix/train/train_data.csv"

    s3_files = [train_data]
    boto_mock.resource("s3").Bucket(BUCKET_NAME).objects.filter.return_value = s3_files

    obj_mock = Mock()
    boto_mock.resource("s3").Object.return_value = obj_mock

    # all the S3 mocks are set, the test itself begins now.
    sagemaker.utils.download_folder(BUCKET_NAME, "/prefix/train/train_data.csv", "/tmp", session)

    obj_mock.download_file.assert_called()
    calls = [call(os.path.join("/tmp", "train_data.csv"))]
    obj_mock.download_file.assert_has_calls(calls)
    boto_mock.resource("s3").Bucket(BUCKET_NAME).objects.filter.assert_not_called()
    obj_mock.reset_mock()


def test_download_file():
    boto_mock = Mock(name="boto_session")
    boto_mock.client("sts").get_caller_identity.return_value = {"Account": "123"}
    bucket_mock = Mock()
    boto_mock.resource("s3").Bucket.return_value = bucket_mock
    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())

    sagemaker.utils.download_file(
        BUCKET_NAME, "/prefix/path/file.tar.gz", "/tmp/file.tar.gz", session
    )

    bucket_mock.download_file.assert_called_with("prefix/path/file.tar.gz", "/tmp/file.tar.gz")


@patch("tarfile.open")
def test_create_tar_file_with_provided_path(open):
    files = mock_tarfile(open)

    file_list = ["/tmp/a", "/tmp/b"]

    path = sagemaker.utils.create_tar_file(file_list, target="/my/custom/path.tar.gz")
    assert path == "/my/custom/path.tar.gz"
    assert files == [["/tmp/a", "a"], ["/tmp/b", "b"]]


def mock_tarfile(open):
    open.return_value = open
    files = []

    def add_files(filename, arcname):
        files.append([filename, arcname])

    open.__enter__ = Mock()
    open.__enter__().add = add_files
    open.__exit__ = Mock(return_value=None)
    return files


@patch("tarfile.open")
@patch("tempfile.mkstemp", Mock(return_value=(None, "/auto/generated/path")))
def test_create_tar_file_with_auto_generated_path(open):
    files = mock_tarfile(open)

    path = sagemaker.utils.create_tar_file(["/tmp/a", "/tmp/b"])
    assert path == "/auto/generated/path"
    assert files == [["/tmp/a", "a"], ["/tmp/b", "b"]]


def create_file_tree(root, tree):
    for file in tree:
        try:
            os.makedirs(os.path.join(root, os.path.dirname(file)))
        except:  # noqa: E722 Using bare except because p2/3 incompatibility issues.
            pass
        with open(os.path.join(root, file), "a") as f:
            f.write(file)


@pytest.fixture()
def tmp(tmpdir):
    yield str(tmpdir)


def test_repack_model_without_source_dir(tmp, fake_s3):

    create_file_tree(
        tmp,
        [
            "model-dir/model",
            "dependencies/a",
            "dependencies/some/dir/b",
            "aa",
            "bb",
            "source-dir/inference.py",
            "source-dir/this-file-should-not-be-included.py",
        ],
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")

    sagemaker.utils.repack_model(
        inference_script=os.path.join(tmp, "source-dir/inference.py"),
        source_directory=None,
        dependencies=[
            os.path.join(tmp, "dependencies/a"),
            os.path.join(tmp, "dependencies/some/dir"),
            os.path.join(tmp, "aa"),
            os.path.join(tmp, "bb"),
        ],
        model_uri="s3://fake/location",
        repacked_model_uri="s3://destination-bucket/model.tar.gz",
        sagemaker_session=fake_s3.sagemaker_session,
    )

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {
        "/model",
        "/code/lib/a",
        "/code/lib/aa",
        "/code/lib/bb",
        "/code/lib/dir/b",
        "/code/inference.py",
    }


def test_repack_model_with_entry_point_without_path_without_source_dir(tmp, fake_s3):

    create_file_tree(
        tmp,
        [
            "model-dir/model",
            "source-dir/inference.py",
            "source-dir/this-file-should-not-be-included.py",
        ],
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")

    cwd = os.getcwd()
    try:
        os.chdir(os.path.join(tmp, "source-dir"))

        sagemaker.utils.repack_model(
            "inference.py",
            None,
            None,
            "s3://fake/location",
            "s3://destination-bucket/model.tar.gz",
            fake_s3.sagemaker_session,
        )
    finally:
        os.chdir(cwd)

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {"/code/inference.py", "/model"}


def test_repack_model_from_s3_to_s3(tmp, fake_s3):

    create_file_tree(
        tmp,
        [
            "model-dir/model",
            "source-dir/inference.py",
            "source-dir/this-file-should-be-included.py",
        ],
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")

    sagemaker.utils.repack_model(
        "inference.py",
        os.path.join(tmp, "source-dir"),
        None,
        "s3://fake/location",
        "s3://destination-bucket/model.tar.gz",
        fake_s3.sagemaker_session,
    )

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {
        "/code/this-file-should-be-included.py",
        "/code/inference.py",
        "/model",
    }


def test_repack_model_from_file_to_file(tmp):
    create_file_tree(tmp, ["model", "dependencies/a", "source-dir/inference.py"])

    model_tar_path = os.path.join(tmp, "model.tar.gz")
    sagemaker.utils.create_tar_file([os.path.join(tmp, "model")], model_tar_path)

    sagemaker_session = MagicMock()

    file_mode_path = "file://%s" % model_tar_path
    destination_path = "file://%s" % os.path.join(tmp, "repacked-model.tar.gz")

    sagemaker.utils.repack_model(
        "inference.py",
        os.path.join(tmp, "source-dir"),
        [os.path.join(tmp, "dependencies/a")],
        file_mode_path,
        destination_path,
        sagemaker_session,
    )

    assert list_tar_files(destination_path, tmp) == {"/code/lib/a", "/code/inference.py", "/model"}


def test_repack_model_with_inference_code_should_replace_the_code(tmp, fake_s3):
    create_file_tree(
        tmp, ["model-dir/model", "source-dir/new-inference.py", "model-dir/code/old-inference.py"]
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")

    sagemaker.utils.repack_model(
        "inference.py",
        os.path.join(tmp, "source-dir"),
        None,
        "s3://fake/location",
        "s3://destination-bucket/repacked-model",
        fake_s3.sagemaker_session,
    )

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {"/code/new-inference.py", "/model"}


def test_repack_model_from_file_to_folder(tmp):
    create_file_tree(tmp, ["model", "source-dir/inference.py"])

    model_tar_path = os.path.join(tmp, "model.tar.gz")
    sagemaker.utils.create_tar_file([os.path.join(tmp, "model")], model_tar_path)

    file_mode_path = "file://%s" % model_tar_path

    sagemaker.utils.repack_model(
        "inference.py",
        os.path.join(tmp, "source-dir"),
        [],
        file_mode_path,
        "file://%s/repacked-model.tar.gz" % tmp,
        MagicMock(),
    )

    assert list_tar_files("file://%s/repacked-model.tar.gz" % tmp, tmp) == {
        "/code/inference.py",
        "/model",
    }


def test_repack_model_with_inference_code_and_requirements(tmp, fake_s3):
    create_file_tree(
        tmp,
        [
            "new-inference.py",
            "model-dir/model",
            "model-dir/code/old-inference.py",
            "model-dir/code/requirements.txt",
        ],
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")

    sagemaker.utils.repack_model(
        os.path.join(tmp, "new-inference.py"),
        None,
        None,
        "s3://fake/location",
        "s3://destination-bucket/repacked-model",
        fake_s3.sagemaker_session,
    )

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {
        "/code/requirements.txt",
        "/code/new-inference.py",
        "/code/old-inference.py",
        "/model",
    }


def test_repack_model_with_same_inference_file_name(tmp, fake_s3):
    create_file_tree(
        tmp,
        [
            "inference.py",
            "model-dir/model",
            "model-dir/code/inference.py",
            "model-dir/code/requirements.txt",
        ],
    )

    fake_s3.tar_and_upload("model-dir", "s3://fake/location")

    sagemaker.utils.repack_model(
        os.path.join(tmp, "inference.py"),
        None,
        None,
        "s3://fake/location",
        "s3://destination-bucket/repacked-model",
        fake_s3.sagemaker_session,
    )

    assert list_tar_files(fake_s3.fake_upload_path, tmp) == {
        "/code/requirements.txt",
        "/code/inference.py",
        "/model",
    }


class FakeS3(object):
    def __init__(self, tmp):
        self.tmp = tmp
        self.sagemaker_session = MagicMock()
        self.location_map = {}
        self.current_bucket = None

        self.sagemaker_session.boto_session.resource().Bucket().download_file.side_effect = (
            self.download_file
        )
        self.sagemaker_session.boto_session.resource().Bucket.side_effect = self.bucket
        self.fake_upload_path = self.mock_s3_upload()

    def bucket(self, name):
        self.current_bucket = name
        return self

    def download_file(self, path, target):
        key = "%s/%s" % (self.current_bucket, path)
        shutil.copy2(self.location_map[key], target)

    def tar_and_upload(self, path, fake_location):
        tar_location = os.path.join(self.tmp, "model-%s.tar.gz" % time.time())
        with tarfile.open(tar_location, mode="w:gz") as t:
            t.add(os.path.join(self.tmp, path), arcname=os.path.sep)

        self.location_map[fake_location.replace("s3://", "")] = tar_location
        return tar_location

    def mock_s3_upload(self):
        dst = os.path.join(self.tmp, "dst")

        class MockS3Object(object):
            def __init__(self, bucket, key):
                self.bucket = bucket
                self.key = key

            def upload_file(self, target, **kwargs):
                if self.bucket in BUCKET_WITHOUT_WRITING_PERMISSION:
                    raise exceptions.S3UploadFailedError()
                shutil.copy2(target, dst)

        self.sagemaker_session.boto_session.resource().Object = MockS3Object
        return dst


@pytest.fixture()
def fake_s3(tmp):
    return FakeS3(tmp)


def list_tar_files(tar_ball, tmp):
    tar_ball = tar_ball.replace("file://", "")
    startpath = os.path.join(tmp, "startpath")
    os.mkdir(startpath)

    with tarfile.open(name=tar_ball, mode="r:gz") as t:
        t.extractall(path=startpath)

    def walk():
        for root, dirs, files in os.walk(startpath):
            path = root.replace(startpath, "")
            for f in files:
                yield "%s/%s" % (path, f)

    result = set(walk())
    return result if result else {}


def test_get_ecr_image_uri_prefix():
    ecr_prefix = sagemaker.utils.get_ecr_image_uri_prefix("123456789012", "us-west-2")
    assert ecr_prefix == "123456789012.dkr.ecr.us-west-2.amazonaws.com"

    ecr_prefix = sagemaker.utils.get_ecr_image_uri_prefix("123456789012", "us-iso-east-1")
    assert ecr_prefix == "123456789012.dkr.ecr.us-iso-east-1.c2s.ic.gov"


def test_sts_regional_endpoint():
    endpoint = sagemaker.utils.sts_regional_endpoint("us-west-2")
    assert endpoint == "https://sts.us-west-2.amazonaws.com"
    assert botocore.utils.is_valid_endpoint_url(endpoint)

    endpoint = sagemaker.utils.sts_regional_endpoint("us-iso-east-1")
    assert endpoint == "https://sts.us-iso-east-1.c2s.ic.gov"
    assert botocore.utils.is_valid_endpoint_url(endpoint)


def test_partition_by_region():
    assert sagemaker.utils._aws_partition("us-west-2") == "aws"
    assert sagemaker.utils._aws_partition("cn-north-1") == "aws-cn"
    assert sagemaker.utils._aws_partition("us-gov-east-1") == "aws-us-gov"
    assert sagemaker.utils._aws_partition("us-iso-east-1") == "aws-iso"
    assert sagemaker.utils._aws_partition("us-isob-east-1") == "aws-iso-b"
