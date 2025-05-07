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

import sys

import pytest
from mock import patch, Mock

import sagemaker.amazon
import sagemaker.utils.local.data


@patch("sagemaker.local.data.LocalFileDataSource")
def test_get_data_source_instance_with_file(LocalFileDataSource, sagemaker_local_session):
    # file
    data_source = sagemaker.utils.local.data.get_data_source_instance(
        "file:///my/file", sagemaker_local_session
    )
    LocalFileDataSource.assert_called_with("/my/file")
    assert data_source is not None

    data_source = sagemaker.utils.local.data.get_data_source_instance(
        "file://relative/path", sagemaker_local_session
    )
    LocalFileDataSource.assert_called_with("relative/path")
    assert data_source is not None


@patch("sagemaker.local.data.S3DataSource")
def test_get_data_source_instance_with_s3(S3DataSource, sagemaker_local_session):
    data_source = sagemaker.utils.local.data.get_data_source_instance(
        "s3://bucket/path", sagemaker_local_session
    )
    S3DataSource.assert_called_with("bucket", "/path", sagemaker_local_session)
    assert data_source is not None


@patch("os.path.exists", Mock(return_value=True))
@patch("os.path.abspath", lambda x: x)
@patch("os.path.isdir", lambda x: x[-1] == "/")
@patch("os.path.isfile", lambda x: x[-1] != "/")
@patch("os.listdir")
def test_file_data_source_get_file_list_with_folder(listdir):
    data_source = sagemaker.utils.local.data.LocalFileDataSource("/some/path/")
    listdir.return_value = ["/some/path/a", "/some/path/b", "/some/path/c/", "/some/path/c/a"]
    expected = ["/some/path/a", "/some/path/b", "/some/path/c/a"]
    result = data_source.get_file_list()
    assert result == expected


@patch("os.path.exists", Mock(return_value=True))
@patch("os.path.abspath", lambda x: x)
@patch("os.path.isdir", lambda x: x[-1] == "/")
@patch("os.path.isfile", lambda x: x[-1] != "/")
def test_file_data_source_get_file_list_with_single_file():
    data_source = sagemaker.utils.local.data.LocalFileDataSource("/some/batch/file.csv")
    assert data_source.get_file_list() == ["/some/batch/file.csv"]


@patch("os.path.exists", Mock(return_value=True))
@patch("os.path.abspath", lambda x: x)
@patch("os.path.isdir", lambda x: x[-1] == "/")
def test_file_data_source_get_root():
    data_source = sagemaker.utils.local.data.LocalFileDataSource("/some/path/")
    assert data_source.get_root_dir() == "/some/path/"

    data_source = sagemaker.utils.local.data.LocalFileDataSource("/some/path/my_file.csv")
    assert data_source.get_root_dir() == "/some/path"


@patch("sagemaker.local.data.LocalFileDataSource")
@patch("sagemaker.utils.download_folder")
@patch("tempfile.mkdtemp", lambda dir: "/tmp/working_dir")
def test_s3_data_source(download_folder, LocalFileDataSource, sagemaker_local_session):
    data_source = sagemaker.utils.local.data.S3DataSource(
        "my_bucket", "/transform/data", sagemaker_local_session
    )
    download_folder.assert_called()
    data_source.get_file_list()
    LocalFileDataSource().get_file_list.assert_called()
    data_source.get_root_dir()
    LocalFileDataSource().get_root_dir.assert_called()


def test_get_splitter_instance_with_valid_types():
    splitter = sagemaker.utils.local.data.get_splitter_instance(None)
    assert isinstance(splitter, sagemaker.utils.local.data.NoneSplitter)

    splitter = sagemaker.utils.local.data.get_splitter_instance("Line")
    assert isinstance(splitter, sagemaker.utils.local.data.LineSplitter)

    splitter = sagemaker.utils.local.data.get_splitter_instance("RecordIO")
    assert isinstance(splitter, sagemaker.utils.local.data.RecordIOSplitter)


def test_get_splitter_instance_with_invalid_types():
    with pytest.raises(ValueError):
        sagemaker.utils.local.data.get_splitter_instance("SomethingInvalid")


def test_none_splitter(tmpdir):
    splitter = sagemaker.utils.local.data.NoneSplitter()

    test_file_path = tmpdir.join("none_test.txt")

    with test_file_path.open("w") as f:
        f.write("this\nis\na\ntest")

    data = [x for x in splitter.split(str(test_file_path))]
    assert data == ["this\nis\na\ntest"]

    test_bin_file_path = tmpdir.join("none_test.bin")

    with test_bin_file_path.open("wb") as f:
        f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00C")

    data = [x for x in splitter.split(str(test_bin_file_path))]
    assert data == [b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00C"]


def test_line_splitter(tmpdir):
    test_file_path = tmpdir.join("line_test.txt")

    with test_file_path.open("w") as f:
        for i in range(10):
            f.write("%s\n" % i)

    splitter = sagemaker.utils.local.data.LineSplitter()
    data = [x for x in splitter.split(str(test_file_path))]
    assert len(data) == 10
    for i in range(10):
        assert data[i] == "%s\n" % str(i)


def test_recordio_splitter(tmpdir):
    test_file_path = tmpdir.join("recordio_test.txt")
    with test_file_path.open("wb") as f:
        for i in range(10):
            data = str(i).encode("utf-8")
            sagemaker.amazon.common._write_recordio(f, data)

    splitter = sagemaker.utils.local.data.RecordIOSplitter()
    data = [x for x in splitter.split(str(test_file_path))]

    assert len(data) == 10


def test_get_batch_strategy_instance_with_valid_type():
    # Single Record
    strategy = sagemaker.utils.local.data.get_batch_strategy_instance("SingleRecord", None)
    assert isinstance(strategy, sagemaker.utils.local.data.SingleRecordStrategy)

    # Multi Record
    strategy = sagemaker.utils.local.data.get_batch_strategy_instance("MultiRecord", None)
    assert isinstance(strategy, sagemaker.utils.local.data.MultiRecordStrategy)


def test_get_batch_strategy_instance_with_invalid_type():
    with pytest.raises(ValueError):
        # something invalid
        sagemaker.utils.local.data.get_batch_strategy_instance("NiceRecord", None)


def test_single_record_strategy_with_small_records():
    splitter = Mock()

    single_record = sagemaker.utils.local.data.SingleRecordStrategy(splitter)
    data = ["123", "456", "789"]
    splitter.split.return_value = data

    # given 3 small records the output should be the same 3 records
    batch_records = [r for r in single_record.pad("some_file", 6)]
    assert data == batch_records


def test_single_record_strategy_with_large_records():
    splitter = Mock()
    mb = 1024 * 1024

    single_record = sagemaker.utils.local.data.SingleRecordStrategy(splitter)
    # We will construct a huge record greater than 1MB and we expect an exception
    # since there is no way to fit this with the payload size.
    buffer = ""
    while sys.getsizeof(buffer) < 2 * mb:
        buffer += "1" * 100

    data = [buffer]
    with pytest.raises(RuntimeError):
        splitter.split.return_value = data
        batch_records = [r for r in single_record.pad("some_file", 1)]
        print(batch_records)


def test_single_record_strategy_with_no_payload_limit():
    # passing 0 as the max_payload_size should work and a 1MB record should be returned
    # correctly.
    splitter = Mock()
    mb = 1024 * 1024

    buffer = ""
    while sys.getsizeof(buffer) < 2 * mb:
        buffer += "1" * 100
    splitter.split.return_value = [buffer]

    single_record = sagemaker.utils.local.data.SingleRecordStrategy(splitter)
    batch_records = [r for r in single_record.pad("some_file", 0)]
    assert len(batch_records) == 1


def test_multi_record_strategy_with_small_records():
    splitter = Mock()

    multi_record = sagemaker.utils.local.data.MultiRecordStrategy(splitter)
    data = ["123", "456", "789"]
    splitter.split.return_value = data

    # given 3 small records, the output should be 1 single record with the data from all 3 combined
    batch_records = [r for r in multi_record.pad("some_file", 6)]
    assert len(batch_records) == 1
    assert batch_records[0] == "123456789"


def test_multi_record_strategy_with_large_records():
    splitter = Mock()
    mb = 1024 * 1024

    multi_record = sagemaker.utils.local.data.MultiRecordStrategy(splitter)
    # we will construct several large records and we expect them to be merged into <1MB ones
    buffer = ""
    while sys.getsizeof(buffer) < 0.5 * mb:
        buffer += "1" * 100

    # buffer should be aprox 0.5 MB. We will make the data total 10 MB made out of 0.5mb records
    # with a max_payload size of 1MB the expectation is to have ~10 output records.

    data = [buffer for _ in range(10)]
    splitter.split.return_value = data

    batch_records = [r for r in multi_record.pad("some_file", 1)]
    # check with 11 because there may be a bit of leftover.
    assert len(batch_records) <= 11
