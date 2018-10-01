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

import pytest
import sys

from mock import patch, Mock

import sagemaker.local.data
import sagemaker.amazon


@patch('os.path.exists', Mock(return_value=True))
@patch('sagemaker.local.data.download_folder', Mock())
def test_data_source_factory(sagemaker_local_session):
    factory = sagemaker.local.data.DataSourceFactory()
    # file
    data_source = factory.get_instance('file:///my/file', sagemaker_local_session)
    assert isinstance(data_source, sagemaker.local.data.LocalFileDataSource)

    # s3
    data_source = factory.get_instance('s3://bucket/path', sagemaker_local_session)
    assert isinstance(data_source, sagemaker.local.data.S3DataSource)


@patch('os.path.exists', Mock(return_value=True))
@patch('os.path.abspath', lambda x: x)
@patch('os.path.isdir', lambda x: x[-1] == '/')
@patch('os.path.isfile', lambda x: x[-1] != '/')
@patch('os.listdir')
def test_file_data_source_get_file_list(listdir):
    data_source = sagemaker.local.data.LocalFileDataSource('/some/path/')
    listdir.return_value = [
        '/some/path/a',
        '/some/path/b',
        '/some/path/c/',
        '/some/path/c/a'
    ]
    expected = [
        '/some/path/a',
        '/some/path/b',
        '/some/path/c/a'
    ]
    result = data_source.get_file_list()
    assert result == expected

    data_source = sagemaker.local.data.LocalFileDataSource('/some/batch/file.csv')
    assert data_source.get_file_list() == ['/some/batch/file.csv']


@patch('os.path.exists', Mock(return_value=True))
@patch('os.path.abspath', lambda x: x)
@patch('os.path.isdir', lambda x: x[-1] == '/')
def test_file_data_source_get_root():
    data_source = sagemaker.local.data.LocalFileDataSource('/some/path/')
    assert data_source.get_root_dir() == '/some/path/'

    data_source = sagemaker.local.data.LocalFileDataSource('/some/path/my_file.csv')
    assert data_source.get_root_dir() == '/some/path'


@patch('sagemaker.local.data.LocalFileDataSource')
@patch('sagemaker.local.data.download_folder')
@patch('tempfile.mkdtemp', lambda dir: '/tmp/working_dir')
def test_s3_data_source(download_folder, LocalFileDataSource, sagemaker_local_session):
    data_source = sagemaker.local.data.S3DataSource('my_bucket', '/transform/data', sagemaker_local_session)
    download_folder.assert_called()
    data_source.get_file_list()
    LocalFileDataSource().get_file_list.assert_called()
    data_source.get_root_dir()
    LocalFileDataSource().get_root_dir.assert_called()


def test_splitter_factory():
    factory = sagemaker.local.data.SplitterFactory()
    # file
    splitter = factory.get_instance(None)
    assert isinstance(splitter, sagemaker.local.data.NoneSplitter)

    splitter = factory.get_instance('Line')
    assert isinstance(splitter, sagemaker.local.data.LineSplitter)

    splitter = factory.get_instance('RecordIO')
    assert isinstance(splitter, sagemaker.local.data.RecordIOSplitter)

    with pytest.raises(ValueError):
        # something invalid
        factory.get_instance('JSON')


def test_none_splitter(tmpdir):
    test_file_path = tmpdir.join('none_test.txt')

    with test_file_path.open('w') as f:
        f.write('this\nis\na\ntest')

    splitter = sagemaker.local.data.NoneSplitter()
    data = [x for x in splitter.split(str(test_file_path))]
    assert data == ['this\nis\na\ntest']


def test_line_splitter(tmpdir):
    test_file_path = tmpdir.join('line_test.txt')

    with test_file_path.open('w') as f:
        for i in range(10):
            f.write('%s\n' % i)

    splitter = sagemaker.local.data.LineSplitter()
    data = [x for x in splitter.split(str(test_file_path))]
    assert len(data) == 10
    for i in range(10):
        assert data[i] == '%s\n' % str(i)


def test_recordio_splitter(tmpdir):

    test_file_path = tmpdir.join('recordio_test.txt')
    with test_file_path.open('wb') as f:
        for i in range(10):
            data = str(i).encode('utf-8')
            sagemaker.amazon.common._write_recordio(f, data)

    splitter = sagemaker.local.data.RecordIOSplitter()
    data = [x for x in splitter.split(str(test_file_path))]

    assert len(data) == 10


def test_batch_strategy_factory():
    factory = sagemaker.local.data.BatchStrategyFactory()
    # Single Record
    strategy = factory.get_instance('SingleRecord', None)
    assert isinstance(strategy, sagemaker.local.data.SingleRecordStrategy)

    # Multi Record
    strategy = factory.get_instance('MultiRecord', None)
    assert isinstance(strategy, sagemaker.local.data.MultiRecordStrategy)

    with pytest.raises(ValueError):
        # something invalid
        factory.get_instance('NiceRecord', None)


def test_single_record_strategy():
    splitter = Mock()
    mb = 1024 * 1024

    single_record = sagemaker.local.data.SingleRecordStrategy(splitter)
    data = ['123', '456', '789']
    splitter.split.return_value = data

    # given 3 small records the output should be the same 3 records
    batch_records = [r for r in single_record.pad('some_file', 6)]
    assert data == batch_records

    # now we will construct a huge record greater than 1MB and we expect an exception
    # since there is no way to fit this with the payload size.
    buffer = ''
    while sys.getsizeof(buffer) < 2 * mb:
        buffer += '1' * 100

    data = [buffer]
    with pytest.raises(RuntimeError):
        splitter.split.return_value = data
        batch_records = [r for r in single_record.pad('some_file', 1)]
        print(batch_records)

    # passing 0 as the max_payload_size should work and the same record above should be returned
    # correctly.
    batch_records = [r for r in single_record.pad('some_file', 0)]
    assert len(batch_records) == 1


def test_multi_record_strategy():
    splitter = Mock()
    mb = 1024 * 1024

    multi_record = sagemaker.local.data.MultiRecordStrategy(splitter)
    data = ['123', '456', '789']
    splitter.split.return_value = data

    # given 3 small records, the output should be 1 single record with the data from all 3 combined
    batch_records = [r for r in multi_record.pad('some_file', 6)]
    assert len(batch_records) == 1
    assert batch_records[0] == '123456789'

    # now we will construct several large records and we expect them to be merged into <1MB ones
    buffer = ''
    while sys.getsizeof(buffer) < 0.5 * mb:
        buffer += '1' * 100

    # buffer should be aprox 0.5 MB. We will make the data total 10 MB made out of 0.5mb records
    # with a max_payload size of 1MB the expectation is to have ~10 output records.

    data = [buffer for _ in range(10)]
    splitter.split.return_value = data

    batch_records = [r for r in multi_record.pad('some_file', 1)]
    # check with 11 because there may be a bit of leftover.
    assert len(batch_records) <= 11
