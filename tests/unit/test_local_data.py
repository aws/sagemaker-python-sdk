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
