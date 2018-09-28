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

import os
import pytest
import urllib3

from botocore.exceptions import ClientError
from mock import call, patch, Mock, MagicMock

import sagemaker.local.data


REGION = 'us-west-2'
BUCKET_NAME = 'mybucket'
EXPANDED_ROLE = 'arn:aws:iam::111111111111:role/ExpandedRole'

@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    boto_mock.client('sts').get_caller_identity.return_value = {'Account': '123'}
    boto_mock.resource('s3').Bucket(BUCKET_NAME).objects.filter.return_value = []

    sms = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())

    sms.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    sms.expand_role = Mock(return_value=EXPANDED_ROLE)

    return sms


@patch('os.path.exists', Mock(return_value=True))
def test_data_source_factory(sagemaker_session):
    factory = sagemaker.local.data.DataSourceFactory()
    # file
    data_source = factory.get_instance('file:///my/file', sagemaker_session)
    assert isinstance(data_source, sagemaker.local.data.LocalFileDataSource)

    # s3

    data_source = factory.get_instance('s3://bucket/path', sagemaker_session)
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


def test_s3_data_source(sagemaker_session):
    pass
