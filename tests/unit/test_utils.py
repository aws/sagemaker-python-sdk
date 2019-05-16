# -*- coding: utf-8 -*-

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
from mock import call, patch, Mock, MagicMock

import sagemaker


NAME = 'base_name'
BUCKET_NAME = 'some_bucket'


def test_get_config_value():

    config = {
        'local': {
            'region_name': 'us-west-2',
            'port': '123'
        },
        'other': {
            'key': 1
        }
    }

    assert sagemaker.utils.get_config_value('local.region_name', config) == 'us-west-2'
    assert sagemaker.utils.get_config_value('local', config) == {'region_name': 'us-west-2', 'port': '123'}

    assert sagemaker.utils.get_config_value('does_not.exist', config) is None
    assert sagemaker.utils.get_config_value('other.key', None) is None


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


@patch('sagemaker.utils.sagemaker_timestamp')
def test_name_from_base(sagemaker_timestamp):
    sagemaker.utils.name_from_base(NAME, short=False)
    assert sagemaker_timestamp.called_once


@patch('sagemaker.utils.sagemaker_short_timestamp')
def test_name_from_base_short(sagemaker_short_timestamp):
    sagemaker.utils.name_from_base(NAME, short=True)
    assert sagemaker_short_timestamp.called_once


def test_unique_name_from_base():
    assert re.match(r'base-\d{10}-[a-f0-9]{4}', sagemaker.utils.unique_name_from_base('base'))


def test_unique_name_from_base_truncated():
    assert re.match(r'real-\d{10}-[a-f0-9]{4}',
                    sagemaker.utils.unique_name_from_base('really-long-name', max_length=20))


def test_to_str_with_native_string():
    value = 'some string'
    assert sagemaker.utils.to_str(value) == value


def test_to_str_with_unicode_string():
    value = u'åñøthér strîng'
    assert sagemaker.utils.to_str(value) == value


def test_name_from_tuning_arn():
    arn = 'arn:aws:sagemaker:us-west-2:968277160000:hyper-parameter-tuning-job/resnet-sgd-tuningjob-11-07-34-11'
    name = sagemaker.utils.extract_name_from_job_arn(arn)
    assert name == 'resnet-sgd-tuningjob-11-07-34-11'


def test_name_from_training_arn():
    arn = 'arn:aws:sagemaker:us-west-2:968277160000:training-job/resnet-sgd-tuningjob-11-22-38-46-002-2927640b'
    name = sagemaker.utils.extract_name_from_job_arn(arn)
    assert name == 'resnet-sgd-tuningjob-11-22-38-46-002-2927640b'


MESSAGE = 'message'
STATUS = 'status'
TRAINING_JOB_DESCRIPTION_1 = {
    'SecondaryStatusTransitions': [{'StatusMessage': MESSAGE, 'Status': STATUS}]
}
TRAINING_JOB_DESCRIPTION_2 = {
    'SecondaryStatusTransitions': [{'StatusMessage': 'different message', 'Status': STATUS}]
}

TRAINING_JOB_DESCRIPTION_EMPTY = {
    'SecondaryStatusTransitions': []
}


def test_secondary_training_status_changed_true():
    changed = sagemaker.utils.secondary_training_status_changed(TRAINING_JOB_DESCRIPTION_1, TRAINING_JOB_DESCRIPTION_2)
    assert changed is True


def test_secondary_training_status_changed_false():
    changed = sagemaker.utils.secondary_training_status_changed(TRAINING_JOB_DESCRIPTION_1, TRAINING_JOB_DESCRIPTION_1)
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
    changed = sagemaker.utils.secondary_training_status_changed(TRAINING_JOB_DESCRIPTION_EMPTY,
                                                                TRAINING_JOB_DESCRIPTION_1)
    assert changed is False


def test_secondary_training_status_message_status_changed():
    now = datetime.now()
    TRAINING_JOB_DESCRIPTION_1['LastModifiedTime'] = now
    expected = '{} {} - {}'.format(
        datetime.utcfromtimestamp(time.mktime(now.timetuple())).strftime('%Y-%m-%d %H:%M:%S'),
        STATUS,
        MESSAGE
    )
    assert sagemaker.utils.secondary_training_status_message(TRAINING_JOB_DESCRIPTION_1,
                                                             TRAINING_JOB_DESCRIPTION_EMPTY) == expected


def test_secondary_training_status_message_status_not_changed():
    now = datetime.now()
    TRAINING_JOB_DESCRIPTION_1['LastModifiedTime'] = now
    expected = '{} {} - {}'.format(
        datetime.utcfromtimestamp(time.mktime(now.timetuple())).strftime('%Y-%m-%d %H:%M:%S'),
        STATUS,
        MESSAGE
    )
    assert sagemaker.utils.secondary_training_status_message(TRAINING_JOB_DESCRIPTION_1,
                                                             TRAINING_JOB_DESCRIPTION_2) == expected


def test_secondary_training_status_message_prev_missing():
    now = datetime.now()
    TRAINING_JOB_DESCRIPTION_1['LastModifiedTime'] = now
    expected = '{} {} - {}'.format(
        datetime.utcfromtimestamp(time.mktime(now.timetuple())).strftime('%Y-%m-%d %H:%M:%S'),
        STATUS,
        MESSAGE
    )
    assert sagemaker.utils.secondary_training_status_message(TRAINING_JOB_DESCRIPTION_1, {}) == expected


@patch('os.makedirs')
def test_download_folder(makedirs):
    boto_mock = Mock(name='boto_session')
    boto_mock.client('sts').get_caller_identity.return_value = {'Account': '123'}

    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())

    train_data = Mock()
    validation_data = Mock()

    train_data.bucket_name.return_value = BUCKET_NAME
    train_data.key = 'prefix/train/train_data.csv'
    validation_data.bucket_name.return_value = BUCKET_NAME
    validation_data.key = 'prefix/train/validation_data.csv'

    s3_files = [train_data, validation_data]
    boto_mock.resource('s3').Bucket(BUCKET_NAME).objects.filter.return_value = s3_files

    obj_mock = Mock()
    boto_mock.resource('s3').Object.return_value = obj_mock

    # all the S3 mocks are set, the test itself begins now.
    sagemaker.utils.download_folder(BUCKET_NAME, '/prefix', '/tmp', session)

    obj_mock.download_file.assert_called()
    calls = [call(os.path.join('/tmp', 'train/train_data.csv')),
             call(os.path.join('/tmp', 'train/validation_data.csv'))]
    obj_mock.download_file.assert_has_calls(calls)
    obj_mock.reset_mock()

    # Testing with a trailing slash for the prefix.
    sagemaker.utils.download_folder(BUCKET_NAME, '/prefix/', '/tmp', session)
    obj_mock.download_file.assert_called()
    obj_mock.download_file.assert_has_calls(calls)


@patch('os.makedirs')
def test_download_folder_points_to_single_file(makedirs):
    boto_mock = Mock(name='boto_session')
    boto_mock.client('sts').get_caller_identity.return_value = {'Account': '123'}

    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())

    train_data = Mock()

    train_data.bucket_name.return_value = BUCKET_NAME
    train_data.key = 'prefix/train/train_data.csv'

    s3_files = [train_data]
    boto_mock.resource('s3').Bucket(BUCKET_NAME).objects.filter.return_value = s3_files

    obj_mock = Mock()
    boto_mock.resource('s3').Object.return_value = obj_mock

    # all the S3 mocks are set, the test itself begins now.
    sagemaker.utils.download_folder(BUCKET_NAME, '/prefix/train/train_data.csv', '/tmp', session)

    obj_mock.download_file.assert_called()
    calls = [call(os.path.join('/tmp', 'train_data.csv'))]
    obj_mock.download_file.assert_has_calls(calls)
    assert boto_mock.resource('s3').Bucket(BUCKET_NAME).objects.filter.call_count == 1
    obj_mock.reset_mock()


def test_download_file():
    boto_mock = Mock(name='boto_session')
    boto_mock.client('sts').get_caller_identity.return_value = {'Account': '123'}
    bucket_mock = Mock()
    boto_mock.resource('s3').Bucket.return_value = bucket_mock
    session = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())

    sagemaker.utils.download_file(BUCKET_NAME, '/prefix/path/file.tar.gz',
                                  '/tmp/file.tar.gz', session)

    bucket_mock.download_file.assert_called_with('prefix/path/file.tar.gz', '/tmp/file.tar.gz')


@patch('tarfile.open')
def test_create_tar_file_with_provided_path(open):
    files = mock_tarfile(open)

    file_list = ['/tmp/a', '/tmp/b']

    path = sagemaker.utils.create_tar_file(file_list, target='/my/custom/path.tar.gz')
    assert path == '/my/custom/path.tar.gz'
    assert files == [['/tmp/a', 'a'], ['/tmp/b', 'b']]


def mock_tarfile(open):
    open.return_value = open
    files = []

    def add_files(filename, arcname):
        files.append([filename, arcname])

    open.__enter__ = Mock()
    open.__enter__().add = add_files
    open.__exit__ = Mock(return_value=None)
    return files


@patch('tarfile.open')
@patch('tempfile.mkstemp', Mock(return_value=(None, '/auto/generated/path')))
def test_create_tar_file_with_auto_generated_path(open):
    files = mock_tarfile(open)

    path = sagemaker.utils.create_tar_file(['/tmp/a', '/tmp/b'])
    assert path == '/auto/generated/path'
    assert files == [['/tmp/a', 'a'], ['/tmp/b', 'b']]


def write_file(path, content):
    with open(path, 'a') as f:
        f.write(content)


def test_repack_model_without_source_dir(tmpdir):

    tmp = str(tmpdir)

    model_path = os.path.join(tmp, 'model')
    write_file(model_path, 'model data')

    source_dir = os.path.join(tmp, 'source-dir')
    os.mkdir(source_dir)
    script_path = os.path.join(source_dir, 'inference.py')
    write_file(script_path, 'inference script')

    contents = [model_path]

    sagemaker_session = MagicMock()
    mock_s3_model_tar(contents, sagemaker_session, tmp)
    fake_upload_path = mock_s3_upload(sagemaker_session, tmp)

    model_uri = 's3://fake/location'

    new_model_uri = sagemaker.utils.repack_model(os.path.join(source_dir, 'inference.py'),
                                                 None,
                                                 model_uri,
                                                 sagemaker_session)

    assert list_tar_files(fake_upload_path, tmpdir) == {'/code/inference.py', '/model'}
    assert re.match(r'^s3://fake/model-\d+-\d+.tar.gz$', new_model_uri)


def test_repack_model_from_s3_saved_model_to_s3(tmpdir):

    tmp = str(tmpdir)

    model_path = os.path.join(tmp, 'model')
    write_file(model_path, 'model data')

    source_dir = os.path.join(tmp, 'source-dir')
    os.mkdir(source_dir)
    script_path = os.path.join(source_dir, 'inference.py')
    write_file(script_path, 'inference script')

    contents = [model_path]

    sagemaker_session = MagicMock()
    mock_s3_model_tar(contents, sagemaker_session, tmp)
    fake_upload_path = mock_s3_upload(sagemaker_session, tmp)

    model_uri = 's3://fake/location'

    new_model_uri = sagemaker.utils.repack_model('inference.py',
                                                 source_dir,
                                                 model_uri,
                                                 sagemaker_session)

    assert list_tar_files(fake_upload_path, tmpdir) == {'/code/inference.py', '/model'}
    assert re.match(r'^s3://fake/model-\d+-\d+.tar.gz$', new_model_uri)


def test_repack_model_from_file_saves_model_to_file(tmpdir):

    tmp = str(tmpdir)

    model_path = os.path.join(tmp, 'model')
    write_file(model_path, 'model data')

    source_dir = os.path.join(tmp, 'source-dir')
    os.mkdir(source_dir)
    script_path = os.path.join(source_dir, 'inference.py')
    write_file(script_path, 'inference script')

    model_tar_path = os.path.join(tmp, 'model.tar.gz')
    sagemaker.utils.create_tar_file([model_path], model_tar_path)

    sagemaker_session = MagicMock()

    file_mode_path = 'file://%s' % model_tar_path
    new_model_uri = sagemaker.utils.repack_model('inference.py',
                                                 source_dir,
                                                 file_mode_path,
                                                 sagemaker_session)

    assert os.path.dirname(new_model_uri) == os.path.dirname(file_mode_path)
    assert list_tar_files(new_model_uri, tmpdir) == {'/code/inference.py', '/model'}


def test_repack_model_with_inference_code_should_replace_the_code(tmpdir):

    tmp = str(tmpdir)

    model_path = os.path.join(tmp, 'model')
    write_file(model_path, 'model data')

    source_dir = os.path.join(tmp, 'source-dir')
    os.mkdir(source_dir)
    script_path = os.path.join(source_dir, 'new-inference.py')
    write_file(script_path, 'inference script')

    old_code_path = os.path.join(tmp, 'code')
    os.mkdir(old_code_path)
    old_script_path = os.path.join(old_code_path, 'old-inference.py')
    write_file(old_script_path, 'old inference script')
    contents = [model_path, old_code_path]

    sagemaker_session = MagicMock()
    mock_s3_model_tar(contents, sagemaker_session, tmp)
    fake_upload_path = mock_s3_upload(sagemaker_session, tmp)

    model_uri = 's3://fake/location'

    new_model_uri = sagemaker.utils.repack_model('inference.py',
                                                 source_dir,
                                                 model_uri,
                                                 sagemaker_session)

    assert list_tar_files(fake_upload_path, tmpdir) == {'/code/new-inference.py', '/model'}
    assert re.match(r'^s3://fake/model-\d+-\d+.tar.gz$', new_model_uri)


def mock_s3_model_tar(contents, sagemaker_session, tmp):
    model_tar_path = os.path.join(tmp, 'model.tar.gz')
    sagemaker.utils.create_tar_file(contents, model_tar_path)
    mock_s3_download(sagemaker_session, model_tar_path)


def mock_s3_download(sagemaker_session, model_tar_path):
    def download_file(_, target):
        shutil.copy2(model_tar_path, target)

    sagemaker_session.boto_session.resource().Bucket().download_file.side_effect = download_file


def mock_s3_upload(sagemaker_session, tmp):
    dst = os.path.join(tmp, 'dst')

    class MockS3Object(object):

        def __init__(self, bucket, key):
            self.bucket = bucket
            self.key = key

        def upload_file(self, target):
            shutil.copy2(target, dst)

    sagemaker_session.boto_session.resource().Object = MockS3Object
    return dst


def list_tar_files(tar_ball, tmpdir):
    tar_ball = tar_ball.replace('file://', '')
    startpath = str(tmpdir.ensure('tmp', dir=True))

    with tarfile.open(name=tar_ball, mode='r:gz') as t:
        t.extractall(path=startpath)

    def walk():
        for root, dirs, files in os.walk(startpath):
            path = root.replace(startpath, '')
            for f in files:
                yield '%s/%s' % (path, f)

    result = set(walk())
    return result if result else {}
