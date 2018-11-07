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

import inspect
import os

import pytest
from mock import Mock, patch

from sagemaker.fw_utils import create_image_uri, framework_name_from_image, \
    framework_version_from_tag, \
    model_code_key_prefix
from sagemaker.fw_utils import tar_and_upload_dir, parse_s3_url, UploadedCode, validate_source_dir
from sagemaker.utils import name_from_image

DATA_DIR = 'data_dir'
BUCKET_NAME = 'mybucket'
ROLE = 'Sagemaker'
REGION = 'us-west-2'
SCRIPT_PATH = 'script.py'
TIMESTAMP = '2017-10-10-14-14-15'


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    session_mock = Mock(name='sagemaker_session', boto_session=boto_mock)
    session_mock.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    session_mock.expand_role = Mock(name="expand_role", return_value=ROLE)
    session_mock.sagemaker_client.describe_training_job = \
        Mock(return_value={'ModelArtifacts': {'S3ModelArtifacts': 's3://m/m.tar.gz'}})
    return session_mock


def test_create_image_uri_cpu():
    image_uri = create_image_uri('mars-south-3', 'mlfw', 'ml.c4.large', '1.0rc', 'py2', '23')
    assert image_uri == '23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-cpu-py2'

    image_uri = create_image_uri('mars-south-3', 'mlfw', 'local', '1.0rc', 'py2', '23')
    assert image_uri == '23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-cpu-py2'


def test_create_image_uri_no_python():
    image_uri = create_image_uri('mars-south-3', 'mlfw', 'ml.c4.large', '1.0rc', account='23')
    assert image_uri == '23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-cpu'


def test_create_image_uri_bad_python():
    with pytest.raises(ValueError):
        create_image_uri('mars-south-3', 'mlfw', 'ml.c4.large', '1.0rc', 'py0')


def test_create_image_uri_gpu():
    image_uri = create_image_uri('mars-south-3', 'mlfw', 'ml.p3.2xlarge', '1.0rc', 'py3', '23')
    assert image_uri == '23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3'

    image_uri = create_image_uri('mars-south-3', 'mlfw', 'local_gpu', '1.0rc', 'py3', '23')
    assert image_uri == '23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3'


def test_create_image_uri_default_account():
    image_uri = create_image_uri('mars-south-3', 'mlfw', 'ml.p3.2xlarge', '1.0rc', 'py3')
    assert image_uri == '520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3'


def test_create_image_uri_gov_cloud():
    image_uri = create_image_uri('us-gov-west-1', 'mlfw', 'ml.p3.2xlarge', '1.0rc', 'py3')
    assert image_uri == '246785580436.dkr.ecr.us-gov-west-1.amazonaws.com/sagemaker-mlfw:1.0rc-gpu-py3'


def test_invalid_instance_type():
    # instance type is missing 'ml.' prefix
    with pytest.raises(ValueError):
        create_image_uri('mars-south-3', 'mlfw', 'p3.2xlarge', '1.0.0', 'py3')


def test_optimized_family():
    image_uri = create_image_uri('mars-south-3', 'mlfw', 'ml.p3.2xlarge', '1.0.0', 'py3',
                                 optimized_families=['c5', 'p3'])
    assert image_uri == '520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0.0-p3-py3'


def test_unoptimized_cpu_family():
    image_uri = create_image_uri('mars-south-3', 'mlfw', 'ml.m4.xlarge', '1.0.0', 'py3',
                                 optimized_families=['c5', 'p3'])
    assert image_uri == '520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0.0-cpu-py3'


def test_unoptimized_gpu_family():
    image_uri = create_image_uri('mars-south-3', 'mlfw', 'ml.p2.xlarge', '1.0.0', 'py3',
                                 optimized_families=['c5', 'p3'])
    assert image_uri == '520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw:1.0.0-gpu-py3'


def test_tar_and_upload_dir_s3(sagemaker_session):
    bucket = 'mybucker'
    s3_key_prefix = 'something/source'
    script = 'mnist.py'
    directory = 's3://m'
    result = tar_and_upload_dir(sagemaker_session, bucket, s3_key_prefix, script, directory)
    assert result == UploadedCode('s3://m', 'mnist.py')


def test_validate_source_dir_does_not_exits(sagemaker_session):
    script = 'mnist.py'
    directory = ' !@#$%^&*()path probably in not there.!@#$%^&*()'
    with pytest.raises(ValueError):
        validate_source_dir(script, directory)


def test_validate_source_dir_is_not_directory(sagemaker_session):
    script = 'mnist.py'
    directory = inspect.getfile(inspect.currentframe())
    with pytest.raises(ValueError):
        validate_source_dir(script, directory)


def test_validate_source_dir_file_not_in_dir():
    script = ' !@#$%^&*() .myscript. !@#$%^&*() '
    directory = '.'
    with pytest.raises(ValueError):
        validate_source_dir(script, directory)


def test_tar_and_upload_dir_not_s3(sagemaker_session):
    bucket = 'mybucker'
    s3_key_prefix = 'something/source'
    script = os.path.basename(__file__)
    directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    result = tar_and_upload_dir(sagemaker_session, bucket, s3_key_prefix, script, directory)
    assert result == UploadedCode('s3://{}/{}/sourcedir.tar.gz'.format(bucket, s3_key_prefix),
                                  script)


def test_framework_name_from_image_mxnet():
    image_name = '123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet:1.1-gpu-py3'
    assert ('mxnet', 'py3', '1.1-gpu-py3') == framework_name_from_image(image_name)


def test_framework_name_from_image_tf():
    image_name = '123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow:1.6-cpu-py2'
    assert ('tensorflow', 'py2', '1.6-cpu-py2') == framework_name_from_image(image_name)


def test_legacy_name_from_framework_image():
    image_name = '123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py3-gpu:2.5.6-gpu-py2'
    framework, py_ver, tag = framework_name_from_image(image_name)
    assert framework == 'mxnet'
    assert py_ver == 'py3'
    assert tag == '2.5.6-gpu-py2'


def test_legacy_name_from_wrong_framework():
    framework, py_ver, tag = framework_name_from_image(
        '123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py2-gpu:1')
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_legacy_name_from_wrong_python():
    framework, py_ver, tag = framework_name_from_image(
        '123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py4-gpu:1')
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_legacy_name_from_wrong_device():
    framework, py_ver, tag = framework_name_from_image(
        '123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py4-gpu:1')
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_legacy_name_from_image_any_tag():
    image_name = '123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tensorflow-py2-cpu:any-tag'
    framework, py_ver, tag = framework_name_from_image(image_name)
    assert framework == 'tensorflow'
    assert py_ver == 'py2'
    assert tag == 'any-tag'


def test_framework_version_from_tag():
    version = framework_version_from_tag('1.5rc-keras-gpu-py2')
    assert version == '1.5rc-keras'


def test_framework_version_from_tag_other():
    version = framework_version_from_tag('weird-tag-py2')
    assert version is None


def test_parse_s3_url():
    bucket, key_prefix = parse_s3_url('s3://bucket/code_location')
    assert 'bucket' == bucket
    assert 'code_location' == key_prefix


def test_parse_s3_url_fail():
    with pytest.raises(ValueError) as error:
        parse_s3_url('t3://code_location')
    assert 'Expecting \'s3\' scheme' in str(error)


def test_model_code_key_prefix_with_all_values_present():
    key_prefix = model_code_key_prefix('prefix', 'model_name', 'image_name')
    assert key_prefix == 'prefix/model_name'


def test_model_code_key_prefix_with_no_prefix_and_all_other_values_present():
    key_prefix = model_code_key_prefix(None, 'model_name', 'image_name')
    assert key_prefix == 'model_name'


@patch('time.strftime', return_value=TIMESTAMP)
def test_model_code_key_prefix_with_only_image_present(time):
    key_prefix = model_code_key_prefix(None, None, 'image_name')
    assert key_prefix == name_from_image('image_name')


@patch('time.strftime', return_value=TIMESTAMP)
def test_model_code_key_prefix_and_image_present(time):
    key_prefix = model_code_key_prefix('prefix', None, 'image_name')
    assert key_prefix == 'prefix/' + name_from_image('image_name')


def test_model_code_key_prefix_with_prefix_present_and_others_none_fail():
    with pytest.raises(TypeError) as error:
        model_code_key_prefix('prefix', None, None)
    assert 'expected string' in str(error)


def test_model_code_key_prefix_with_all_none_fail():
    with pytest.raises(TypeError) as error:
        model_code_key_prefix(None, None, None)
    assert 'expected string' in str(error)
