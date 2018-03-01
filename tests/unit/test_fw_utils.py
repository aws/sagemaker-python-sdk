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
import inspect
from mock import Mock
import os
from sagemaker.fw_utils import create_image_uri, framework_name_from_image, framework_version_from_tag
from sagemaker.fw_utils import tar_and_upload_dir, parse_s3_url, UploadedCode
import pytest


DATA_DIR = 'data_dir'
BUCKET_NAME = 'mybucket'
ROLE = 'Sagemaker'
REGION = 'us-west-2'
SCRIPT_PATH = 'script.py'


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session', region_name=REGION)
    ims = Mock(name='sagemaker_session', boto_session=boto_mock)
    ims.default_bucket = Mock(name='default_bucket', return_value=BUCKET_NAME)
    ims.expand_role = Mock(name="expand_role", return_value=ROLE)
    ims.sagemaker_client.describe_training_job = Mock(return_value={'ModelArtifacts':
                                                                    {'S3ModelArtifacts': 's3://m/m.tar.gz'}})
    return ims


def test_create_image_uri_cpu():
    image_uri = create_image_uri('mars-south-3', 'mlfw', 'any-non-gpu-device', '1.0rc', 'py2', '23')
    assert image_uri == '23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw-py2-cpu:1.0rc-cpu-py2'


def test_create_image_uri_gpu():
    image_uri = create_image_uri('mars-south-3', 'mlfw', 'ml.p3.2xlarge', '1.0rc', 'py3', '23')
    assert image_uri == '23.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw-py3-gpu:1.0rc-gpu-py3'


def test_create_image_uri_default_account():
    image_uri = create_image_uri('mars-south-3', 'mlfw', 'ml.p3.2xlarge', '1.0rc', 'py3')
    assert image_uri == '520713654638.dkr.ecr.mars-south-3.amazonaws.com/sagemaker-mlfw-py3-gpu:1.0rc-gpu-py3'


def test_tar_and_upload_dir_s3(sagemaker_session):
    bucket = 'mybucker'
    s3_key_prefix = 'something/source'
    script = 'mnist.py'
    directory = 's3://m'
    result = tar_and_upload_dir(sagemaker_session, bucket, s3_key_prefix, script, directory)
    assert result == UploadedCode('s3://m', 'mnist.py')


def test_tar_and_upload_dir_does_not_exits(sagemaker_session):
    bucket = 'mybucker'
    s3_key_prefix = 'something/source'
    script = 'mnist.py'
    directory = ' !@#$%^&*()path probably in not there.!@#$%^&*()'
    with pytest.raises(ValueError) as error:
        tar_and_upload_dir(sagemaker_session, bucket, s3_key_prefix, script, directory)
    assert 'does not exist' in str(error)


def test_tar_and_upload_dir_is_not_directory(sagemaker_session):
    bucket = 'mybucker'
    s3_key_prefix = 'something/source'
    script = 'mnist.py'
    directory = inspect.getfile(inspect.currentframe())
    with pytest.raises(ValueError) as error:
        tar_and_upload_dir(sagemaker_session, bucket, s3_key_prefix, script, directory)
    assert 'is not a directory' in str(error)


def test_tar_and_upload_dir_file_not_in_dir(sagemaker_session):
    bucket = 'mybucker'
    s3_key_prefix = 'something/source'
    script = ' !@#$%^&*() .myscript. !@#$%^&*() '
    directory = '.'
    with pytest.raises(ValueError) as error:
        tar_and_upload_dir(sagemaker_session, bucket, s3_key_prefix, script, directory)
    assert 'No file named' in str(error)


def test_tar_and_upload_dir_not_s3(sagemaker_session):
    bucket = 'mybucker'
    s3_key_prefix = 'something/source'
    script = os.path.basename(__file__)
    directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    result = tar_and_upload_dir(sagemaker_session, bucket, s3_key_prefix, script, directory)
    assert result == UploadedCode('s3://{}/{}/sourcedir.tar.gz'.format(bucket, s3_key_prefix), script)


def test_framework_name_from_framework_image():
    image_name = '123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-mxnet-py3-gpu:2.5.6-gpu-py2'
    framework, py_ver, tag = framework_name_from_image(image_name)
    assert framework == 'mxnet'
    assert py_ver == 'py3'
    assert tag == '2.5.6-gpu-py2'


def test_framework_name_from_wrong_framework():
    framework, py_ver, tag = framework_name_from_image('123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py2-gpu:1')
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_framework_name_from_wrong_python():
    framework, py_ver, tag = framework_name_from_image('123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py4-gpu:1')
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_framework_name_from_wrong_device():
    framework, py_ver, tag = framework_name_from_image('123.dkr.ecr.us-west-2.amazonaws.com/sagemaker-myown-py4-gpu:1')
    assert framework is None
    assert py_ver is None
    assert tag is None


def test_framework_name_from_image_any_tag():
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
