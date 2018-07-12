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
from mock import Mock

from sagemaker.transformer import Transformer, _TransformJob

MODEL_NAME = 'model'
INSTANCE_COUNT = 1
INSTANCE_TYPE = 'ml.m4.xlarge'
JOB_NAME = 'job'
DATA = 's3://bucket/input-data'
S3_DATA_TYPE = 'S3Prefix'
OUTPUT_PATH = 's3://bucket/output'


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name='boto_session')
    return Mock(name='sagemaker_session', boto_session=boto_mock)


@pytest.fixture()
def transformer(sagemaker_session):
    return Transformer(MODEL_NAME, INSTANCE_COUNT, INSTANCE_TYPE,
                       output_path=OUTPUT_PATH, sagemaker_session=sagemaker_session)


# _TransformJob tests

def test_start_new(transformer, sagemaker_session):
    transformer._current_job_name = JOB_NAME

    job = _TransformJob(sagemaker_session, JOB_NAME)
    started_job = job.start_new(transformer, DATA, S3_DATA_TYPE, None, None, None)

    assert started_job.sagemaker_session == sagemaker_session
    sagemaker_session.transform.assert_called_once()


def test_load_config(transformer):
    expected_config = {
        'input_config': {
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': S3_DATA_TYPE,
                    'S3Uri': DATA,
                },
            },
        },
        'output_config': {
            'S3OutputPath': OUTPUT_PATH,
        },
        'resource_config': {
            'InstanceCount': INSTANCE_COUNT,
            'InstanceType': INSTANCE_TYPE,
        },
    }

    actual_config = _TransformJob._load_config(DATA, S3_DATA_TYPE, None, None, None, transformer)
    assert actual_config == expected_config


def test_format_inputs_to_input_config():
    expected_config = {
        'DataSource': {
            'S3DataSource': {
                'S3DataType': S3_DATA_TYPE,
                'S3Uri': DATA,
            },
        },
    }

    actual_config = _TransformJob._format_inputs_to_input_config(DATA, S3_DATA_TYPE, None, None, None)
    assert actual_config == expected_config


def test_format_inputs_to_input_config_with_optional_params():
    compression = 'Gzip'
    content_type = 'text/csv'
    split = 'Line'

    expected_config = {
        'DataSource': {
            'S3DataSource': {
                'S3DataType': S3_DATA_TYPE,
                'S3Uri': DATA,
            },
        },
        'CompressionType': compression,
        'ContentType': content_type,
        'SplitType': split,
    }

    actual_config = _TransformJob._format_inputs_to_input_config(DATA, S3_DATA_TYPE, content_type, compression, split)
    assert actual_config == expected_config


def test_prepare_output_config():
    config = _TransformJob._prepare_output_config(OUTPUT_PATH, None, None, None)

    assert config == {'S3OutputPath': OUTPUT_PATH}


def test_prepare_output_config_with_optional_params():
    kms_key = 'key'
    assemble_with = 'Line'
    accept = 'text/csv'

    expected_config = {
        'S3OutputPath': OUTPUT_PATH,
        'KmsKeyId': kms_key,
        'AssembleWith': assemble_with,
        'Accept': accept,
    }

    actual_config = _TransformJob._prepare_output_config(OUTPUT_PATH, kms_key, assemble_with, accept)
    assert actual_config == expected_config


def test_prepare_resource_config():
    config = _TransformJob._prepare_resource_config(INSTANCE_COUNT, INSTANCE_TYPE)
    assert config == {'InstanceCount': INSTANCE_COUNT, 'InstanceType': INSTANCE_TYPE}
