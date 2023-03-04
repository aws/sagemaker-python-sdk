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

import os

import pytest
import yaml

from sagemaker.config import SageMakerConfig
from sagemaker.s3 import S3Uploader
from tests.integ.kms_utils import get_or_create_kms_key


@pytest.fixture()
def get_data_dir():
    return os.path.join(os.path.dirname(__file__), "..", "data", "config")


@pytest.fixture(scope="module")
def s3_files_kms_key(sagemaker_session):
    return get_or_create_kms_key(sagemaker_session=sagemaker_session)


@pytest.fixture()
def expected_merged_config(get_data_dir):
    expected_merged_config_file_path = os.path.join(
        get_data_dir, "expected_output_config_after_merge.yaml"
    )
    return yaml.safe_load(open(expected_merged_config_file_path, "r").read())


def test_config_download_from_s3_and_merge(
    sagemaker_session,
    s3_files_kms_key,
    get_data_dir,
    expected_merged_config,
):

    # Note: not using unique_name_from_base() here because the config contents are expected to
    # change very rarely (if ever), so rather than writing new files and deleting them every time
    # we can just use the same S3 paths
    s3_uri_prefix = os.path.join(
        "s3://",
        sagemaker_session.default_bucket(),
        "integ-test-sagemaker_config",
    )

    config_file_1_local_path = os.path.join(get_data_dir, "sample_config_for_merge.yaml")
    config_file_2_local_path = os.path.join(get_data_dir, "sample_additional_config_for_merge.yaml")

    config_file_1_as_yaml = open(config_file_1_local_path, "r").read()
    config_file_2_as_yaml = open(config_file_2_local_path, "r").read()

    s3_uri_config_1 = os.path.join(s3_uri_prefix, "config_1.yaml")
    s3_uri_config_2 = os.path.join(s3_uri_prefix, "config_2.yaml")

    # Upload S3 files in case they dont already exist
    S3Uploader.upload_string_as_file_body(
        body=config_file_1_as_yaml,
        desired_s3_uri=s3_uri_config_1,
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )
    S3Uploader.upload_string_as_file_body(
        body=config_file_2_as_yaml,
        desired_s3_uri=s3_uri_config_2,
        kms_key=s3_files_kms_key,
        sagemaker_session=sagemaker_session,
    )

    # The thing being tested.
    sagemaker_config = SageMakerConfig(additional_config_paths=[s3_uri_config_1, s3_uri_config_2])

    assert sagemaker_config.config == expected_merged_config
