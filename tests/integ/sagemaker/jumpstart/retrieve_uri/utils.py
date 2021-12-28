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

import uuid
from typing import Match, Optional, Tuple
import boto3
import pandas as pd
import regex
import os

from tests.integ.sagemaker.jumpstart.retrieve_uri.constants import (
    HYPERPARAMETER_MODEL_DICT,
    TEST_ASSETS_SPECS,
    TMP_DIRECTORY_PATH,
    TRAINING_DATASET_MODEL_DICT,
)
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket

from sagemaker.s3 import parse_s3_url
from sagemaker.session import Session


def download_file(local_download_path, s3_bucket, s3_key, s3_client) -> None:
    s3_client.download_file(s3_bucket, s3_key, local_download_path)


def extract_role_arn_from_caller_identity(caller_identity_arn: str) -> str:

    ASSUME_ROLE_REGEX = r"^(?P<prefix>.+)sts::(?P<infix>\d+):assumed-role\/(?P<suffix>.+?)\/.*$"

    match: Optional[Match[str]] = regex.match(ASSUME_ROLE_REGEX, caller_identity_arn)

    if match is None:
        # not an assumed role caller identity
        return caller_identity_arn

    prefix, infix, suffix = match.groups()

    return f"{prefix}iam::{infix}:role/{suffix}"


def get_model_tarball_full_uri_from_base_uri(base_uri: str, training_job_name: str) -> str:
    return os.path.join(
        base_uri,
        training_job_name,
        "output",
        "model.tar.gz",
    )


def get_full_hyperparameters(
    base_hyperparameters: dict, job_name: str, model_artifacts_uri: str
) -> dict:

    bucket, key = parse_s3_url(model_artifacts_uri)
    return {
        **base_hyperparameters,
        "sagemaker_job_name": job_name,
        "model-artifact-bucket": bucket,
        "model-artifact-key": key,
    }


def get_hyperparameters_for_model_and_version(model_id: str, version: str) -> dict:
    return HYPERPARAMETER_MODEL_DICT[(model_id, version)]


def get_training_dataset_for_model_and_version(model_id: str, version: str) -> dict:
    return TRAINING_DATASET_MODEL_DICT[(model_id, version)]


def get_test_cache_bucket() -> str:
    bucket_name = Session().default_bucket()
    return bucket_name


def download_inference_assets():

    if not os.path.exists(TMP_DIRECTORY_PATH):
        os.makedirs(TMP_DIRECTORY_PATH)

    for asset, s3_key in TEST_ASSETS_SPECS.items():
        file_path = os.path.join(TMP_DIRECTORY_PATH, str(asset.value))
        if not os.path.exists(file_path):
            download_file(
                os.path.join(TMP_DIRECTORY_PATH, str(asset.value)),
                get_jumpstart_content_bucket(JUMPSTART_DEFAULT_REGION_NAME),
                s3_key,
                boto3.client("s3"),
            )


def get_tabular_data(data_filename: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    asset_file_path = os.path.join(TMP_DIRECTORY_PATH, data_filename)

    test_data = pd.read_csv(asset_file_path, header=None)
    label, features = test_data.iloc[:, :1], test_data.iloc[:, 1:]

    return label, features


def get_test_suite_id() -> str:
    return str(uuid.uuid4())
