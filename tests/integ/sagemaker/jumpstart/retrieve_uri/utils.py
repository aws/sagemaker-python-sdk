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
from typing import Tuple
import boto3
import pandas as pd
import os

from tests.integ.sagemaker.jumpstart.retrieve_uri.constants import (
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


def get_model_tarball_full_uri_from_base_uri(base_uri: str, training_job_name: str) -> str:
    return "/".join(
        [
            base_uri,
            training_job_name,
            "output",
            "model.tar.gz",
        ]
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


def get_training_dataset_for_model_and_version(model_id: str, version: str) -> dict:
    return TRAINING_DATASET_MODEL_DICT[(model_id, version)]


def get_sm_session() -> Session:
    return Session(boto_session=boto3.Session(region_name=JUMPSTART_DEFAULT_REGION_NAME))


def get_test_artifact_bucket() -> str:
    bucket_name = get_sm_session().default_bucket()
    return bucket_name


def download_inference_assets():

    if not os.path.exists(TMP_DIRECTORY_PATH):
        os.makedirs(TMP_DIRECTORY_PATH)

    for asset, s3_key in TEST_ASSETS_SPECS.items():
        file_path = os.path.join(TMP_DIRECTORY_PATH, str(asset.value))
        if not os.path.exists(file_path):
            download_file(
                file_path,
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
