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
import functools
import json

import random
import time
import uuid
from typing import Any, Dict, List, Tuple
import boto3
import pandas as pd
import os

from botocore.config import Config
from botocore.exceptions import ClientError
import pytest


from tests.integ.sagemaker.jumpstart.constants import (
    TEST_ASSETS_SPECS,
    TMP_DIRECTORY_PATH,
    TRAINING_DATASET_MODEL_DICT,
    ContentType,
)
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
from sagemaker.jumpstart.hub.hub import Hub

from sagemaker.session import Session


def get_test_artifact_bucket() -> str:
    bucket_name = get_sm_session().default_bucket()
    return bucket_name


def get_test_suite_id() -> str:
    return str(uuid.uuid4())


def get_sm_session() -> Session:
    return Session(boto_session=boto3.Session(region_name=JUMPSTART_DEFAULT_REGION_NAME))


<<<<<<< HEAD
=======
def get_sm_session_with_override() -> Session:
    # [TODO]: Remove service endpoint override before GA
    # boto3.set_stream_logger(name='botocore', level=logging.DEBUG)
    boto_session = boto3.Session(region_name="us-west-2")
    sagemaker = boto3.client(
        service_name="sagemaker",
        endpoint_url="https://sagemaker.gamma.us-west-2.ml-platform.aws.a2z.com",
    )
    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker,
    )


>>>>>>> ff3eae05 (feat: Adding Bedrock Store model support for HubService (#1539))
def get_training_dataset_for_model_and_version(model_id: str, version: str) -> dict:
    return TRAINING_DATASET_MODEL_DICT[(model_id, version)]


def x_fail_if_ice(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "CapacityError" in str(e):
                pytest.xfail(str(e))
            raise

    return wrapper


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


def download_file(local_download_path, s3_bucket, s3_key, s3_client) -> None:
    s3_client.download_file(s3_bucket, s3_key, local_download_path)


def get_public_hub_model_arn(hub: Hub, model_id: str) -> str:
    filter_value = f"model_id == {model_id}"
    response = hub.list_sagemaker_public_hub_models(filter=filter_value)

    models = response["hub_content_summaries"]

    return models[0]["hub_content_arn"]


def with_exponential_backoff(max_retries=5, initial_delay=1, max_delay=60):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except ClientError as e:
                    if retries >= max_retries or e.response["Error"]["Code"] not in [
                        "ThrottlingException",
                        "TooManyRequestsException",
                    ]:
                        raise
                    delay = min(initial_delay * (2**retries) + random.random(), max_delay)
                    print(
                        f"Retrying {func.__name__} in {delay:.2f} seconds... (Attempt {retries + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                    retries += 1

        return wrapper

    return decorator


class EndpointInvoker:
    def __init__(
        self,
        endpoint_name: str,
        region: str = JUMPSTART_DEFAULT_REGION_NAME,
        boto_config: Config = Config(retries={"max_attempts": 10, "mode": "standard"}),
    ) -> None:
        self.endpoint_name = endpoint_name
        self.region = region
        self.config = boto_config
        self.sagemaker_runtime_client = self.get_sagemaker_runtime_client()

    def _invoke_endpoint(
        self,
        body: Any,
        content_type: ContentType,
    ) -> Dict[str, Any]:
        response = self.sagemaker_runtime_client.invoke_endpoint(
            EndpointName=self.endpoint_name, ContentType=content_type.value, Body=body
        )
        return json.loads(response["Body"].read())

    def invoke_tabular_endpoint(self, data: pd.DataFrame) -> Dict[str, Any]:
        return self._invoke_endpoint(
            body=data.to_csv(header=False, index=False).encode("utf-8"),
            content_type=ContentType.TEXT_CSV,
        )

    def invoke_spc_endpoint(self, text: List[str]) -> Dict[str, Any]:
        return self._invoke_endpoint(
            body=json.dumps(text).encode("utf-8"),
            content_type=ContentType.LIST_TEXT,
        )

    def get_sagemaker_runtime_client(self) -> boto3.client:
        return boto3.client(
            service_name="runtime.sagemaker", config=self.config, region_name=self.region
        )
