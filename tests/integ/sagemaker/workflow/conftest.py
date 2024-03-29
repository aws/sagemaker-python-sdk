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
from botocore.config import Config

from tests.integ import DATA_DIR
from sagemaker import Session, get_execution_role

CUSTOM_S3_OBJECT_KEY_PREFIX = "session-default-prefix"


# Create a sagemaker_session in workflow scope to prevent race condition
# with other tests. Some other tests may change the session `settings`.
@pytest.fixture(scope="module")
def sagemaker_session_for_pipeline(
    sagemaker_client_config,
    boto_session,
):
    sagemaker_client_config.setdefault("config", Config(retries=dict(max_attempts=10)))
    sagemaker_client = (
        boto_session.client("sagemaker", **sagemaker_client_config)
        if sagemaker_client_config
        else None
    )

    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_config={},
        default_bucket_prefix=CUSTOM_S3_OBJECT_KEY_PREFIX,
    )


@pytest.fixture(scope="module")
def smclient(sagemaker_session):
    return sagemaker_session.boto_session.client("sagemaker")


@pytest.fixture(scope="module")
def role(sagemaker_session_for_pipeline):
    return get_execution_role(sagemaker_session_for_pipeline)


@pytest.fixture(scope="module")
def region_name(sagemaker_session_for_pipeline):
    return sagemaker_session_for_pipeline.boto_session.region_name


@pytest.fixture(scope="module")
def script_dir():
    return os.path.join(DATA_DIR, "sklearn_processing")
