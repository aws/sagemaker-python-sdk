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

import pytest
import os
import boto3
import sagemaker
import sagemaker_core.helper.session_helper as core_session

from botocore.config import Config
from sagemaker import Session

DEFAULT_REGION = "us-west-2"
CUSTOM_S3_OBJECT_KEY_PREFIX = "session-default-prefix"


@pytest.fixture(scope="session")
def sagemaker_session(
    sagemaker_client_config, sagemaker_runtime_config, boto_session, sagemaker_metrics_config
):
    """Isolated Session for the serve (ModelBuilder) integ tests.

    Overrides the repo-wide ``sagemaker_session`` fixture (defined in
    ``tests/conftest.py``) for everything under ``tests/integ/sagemaker/serve``.

    ModelBuilder mutates the global ``session.settings._local_download_dir`` to a
    temporary ``/tmp/sagemaker/model-builder/<uuid>`` path. When the shared
    session-scoped fixture is reused by other test modules, that temp dir gets
    cleaned up while the polluted setting lingers, breaking unrelated tests such
    as ``tests/integ/sagemaker/workflow/test_tuning_steps.py::test_tuning_multi_algos``
    (``ValueError: Inputted directory ... does not exist``). Scoping a dedicated
    session to the serve package keeps that mutation contained here.
    """
    sagemaker_client_config.setdefault("config", Config(retries=dict(max_attempts=10)))
    sagemaker_client = (
        boto_session.client("sagemaker", **sagemaker_client_config)
        if sagemaker_client_config
        else None
    )
    runtime_client = (
        boto_session.client("sagemaker-runtime", **sagemaker_runtime_config)
        if sagemaker_runtime_config
        else None
    )
    metrics_client = (
        boto_session.client("sagemaker-metrics", **sagemaker_metrics_config)
        if sagemaker_metrics_config
        else None
    )

    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        sagemaker_metrics_client=metrics_client,
        sagemaker_config={},
        default_bucket_prefix=CUSTOM_S3_OBJECT_KEY_PREFIX,
    )


@pytest.fixture(scope="module")
def mb_sagemaker_session():
    region = os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION
        region_manual_set = True
    else:
        region_manual_set = True

    boto_session = boto3.Session(region_name=os.environ["AWS_DEFAULT_REGION"])
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    yield sagemaker_session

    if region_manual_set and "AWS_DEFAULT_REGION" in os.environ:
        del os.environ["AWS_DEFAULT_REGION"]


@pytest.fixture(scope="module")
def mb_sagemaker_core_session():
    region = os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION
        region_manual_set = True
    else:
        region_manual_set = True

    boto_session = boto3.Session(region_name=os.environ["AWS_DEFAULT_REGION"])
    sagemaker_session = core_session.Session(boto_session=boto_session)

    yield sagemaker_session

    if region_manual_set and "AWS_DEFAULT_REGION" in os.environ:
        del os.environ["AWS_DEFAULT_REGION"]
