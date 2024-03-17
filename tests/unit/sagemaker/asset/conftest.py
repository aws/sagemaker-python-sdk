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

import time
import json
from boto3 import Session as BotoSession
from sagemaker import get_execution_role
from sagemaker.session import Session
from sagemaker.asset import AssetManager
from sagemaker.model import ModelPackageGroup
from sagemaker.feature_store.feature_group import FeatureGroup
from botocore.exceptions import ClientError
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)
from unittest import SkipTest
import pytest
from mock import MagicMock, Mock

@pytest.fixture(scope="module")
def domain_id():
    return "dzd_123"


@pytest.fixture(scope="module")
def project_id():
    return "pj_123"


@pytest.fixture(scope="module")
def environment_id():
    return "env_123"


@pytest.fixture(scope="module")
def sagemaker_client():
    return Mock(name="sagemaker_client")


@pytest.fixture(scope="module")
def datazone_client():
    return Mock(name="datazone_client")


@pytest.fixture(scope="module")
def lf_client():
    return Mock(name="lakeformation_client")
