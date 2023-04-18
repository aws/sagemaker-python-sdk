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

from mock.mock import Mock
import pytest

from sagemaker.session_settings import SessionSettings

REGION_NAME = "us-west-2"
BUCKET_NAME = "some-bucket-name"


@pytest.fixture(scope="module")
def session():
    boto_mock = Mock(region_name=REGION_NAME)
    sms = Mock(
        boto_session=boto_mock,
        boto_region_name=REGION_NAME,
        config=None,
        settings=SessionSettings(),
    )
    sms.default_bucket = Mock(return_value=BUCKET_NAME)
    # For tests which doesn't verify config file injection, operate with empty config
    sms.sagemaker_config = {}
    return sms
