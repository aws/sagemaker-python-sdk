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
from botocore.exceptions import ClientError
from mock import Mock
import sagemaker

ACCOUNT_ID = "123"
REGION = "us-west-2"
DEFAULT_BUCKET_NAME = "sagemaker-{}-{}".format(REGION, ACCOUNT_ID)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    client_mock = Mock()
    client_mock.get_caller_identity.return_value = {
        "UserId": "mock_user_id",
        "Account": "012345678910",
        "Arn": "arn:aws:iam::012345678910:user/mock-user",
    }
    boto_mock.client.return_value = client_mock
    boto_mock.client("sts").get_caller_identity.return_value = {"Account": ACCOUNT_ID}
    ims = sagemaker.Session(boto_session=boto_mock)
    return ims


def test_default_bucket_s3_create_call(sagemaker_session):
    bucket_name = sagemaker_session.default_bucket()

    create_calls = sagemaker_session.boto_session.resource().create_bucket.mock_calls
    _1, _2, create_kwargs = create_calls[0]
    assert bucket_name == DEFAULT_BUCKET_NAME
    assert len(create_calls) == 1
    assert create_kwargs == {
        "CreateBucketConfiguration": {"LocationConstraint": "us-west-2"},
        "Bucket": bucket_name,
    }
    assert sagemaker_session._default_bucket == bucket_name


def test_default_already_cached(sagemaker_session):
    existing_default = "mydefaultbucket"
    sagemaker_session._default_bucket = existing_default

    before_create_calls = sagemaker_session.boto_session.resource().create_bucket.mock_calls

    bucket_name = sagemaker_session.default_bucket()

    after_create_calls = sagemaker_session.boto_session.resource().create_bucket.mock_calls
    assert bucket_name == existing_default
    assert before_create_calls == after_create_calls


def test_default_bucket_exists(sagemaker_session):
    error = ClientError(
        error_response={"Error": {"Code": "BucketAlreadyOwnedByYou", "Message": "message"}},
        operation_name="foo",
    )
    sagemaker_session.boto_session.resource().create_bucket.side_effect = error

    bucket_name = sagemaker_session.default_bucket()
    assert bucket_name == DEFAULT_BUCKET_NAME


def test_concurrent_bucket_modification(sagemaker_session):
    message = "A conflicting conditional operation is currently in progress against this resource. Please try again"
    error = ClientError(
        error_response={"Error": {"Code": "BucketAlreadyOwnedByYou", "Message": message}},
        operation_name="foo",
    )
    sagemaker_session.boto_session.resource().create_bucket.side_effect = error

    bucket_name = sagemaker_session.default_bucket()
    assert bucket_name == DEFAULT_BUCKET_NAME


def test_bucket_creation_client_error():
    with pytest.raises(ClientError):
        boto_mock = Mock(name="boto_session", region_name=REGION)
        client_mock = Mock()
        client_mock.get_caller_identity.return_value = {
            "UserId": "mock_user_id",
            "Account": "012345678910",
            "Arn": "arn:aws:iam::012345678910:user/mock-user",
        }
        boto_mock.client.return_value = client_mock
        boto_mock.client("sts").get_caller_identity.return_value = {"Account": ACCOUNT_ID}

        error = ClientError(
            error_response={"Error": {"Code": "SomethingWrong", "Message": "message"}},
            operation_name="foo",
        )
        boto_mock.resource().create_bucket.side_effect = error

        session = sagemaker.Session(boto_session=boto_mock)
        assert session._default_bucket is None


def test_bucket_creation_other_error():
    with pytest.raises(RuntimeError):
        boto_mock = Mock(name="boto_session", region_name=REGION)
        client_mock = Mock()
        client_mock.get_caller_identity.return_value = {
            "UserId": "mock_user_id",
            "Account": "012345678910",
            "Arn": "arn:aws:iam::012345678910:user/mock-user",
        }
        boto_mock.client.return_value = client_mock
        boto_mock.client("sts").get_caller_identity.return_value = {"Account": ACCOUNT_ID}

        error = RuntimeError()
        boto_mock.resource().create_bucket.side_effect = error

        session = sagemaker.Session(boto_session=boto_mock)
        assert session._default_bucket is None
