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
from botocore.exceptions import ClientError
from mock import MagicMock, patch
import sagemaker

ACCOUNT_ID = "123"
REGION = "us-west-2"
DEFAULT_BUCKET_NAME = "sagemaker-{}-{}".format(REGION, ACCOUNT_ID)


@pytest.fixture()
def sagemaker_session():
    boto_mock = MagicMock(name="boto_session", region_name=REGION)
    boto_mock.client("sts").get_caller_identity.return_value = {"Account": ACCOUNT_ID}
    sagemaker_session = sagemaker.Session(boto_session=boto_mock)
    sagemaker_session.boto_session.resource("s3").Bucket().creation_date = None
    return sagemaker_session


def test_default_bucket_s3_create_call(sagemaker_session):
    error = ClientError(
        error_response={"Error": {"Code": "404", "Message": "Not Found"}},
        operation_name="foo",
    )
    sagemaker_session.boto_session.resource("s3").meta.client.head_bucket.side_effect = error
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


def test_default_bucket_s3_needs_access(sagemaker_session):
    with patch("logging.Logger.error") as mocked_error_log:
        with pytest.raises(ClientError):
            error = ClientError(
                error_response={"Error": {"Code": "403", "Message": "Forbidden"}},
                operation_name="foo",
            )
            sagemaker_session.boto_session.resource(
                "s3"
            ).meta.client.head_bucket.side_effect = error
            sagemaker_session.default_bucket()
            mocked_error_log.assert_called_once_with(
                "Bucket %s exists, but access is forbidden. Please try again after "
                "adding appropriate access.",
                DEFAULT_BUCKET_NAME,
            )
        assert sagemaker_session._default_bucket is None


def test_default_already_cached(sagemaker_session):
    existing_default = "mydefaultbucket"
    sagemaker_session._default_bucket = existing_default

    bucket_name = sagemaker_session.default_bucket()

    create_calls = sagemaker_session.boto_session.resource().create_bucket.mock_calls
    assert bucket_name == existing_default
    assert create_calls == []


def test_default_bucket_exists(sagemaker_session):
    sagemaker_session.boto_session.resource("s3").meta.client.head_bucket.return_value = {
        "ResponseMetadata": {"RequestId": "xxx", "HTTPStatusCode": 200, "RetryAttempts": 0}
    }

    bucket_name = sagemaker_session.default_bucket()
    assert bucket_name == DEFAULT_BUCKET_NAME


def test_concurrent_bucket_modification(sagemaker_session):
    message = "A conflicting conditional operation is currently in progress against this resource. Please try again"
    error = ClientError(
        error_response={"Error": {"Code": "OperationAborted", "Message": message}},
        operation_name="foo",
    )
    sagemaker_session.boto_session.resource().create_bucket.side_effect = error

    bucket_name = sagemaker_session.default_bucket()
    assert bucket_name == DEFAULT_BUCKET_NAME


def test_bucket_creation_client_error(sagemaker_session):
    error = ClientError(
        error_response={"Error": {"Code": "404", "Message": "Not Found"}},
        operation_name="foo",
    )
    sagemaker_session.boto_session.resource("s3").meta.client.head_bucket.side_effect = error
    with pytest.raises(ClientError):
        error = ClientError(
            error_response={"Error": {"Code": "SomethingWrong", "Message": "message"}},
            operation_name="foo",
        )
        sagemaker_session.boto_session.resource().create_bucket.side_effect = error

        sagemaker_session.default_bucket()
    assert sagemaker_session._default_bucket is None


def test_bucket_creation_other_error(sagemaker_session):
    error = ClientError(
        error_response={"Error": {"Code": "404", "Message": "Not Found"}},
        operation_name="foo",
    )
    sagemaker_session.boto_session.resource("s3").meta.client.head_bucket.side_effect = error
    with pytest.raises(RuntimeError):
        error = RuntimeError()
        sagemaker_session.boto_session.resource().create_bucket.side_effect = error

        sagemaker_session.default_bucket()
    assert sagemaker_session._default_bucket is None
