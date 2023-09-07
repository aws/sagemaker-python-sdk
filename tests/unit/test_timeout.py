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
"""
This class tests the timeout.py class in the integration tests.
This is to prevent regressions that cause the timeout function to hide failed tests.
"""
from __future__ import absolute_import

import time

import pytest
from mock import Mock, patch, call
import stopit

from botocore.exceptions import ClientError
from sagemaker.session_settings import SessionSettings

from tests.integ.timeout import (
    timeout,
    timeout_and_delete_endpoint_by_name,
    timeout_and_delete_model_with_transformer,
)


BOTO_SESSION_NAME = "boto_session_name"
SAGEMAKER_SESSION_NAME = "sagemaker_session_name"
DEFAULT_BUCKET_NAME = "default_bucket_name"
TRANSFORMER_NAME = "transformer.name"
REGION = "us-west-2"
BUCKET_NAME = "bucket-name"
ENDPOINT_NAME = "endpoint_name"

EXCEPTION_MESSAGE = "This Exception is expected and should not be swallowed by the timeout."
SHORT_TIMEOUT_TO_FORCE_TIMEOUT_TO_OCCUR = 0.001
LONG_DURATION_TO_EXCEED_TIMEOUT = 0.002
LONG_TIMEOUT_THAT_WILL_NEVER_BE_EXCEEDED = 10
DURATION_TO_SLEEP_TO_ALLOW_BACKGROUND_THREAD_TO_COMPLETE = 0.2
DURATION_TO_SLEEP = 0.01


@pytest.fixture()
def session():
    boto_mock = Mock(name=BOTO_SESSION_NAME, region_name=REGION)
    sms = Mock(
        name=SAGEMAKER_SESSION_NAME,
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=True,
        settings=SessionSettings(),
    )
    sms.default_bucket = Mock(name=DEFAULT_BUCKET_NAME, return_value=BUCKET_NAME)
    # For tests which doesn't verify config file injection, operate with empty config
    sms.sagemaker_config = {}
    return sms


@pytest.fixture()
def transformer():
    return Mock(name=TRANSFORMER_NAME, region_name=REGION)


def test_timeout_fails_correctly_when_method_throws_exception():
    with pytest.raises(ValueError) as exception:
        with timeout(hours=0, minutes=0, seconds=LONG_TIMEOUT_THAT_WILL_NEVER_BE_EXCEEDED):
            raise ValueError(EXCEPTION_MESSAGE)
        assert EXCEPTION_MESSAGE in str(exception.value)


def test_timeout_does_not_throw_exception_when_method_ends_gracefully():
    with timeout(hours=0, minutes=0, seconds=LONG_TIMEOUT_THAT_WILL_NEVER_BE_EXCEEDED):
        pass


@patch("tests.integ.timeout._show_logs", return_value=None, autospec=True)
@patch("tests.integ.timeout._cleanup_logs", return_value=None, autospec=True)
@patch(
    "tests.integ.timeout._delete_schedules_associated_with_endpoint",
    return_value=None,
    autospec=True,
)
def test_timeout_and_delete_endpoint_by_name_fails_when_method_throws_exception(
    _show_logs, _cleanup_logs, _delete_schedules_associated_with_endpoint, session
):
    with pytest.raises(ValueError) as exception:
        with timeout_and_delete_endpoint_by_name(
            endpoint_name=ENDPOINT_NAME,
            sagemaker_session=session,
            hours=0,
            minutes=0,
            seconds=LONG_TIMEOUT_THAT_WILL_NEVER_BE_EXCEEDED,
            sleep_between_cleanup_attempts=0,
        ):
            raise ValueError(EXCEPTION_MESSAGE)
        assert EXCEPTION_MESSAGE in str(exception.value)
    assert session.delete_endpoint.call_count == 1


@patch("tests.integ.timeout._show_logs", return_value=None, autospec=True)
@patch("tests.integ.timeout._cleanup_logs", return_value=None, autospec=True)
@patch(
    "tests.integ.timeout._delete_schedules_associated_with_endpoint",
    return_value=None,
    autospec=True,
)
def test_timeout_and_delete_endpoint_by_name_throws_timeout_exception_when_method_times_out(
    _show_logs, _cleanup_logs, _delete_schedules_associated_with_endpoint, session
):
    with pytest.raises(stopit.utils.TimeoutException):
        with timeout_and_delete_endpoint_by_name(
            endpoint_name=ENDPOINT_NAME,
            sagemaker_session=session,
            hours=0,
            minutes=0,
            seconds=SHORT_TIMEOUT_TO_FORCE_TIMEOUT_TO_OCCUR,
            sleep_between_cleanup_attempts=0,
        ):
            time.sleep(LONG_DURATION_TO_EXCEED_TIMEOUT)


@patch("tests.integ.timeout._show_logs", return_value=None, autospec=True)
@patch("tests.integ.timeout._cleanup_logs", return_value=None, autospec=True)
@patch(
    "tests.integ.timeout._delete_schedules_associated_with_endpoint",
    return_value=None,
    autospec=True,
)
def test_timeout_and_delete_endpoint_by_name_does_not_throw_exception_when_method_ends_gracefully(
    _show_logs, _cleanup_logs, _delete_schedules_associated_with_endpoint, session
):
    with timeout_and_delete_endpoint_by_name(
        endpoint_name=ENDPOINT_NAME,
        sagemaker_session=session,
        hours=0,
        minutes=0,
        seconds=LONG_TIMEOUT_THAT_WILL_NEVER_BE_EXCEEDED,
        sleep_between_cleanup_attempts=0,
    ):
        pass
    assert session.delete_endpoint.call_count == 1


@patch("tests.integ.timeout._show_logs", return_value=None, autospec=True)
@patch("tests.integ.timeout._cleanup_logs", return_value=None, autospec=True)
@patch(
    "tests.integ.timeout._delete_schedules_associated_with_endpoint",
    return_value=None,
    autospec=True,
)
def test_timeout_and_delete_endpoint_by_name_retries_resource_deletion_on_failure(
    _show_logs, _cleanup_logs, _delete_schedules_associated_with_endpoint, session
):
    session.delete_endpoint = Mock(
        side_effect=ClientError(
            error_response={"Error": {"Code": 403, "Message": "ValidationException"}},
            operation_name="Unit Test",
        )
    )

    with timeout_and_delete_endpoint_by_name(
        endpoint_name=ENDPOINT_NAME,
        sagemaker_session=session,
        hours=0,
        minutes=0,
        seconds=LONG_TIMEOUT_THAT_WILL_NEVER_BE_EXCEEDED,
        sleep_between_cleanup_attempts=0,
    ):
        pass
    assert session.delete_endpoint.call_count == 3


@patch("tests.integ.timeout._show_logs", return_value=None, autospec=True)
@patch("tests.integ.timeout._cleanup_logs", return_value=None, autospec=True)
@patch(
    "tests.integ.timeout._delete_schedules_associated_with_endpoint",
    return_value=None,
    autospec=True,
)
@patch("tests.integ.timeout.sleep", return_value=None)
def test_timeout_and_delete_endpoint_by_name_retries_resource_deletion_on_failure_with_exp_sleep(
    mock_sleep, _show_logs, _cleanup_logs, _delete_schedules_associated_with_endpoint, session
):
    session.delete_endpoint = Mock(
        side_effect=ClientError(
            error_response={"Error": {"Code": 403, "Message": "ValidationException"}},
            operation_name="Unit Test",
        )
    )

    with timeout_and_delete_endpoint_by_name(
        endpoint_name=ENDPOINT_NAME,
        sagemaker_session=session,
        hours=0,
        minutes=0,
        seconds=LONG_TIMEOUT_THAT_WILL_NEVER_BE_EXCEEDED,
        sleep_between_cleanup_attempts=DURATION_TO_SLEEP,
        exponential_sleep=True,
    ):
        pass
    assert session.delete_endpoint.call_count == 3
    assert mock_sleep.call_count == 3
    assert mock_sleep.mock_calls == [call(0.01), call(0.02), call(0.03)]


@patch("tests.integ.timeout._show_logs", return_value=None, autospec=True)
@patch("tests.integ.timeout._cleanup_logs", return_value=None, autospec=True)
@patch(
    "tests.integ.timeout._delete_schedules_associated_with_endpoint",
    return_value=None,
    autospec=True,
)
def test_timeout_and_delete_model_with_transformer_fails_when_method_throws_exception(
    _show_logs, _cleanup_logs, _delete_schedules_associated_with_endpoint, session, transformer
):
    with pytest.raises(ValueError) as exception:
        with timeout_and_delete_model_with_transformer(
            sagemaker_session=session,
            transformer=transformer,
            hours=0,
            minutes=1,
            sleep_between_cleanup_attempts=0,
        ):
            raise ValueError(EXCEPTION_MESSAGE)
        assert EXCEPTION_MESSAGE in str(exception.value)
    assert transformer.delete_model.call_count == 1


@patch("tests.integ.timeout._show_logs", return_value=None, autospec=True)
@patch("tests.integ.timeout._cleanup_logs", return_value=None, autospec=True)
@patch(
    "tests.integ.timeout._delete_schedules_associated_with_endpoint",
    return_value=None,
    autospec=True,
)
def test_timeout_and_delete_model_with_transformer_throws_timeout_exception_when_method_times_out(
    _show_logs, _cleanup_logs, _delete_schedules_associated_with_endpoint, session, transformer
):
    with pytest.raises(stopit.utils.TimeoutException):
        with timeout_and_delete_model_with_transformer(
            sagemaker_session=session,
            transformer=transformer,
            hours=0,
            minutes=0,
            seconds=SHORT_TIMEOUT_TO_FORCE_TIMEOUT_TO_OCCUR,
            sleep_between_cleanup_attempts=0,
        ):
            time.sleep(LONG_DURATION_TO_EXCEED_TIMEOUT)


@patch("tests.integ.timeout._show_logs", return_value=None, autospec=True)
@patch("tests.integ.timeout._cleanup_logs", return_value=None, autospec=True)
@patch(
    "tests.integ.timeout._delete_schedules_associated_with_endpoint",
    return_value=None,
    autospec=True,
)
def test_timeout_and_delete_model_with_transformer_does_not_throw_when_method_ends_gracefully(
    _show_logs, _cleanup_logs, _delete_schedules_associated_with_endpoint, session, transformer
):
    with timeout_and_delete_model_with_transformer(
        sagemaker_session=session,
        transformer=transformer,
        hours=0,
        minutes=0,
        seconds=LONG_TIMEOUT_THAT_WILL_NEVER_BE_EXCEEDED,
        sleep_between_cleanup_attempts=0,
    ):
        pass
    assert transformer.delete_model.call_count == 1


@patch("tests.integ.timeout._show_logs", return_value=None, autospec=True)
@patch("tests.integ.timeout._cleanup_logs", return_value=None, autospec=True)
@patch(
    "tests.integ.timeout._delete_schedules_associated_with_endpoint",
    return_value=None,
    autospec=True,
)
def test_timeout_and_delete_model_with_transformer_retries_resource_deletion_on_failure(
    _show_logs, _cleanup_logs, _delete_schedules_associated_with_endpoint, session, transformer
):
    transformer.delete_model = Mock(
        side_effect=ClientError(
            error_response={"Error": {"Code": 403, "Message": "ValidationException"}},
            operation_name="Unit Test",
        )
    )

    with timeout_and_delete_model_with_transformer(
        sagemaker_session=session,
        transformer=transformer,
        hours=0,
        minutes=0,
        seconds=LONG_TIMEOUT_THAT_WILL_NEVER_BE_EXCEEDED,
        sleep_between_cleanup_attempts=0,
    ):
        pass
    assert transformer.delete_model.call_count == 3
