# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
from mock import MagicMock, patch

from tests.integ.timeout import (
    timeout,
    timeout_and_delete_endpoint_by_name,
    timeout_and_delete_model_with_transformer,
)


EXCEPTION_MESSAGE = "This Exception is expected and should not be swallowed by the timeout."


def test_timeout_fails_correctly_when_calling_test_throws_exception():
    with pytest.raises(ValueError) as exception:
        with timeout(minutes=1):
            raise ValueError(EXCEPTION_MESSAGE)
        assert EXCEPTION_MESSAGE in str(exception.value)


@patch("sagemaker.session")
def test_timeout_and_delete_endpoint_by_name_fails_when_calling_test_throws_exception(session):
    session.delete_endpoint = MagicMock()

    with pytest.raises(ValueError) as exception:
        with timeout_and_delete_endpoint_by_name(
            endpoint_name="fake-endpoint_name",
            sagemaker_session=session,
            minutes=1,
            sleep_between_cleanup_attempts=0,
        ):
            raise ValueError(EXCEPTION_MESSAGE)
        assert EXCEPTION_MESSAGE in str(exception.value)
    assert session.delete_endpoint.call_count == 3


@patch("sagemaker.session")
def test_timeout_and_delete_model_with_transformer_fails_when_calling_test_throws_exception(
    session
):
    transformer = MagicMock()

    with pytest.raises(ValueError) as exception:
        with timeout_and_delete_model_with_transformer(
            sagemaker_session=session,
            transformer=transformer,
            minutes=1,
            sleep_between_cleanup_attempts=0,
        ):
            raise ValueError(EXCEPTION_MESSAGE)
        assert EXCEPTION_MESSAGE in str(exception.value)
    assert transformer.delete_model.call_count == 3
