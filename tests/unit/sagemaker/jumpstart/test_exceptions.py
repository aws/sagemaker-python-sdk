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

from sagemaker.jumpstart.exceptions import (
    get_wildcard_model_version_msg,
    get_old_model_version_msg,
    get_proprietary_model_subscription_error,
    MarketplaceModelSubscriptionError,
)


def test_get_wildcard_model_version_msg():
    assert (
        "Using model 'mother_of_all_models' with wildcard version identifier '*'. "
        "You can pin to version '1.2.3' for more stable results. "
        "Note that models may have different input/output signatures after a "
        "major version upgrade."
        == get_wildcard_model_version_msg("mother_of_all_models", "*", "1.2.3")
    )


def test_get_old_model_version_msg():
    assert (
        "Using model 'mother_of_all_models' with version '1.0.0'. "
        "You can upgrade to version '1.2.3' to get the latest model specifications. "
        "Note that models may have different input/output signatures after a major "
        "version upgrade." == get_old_model_version_msg("mother_of_all_models", "1.0.0", "1.2.3")
    )


def test_get_marketplace_subscription_error():
    error = ClientError(
        error_response={
            "Error": {
                "Code": "ValidationException",
                "Message": "Caller is not subscribed to the Marketplace listing.",
            },
        },
        operation_name="mock-operation",
    )
    with pytest.raises(MarketplaceModelSubscriptionError):
        get_proprietary_model_subscription_error(error, subscription_link="mock-link")

    error = ClientError(
        error_response={
            "Error": {
                "Code": "UnknownException",
                "Message": "Unknown error raised.",
            },
        },
        operation_name="mock-operation",
    )

    try:
        get_proprietary_model_subscription_error(error, subscription_link="mock-link")
    except MarketplaceModelSubscriptionError:
        pytest.fail("MarketplaceModelSubscriptionError should not be raised for unknown error.")
