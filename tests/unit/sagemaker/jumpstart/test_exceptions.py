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

from sagemaker.jumpstart.exceptions import (
    get_wildcard_model_version_msg,
    get_old_model_version_msg,
)


def test_get_wildcard_model_version_msg():
    assert (
        "Using model 'mother_of_all_models' with wildcard version identifier '*'. "
        "Please consider pinning to version '1.2.3' to ensure stable results. "
        "Note that models may have different input/output signatures after a "
        "major version upgrade."
        == get_wildcard_model_version_msg("mother_of_all_models", "*", "1.2.3")
    )


def test_get_old_model_version_msg():
    assert (
        "Using model 'mother_of_all_models' with old version '1.0.0'. "
        "Please consider upgrading to version '1.2.3'. Note that models "
        "may have different input/output signatures after a major "
        "version upgrade." == get_old_model_version_msg("mother_of_all_models", "1.0.0", "1.2.3")
    )
