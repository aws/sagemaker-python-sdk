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
import pathlib


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "core: subset of integ tests that must run for every pull request"
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/integ/core" in str(item.fspath):
            item.add_marker(pytest.mark.core)
