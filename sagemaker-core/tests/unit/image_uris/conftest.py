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

import os
import json
import pytest


# Get the path relative to this file's location
# conftest.py is in sagemaker-core/tests/unit/image_uris/
# config files are in sagemaker-core/src/sagemaker/core/image_uri_config/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(CURRENT_DIR, "../../../src/sagemaker/core/image_uri_config/")


def get_config(config_file_name):
    config_file_path = os.path.join(CONFIG_DIR, config_file_name)
    with open(config_file_path, "r") as config_file:
        return json.load(config_file)


@pytest.fixture(scope="module")
def load_config(request):
    config_file_name = request.param
    return get_config(config_file_name)


@pytest.fixture(scope="module")
def load_config_and_file_name(request):
    config_file_name = request.param
    return get_config(config_file_name), config_file_name


@pytest.fixture(scope="module")
def extract_versions_for_image_scope(load_config, request):
    scope_val = request.param
    return load_config[scope_val]["versions"]


@pytest.fixture
def sklearn_version():
    return "1.0-1"


@pytest.fixture
def graviton_xgboost_versions():
    return ["1.3-1", "1.5-1"]


@pytest.fixture
def graviton_xgboost_unsupported_versions():
    return ["1.0-1", "1.2-1"]


@pytest.fixture
def graviton_sklearn_versions():
    return ["1.0-1"]


@pytest.fixture
def graviton_sklearn_unsupported_versions():
    return ["0.20.0", "0.23-1"]
