# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import pytest


@pytest.fixture(scope='module', params=["1.4", "1.4.1", "1.5", "1.5.0"])
def tf_version(request):
    return request.param


@pytest.fixture(scope='module', params=["0.12", "0.12.1", "1.0", "1.0.0"])
def mxnet_version(request):
    return request.param


@pytest.fixture(scope='module', params=["1.4.1", "1.5.0"])
def tf_full_version(request):
    return request.param


@pytest.fixture(scope='module', params=["0.12.1", "1.0.0"])
def mxnet_full_version(request):
    return request.param
