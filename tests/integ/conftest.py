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
from __future__ import absolute_import

import os

import pytest


def create_sagemaker_local_network():
    """
    Docker has a known race condition which allows two parallel processes
    to create duplicated networks with the same name. This function
    creates the network sagemaker-local beforehand, avoiding this issue
    in CI.
    """
    os.system("docker network create sagemaker-local")


create_sagemaker_local_network()


@pytest.fixture(scope="session", params=["local", "ml.c4.xlarge"])
def instance_type(request):
    return request.param
