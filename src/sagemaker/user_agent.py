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

import pkg_resources
import platform
import sys

import boto3
import botocore

SDK_VERSION = pkg_resources.require('sagemaker')[0].version
OS_NAME = platform.system() or 'UnresolvedOS'
OS_VERSION = platform.release() or 'UnresolvedOSVersion'
PYTHON_VERSION = '{}.{}.{}'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)


def determine_prefix():
    prefix = 'AWS-SageMaker-Python-SDK/{} Python/{} {}/{} Boto3/{} Botocore/{}'\
        .format(SDK_VERSION, PYTHON_VERSION, OS_NAME, OS_VERSION, boto3.__version__, botocore.__version__)

    try:
        with open('/etc/opt/ml/sagemaker-notebook-instance-version.txt') as sagemaker_nbi_file:
            prefix = 'AWS-SageMaker-Notebook-Instance/{} {}'.format(sagemaker_nbi_file.read().strip(), prefix)
    except IOError:
        # This file isn't expected to always exist, and we DO want to silently ignore failures.
        pass

    return prefix


def prepend_user_agent(client):
    prefix = determine_prefix()

    if client._client_config.user_agent is None:
        client._client_config.user_agent = prefix
    else:
        client._client_config.user_agent = '{} {}'.format(prefix, client._client_config.user_agent)
