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

from sagemaker import instance_types_gpu_info

REGION = "us-west-2"
INVALID_REGION = "invalid-region"


def test_retrieve_success():
    data = instance_types_gpu_info.retrieve(REGION)

    assert len(data) > 0


def test_retrieve_throws():
    data = instance_types_gpu_info.retrieve(INVALID_REGION)

    assert len(data) == 0
