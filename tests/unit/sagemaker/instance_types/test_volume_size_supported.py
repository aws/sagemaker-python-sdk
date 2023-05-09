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

from sagemaker import instance_types


def test_volume_size_supported():
    instances_that_support_volume_size = [
        "ml.inf1.xlarge",
        "ml.inf1.2xlarge",
        "ml.inf1.6xlarge",
        "ml.inf1.24xlarge",
        "ml.inf2.xlarge",
        "ml.inf2.8xlarge",
        "ml.inf2.24xlarge",
        "ml.inf2.48xlarge",
        "ml.m5.large",
        "ml.m5.xlarge",
        "ml.m5.2xlarge",
        "ml.m5.4xlarge",
        "ml.m5.8xlarge",
        "ml.m5.12xlarge",
        "ml.m5.16xlarge",
        "ml.m5.24xlarge",
        "ml.m5.metal",
        "ml.c5.large",
        "ml.c5.xlarge",
        "ml.c5.2xlarge",
        "ml.c5.4xlarge",
        "ml.c5.9xlarge",
        "ml.c5.12xlarge",
        "ml.c5.18xlarge",
        "ml.c5.24xlarge",
        "ml.c5.metal",
        "ml.p3.2xlarge",
        "ml.p3.8xlarge",
        "ml.p3.16xlarge",
    ]

    for instance in instances_that_support_volume_size:
        assert instance_types.volume_size_supported(instance)


def test_volume_size_not_supported():
    instances_that_dont_support_volume_size = [
        "ml.p4d.xlarge",
        "ml.p4d.2xlarge",
        "ml.p4d.4xlarge",
        "ml.p4d.8xlarge",
        "ml.p4de.xlarge",
        "ml.p4de.2xlarge",
        "ml.p4de.4xlarge",
        "ml.p4de.8xlarge",
        "ml.g4dn.xlarge",
        "ml.g4dn.2xlarge",
        "ml.g4dn.4xlarge",
        "ml.g4dn.8xlarge",
        "ml.g5.xlarge",
        "ml.g5.2xlarge",
        "ml.g5.4xlarge",
        "ml.g5.8xlarge",
    ]

    for instance in instances_that_dont_support_volume_size:
        assert not instance_types.volume_size_supported(instance)


def test_volume_size_badly_formatted():
    with pytest.raises(ValueError):
        instance_types.volume_size_supported("blah")

    with pytest.raises(ValueError):
        instance_types.volume_size_supported(float("inf"))

    with pytest.raises(ValueError):
        instance_types.volume_size_supported("ml.p2")

    with pytest.raises(ValueError):
        instance_types.volume_size_supported({})
