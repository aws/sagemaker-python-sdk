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
"""Unit tests for workflow check_job_config."""
from __future__ import absolute_import

from sagemaker.mlops.workflow.check_job_config import CheckJobConfig


def test_check_job_config_init():
    config = CheckJobConfig(
        role="arn:aws:iam::123456789012:role/test-role",
        instance_count=1,
        instance_type="ml.m5.large"
    )
    assert config.role == "arn:aws:iam::123456789012:role/test-role"
    assert config.instance_count == 1
    assert config.instance_type == "ml.m5.large"
