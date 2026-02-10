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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from sagemaker.mlops.feature_store.feature_processor.lineage._pipeline_trigger import PipelineTrigger


def test_pipeline_trigger():

    trigger = PipelineTrigger(
        trigger_name="test_trigger",
        trigger_arn="test_arn",
        event_pattern="test_pattern",
        pipeline_name="test_pipeline",
        state="Enabled",
    )

    assert trigger.trigger_name == "test_trigger"
    assert trigger.trigger_arn == "test_arn"
    assert trigger.event_pattern == "test_pattern"
    assert trigger.pipeline_name == "test_pipeline"
    assert trigger.state == "Enabled"
