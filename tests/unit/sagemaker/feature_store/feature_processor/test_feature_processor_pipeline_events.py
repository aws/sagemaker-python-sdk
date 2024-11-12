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

from sagemaker.feature_store.feature_processor import (
    FeatureProcessorPipelineEvents,
    FeatureProcessorPipelineExecutionStatus,
)


def test_feature_processor_pipeline_events():
    fe_pipeline_events = FeatureProcessorPipelineEvents(
        pipeline_name="pipeline_name",
        pipeline_execution_status=[FeatureProcessorPipelineExecutionStatus.EXECUTING],
    )
    assert fe_pipeline_events.pipeline_name == "pipeline_name"
    assert fe_pipeline_events.pipeline_execution_status == [
        FeatureProcessorPipelineExecutionStatus.EXECUTING
    ]
