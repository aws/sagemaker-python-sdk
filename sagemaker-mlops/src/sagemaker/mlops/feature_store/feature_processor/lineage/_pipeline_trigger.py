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
"""Contains class to store the Pipeline Schedule"""
from __future__ import absolute_import
import attr


@attr.s
class PipelineTrigger:
    """An evnet based trigger definition for FeatureProcessor Lineage.

    Attributes:
        trigger_name (str): Trigger Name.
        trigger_arn (str): The ARN of the Trigger.
        event_pattern (str): The event pattern. For more information, see Amazon EventBridge
            event patterns in the Amazon EventBridge User Guide.
        pipeline_name (str): The SageMaker Pipeline name that will be triggered.
        state (str): Specifies whether the trigger is enabled or disabled. Valid values are
            ENABLED and DISABLED.
    """

    trigger_name: str = attr.ib()
    trigger_arn: str = attr.ib()
    event_pattern: str = attr.ib()
    pipeline_name: str = attr.ib()
    state: str = attr.ib()
