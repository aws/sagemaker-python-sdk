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
"""Contains classes for EventBridge Schedule management for a feature processor."""
from __future__ import absolute_import

import json
import logging
import re
from typing import Dict, List, Tuple, Optional, Any
import attr
from botocore.exceptions import ClientError
from botocore.paginate import PageIterator
from sagemaker import Session
from sagemaker.feature_store.feature_processor._feature_processor_pipeline_events import (
    FeatureProcessorPipelineEvents,
)
from sagemaker.feature_store.feature_processor._constants import (
    RESOURCE_NOT_FOUND_EXCEPTION,
    PIPELINE_ARN_REGEX_PATTERN,
    BASE_EVENT_PATTERN,
)
from sagemaker.feature_store.feature_processor._enums import (
    FeatureProcessorPipelineExecutionStatus,
)
from sagemaker.utils import TagsDict

logger = logging.getLogger("sagemaker")


@attr.s
class EventBridgeRuleHelper:
    """Contains helper methods for managing EventBridge rules for a feature processor."""

    sagemaker_session: Session = attr.ib()
    event_bridge_rule_client = attr.ib()

    def put_rule(
        self,
        source_pipeline_events: List[FeatureProcessorPipelineEvents],
        target_pipeline: str,
        event_pattern: str,
        state: str,
    ) -> str:
        """Creates an EventBridge Rule for a given target pipeline.

        Args:
            source_pipeline_events: The list of pipeline events that trigger the EventBridge Rule.
            target_pipeline: The name of the pipeline that is triggered by the EventBridge Rule.
            event_pattern: The EventBridge EventPattern that triggers the EventBridge Rule.
                If specified, will override source_pipeline_events.
            state: Indicates whether the rule is enabled or disabled.

        Returns:
            The Amazon Resource Name (ARN) of the rule.
        """
        self._validate_feature_processor_pipeline_events(source_pipeline_events)
        rule_name = target_pipeline
        _event_patterns = (
            event_pattern
            or self._generate_event_pattern_from_feature_processor_pipeline_events(
                source_pipeline_events
            )
        )
        rule_arn = self.event_bridge_rule_client.put_rule(
            Name=rule_name, EventPattern=_event_patterns, State=state
        )["RuleArn"]
        return rule_arn

    def put_target(
        self,
        rule_name: str,
        target_pipeline: str,
        target_pipeline_parameters: Dict[str, str],
        role_arn: str,
    ) -> None:
        """Attach target pipeline to an event based trigger.

        Args:
            rule_name: The name of the EventBridge Rule.
            target_pipeline: The name of the pipeline that is triggered by the EventBridge Rule.
            target_pipeline_parameters: The list of parameters to start execution of a pipeline.
            role_arn: The Amazon Resource Name (ARN) of the IAM role associated with the rule.
        """
        target_pipeline_arn_and_name = self._generate_pipeline_arn_and_name(target_pipeline)
        target_pipeline_name = target_pipeline_arn_and_name["pipeline_name"]
        target_pipeline_arn = target_pipeline_arn_and_name["pipeline_arn"]
        target_request_dict = {
            "Id": target_pipeline_name,
            "Arn": target_pipeline_arn,
            "RoleArn": role_arn,
        }
        if target_pipeline_parameters:
            target_request_dict["SageMakerPipelineParameters"] = {
                "PipelineParameterList": target_pipeline_parameters
            }
        put_targets_response = self.event_bridge_rule_client.put_targets(
            Rule=rule_name,
            Targets=[target_request_dict],
        )
        if put_targets_response["FailedEntryCount"] != 0:
            error_msg = put_targets_response["FailedEntries"][0]["ErrorMessage"]
            raise Exception(f"Failed to add target pipeline to rule. Failure reason: {error_msg}")

    def delete_rule(self, rule_name: str) -> None:
        """Deletes an EventBridge Rule of a given pipeline if there is one.

        Args:
            rule_name: The name of the EventBridge Rule.
        """
        self.event_bridge_rule_client.delete_rule(Name=rule_name)

    def remove_targets(self, rule_name: str, ids: List[str]) -> None:
        """Deletes an EventBridge Targets of a given rule if there is one.

        Args:
            rule_name: The name of the EventBridge Rule.
            ids: The ids of the EventBridge Target.
        """
        self.event_bridge_rule_client.remove_targets(Rule=rule_name, Ids=ids)

    def list_targets_by_rule(self, rule_name: str) -> PageIterator:
        """List EventBridge Targets of a given rule.

        Args:
            rule_name: The name of the EventBridge Rule.

        Returns:
            The page iterator of list_targets_by_rule call.
        """
        return self.event_bridge_rule_client.get_paginator("list_targets_by_rule").paginate(
            Rule=rule_name
        )

    def describe_rule(self, rule_name: str) -> Optional[Dict[str, Any]]:
        """Describe the EventBridge Rule ARN corresponding to a sagemaker pipeline

        Args:
            rule_name: The name of the EventBridge Rule.
        Returns:
            Optional[Dict[str, str]] : Describe EventBridge Rule response if exists.
        """
        try:
            event_bridge_rule_response = self.event_bridge_rule_client.describe_rule(Name=rule_name)
            return event_bridge_rule_response
        except ClientError as e:
            if RESOURCE_NOT_FOUND_EXCEPTION == e.response["Error"]["Code"]:
                logger.info("No EventBridge Rule found for pipeline %s.", rule_name)
                return None
            raise e

    def enable_rule(self, rule_name: str) -> None:
        """Enables an EventBridge Rule of a given pipeline if there is one.

        Args:
            rule_name: The name of the EventBridge Rule.
        """
        self.event_bridge_rule_client.enable_rule(Name=rule_name)
        logger.info("Enabled EventBridge Rule for pipeline %s.", rule_name)

    def disable_rule(self, rule_name: str) -> None:
        """Disables an EventBridge Rule of a given pipeline if there is one.

        Args:
            rule_name: The name of the EventBridge Rule.
        """
        self.event_bridge_rule_client.disable_rule(Name=rule_name)
        logger.info("Disabled EventBridge Rule for pipeline %s.", rule_name)

    def add_tags(self, rule_arn: str, tags: List[TagsDict]) -> None:
        """Adds tags to the EventBridge Rule.

        Args:
            rule_arn: The ARN of the EventBridge Rule.
            tags: List of tags to be added.
        """
        self.event_bridge_rule_client.tag_resource(ResourceARN=rule_arn, Tags=tags)

    def _generate_event_pattern_from_feature_processor_pipeline_events(
        self, pipeline_events: List[FeatureProcessorPipelineEvents]
    ) -> str:
        """Generates the event pattern json string from the pipeline events.

        Args:
            pipeline_events: List of pipeline events.
        Returns:
            str: The event pattern json string.

        Raises:
            ValueError: If pipeline events contain duplicate pipeline names.
        """

        result_event_pattern = {
            "detail-type": ["SageMaker Model Building Pipeline Execution Status Change"],
        }
        filters = []
        desired_status_to_pipeline_names_map = (
            self._aggregate_pipeline_events_with_same_desired_status(pipeline_events)
        )
        for desired_status in desired_status_to_pipeline_names_map:
            pipeline_arns = [
                self._generate_pipeline_arn_and_name(pipeline_name)["pipeline_arn"]
                for pipeline_name in desired_status_to_pipeline_names_map[desired_status]
            ]
            curr_filter = BASE_EVENT_PATTERN.copy()
            curr_filter["detail"]["pipelineArn"] = pipeline_arns
            curr_filter["detail"]["currentPipelineExecutionStatus"] = [
                status_enum.value for status_enum in desired_status
            ]
            filters.append(curr_filter)
        if len(filters) > 1:
            result_event_pattern["$or"] = filters
        else:
            result_event_pattern.update(filters[0])
        return json.dumps(result_event_pattern)

    def _validate_feature_processor_pipeline_events(
        self, pipeline_events: List[FeatureProcessorPipelineEvents]
    ) -> None:
        """Validates the pipeline events.

        Args:
            pipeline_events: List of pipeline events.
        Raises:
            ValueError: If pipeline events contain duplicate pipeline names.
        """

        unique_pipelines = {event.pipeline_name for event in pipeline_events}
        potential_infinite_loop = []
        if len(unique_pipelines) != len(pipeline_events):
            raise ValueError("Pipeline names in pipeline_events must be unique.")

        for event in pipeline_events:
            if FeatureProcessorPipelineExecutionStatus.EXECUTING in event.pipeline_execution_status:
                potential_infinite_loop.append(event.pipeline_name)
        if potential_infinite_loop:
            logger.warning(
                "Potential infinite loop detected for pipelines %s. "
                "Setting pipeline_execution_status to EXECUTING might cause infinite loop. "
                "Please consider a terminal status instead.",
                potential_infinite_loop,
            )

    def _aggregate_pipeline_events_with_same_desired_status(
        self, pipeline_events: List[FeatureProcessorPipelineEvents]
    ) -> Dict[Tuple, List[str]]:
        """Aggregate pipeline events with same desired status.

            e.g.
            {
                (FeatureProcessorPipelineExecutionStatus.FAILED,
                FeatureProcessorPipelineExecutionStatus.STOPPED):
                    ["pipeline_name_1", "pipeline_name_2"],
                (FeatureProcessorPipelineExecutionStatus.STOPPED,
                FeatureProcessorPipelineExecutionStatus.STOPPED):
                    ["pipeline_name_3"],
            }
        Args:
            pipeline_events: List of pipeline events.
        Returns:
            Dict[Tuple, List[str]]: A dictionary of desired status keys and corresponding pipeline
            names.
        """
        events_by_desired_status = {}

        for event in pipeline_events:
            sorted_execution_status = sorted(event.pipeline_execution_status, key=lambda x: x.value)
            desired_status_keys = tuple(sorted_execution_status)

            if desired_status_keys not in events_by_desired_status:
                events_by_desired_status[desired_status_keys] = []
            events_by_desired_status[desired_status_keys].append(event.pipeline_name)

        return events_by_desired_status

    def _generate_pipeline_arn_and_name(self, pipeline_uri: str) -> Dict[str, str]:
        """Generate pipeline arn and pipeline name from pipeline uri.

        Args:
            pipeline_uri: The name or arn of the pipeline.
        Returns:
            Dict[str, str]: The arn and name of the pipeline.
        """
        match = re.match(PIPELINE_ARN_REGEX_PATTERN, pipeline_uri)
        pipeline_arn = ""
        pipeline_name = ""
        if not match:
            pipeline_name = pipeline_uri
            describe_pipeline_response = self.sagemaker_session.sagemaker_client.describe_pipeline(
                PipelineName=pipeline_name
            )
            pipeline_arn = describe_pipeline_response["PipelineArn"]
        else:
            pipeline_arn = pipeline_uri
            pipeline_name = match.group(4)
        return dict(pipeline_arn=pipeline_arn, pipeline_name=pipeline_name)
