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

import copy
import datetime
from typing import Iterator, List

import pytest
from mock import call, patch, Mock

from sagemaker import Session
from sagemaker.feature_store.feature_processor._event_bridge_scheduler_helper import (
    EventBridgeSchedulerHelper,
)
from sagemaker.feature_store.feature_processor._event_bridge_rule_helper import (
    EventBridgeRuleHelper,
)
from sagemaker.feature_store.feature_processor.lineage.constants import (
    TRANSFORMATION_CODE_STATUS_INACTIVE,
)
from sagemaker.lineage.context import Context
from sagemaker.lineage.artifact import Artifact
from test_constants import (
    FEATURE_GROUP_DATA_SOURCE,
    FEATURE_GROUP_INPUT,
    LAST_UPDATE_TIME,
    PIPELINE,
    PIPELINE_ARN,
    PIPELINE_CONTEXT,
    PIPELINE_NAME,
    PIPELINE_VERSION_CONTEXT,
    RAW_DATA_INPUT,
    RAW_DATA_INPUT_ARTIFACTS,
    RESOURCE_NOT_FOUND_EXCEPTION,
    SAGEMAKER_SESSION_MOCK,
    SCHEDULE_ARTIFACT_RESULT,
    PIPELINE_TRIGGER_ARTIFACT,
    TRANSFORMATION_CODE_ARTIFACT_1,
    TRANSFORMATION_CODE_ARTIFACT_2,
    TRANSFORMATION_CODE_INPUT_1,
    TRANSFORMATION_CODE_INPUT_2,
    ARTIFACT_SUMMARY,
    ARTIFACT_RESULT,
)

from sagemaker.feature_store.feature_processor.lineage._feature_group_lineage_entity_handler import (
    FeatureGroupLineageEntityHandler,
)
from sagemaker.feature_store.feature_processor.lineage._feature_processor_lineage import (
    FeatureProcessorLineageHandler,
)
from sagemaker.feature_store.feature_processor.lineage._lineage_association_handler import (
    LineageAssociationHandler,
)
from sagemaker.feature_store.feature_processor.lineage._pipeline_lineage_entity_handler import (
    PipelineLineageEntityHandler,
)
from sagemaker.feature_store.feature_processor.lineage._pipeline_schedule import (
    PipelineSchedule,
)
from sagemaker.feature_store.feature_processor.lineage._pipeline_trigger import (
    PipelineTrigger,
)
from sagemaker.feature_store.feature_processor.lineage._pipeline_version_lineage_entity_handler import (
    PipelineVersionLineageEntityHandler,
)
from sagemaker.feature_store.feature_processor.lineage._s3_lineage_entity_handler import (
    S3LineageEntityHandler,
)
from sagemaker.lineage._api_types import AssociationSummary

SCHEDULE_ARN = ""
SCHEDULE_EXPRESSION = ""
STATE = ""
TRIGGER_ARN = ""
EVENT_PATTERN = ""
START_DATE = datetime.datetime(2023, 4, 28, 21, 53, 47, 912000)
TAGS = [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]


@pytest.fixture
def sagemaker_session():
    boto_session = Mock()
    boto_session.client("scheduler").return_value = Mock()
    return Mock(Session, boto_session=boto_session)


@pytest.fixture
def event_bridge_scheduler_helper(sagemaker_session):
    return EventBridgeSchedulerHelper(
        sagemaker_session, sagemaker_session.boto_session.client("scheduler")
    )


def test_create_lineage_when_no_lineage_exists_with_fg_only():
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[
                FEATURE_GROUP_INPUT[0],
                FEATURE_GROUP_INPUT[1],
                FEATURE_GROUP_INPUT[0],
            ],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=TRANSFORMATION_CODE_ARTIFACT_1,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            side_effect=RESOURCE_NOT_FOUND_EXCEPTION,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineLineageEntityHandler,
            "create_pipeline_context",
            return_value=PIPELINE_CONTEXT,
        ),
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ),
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                [],
                generate_pipeline_version_upstream_transformation_code(),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
    ):
        lineage_handler.create_lineage()

    retrieve_feature_group_context_arns_method.assert_has_calls(
        [
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[1].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == retrieve_feature_group_context_arns_method.call_count

    retrieve_raw_data_artifact_method.assert_not_called()

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        creation_time=PIPELINE["CreationTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_version_context_method.assert_not_called()
    list_upstream_associations_method.assert_not_called()
    list_downstream_associations_method.assert_not_called()
    update_pipeline_context_method.assert_not_called()

    add_upstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_inputs=FEATURE_GROUP_INPUT,
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_downstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_output=FEATURE_GROUP_INPUT[0],
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_raw_data_associations_method.assert_called_once_with(
        raw_data_inputs=[],
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_transformation_code_associations_method.assert_called_once_with(
        transformation_code_artifact=TRANSFORMATION_CODE_ARTIFACT_1,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_pipeline_and_pipeline_version_association_method.assert_called_once_with(
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )


def test_create_lineage_when_no_lineage_exists_with_raw_data_only():
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=RAW_DATA_INPUT,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[
                FEATURE_GROUP_INPUT[0],
                FEATURE_GROUP_INPUT[1],
                FEATURE_GROUP_INPUT[0],
            ],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=TRANSFORMATION_CODE_ARTIFACT_1,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            side_effect=RESOURCE_NOT_FOUND_EXCEPTION,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineLineageEntityHandler,
            "create_pipeline_context",
            return_value=PIPELINE_CONTEXT,
        ),
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ),
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                [],
                generate_pipeline_version_upstream_transformation_code(),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_lineage(TAGS)

    retrieve_feature_group_context_arns_method.assert_called_once_with(
        feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[2], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[3], sagemaker_session=SAGEMAKER_SESSION_MOCK),
        ]
    )
    assert 4 == retrieve_raw_data_artifact_method.call_count

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        creation_time=PIPELINE["CreationTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_version_context_method.assert_not_called()
    list_upstream_associations_method.assert_not_called()
    list_downstream_associations_method.assert_not_called()
    update_pipeline_context_method.assert_not_called()

    add_upstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_inputs=[],
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_downstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_output=FEATURE_GROUP_INPUT[0],
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_raw_data_associations_method.assert_called_once_with(
        raw_data_inputs=RAW_DATA_INPUT_ARTIFACTS,
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_transformation_code_associations_method.assert_called_once_with(
        transformation_code_artifact=TRANSFORMATION_CODE_ARTIFACT_1,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_pipeline_and_pipeline_version_association_method.assert_called_once_with(
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_set_tags.assert_called_once_with(TAGS)


def test_create_lineage_when_no_lineage_exists_with_fg_and_raw_data_with_tags():
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=RAW_DATA_INPUT + FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[
                FEATURE_GROUP_INPUT[0],
                FEATURE_GROUP_INPUT[1],
                FEATURE_GROUP_INPUT[0],
            ],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=TRANSFORMATION_CODE_ARTIFACT_1,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            side_effect=RESOURCE_NOT_FOUND_EXCEPTION,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineLineageEntityHandler,
            "create_pipeline_context",
            return_value=PIPELINE_CONTEXT,
        ),
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ),
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                [],
                generate_pipeline_version_upstream_transformation_code(),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_lineage(TAGS)

    retrieve_feature_group_context_arns_method.assert_has_calls(
        [
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[1].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == retrieve_feature_group_context_arns_method.call_count

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[2], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[3], sagemaker_session=SAGEMAKER_SESSION_MOCK),
        ]
    )
    assert 4 == retrieve_raw_data_artifact_method.call_count

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        creation_time=PIPELINE["CreationTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_version_context_method.assert_not_called()
    list_upstream_associations_method.assert_not_called()
    list_downstream_associations_method.assert_not_called()
    update_pipeline_context_method.assert_not_called()

    add_upstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_inputs=FEATURE_GROUP_INPUT,
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_downstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_output=FEATURE_GROUP_INPUT[0],
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_raw_data_associations_method.assert_called_once_with(
        raw_data_inputs=RAW_DATA_INPUT_ARTIFACTS,
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_transformation_code_associations_method.assert_called_once_with(
        transformation_code_artifact=TRANSFORMATION_CODE_ARTIFACT_1,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_pipeline_and_pipeline_version_association_method.assert_called_once_with(
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_set_tags.assert_called_once_with(TAGS)


def test_create_lineage_when_no_lineage_exists_with_no_transformation_code():
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=RAW_DATA_INPUT + FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[
                FEATURE_GROUP_INPUT[0],
                FEATURE_GROUP_INPUT[1],
                FEATURE_GROUP_INPUT[0],
            ],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=None,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            side_effect=RESOURCE_NOT_FOUND_EXCEPTION,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineLineageEntityHandler,
            "create_pipeline_context",
            return_value=PIPELINE_CONTEXT,
        ),
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ),
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                [],
                generate_pipeline_version_upstream_transformation_code(),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_lineage(TAGS)

    retrieve_feature_group_context_arns_method.assert_has_calls(
        [
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[1].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == retrieve_feature_group_context_arns_method.call_count

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[2], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[3], sagemaker_session=SAGEMAKER_SESSION_MOCK),
        ]
    )
    assert 4 == retrieve_raw_data_artifact_method.call_count

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=None,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        creation_time=PIPELINE["CreationTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_version_context_method.assert_not_called()
    list_upstream_associations_method.assert_not_called()
    list_downstream_associations_method.assert_not_called()
    update_pipeline_context_method.assert_not_called()

    add_upstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_inputs=FEATURE_GROUP_INPUT,
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_downstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_output=FEATURE_GROUP_INPUT[0],
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_raw_data_associations_method.assert_called_once_with(
        raw_data_inputs=RAW_DATA_INPUT_ARTIFACTS,
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_transformation_code_associations_method.assert_not_called()

    add_pipeline_and_pipeline_version_association_method.assert_called_once_with(
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_set_tags.assert_not_called()


def test_create_lineage_when_already_exist_with_no_version_change():
    transformation_code_1 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=RAW_DATA_INPUT + FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[
                FEATURE_GROUP_INPUT[0],
                FEATURE_GROUP_INPUT[1],
                FEATURE_GROUP_INPUT[0],
            ],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=TRANSFORMATION_CODE_ARTIFACT_1,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=PIPELINE_CONTEXT,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                generate_pipeline_version_upstream_raw_data_list(),
                generate_pipeline_version_upstream_transformation_code(),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            S3LineageEntityHandler,
            "load_artifact_from_arn",
            return_value=transformation_code_1,
        ) as load_artifact_from_arn_method,
        patch.object(
            S3LineageEntityHandler,
            "update_transformation_code_artifact",
        ) as update_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as create_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_lineage(TAGS)

    retrieve_feature_group_context_arns_method.assert_has_calls(
        [
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[1].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == retrieve_feature_group_context_arns_method.call_count

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[2], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[3], sagemaker_session=SAGEMAKER_SESSION_MOCK),
        ]
    )
    assert 4 == retrieve_raw_data_artifact_method.call_count

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_has_calls(
        [
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 2 == load_pipeline_context_method.call_count
    load_pipeline_version_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        last_update_time=PIPELINE_CONTEXT.properties["LastUpdateTime"],
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    list_upstream_associations_method.assert_has_calls(
        [
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="FeatureGroupPipelineVersion",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="DataSet",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="TransformationCode",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == list_upstream_associations_method.call_count

    list_downstream_associations_method.assert_called_once_with(
        entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        destination_type="FeatureGroupPipelineVersion",
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_artifact_from_arn_method.assert_called_once_with(
        artifact_arn=TRANSFORMATION_CODE_ARTIFACT_1.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    transformation_code_2 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    transformation_code_2.properties["state"] = TRANSFORMATION_CODE_STATUS_INACTIVE
    transformation_code_2.properties["exclusive_end_date"] = PIPELINE["LastModifiedTime"].strftime(
        "%s"
    )
    update_transformation_code_artifact_method.assert_called_once_with(
        transformation_code_artifact=transformation_code_2
    )

    add_upstream_transformation_code_associations_method.assert_called_once_with(
        transformation_code_artifact=TRANSFORMATION_CODE_ARTIFACT_1,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    create_pipeline_version_context_method.assert_not_called()
    update_pipeline_context_method.assert_not_called()
    add_upstream_feature_group_data_associations_method.assert_not_called()
    add_downstream_feature_group_data_associations_method.assert_not_called()
    add_upstream_raw_data_associations_method.assert_not_called()
    add_pipeline_and_pipeline_version_association_method.assert_not_called()

    artifact_set_tags.assert_called_once_with(TAGS)


def test_create_lineage_when_already_exist_with_changed_raw_data():
    transformation_code_1 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    pipeline_context = copy.deepcopy(PIPELINE_CONTEXT)
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=[RAW_DATA_INPUT[0], RAW_DATA_INPUT[1]] + FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[
                FEATURE_GROUP_INPUT[0],
                FEATURE_GROUP_INPUT[1],
                FEATURE_GROUP_INPUT[0],
            ],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[RAW_DATA_INPUT_ARTIFACTS[0], RAW_DATA_INPUT_ARTIFACTS[1]],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=TRANSFORMATION_CODE_ARTIFACT_1,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=pipeline_context,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                generate_pipeline_version_upstream_raw_data_list(),
                generate_pipeline_version_upstream_transformation_code(),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            S3LineageEntityHandler,
            "load_artifact_from_arn",
            return_value=transformation_code_1,
        ) as load_artifact_from_arn_method,
        patch.object(
            S3LineageEntityHandler,
            "update_transformation_code_artifact",
        ) as update_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ),
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_lineage(TAGS)

    retrieve_feature_group_context_arns_method.assert_has_calls(
        [
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[1].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == retrieve_feature_group_context_arns_method.call_count

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=SAGEMAKER_SESSION_MOCK),
        ]
    )
    assert 2 == retrieve_raw_data_artifact_method.call_count

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_has_calls(
        [
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 2 == load_pipeline_context_method.call_count
    load_pipeline_version_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        last_update_time=LAST_UPDATE_TIME,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    list_upstream_associations_method.assert_has_calls(
        [
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="FeatureGroupPipelineVersion",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="DataSet",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="TransformationCode",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == list_upstream_associations_method.call_count

    list_downstream_associations_method.assert_called_once_with(
        entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        destination_type="FeatureGroupPipelineVersion",
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_artifact_from_arn_method.assert_called_once_with(
        artifact_arn=TRANSFORMATION_CODE_ARTIFACT_1.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    transformation_code_2 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    transformation_code_2.properties["state"] = TRANSFORMATION_CODE_STATUS_INACTIVE
    transformation_code_2.properties["exclusive_end_date"] = PIPELINE["LastModifiedTime"].strftime(
        "%s"
    )
    update_transformation_code_artifact_method.assert_called_once_with(
        transformation_code_artifact=transformation_code_2
    )

    add_upstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_inputs=FEATURE_GROUP_INPUT,
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_downstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_output=FEATURE_GROUP_INPUT[0],
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_raw_data_associations_method.assert_called_once_with(
        raw_data_inputs=[RAW_DATA_INPUT_ARTIFACTS[0], RAW_DATA_INPUT_ARTIFACTS[1]],
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_transformation_code_associations_method.assert_called_once_with(
        transformation_code_artifact=TRANSFORMATION_CODE_ARTIFACT_1,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    assert pipeline_context.properties["LastUpdateTime"] == PIPELINE["LastModifiedTime"].strftime(
        "%s"
    )
    update_pipeline_context_method.assert_called_once_with(pipeline_context=pipeline_context)

    add_pipeline_and_pipeline_version_association_method.assert_called_once_with(
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_set_tags.assert_called_once_with(TAGS)


def test_create_lineage_when_already_exist_with_changed_input_fg():
    transformation_code_1 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    pipeline_context = copy.deepcopy(PIPELINE_CONTEXT)
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=RAW_DATA_INPUT + [FEATURE_GROUP_DATA_SOURCE[0]],
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[FEATURE_GROUP_INPUT[0], FEATURE_GROUP_INPUT[0]],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=TRANSFORMATION_CODE_ARTIFACT_1,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=pipeline_context,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                generate_pipeline_version_upstream_raw_data_list(),
                generate_pipeline_version_upstream_transformation_code(),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            S3LineageEntityHandler,
            "load_artifact_from_arn",
            return_value=transformation_code_1,
        ) as load_artifact_from_arn_method,
        patch.object(
            S3LineageEntityHandler,
            "update_transformation_code_artifact",
        ) as update_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ),
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_lineage(TAGS)

    retrieve_feature_group_context_arns_method.assert_has_calls(
        [
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 2 == retrieve_feature_group_context_arns_method.call_count

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[2], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[3], sagemaker_session=SAGEMAKER_SESSION_MOCK),
        ]
    )
    assert 4 == retrieve_raw_data_artifact_method.call_count

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_has_calls(
        [
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 2 == load_pipeline_context_method.call_count
    load_pipeline_version_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        last_update_time=LAST_UPDATE_TIME,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    list_upstream_associations_method.assert_has_calls(
        [
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="FeatureGroupPipelineVersion",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="DataSet",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="TransformationCode",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == list_upstream_associations_method.call_count

    list_downstream_associations_method.assert_called_once_with(
        entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        destination_type="FeatureGroupPipelineVersion",
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_artifact_from_arn_method.assert_called_once_with(
        artifact_arn=TRANSFORMATION_CODE_ARTIFACT_1.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    transformation_code_2 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    transformation_code_2.properties["state"] = TRANSFORMATION_CODE_STATUS_INACTIVE
    transformation_code_2.properties["exclusive_end_date"] = PIPELINE["LastModifiedTime"].strftime(
        "%s"
    )
    update_transformation_code_artifact_method.assert_called_once_with(
        transformation_code_artifact=transformation_code_2
    )

    add_upstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_inputs=[FEATURE_GROUP_INPUT[0]],
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_downstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_output=FEATURE_GROUP_INPUT[0],
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_raw_data_associations_method.assert_called_once_with(
        raw_data_inputs=RAW_DATA_INPUT_ARTIFACTS,
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_transformation_code_associations_method.assert_called_once_with(
        transformation_code_artifact=TRANSFORMATION_CODE_ARTIFACT_1,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    assert pipeline_context.properties["LastUpdateTime"] == PIPELINE["LastModifiedTime"].strftime(
        "%s"
    )
    update_pipeline_context_method.assert_called_once_with(pipeline_context=pipeline_context)

    add_pipeline_and_pipeline_version_association_method.assert_called_once_with(
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_set_tags.assert_called_once_with(TAGS)


def test_create_lineage_when_already_exist_with_changed_output_fg():
    transformation_code_1 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    pipeline_context = copy.deepcopy(PIPELINE_CONTEXT)
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=RAW_DATA_INPUT + FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[1].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[
                FEATURE_GROUP_INPUT[0],
                FEATURE_GROUP_INPUT[1],
                FEATURE_GROUP_INPUT[1],
            ],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=TRANSFORMATION_CODE_ARTIFACT_1,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=pipeline_context,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                generate_pipeline_version_upstream_raw_data_list(),
                generate_pipeline_version_upstream_transformation_code(),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            S3LineageEntityHandler,
            "load_artifact_from_arn",
            return_value=transformation_code_1,
        ) as load_artifact_from_arn_method,
        patch.object(
            S3LineageEntityHandler,
            "update_transformation_code_artifact",
        ) as update_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ),
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_lineage(TAGS)

    retrieve_feature_group_context_arns_method.assert_has_calls(
        [
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[1].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[1].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == retrieve_feature_group_context_arns_method.call_count

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[2], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[3], sagemaker_session=SAGEMAKER_SESSION_MOCK),
        ]
    )
    assert 4 == retrieve_raw_data_artifact_method.call_count

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_has_calls(
        [
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 2 == load_pipeline_context_method.call_count
    load_pipeline_version_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        last_update_time=LAST_UPDATE_TIME,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    list_upstream_associations_method.assert_has_calls(
        [
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="FeatureGroupPipelineVersion",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="DataSet",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="TransformationCode",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == list_upstream_associations_method.call_count

    list_downstream_associations_method.assert_called_once_with(
        entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        destination_type="FeatureGroupPipelineVersion",
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_artifact_from_arn_method.assert_called_once_with(
        artifact_arn=TRANSFORMATION_CODE_ARTIFACT_1.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    transformation_code_2 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    transformation_code_2.properties["state"] = TRANSFORMATION_CODE_STATUS_INACTIVE
    transformation_code_2.properties["exclusive_end_date"] = PIPELINE["LastModifiedTime"].strftime(
        "%s"
    )
    update_transformation_code_artifact_method.assert_called_once_with(
        transformation_code_artifact=transformation_code_2
    )

    add_upstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_inputs=FEATURE_GROUP_INPUT,
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_downstream_feature_group_data_associations_method.assert_called_once_with(
        feature_group_output=FEATURE_GROUP_INPUT[1],
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_raw_data_associations_method.assert_called_once_with(
        raw_data_inputs=RAW_DATA_INPUT_ARTIFACTS,
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    add_upstream_transformation_code_associations_method.assert_called_once_with(
        transformation_code_artifact=TRANSFORMATION_CODE_ARTIFACT_1,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    assert pipeline_context.properties["LastUpdateTime"] == PIPELINE["LastModifiedTime"].strftime(
        "%s"
    )
    update_pipeline_context_method.assert_called_once_with(pipeline_context=pipeline_context)

    add_pipeline_and_pipeline_version_association_method.assert_called_once_with(
        pipeline_context_arn=PIPELINE_CONTEXT.context_arn,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_set_tags.assert_called_once_with(TAGS)


def test_create_lineage_when_already_exist_with_changed_transformation_code():
    transformation_code_1 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    pipeline_context = copy.deepcopy(PIPELINE_CONTEXT)
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=RAW_DATA_INPUT + FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_2,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[
                FEATURE_GROUP_INPUT[0],
                FEATURE_GROUP_INPUT[1],
                FEATURE_GROUP_INPUT[0],
            ],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=TRANSFORMATION_CODE_ARTIFACT_2,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=pipeline_context,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                generate_pipeline_version_upstream_raw_data_list(),
                generate_pipeline_version_upstream_transformation_code(),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            S3LineageEntityHandler,
            "load_artifact_from_arn",
            return_value=transformation_code_1,
        ) as load_artifact_from_arn_method,
        patch.object(
            S3LineageEntityHandler,
            "update_transformation_code_artifact",
        ) as update_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ),
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_lineage(TAGS)

    retrieve_feature_group_context_arns_method.assert_has_calls(
        [
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[1].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == retrieve_feature_group_context_arns_method.call_count

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[2], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[3], sagemaker_session=SAGEMAKER_SESSION_MOCK),
        ]
    )
    assert 4 == retrieve_raw_data_artifact_method.call_count

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=TRANSFORMATION_CODE_INPUT_2,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_has_calls(
        [
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 2 == load_pipeline_context_method.call_count
    load_pipeline_version_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        last_update_time=LAST_UPDATE_TIME,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    list_upstream_associations_method.assert_has_calls(
        [
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="FeatureGroupPipelineVersion",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="DataSet",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="TransformationCode",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == list_upstream_associations_method.call_count

    list_downstream_associations_method.assert_called_once_with(
        entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        destination_type="FeatureGroupPipelineVersion",
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_artifact_from_arn_method.assert_called_once_with(
        artifact_arn=TRANSFORMATION_CODE_ARTIFACT_1.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    transformation_code_2 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    transformation_code_2.properties["state"] = TRANSFORMATION_CODE_STATUS_INACTIVE
    transformation_code_2.properties["exclusive_end_date"] = PIPELINE["LastModifiedTime"].strftime(
        "%s"
    )
    update_transformation_code_artifact_method.assert_called_once_with(
        transformation_code_artifact=transformation_code_2
    )

    assert pipeline_context.properties["LastUpdateTime"] == LAST_UPDATE_TIME

    update_pipeline_context_method.assert_not_called()
    add_upstream_feature_group_data_associations_method.assert_not_called()
    add_downstream_feature_group_data_associations_method.assert_not_called()
    add_upstream_raw_data_associations_method.assert_not_called()
    add_pipeline_and_pipeline_version_association_method.assert_not_called()

    add_upstream_transformation_code_associations_method.assert_called_once_with(
        transformation_code_artifact=TRANSFORMATION_CODE_ARTIFACT_2,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_set_tags.assert_called_once_with(TAGS)


def test_create_lineage_when_already_exist_with_last_transformation_code_as_none():
    transformation_code_1 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    transformation_code_1.properties["state"] = TRANSFORMATION_CODE_STATUS_INACTIVE
    transformation_code_1.properties["exclusive_end_date"] = PIPELINE["LastModifiedTime"].strftime(
        "%s"
    )
    pipeline_context = copy.deepcopy(PIPELINE_CONTEXT)
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=RAW_DATA_INPUT + FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_2,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[
                FEATURE_GROUP_INPUT[0],
                FEATURE_GROUP_INPUT[1],
                FEATURE_GROUP_INPUT[0],
            ],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=TRANSFORMATION_CODE_ARTIFACT_2,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=pipeline_context,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                generate_pipeline_version_upstream_raw_data_list(),
                generate_pipeline_version_upstream_transformation_code(),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            S3LineageEntityHandler,
            "load_artifact_from_arn",
            return_value=transformation_code_1,
        ) as load_artifact_from_arn_method,
        patch.object(
            S3LineageEntityHandler,
            "update_transformation_code_artifact",
        ) as update_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ),
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_lineage(TAGS)

    retrieve_feature_group_context_arns_method.assert_has_calls(
        [
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[1].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == retrieve_feature_group_context_arns_method.call_count

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[2], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[3], sagemaker_session=SAGEMAKER_SESSION_MOCK),
        ]
    )
    assert 4 == retrieve_raw_data_artifact_method.call_count

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=TRANSFORMATION_CODE_INPUT_2,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_has_calls(
        [
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 2 == load_pipeline_context_method.call_count
    load_pipeline_version_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        last_update_time=LAST_UPDATE_TIME,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    list_upstream_associations_method.assert_has_calls(
        [
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="FeatureGroupPipelineVersion",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="DataSet",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="TransformationCode",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == list_upstream_associations_method.call_count

    list_downstream_associations_method.assert_called_once_with(
        entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        destination_type="FeatureGroupPipelineVersion",
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_artifact_from_arn_method.assert_called_once_with(
        artifact_arn=TRANSFORMATION_CODE_ARTIFACT_1.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    update_transformation_code_artifact_method.assert_not_called()

    assert pipeline_context.properties["LastUpdateTime"] == LAST_UPDATE_TIME

    update_pipeline_context_method.assert_not_called()
    add_upstream_feature_group_data_associations_method.assert_not_called()
    add_downstream_feature_group_data_associations_method.assert_not_called()
    add_upstream_raw_data_associations_method.assert_not_called()
    add_pipeline_and_pipeline_version_association_method.assert_not_called()

    add_upstream_transformation_code_associations_method.assert_called_once_with(
        transformation_code_artifact=TRANSFORMATION_CODE_ARTIFACT_2,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_set_tags.assert_called_once_with(TAGS)


def test_create_lineage_when_already_exist_with_all_previous_transformation_code_as_none():
    pipeline_context = copy.deepcopy(PIPELINE_CONTEXT)
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=RAW_DATA_INPUT + FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_2,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[
                FEATURE_GROUP_INPUT[0],
                FEATURE_GROUP_INPUT[1],
                FEATURE_GROUP_INPUT[0],
            ],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=TRANSFORMATION_CODE_ARTIFACT_2,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=pipeline_context,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                generate_pipeline_version_upstream_raw_data_list(),
                iter([]),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            S3LineageEntityHandler,
            "load_artifact_from_arn",
        ) as load_artifact_from_arn_method,
        patch.object(
            S3LineageEntityHandler,
            "update_transformation_code_artifact",
        ) as update_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ),
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_lineage(TAGS)

    retrieve_feature_group_context_arns_method.assert_has_calls(
        [
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[1].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == retrieve_feature_group_context_arns_method.call_count

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[2], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[3], sagemaker_session=SAGEMAKER_SESSION_MOCK),
        ]
    )
    assert 4 == retrieve_raw_data_artifact_method.call_count

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=TRANSFORMATION_CODE_INPUT_2,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_has_calls(
        [
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 2 == load_pipeline_context_method.call_count
    load_pipeline_version_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        last_update_time=LAST_UPDATE_TIME,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    list_upstream_associations_method.assert_has_calls(
        [
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="FeatureGroupPipelineVersion",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="DataSet",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="TransformationCode",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == list_upstream_associations_method.call_count

    list_downstream_associations_method.assert_called_once_with(
        entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        destination_type="FeatureGroupPipelineVersion",
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_artifact_from_arn_method.assert_not_called()
    update_transformation_code_artifact_method.assert_not_called()

    assert pipeline_context.properties["LastUpdateTime"] == LAST_UPDATE_TIME

    update_pipeline_context_method.assert_not_called()
    add_upstream_feature_group_data_associations_method.assert_not_called()
    add_downstream_feature_group_data_associations_method.assert_not_called()
    add_upstream_raw_data_associations_method.assert_not_called()
    add_pipeline_and_pipeline_version_association_method.assert_not_called()

    add_upstream_transformation_code_associations_method.assert_called_once_with(
        transformation_code_artifact=TRANSFORMATION_CODE_ARTIFACT_2,
        pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_set_tags.assert_called_once_with(TAGS)


def test_create_lineage_when_already_exist_with_removed_transformation_code():
    transformation_code_1 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    pipeline_context = copy.deepcopy(PIPELINE_CONTEXT)
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=RAW_DATA_INPUT + FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            FeatureGroupLineageEntityHandler,
            "retrieve_feature_group_context_arns",
            side_effect=[
                FEATURE_GROUP_INPUT[0],
                FEATURE_GROUP_INPUT[1],
                FEATURE_GROUP_INPUT[0],
            ],
        ) as retrieve_feature_group_context_arns_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            S3LineageEntityHandler,
            "create_transformation_code_artifact",
            return_value=None,
        ) as create_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=pipeline_context,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                generate_pipeline_version_upstream_raw_data_list(),
                generate_pipeline_version_upstream_transformation_code(),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            S3LineageEntityHandler,
            "load_artifact_from_arn",
            return_value=transformation_code_1,
        ) as load_artifact_from_arn_method,
        patch.object(
            S3LineageEntityHandler,
            "update_transformation_code_artifact",
        ) as update_transformation_code_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "update_pipeline_context",
        ) as update_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "create_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ),
        patch.object(
            LineageAssociationHandler, "add_upstream_feature_group_data_associations"
        ) as add_upstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_downstream_feature_group_data_associations"
        ) as add_downstream_feature_group_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_raw_data_associations"
        ) as add_upstream_raw_data_associations_method,
        patch.object(
            LineageAssociationHandler, "add_upstream_transformation_code_associations"
        ) as add_upstream_transformation_code_associations_method,
        patch.object(
            LineageAssociationHandler, "add_pipeline_and_pipeline_version_association"
        ) as add_pipeline_and_pipeline_version_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_lineage(TAGS)

    retrieve_feature_group_context_arns_method.assert_has_calls(
        [
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[1].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                feature_group_name=FEATURE_GROUP_DATA_SOURCE[0].name,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == retrieve_feature_group_context_arns_method.call_count

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[2], sagemaker_session=SAGEMAKER_SESSION_MOCK),
            call(raw_data=RAW_DATA_INPUT[3], sagemaker_session=SAGEMAKER_SESSION_MOCK),
        ]
    )
    assert 4 == retrieve_raw_data_artifact_method.call_count

    create_transformation_code_artifact_method.assert_called_once_with(
        transformation_code=None,
        pipeline_last_update_time=PIPELINE["LastModifiedTime"].strftime("%s"),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_pipeline_context_method.assert_has_calls(
        [
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                pipeline_name=PIPELINE_NAME,
                creation_time=PIPELINE["CreationTime"].strftime("%s"),
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 2 == load_pipeline_context_method.call_count
    load_pipeline_version_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        last_update_time=LAST_UPDATE_TIME,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    list_upstream_associations_method.assert_has_calls(
        [
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="FeatureGroupPipelineVersion",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="DataSet",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_type="TransformationCode",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 3 == list_upstream_associations_method.call_count

    list_downstream_associations_method.assert_called_once_with(
        entity_arn=PIPELINE_VERSION_CONTEXT.context_arn,
        destination_type="FeatureGroupPipelineVersion",
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    load_artifact_from_arn_method.assert_called_once_with(
        artifact_arn=TRANSFORMATION_CODE_ARTIFACT_1.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    transformation_code_2 = copy.deepcopy(TRANSFORMATION_CODE_ARTIFACT_1)
    transformation_code_2.properties["state"] = TRANSFORMATION_CODE_STATUS_INACTIVE
    transformation_code_2.properties["exclusive_end_date"] = PIPELINE["LastModifiedTime"].strftime(
        "%s"
    )
    update_transformation_code_artifact_method.assert_called_once_with(
        transformation_code_artifact=transformation_code_2
    )

    update_pipeline_context_method.assert_not_called()
    add_upstream_feature_group_data_associations_method.assert_not_called()
    add_downstream_feature_group_data_associations_method.assert_not_called()
    add_upstream_raw_data_associations_method.assert_not_called()
    add_upstream_transformation_code_associations_method.assert_not_called()
    add_pipeline_and_pipeline_version_association_method.assert_not_called()

    artifact_set_tags.assert_not_called()


def test_get_pipeline_lineage_names_when_no_lineage_exists():
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with patch.object(
        PipelineLineageEntityHandler,
        "load_pipeline_context",
        side_effect=RESOURCE_NOT_FOUND_EXCEPTION,
    ) as load_pipeline_context_method:
        return_value = lineage_handler.get_pipeline_lineage_names()

        assert return_value is None

        load_pipeline_context_method.assert_called_once_with(
            pipeline_name=PIPELINE_NAME,
            creation_time=PIPELINE["CreationTime"].strftime("%s"),
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )


def test_get_pipeline_lineage_names_when_lineage_exists():
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_1,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=PIPELINE_CONTEXT,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
    ):
        return_value = lineage_handler.get_pipeline_lineage_names()

        assert return_value == dict(
            pipeline_context_name=PIPELINE_CONTEXT.context_name,
            pipeline_version_context_name=PIPELINE_VERSION_CONTEXT.context_name,
        )

        load_pipeline_context_method.assert_has_calls(
            [
                call(
                    pipeline_name=PIPELINE_NAME,
                    creation_time=PIPELINE["CreationTime"].strftime("%s"),
                    sagemaker_session=SAGEMAKER_SESSION_MOCK,
                ),
                call(
                    pipeline_name=PIPELINE_NAME,
                    creation_time=PIPELINE["CreationTime"].strftime("%s"),
                    sagemaker_session=SAGEMAKER_SESSION_MOCK,
                ),
            ]
        )
        assert 2 == load_pipeline_context_method.call_count

        load_pipeline_version_context_method.assert_called_once_with(
            pipeline_name=PIPELINE_NAME,
            last_update_time=PIPELINE_CONTEXT.properties["LastUpdateTime"],
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )


def test_create_schedule_lineage():
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=PIPELINE_CONTEXT,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_pipeline_schedule_artifact",
            return_value=SCHEDULE_ARTIFACT_RESULT,
        ) as retrieve_pipeline_schedule_artifact_method,
        patch.object(
            LineageAssociationHandler,
            "add_upstream_schedule_associations",
        ) as add_upstream_schedule_associations_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_schedule_lineage(
            pipeline_name=PIPELINE_NAME,
            schedule_arn=SCHEDULE_ARN,
            schedule_expression=SCHEDULE_EXPRESSION,
            state=STATE,
            start_date=START_DATE,
            tags=TAGS,
        )

        load_pipeline_context_method.assert_called_once_with(
            pipeline_name=PIPELINE_NAME,
            creation_time=PIPELINE["CreationTime"].strftime("%s"),
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

        load_pipeline_version_context_method.assert_called_once_with(
            pipeline_name=PIPELINE_NAME,
            last_update_time=PIPELINE_CONTEXT.properties["LastUpdateTime"],
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

        retrieve_pipeline_schedule_artifact_method.assert_called_once_with(
            pipeline_schedule=PipelineSchedule(
                schedule_name=PIPELINE_NAME,
                schedule_arn=SCHEDULE_ARN,
                schedule_expression=SCHEDULE_EXPRESSION,
                pipeline_name=PIPELINE_NAME,
                state=STATE,
                start_date=START_DATE.strftime("%s"),
            ),
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

        add_upstream_schedule_associations_method.assert_called_once_with(
            schedule_artifact=SCHEDULE_ARTIFACT_RESULT,
            pipeline_version_context_arn=PIPELINE_VERSION_CONTEXT.context_arn,
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

        artifact_set_tags.assert_called_once_with(TAGS)


def test_create_trigger_lineage():
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
    with (
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=PIPELINE_CONTEXT,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            S3LineageEntityHandler,
            "retrieve_pipeline_trigger_artifact",
            return_value=PIPELINE_TRIGGER_ARTIFACT,
        ) as retrieve_pipeline_trigger_artifact_method,
        patch.object(
            LineageAssociationHandler,
            "_add_association",
        ) as add_association_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
    ):
        lineage_handler.create_trigger_lineage(
            pipeline_name=PIPELINE_NAME,
            trigger_arn=TRIGGER_ARN,
            event_pattern=EVENT_PATTERN,
            state=STATE,
            tags=TAGS,
        )

        load_pipeline_context_method.assert_called_once_with(
            pipeline_name=PIPELINE_NAME,
            creation_time=PIPELINE["CreationTime"].strftime("%s"),
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

        load_pipeline_version_context_method.assert_called_once_with(
            pipeline_name=PIPELINE_NAME,
            last_update_time=PIPELINE_CONTEXT.properties["LastUpdateTime"],
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

        retrieve_pipeline_trigger_artifact_method.assert_called_once_with(
            pipeline_trigger=PipelineTrigger(
                trigger_name=PIPELINE_NAME,
                trigger_arn=TRIGGER_ARN,
                pipeline_name=PIPELINE_NAME,
                event_pattern=EVENT_PATTERN,
                state=STATE,
            ),
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

        add_association_method.assert_called_once_with(
            source_arn=PIPELINE_TRIGGER_ARTIFACT.artifact_arn,
            destination_arn=PIPELINE_VERSION_CONTEXT.context_arn,
            association_type="ContributedTo",
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

        artifact_set_tags.assert_called_once_with(TAGS)


def test_upsert_tags_for_lineage_resources():
    pipeline_context = copy.deepcopy(PIPELINE_CONTEXT)
    mock_session = Mock(Session)
    lineage_handler = FeatureProcessorLineageHandler(
        pipeline_name=PIPELINE_NAME,
        pipeline_arn=PIPELINE_ARN,
        pipeline=PIPELINE,
        inputs=RAW_DATA_INPUT + FEATURE_GROUP_DATA_SOURCE,
        output=FEATURE_GROUP_DATA_SOURCE[0].name,
        transformation_code=TRANSFORMATION_CODE_INPUT_2,
        sagemaker_session=mock_session,
    )
    lineage_handler.sagemaker_session.boto_session = Mock()
    lineage_handler.sagemaker_session.sagemaker_client = Mock()
    with (
        patch.object(
            S3LineageEntityHandler,
            "retrieve_raw_data_artifact",
            side_effect=[
                RAW_DATA_INPUT_ARTIFACTS[0],
                RAW_DATA_INPUT_ARTIFACTS[1],
                RAW_DATA_INPUT_ARTIFACTS[2],
                RAW_DATA_INPUT_ARTIFACTS[3],
            ],
        ) as retrieve_raw_data_artifact_method,
        patch.object(
            PipelineLineageEntityHandler,
            "load_pipeline_context",
            return_value=pipeline_context,
        ) as load_pipeline_context_method,
        patch.object(
            PipelineVersionLineageEntityHandler,
            "load_pipeline_version_context",
            return_value=PIPELINE_VERSION_CONTEXT,
        ) as load_pipeline_version_context_method,
        patch.object(
            LineageAssociationHandler,
            "list_upstream_associations",
            side_effect=[
                generate_pipeline_version_upstream_feature_group_list(),
                generate_pipeline_version_upstream_raw_data_list(),
                iter([]),
            ],
        ) as list_upstream_associations_method,
        patch.object(
            LineageAssociationHandler,
            "list_downstream_associations",
            return_value=generate_pipeline_version_downstream_feature_group(),
        ) as list_downstream_associations_method,
        patch.object(
            S3LineageEntityHandler, "load_artifact_from_arn", return_value=ARTIFACT_RESULT
        ) as load_artifact_from_arn_method,
        patch.object(
            S3LineageEntityHandler, "_load_artifact_from_s3_uri", return_value=ARTIFACT_SUMMARY
        ) as load_artifact_from_s3_uri_method,
        patch.object(
            Artifact,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as artifact_set_tags,
        patch.object(
            Context,
            "set_tags",
            return_value={
                "Tags": [dict(Key="key_1", Value="value_1"), dict(Key="key_2", Value="value_2")]
            },
        ) as context_set_tags,
        patch.object(
            EventBridgeSchedulerHelper, "describe_schedule", return_value=dict(Arn="schedule_arn")
        ) as get_event_bridge_schedule,
        patch.object(
            EventBridgeRuleHelper, "describe_rule", return_value=dict(Arn="rule_arn")
        ) as get_event_bridge_rule,
    ):
        lineage_handler.upsert_tags_for_lineage_resources(TAGS)

    retrieve_raw_data_artifact_method.assert_has_calls(
        [
            call(raw_data=RAW_DATA_INPUT[0], sagemaker_session=mock_session),
            call(raw_data=RAW_DATA_INPUT[1], sagemaker_session=mock_session),
            call(raw_data=RAW_DATA_INPUT[2], sagemaker_session=mock_session),
            call(raw_data=RAW_DATA_INPUT[3], sagemaker_session=mock_session),
        ]
    )

    load_pipeline_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        creation_time=PIPELINE["CreationTime"].strftime("%s"),
        sagemaker_session=mock_session,
    )

    load_pipeline_version_context_method.assert_called_once_with(
        pipeline_name=PIPELINE_NAME,
        last_update_time=LAST_UPDATE_TIME,
        sagemaker_session=mock_session,
    )

    list_upstream_associations_method.assert_not_called()
    list_downstream_associations_method.assert_not_called()
    load_artifact_from_s3_uri_method.assert_has_calls(
        [
            call(s3_uri="schedule_arn", sagemaker_session=mock_session),
            call(s3_uri="rule_arn", sagemaker_session=mock_session),
        ]
    )
    get_event_bridge_schedule.assert_called_once_with(PIPELINE_NAME)
    get_event_bridge_rule.assert_called_once_with(PIPELINE_NAME)
    load_artifact_from_arn_method.assert_called_with(
        artifact_arn=ARTIFACT_SUMMARY.artifact_arn, sagemaker_session=mock_session
    )

    # three raw data artifact, one schedule artifact and one trigger artifact
    artifact_set_tags.assert_has_calls(
        [
            call(TAGS),
            call(TAGS),
            call(TAGS),
            call(TAGS),
            call(TAGS),
        ]
    )
    # pipeline context and current pipeline version context
    context_set_tags.assert_has_calls(
        [
            call(TAGS),
            call(TAGS),
        ]
    )


def generate_pipeline_version_upstream_feature_group_list() -> Iterator[AssociationSummary]:
    pipeline_version_upstream_fg: List[AssociationSummary] = list()
    for feature_group in FEATURE_GROUP_INPUT:
        pipeline_version_upstream_fg.append(
            AssociationSummary(
                source_arn=feature_group.pipeline_version_context_arn,
                source_name=f"{feature_group.name}-pipeline-version",
                destination_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                destination_name=PIPELINE_VERSION_CONTEXT.context_name,
                association_type="ContributedTo",
            )
        )
    return iter(pipeline_version_upstream_fg)


def generate_pipeline_version_upstream_raw_data_list() -> Iterator[AssociationSummary]:
    pipeline_version_upstream_fg: List[AssociationSummary] = list()
    for raw_data in RAW_DATA_INPUT_ARTIFACTS:
        pipeline_version_upstream_fg.append(
            AssociationSummary(
                source_arn=raw_data.artifact_arn,
                source_name="sm-fs-fe-raw-data",
                destination_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                destination_name=PIPELINE_VERSION_CONTEXT.context_name,
                association_type="ContributedTo",
            )
        )
    return iter(pipeline_version_upstream_fg)


def generate_pipeline_version_upstream_transformation_code() -> Iterator[AssociationSummary]:
    return iter(
        [
            AssociationSummary(
                source_arn=TRANSFORMATION_CODE_ARTIFACT_1.artifact_arn,
                source_name=TRANSFORMATION_CODE_ARTIFACT_1.artifact_name,
                destination_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                destination_name=PIPELINE_VERSION_CONTEXT.context_name,
                association_type="ContributedTo",
            )
        ]
    )


def generate_pipeline_version_downstream_feature_group() -> Iterator[AssociationSummary]:
    return iter(
        [
            AssociationSummary(
                source_arn=PIPELINE_VERSION_CONTEXT.context_arn,
                source_name=PIPELINE_VERSION_CONTEXT.context_name,
                destination_arn=FEATURE_GROUP_INPUT[0].pipeline_version_context_arn,
                destination_name=f"{FEATURE_GROUP_INPUT[0].name}-pipeline-version",
                association_type="ContributedTo",
            )
        ]
    )
