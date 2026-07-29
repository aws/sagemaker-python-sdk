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

from mock import patch
from test_constants import (
    ARTIFACT_RESULT,
    ARTIFACT_SUMMARY,
    PIPELINE_SCHEDULE,
    PIPELINE_SCHEDULE_2,
    PIPELINE_TRIGGER,
    PIPELINE_TRIGGER_2,
    PIPELINE_TRIGGER_ARTIFACT,
    PIPELINE_TRIGGER_ARTIFACT_SUMMARY,
    SCHEDULE_ARTIFACT_RESULT,
    TRANSFORMATION_CODE_ARTIFACT_1,
    TRANSFORMATION_CODE_INPUT_1,
    LAST_UPDATE_TIME,
    MockDataSource,
)
from test_pipeline_lineage_entity_handler import SAGEMAKER_SESSION_MOCK

from sagemaker.mlops.feature_store.feature_processor import CSVDataSource
from sagemaker.mlops.feature_store.feature_processor.lineage._s3_lineage_entity_handler import (
    S3LineageEntityHandler,
)
from sagemaker.mlops.feature_store.feature_processor.lineage._transformation_code import (
    TransformationCode,
)
from sagemaker.core.lineage.artifact import Artifact

raw_data = CSVDataSource(
    s3_uri="s3://sagemaker-us-west-2-789975069016/transform-2023-04-28-21-50-14-616/"
    "transform-2023-04-28-21-50-14-616/output/model.tar.gz"
)


def test_retrieve_raw_data_artifact_when_artifact_already_exist():
    with patch.object(Artifact, "list", return_value=[ARTIFACT_SUMMARY]) as artifact_list_method:
        with patch.object(Artifact, "load", return_value=ARTIFACT_RESULT) as artifact_load_method:
            with patch.object(
                Artifact, "create", return_value=ARTIFACT_RESULT
            ) as artifact_create_method:
                result = S3LineageEntityHandler.retrieve_raw_data_artifact(
                    raw_data=raw_data, sagemaker_session=SAGEMAKER_SESSION_MOCK
                )

    assert result == ARTIFACT_RESULT

    artifact_list_method.assert_called_once_with(
        source_uri=raw_data.s3_uri, sagemaker_session=SAGEMAKER_SESSION_MOCK
    )

    artifact_load_method.assert_called_once_with(
        artifact_arn=ARTIFACT_SUMMARY.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_create_method.assert_not_called()


def test_retrieve_raw_data_artifact_when_artifact_does_not_exist():
    with patch.object(Artifact, "list", return_value=[]) as artifact_list_method:
        with patch.object(Artifact, "load", return_value=ARTIFACT_RESULT) as artifact_load_method:
            with patch.object(
                Artifact, "create", return_value=ARTIFACT_RESULT
            ) as artifact_create_method:
                result = S3LineageEntityHandler.retrieve_raw_data_artifact(
                    raw_data=raw_data, sagemaker_session=SAGEMAKER_SESSION_MOCK
                )

    assert result == ARTIFACT_RESULT

    artifact_list_method.assert_called_once_with(
        source_uri=raw_data.s3_uri, sagemaker_session=SAGEMAKER_SESSION_MOCK
    )

    artifact_load_method.assert_not_called()

    artifact_create_method.assert_called_once_with(
        source_uri=raw_data.s3_uri,
        artifact_type="DataSet",
        artifact_name="sm-fs-fe-raw-data",
        properties=None,
        source_types=None,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )


def test_retrieve_user_defined_raw_data_artifact_when_artifact_already_exist():
    data_source = MockDataSource()
    with patch.object(Artifact, "list", return_value=[ARTIFACT_SUMMARY]) as artifact_list_method:
        with patch.object(Artifact, "load", return_value=ARTIFACT_RESULT) as artifact_load_method:
            with patch.object(
                Artifact, "create", return_value=ARTIFACT_RESULT
            ) as artifact_create_method:
                result = S3LineageEntityHandler.retrieve_raw_data_artifact(
                    raw_data=data_source, sagemaker_session=SAGEMAKER_SESSION_MOCK
                )

    assert result == ARTIFACT_RESULT

    artifact_list_method.assert_called_once_with(
        source_uri=data_source.data_source_unique_id, sagemaker_session=SAGEMAKER_SESSION_MOCK
    )

    artifact_load_method.assert_called_once_with(
        artifact_arn=ARTIFACT_SUMMARY.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_create_method.assert_not_called()


def test_retrieve_user_defined_raw_data_artifact_when_artifact_does_not_exist():
    data_source = MockDataSource()
    with patch.object(Artifact, "list", return_value=[]) as artifact_list_method:
        with patch.object(Artifact, "load", return_value=ARTIFACT_RESULT) as artifact_load_method:
            with patch.object(
                Artifact, "create", return_value=ARTIFACT_RESULT
            ) as artifact_create_method:
                result = S3LineageEntityHandler.retrieve_raw_data_artifact(
                    raw_data=data_source, sagemaker_session=SAGEMAKER_SESSION_MOCK
                )

    assert result == ARTIFACT_RESULT

    artifact_list_method.assert_called_once_with(
        source_uri=data_source.data_source_unique_id, sagemaker_session=SAGEMAKER_SESSION_MOCK
    )

    artifact_load_method.assert_not_called()

    artifact_create_method.assert_called_once_with(
        source_uri=data_source.data_source_unique_id,
        artifact_type="DataSet",
        artifact_name=data_source.data_source_name,
        properties=None,
        source_types=None,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )


def test_create_transformation_code_artifact():
    with patch.object(
        Artifact, "create", return_value=TRANSFORMATION_CODE_ARTIFACT_1
    ) as artifact_create_method:

        result = S3LineageEntityHandler.create_transformation_code_artifact(
            transformation_code=TRANSFORMATION_CODE_INPUT_1,
            pipeline_last_update_time=LAST_UPDATE_TIME,
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

    assert result == TRANSFORMATION_CODE_ARTIFACT_1

    artifact_create_method.assert_called_once_with(
        source_uri=TRANSFORMATION_CODE_INPUT_1.s3_uri,
        source_types=[dict(SourceIdType="Custom", Value=LAST_UPDATE_TIME)],
        artifact_type="TransformationCode",
        artifact_name=f"sm-fs-fe-transformation-code-{LAST_UPDATE_TIME}",
        properties=dict(
            name=TRANSFORMATION_CODE_INPUT_1.name,
            author=TRANSFORMATION_CODE_INPUT_1.author,
            state="Active",
            inclusive_start_date=LAST_UPDATE_TIME,
        ),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )


def test_create_transformation_code_artifact_when_no_author_or_name():
    transformation_code_input = TransformationCode(s3_uri=TRANSFORMATION_CODE_INPUT_1.s3_uri)
    with patch.object(
        Artifact, "create", return_value=TRANSFORMATION_CODE_ARTIFACT_1
    ) as artifact_create_method:

        result = S3LineageEntityHandler.create_transformation_code_artifact(
            transformation_code=transformation_code_input,
            pipeline_last_update_time=LAST_UPDATE_TIME,
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

    assert result == TRANSFORMATION_CODE_ARTIFACT_1

    artifact_create_method.assert_called_once_with(
        source_uri=TRANSFORMATION_CODE_INPUT_1.s3_uri,
        source_types=[dict(SourceIdType="Custom", Value=LAST_UPDATE_TIME)],
        artifact_type="TransformationCode",
        artifact_name=f"sm-fs-fe-transformation-code-{LAST_UPDATE_TIME}",
        properties=dict(
            state="Active",
            inclusive_start_date=LAST_UPDATE_TIME,
        ),
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )


def test_create_transformation_code_artifact_when_no_code_provided():
    with patch.object(
        Artifact, "create", return_value=TRANSFORMATION_CODE_ARTIFACT_1
    ) as artifact_create_method:

        result = S3LineageEntityHandler.create_transformation_code_artifact(
            transformation_code=None,
            pipeline_last_update_time=LAST_UPDATE_TIME,
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

    assert result is None

    artifact_create_method.assert_not_called()


def test_retrieve_pipeline_schedule_artifact_when_artifact_does_not_exist():
    with patch.object(Artifact, "list", return_value=[]) as artifact_list_method:
        with patch.object(Artifact, "load", return_value=ARTIFACT_RESULT) as artifact_load_method:
            with patch.object(
                Artifact, "create", return_value=ARTIFACT_RESULT
            ) as artifact_create_method:
                result = S3LineageEntityHandler.retrieve_pipeline_schedule_artifact(
                    pipeline_schedule=PIPELINE_SCHEDULE,
                    sagemaker_session=SAGEMAKER_SESSION_MOCK,
                )

    assert result == ARTIFACT_RESULT

    artifact_list_method.assert_called_once_with(
        source_uri=PIPELINE_SCHEDULE.schedule_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_load_method.assert_not_called()

    artifact_create_method.assert_called_once_with(
        source_uri=PIPELINE_SCHEDULE.schedule_arn,
        artifact_type="PipelineSchedule",
        artifact_name=f"sm-fs-fe-{PIPELINE_SCHEDULE.schedule_name}",
        properties=dict(
            pipeline_name=PIPELINE_SCHEDULE.pipeline_name,
            schedule_expression=PIPELINE_SCHEDULE.schedule_expression,
            state=PIPELINE_SCHEDULE.state,
            start_date=PIPELINE_SCHEDULE.start_date,
        ),
        source_types=None,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )


def test_retrieve_pipeline_schedule_artifact_when_artifact_exists():
    with patch.object(Artifact, "list", return_value=[ARTIFACT_SUMMARY]) as artifact_list_method:
        with patch.object(
            Artifact, "load", return_value=SCHEDULE_ARTIFACT_RESULT
        ) as artifact_load_method:
            with patch.object(SCHEDULE_ARTIFACT_RESULT, "save") as artifact_save_method:
                with patch.object(
                    Artifact, "create", return_value=SCHEDULE_ARTIFACT_RESULT
                ) as artifact_create_method:
                    result = S3LineageEntityHandler.retrieve_pipeline_schedule_artifact(
                        pipeline_schedule=PIPELINE_SCHEDULE,
                        sagemaker_session=SAGEMAKER_SESSION_MOCK,
                    )

    assert result == SCHEDULE_ARTIFACT_RESULT

    artifact_list_method.assert_called_once_with(
        source_uri=PIPELINE_SCHEDULE.schedule_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_load_method.assert_called_once_with(
        artifact_arn=ARTIFACT_SUMMARY.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_save_method.assert_called_once_with()

    artifact_create_method.assert_not_called()


def test_retrieve_pipeline_schedule_artifact_when_artifact_updated():
    schedule_artifact_result = copy.deepcopy(SCHEDULE_ARTIFACT_RESULT)
    with patch.object(Artifact, "list", return_value=[ARTIFACT_SUMMARY]) as artifact_list_method:
        with patch.object(
            Artifact, "load", return_value=schedule_artifact_result
        ) as artifact_load_method:
            with patch.object(schedule_artifact_result, "save") as artifact_save_method:
                with patch.object(
                    Artifact, "create", return_value=schedule_artifact_result
                ) as artifact_create_method:
                    result = S3LineageEntityHandler.retrieve_pipeline_schedule_artifact(
                        pipeline_schedule=PIPELINE_SCHEDULE_2,
                        sagemaker_session=SAGEMAKER_SESSION_MOCK,
                    )

    assert result == schedule_artifact_result
    assert schedule_artifact_result != SCHEDULE_ARTIFACT_RESULT
    assert result.properties["pipeline_name"] == PIPELINE_SCHEDULE_2.pipeline_name
    assert result.properties["schedule_expression"] == PIPELINE_SCHEDULE_2.schedule_expression
    assert result.properties["state"] == PIPELINE_SCHEDULE_2.state
    assert result.properties["start_date"] == PIPELINE_SCHEDULE_2.start_date

    artifact_list_method.assert_called_once_with(
        source_uri=PIPELINE_SCHEDULE.schedule_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_load_method.assert_called_once_with(
        artifact_arn=ARTIFACT_SUMMARY.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_save_method.assert_called_once_with()

    artifact_create_method.assert_not_called()


def test_retrieve_pipeline_trigger_artifact_when_artifact_does_not_exist():
    with patch.object(Artifact, "list", return_value=[]) as artifact_list_method:
        with patch.object(
            Artifact, "load", return_value=PIPELINE_TRIGGER_ARTIFACT
        ) as artifact_load_method:
            with patch.object(
                Artifact, "create", return_value=PIPELINE_TRIGGER_ARTIFACT
            ) as artifact_create_method:
                result = S3LineageEntityHandler.retrieve_pipeline_trigger_artifact(
                    pipeline_trigger=PIPELINE_TRIGGER,
                    sagemaker_session=SAGEMAKER_SESSION_MOCK,
                )

    assert result == PIPELINE_TRIGGER_ARTIFACT

    artifact_list_method.assert_called_once_with(
        source_uri=PIPELINE_TRIGGER.trigger_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_load_method.assert_not_called()

    artifact_create_method.assert_called_once_with(
        source_uri=PIPELINE_TRIGGER.trigger_arn,
        artifact_type="PipelineTrigger",
        artifact_name=f"sm-fs-fe-trigger-{PIPELINE_TRIGGER.trigger_name}",
        properties=dict(
            pipeline_name=PIPELINE_TRIGGER.pipeline_name,
            event_pattern=PIPELINE_TRIGGER.event_pattern,
            state=PIPELINE_TRIGGER.state,
        ),
        source_types=None,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )


def test_retrieve_pipeline_trigger_artifact_when_artifact_exists():
    with patch.object(
        Artifact, "list", return_value=[PIPELINE_TRIGGER_ARTIFACT_SUMMARY]
    ) as artifact_list_method:
        with patch.object(
            Artifact, "load", return_value=PIPELINE_TRIGGER_ARTIFACT
        ) as artifact_load_method:
            with patch.object(PIPELINE_TRIGGER_ARTIFACT, "save") as artifact_save_method:
                with patch.object(
                    Artifact, "create", return_value=PIPELINE_TRIGGER_ARTIFACT
                ) as artifact_create_method:
                    result = S3LineageEntityHandler.retrieve_pipeline_trigger_artifact(
                        pipeline_trigger=PIPELINE_TRIGGER,
                        sagemaker_session=SAGEMAKER_SESSION_MOCK,
                    )

    assert result == PIPELINE_TRIGGER_ARTIFACT

    artifact_list_method.assert_called_once_with(
        source_uri=PIPELINE_TRIGGER.trigger_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_load_method.assert_called_once_with(
        artifact_arn=PIPELINE_TRIGGER_ARTIFACT_SUMMARY.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_save_method.assert_called_once_with()

    artifact_create_method.assert_not_called()


def test_retrieve_pipeline_trigger_artifact_when_artifact_updated():
    trigger_artifact_result = copy.deepcopy(PIPELINE_TRIGGER_ARTIFACT)
    with patch.object(
        Artifact, "list", return_value=[PIPELINE_TRIGGER_ARTIFACT_SUMMARY]
    ) as artifact_list_method:
        with patch.object(
            Artifact, "load", return_value=trigger_artifact_result
        ) as artifact_load_method:
            with patch.object(trigger_artifact_result, "save") as artifact_save_method:
                with patch.object(
                    Artifact, "create", return_value=trigger_artifact_result
                ) as artifact_create_method:
                    result = S3LineageEntityHandler.retrieve_pipeline_trigger_artifact(
                        pipeline_trigger=PIPELINE_TRIGGER_2,
                        sagemaker_session=SAGEMAKER_SESSION_MOCK,
                    )

    assert result == trigger_artifact_result
    assert trigger_artifact_result != PIPELINE_TRIGGER_ARTIFACT
    assert result.properties["pipeline_name"] == PIPELINE_TRIGGER_2.pipeline_name
    assert result.properties["event_pattern"] == PIPELINE_TRIGGER_2.event_pattern
    assert result.properties["state"] == PIPELINE_TRIGGER_2.state

    artifact_list_method.assert_called_once_with(
        source_uri=PIPELINE_TRIGGER.trigger_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_load_method.assert_called_once_with(
        artifact_arn=PIPELINE_TRIGGER_ARTIFACT_SUMMARY.artifact_arn,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )

    artifact_save_method.assert_called_once_with()

    artifact_create_method.assert_not_called()
