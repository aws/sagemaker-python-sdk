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
from mock import patch, call
import pytest

from sagemaker.mlops.feature_store.feature_processor.lineage._lineage_association_handler import (
    LineageAssociationHandler,
)
from sagemaker.core.lineage.association import Association
from botocore.exceptions import ClientError

from test_constants import (
    FEATURE_GROUP_INPUT,
    RAW_DATA_INPUT_ARTIFACTS,
    VALIDATION_EXCEPTION,
    NON_VALIDATION_EXCEPTION,
    SAGEMAKER_SESSION_MOCK,
    TRANSFORMATION_CODE_ARTIFACT_1,
)


def test_add_upstream_feature_group_data_associations():
    with patch.object(Association, "create") as create_association_method:
        LineageAssociationHandler.add_upstream_feature_group_data_associations(
            feature_group_inputs=FEATURE_GROUP_INPUT,
            pipeline_context_arn="pipeline-context-arn",
            pipeline_version_context_arn="pipeline-version-context-arn",
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

    for feature_group in FEATURE_GROUP_INPUT:
        create_association_method.assert_has_calls(
            [
                call(
                    source_arn=feature_group.pipeline_context_arn,
                    destination_arn="pipeline-context-arn",
                    association_type="ContributedTo",
                    sagemaker_session=SAGEMAKER_SESSION_MOCK,
                ),
                call(
                    source_arn=feature_group.pipeline_version_context_arn,
                    destination_arn="pipeline-version-context-arn",
                    association_type="ContributedTo",
                    sagemaker_session=SAGEMAKER_SESSION_MOCK,
                ),
            ]
        )
        assert len(FEATURE_GROUP_INPUT) * 2 == create_association_method.call_count


def test_add_upstream_feature_group_data_associations_when_association_already_exists():
    with patch.object(
        Association, "create", side_effect=VALIDATION_EXCEPTION
    ) as create_association_method:
        LineageAssociationHandler.add_upstream_feature_group_data_associations(
            feature_group_inputs=FEATURE_GROUP_INPUT,
            pipeline_context_arn="pipeline-context-arn",
            pipeline_version_context_arn="pipeline-version-context-arn",
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

    for feature_group in FEATURE_GROUP_INPUT:
        create_association_method.assert_has_calls(
            [
                call(
                    source_arn=feature_group.pipeline_context_arn,
                    destination_arn="pipeline-context-arn",
                    association_type="ContributedTo",
                    sagemaker_session=SAGEMAKER_SESSION_MOCK,
                ),
                call(
                    source_arn=feature_group.pipeline_version_context_arn,
                    destination_arn="pipeline-version-context-arn",
                    association_type="ContributedTo",
                    sagemaker_session=SAGEMAKER_SESSION_MOCK,
                ),
            ]
        )
        assert len(FEATURE_GROUP_INPUT) * 2 == create_association_method.call_count


def test_add_upstream_feature_group_data_associations_when_non_validation_exception():
    with patch.object(Association, "create", side_effect=NON_VALIDATION_EXCEPTION):
        with pytest.raises(ClientError):
            LineageAssociationHandler.add_upstream_feature_group_data_associations(
                feature_group_inputs=FEATURE_GROUP_INPUT,
                pipeline_context_arn="pipeline-context-arn",
                pipeline_version_context_arn="pipeline-version-context-arn",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            )


def test_add_upstream_raw_data_associations():
    with patch.object(Association, "create") as create_association_method:
        LineageAssociationHandler.add_upstream_raw_data_associations(
            raw_data_inputs=RAW_DATA_INPUT_ARTIFACTS,
            pipeline_context_arn="pipeline-context-arn",
            pipeline_version_context_arn="pipeline-version-context-arn",
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

    for raw_data in RAW_DATA_INPUT_ARTIFACTS:
        create_association_method.assert_has_calls(
            [
                call(
                    source_arn=raw_data.artifact_arn,
                    destination_arn="pipeline-context-arn",
                    association_type="ContributedTo",
                    sagemaker_session=SAGEMAKER_SESSION_MOCK,
                ),
                call(
                    source_arn=raw_data.artifact_arn,
                    destination_arn="pipeline-version-context-arn",
                    association_type="ContributedTo",
                    sagemaker_session=SAGEMAKER_SESSION_MOCK,
                ),
            ]
        )
        assert len(RAW_DATA_INPUT_ARTIFACTS) * 2 == create_association_method.call_count


def test_add_upstream_transformation_code_associations():
    with patch.object(Association, "create") as create_association_method:
        LineageAssociationHandler.add_upstream_transformation_code_associations(
            transformation_code_artifact=TRANSFORMATION_CODE_ARTIFACT_1,
            pipeline_version_context_arn="pipeline-version-context-arn",
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

    create_association_method.assert_called_once_with(
        source_arn=TRANSFORMATION_CODE_ARTIFACT_1.artifact_arn,
        destination_arn="pipeline-version-context-arn",
        association_type="ContributedTo",
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )


def test_add_downstream_feature_group_data_associations():
    with patch.object(Association, "create") as create_association_method:
        LineageAssociationHandler.add_downstream_feature_group_data_associations(
            feature_group_output=FEATURE_GROUP_INPUT[0],
            pipeline_context_arn="pipeline-context-arn",
            pipeline_version_context_arn="pipeline-version-context-arn",
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

    create_association_method.assert_has_calls(
        [
            call(
                source_arn="pipeline-context-arn",
                destination_arn=FEATURE_GROUP_INPUT[0].pipeline_context_arn,
                association_type="ContributedTo",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                source_arn="pipeline-version-context-arn",
                destination_arn=FEATURE_GROUP_INPUT[0].pipeline_version_context_arn,
                association_type="ContributedTo",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 2 == create_association_method.call_count


def test_add_pipeline_and_pipeline_version_association():
    with patch.object(Association, "create") as create_association_method:
        LineageAssociationHandler.add_pipeline_and_pipeline_version_association(
            pipeline_context_arn="pipeline-context-arn",
            pipeline_version_context_arn="pipeline-version-context-arn",
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

    create_association_method.assert_called_once_with(
        source_arn="pipeline-context-arn",
        destination_arn="pipeline-version-context-arn",
        association_type="AssociatedWith",
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )


def test_list_upstream_associations():
    with patch.object(Association, "list") as list_association_method:
        LineageAssociationHandler.list_upstream_associations(
            entity_arn="pipeline-context-arn",
            source_type="FeatureEngineeringPipelineVersion",
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

    list_association_method.assert_called_once_with(
        source_arn=None,
        source_type="FeatureEngineeringPipelineVersion",
        destination_arn="pipeline-context-arn",
        destination_type=None,
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )


def test_list_downstream_associations():
    with patch.object(Association, "list") as list_association_method:
        LineageAssociationHandler.list_downstream_associations(
            entity_arn="pipeline-context-arn",
            destination_type="FeatureEngineeringPipelineVersion",
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

    list_association_method.assert_called_once_with(
        source_arn="pipeline-context-arn",
        source_type=None,
        destination_arn=None,
        destination_type="FeatureEngineeringPipelineVersion",
        sagemaker_session=SAGEMAKER_SESSION_MOCK,
    )
