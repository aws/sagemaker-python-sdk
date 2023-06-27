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

from sagemaker.feature_store.feature_processor.lineage._feature_group_lineage_entity_handler import (
    FeatureGroupLineageEntityHandler,
)
from sagemaker.lineage.context import Context

from test_constants import (
    SAGEMAKER_SESSION_MOCK,
    CONTEXT_MOCK_01,
    CONTEXT_MOCK_02,
    FEATURE_GROUP,
    FEATURE_GROUP_NAME,
)


def test_retrieve_feature_group_context_arns():
    with patch.object(
        SAGEMAKER_SESSION_MOCK, "describe_feature_group", return_value=FEATURE_GROUP
    ) as fg_describe_method:
        with patch.object(
            Context, "load", side_effect=[CONTEXT_MOCK_01, CONTEXT_MOCK_02]
        ) as context_load:
            type(CONTEXT_MOCK_01).context_arn = "context-arn-fep"
            type(CONTEXT_MOCK_02).context_arn = "context-arn-fep-ver"
            result = FeatureGroupLineageEntityHandler.retrieve_feature_group_context_arns(
                feature_group_name=FEATURE_GROUP_NAME,
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            )

    assert result.name == FEATURE_GROUP_NAME
    assert result.pipeline_context_arn == "context-arn-fep"
    assert result.pipeline_version_context_arn == "context-arn-fep-ver"
    fg_describe_method.assert_called_once_with(feature_group_name=FEATURE_GROUP_NAME)
    context_load.assert_has_calls(
        [
            call(
                context_name=f'{FEATURE_GROUP_NAME}-{FEATURE_GROUP["CreationTime"].strftime("%s")}'
                f"-feature-group-pipeline",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
            call(
                context_name=f'{FEATURE_GROUP_NAME}-{FEATURE_GROUP["CreationTime"].strftime("%s")}'
                f"-feature-group-pipeline-version",
                sagemaker_session=SAGEMAKER_SESSION_MOCK,
            ),
        ]
    )
    assert 2 == context_load.call_count
