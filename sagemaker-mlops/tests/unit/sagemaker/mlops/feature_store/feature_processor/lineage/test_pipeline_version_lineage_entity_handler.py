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

from mock import patch

from sagemaker.mlops.feature_store.feature_processor.lineage._pipeline_version_lineage_entity_handler import (
    PipelineVersionLineageEntityHandler,
)

from sagemaker.core.lineage.context import Context

from test_constants import (
    PIPELINE_NAME,
    PIPELINE_ARN,
    LAST_UPDATE_TIME,
    SAGEMAKER_SESSION_MOCK,
    CONTEXT_MOCK_01,
)


def test_create_pipeline_version_context():
    with patch.object(Context, "create", return_value=CONTEXT_MOCK_01) as create_method:
        result = PipelineVersionLineageEntityHandler.create_pipeline_version_context(
            pipeline_name=PIPELINE_NAME,
            pipeline_arn=PIPELINE_ARN,
            last_update_time=LAST_UPDATE_TIME,
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

        assert result == CONTEXT_MOCK_01
        create_method.assert_called_with(
            context_name=f"sm-fs-fe-{PIPELINE_NAME}-{LAST_UPDATE_TIME}-fep-ver",
            context_type=f"FeatureEngineeringPipelineVersion-{PIPELINE_NAME}",
            source_uri=PIPELINE_ARN,
            source_type=LAST_UPDATE_TIME,
            properties={
                "PipelineName": PIPELINE_NAME,
                "LastUpdateTime": LAST_UPDATE_TIME,
            },
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )


def test_load_pipeline_version_context():
    with patch.object(Context, "load", return_value=CONTEXT_MOCK_01) as load_method:
        result = PipelineVersionLineageEntityHandler.load_pipeline_version_context(
            pipeline_name=PIPELINE_NAME,
            last_update_time=LAST_UPDATE_TIME,
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )

        assert result == CONTEXT_MOCK_01
        load_method.assert_called_once_with(
            context_name=f"sm-fs-fe-{PIPELINE_NAME}-{LAST_UPDATE_TIME}-fep-ver",
            sagemaker_session=SAGEMAKER_SESSION_MOCK,
        )
