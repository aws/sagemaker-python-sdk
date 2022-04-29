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
"""This module contains code to test ``sagemaker.workflow.pipeline_session.PipelineSession``"""
from __future__ import absolute_import

from sagemaker.workflow.pipeline_context import PipelineSession

from botocore.config import Config


def test_pipeline_session_init(sagemaker_client_config, boto_session):
    sagemaker_client_config.setdefault("config", Config(retries=dict(max_attempts=10)))
    sagemaker_client = (
        boto_session.client("sagemaker", **sagemaker_client_config)
        if sagemaker_client_config
        else None
    )

    sess = PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
    )
    assert sess.sagemaker_client is not None
    assert sess.default_bucket() is not None
    assert sess.context is None
