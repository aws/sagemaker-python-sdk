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
"""This module contains code to test SageMaker ``DatasetArtifact``"""
from __future__ import absolute_import


def test_endpoints(
    sagemaker_session,
    model_artifact_associated_endpoints,
    endpoint_deployment_action_obj,
    endpoint_context_obj,
):

    model_list = model_artifact_associated_endpoints.endpoints()
    for model in model_list:
        assert model.source_arn == endpoint_deployment_action_obj.action_arn
        assert model.destination_arn == endpoint_context_obj.context_arn
        assert model.source_type == endpoint_deployment_action_obj.action_type
        assert model.destination_type == endpoint_context_obj.context_type
