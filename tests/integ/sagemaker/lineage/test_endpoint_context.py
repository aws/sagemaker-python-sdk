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
"""This module contains code to test SageMaker ``Contexts``"""
from __future__ import absolute_import


def test_model(
    endpoint_context_associate_with_model,
    model_obj,
    endpoint_action_obj,
    sagemaker_session,
):
    model_list = endpoint_context_associate_with_model.models()
    for model in model_list:
        assert model.source_arn == endpoint_action_obj.action_arn
        assert model.destination_arn == model_obj.context_arn
        assert model.source_type == "ModelDeployment"
        assert model.destination_type == "Model"
