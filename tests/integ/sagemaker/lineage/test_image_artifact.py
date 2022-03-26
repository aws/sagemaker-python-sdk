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
"""This module contains code to test SageMaker ``ImageArtifact``"""
from __future__ import absolute_import

import pytest

from sagemaker.lineage.query import LineageQueryDirectionEnum


@pytest.mark.skip("data inconsistency P61661075")
def test_dataset(static_image_artifact, sagemaker_session):
    artifacts_from_query = static_image_artifact.datasets(
        direction=LineageQueryDirectionEnum.DESCENDANTS
    )
    assert len(artifacts_from_query) > 0
    for artifact in artifacts_from_query:
        assert artifact.artifact_type == "DataSet"
        assert "artifact" in artifact.artifact_arn
