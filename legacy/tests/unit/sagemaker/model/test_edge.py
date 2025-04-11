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

import pytest
from mock import Mock

from sagemaker.model import Model

MODEL_DATA = "s3://bucket/model.tar.gz"
MODEL_IMAGE = "mi"

REGION = "us-west-2"

DESCRIBE_EDGE_PACKAGING_JOB_RESPONSE = {
    "EdgePackagingJobStatus": "Completed",
    "ModelArtifact": "s3://output-path/package-model.tar.gz",
}


@pytest.fixture
def sagemaker_session():
    session = Mock(
        boto_region_name=REGION,
        default_bucket_prefix=None,
    )
    # For tests which doesn't verify config file injection, operate with empty config
    session.sagemaker_config = {}
    return session


def _create_model(sagemaker_session=None):
    model = Model(MODEL_IMAGE, MODEL_DATA, role="role", sagemaker_session=sagemaker_session)
    model._compilation_job_name = "compilation-test-name"
    model._is_compiled_model = True
    return model


def test_package_model(sagemaker_session):
    sagemaker_session.wait_for_edge_packaging_job = Mock(
        return_value=DESCRIBE_EDGE_PACKAGING_JOB_RESPONSE
    )
    model = _create_model(sagemaker_session)
    model.package_for_edge(
        output_path="s3://output",
        role="role",
        model_name="model_name",
        model_version="1.0",
    )
    assert model._is_edge_packaged_model is True


def test_package_validates_compiled():
    sagemaker_session.wait_for_edge_packaging_job = Mock(
        return_value=DESCRIBE_EDGE_PACKAGING_JOB_RESPONSE
    )
    sagemaker_session.package_model_for_edge = Mock()
    model = _create_model()
    model._compilation_job_name = None

    with pytest.raises(ValueError) as e:
        model.package_for_edge(
            output_path="s3://output",
            role="role",
            model_name="model_name",
            model_version="1.0",
        )

    assert "You must first compile this model" in str(e)
