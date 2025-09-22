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

from mock.mock import patch
import pytest

from sagemaker import image_uris
from sagemaker.jumpstart import accessors
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel

from tests.unit.sagemaker.jumpstart.utils import get_prototype_model_spec


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_sklearn_image_uri(patched_get_model_specs, session):

    patched_get_model_specs.side_effect = get_prototype_model_spec

    model_id, model_version = "sklearn-classification-linear", "*"
    instance_type = "ml.m5.xlarge"
    region = "us-west-2"

    model_specs = accessors.JumpStartModelsAccessor.get_model_specs(region, model_id, model_version)

    # inference
    uri = image_uris.retrieve(
        framework=None,
        region=region,
        image_scope="inference",
        model_id=model_id,
        model_version=model_version,
        instance_type=instance_type,
    )

    framework_class_uri = SKLearnModel(
        role="mock_role",
        model_data="mock_data",
        entry_point="mock_entry_point",
        framework_version=model_specs.hosting_ecr_specs.framework_version,
        py_version=model_specs.hosting_ecr_specs.py_version,
        sagemaker_session=session,
    ).serving_image_uri(region, instance_type)

    # framework classes dont use digest.
    assert (
        framework_class_uri
        == "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-1"
        "-cpu-py3"
    )
    assert (
        uri == "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn@"
        "sha256:a7a5a9097f078d3d0ccb7248ec076ad63e10c82e39a798408f5066450afe82bd"

    # training
    uri = image_uris.retrieve(
        framework=None,
        region=region,
        image_scope="training",
        model_id=model_id,
        model_version=model_version,
        instance_type=instance_type,
    )

    framework_class_uri = SKLearn(
        role="mock_role",
        entry_point="mock_entry_point",
        framework_version=model_specs.training_ecr_specs.framework_version,
        py_version=model_specs.training_ecr_specs.py_version,
        instance_type=instance_type,
        instance_count=1,
        image_uri_region=region,
        sagemaker_session=session,
    ).training_image_uri(region=region)

    # framework classes dont use digest.
    assert (
        framework_class_uri
        == "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-1-cpu-py3"
    )
    assert (
        uri == "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn"
        "@sha256:a7a5a9097f078d3d0ccb7248ec076ad63e10c82e39a798408f5066450afe82bd"
    )

    with pytest.raises(ValueError):
        image_uris.retrieve(
            framework=None,
            region="us-west-2",
            image_scope="training",
            model_id=model_id,
            model_version=model_version,
            instance_type="ml.p2.xlarge",
        )
