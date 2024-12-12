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

from sagemaker import image_uris
from sagemaker.huggingface.estimator import HuggingFace
from sagemaker.jumpstart import accessors
from sagemaker.huggingface.model import HuggingFaceModel

from tests.unit.sagemaker.jumpstart.utils import get_prototype_model_spec


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_huggingface_image_uri(patched_get_model_specs, session):

    patched_get_model_specs.side_effect = get_prototype_model_spec

    model_id, model_version = "huggingface-spc-bert-base-cased", "*"
    instance_type = "ml.m5.xlarge"
    training_instance_type = "ml.p3.2xlarge"
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

    framework_class_uri = HuggingFaceModel(
        role="mock_role",
        transformers_version=model_specs.hosting_ecr_specs.huggingface_transformers_version,
        pytorch_version=model_specs.hosting_ecr_specs.framework_version,
        py_version=model_specs.hosting_ecr_specs.py_version,
        sagemaker_session=session,
    ).serving_image_uri(region, instance_type)

    assert uri == framework_class_uri

    assert (
        uri == "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:"
        "1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04"
    )

    # training
    uri = image_uris.retrieve(
        framework=None,
        region=region,
        image_scope="training",
        model_id=model_id,
        model_version=model_version,
        instance_type=training_instance_type,
    )

    framework_class_uri = HuggingFace(
        role="mock_role",
        region=region,
        py_version=model_specs.training_ecr_specs.py_version,
        entry_point="some_entry_point",
        transformers_version=model_specs.training_ecr_specs.huggingface_transformers_version,
        pytorch_version=model_specs.training_ecr_specs.framework_version,
        instance_type=training_instance_type,
        instance_count=1,
        sagemaker_session=session,
    ).training_image_uri(region=region)

    assert (
        uri == "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:"
        "1.6.0-transformers4.4.2-gpu-py36-cu110-ubuntu18.04"
    )

    assert uri == framework_class_uri
