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

from sagemaker import model_uris

from tests.unit.sagemaker.jumpstart.utils import get_special_model_spec


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_jumpstart_combined_artifacts(patched_get_model_specs):

    patched_get_model_specs.side_effect = get_special_model_spec

    model_id_combined_model_artifact = "huggingface-text2text-flan-t5-xxl-fp16"

    uri = model_uris.retrieve(
        region="us-west-2",
        model_scope="inference",
        model_id=model_id_combined_model_artifact,
        model_version="*",
    )
    assert (
        uri == "s3://jumpstart-cache-prod-us-west-2/huggingface-infer/"
        "prepack/v1.0.0/infer-prepack-huggingface-text2text-flan-t5-xxl-fp16.tar.gz"
    )
