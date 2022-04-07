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
"""This module stores notebook utils related to SageMaker JumpStart."""
from __future__ import absolute_import

from sagemaker.jumpstart import accessors
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME


def get_model_url(
    model_id: str, model_version: str, region: str = JUMPSTART_DEFAULT_REGION_NAME
) -> str:
    """Retrieve web url describing pretrained model.

    Args:
        model_id (str): The model ID for which to retrieve the url.
        model_version (str): The model version for which to retrieve the url.
        region (str): Optional. The region from which to retrieve metadata.
            (Default: JUMPSTART_DEFAULT_REGION_NAME)
    """

    model_specs = accessors.JumpStartModelsAccessor.get_model_specs(
        region=region, model_id=model_id, version=model_version
    )
    return model_specs.url
