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

from sagemaker import image_uris

REGION = "us-east-1"
GPU_INSTANCE_TYPE = "ml.p2.xlarge"
NEURONX_INSTANCE_TYPE = "ml.trn1.2xlarge"


def get_full_gpu_image_uri(
    version,
    base_framework_version,
    region=REGION,
    instance_type=GPU_INSTANCE_TYPE,
):
    return image_uris.retrieve(
        "huggingface",
        region,
        version=version,
        instance_type=instance_type,
        image_scope="training",
        base_framework_version=base_framework_version,
        container_version="cu110-ubuntu18.04",
    )


def get_full_neuronx_image_uri(
    version,
    base_framework_version,
    region=REGION,
    instance_type=NEURONX_INSTANCE_TYPE,
):
    return image_uris.retrieve(
        "huggingface",
        region,
        version=version,
        instance_type=instance_type,
        image_scope="training",
        base_framework_version=base_framework_version,
        container_version="cu110-ubuntu18.04",
        inference_tool="neuronx",
    )
