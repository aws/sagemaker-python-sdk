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

from sagemaker.core import image_uris
from sagemaker.core.common_utils import ALTERNATE_DOMAINS

GPU_INSTANCE = "ml.g5.2xlarge"
CPU_INSTANCE = "ml.m5.xlarge"
DEFAULT_DOMAIN = "amazonaws.com"

# The amzn2023 unified `pytorch` repo (framework "pytorch-amzn2023"). Its GPU tag
# encodes CUDA directly (e.g. 2.13-cu133-amzn2023-sagemaker) with no "gpu" token,
# so these versions set "processor_in_tag": false and bake the accelerator into
# container_version.
AMZN2023_VERSIONS = ["2.11", "2.12", "2.13"]


@pytest.mark.parametrize("load_config", ["pytorch-amzn2023.json"], indirect=True)
def test_pytorch_amzn2023_training_uris(load_config):
    """pytorch-amzn2023 resolves both cpu and gpu; the gpu tag carries cuNNN, not "gpu"."""
    training = load_config["training"]
    assert sorted(training["versions"]) == sorted(AMZN2023_VERSIONS)
    for version in AMZN2023_VERSIONS:
        version_config = training["versions"][version]
        assert version_config["repository"] == "pytorch"
        assert version_config["processor_in_tag"] is False
        container_version = version_config["container_version"]
        for region, account in version_config["registries"].items():
            domain = ALTERNATE_DOMAINS.get(region, DEFAULT_DOMAIN)

            gpu_uri = image_uris.retrieve(
                framework="pytorch-amzn2023",
                region=region,
                version=version,
                image_scope="training",
                instance_type=GPU_INSTANCE,
            )
            assert gpu_uri == (
                f"{account}.dkr.ecr.{region}.{domain}"
                f"/pytorch:{version}-{container_version['gpu']}"
            )
            assert "-gpu-" not in gpu_uri

            cpu_uri = image_uris.retrieve(
                framework="pytorch-amzn2023",
                region=region,
                version=version,
                image_scope="training",
                instance_type=CPU_INSTANCE,
            )
            assert cpu_uri == (
                f"{account}.dkr.ecr.{region}.{domain}"
                f"/pytorch:{version}-{container_version['cpu']}"
            )


def test_pytorch_training_default_stays_ubuntu():
    """The `pytorch` (Ubuntu) training default must NOT move onto the amzn2023 repo:
    with no version, retrieve() resolves to a pytorch-training image, not `pytorch`."""
    uri = image_uris.retrieve(
        framework="pytorch",
        region="us-west-2",
        image_scope="training",
        instance_type=GPU_INSTANCE,
    )
    assert "/pytorch-training:" in uri
    assert "-amzn2023-" not in uri


def test_pytorch_training_new_ubuntu_versions():
    """The newly added Ubuntu training versions resolve on pytorch-training."""
    for version, py in [("2.8.0", "py312"), ("2.9.0", "py312"), ("2.10.0", "py313")]:
        uri = image_uris.retrieve(
            framework="pytorch",
            region="us-west-2",
            version=version,
            image_scope="training",
            instance_type=GPU_INSTANCE,
        )
        assert uri.endswith(f"/pytorch-training:{version}-gpu-{py}")
