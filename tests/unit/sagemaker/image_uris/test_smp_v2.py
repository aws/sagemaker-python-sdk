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
from sagemaker import image_uris
from tests.unit.sagemaker.image_uris import expected_uris

CONTAINER_VERSIONS = {"ml.p4d.24xlarge": "cu118", "ml.p5.24xlarge": "cu121"}


@pytest.mark.parametrize("load_config", ["pytorch-smp.json"], indirect=True)
def test_smp_v2(load_config):
    VERSIONS = load_config["training"]["versions"]
    PROCESSORS = load_config["training"]["processors"]
    distribution = {
        "torch_distributed": {"enabled": True},
        "smdistributed": {"modelparallel": {"enabled": True}},
    }
    for processor in PROCESSORS:
        for version in VERSIONS:
            ACCOUNTS = load_config["training"]["versions"][version]["registries"]
            PY_VERSIONS = load_config["training"]["versions"][version]["py_versions"]
            for py_version in PY_VERSIONS:
                for region in ACCOUNTS.keys():
                    for instance_type in CONTAINER_VERSIONS.keys():
                        cuda_vers = CONTAINER_VERSIONS[instance_type]
                        if "2.1" in version or "2.2" in version:
                            cuda_vers = "cu121"

                        uri = image_uris.get_training_image_uri(
                            region,
                            framework="pytorch",
                            framework_version=version,
                            py_version=py_version,
                            distribution=distribution,
                            instance_type=instance_type,
                        )
                        expected = expected_uris.framework_uri(
                            repo="smdistributed-modelparallel",
                            fw_version=version,
                            py_version=f"{py_version}-{cuda_vers}",
                            processor=processor,
                            region=region,
                            account=ACCOUNTS[region],
                        )
                        assert expected == uri
