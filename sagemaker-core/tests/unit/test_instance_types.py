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
"""``sagemaker.core.instance_types`` lists SageMaker instance types and describes their hardware.

Before this change the module held only the JumpStart ``retrieve`` functions, so
``from sagemaker.core.instance_types import InstanceSpec`` raised ``ImportError``. The only
hardware lookups lived in two private ``ModelBuilder`` methods (``_get_instance_resources``
and ``_infer_accelerator_count_from_instance_type``), each with its own EC2
``describe_instance_types`` call, its own ``ml.`` strip, and no batching.
"""

from __future__ import absolute_import

import logging
from unittest.mock import Mock

import boto3
import pytest
from botocore.exceptions import ClientError, NoRegionError

from sagemaker.core.instance_types import (
    InstanceSpec,
    describe_instance_type,
    describe_instance_types,
    list_instance_types,
)

G5_XLARGE_INFO = {
    "InstanceType": "g5.xlarge",
    "VCpuInfo": {"DefaultVCpus": 4},
    "MemoryInfo": {"SizeInMiB": 16384},
    "GpuInfo": {
        "Gpus": [{"Name": "A10G", "Count": 1, "MemoryInfo": {"SizeInMiB": 24576}}],
        "TotalGpuMemoryInMiB": 24576,
    },
}
P4D_24XLARGE_INFO = {
    "InstanceType": "p4d.24xlarge",
    "VCpuInfo": {"DefaultVCpus": 96},
    "MemoryInfo": {"SizeInMiB": 1179648},
    "GpuInfo": {
        "Gpus": [{"Name": "A100", "Count": 8, "MemoryInfo": {"SizeInMiB": 40960}}],
        "TotalGpuMemoryInMiB": 327680,
    },
}
M5_LARGE_INFO = {
    "InstanceType": "m5.large",
    "VCpuInfo": {"DefaultVCpus": 2},
    "MemoryInfo": {"SizeInMiB": 8192},
}
KNOWN_INFOS = {
    info["InstanceType"]: info for info in (G5_XLARGE_INFO, P4D_24XLARGE_INFO, M5_LARGE_INFO)
}


def _invalid_instance_type_error(names):
    return ClientError(
        {
            "Error": {
                "Code": "InvalidInstanceType",
                "Message": f"The following supplied instance types do not exist: [{names}]",
            }
        },
        "DescribeInstanceTypes",
    )


def _ec2_client(known_infos):
    """An EC2 client stub that, like EC2, rejects a whole request with one unknown name."""

    def describe_instance_types(**request):
        unknown = [name for name in request["InstanceTypes"] if name not in known_infos]
        if unknown:
            raise _invalid_instance_type_error(", ".join(unknown))
        return {"InstanceTypes": [known_infos[name] for name in request["InstanceTypes"]]}

    client = Mock()
    client.describe_instance_types.side_effect = describe_instance_types
    return client


def _session(ec2_client):
    session = Mock(spec=boto3.Session)
    session.client.return_value = ec2_client
    return session


def _requested_names(ec2_client):
    return [
        call.kwargs["InstanceTypes"] for call in ec2_client.describe_instance_types.call_args_list
    ]


def test_list_instance_types_reads_the_enum_of_the_loaded_service_model():
    sagemaker_client = Mock()
    sagemaker_client.meta.service_model.shape_for.return_value.enum = [
        "ml.t2.medium",
        "ml.g5.xlarge",
    ]
    session = _session(sagemaker_client)

    names = list_instance_types(region="eu-west-1", session=session)

    assert names == ["ml.t2.medium", "ml.g5.xlarge"]
    session.client.assert_called_once_with("sagemaker", region_name="eu-west-1")
    sagemaker_client.meta.service_model.shape_for.assert_called_once_with(
        "ProductionVariantInstanceType"
    )


def test_list_instance_types_reads_the_installed_botocore_model_without_a_request():
    session = boto3.Session(
        region_name="us-west-2", aws_access_key_id="unused", aws_secret_access_key="unused"
    )

    names = list_instance_types(session=session)

    assert names[0] == "ml.t2.medium"
    assert "ml.g5.xlarge" in names
    assert all(name.startswith("ml.") for name in names)
    assert len(names) > 200


def test_describe_instance_types_sends_at_most_100_names_per_ec2_request():
    names = [f"ml.c5.{index}xlarge" for index in range(150)]
    known_infos = {
        name[3:]: {
            "InstanceType": name[3:],
            "VCpuInfo": {"DefaultVCpus": 2},
            "MemoryInfo": {"SizeInMiB": 4096},
        }
        for name in names
    }
    ec2_client = _ec2_client(known_infos)

    specs = describe_instance_types(names, session=_session(ec2_client))

    requested = _requested_names(ec2_client)
    assert [len(batch) for batch in requested] == [100, 50]
    assert requested[0][0] == "c5.0xlarge"
    assert requested[1][-1] == "c5.149xlarge"
    assert list(specs) == names
    assert specs["ml.c5.149xlarge"] == InstanceSpec(
        name="ml.c5.149xlarge",
        vcpus=2,
        memory_mib=4096,
        gpu_count=0,
        gpu_memory_mib=None,
        gpu_name=None,
    )


def test_describe_instance_types_maps_gpu_fields_and_strips_the_ml_prefix():
    ec2_client = _ec2_client(KNOWN_INFOS)

    specs = describe_instance_types(
        ["ml.g5.xlarge", "ml.p4d.24xlarge", "ml.m5.large"], session=_session(ec2_client)
    )

    assert _requested_names(ec2_client) == [["g5.xlarge", "p4d.24xlarge", "m5.large"]]
    assert specs["ml.g5.xlarge"] == InstanceSpec(
        name="ml.g5.xlarge",
        vcpus=4,
        memory_mib=16384,
        gpu_count=1,
        gpu_memory_mib=24576,
        gpu_name="A10G",
    )
    assert specs["ml.p4d.24xlarge"] == InstanceSpec(
        name="ml.p4d.24xlarge",
        vcpus=96,
        memory_mib=1179648,
        gpu_count=8,
        gpu_memory_mib=327680,
        gpu_name="A100",
    )
    assert specs["ml.m5.large"] == InstanceSpec(
        name="ml.m5.large",
        vcpus=2,
        memory_mib=8192,
        gpu_count=0,
        gpu_memory_mib=None,
        gpu_name=None,
    )


def test_describe_instance_types_reads_every_page_of_the_ec2_response():
    ec2_client = Mock()
    ec2_client.describe_instance_types.side_effect = [
        {"InstanceTypes": [G5_XLARGE_INFO], "NextToken": "page-2"},
        {"InstanceTypes": [M5_LARGE_INFO]},
    ]

    specs = describe_instance_types(["ml.g5.xlarge", "ml.m5.large"], session=_session(ec2_client))

    assert ec2_client.describe_instance_types.call_args_list[1].kwargs["NextToken"] == "page-2"
    assert specs["ml.g5.xlarge"].gpu_name == "A10G"
    assert specs["ml.m5.large"].vcpus == 2


def test_unknown_instance_type_gets_an_empty_spec_and_a_warning(caplog):
    ec2_client = _ec2_client(KNOWN_INFOS)

    with caplog.at_level(logging.WARNING, logger="sagemaker.core.instance_types"):
        specs = describe_instance_types(
            ["ml.g5.xlarge", "ml.retired.xlarge", "ml.m5.large"], session=_session(ec2_client)
        )

    assert specs["ml.retired.xlarge"] == InstanceSpec(
        name="ml.retired.xlarge",
        vcpus=None,
        memory_mib=None,
        gpu_count=0,
        gpu_memory_mib=None,
        gpu_name=None,
    )
    # After the rejected request the function asks for each name alone, so the known names
    # still resolve.
    assert specs["ml.g5.xlarge"].gpu_count == 1
    assert specs["ml.m5.large"].vcpus == 2
    assert _requested_names(ec2_client) == [
        ["g5.xlarge", "retired.xlarge", "m5.large"],
        ["g5.xlarge"],
        ["retired.xlarge"],
        ["m5.large"],
    ]
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "ml.retired.xlarge" in warnings[0].getMessage()


def test_describe_instance_type_returns_the_one_spec():
    ec2_client = _ec2_client(KNOWN_INFOS)
    session = _session(ec2_client)

    spec = describe_instance_type("ml.g5.xlarge", session=session, region="eu-west-1")

    assert spec.name == "ml.g5.xlarge"
    assert (spec.vcpus, spec.memory_mib, spec.gpu_count) == (4, 16384, 1)
    assert _requested_names(ec2_client) == [["g5.xlarge"]]
    session.client.assert_called_once_with("ec2", region_name="eu-west-1")


def test_describe_instance_types_without_a_region_raises_no_region_error(monkeypatch):
    for variable in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent/aws-config")
    session = boto3.Session(aws_access_key_id="unused", aws_secret_access_key="unused")

    with pytest.raises(NoRegionError):
        describe_instance_types(["ml.g5.xlarge"], session=session)


def test_describe_instance_type_of_an_unknown_type_has_no_hardware():
    spec = describe_instance_type("ml.retired.xlarge", session=_session(_ec2_client(KNOWN_INFOS)))

    assert spec == InstanceSpec("ml.retired.xlarge", None, None, 0, None, None)


def test_describe_instance_types_raises_every_other_ec2_error():
    ec2_client = Mock()
    ec2_client.describe_instance_types.side_effect = ClientError(
        {"Error": {"Code": "UnauthorizedOperation", "Message": "not allowed"}},
        "DescribeInstanceTypes",
    )

    with pytest.raises(ClientError, match="UnauthorizedOperation"):
        describe_instance_types(["ml.g5.xlarge", "ml.m5.large"], session=_session(ec2_client))

    assert ec2_client.describe_instance_types.call_count == 1


def test_describe_instance_types_with_no_names_makes_no_request():
    ec2_client = _ec2_client(KNOWN_INFOS)

    assert describe_instance_types([], session=_session(ec2_client)) == {}
    ec2_client.describe_instance_types.assert_not_called()


def test_instance_spec_is_frozen():
    spec = InstanceSpec("ml.g5.xlarge", 4, 16384, 1, 24576, "A10G")

    with pytest.raises(AttributeError):
        spec.gpu_count = 2
