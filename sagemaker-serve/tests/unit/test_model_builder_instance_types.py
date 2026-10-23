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
"""``ModelBuilder`` reads instance hardware through ``sagemaker.core.instance_types``.

Before this change ``_get_instance_resources`` and ``_infer_accelerator_count_from_instance_type``
each made their own EC2 ``describe_instance_types`` call with its own ``ml.`` strip.
``sagemaker.serve.model_builder`` had no ``describe_instance_type`` to patch, and the two
private copies could drift from each other and from the public module.
"""

from __future__ import absolute_import

import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from sagemaker.core.instance_types import InstanceSpec
from sagemaker.core.resources import TrainingJob
from sagemaker.serve.model_builder import ModelBuilder

G5_12XLARGE_INFO = {
    "InstanceType": "g5.12xlarge",
    "VCpuInfo": {"DefaultVCpus": 48},
    "MemoryInfo": {"SizeInMiB": 196608},
    "GpuInfo": {"Gpus": [{"Name": "A10G", "Count": 4, "MemoryInfo": {"SizeInMiB": 24576}}]},
}
M5_XLARGE_INFO = {
    "InstanceType": "m5.xlarge",
    "VCpuInfo": {"DefaultVCpus": 4},
    "MemoryInfo": {"SizeInMiB": 16384},
}
UNKNOWN_TYPE_ERROR = ClientError(
    {"Error": {"Code": "InvalidInstanceType", "Message": "retired.xlarge does not exist"}},
    "DescribeInstanceTypes",
)


class ModelBuilderInstanceTypesTest(unittest.TestCase):
    """The two hardware helpers delegate to ``describe_instance_type`` and keep their fallbacks."""

    def setUp(self):
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-east-1"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-east-1"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.ec2_client = Mock()
        self.mock_session.boto_session.client.return_value = self.ec2_client
        self.builder = ModelBuilder(
            model=Mock(spec=TrainingJob),
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session,
        )

    def _ec2_returns(self, *infos):
        self.ec2_client.describe_instance_types.return_value = {"InstanceTypes": list(infos)}

    @patch("sagemaker.serve.model_builder.describe_instance_type")
    def test_get_instance_resources_passes_the_boto_session_to_the_public_helper(self, describe):
        describe.return_value = InstanceSpec("ml.g5.12xlarge", 48, 196608, 4, 98304, "A10G")

        self.assertEqual(self.builder._get_instance_resources("ml.g5.12xlarge"), (48, 196608))
        describe.assert_called_once_with("ml.g5.12xlarge", session=self.mock_session.boto_session)

    def test_get_instance_resources_reads_vcpus_and_memory_from_ec2(self):
        self._ec2_returns(G5_12XLARGE_INFO)

        self.assertEqual(self.builder._get_instance_resources("ml.g5.12xlarge"), (48, 196608))
        self.ec2_client.describe_instance_types.assert_called_once_with(
            InstanceTypes=["g5.12xlarge"]
        )

    def test_get_instance_resources_falls_back_to_none_for_an_unknown_type(self):
        self.ec2_client.describe_instance_types.side_effect = UNKNOWN_TYPE_ERROR

        self.assertEqual(self.builder._get_instance_resources("ml.retired.xlarge"), (None, None))

    def test_get_instance_resources_falls_back_to_none_when_the_call_fails(self):
        self.ec2_client.describe_instance_types.side_effect = ClientError(
            {"Error": {"Code": "UnauthorizedOperation", "Message": "not allowed"}},
            "DescribeInstanceTypes",
        )

        self.assertEqual(self.builder._get_instance_resources("ml.g5.12xlarge"), (None, None))

    def test_infer_accelerator_count_sums_the_gpus(self):
        self._ec2_returns(G5_12XLARGE_INFO)

        self.assertEqual(
            self.builder._infer_accelerator_count_from_instance_type("ml.g5.12xlarge"), 4
        )
        self.assertTrue(self.builder._is_gpu_instance("ml.g5.12xlarge"))

    def test_infer_accelerator_count_is_none_for_a_cpu_instance(self):
        self._ec2_returns(M5_XLARGE_INFO)

        self.assertIsNone(self.builder._infer_accelerator_count_from_instance_type("ml.m5.xlarge"))
        self.assertFalse(self.builder._is_gpu_instance("ml.m5.xlarge"))

    def test_infer_accelerator_count_is_none_for_an_unknown_type(self):
        self.ec2_client.describe_instance_types.side_effect = UNKNOWN_TYPE_ERROR

        self.assertIsNone(
            self.builder._infer_accelerator_count_from_instance_type("ml.retired.xlarge")
        )
        self.assertFalse(self.builder._is_gpu_instance("ml.retired.xlarge"))
