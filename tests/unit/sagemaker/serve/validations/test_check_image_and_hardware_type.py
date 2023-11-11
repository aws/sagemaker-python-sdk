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

import unittest
from unittest.mock import patch

from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.validations.check_image_and_hardware_type import (
    validate_image_uri_and_hardware,
)


CPU_IMAGE_TORCHSERVE = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.0-cpu-py310"
)
GPU_IMAGE_TORCHSERVE = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.0-gpu-py310"
)
CPU_IMAGE_TRITON = (
    "301217895009.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:23.08-py3-cpu"
)
GPU_IMAGE_TRITON = "301217895009.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:23.08-py3"

CPU_INSTANCE = "ml.c5.xlarge"
GPU_INSTANCE = "ml.g4dn.xlarge"


class TestValidateImageAndHardware(unittest.TestCase):
    def test_torchserve_cpu_image_with_cpu_instance(self):

        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=CPU_IMAGE_TORCHSERVE,
                instance_type=CPU_INSTANCE,
                model_server=ModelServer.TORCHSERVE,
            )
            mock_logger.assert_not_called()

    def test_torchserve_gpu_image_with_gpu_instance(self):

        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=GPU_IMAGE_TORCHSERVE,
                instance_type=GPU_INSTANCE,
                model_server=ModelServer.TORCHSERVE,
            )
            mock_logger.assert_not_called()

    def test_torchserve_cpu_image_with_gpu_instance(self):
        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=CPU_IMAGE_TORCHSERVE,
                instance_type=GPU_INSTANCE,
                model_server=ModelServer.TORCHSERVE,
            )

            mock_logger.assert_called_once()

    def test_torchserve_gpu_image_with_cpu_instance(self):
        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=GPU_IMAGE_TORCHSERVE,
                instance_type=CPU_INSTANCE,
                model_server=ModelServer.TORCHSERVE,
            )

            mock_logger.assert_called_once()

    def test_triton_cpu_image_with_cpu_instance(self):

        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=CPU_IMAGE_TRITON,
                instance_type=CPU_INSTANCE,
                model_server=ModelServer.TRITON,
            )
            mock_logger.assert_not_called()

    def test_triton_gpu_image_with_gpu_instance(self):

        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=GPU_IMAGE_TRITON,
                instance_type=GPU_INSTANCE,
                model_server=ModelServer.TRITON,
            )
            mock_logger.assert_not_called()

    def test_triton_cpu_image_with_gpu_instance(self):
        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=CPU_IMAGE_TRITON,
                instance_type=GPU_INSTANCE,
                model_server=ModelServer.TRITON,
            )

            mock_logger.assert_called_once()

    def test_triton_gpu_image_with_cpu_instance(self):
        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=GPU_IMAGE_TRITON,
                instance_type=CPU_INSTANCE,
                model_server=ModelServer.TRITON,
            )

            mock_logger.assert_called_once()
