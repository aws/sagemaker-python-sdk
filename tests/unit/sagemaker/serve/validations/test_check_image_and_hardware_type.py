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
GRAVITON_IMAGE_TORCHSERVE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-inference-graviton:2.1.0-cpu-py310-ubuntu20.04-sagemaker"
)
INF1_IMAGE_TORCHSERVE = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com"
    "/pytorch-inference-neuron:1.13.1-neuron-py310-sdk2.15.0-ubuntu20.04"
)

INF2_IMAGE_TORCHSERVE = (
    "763104351884.dkr.ecr.us-west-2.amazonaws.com"
    "/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.15.0-ubuntu20.04"
)

CPU_INSTANCE = "ml.c5.xlarge"
GPU_INSTANCE = "ml.g4dn.xlarge"
INF1_INSTANCE = "ml.inf1.xlarge"
INF2_INSTANCE = "ml.inf2.xlarge"
GRAVITON_INSTANCE = "ml.c7g.xlarge"


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

    def test_torchserve_inf1_image_with_inf1_instance(self):

        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=INF1_IMAGE_TORCHSERVE,
                instance_type=INF1_INSTANCE,
                model_server=ModelServer.TORCHSERVE,
            )
            mock_logger.assert_not_called()

    def test_torchserve_inf2_image_with_inf2_instance(self):

        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=INF2_IMAGE_TORCHSERVE,
                instance_type=INF2_INSTANCE,
                model_server=ModelServer.TORCHSERVE,
            )
            mock_logger.assert_not_called()

    def test_torchserve_graviton_image_with_graviton_instance(self):

        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=GRAVITON_IMAGE_TORCHSERVE,
                instance_type=GRAVITON_INSTANCE,
                model_server=ModelServer.TORCHSERVE,
            )
            mock_logger.assert_not_called()

    def test_torchserve_inf1_image_with_cpu_instance(self):

        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=INF1_IMAGE_TORCHSERVE,
                instance_type=CPU_INSTANCE,
                model_server=ModelServer.TORCHSERVE,
            )
            mock_logger.assert_called_once()

    def test_torchserve_graviton_image_with_cpu_instance(self):

        with patch("logging.Logger.warning") as mock_logger:
            validate_image_uri_and_hardware(
                image_uri=GRAVITON_IMAGE_TORCHSERVE,
                instance_type=CPU_INSTANCE,
                model_server=ModelServer.TORCHSERVE,
            )
            mock_logger.assert_called_once()
