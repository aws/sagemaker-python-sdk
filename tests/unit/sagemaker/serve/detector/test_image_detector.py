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
import sys
import logging
import unittest
from typing import Tuple
from mock import patch, MagicMock, Mock
from sagemaker.serve.detector.image_detector import auto_detect_container

logger = logging.getLogger(__name__)

REGION = "us-west-2"
CPU = "ml.c5.xlarge"
GPU = "ml.g5dn.xlarge"


class xgboost:
    pass


class torch:
    pass


class tensorflow:
    pass


class keras(tensorflow):
    pass


class TestImageDetector(unittest.TestCase):
    # Test xgboost Success
    def test_detect_latest_downcast_xgb(self):
        xgb_model = Mock()
        xgb_model.__class__.__bases__ = (xgboost,)
        self.assert_dlc_is_expected(
            "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:1.7-1", xgb_model
        )

    @patch.dict(sys.modules, {"xgboost": MagicMock(__version__="1.6.5")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_earliest_upcast_xgb(self, platform):
        platform.python_version_tuple.return_value = (3, 8)
        xgb_model = Mock()
        xgb_model.__class__.__bases__ = (xgboost,)
        self.assert_dlc_is_expected(
            "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:1.7-1", xgb_model
        )

    @patch.dict(sys.modules, {"xgboost": MagicMock(__version__="1.5.1")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_exact_match_xgb(self, platform):
        platform.python_version_tuple.return_value = (3, 8)
        xgb_model = Mock()
        xgb_model.__class__.__bases__ = (xgboost,)
        self.assert_dlc_is_expected(
            "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:1.5-1", xgb_model
        )

    # Test Torch Success
    @patch.dict(sys.modules, {"torch": MagicMock(__version__="2.0.1")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_latest_downcast_pt(self, platform):
        platform.python_version_tuple.return_value = (3, 10)
        pt_model = Mock()
        pt_model.__class__.__bases__ = (torch,)
        self.assert_dlc_is_expected(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
            pt_model,
        )

    @patch.dict(sys.modules, {"torch": MagicMock(__version__="2.1.0")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_latest_downcast_pt_2_1_0(self, platform):
        platform.python_version_tuple.return_value = (3, 10)
        pt_model = Mock()
        pt_model.__class__.__bases__ = (torch,)
        self.assert_dlc_is_expected(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.1-cpu-py310",
            pt_model,
        )

    @patch.dict(sys.modules, {"torch": MagicMock(__version__="1.13.0")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_earliest_upcast_pt(self, platform):
        platform.python_version_tuple.return_value = (3, 9)
        pt_model = Mock()
        pt_model.__class__.__bases__ = (torch,)
        self.assert_dlc_is_expected(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.1-cpu-py38",
            pt_model,
        )

    @patch.dict(sys.modules, {"torch": MagicMock(__version__="2.0.0")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_exact_match_pt(self, platform):
        platform.python_version_tuple.return_value = (3, 10)
        pt_model = Mock()
        pt_model.__class__.__bases__ = (torch,)
        self.assert_dlc_is_expected(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.0-cpu-py310",
            pt_model,
        )

    # Test Torch GPU
    @patch.dict(sys.modules, {"torch": MagicMock(__version__="2.0.1")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_latest_downcast_pt_gpu(self, platform):
        platform.python_version_tuple.return_value = (3, 10)
        pt_model = Mock()
        pt_model.__class__.__bases__ = (torch,)
        self.assert_dlc_is_expected(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.1-gpu-py310",
            pt_model,
            GPU,
        )

    @patch.dict(sys.modules, {"torch": MagicMock(__version__="1.13.0")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_earliest_upcast_pt_gpu(self, platform):
        platform.python_version_tuple.return_value = (3, 9)
        pt_model = Mock()
        pt_model.__class__.__bases__ = (torch,)
        self.assert_dlc_is_expected(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.13.1-gpu-py39",
            pt_model,
            GPU,
        )

    @patch.dict(sys.modules, {"torch": MagicMock(__version__="2.0.0")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_exact_match_pt_gpu(self, platform):
        platform.python_version_tuple.return_value = (3, 10)
        pt_model = Mock()
        pt_model.__class__.__bases__ = (torch,)
        self.assert_dlc_is_expected(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0.0-gpu-py310",
            pt_model,
            GPU,
        )

    # Test Tensorflow success
    @patch.dict(sys.modules, {"tensorflow": MagicMock(__version__="2.13.0")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_latest_exact_match_tf(self, platform):
        platform.python_version_tuple.return_value = (3, 10)
        tf_model = Mock()
        tf_model.__class__.__bases__ = (keras,)
        self.assert_dlc_is_expected(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.13.0-cpu",
            tf_model,
        )

    # Test Tensorflow GPU
    @patch.dict(sys.modules, {"tensorflow": MagicMock(__version__="2.13.0")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_latest_exact_match_tf_gpu(self, platform):
        platform.python_version_tuple.return_value = (3, 10)
        tf_model = Mock()
        tf_model.__class__.__bases__ = (keras,)
        self.assert_dlc_is_expected(
            "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.13.0-gpu",
            tf_model,
            GPU,
        )

    # failure cases
    @patch.dict(sys.modules, {"torch": MagicMock(__version__="8.0.0")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_unsupported_major_vs(self, platform):
        platform.python_version_tuple.return_value = (3, 8)
        pt_model = Mock()
        pt_model.__class__.__bases__ = (torch,)
        self.assertRaises(ValueError, auto_detect_container, pt_model, REGION, CPU)

    @patch.dict(sys.modules, {"torch": MagicMock(__version__="2.0.0")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_unsupported_python_vs(self, platform):
        platform.python_version_tuple.return_value = (3, 12)
        pt_model = Mock()
        pt_model.__class__.__bases__ = (torch,)
        self.assertRaises(ValueError, auto_detect_container, pt_model, REGION, CPU)

    # corner cases
    @patch.dict(sys.modules, {"xgboost": MagicMock(__version__="1.6.5")})
    @patch("sagemaker.serve.detector.image_detector.platform")
    def test_detect_model_no_base(self, platform):
        platform.python_version_tuple.return_value = (3, 8)
        xgb_model = Mock()
        xgb_model.__class__ = xgboost
        self.assert_dlc_is_expected(
            "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:1.7-1", xgb_model
        )

    # helpers
    def assert_dlc_is_expected(self, expected_dlc, model, instance_type=CPU):
        # TODO: Only check the major verison and relax the minor version
        dlc = auto_detect_container(model, REGION, instance_type)
        expected_ecr_uri, expected_repo, expected_tag = self.parse_image_uri(expected_dlc)
        actual_ecr_uri, actual_repo, actual_tag = self.parse_image_uri(dlc)

        self.assertEqual(expected_ecr_uri, actual_ecr_uri)
        self.assertEqual(expected_repo, actual_repo)
        self.assertEqual(expected_tag[0], actual_tag[0])

    def parse_image_uri(self, image_uri: str) -> Tuple:
        ecr_uri, repo_tag = image_uri.split("/")
        repo, tag = repo_tag.split(":")

        return (ecr_uri, repo, tag)
