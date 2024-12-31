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
from unittest.mock import patch, mock_open, call
from pathlib import Path

from sagemaker.serve.detector.dependency_manager import _parse_dependency_list, capture_dependencies


DEPENDENCY_LIST = [
    "requests==2.26.0",
    "numpy==2.0",
    "pandas==2.2.3",
    "matplotlib<3.5.0",
    "scikit-learn>0.24.1",
    "Django!=4.0.0",
    "attrs>=23.1.0,<24",
    "torch@https://download.pytorch.org/whl/cpu/torch-2.0.0%2Bcpu-cp310-cp310-linux_x86_64.whl",
    "# some comment",
    "boto3==1.26.*",
]

EXPECTED_DEPENDENCY_MAP = {
    "requests": "==2.26.0",
    "numpy": "==2.0",
    "pandas": "==2.2.3",
    "matplotlib": "<3.5.0",
    "scikit-learn": ">0.24.1",
    "Django": "!=4.0.0",
    "attrs": ">=23.1.0,<24",
    "torch": "@https://download.pytorch.org/whl/cpu/torch-2.0.0%2Bcpu-cp310-cp310-linux_x86_64.whl",
    "boto3": "==1.26.*",
}

DEPENDENCY_CONFIG = {
    "auto": True,
    "requirements": "/path/to/requirements.txt",
    "custom": ["custom_module==1.2.3", "other_module@http://some/website.whl"],
}

NO_AUTO_DEPENDENCY_CONFIG = {
    "auto": False,
    "requirements": "/path/to/requirements.txt",
    "custom": ["custom_module==1.2.3", "other_module@http://some/website.whl"],
}

WORK_DIR = Path("/path/to/working/dir")

AUTODETECTED_REQUIREMENTS = """module==1.2
custom_module==1.2.0
numpy==2.0
boto3==1.26.135
"""

NO_AUTODETECTED_REQUIREMENTS = """
"""

CUSTOM_REQUIREMENT_FILE = """boto3=1.28.*
"""


class DepedencyManagerTest(unittest.TestCase):
    @patch("sagemaker.serve.detector.dependency_manager.Path")
    @patch("builtins.open", new_callable=mock_open, read_data=AUTODETECTED_REQUIREMENTS)
    @patch("sagemaker.serve.detector.dependency_manager.subprocess")
    def test_capture_dependencies(self, mock_subprocess, mock_file, mock_path):
        mock_open_custom_file = mock_open(read_data=CUSTOM_REQUIREMENT_FILE)
        mock_open_write = mock_open(read_data="")
        handlers = (
            mock_file.return_value,
            mock_open_custom_file.return_value,
            mock_open_write.return_value,
        )
        mock_file.side_effect = handlers

        mock_path.is_file.return_value = True

        capture_dependencies(dependencies=DEPENDENCY_CONFIG, work_dir=WORK_DIR, capture_all=False)
        mock_subprocess.run.assert_called_once()

        mocked_writes = mock_open_write.return_value.__enter__().write

        assert 6 == mocked_writes.call_count

        expected_calls = [
            call("module==1.2\n"),
            call("custom_module==1.2.3\n"),
            call("numpy==2.0\n"),
            call("boto3=1.28.*\n"),
            call("sagemaker[huggingface]>=2.199\n"),
            call("other_module@http://some/website.whl\n"),
        ]
        mocked_writes.assert_has_calls(expected_calls)

    @patch("sagemaker.serve.detector.dependency_manager.Path")
    @patch("builtins.open", new_callable=mock_open, read_data=NO_AUTODETECTED_REQUIREMENTS)
    @patch("sagemaker.serve.detector.dependency_manager.subprocess")
    def test_capture_dependencies_no_auto_detect(self, mock_path, mock_file, mock_subprocess):
        mock_open_custom_file = mock_open(read_data=CUSTOM_REQUIREMENT_FILE)
        mock_open_write = mock_open(read_data="")
        handlers = (
            mock_open_custom_file.return_value,
            mock_open_write.return_value,
        )
        mock_file.side_effect = handlers

        mock_path.is_file.return_value = True

        capture_dependencies(
            dependencies=NO_AUTO_DEPENDENCY_CONFIG, work_dir=WORK_DIR, capture_all=False
        )
        mock_subprocess.run().assert_not_called()
        mocked_writes = mock_open_write.return_value.__enter__().write

        assert 4 == mocked_writes.call_count

        expected_calls = [
            call("boto3=1.28.*\n"),
            call("custom_module==1.2.3\n"),
            call("other_module@http://some/website.whl\n"),
        ]
        mocked_writes.assert_has_calls(expected_calls)

    def test_parse_dependency_list(self):
        dependency_map = _parse_dependency_list(DEPENDENCY_LIST)

        for key, value in dependency_map.items():
            self.assertTrue(key in EXPECTED_DEPENDENCY_MAP.keys())
            self.assertEqual(value, EXPECTED_DEPENDENCY_MAP.get(key))
