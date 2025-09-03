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
from sagemaker.serve.detector.pickle_dependencies import get_requirements_for_pkl_file

import boto3
import json
import io
import sys
import subprocess
from mock import MagicMock, mock_open, patch, call
from types import ModuleType

import logging

logger = logging.getLogger(__name__)


def create_mock_modules(name, doc, file):
    m = ModuleType(name, doc)
    m.__file__ = file
    return m


INSTALLED_PKG_JSON = [
    {"name": "sagemaker", "version": "1.2.3"},
    {"name": "boto3", "version": "1.2.3"},
    {"name": "xgboost", "version": "1.2.3"},
    {"name": "scipy", "version": "1.2.3"},
]
INSTALLED_PKG_LST = ["sagemaker", "boto3", "xgboost", "scipy"]
INSTALLED_PKG_JSON_UNUSED = INSTALLED_PKG_JSON + [{"name": "unused", "version": "1.2.3"}]
INSTALLED_PKG_LST_WITH_UNUSED = INSTALLED_PKG_LST + ["unused"]
INSTALLED_PKG_READLINE_LST = [
    "Name: sagemaker",
    "Location: /tmp/to/site-packages/sagemaker",
    "Files: sagemaker/__init__.py",
    "---",
    "Name: boto3",
    "Location: /tmp/to/site-packages/boto3",
    "Files: boto3/__init__.py",
    "---",
    "Name: xgboost",
    "Location: /tmp/to/site-packages/xgboost",
    "Files: xgboost/__init__.py",
    "---",
    "Name: scipy",
    "Location: /tmp/to/site-packages/scipy",
    "Files: scipy/__init__.py",
]
INSTALLED_PKG_READLINE = io.BytesIO("\n".join(INSTALLED_PKG_READLINE_LST).encode("utf-8"))
INSTALLED_PKG_READLINE_UNUSED_LST = INSTALLED_PKG_READLINE_LST + [
    "---",
    "Name: unused",
    "Location: /tmp/to/site-packages/unused",
    "Files: unused/__init__.py",
]
INSTALLED_PKG_READLINE_UNUSED = io.BytesIO(
    "\n".join(INSTALLED_PKG_READLINE_UNUSED_LST).encode("utf-8")
)
CURRENTLY_USED_FILES = {
    "sagemaker": create_mock_modules(
        "sagemaker",
        "mock sagemaker module",
        "/tmp/to/site-packages/sagemaker/sagemaker/__init__.py",
    ),
    "boto3": create_mock_modules(
        "boto3", "mock boto3 module", "/tmp/to/site-packages/boto3/boto3/__init__.py"
    ),
    "xgboost": create_mock_modules(
        "xgboost",
        "mock xgboost module",
        "/tmp/to/site-packages/xgboost/xgboost/__init__.py",
    ),
    "scipy": create_mock_modules(
        "scipy", "mock scipy module", "/tmp/to/site-packages/scipy/scipy/__init__.py"
    ),
}
EXPECTED_SM_WHL = "/opt/ml/model/whl/sagemaker-2.197.1.dev0-py2.py3-none-any.whl"
EXPECTED_BOTO3_MAPPING = f"boto3=={boto3.__version__}"
EXPECTED_SUBPROCESS_CMD = [sys.executable, "-m", "pip", "--disable-pip-version-check"]


# happy case
def test_generate_requirements_exact_match(monkeypatch):
    with (
        patch("cloudpickle.load"),
        patch("tqdm.tqdm"),
        patch("sagemaker.serve.detector.pickle_dependencies.subprocess.run") as subprocess_run,
        patch("sagemaker.serve.detector.pickle_dependencies.subprocess.Popen") as subprocess_popen,
        patch("builtins.open") as mocked_open,
        monkeypatch.context() as m,
    ):
        mock_run_stdout = MagicMock()
        mock_run_stdout.stdout = json.dumps(INSTALLED_PKG_JSON).encode("utf-8")
        subprocess_run.return_value = mock_run_stdout

        mock_popen_stdout = MagicMock()
        mock_popen_stdout.stdout = INSTALLED_PKG_READLINE
        subprocess_popen.return_value = mock_popen_stdout

        m.setattr(sys, "modules", CURRENTLY_USED_FILES)

        open_dest_file = mock_open()
        mocked_open.side_effect = [
            mock_open().return_value,
            open_dest_file.return_value,
        ]

        get_requirements_for_pkl_file("/path/to/serve.pkl", "path/to/requirements.txt")

        mocked_open.assert_any_call("/path/to/serve.pkl", mode="rb")
        mocked_open.assert_any_call("path/to/requirements.txt", mode="w+")

        assert 2 == subprocess_run.call_count
        subprocess_run_call = call(
            EXPECTED_SUBPROCESS_CMD + ["list", "--format", "json"],
            stdout=subprocess.PIPE,
            check=True,
        )
        subprocess_run.assert_has_calls([subprocess_run_call, subprocess_run_call])

        subprocess_popen.assert_called_once_with(
            EXPECTED_SUBPROCESS_CMD + ["show", "-f"] + INSTALLED_PKG_LST,
            stdout=subprocess.PIPE,
        )

        mocked_writes = open_dest_file.return_value.__enter__().write
        assert 4 == mocked_writes.call_count

        expected_calls = [
            call("sagemaker==1.2.3\n"),
            call(f"{EXPECTED_BOTO3_MAPPING}\n"),
            call("xgboost==1.2.3\n"),
            call("scipy==1.2.3\n"),
        ]
        mocked_writes.assert_has_calls(expected_calls)


def test_generate_requirements_txt_pruning_unused_packages(monkeypatch):
    with (
        patch("cloudpickle.load"),
        patch("tqdm.tqdm"),
        patch("sagemaker.serve.detector.pickle_dependencies.subprocess.run") as subprocess_run,
        patch("sagemaker.serve.detector.pickle_dependencies.subprocess.Popen") as subprocess_popen,
        patch("builtins.open") as mocked_open,
        monkeypatch.context() as m,
    ):
        mock_run_stdout = MagicMock()
        mock_run_stdout.stdout = json.dumps(INSTALLED_PKG_JSON_UNUSED).encode("utf-8")
        subprocess_run.return_value = mock_run_stdout

        mock_popen_stdout = MagicMock()
        mock_popen_stdout.stdout = INSTALLED_PKG_READLINE_UNUSED
        subprocess_popen.return_value = mock_popen_stdout

        m.setattr(sys, "modules", CURRENTLY_USED_FILES)

        open_dest_file = mock_open()
        mocked_open.side_effect = [
            mock_open().return_value,
            open_dest_file.return_value,
        ]

        get_requirements_for_pkl_file("/path/to/serve.pkl", "path/to/requirements.txt")

        mocked_open.assert_any_call("/path/to/serve.pkl", mode="rb")
        mocked_open.assert_any_call("path/to/requirements.txt", mode="w+")

        assert 2 == subprocess_run.call_count
        subprocess_run_call = call(
            EXPECTED_SUBPROCESS_CMD + ["list", "--format", "json"],
            stdout=subprocess.PIPE,
            check=True,
        )
        subprocess_run.assert_has_calls([subprocess_run_call, subprocess_run_call])

        subprocess_popen.assert_called_once_with(
            EXPECTED_SUBPROCESS_CMD + ["show", "-f"] + INSTALLED_PKG_LST_WITH_UNUSED,
            stdout=subprocess.PIPE,
        )

        mocked_writes = open_dest_file.return_value.__enter__().write
        assert 4 == mocked_writes.call_count

        expected_calls = [
            call("sagemaker==1.2.3\n"),
            call(f"{EXPECTED_BOTO3_MAPPING}\n"),
            call("xgboost==1.2.3\n"),
            call("scipy==1.2.3\n"),
        ]
        mocked_writes.assert_has_calls(expected_calls)


def test_generate_requirements_txt_no_currently_used_packages(monkeypatch):
    with (
        patch("cloudpickle.load"),
        patch("tqdm.tqdm"),
        patch("sagemaker.serve.detector.pickle_dependencies.subprocess.run") as subprocess_run,
        patch("sagemaker.serve.detector.pickle_dependencies.subprocess.Popen") as subprocess_popen,
        patch("builtins.open") as mocked_open,
        monkeypatch.context() as m,
    ):
        mock_run_stdout = MagicMock()
        mock_run_stdout.stdout = json.dumps([]).encode("utf-8")
        subprocess_run.return_value = mock_run_stdout

        mock_popen_stdout = MagicMock()
        mock_popen_stdout.stdout = []
        subprocess_popen.return_value = mock_popen_stdout

        m.setattr(sys, "modules", CURRENTLY_USED_FILES)

        open_dest_file = mock_open()
        mocked_open.side_effect = [
            mock_open().return_value,
            open_dest_file.return_value,
        ]

        get_requirements_for_pkl_file("/path/to/serve.pkl", "path/to/requirements.txt")

        mocked_open.assert_any_call("/path/to/serve.pkl", mode="rb")
        mocked_open.assert_any_call("path/to/requirements.txt", mode="w+")

        assert 2 == subprocess_run.call_count
        subprocess_run_call = call(
            EXPECTED_SUBPROCESS_CMD + ["list", "--format", "json"],
            stdout=subprocess.PIPE,
            check=True,
        )
        subprocess_run.assert_has_calls([subprocess_run_call, subprocess_run_call])

        assert 0 == subprocess_popen.call_count

        mocked_writes = open_dest_file.return_value.__enter__().write

        assert 0 == mocked_writes.call_count
