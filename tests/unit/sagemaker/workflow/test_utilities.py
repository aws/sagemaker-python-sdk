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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import tempfile

from sagemaker.workflow.pipeline_context import _PipelineConfig
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.utilities import (
    hash_file,
    hash_files_or_dirs,
    trim_request_dict,
    _collect_parameters,
)
from pathlib import Path
import unittest


def test_hash_file():
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write("hashme".encode())
        hash = hash_file(tmp.name)
        assert hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hash_file_uri():
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write("hashme".encode())
        hash = hash_file(f"file:///{tmp.name}")
        assert hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_hash_files_or_dirs_with_file():
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write("hashme".encode())
        hash1 = hash_files_or_dirs([f"file:///{tmp.name}"])
        # compute hash again with no change to file
        hash2 = hash_files_or_dirs([f"file:///{tmp.name}"])
        assert hash1 == hash2


def test_hash_files_or_dirs_with_directory():
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        file_name = temp_dir / "test.txt"
        file_name.write_text("foo bar")
        hash1 = hash_files_or_dirs([tmpdirname])
        # compute hash again with no change to directory
        hash2 = hash_files_or_dirs([tmpdirname])
        assert hash1 == hash2


def test_hash_files_or_dirs_change_file_content():
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        file_name = temp_dir / "test.txt"
        file_name.write_text("foo bar")
        hash1 = hash_files_or_dirs([tmpdirname])
        # change file content
        file_name.write_text("new text")
        hash2 = hash_files_or_dirs([tmpdirname])
        assert hash1 != hash2


def test_hash_files_or_dirs_rename_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        file_name = temp_dir / "test.txt"
        file_name.write_text("foo bar")
        hash1 = hash_files_or_dirs([tmpdirname])
        # rename file
        file_name.rename(temp_dir / "test1.txt")
        hash2 = hash_files_or_dirs([tmpdirname])
        assert hash1 != hash2


def test_hash_files_or_dirs_add_new_file():
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        file_name = temp_dir / "test.txt"
        file_name.write_text("foo bar")
        hash1 = hash_files_or_dirs([tmpdirname])
        # add new file
        file_name2 = temp_dir / "test2.txt"
        file_name2.write_text("test test")
        hash2 = hash_files_or_dirs([tmpdirname])
        assert hash1 != hash2


def test_hash_files_or_dirs_unsorted_input_list():
    with tempfile.NamedTemporaryFile() as tmp1:
        tmp1.write("hashme".encode())
        with tempfile.NamedTemporaryFile() as tmp2:
            tmp2.write("hashme".encode())
            hash1 = hash_files_or_dirs([tmp1.name, tmp2.name])
            hash2 = hash_files_or_dirs([tmp2.name, tmp1.name])
            assert hash1 == hash2


def test_trim_request_dict():
    config = PipelineDefinitionConfig(use_custom_job_prefix=True)
    _pipeline_config = _PipelineConfig(
        pipeline_name="test",
        step_name="test-step",
        sagemaker_session=None,
        code_hash="123467890",
        config_hash="123467890",
        pipeline_build_time="some_time",
        pipeline_definition_config=config,
    )
    custom_job_prefix = "MyTrainingJobNamePrefix"
    sample_training_job_name = f"{custom_job_prefix}-2023-06-22-23-39-36-766"
    request_dict = {"TrainingJobName": sample_training_job_name}
    assert (
        custom_job_prefix
        == trim_request_dict(
            request_dict=request_dict, job_key="TrainingJobName", config=_pipeline_config
        )["TrainingJobName"]
    )
    request_dict = {"TrainingJobName": custom_job_prefix}
    assert (
        custom_job_prefix
        == trim_request_dict(
            request_dict=request_dict, job_key="TrainingJobName", config=_pipeline_config
        )["TrainingJobName"]
    )
    request_dict = {"NotSupportedJobName": custom_job_prefix}
    # noop in the case our job_key is invalid
    assert request_dict == trim_request_dict(
        request_dict=request_dict, job_key="NotSupportedJobName", config=_pipeline_config
    )


class TestCollectParametersDecorator(unittest.TestCase):
    def setUp(self):
        class BaseClass:
            def __init__(self, param1):
                self.param1 = param1

        class SampleClass(BaseClass):
            @_collect_parameters
            def __init__(self, param1, param2, param3="default_value"):
                super(SampleClass, self).__init__(param1)
                pass

            @_collect_parameters
            def abc(self, param2, param4="default_value", param5=None):
                pass

        self.SampleClass = SampleClass

    def test_collect_parameters_for_init(self):
        obj = self.SampleClass(param1=42, param2="Hello")

        self.assertEqual(obj.param1, 42)
        self.assertEqual(obj.param2, "Hello")
        self.assertEqual(obj.param3, "default_value")

    def test_collect_parameters_override_for_init(self):
        obj = self.SampleClass(param1=42, param2="Hello", param3="new_value")

        self.assertEqual(obj.param1, 42)
        self.assertEqual(obj.param2, "Hello")
        self.assertEqual(obj.param3, "new_value")

    def test_collect_parameters_function(self):
        obj = self.SampleClass(param1=42, param2="Hello", param3="new_value")
        obj.abc(param2="param2_overridden", param5="param5_value")

        self.assertEqual(obj.param1, 42)
        self.assertEqual(obj.param2, "param2_overridden")
        self.assertEqual(obj.param3, "new_value")
        self.assertEqual(obj.param4, "default_value")
        self.assertEqual(obj.param5, "param5_value")
