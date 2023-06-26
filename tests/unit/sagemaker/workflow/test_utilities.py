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
from sagemaker.workflow.utilities import (
    hash_file,
    hash_files_or_dirs,
    strip_timestamp_from_job_name,
)
from pathlib import Path


def test_hash_file():
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write("hashme".encode())
        hash = hash_file(tmp.name)
        assert hash == "d41d8cd98f00b204e9800998ecf8427e"


def test_hash_file_uri():
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write("hashme".encode())
        hash = hash_file(f"file:///{tmp.name}")
        assert hash == "d41d8cd98f00b204e9800998ecf8427e"


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


def test_strip_timestamp_from_job_name():
    custom_job_prefix = "MyTrainingJobNamePrefix"
    sample_training_job_name = f"{custom_job_prefix}-2023-06-22-23-39-36-766"
    request_dict = {"TrainingJobName": sample_training_job_name}
    assert (
        custom_job_prefix
        == strip_timestamp_from_job_name(request_dict=request_dict, job_key="TrainingJobName")[
            "TrainingJobName"
        ]
    )
    request_dict = {"TrainingJobName": custom_job_prefix}
    assert (
        custom_job_prefix
        == strip_timestamp_from_job_name(request_dict=request_dict, job_key="TrainingJobName")[
            "TrainingJobName"
        ]
    )
    request_dict = {
        "NotSupportedJobName": custom_job_prefix
    }  # do nothing in the case our jobKey is invalid
    assert request_dict == strip_timestamp_from_job_name(
        request_dict=request_dict, job_key="NotSupportedJobName"
    )
