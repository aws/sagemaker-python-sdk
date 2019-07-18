# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import filecmp
import os
import random
import shutil

from sagemaker.tensorflow.estimator import Tensorboard


def create_test_directory(directory, variable_content="hello world"):
    """Create dummy data for testing Tensorboard._sync_directories with the
    following structure:

    <directory>
    |_ child_directory
        |_ hello.txt
    |_ foo1.txt
    |_ foo2.txt

    Args:
        directory (str): The path to a directory to create with fake files
        variable_content (str): Content to put in one of the files
    """
    child_dir = os.path.join(directory, "child_directory")
    os.mkdir(child_dir)
    with open(os.path.join(directory, "foo1.txt"), "w") as f:
        f.write("bar1")
    with open(os.path.join(directory, "foo2.txt"), "w") as f:
        f.write("bar2")
    with open(os.path.join(child_dir, "hello.txt"), "w") as f:
        f.write(variable_content)


def same_dirs(a, b):
    """Check that structure and files are the same for directories a and b

    Args:
        a (str): The path to the first directory
        b (str): The path to the second directory
    """
    comp = filecmp.dircmp(a, b)
    common = sorted(comp.common)
    left = sorted(comp.left_list)
    right = sorted(comp.right_list)
    if left != common or right != common:
        return False
    if len(comp.diff_files):
        return False
    for subdir in comp.common_dirs:
        left_subdir = os.path.join(a, subdir)
        right_subdir = os.path.join(b, subdir)
        return same_dirs(left_subdir, right_subdir)
    return True


def test_to_directory_doesnt_exist():
    with Tensorboard._temporary_directory() as from_dir:
        create_test_directory(from_dir)
        to_dir = "./not_a_real_place_{}".format(random.getrandbits(64))
        Tensorboard._sync_directories(from_dir, to_dir)
        assert same_dirs(from_dir, to_dir)
        shutil.rmtree(to_dir)


def test_only_root_of_to_directory_exists():
    with Tensorboard._temporary_directory() as from_dir:
        with Tensorboard._temporary_directory() as to_dir:
            create_test_directory(from_dir)
            assert not same_dirs(from_dir, to_dir)
            Tensorboard._sync_directories(from_dir, to_dir)
            assert same_dirs(from_dir, to_dir)


def test_files_are_overwritten_when_they_already_exist():
    with Tensorboard._temporary_directory() as from_dir:
        with Tensorboard._temporary_directory() as to_dir:
            create_test_directory(from_dir)
            create_test_directory(to_dir, "foo bar")
            assert not same_dirs(from_dir, to_dir)
            Tensorboard._sync_directories(from_dir, to_dir)
            assert same_dirs(from_dir, to_dir)
