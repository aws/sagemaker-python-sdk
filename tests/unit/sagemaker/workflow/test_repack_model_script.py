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
from sagemaker.workflow import _repack_model

from pathlib import Path
import shutil
import tarfile
import os
import pytest
import time


@pytest.mark.skip(
    reason="""This test operates on the root file system
                            and will likely fail due to permission errors.
                            Temporarily remove this skip decorator and run
                            the test after making changes to _repack_model.py"""
)
def test_repack_entry_point_only(tmp):
    model_name = "xg-boost-model"
    fake_model_path = os.path.join(tmp, model_name)

    # create a fake model
    open(fake_model_path, "w")

    # create model.tar.gz
    model_tar_path = "s3://my-bucket/model-%s.tar.gz" % time.time()
    model_tar_name = model_tar_path.split("/")[-1]
    model_tar_location = os.path.join(tmp, model_tar_name)
    with tarfile.open(model_tar_location, mode="w:gz") as t:
        t.add(fake_model_path, arcname=model_name)

    # move model.tar.gz to /opt/ml/input/data/training
    Path("/opt/ml/input/data/training").mkdir(parents=True, exist_ok=True)
    shutil.move(model_tar_location, os.path.join("/opt/ml/input/data/training", model_tar_name))

    # create files that will be added to model.tar.gz
    create_file_tree(
        "/opt/ml/code",
        [
            "inference.py",
        ],
    )

    # repack
    _repack_model.repack(inference_script="inference.py", model_archive=model_tar_path)

    # /opt/ml/model should now have the original model and the inference script
    assert os.path.exists(os.path.join("/opt/ml/model", model_name))
    assert os.path.exists(os.path.join("/opt/ml/model/code", "inference.py"))


@pytest.mark.skip(
    reason="""This test operates on the root file system
                            and will likely fail due to permission errors.
                            Temporarily remove this skip decorator and run
                            the test after making changes to _repack_model.py"""
)
def test_repack_with_dependencies(tmp):
    model_name = "xg-boost-model"
    fake_model_path = os.path.join(tmp, model_name)

    # create a fake model
    open(fake_model_path, "w")

    # create model.tar.gz
    model_tar_name = "model-%s.tar.gz" % time.time()
    model_tar_location = os.path.join(tmp, model_tar_name)
    with tarfile.open(model_tar_location, mode="w:gz") as t:
        t.add(fake_model_path, arcname=model_name)

    # move model.tar.gz to /opt/ml/input/data/training
    Path("/opt/ml/input/data/training").mkdir(parents=True, exist_ok=True)
    shutil.move(model_tar_location, os.path.join("/opt/ml/input/data/training", model_tar_name))

    # create files that will be added to model.tar.gz
    create_file_tree(
        "/opt/ml/code",
        ["inference.py", "dependencies/a", "bb", "dependencies/some/dir/b"],
    )

    # repack
    _repack_model.repack(
        inference_script="inference.py",
        model_archive=model_tar_name,
        dependencies="dependencies/a bb dependencies/some/dir",
    )

    # /opt/ml/model should now have the original model and the inference script
    assert os.path.exists(os.path.join("/opt/ml/model", model_name))
    assert os.path.exists(os.path.join("/opt/ml/model/code", "inference.py"))
    assert os.path.exists(os.path.join("/opt/ml/model/code/lib", "a"))
    assert os.path.exists(os.path.join("/opt/ml/model/code/lib", "bb"))
    assert os.path.exists(os.path.join("/opt/ml/model/code/lib/dir", "b"))


@pytest.mark.skip(
    reason="""This test operates on the root file system
                            and will likely fail due to permission errors.
                            Temporarily remove this skip decorator and run
                            the test after making changes to _repack_model.py"""
)
def test_repack_with_source_dir_and_dependencies(tmp):
    model_name = "xg-boost-model"
    fake_model_path = os.path.join(tmp, model_name)

    # create a fake model
    open(fake_model_path, "w")

    # create model.tar.gz
    model_tar_name = "model-%s.tar.gz" % time.time()
    model_tar_location = os.path.join(tmp, model_tar_name)
    with tarfile.open(model_tar_location, mode="w:gz") as t:
        t.add(fake_model_path, arcname=model_name)

    # move model.tar.gz to /opt/ml/input/data/training
    Path("/opt/ml/input/data/training").mkdir(parents=True, exist_ok=True)
    shutil.move(model_tar_location, os.path.join("/opt/ml/input/data/training", model_tar_name))

    # create files that will be added to model.tar.gz
    create_file_tree(
        "/opt/ml/code",
        [
            "inference.py",
            "dependencies/a",
            "bb",
            "dependencies/some/dir/b",
            "sourcedir/foo.py",
            "sourcedir/some/dir/a",
        ],
    )

    # repack
    _repack_model.repack(
        inference_script="inference.py",
        model_archive=model_tar_name,
        dependencies="dependencies/a bb dependencies/some/dir",
        source_dir="sourcedir",
    )

    # /opt/ml/model should now have the original model and the inference script
    assert os.path.exists(os.path.join("/opt/ml/model", model_name))
    assert os.path.exists(os.path.join("/opt/ml/model/code", "inference.py"))
    assert os.path.exists(os.path.join("/opt/ml/model/code/lib", "a"))
    assert os.path.exists(os.path.join("/opt/ml/model/code/lib", "bb"))
    assert os.path.exists(os.path.join("/opt/ml/model/code/lib/dir", "b"))
    assert os.path.exists(os.path.join("/opt/ml/model/code/", "foo.py"))
    assert os.path.exists(os.path.join("/opt/ml/model/code/some/dir", "a"))


def create_file_tree(root, tree):
    for file in tree:
        try:
            os.makedirs(os.path.join(root, os.path.dirname(file)))
        except:  # noqa: E722 Using bare except because p2/3 incompatibility issues.
            pass
        with open(os.path.join(root, file), "a") as f:
            f.write(file)


@pytest.fixture()
def tmp(tmpdir):
    yield str(tmpdir)
