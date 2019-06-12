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

import os

from sagemaker.git_utils import git_clone_repo_and_enter

GIT_REPO = 'https://github.com/GaryTu1020/python-sdk-testing.git'
BRANCH = 'branch1'
COMMIT = '4893e528afa4a790331e1b5286954f073b0f14a2'


def test_git_clone_repo_and_enter():
    git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
    entry_point = 'entry_point'
    source_dir = 'source_dir'
    dependencies = ['foo', 'foo/bar']
    git_clone_repo_and_enter(git_config, entry_point, source_dir, dependencies)
    assert os.path.isfile(os.path.join(source_dir, entry_point))
    for directory in dependencies:
        assert os.path.exists(directory)
