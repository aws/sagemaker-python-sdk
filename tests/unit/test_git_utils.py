# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
import subprocess
from mock import patch

from sagemaker import git_utils

REPO_DIR = '/tmp/repo_dir'
GIT_REPO = 'https://github.com/GaryTu1020/python-sdk-testing.git'
BRANCH = 'branch1'
COMMIT = 'b61c450200d6a309c8d24ac14b8adddc405acc56'


@patch('subprocess.check_call')
@patch('tempfile.mkdtemp', return_value=REPO_DIR)
@patch('sagemaker.git_utils._validate_git_config')
@patch('sagemaker.git_utils._checkout_branch_and_commit')
@patch('os.path.isfile', return_value=True)
@patch('os.path.isdir', return_value=True)
@patch('os.path.exists', return_value=True)
def test_git_clone_repo_succeed(exists, isdir, isfile, checkout_branch_and_commit,
                                validate_git_config, mkdtemp, check_call):
    git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
    entry_point = 'entry_point'
    source_dir = 'source_dir'
    dependencies = ['foo', 'bar']
    ret = git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    validate_git_config.assert_called_with(git_config)
    check_call.assert_called_with(['git', 'clone', git_config['repo'], REPO_DIR])
    mkdtemp.assert_called_once()
    checkout_branch_and_commit.assert_called_with(git_config, REPO_DIR)
    assert ret['entry_point'] == 'entry_point'
    assert ret['source_dir'] == '/tmp/repo_dir/source_dir'
    assert ret['dependencies'] == ['/tmp/repo_dir/foo', '/tmp/repo_dir/bar']


@patch('subprocess.check_call')
@patch('tempfile.mkdtemp', return_value=REPO_DIR)
@patch('sagemaker.git_utils._validate_git_config',
       side_effect=ValueError('Please provide a repo for git_config.'))
@patch('sagemaker.git_utils._checkout_branch_and_commit')
@patch('os.path.isfile', return_value=True)
@patch('os.path.isdir', return_value=True)
@patch('os.path.exists', return_value=True)
def test_git_clone_repo_repo_not_provided(exists, isdir, isfile, checkout_branch_and_commit,
                                          validate_git_config, mkdtemp, check_call):
    git_config = {'branch': BRANCH, 'commit': COMMIT}
    entry_point = 'entry_point_that_does_not_exist'
    source_dir = 'source_dir'
    dependencies = ['foo', 'bar']
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert 'Please provide a repo for git_config.' in str(error)


@patch('subprocess.check_call',
       side_effect=subprocess.CalledProcessError(returncode=1, cmd='git clone {} {}'.format(GIT_REPO, REPO_DIR)))
@patch('tempfile.mkdtemp', return_value=REPO_DIR)
@patch('sagemaker.git_utils._validate_git_config')
@patch('sagemaker.git_utils._checkout_branch_and_commit')
@patch('os.path.isfile', return_value=True)
@patch('os.path.isdir', return_value=True)
@patch('os.path.exists', return_value=True)
def test_git_clone_repo_clone_fail(exists, isdir, isfile, checkout_branch_and_commit,
                                   validate_git_config, mkdtemp, check_call):
    git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
    entry_point = 'entry_point'
    source_dir = 'source_dir'
    dependencies = ['foo', 'bar']
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert 'returned non-zero exit status' in str(error)


@patch('subprocess.check_call')
@patch('tempfile.mkdtemp', return_value=REPO_DIR)
@patch('sagemaker.git_utils._validate_git_config')
@patch('sagemaker.git_utils._checkout_branch_and_commit',
       side_effect=subprocess.CalledProcessError(returncode=1, cmd='git checkout {}'.format(BRANCH)))
@patch('os.path.isfile', return_value=True)
@patch('os.path.isdir', return_value=True)
@patch('os.path.exists', return_value=True)
def test_git_clone_repo_branch_not_exist(exists, isdir, isfile, checkout_branch_and_commit,
                                         validate_git_config, mkdtemp, check_call):
    git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
    entry_point = 'entry_point'
    source_dir = 'source_dir'
    dependencies = ['foo', 'bar']
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert 'returned non-zero exit status' in str(error)


@patch('subprocess.check_call')
@patch('tempfile.mkdtemp', return_value=REPO_DIR)
@patch('sagemaker.git_utils._validate_git_config')
@patch('sagemaker.git_utils._checkout_branch_and_commit',
       side_effect=subprocess.CalledProcessError(returncode=1, cmd='git checkout {}'.format(COMMIT)))
@patch('os.path.isfile', return_value=True)
@patch('os.path.isdir', return_value=True)
@patch('os.path.exists', return_value=True)
def test_git_clone_repo_commit_not_exist(exists, isdir, isfile, checkout_branch_and_commit,
                                         validate_git_config, mkdtemp, check_call):
    git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
    entry_point = 'entry_point'
    source_dir = 'source_dir'
    dependencies = ['foo', 'bar']
    with pytest.raises(subprocess.CalledProcessError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert 'returned non-zero exit status' in str(error)


@patch('subprocess.check_call')
@patch('tempfile.mkdtemp', return_value=REPO_DIR)
@patch('sagemaker.git_utils._validate_git_config')
@patch('sagemaker.git_utils._checkout_branch_and_commit')
@patch('os.path.isfile', return_value=False)
@patch('os.path.isdir', return_value=True)
@patch('os.path.exists', return_value=True)
def test_git_clone_repo_entry_point_not_exist(exists, isdir, isfile, checkout_branch_and_commit,
                                              validate_git_config, mkdtemp, check_call):
    git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
    entry_point = 'entry_point_that_does_not_exist'
    source_dir = 'source_dir'
    dependencies = ['foo', 'bar']
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert 'Entry point does not exist in the repo.' in str(error)


@patch('subprocess.check_call')
@patch('tempfile.mkdtemp', return_value=REPO_DIR)
@patch('sagemaker.git_utils._validate_git_config')
@patch('sagemaker.git_utils._checkout_branch_and_commit')
@patch('os.path.isfile', return_value=True)
@patch('os.path.isdir', return_value=False)
@patch('os.path.exists', return_value=True)
def test_git_clone_repo_source_dir_not_exist(exists, isdir, isfile, checkout_branch_and_commit,
                                             validate_git_config, mkdtemp, check_call):
    git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
    entry_point = 'entry_point'
    source_dir = 'source_dir_that_does_not_exist'
    dependencies = ['foo', 'bar']
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert 'Source directory does not exist in the repo.' in str(error)


@patch('subprocess.check_call')
@patch('tempfile.mkdtemp', return_value=REPO_DIR)
@patch('sagemaker.git_utils._validate_git_config')
@patch('sagemaker.git_utils._checkout_branch_and_commit')
@patch('os.path.isfile', return_value=True)
@patch('os.path.isdir', return_value=True)
@patch('os.path.exists', side_effect=[True, False])
def test_git_clone_repo_dependencies_not_exist(exists, isdir, isfile, checkout_branch_and_commit,
                                               validate_git_config, mkdtemp, check_call):
    git_config = {'repo': GIT_REPO, 'branch': BRANCH, 'commit': COMMIT}
    entry_point = 'entry_point'
    source_dir = 'source_dir'
    dependencies = ['foo', 'dep_that_does_not_exist']
    with pytest.raises(ValueError) as error:
        git_utils.git_clone_repo(git_config, entry_point, source_dir, dependencies)
    assert 'does not exist in the repo.' in str(error)
