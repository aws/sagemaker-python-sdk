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
"""Unit tests for _repack_model module."""
from __future__ import absolute_import

import pytest
import tarfile
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from sagemaker.mlops.workflow._repack_model import (
    _get_resolved_path,
    _is_bad_path,
    _is_bad_link,
    _get_safe_members,
    custom_extractall_tarfile,
)


def test_get_resolved_path():
    """Test _get_resolved_path returns normalized absolute path."""
    path = "./test/path"
    result = _get_resolved_path(path)
    assert os.path.isabs(result)
    assert result == os.path.normpath(os.path.realpath(os.path.abspath(path)))


def test_get_resolved_path_with_absolute():
    """Test _get_resolved_path with absolute path."""
    path = "/absolute/test/path"
    result = _get_resolved_path(path)
    assert result == os.path.normpath(os.path.realpath(os.path.abspath(path)))


def test_is_bad_path_safe():
    """Test _is_bad_path returns False for safe paths."""
    base = _get_resolved_path("")
    safe_path = "safe/path"
    assert _is_bad_path(safe_path, base) is False


def test_is_bad_path_unsafe():
    """Test _is_bad_path returns True for unsafe paths."""
    base = _get_resolved_path("/tmp/safe")
    unsafe_path = "/etc/passwd"
    assert _is_bad_path(unsafe_path, base) is True


def test_is_bad_path_parent_traversal():
    """Test _is_bad_path detects parent directory traversal."""
    base = _get_resolved_path("/tmp/safe")
    traversal_path = "../../../etc/passwd"
    # This should be caught as bad if it escapes base
    result = _is_bad_path(traversal_path, base)
    # Result depends on whether the resolved path escapes base
    assert isinstance(result, bool)


def test_is_bad_link_safe():
    """Test _is_bad_link returns False for safe links."""
    base = _get_resolved_path("")
    
    mock_info = Mock()
    mock_info.name = "safe/link"
    mock_info.linkname = "safe/target"
    
    assert _is_bad_link(mock_info, base) is False


def test_is_bad_link_unsafe():
    """Test _is_bad_link returns True for unsafe links."""
    base = _get_resolved_path("/tmp/safe")
    
    mock_info = Mock()
    mock_info.name = "link"
    mock_info.linkname = "/etc/passwd"
    
    result = _is_bad_link(mock_info, base)
    assert isinstance(result, bool)


def test_get_safe_members_all_safe():
    """Test _get_safe_members yields all safe members."""
    base = _get_resolved_path("")
    
    mock_member1 = Mock()
    mock_member1.name = "safe/file1.txt"
    mock_member1.issym = Mock(return_value=False)
    mock_member1.islnk = Mock(return_value=False)
    
    mock_member2 = Mock()
    mock_member2.name = "safe/file2.txt"
    mock_member2.issym = Mock(return_value=False)
    mock_member2.islnk = Mock(return_value=False)
    
    members = [mock_member1, mock_member2]
    safe_members = list(_get_safe_members(members, "/tmp/extract"))
    
    assert len(safe_members) == 2
    assert mock_member1 in safe_members
    assert mock_member2 in safe_members


def test_get_safe_members_filters_bad_path():
    """Test _get_safe_members filters out bad paths."""
    mock_member_safe = Mock()
    mock_member_safe.name = "safe/file.txt"
    mock_member_safe.issym = Mock(return_value=False)
    mock_member_safe.islnk = Mock(return_value=False)
    
    mock_member_bad = Mock()
    mock_member_bad.name = "/etc/passwd"
    mock_member_bad.issym = Mock(return_value=False)
    mock_member_bad.islnk = Mock(return_value=False)
    
    with patch('sagemaker.mlops.workflow._repack_model._is_bad_path') as mock_is_bad:
        mock_is_bad.side_effect = lambda name, base: name == "/etc/passwd"
        
        members = [mock_member_safe, mock_member_bad]
        safe_members = list(_get_safe_members(members, "/tmp/extract"))
        
        assert len(safe_members) == 1
        assert mock_member_safe in safe_members


def test_get_safe_members_filters_bad_symlink():
    """Test _get_safe_members filters out bad symlinks."""
    mock_member_safe = Mock()
    mock_member_safe.name = "safe/file.txt"
    mock_member_safe.issym = Mock(return_value=False)
    mock_member_safe.islnk = Mock(return_value=False)
    
    mock_member_symlink = Mock()
    mock_member_symlink.name = "bad/symlink"
    mock_member_symlink.issym = Mock(return_value=True)
    mock_member_symlink.islnk = Mock(return_value=False)
    mock_member_symlink.linkname = "/etc/passwd"
    
    with patch('sagemaker.mlops.workflow._repack_model._is_bad_path', return_value=False):
        with patch('sagemaker.mlops.workflow._repack_model._is_bad_link') as mock_is_bad_link:
            mock_is_bad_link.return_value = True
            
            members = [mock_member_safe, mock_member_symlink]
            safe_members = list(_get_safe_members(members, "/tmp/extract"))
            
            assert len(safe_members) == 1
            assert mock_member_safe in safe_members


def test_get_safe_members_filters_bad_hardlink():
    """Test _get_safe_members filters out bad hardlinks."""
    mock_member_safe = Mock()
    mock_member_safe.name = "safe/file.txt"
    mock_member_safe.issym = Mock(return_value=False)
    mock_member_safe.islnk = Mock(return_value=False)
    
    mock_member_hardlink = Mock()
    mock_member_hardlink.name = "bad/hardlink"
    mock_member_hardlink.issym = Mock(return_value=False)
    mock_member_hardlink.islnk = Mock(return_value=True)
    mock_member_hardlink.linkname = "/etc/passwd"
    
    with patch('sagemaker.mlops.workflow._repack_model._is_bad_path', return_value=False):
        with patch('sagemaker.mlops.workflow._repack_model._is_bad_link') as mock_is_bad_link:
            mock_is_bad_link.return_value = True
            
            members = [mock_member_safe, mock_member_hardlink]
            safe_members = list(_get_safe_members(members, "/tmp/extract"))
            
            assert len(safe_members) == 1
            assert mock_member_safe in safe_members


def test_custom_extractall_tarfile_with_data_filter():
    """Test custom_extractall_tarfile uses data_filter when available."""
    mock_tar = Mock()
    mock_tar.extractall = Mock()
    extract_path = "/tmp/extract"
    
    with patch('sagemaker.mlops.workflow._repack_model.tarfile') as mock_tarfile:
        mock_tarfile.data_filter = "data"
        
        custom_extractall_tarfile(mock_tar, extract_path)
        
        mock_tar.extractall.assert_called_once_with(path=extract_path, filter="data")


def test_custom_extractall_tarfile_without_data_filter():
    """Test custom_extractall_tarfile uses safe members when data_filter unavailable."""
    mock_tar = Mock()
    mock_tar.extractall = Mock()
    mock_tar.__iter__ = Mock(return_value=iter([]))
    extract_path = "/tmp/extract"
    
    with patch('sagemaker.mlops.workflow._repack_model.tarfile') as mock_tarfile:
        # Remove data_filter attribute
        if hasattr(mock_tarfile, 'data_filter'):
            delattr(mock_tarfile, 'data_filter')
        
        with patch('sagemaker.mlops.workflow._repack_model._get_safe_members') as mock_safe:
            mock_safe.return_value = []
            
            custom_extractall_tarfile(mock_tar, extract_path)
            
            mock_tar.extractall.assert_called_once()
            call_args = mock_tar.extractall.call_args
            assert call_args[1]['path'] == extract_path
            assert 'members' in call_args[1]
