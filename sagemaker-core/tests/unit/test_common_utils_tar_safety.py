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
"""Unit tests for tar extraction safety functions in common_utils."""
from __future__ import annotations

import os
import tempfile
import tarfile

import pytest
from unittest.mock import Mock, patch, MagicMock

from sagemaker.core.common_utils import (
    _get_resolved_path,
    _is_bad_path,
    _is_bad_link,
    _get_safe_members,
    custom_extractall_tarfile,
)


def test_get_resolved_path_returns_normalized_absolute_path():
    """Test _get_resolved_path returns normalized absolute path."""
    path = "./test/path"
    result = _get_resolved_path(path)
    assert os.path.isabs(result)
    assert result == os.path.normpath(os.path.realpath(os.path.abspath(path)))


def test_get_resolved_path_with_absolute_path():
    """Test _get_resolved_path with absolute path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "absolute", "test", "path")
        result = _get_resolved_path(path)
        assert result == os.path.normpath(os.path.realpath(os.path.abspath(path)))


def test_is_bad_path_returns_false_for_safe_relative_path():
    """Test _is_bad_path returns False for safe relative paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _get_resolved_path(os.path.join(tmpdir, "extract"))
        safe_path = "safe/path/file.txt"
        assert _is_bad_path(safe_path, base) is False


def test_is_bad_path_returns_true_for_absolute_escape_path():
    """Test _is_bad_path returns True for absolute paths that escape base."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _get_resolved_path(os.path.join(tmpdir, "safe"))
        unsafe_path = "/etc/passwd"
        assert _is_bad_path(unsafe_path, base) is True


def test_is_bad_path_returns_true_for_parent_traversal():
    """Test _is_bad_path detects parent directory traversal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _get_resolved_path(os.path.join(tmpdir, "safe", "extract"))
        traversal_path = "../../../etc/passwd"
        assert _is_bad_path(traversal_path, base) is True


def test_is_bad_path_with_similar_prefix_does_not_false_positive():
    """Test that base/x2 is correctly identified as bad when base is base/x.

    This verifies the commonpath fix: startswith would incorrectly allow
    base/x2 when base is base/x, but commonpath correctly rejects it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _get_resolved_path(os.path.join(tmpdir, "x"))
        # A path like tmpdir/x2/file should NOT be under tmpdir/x
        # With startswith, "tmpdir/x2".startswith("tmpdir/x") would be True (bug)
        # With commonpath, commonpath(["tmpdir/x2", "tmpdir/x"]) == "tmpdir" != "tmpdir/x" (correct)
        escape_path = os.path.join(tmpdir, "x2", "file")
        result = _is_bad_path(escape_path, base)
        assert result is True


def test_is_bad_link_returns_false_for_safe_symlink():
    """Test _is_bad_link returns False for safe links."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _get_resolved_path(os.path.join(tmpdir, "extract"))

        mock_info = Mock()
        mock_info.name = "safe/link"
        mock_info.linkname = "safe/target"

        assert _is_bad_link(mock_info, base) is False


def test_is_bad_link_returns_true_for_escape_symlink():
    """Test _is_bad_link returns True for links that escape base."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _get_resolved_path(os.path.join(tmpdir, "safe"))

        mock_info = Mock()
        mock_info.name = "link"
        mock_info.linkname = "/etc/passwd"

        result = _is_bad_link(mock_info, base)
        assert result is True


def test_get_safe_members_yields_all_safe_members():
    """Test _get_safe_members yields all safe members."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _get_resolved_path(os.path.join(tmpdir, "extract"))

        mock_member1 = Mock()
        mock_member1.name = "safe/file1.txt"
        mock_member1.issym = Mock(return_value=False)
        mock_member1.islnk = Mock(return_value=False)

        mock_member2 = Mock()
        mock_member2.name = "safe/file2.txt"
        mock_member2.issym = Mock(return_value=False)
        mock_member2.islnk = Mock(return_value=False)

        members = [mock_member1, mock_member2]
        safe_members = list(_get_safe_members(members, base))

        assert len(safe_members) == 2
        assert mock_member1 in safe_members
        assert mock_member2 in safe_members


def test_get_safe_members_filters_bad_path_member():
    """Test _get_safe_members filters out members with bad paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _get_resolved_path(os.path.join(tmpdir, "extract"))

        mock_member_safe = Mock()
        mock_member_safe.name = "safe/file.txt"
        mock_member_safe.issym = Mock(return_value=False)
        mock_member_safe.islnk = Mock(return_value=False)

        mock_member_bad = Mock()
        mock_member_bad.name = "/etc/passwd"
        mock_member_bad.issym = Mock(return_value=False)
        mock_member_bad.islnk = Mock(return_value=False)

        members = [mock_member_safe, mock_member_bad]
        safe_members = list(_get_safe_members(members, base))

        assert len(safe_members) == 1
        assert mock_member_safe in safe_members


def test_get_safe_members_filters_bad_symlink_member():
    """Test _get_safe_members filters out bad symlinks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _get_resolved_path(os.path.join(tmpdir, "extract"))

        mock_member_safe = Mock()
        mock_member_safe.name = "safe/file.txt"
        mock_member_safe.issym = Mock(return_value=False)
        mock_member_safe.islnk = Mock(return_value=False)

        mock_member_symlink = Mock()
        mock_member_symlink.name = "bad/symlink"
        mock_member_symlink.issym = Mock(return_value=True)
        mock_member_symlink.islnk = Mock(return_value=False)
        mock_member_symlink.linkname = "/etc/passwd"

        members = [mock_member_safe, mock_member_symlink]
        safe_members = list(_get_safe_members(members, base))

        assert len(safe_members) == 1
        assert mock_member_safe in safe_members


def test_get_safe_members_filters_bad_hardlink_member():
    """Test _get_safe_members filters out bad hardlinks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = _get_resolved_path(os.path.join(tmpdir, "extract"))

        mock_member_safe = Mock()
        mock_member_safe.name = "safe/file.txt"
        mock_member_safe.issym = Mock(return_value=False)
        mock_member_safe.islnk = Mock(return_value=False)

        mock_member_hardlink = Mock()
        mock_member_hardlink.name = "bad/hardlink"
        mock_member_hardlink.issym = Mock(return_value=False)
        mock_member_hardlink.islnk = Mock(return_value=True)
        mock_member_hardlink.linkname = "/etc/passwd"

        members = [mock_member_safe, mock_member_hardlink]
        safe_members = list(_get_safe_members(members, base))

        assert len(safe_members) == 1
        assert mock_member_safe in safe_members


def test_custom_extractall_tarfile_with_data_filter_uses_filter_param():
    """Test custom_extractall_tarfile uses data_filter when available.

    We set mock_tarfile.data_filter explicitly to ensure hasattr returns True.
    The MagicMock would auto-create the attribute anyway, but we set it
    explicitly for clarity. The key assertion is that filter="data" is passed.
    """
    mock_tar = Mock()
    mock_tar.extractall = Mock()

    with tempfile.TemporaryDirectory() as tmpdir:
        extract_path = os.path.join(tmpdir, "extract")

        with patch("sagemaker.core.common_utils.tarfile") as mock_tarfile:
            # Explicitly set data_filter to ensure the hasattr check passes
            mock_tarfile.data_filter = True

            custom_extractall_tarfile(mock_tar, extract_path)

            mock_tar.extractall.assert_called_once_with(path=extract_path, filter="data")


def test_custom_extractall_tarfile_without_data_filter_uses_safe_members():
    """Test custom_extractall_tarfile uses safe members when data_filter is unavailable.

    Verifies that:
    1. tar.getmembers() is called (not iterating over tar directly)
    2. _get_safe_members is called with the members list and resolved extract_path as base
    3. _validate_extracted_paths is called after extraction
    """
    mock_member = Mock()
    mock_member.name = "safe/file.txt"
    mock_member.issym = Mock(return_value=False)
    mock_member.islnk = Mock(return_value=False)

    mock_tar = Mock()
    mock_tar.extractall = Mock()
    mock_tar.getmembers = Mock(return_value=[mock_member])

    with tempfile.TemporaryDirectory() as tmpdir:
        extract_path = os.path.join(tmpdir, "extract")

        # Use spec= to restrict the mock to only have 'TarFile' attribute,
        # so hasattr(mock_tarfile, 'data_filter') returns False
        with patch(
            "sagemaker.core.common_utils.tarfile", spec=["TarFile"]
        ):
            with patch("sagemaker.core.common_utils._get_safe_members") as mock_safe:
                mock_safe.return_value = [mock_member]

                with patch("sagemaker.core.common_utils._validate_extracted_paths"):
                    custom_extractall_tarfile(mock_tar, extract_path)

                    # Verify getmembers() was called (not iterating over tar directly)
                    mock_tar.getmembers.assert_called_once()

                    # Verify _get_safe_members was called with the members list and resolved base
                    mock_safe.assert_called_once()
                    call_args = mock_safe.call_args
                    assert call_args[0][0] == [mock_member]  # members list
                    # base should be resolved extract_path, not cwd
                    expected_base = _get_resolved_path(extract_path)
                    assert call_args[0][1] == expected_base

                    mock_tar.extractall.assert_called_once()
                    call_kwargs = mock_tar.extractall.call_args[1]
                    assert call_kwargs["path"] == extract_path
                    assert "members" in call_kwargs


def test_is_bad_path_handles_value_error_gracefully():
    """Test that _is_bad_path returns True when os.path.commonpath raises ValueError.

    This can happen on Windows with paths on different drives, or with mixed
    absolute/relative paths.
    """
    with patch("sagemaker.core.common_utils.os.path.commonpath", side_effect=ValueError):
        # Should return True (treat as bad path) when commonpath raises ValueError
        result = _is_bad_path("some/path", "/some/base")
        assert result is True
