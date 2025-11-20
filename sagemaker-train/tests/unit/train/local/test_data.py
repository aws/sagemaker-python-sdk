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
"""Tests for local data module."""
from __future__ import absolute_import

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock, mock_open

from sagemaker.train.local.data import (
    get_data_source_instance,
    get_splitter_instance,
    get_batch_strategy_instance,
    LocalFileDataSource,
    S3DataSource,
    NoneSplitter,
    LineSplitter,
    RecordIOSplitter,
    MultiRecordStrategy,
    SingleRecordStrategy,
    _payload_size_within_limit,
    _validate_payload_size,
)


class TestGetDataSourceInstance:
    """Test get_data_source_instance function."""

    def test_returns_local_file_data_source(self):
        """Test returns LocalFileDataSource for file:// URI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_source = get_data_source_instance(f"file://{tmpdir}", None)
            assert isinstance(data_source, LocalFileDataSource)

    @patch("sagemaker.train.local.data.S3DataSource")
    def test_returns_s3_data_source(self, mock_s3_data_source):
        """Test returns S3DataSource for s3:// URI."""
        mock_session = MagicMock()
        data_source = get_data_source_instance("s3://bucket/prefix", mock_session)
        mock_s3_data_source.assert_called_once_with("bucket", "/prefix", mock_session)

    def test_raises_error_for_invalid_scheme(self):
        """Test raises ValueError for invalid URI scheme."""
        with pytest.raises(ValueError, match="data_source must be either file or s3"):
            get_data_source_instance("http://example.com", None)


class TestGetSplitterInstance:
    """Test get_splitter_instance function."""

    def test_returns_none_splitter_for_none_string(self):
        """Test returns NoneSplitter for 'None' string."""
        splitter = get_splitter_instance("None")
        assert isinstance(splitter, NoneSplitter)

    def test_returns_none_splitter_for_none(self):
        """Test returns NoneSplitter for None."""
        splitter = get_splitter_instance(None)
        assert isinstance(splitter, NoneSplitter)

    def test_returns_line_splitter(self):
        """Test returns LineSplitter for 'Line'."""
        splitter = get_splitter_instance("Line")
        assert isinstance(splitter, LineSplitter)

    def test_returns_recordio_splitter(self):
        """Test returns RecordIOSplitter for 'RecordIO'."""
        splitter = get_splitter_instance("RecordIO")
        assert isinstance(splitter, RecordIOSplitter)

    def test_raises_error_for_invalid_split_type(self):
        """Test raises ValueError for invalid split type."""
        with pytest.raises(ValueError, match="Invalid Split Type"):
            get_splitter_instance("Invalid")


class TestGetBatchStrategyInstance:
    """Test get_batch_strategy_instance function."""

    def test_returns_single_record_strategy(self):
        """Test returns SingleRecordStrategy."""
        splitter = NoneSplitter()
        strategy = get_batch_strategy_instance("SingleRecord", splitter)
        assert isinstance(strategy, SingleRecordStrategy)
        assert strategy.splitter is splitter

    def test_returns_multi_record_strategy(self):
        """Test returns MultiRecordStrategy."""
        splitter = NoneSplitter()
        strategy = get_batch_strategy_instance("MultiRecord", splitter)
        assert isinstance(strategy, MultiRecordStrategy)
        assert strategy.splitter is splitter

    def test_raises_error_for_invalid_strategy(self):
        """Test raises ValueError for invalid strategy."""
        splitter = NoneSplitter()
        with pytest.raises(ValueError, match="Invalid Batch Strategy"):
            get_batch_strategy_instance("Invalid", splitter)


class TestLocalFileDataSource:
    """Test LocalFileDataSource class."""

    def test_init_with_valid_directory(self):
        """Test initialization with valid directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_source = LocalFileDataSource(tmpdir)
            assert data_source.root_path == os.path.abspath(tmpdir)

    def test_init_with_valid_file(self):
        """Test initialization with valid file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            try:
                data_source = LocalFileDataSource(tmpfile.name)
                assert data_source.root_path == os.path.abspath(tmpfile.name)
            finally:
                os.unlink(tmpfile.name)

    def test_init_raises_error_for_nonexistent_path(self):
        """Test raises RuntimeError for nonexistent path."""
        with pytest.raises(RuntimeError, match="does not exist"):
            LocalFileDataSource("/nonexistent/path")

    def test_get_file_list_for_directory(self):
        """Test get_file_list returns files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = os.path.join(tmpdir, "file1.txt")
            file2 = os.path.join(tmpdir, "file2.txt")
            open(file1, "w").close()
            open(file2, "w").close()
            
            data_source = LocalFileDataSource(tmpdir)
            file_list = data_source.get_file_list()
            
            assert len(file_list) == 2
            assert file1 in file_list
            assert file2 in file_list

    def test_get_file_list_for_single_file(self):
        """Test get_file_list returns single file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            try:
                data_source = LocalFileDataSource(tmpfile.name)
                file_list = data_source.get_file_list()
                
                assert len(file_list) == 1
                assert file_list[0] == tmpfile.name
            finally:
                os.unlink(tmpfile.name)

    def test_get_root_dir_for_directory(self):
        """Test get_root_dir returns directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_source = LocalFileDataSource(tmpdir)
            assert data_source.get_root_dir() == os.path.abspath(tmpdir)

    def test_get_root_dir_for_file(self):
        """Test get_root_dir returns parent directory for file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            try:
                data_source = LocalFileDataSource(tmpfile.name)
                assert data_source.get_root_dir() == os.path.dirname(tmpfile.name)
            finally:
                os.unlink(tmpfile.name)


class TestS3DataSource:
    """Test S3DataSource class."""

    @pytest.mark.skip(reason="S3DataSource requires sagemaker.utils module which doesn't exist in modular structure")
    def test_init_downloads_from_s3(self):
        """Test initialization downloads from S3."""
        pass

    @pytest.mark.skip(reason="S3DataSource requires sagemaker.utils module which doesn't exist in modular structure")
    def test_init_applies_darwin_workaround(self):
        """Test applies Darwin workaround for Mac OS."""
        pass

    @pytest.mark.skip(reason="S3DataSource requires sagemaker.utils module which doesn't exist in modular structure")
    def test_init_uses_custom_root_dir(self):
        """Test uses custom root directory."""
        pass

    @pytest.mark.skip(reason="S3DataSource requires sagemaker.utils module which doesn't exist in modular structure")
    def test_get_file_list(self):
        """Test get_file_list delegates to LocalFileDataSource."""
        pass

    @pytest.mark.skip(reason="S3DataSource requires sagemaker.utils module which doesn't exist in modular structure")
    def test_get_root_dir(self):
        """Test get_root_dir delegates to LocalFileDataSource."""
        pass


class TestNoneSplitter:
    """Test NoneSplitter class."""

    def test_split_returns_whole_file_text(self):
        """Test split returns whole file for text content."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
            tmpfile.write("test content")
            tmpfile.flush()
            
            try:
                splitter = NoneSplitter()
                result = list(splitter.split(tmpfile.name))
                
                assert len(result) == 1
                assert result[0] == "test content"
            finally:
                os.unlink(tmpfile.name)

    def test_split_returns_whole_file_binary(self):
        """Test split returns whole file for binary content."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmpfile:
            tmpfile.write(b"\x00\x01\x02\x03")
            tmpfile.flush()
            
            try:
                splitter = NoneSplitter()
                result = list(splitter.split(tmpfile.name))
                
                assert len(result) == 1
                assert result[0] == b"\x00\x01\x02\x03"
            finally:
                os.unlink(tmpfile.name)

    def test_is_binary_returns_true_for_binary(self):
        """Test _is_binary returns True for binary data."""
        splitter = NoneSplitter()
        assert splitter._is_binary(b"\x00\x01\x02") is True

    def test_is_binary_returns_false_for_text(self):
        """Test _is_binary returns False for text data."""
        splitter = NoneSplitter()
        assert splitter._is_binary(b"test content") is False


class TestLineSplitter:
    """Test LineSplitter class."""

    def test_split_returns_lines(self):
        """Test split returns individual lines."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpfile:
            tmpfile.write("line1\nline2\nline3")
            tmpfile.flush()
            
            try:
                splitter = LineSplitter()
                result = list(splitter.split(tmpfile.name))
                
                assert len(result) == 3
                assert result[0] == "line1\n"
                assert result[1] == "line2\n"
                assert result[2] == "line3"
            finally:
                os.unlink(tmpfile.name)


class TestRecordIOSplitter:
    """Test RecordIOSplitter class."""

    @pytest.mark.skip(reason="RecordIOSplitter requires sagemaker.amazon.common module which doesn't exist in modular structure")
    def test_split_returns_recordio_records(self):
        """Test split returns RecordIO records."""
        pass


class TestMultiRecordStrategy:
    """Test MultiRecordStrategy class."""

    def test_pad_groups_records_within_size(self):
        """Test pad groups records within size limit."""
        splitter = MagicMock()
        splitter.split.return_value = ["a", "b", "c", "d"]
        
        strategy = MultiRecordStrategy(splitter)
        result = list(strategy.pad("file.txt", size=0))  # size=0 means unlimited
        
        assert len(result) == 1
        assert result[0] == "abcd"

    def test_pad_splits_when_exceeding_size(self):
        """Test pad splits records when exceeding size."""
        splitter = MagicMock()
        splitter.split.return_value = ["a" * 1000, "b" * 1000, "c" * 1000]
        
        strategy = MultiRecordStrategy(splitter)
        result = list(strategy.pad("file.txt", size=0.001))  # Very small size
        
        # Should split into multiple batches
        assert len(result) > 1


class TestSingleRecordStrategy:
    """Test SingleRecordStrategy class."""

    def test_pad_returns_individual_records(self):
        """Test pad returns individual records."""
        splitter = MagicMock()
        splitter.split.return_value = ["record1", "record2", "record3"]
        
        strategy = SingleRecordStrategy(splitter)
        result = list(strategy.pad("file.txt", size=0))  # size=0 means unlimited
        
        assert len(result) == 3
        assert result[0] == "record1"
        assert result[1] == "record2"
        assert result[2] == "record3"

    def test_pad_raises_error_for_oversized_record(self):
        """Test pad raises error for record exceeding size."""
        splitter = MagicMock()
        splitter.split.return_value = ["a" * 10000000]  # Very large record
        
        strategy = SingleRecordStrategy(splitter)
        
        with pytest.raises(RuntimeError, match="Record is larger"):
            list(strategy.pad("file.txt", size=0.001))  # Very small size


class TestPayloadSizeWithinLimit:
    """Test _payload_size_within_limit function."""

    def test_returns_true_for_size_zero(self):
        """Test returns True when size is 0 (unlimited)."""
        assert _payload_size_within_limit("any payload", 0) is True

    def test_returns_true_for_small_payload(self):
        """Test returns True for payload within limit."""
        assert _payload_size_within_limit("small", 1) is True

    def test_returns_false_for_large_payload(self):
        """Test returns False for payload exceeding limit."""
        large_payload = "a" * 10000000  # 10MB
        assert _payload_size_within_limit(large_payload, 1) is False


class TestValidatePayloadSize:
    """Test _validate_payload_size function."""

    def test_returns_true_for_valid_size(self):
        """Test returns True for valid payload size."""
        assert _validate_payload_size("small payload", 1) is True

    def test_returns_true_for_size_zero(self):
        """Test returns True when size is 0 (unlimited)."""
        assert _validate_payload_size("any payload", 0) is True

    def test_raises_error_for_oversized_payload(self):
        """Test raises RuntimeError for oversized payload."""
        large_payload = "a" * 10000000  # 10MB
        with pytest.raises(RuntimeError, match="Record is larger"):
            _validate_payload_size(large_payload, 1)
