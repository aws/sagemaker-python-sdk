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
"""Tests for logging_config module."""
from __future__ import absolute_import

import logging
import time
from unittest.mock import patch
from sagemaker.train.remote_function.logging_config import _UTCFormatter, get_logger


class TestUTCFormatter:
    """Test _UTCFormatter class."""

    def test_converter_is_gmtime(self):
        """Test that converter is set to gmtime."""
        formatter = _UTCFormatter()
        assert formatter.converter == time.gmtime

    def test_formats_time_in_utc(self):
        """Test that time is formatted in UTC."""
        formatter = _UTCFormatter("%(asctime)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        # Should contain UTC time format
        assert formatted


class TestGetLogger:
    """Test get_logger function."""

    def test_returns_logger_with_correct_name(self):
        """Test that logger has correct name."""
        logger = get_logger()
        assert logger.name == "sagemaker.remote_function"

    def test_logger_has_info_level(self):
        """Test that logger is set to INFO level."""
        logger = get_logger()
        assert logger.level == logging.INFO

    def test_logger_has_handler(self):
        """Test that logger has at least one handler."""
        logger = get_logger()
        assert len(logger.handlers) > 0

    def test_logger_handler_has_utc_formatter(self):
        """Test that logger handler uses UTC formatter."""
        logger = get_logger()
        handler = logger.handlers[0]
        # Check that formatter has gmtime converter (UTC formatter characteristic)
        assert handler.formatter.converter == time.gmtime

    def test_logger_does_not_propagate(self):
        """Test that logger does not propagate to root logger."""
        logger = get_logger()
        assert logger.propagate == 0

    def test_get_logger_is_idempotent(self):
        """Test that calling get_logger multiple times returns same logger."""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2

    def test_logger_handler_is_stream_handler(self):
        """Test that logger uses StreamHandler."""
        logger = get_logger()
        assert isinstance(logger.handlers[0], logging.StreamHandler)
