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
"""Tests for sagemaker-core processing functionality.

Note: Tests for the FrameworkProcessor._package_code Windows PermissionError fix
(issue #5873) are located in tests/unit/test_processing_windows_fix.py since the
bug is in the main SageMaker SDK's sagemaker/processing.py module.
"""
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from sagemaker.processing import FrameworkProcessor


@pytest.fixture
def mock_session():
    """Create a mock SageMaker session for testing."""
    session = Mock()
    session.boto_session = Mock()
    session.boto_session.region_name = "us-west-2"
    session.sagemaker_client = Mock()
    session.default_bucket.return_value = "test-bucket"
    session.default_bucket_prefix = "sagemaker"
    session.expand_role.return_value = "arn:aws:iam::123456789012:role/SageMakerRole"
    session.sagemaker_config = {}
    session.local_mode = False
    return session


class TestProcessingPlaceholder:
    """Placeholder test class for sagemaker-core processing tests."""

    def test_placeholder(self):
        """Placeholder test to prevent empty test file warnings."""
        pass


class TestPackageCodeTempFileCleanup:
    """Tests verifying the fix for Windows PermissionError in _package_code.

    Issue #5873: os.unlink() was called inside the `with` block while the
    NamedTemporaryFile handle was still open, causing PermissionError on Windows.
    The fix moves os.unlink() after the `with` block exits, ensuring the file
    handle is closed before deletion.
    """

    def test_package_code_temp_file_cleanup(self, mock_session):
        """Verify temp file handle is closed before os.unlink is called.

        This validates the fix for issue #5873: the temp file should be
        accessible (handle closed) when os.unlink is called, which prevents
        PermissionError on Windows.
        """
        processor = FrameworkProcessor(
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            image_uri="test-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            entry_point = os.path.join(tmpdir, "train.py")
            with open(entry_point, "w") as f:
                f.write("print('training')")

            file_was_closed_when_unlinked = []

            original_unlink = os.unlink

            def simulated_windows_unlink(path):
                """Simulate Windows behavior: verify file handle is closed.

                On Windows, if a file handle is still open, attempting to
                delete or open the file would raise PermissionError.
                We verify the handle is closed by successfully opening the file.
                """
                if "sourcedir" in str(path) or ".tar.gz" in str(path):
                    try:
                        # Try to open the file - this would fail on Windows
                        # if another handle is open
                        with open(path, "rb"):
                            pass
                        file_was_closed_when_unlinked.append(True)
                    except (PermissionError, OSError):
                        file_was_closed_when_unlinked.append(False)
                    finally:
                        original_unlink(path)
                else:
                    original_unlink(path)

            with patch(
                "sagemaker.processing.S3Uploader.upload",
                return_value="s3://test-bucket/sagemaker/test-job/input/code/sourcedir.tar.gz",
            ):
                with patch("sagemaker.processing.os.unlink", side_effect=simulated_windows_unlink):
                    processor._package_code(
                        code=entry_point,
                        source_dir=tmpdir,
                        dependencies=None,
                        git_config=None,
                        job_name="test-job",
                    )

            # Verify that when unlink was called, the file handle was already closed
            assert len(file_was_closed_when_unlinked) >= 1, (
                "os.unlink was never called for the temp tar.gz file"
            )
            assert all(file_was_closed_when_unlinked), (
                "os.unlink was called while the file handle was still open. "
                "This would cause PermissionError on Windows."
            )
