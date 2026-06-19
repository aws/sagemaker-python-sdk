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
"""Tests for the Windows PermissionError fix in FrameworkProcessor._package_code.

Issue #5873: On Windows, FrameworkProcessor.run() with a local source_dir fails
during code packaging with PermissionError: [WinError 32]. The failure occurs in
_package_code() at os.unlink(tmp.name) inside the `with` block while the
NamedTemporaryFile handle is still open.

Fix: Move os.unlink() after the `with` block closes the handle.
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


class TestFrameworkProcessorPackageCodeWindowsFix:
    """Tests verifying the fix for Windows PermissionError in _package_code.

    Issue #5873: os.unlink() was called inside the `with` block while the
    NamedTemporaryFile handle was still open, causing PermissionError on Windows.
    The fix moves os.unlink() after the `with` block exits.
    """

    def test_package_code_with_source_dir_does_not_raise_permission_error(
        self, mock_session
    ):
        """Verify _package_code completes without PermissionError on any platform.

        This test creates a real temp directory with an entry point file and calls
        _package_code with a mocked S3Uploader to ensure no PermissionError is
        raised during temp file cleanup.
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

            with patch(
                "sagemaker.processing.S3Uploader.upload",
                return_value="s3://test-bucket/sagemaker/test-job/input/code/sourcedir.tar.gz",
            ):
                # This should not raise PermissionError on any platform
                result = processor._package_code(
                    code=entry_point,
                    source_dir=tmpdir,
                    dependencies=None,
                    git_config=None,
                    job_name="test-job",
                )
                assert "s3://" in result

    def test_package_code_temp_file_deleted_after_handle_closed(
        self, mock_session
    ):
        """Verify os.unlink is called only after the temp file handle is closed.

        This test patches os.unlink to verify that the temp file can be opened
        (i.e., no other handle holds it) when unlink is called, confirming the
        fix for the Windows PermissionError.
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
                """Simulate Windows behavior: fail if file handle is open."""
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

            # Verify that when unlink was called, the file handle was closed
            assert len(file_was_closed_when_unlinked) >= 1
            assert all(file_was_closed_when_unlinked), (
                "os.unlink was called while the file handle was still open. "
                "This would cause PermissionError on Windows."
            )

    def test_package_code_temp_file_cleaned_up_on_upload_exception(
        self, mock_session
    ):
        """Verify temp file is cleaned up even when S3 upload raises an exception.

        The fix ensures os.unlink is called in a finally block after the
        with block exits, even when an exception occurs during S3 upload.
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

            unlinked_files = []
            original_unlink = os.unlink

            def tracking_unlink(path):
                """Track unlink calls for tar.gz files."""
                if ".tar.gz" in str(path):
                    unlinked_files.append(str(path))
                original_unlink(path)

            with patch(
                "sagemaker.processing.S3Uploader.upload",
                side_effect=RuntimeError("Upload failed"),
            ):
                with patch("sagemaker.processing.os.unlink", side_effect=tracking_unlink):
                    with pytest.raises(RuntimeError, match="Upload failed"):
                        processor._package_code(
                            code=entry_point,
                            source_dir=tmpdir,
                            dependencies=None,
                            git_config=None,
                            job_name="test-job",
                        )

            # Verify the temp file was still cleaned up despite the exception
            assert len(unlinked_files) >= 1, (
                "Temp tar.gz file was not cleaned up after upload exception"
            )

    def test_package_code_without_source_dir_does_not_raise_permission_error(
        self, mock_session
    ):
        """Verify _package_code without source_dir completes without PermissionError.

        When source_dir is None, _package_code should still handle temp files
        correctly without raising PermissionError on Windows.
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

            with patch(
                "sagemaker.processing.S3Uploader.upload",
                return_value="s3://test-bucket/sagemaker/test-job/input/code/sourcedir.tar.gz",
            ):
                # This should not raise PermissionError on any platform
                result = processor._package_code(
                    code=entry_point,
                    source_dir=None,
                    dependencies=None,
                    git_config=None,
                    job_name="test-job",
                )
                assert "s3://" in result
