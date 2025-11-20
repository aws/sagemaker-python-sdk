"""Unit tests for uploader.py to increase coverage."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


class TestGetDirSize(unittest.TestCase):
    """Test _get_dir_size function."""

    def test_get_dir_size_with_files(self):
        """Test _get_dir_size calculates directory size."""
        from sagemaker.serve.utils.uploader import _get_dir_size
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = os.path.join(tmpdir, "file1.txt")
            file2 = os.path.join(tmpdir, "file2.txt")
            with open(file1, "w") as f:
                f.write("a" * 100)
            with open(file2, "w") as f:
                f.write("b" * 200)
            
            size = _get_dir_size(tmpdir)
            
            self.assertEqual(size, 300)

    def test_get_dir_size_with_subdirs(self):
        """Test _get_dir_size with subdirectories."""
        from sagemaker.serve.utils.uploader import _get_dir_size
        
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            
            file1 = os.path.join(tmpdir, "file1.txt")
            file2 = os.path.join(subdir, "file2.txt")
            with open(file1, "w") as f:
                f.write("a" * 100)
            with open(file2, "w") as f:
                f.write("b" * 200)
            
            size = _get_dir_size(tmpdir)
            
            self.assertEqual(size, 300)


class TestUploaderObserve(unittest.TestCase):
    """Test Uploader observe method."""

    def test_observe_updates_progress(self):
        """Test observe updates progress bar."""
        from sagemaker.serve.utils.uploader import Uploader
        
        uploader = Uploader()
        uploader.total_left = 1000
        uploader.pbar = Mock()
        
        uploader.observe(100)
        
        self.assertEqual(uploader.total_left, 900)
        uploader.pbar.update.assert_called_once_with(100)


class TestUploaderUpload(unittest.TestCase):
    """Test Uploader upload method."""

    @patch('sagemaker.serve.utils.uploader.tqdm.tqdm')
    @patch('sagemaker.serve.utils.uploader.boto3.session.Session')
    @patch('sagemaker.serve.utils.uploader.create_tar_file')
    @patch('sagemaker.serve.utils.uploader.tempfile.mkdtemp')
    @patch('os.listdir')
    @patch('os.remove')
    def test_upload_creates_tar_and_uploads(self, mock_remove, mock_listdir, mock_mkdtemp, 
                                             mock_create_tar, mock_boto_session, mock_tqdm):
        """Test upload creates tar and uploads to S3."""
        from sagemaker.serve.utils.uploader import Uploader
        
        mock_listdir.return_value = ["file1.txt", "file2.txt"]
        mock_mkdtemp.return_value = "/tmp/test"
        mock_create_tar.return_value = "/tmp/test/model.tar.gz"
        
        mock_s3_client = Mock()
        mock_session = Mock()
        mock_session.client.return_value = mock_s3_client
        mock_boto_session.return_value = mock_session
        
        mock_pbar = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        mock_credentials = Mock()
        mock_credentials.access_key = "access"
        mock_credentials.secret_key = "secret"
        mock_credentials.token = "token"
        
        uploader = Uploader()
        uploader.upload("/model/dir", 1000, mock_credentials, "us-west-2", "bucket", "key")
        
        mock_s3_client.upload_file.assert_called_once()
        mock_remove.assert_called_once()


class TestUploaderUploadUncompressed(unittest.TestCase):
    """Test Uploader upload_uncompressed method."""

    @patch('sagemaker.serve.utils.uploader.tqdm.tqdm')
    @patch('sagemaker.serve.utils.uploader.S3Uploader')
    def test_upload_uncompressed(self, mock_s3_uploader, mock_tqdm):
        """Test upload_uncompressed uploads to S3."""
        from sagemaker.serve.utils.uploader import Uploader
        
        mock_pbar = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        mock_session = Mock()
        
        uploader = Uploader()
        uploader.upload_uncompressed("/model/dir", mock_session, "bucket", "prefix", 1000)
        
        mock_s3_uploader.upload.assert_called_once()


class TestUploadFunction(unittest.TestCase):
    """Test upload wrapper function."""

    @patch('sagemaker.serve.utils.uploader.Uploader')
    @patch('sagemaker.serve.utils.uploader._get_dir_size')
    def test_upload_function(self, mock_get_size, mock_uploader_class):
        """Test upload function."""
        from sagemaker.serve.utils.uploader import upload
        
        mock_get_size.return_value = 1000
        mock_uploader = Mock()
        mock_uploader_class.return_value = mock_uploader
        
        mock_session = Mock()
        mock_session.boto_session.get_credentials.return_value = Mock()
        mock_session.boto_session.region_name = "us-west-2"
        
        result = upload(mock_session, "/model/dir", "bucket", "prefix")
        
        mock_uploader.upload.assert_called_once()
        self.assertIn("s3://", result)


class TestUploadUncompressedFunction(unittest.TestCase):
    """Test upload_uncompressed wrapper function."""

    @patch('sagemaker.serve.utils.uploader.Uploader')
    @patch('sagemaker.serve.utils.uploader._get_dir_size')
    def test_upload_uncompressed_function(self, mock_get_size, mock_uploader_class):
        """Test upload_uncompressed function."""
        from sagemaker.serve.utils.uploader import upload_uncompressed
        
        mock_get_size.return_value = 1000
        mock_uploader = Mock()
        mock_uploader_class.return_value = mock_uploader
        
        mock_session = Mock()
        
        result = upload_uncompressed(mock_session, "/model/dir", "bucket", "prefix")
        
        mock_uploader.upload_uncompressed.assert_called_once()
        self.assertIn("s3://", result)


if __name__ == "__main__":
    unittest.main()
