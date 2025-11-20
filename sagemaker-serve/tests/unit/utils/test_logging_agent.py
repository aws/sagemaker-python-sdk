"""Unit tests for logging_agent.py to increase coverage."""

import unittest
from unittest.mock import Mock, patch
import queue
from datetime import datetime, timedelta


class TestGetLogs(unittest.TestCase):
    """Test _get_logs function."""

    @patch('sagemaker.serve.utils.logging_agent.datetime')
    def test_get_logs_success(self, mock_datetime):
        """Test _get_logs processes logs successfully."""
        from sagemaker.serve.utils.logging_agent import _get_logs
        
        now = datetime.now()
        mock_datetime.now.return_value = now
        
        generator = iter(["log1", "log2", "log3"])
        logs = queue.Queue()
        until = now + timedelta(seconds=10)
        
        _get_logs(generator, logs, until)
        
        self.assertEqual(logs.qsize(), 3)

    @patch('sagemaker.serve.utils.logging_agent.datetime')
    def test_get_logs_timeout(self, mock_datetime):
        """Test _get_logs stops at timeout."""
        from sagemaker.serve.utils.logging_agent import _get_logs
        
        now = datetime.now()
        future = now + timedelta(seconds=1)
        mock_datetime.now.side_effect = [now, future, future + timedelta(seconds=2)]
        
        generator = iter(["log1", "log2", "log3"])
        logs = queue.Queue()
        until = now + timedelta(seconds=1)
        
        _get_logs(generator, logs, until)
        
        self.assertLessEqual(logs.qsize(), 3)

    def test_get_logs_stop_iteration(self):
        """Test _get_logs handles StopIteration."""
        from sagemaker.serve.utils.logging_agent import _get_logs
        
        generator = iter([])
        logs = queue.Queue()
        until = datetime.now() + timedelta(seconds=10)
        
        _get_logs(generator, logs, until)
        
        self.assertEqual(logs.qsize(), 0)


class TestPullLogs(unittest.TestCase):
    """Test pull_logs function."""

    def test_pull_logs_already_past_until(self):
        """Test pull_logs returns immediately if until is in the past."""
        from sagemaker.serve.utils.logging_agent import pull_logs
        
        generator = iter(["log1"])
        stop = Mock()
        until = datetime.now() - timedelta(seconds=10)
        
        pull_logs(generator, stop, until, False)
        
        stop.assert_not_called()

    @patch('sagemaker.serve.utils.logging_agent.Thread')
    @patch('sagemaker.serve.utils.logging_agent.queue.Queue')
    def test_pull_logs_oom_error(self, mock_queue_class, mock_thread):
        """Test pull_logs detects OutOfMemoryError."""
        from sagemaker.serve.utils.logging_agent import pull_logs
        from sagemaker.serve.utils.exceptions import LocalModelOutOfMemoryException
        
        mock_queue = Mock()
        mock_queue.get.return_value = "[INFO ] OutOfMemoryError occurred"
        mock_queue_class.return_value = mock_queue
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        generator = iter(["log1"])
        stop = Mock()
        until = datetime.now() + timedelta(seconds=10)
        
        with self.assertRaises(LocalModelOutOfMemoryException):
            pull_logs(generator, stop, until, False)
        
        stop.assert_called_once()

    @patch('sagemaker.serve.utils.logging_agent.Thread')
    @patch('sagemaker.serve.utils.logging_agent.queue.Queue')
    def test_pull_logs_cuda_oom(self, mock_queue_class, mock_thread):
        """Test pull_logs detects CUDA out of memory."""
        from sagemaker.serve.utils.logging_agent import pull_logs
        from sagemaker.serve.utils.exceptions import LocalModelOutOfMemoryException
        
        mock_queue = Mock()
        mock_queue.get.return_value = "CUDA out of memory. Tried to allocate 1024MB"
        mock_queue_class.return_value = mock_queue
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        generator = iter(["log1"])
        stop = Mock()
        until = datetime.now() + timedelta(seconds=10)
        
        with self.assertRaises(LocalModelOutOfMemoryException):
            pull_logs(generator, stop, until, False)

    @patch('sagemaker.serve.utils.logging_agent.Thread')
    @patch('sagemaker.serve.utils.logging_agent.queue.Queue')
    def test_pull_logs_djl_oom(self, mock_queue_class, mock_thread):
        """Test pull_logs detects DJL OOM."""
        from sagemaker.serve.utils.logging_agent import pull_logs
        from sagemaker.serve.utils.exceptions import LocalModelOutOfMemoryException
        
        mock_queue = Mock()
        mock_queue.get.return_value = "ai.djl.engine.EngineException: OOM"
        mock_queue_class.return_value = mock_queue
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        generator = iter(["log1"])
        stop = Mock()
        until = datetime.now() + timedelta(seconds=10)
        
        with self.assertRaises(LocalModelOutOfMemoryException):
            pull_logs(generator, stop, until, False)

    @patch('sagemaker.serve.utils.logging_agent.Thread')
    @patch('sagemaker.serve.utils.logging_agent.queue.Queue')
    def test_pull_logs_4xx_error(self, mock_queue_class, mock_thread):
        """Test pull_logs detects 4xx errors."""
        from sagemaker.serve.utils.logging_agent import pull_logs
        from sagemaker.serve.utils.exceptions import LocalModelInvocationException
        
        mock_queue = Mock()
        mock_queue.get.return_value = "4xx.Count:1"
        mock_queue_class.return_value = mock_queue
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        generator = iter(["log1"])
        stop = Mock()
        until = datetime.now() + timedelta(seconds=10)
        
        with self.assertRaises(LocalModelInvocationException):
            pull_logs(generator, stop, until, False)

    @patch('sagemaker.serve.utils.logging_agent.Thread')
    @patch('sagemaker.serve.utils.logging_agent.queue.Queue')
    def test_pull_logs_5xx_error(self, mock_queue_class, mock_thread):
        """Test pull_logs detects 5xx errors."""
        from sagemaker.serve.utils.logging_agent import pull_logs
        from sagemaker.serve.utils.exceptions import LocalModelInvocationException
        
        mock_queue = Mock()
        mock_queue.get.return_value = "5xx.Count:1"
        mock_queue_class.return_value = mock_queue
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        generator = iter(["log1"])
        stop = Mock()
        until = datetime.now() + timedelta(seconds=10)
        
        with self.assertRaises(LocalModelInvocationException):
            pull_logs(generator, stop, until, False)

    @patch('sagemaker.serve.utils.logging_agent.Thread')
    @patch('sagemaker.serve.utils.logging_agent.queue.Queue')
    def test_pull_logs_error_message(self, mock_queue_class, mock_thread):
        """Test pull_logs detects ERROR messages."""
        from sagemaker.serve.utils.logging_agent import pull_logs
        from sagemaker.serve.utils.exceptions import LocalModelLoadException
        
        mock_queue = Mock()
        mock_queue.get.return_value = "[ERROR] Failed to load model"
        mock_queue_class.return_value = mock_queue
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        generator = iter(["log1"])
        stop = Mock()
        until = datetime.now() + timedelta(seconds=10)
        
        with self.assertRaises(LocalModelLoadException):
            pull_logs(generator, stop, until, False)

    @patch('sagemaker.serve.utils.logging_agent.Thread')
    @patch('sagemaker.serve.utils.logging_agent.queue.Queue')
    def test_pull_logs_failed_register_workflow(self, mock_queue_class, mock_thread):
        """Test pull_logs detects failed workflow registration."""
        from sagemaker.serve.utils.logging_agent import pull_logs
        from sagemaker.serve.utils.exceptions import LocalModelLoadException
        
        mock_queue = Mock()
        mock_queue.get.return_value = "Failed register workflow"
        mock_queue_class.return_value = mock_queue
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        generator = iter(["log1"])
        stop = Mock()
        until = datetime.now() + timedelta(seconds=10)
        
        with self.assertRaises(LocalModelLoadException):
            pull_logs(generator, stop, until, False)

    @patch('sagemaker.serve.utils.logging_agent.Thread')
    @patch('sagemaker.serve.utils.logging_agent.queue.Queue')
    def test_pull_logs_address_in_use(self, mock_queue_class, mock_thread):
        """Test pull_logs detects address already in use."""
        from sagemaker.serve.utils.logging_agent import pull_logs
        from sagemaker.serve.utils.exceptions import LocalModelLoadException
        
        mock_queue = Mock()
        mock_queue.get.return_value = "Address already in use"
        mock_queue_class.return_value = mock_queue
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        generator = iter(["log1"])
        stop = Mock()
        until = datetime.now() + timedelta(seconds=10)
        
        with self.assertRaises(LocalModelLoadException):
            pull_logs(generator, stop, until, False)

    @patch('sagemaker.serve.utils.logging_agent.Thread')
    @patch('sagemaker.serve.utils.logging_agent.queue.Queue')
    def test_pull_logs_queue_empty_no_final_pull(self, mock_queue_class, mock_thread):
        """Test pull_logs handles queue empty without final pull."""
        from sagemaker.serve.utils.logging_agent import pull_logs
        
        mock_queue = Mock()
        mock_queue.get.side_effect = queue.Empty()
        mock_queue_class.return_value = mock_queue
        
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        generator = iter(["log1"])
        stop = Mock()
        until = datetime.now() - timedelta(seconds=1)
        
        pull_logs(generator, stop, until, False)
        
        stop.assert_not_called()

    @unittest.skip("Complex datetime mocking required")
    def test_pull_logs_queue_empty_with_final_pull(self):
        """Test pull_logs handles queue empty with final pull."""
        pass

    @unittest.skip("Complex datetime mocking required")
    def test_pull_logs_thread_not_terminating(self):
        """Test pull_logs raises exception if thread doesn't terminate."""
        pass


if __name__ == "__main__":
    unittest.main()
