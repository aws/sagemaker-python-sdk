import unittest
from sagemaker.serve.utils.exceptions import (
    ModelBuilderException,
    LocalDeepPingException,
    InProcessDeepPingException,
    LocalModelOutOfMemoryException,
    LocalModelLoadException,
    LocalModelInvocationException,
    SkipTuningComboException,
    TaskNotFoundException
)


class TestModelBuilderException(unittest.TestCase):
    def test_base_exception(self):
        exc = ModelBuilderException()
        self.assertIsInstance(exc, Exception)
        self.assertIn("unspecified error", str(exc))

    def test_exception_with_kwargs(self):
        exc = ModelBuilderException(test_param="value")
        self.assertEqual(exc.kwargs, {"test_param": "value"})


class TestLocalDeepPingException(unittest.TestCase):
    def test_exception_message(self):
        exc = LocalDeepPingException(message="Deep ping failed")
        self.assertIn("Deep ping failed", str(exc))
        self.assertEqual(exc.model_builder_error_code, 1)


class TestInProcessDeepPingException(unittest.TestCase):
    def test_exception_message(self):
        exc = InProcessDeepPingException(message="In-process ping failed")
        self.assertIn("In-process ping failed", str(exc))
        self.assertEqual(exc.model_builder_error_code, 1)


class TestLocalModelOutOfMemoryException(unittest.TestCase):
    def test_exception_message(self):
        exc = LocalModelOutOfMemoryException(message="Out of memory")
        self.assertIn("Out of memory", str(exc))
        self.assertEqual(exc.model_builder_error_code, 2)


class TestLocalModelLoadException(unittest.TestCase):
    def test_exception_message(self):
        exc = LocalModelLoadException(message="Failed to load model")
        self.assertIn("Failed to load model", str(exc))
        self.assertEqual(exc.model_builder_error_code, 3)


class TestLocalModelInvocationException(unittest.TestCase):
    def test_exception_message(self):
        exc = LocalModelInvocationException(message="Invocation failed")
        self.assertIn("Invocation failed", str(exc))
        self.assertEqual(exc.model_builder_error_code, 4)


class TestSkipTuningComboException(unittest.TestCase):
    def test_exception_message(self):
        exc = SkipTuningComboException(message="Skip tuning combo")
        self.assertIn("Skip tuning combo", str(exc))


class TestTaskNotFoundException(unittest.TestCase):
    def test_exception_message(self):
        exc = TaskNotFoundException(message="Task not found")
        self.assertIn("Task not found", str(exc))


if __name__ == "__main__":
    unittest.main()
