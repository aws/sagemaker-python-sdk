import unittest
from unittest.mock import Mock, patch
from sagemaker.serve.spec.inference_base import CustomOrchestrator, AsyncCustomOrchestrator


class ConcreteOrchestrator(CustomOrchestrator):
    def handle(self, data, context=None):
        return "handled"


class ConcreteAsyncOrchestrator(AsyncCustomOrchestrator):
    async def handle(self, data, context=None):
        return "async_handled"


class TestCustomOrchestrator(unittest.TestCase):
    def test_init(self):
        orchestrator = ConcreteOrchestrator()
        self.assertIsNotNone(orchestrator)

    @patch("boto3.Session")
    def test_client_property(self, mock_session):
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client
        
        orchestrator = ConcreteOrchestrator()
        client = orchestrator.client
        self.assertEqual(client, mock_client)

    def test_handle(self):
        orchestrator = ConcreteOrchestrator()
        result = orchestrator.handle("data")
        self.assertEqual(result, "handled")


class TestAsyncCustomOrchestrator(unittest.TestCase):
    async def test_async_handle(self):
        orchestrator = ConcreteAsyncOrchestrator()
        result = await orchestrator.handle("data")
        self.assertEqual(result, "async_handled")


if __name__ == "__main__":
    unittest.main()
