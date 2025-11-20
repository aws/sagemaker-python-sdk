import unittest
from sagemaker.serve.constants import (
    Framework,
    LOCAL_MODES,
    SUPPORTED_MODEL_SERVERS,
    DEFAULT_SERIALIZERS_BY_FRAMEWORK
)
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.types import ModelServer


class TestFramework(unittest.TestCase):
    def test_framework_values(self):
        self.assertEqual(Framework.PYTORCH.value, "PyTorch")
        self.assertEqual(Framework.TENSORFLOW.value, "TensorFlow")
        self.assertEqual(Framework.XGBOOST.value, "XGBoost")
        self.assertEqual(Framework.SKLEARN.value, "SKLearn")
        self.assertEqual(Framework.HUGGINGFACE.value, "HuggingFace")

    def test_all_frameworks_exist(self):
        expected_frameworks = [
            "XGBOOST", "LDA", "PYTORCH", "TENSORFLOW", "MXNET",
            "CHAINER", "SKLEARN", "HUGGINGFACE", "DJL", "SPARKML", "NTM", "SMD"
        ]
        for fw in expected_frameworks:
            self.assertTrue(hasattr(Framework, fw))


class TestLocalModes(unittest.TestCase):
    def test_local_modes_contains_expected_modes(self):
        self.assertIn(Mode.LOCAL_CONTAINER, LOCAL_MODES)
        self.assertIn(Mode.IN_PROCESS, LOCAL_MODES)
        self.assertEqual(len(LOCAL_MODES), 2)


class TestSupportedModelServers(unittest.TestCase):
    def test_supported_model_servers(self):
        expected_servers = {
            ModelServer.TORCHSERVE,
            ModelServer.TRITON,
            ModelServer.DJL_SERVING,
            ModelServer.TENSORFLOW_SERVING,
            ModelServer.MMS,
            ModelServer.TGI,
            ModelServer.TEI,
            ModelServer.SMD,
        }
        self.assertEqual(SUPPORTED_MODEL_SERVERS, expected_servers)


class TestDefaultSerializersByFramework(unittest.TestCase):
    def test_all_frameworks_have_serializers(self):
        for framework in Framework:
            self.assertIn(framework, DEFAULT_SERIALIZERS_BY_FRAMEWORK)
            serializer, deserializer = DEFAULT_SERIALIZERS_BY_FRAMEWORK[framework]
            self.assertIsNotNone(serializer)
            self.assertIsNotNone(deserializer)

    def test_pytorch_serializers(self):
        serializer, deserializer = DEFAULT_SERIALIZERS_BY_FRAMEWORK[Framework.PYTORCH]
        self.assertEqual(serializer.__class__.__name__, "TorchTensorSerializer")
        self.assertEqual(deserializer.__class__.__name__, "JSONDeserializer")

    def test_tensorflow_serializers(self):
        serializer, deserializer = DEFAULT_SERIALIZERS_BY_FRAMEWORK[Framework.TENSORFLOW]
        self.assertEqual(serializer.__class__.__name__, "NumpySerializer")
        self.assertEqual(deserializer.__class__.__name__, "JSONDeserializer")

    def test_sklearn_serializers(self):
        serializer, deserializer = DEFAULT_SERIALIZERS_BY_FRAMEWORK[Framework.SKLEARN]
        self.assertEqual(serializer.__class__.__name__, "NumpySerializer")
        self.assertEqual(deserializer.__class__.__name__, "NumpyDeserializer")


if __name__ == "__main__":
    unittest.main()
