import unittest
from sagemaker.serve.utils.types import ModelServer, HardwareType, ImageUriOption, ModelHub


class TestModelServer(unittest.TestCase):
    def test_model_server_values(self):
        self.assertEqual(ModelServer.TORCHSERVE.value, 1)
        self.assertEqual(ModelServer.TGI.value, 6)
        self.assertEqual(ModelServer.TEI.value, 7)

    def test_model_server_str(self):
        self.assertEqual(str(ModelServer.TORCHSERVE), "TORCHSERVE")
        self.assertEqual(str(ModelServer.DJL_SERVING), "DJL_SERVING")


class TestHardwareType(unittest.TestCase):
    def test_hardware_type_values(self):
        self.assertEqual(HardwareType.CPU.value, 1)
        self.assertEqual(HardwareType.GPU.value, 2)
        self.assertEqual(HardwareType.INFERENTIA_2.value, 4)

    def test_hardware_type_str(self):
        self.assertEqual(str(HardwareType.CPU), "CPU")
        self.assertEqual(str(HardwareType.GPU), "GPU")


class TestImageUriOption(unittest.TestCase):
    def test_image_uri_option_values(self):
        self.assertEqual(ImageUriOption.CUSTOM_IMAGE.value, 1)
        self.assertEqual(ImageUriOption.DEFAULT_IMAGE.value, 3)

    def test_image_uri_option_str(self):
        self.assertEqual(str(ImageUriOption.CUSTOM_IMAGE), "CUSTOM_IMAGE")


class TestModelHub(unittest.TestCase):
    def test_model_hub_values(self):
        self.assertEqual(ModelHub.JUMPSTART.value, 1)
        self.assertEqual(ModelHub.HUGGINGFACE.value, 2)

    def test_model_hub_str(self):
        self.assertEqual(str(ModelHub.JUMPSTART), "JUMPSTART")
        self.assertEqual(str(ModelHub.HUGGINGFACE), "HUGGINGFACE")


if __name__ == "__main__":
    unittest.main()
