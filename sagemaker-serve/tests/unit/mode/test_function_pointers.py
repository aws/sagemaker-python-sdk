import unittest
from sagemaker.serve.mode.function_pointers import Mode


class TestMode(unittest.TestCase):
    def test_mode_values(self):
        self.assertEqual(Mode.IN_PROCESS.value, 1)
        self.assertEqual(Mode.LOCAL_CONTAINER.value, 2)
        self.assertEqual(Mode.SAGEMAKER_ENDPOINT.value, 3)

    def test_mode_str(self):
        self.assertEqual(str(Mode.IN_PROCESS), "IN_PROCESS")
        self.assertEqual(str(Mode.LOCAL_CONTAINER), "LOCAL_CONTAINER")
        self.assertEqual(str(Mode.SAGEMAKER_ENDPOINT), "SAGEMAKER_ENDPOINT")


if __name__ == "__main__":
    unittest.main()
