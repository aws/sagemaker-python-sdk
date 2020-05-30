import unittest


class FrameworkVersion(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
