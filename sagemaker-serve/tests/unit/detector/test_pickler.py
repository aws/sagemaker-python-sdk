import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from sagemaker.serve.detector.pickler import save_pkl, save_xgboost, save_sklearn, load_xgboost_from_json


class TestPickler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = Path(self.temp_dir)

    def test_save_pkl(self):
        obj = {"key": "value"}
        save_pkl(self.save_path, obj)
        pkl_file = self.save_path / "serve.pkl"
        self.assertTrue(pkl_file.exists())

    def test_save_xgboost(self):
        mock_model = Mock()
        save_xgboost(self.save_path, mock_model)
        mock_model.save_model.assert_called_once()

    def test_save_sklearn(self):
        mock_model = Mock()
        with patch("joblib.dump") as mock_dump:
            save_sklearn(str(self.save_path), mock_model)
            mock_dump.assert_called_once()

    def test_load_xgboost_from_json(self):
        with patch("sagemaker.serve.detector.pickler._get_class_from_name") as mock_get_class:
            mock_class = Mock()
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            mock_get_class.return_value = mock_class
            
            result = load_xgboost_from_json("model.json", "xgboost.XGBClassifier")
            self.assertEqual(result, mock_instance)


if __name__ == "__main__":
    unittest.main()
