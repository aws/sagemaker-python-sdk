from sagemaker.train.common import FineTuningOptions


class TestFineTuningOptionsToDict:
    """Tests for FineTuningOptions.to_dict() None value handling."""

    def test_to_dict_skips_none_values(self):
        """None-valued hyperparameters should be omitted from to_dict output."""
        options = FineTuningOptions({
            "learning_rate": {"default": 0.0002, "type": "float"},
            "resume_from_path": {"default": None, "type": "string"},
            "global_batch_size": {"default": 64, "type": "integer"},
        })
        result = options.to_dict()
        assert "resume_from_path" not in result
        assert result == {"learning_rate": "0.0002", "global_batch_size": "64"}

    def test_to_dict_includes_non_none_values(self):
        """Non-None values should be included as strings."""
        options = FineTuningOptions({
            "learning_rate": {"default": 0.001, "type": "float"},
            "max_epochs": {"default": 3, "type": "integer"},
            "model_name": {"default": "my-model", "type": "string"},
        })
        result = options.to_dict()
        assert result == {
            "learning_rate": "0.001",
            "max_epochs": "3",
            "model_name": "my-model",
        }

    def test_to_dict_empty_string_is_included(self):
        """Empty string is a valid value and should not be skipped."""
        options = FineTuningOptions({
            "mlflow_run_id": {"default": "", "type": "string"},
        })
        result = options.to_dict()
        assert result == {"mlflow_run_id": ""}

    def test_to_dict_after_user_sets_none_to_value(self):
        """If user overrides a None default with a real value, it should appear."""
        options = FineTuningOptions({
            "resume_from_path": {"default": None, "type": "string"},
        })
        options.resume_from_path = "/path/to/checkpoint"
        result = options.to_dict()
        assert result == {"resume_from_path": "/path/to/checkpoint"}

    def test_to_dict_all_none_returns_empty(self):
        """If all values are None, to_dict should return empty dict."""
        options = FineTuningOptions({
            "param_a": {"default": None, "type": "string"},
            "param_b": {"default": None, "type": "string"},
        })
        result = options.to_dict()
        assert result == {}
