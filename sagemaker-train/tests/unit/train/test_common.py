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


import pytest


class TestValidateLengthConstraints:
    """FineTuningOptions.validate_length_constraints() — sum vs context_length."""

    def _rl_options(self, prompt, response, context_length):
        # Per-field max defaults to context_length; when unknown, omit it so the
        # per-field range check is a no-op and only the sum-guard is exercised.
        prompt_spec = {"type": "integer", "default": 1024, "min": 512}
        response_spec = {"type": "integer", "default": 2048, "min": 100}
        if context_length is not None:
            prompt_spec["max"] = context_length
            response_spec["max"] = context_length
        opts = FineTuningOptions(
            {"max_prompt_length": prompt_spec, "max_response_length": response_spec},
            context_length=context_length,
        )
        object.__setattr__(opts, "max_prompt_length", prompt)
        object.__setattr__(opts, "max_response_length", response)
        return opts

    def test_rl_sum_within_context_length_passes(self):
        opts = self._rl_options(prompt=4096, response=8192, context_length=262144)
        opts.validate_length_constraints()  # no raise

    def test_rl_sum_equal_to_context_length_passes(self):
        opts = self._rl_options(prompt=100000, response=162144, context_length=262144)
        opts.validate_length_constraints()  # sum == ctx, allowed

    def test_rl_sum_exceeds_context_length_raises(self):
        opts = self._rl_options(prompt=200000, response=200000, context_length=262144)
        with pytest.raises(ValueError) as exc:
            opts.validate_length_constraints()
        msg = str(exc.value)
        assert "400000" in msg and "262144" in msg

    def test_sft_max_length_within_context_length_passes(self):
        opts = FineTuningOptions(
            {"max_length": {"type": "integer", "default": 4096, "min": 4096, "max": 131072}},
            context_length=131072,
        )
        opts.max_length = 65536
        opts.validate_length_constraints()  # no raise

    def test_sft_max_length_exceeds_context_length_raises(self):
        opts = FineTuningOptions(
            {"max_length": {"type": "integer", "default": 4096, "min": 4096, "max": 131072}},
            context_length=131072,
        )
        # max is 131072, so set via _specs bypass to test the sum-guard directly
        object.__setattr__(opts, "max_length", 200000)
        with pytest.raises(ValueError) as exc:
            opts.validate_length_constraints()
        assert "131072" in str(exc.value)

    def test_no_context_length_is_noop(self):
        opts = self._rl_options(prompt=200000, response=200000, context_length=None)
        opts.validate_length_constraints()  # unknown ceiling -> no raise
