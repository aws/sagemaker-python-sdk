from unittest.mock import patch

import pytest

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
    """FineTuningOptions.validate_length_constraints() — sum vs sequence_length."""

    def _rl_options(self, prompt, response, sequence_length):
        # Per-field max defaults to sequence_length; when unknown, omit it so the
        # per-field range check is a no-op and only the sum-guard is exercised.
        prompt_spec = {"type": "integer", "default": 1024, "min": 512}
        response_spec = {"type": "integer", "default": 2048, "min": 100}
        if sequence_length is not None:
            prompt_spec["max"] = sequence_length
            response_spec["max"] = sequence_length
        opts = FineTuningOptions(
            {"max_prompt_length": prompt_spec, "max_response_length": response_spec},
            sequence_length=sequence_length,
        )
        object.__setattr__(opts, "max_prompt_length", prompt)
        object.__setattr__(opts, "max_response_length", response)
        return opts

    def test_rl_sum_within_sequence_length_passes(self):
        opts = self._rl_options(prompt=4096, response=8192, sequence_length=262144)
        opts.validate_length_constraints()  # no raise

    def test_rl_sum_equal_to_sequence_length_passes(self):
        opts = self._rl_options(prompt=100000, response=162144, sequence_length=262144)
        opts.validate_length_constraints()  # sum == ceiling, allowed

    def test_rl_sum_exceeds_sequence_length_raises(self):
        opts = self._rl_options(prompt=200000, response=200000, sequence_length=262144)
        with pytest.raises(ValueError) as exc:
            opts.validate_length_constraints()
        msg = str(exc.value)
        assert "400000" in msg and "262144" in msg

    def test_no_sequence_length_is_noop(self):
        opts = self._rl_options(prompt=200000, response=200000, sequence_length=None)
        opts.validate_length_constraints()  # unknown ceiling -> no raise

    # verl/llmft SFT/DPO recipes vend the single-example ceiling as
    # "dataset_max_len". Validation must enforce it so the ceiling is not
    # silently unenforced for these frameworks.
    def test_sft_dataset_max_len_within_sequence_length_passes(self):
        opts = FineTuningOptions(
            {"dataset_max_len": {"type": "integer", "default": 4096, "min": 4096, "max": 131072}},
            sequence_length=131072,
        )
        opts.dataset_max_len = 65536
        opts.validate_length_constraints()  # no raise

    def test_sft_dataset_max_len_exceeds_sequence_length_raises(self):
        opts = FineTuningOptions(
            {"dataset_max_len": {"type": "integer", "default": 4096, "min": 4096, "max": 131072}},
            sequence_length=131072,
        )
        # Bypass the per-field max so the sequence-length guard is what raises.
        object.__setattr__(opts, "dataset_max_len", 200000)
        with pytest.raises(ValueError) as exc:
            opts.validate_length_constraints()
        msg = str(exc.value)
        assert "dataset_max_len" in msg and "131072" in msg

    def test_framework_agnostic_gated_on_sequence_length_metadata_only(self):
        # No sequence_length metadata (e.g. a recipe that does not vend it) -> the
        # ceiling is unknown and validation is a no-op regardless of param name.
        opts = FineTuningOptions(
            {"dataset_max_len": {"type": "integer", "default": 4096, "min": 4096}},
            sequence_length=None,
        )
        object.__setattr__(opts, "dataset_max_len", 999999)
        opts.validate_length_constraints()  # unknown ceiling -> no raise


class TestFineTuningOptionsValidationTelemetry:
    """Failure-only telemetry emitted from FineTuningOptions.__setattr__.

    A single FAILURE event (via the core _emit_failure_telemetry helper) is emitted
    when a caller sets an invalid option name or an out-of-spec value, and NOTHING is
    emitted on the happy path (valid sets, internal sets, or construction).
    """

    _SPECS = {
        "learning_rate": {"default": 1e-4, "type": "float", "min": 1e-7, "max": 1.0},
        "num_epochs": {"default": 3, "type": "integer", "min": 1, "max": 100},
    }

    @patch("sagemaker.train.common._emit_failure_telemetry")
    def test_invalid_option_name_emits_failure(self, mock_emit):
        options = FineTuningOptions(self._SPECS)
        with pytest.raises(AttributeError):
            options.not_a_real_param = 5

        mock_emit.assert_called_once()
        args = mock_emit.call_args[0]
        assert args[1] == "FineTuningOptions.__setattr__"  # func_name
        assert isinstance(args[2], AttributeError)  # exc

    @patch("sagemaker.train.common._emit_failure_telemetry")
    def test_out_of_range_value_emits_failure(self, mock_emit):
        options = FineTuningOptions(self._SPECS)
        with pytest.raises(ValueError):
            options.learning_rate = 999.0  # exceeds max=1.0

        mock_emit.assert_called_once()
        assert isinstance(mock_emit.call_args[0][2], ValueError)

    @patch("sagemaker.train.common._emit_failure_telemetry")
    def test_wrong_type_value_emits_failure(self, mock_emit):
        options = FineTuningOptions(self._SPECS)
        with pytest.raises(ValueError):
            options.num_epochs = "three"  # not an integer

        mock_emit.assert_called_once()
        assert isinstance(mock_emit.call_args[0][2], ValueError)

    @patch("sagemaker.train.common._emit_failure_telemetry")
    def test_valid_set_does_not_emit(self, mock_emit):
        options = FineTuningOptions(self._SPECS)
        options.learning_rate = 1e-3  # within range -> no telemetry
        assert options.learning_rate == 1e-3
        mock_emit.assert_not_called()

    @patch("sagemaker.train.common._emit_failure_telemetry")
    def test_construction_does_not_emit(self, mock_emit):
        # __init__ sets defaults via super().__setattr__ and internal _-prefixed attrs,
        # none of which should emit telemetry.
        FineTuningOptions(self._SPECS)
        mock_emit.assert_not_called()

    @patch("sagemaker.train.common._emit_failure_telemetry")
    def test_emits_model_customization_feature(self, mock_emit):
        from sagemaker.core.telemetry.constants import Feature

        options = FineTuningOptions(self._SPECS)
        with pytest.raises(AttributeError):
            options.bogus = 1
        assert mock_emit.call_args[0][0] == Feature.MODEL_CUSTOMIZATION
