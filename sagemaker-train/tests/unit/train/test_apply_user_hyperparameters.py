"""Unit tests for BaseTrainer._apply_user_hyperparameters.

These lock in that hyperparameters supplied at trainer construction are re-applied
onto the spec-backed FineTuningOptions (through its validating __setattr__) instead of
being silently dropped when the trainer rebuilds hyperparameters from the model spec.
"""
import pytest

from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.train.common import FineTuningOptions


def _make_options():
    return FineTuningOptions(
        {
            "learning_rate": {"type": "float", "default": 0.0001, "min": 0.0, "max": 1.0},
            "epochs": {"type": "integer", "default": 1, "min": 1, "max": 10},
        }
    )


class _DummyTrainer:
    """Minimal stand-in exposing only what the helper touches."""


def _apply(hyperparameters, user_hyperparameters):
    trainer = _DummyTrainer()
    trainer.hyperparameters = hyperparameters
    # Call the unbound helper; it only depends on self.hyperparameters.
    BaseTrainer._apply_user_hyperparameters(trainer, user_hyperparameters)
    return trainer


def test_applies_valid_values_and_marks_user_set():
    options = _make_options()
    trainer = _apply(options, {"learning_rate": 0.001, "epochs": 3})

    assert trainer.hyperparameters.learning_rate == 0.001
    assert trainer.hyperparameters.epochs == 3
    # Values applied via __setattr__ are tracked as explicitly user-set.
    assert trainer.hyperparameters._user_set == {"learning_rate", "epochs"}


def test_invalid_option_name_is_ignored_with_warning(caplog):
    options = _make_options()
    with caplog.at_level("WARNING"):
        trainer = _apply(options, {"not_a_real_option": 1, "learning_rate": 0.001})

    # The overridable value is applied; the non-overridable one is skipped.
    assert trainer.hyperparameters.learning_rate == 0.001
    assert not hasattr(trainer.hyperparameters, "not_a_real_option")
    assert trainer.hyperparameters._user_set == {"learning_rate"}
    # A warning names the ignored, non-overridable hyperparameter.
    assert "not_a_real_option" in caplog.text
    assert "not overridable" in caplog.text.lower()


def test_out_of_spec_value_raises():
    options = _make_options()
    with pytest.raises(ValueError):
        _apply(options, {"learning_rate": 5.0})  # overridable name, but exceeds max of 1.0


def test_empty_user_hyperparameters_is_noop():
    options = _make_options()
    trainer = _apply(options, {})
    assert trainer.hyperparameters._user_set == set()

    trainer_none = _apply(options, None)
    assert trainer_none.hyperparameters._user_set == set()


def test_non_finetuning_options_container_is_noop():
    # A plain dict has no ``_specs``; the helper must not raise or mutate it.
    plain = {"existing": 1}
    trainer = _apply(plain, {"learning_rate": 0.001})
    assert trainer.hyperparameters == {"existing": 1}
