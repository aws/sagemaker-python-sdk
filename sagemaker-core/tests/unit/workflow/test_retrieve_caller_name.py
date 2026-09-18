# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Tests for `retrieve_caller_name`, including the CreateJob declared-name path.

`retrieve_caller_name` decides the `caller_name` carried on the `_StepArguments` a
`@runnable_by_pipeline` producer returns, and that value is what every step class
checks through `validate_step_args_input(expected_caller=...)`. It is a distinct
namespace from the `func_name` passed to `PipelineSession._intercept_create_request`
-- `ProcessingStep` expects `"run"` from here while interception is called with
`"process"` -- so a step guard cannot be satisfied by the interception name alone.
"""
from __future__ import absolute_import

import pytest

from sagemaker.core.workflow.pipeline_context import (
    _DECLARABLE_CALLER_NAMES,
    _StepArguments,
    retrieve_caller_name,
)
from sagemaker.core.workflow.utilities import validate_step_args_input


class _Declared:
    """A CreateJob-backed producer, declaring its caller name."""

    _pipeline_caller_name = "create_job"


class _DeclaredAndTrainerShaped(_Declared):
    """Declares create_job AND carries the ModelTrainer structural signature.

    A synthetic shape, kept because it pins the ordering against the `training_image`
    marker specifically -- no real producer has this combination. For MultiTurnRLTrainer's
    ACTUAL shape, and the fixture that would catch a live regression in it, see
    `_DeclaredAndFinetuneShaped` below. MTRL has no training_image, so this fixture does
    not cover it.
    """

    training_image = "an-image"

    def train(self):  # pragma: no cover - never invoked
        raise AssertionError("train() must not be called by name resolution")


class _UndeclaredCreateJobShaped:
    """A CreateJob producer that has NOT opted in -- e.g. GenAIEvaluator today."""

    def evaluate(self):  # pragma: no cover - never invoked
        raise AssertionError("evaluate() must not be called by name resolution")

    def build_config(self):  # pragma: no cover - never invoked
        raise AssertionError("build_config() must not be called by name resolution")


class _TrainerShaped:
    training_image = "an-image"

    def train(self):  # pragma: no cover - never invoked
        raise AssertionError("train() must not be called by name resolution")


class _TunerShaped:
    model_trainer = object()

    def tune(self):  # pragma: no cover - never invoked
        raise AssertionError("tune() must not be called by name resolution")


def test_declared_create_job_producer_resolves_to_create_job():
    assert retrieve_caller_name(_Declared()) == "create_job"


def test_declared_name_is_resolved_before_the_structural_trainer_check():
    """The ordering guard. A declared CreateJob producer must not be read as a trainer."""
    assert retrieve_caller_name(_DeclaredAndTrainerShaped()) == "create_job"


def test_an_undeclared_create_job_producer_still_resolves_to_none():
    """Fail-closed: opting in is explicit, so today's undecorated producers are unchanged."""
    assert retrieve_caller_name(_UndeclaredCreateJobShaped()) is None


@pytest.mark.parametrize("claimed", ["train", "transform", "tune", "run", "process", ""])
def test_a_producer_cannot_declare_a_name_outside_the_allowlist(claimed):
    """A producer must not be able to claim a name resolved structurally elsewhere.

    Declaring "train" would let a CreateJob producer compose a TrainingStep over a
    different create API -- the same class of confusion `JobStep`'s guard exists to
    prevent.
    """

    class _Claiming:
        _pipeline_caller_name = claimed

    assert retrieve_caller_name(_Claiming()) is None


def test_the_declarable_set_is_pinned():
    """Widening the set is a deliberate change, so it has to move this assertion too."""
    assert _DECLARABLE_CALLER_NAMES == frozenset({"create_job"})


def test_structural_resolution_is_unchanged():
    assert retrieve_caller_name(_TrainerShaped()) == "train"
    assert retrieve_caller_name(_TunerShaped()) == "tune"
    assert retrieve_caller_name(object()) is None


def test_a_declared_producer_satisfies_the_job_step_guard():
    """The payoff: what `JobStep` pins is now reachable.

    `JobStep` calls validate_step_args_input(expected_caller={"create_job"}), which
    reads `_StepArguments.caller_name`. Before this branch existed no producer could
    produce that value, so the guard rejected every capturable producer.
    """
    step_args = _StepArguments(retrieve_caller_name(_Declared()))

    validate_step_args_input(
        step_args=step_args,
        expected_caller={"create_job"},
        error_message="should not raise",
    )


def test_the_job_step_guard_still_rejects_a_trainer_capture():
    step_args = _StepArguments(retrieve_caller_name(_TrainerShaped()))

    with pytest.raises(ValueError):
        validate_step_args_input(
            step_args=step_args,
            expected_caller={"create_job"},
            error_message="a TrainingStep capture must not compose a JobStep",
        )


# ---------------------------------------------------------------------------
# The BaseTrainer family. The train branch keys on `training_image` OR
# `input_data_config`, because the four V3 finetune trainers carry no
# training_image -- they resolve the image from the recipe.
# ---------------------------------------------------------------------------


class _FinetuneTrainerShaped:
    """An SFT/DPO/RLVR/RLAIF trainer: train() and input_data_config, no training_image."""

    input_data_config = None

    def train(self):  # pragma: no cover - never invoked
        raise AssertionError("train() must not be called by name resolution")


class _DeclaredAndFinetuneShaped(_Declared):
    """MultiTurnRLTrainer's ACTUAL shape, which the training_image fixture does not cover.

    MTRL subclasses BaseTrainer, so it already carries `input_data_config` and already
    satisfies the train branch structurally. It has no training_image, so
    `_DeclaredAndTrainerShaped` above would not catch a regression here.
    """

    input_data_config = None

    def train(self):  # pragma: no cover - never invoked
        raise AssertionError("train() must not be called by name resolution")


class _DataPreparerShaped:
    """A CreateJob producer with no train(): attach()/stop(), plus a config attribute."""

    input_data_config = None

    def attach(self):  # pragma: no cover - never invoked
        raise AssertionError("attach() must not be called by name resolution")

    def stop(self):  # pragma: no cover - never invoked
        raise AssertionError("stop() must not be called by name resolution")


def test_a_finetune_trainer_resolves_to_train():
    """The fix. Keying on training_image alone left these four unrecognised.

    An unrecognised producer returns None, `_StepArguments` is built with it happily,
    and TrainingStep then rejects it at construction -- before the producer's own
    interception block is ever reached.
    """
    assert retrieve_caller_name(_FinetuneTrainerShaped()) == "train"


def test_a_finetune_trainer_satisfies_the_training_step_guard():
    """What TrainingStep pins is now reachable for the BaseTrainer family."""
    step_args = _StepArguments(retrieve_caller_name(_FinetuneTrainerShaped()))

    validate_step_args_input(
        step_args=step_args,
        expected_caller={"train"},
        error_message="should not raise",
    )


def test_the_declaration_is_what_keeps_mtrl_out_of_the_train_branch():
    """The ordering guard, at MTRL's real shape rather than a training_image stand-in.

    This is now load-bearing rather than hypothetical: MTRL carries input_data_config
    from BaseTrainer, so it satisfies the train branch structurally today. Only the
    declared name, resolved first, keeps it composing a JobStep instead of a
    TrainingStep over CreateJob arguments.
    """
    assert retrieve_caller_name(_DeclaredAndFinetuneShaped()) == "create_job"


def test_both_train_markers_are_independently_sufficient():
    """A union, not a replacement -- neither marker may become required."""
    assert retrieve_caller_name(_TrainerShaped()) == "train"  # training_image only
    assert retrieve_caller_name(_FinetuneTrainerShaped()) == "train"  # input_data_config only


def test_the_config_marker_alone_does_not_resolve_to_train():
    """`train` stays a required conjunct, so widening cannot reach a non-trainer."""

    class _ConfigOnly:
        input_data_config = None

    assert retrieve_caller_name(_ConfigOnly()) is None


def test_an_undeclared_create_job_producer_without_train_is_still_unrecognised():
    """DataPreparer's shape. It fails the `train` conjunct, so the widening cannot claim it.

    DataPreparer is the CreateJob producer that is not yet pipeline-capable. When it is
    onboarded it declares `create_job` like the others; until then it must not be read
    as a trainer merely for holding a config attribute.
    """
    assert retrieve_caller_name(_DataPreparerShaped()) is None
