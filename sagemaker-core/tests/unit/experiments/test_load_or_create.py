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
"""Regression tests for the load-or-create flows in the experiments modules.

The service changed its duplicate-name error wording from '... already exists'
to '... names must be unique within an AWS account ...'. These tests pin the
already-exists branch against BOTH wordings so a future wording change cannot
silently turn load-or-create back into a hard failure.
"""
from __future__ import absolute_import

import pytest
from unittest.mock import patch
from botocore.exceptions import ClientError

from sagemaker.core.experiments.experiment import Experiment
from sagemaker.core.experiments.trial import _Trial
from sagemaker.core.experiments.trial_component import _TrialComponent


LEGACY_MESSAGE = "Experiment exp-1 already exists"
NEW_MESSAGE = "Experiment names must be unique within an AWS account and region"


def _validation_error(message):
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": message}}, "create"
    )


@pytest.mark.parametrize("message", [LEGACY_MESSAGE, NEW_MESSAGE])
def test_experiment_load_or_create_loads_on_duplicate_name(message):
    with patch.object(Experiment, "create", side_effect=_validation_error(message)):
        with patch.object(Experiment, "load") as mock_load:
            result = Experiment._load_or_create(experiment_name="exp-1")

    mock_load.assert_called_once_with("exp-1", None)
    assert result is mock_load.return_value


def test_experiment_load_or_create_reraises_unrelated_validation_error():
    with patch.object(
        Experiment, "create", side_effect=_validation_error("1 validation error detected")
    ):
        with patch.object(Experiment, "load") as mock_load:
            with pytest.raises(ClientError):
                Experiment._load_or_create(experiment_name="exp-1")

    mock_load.assert_not_called()


@pytest.mark.parametrize("message", [LEGACY_MESSAGE, NEW_MESSAGE])
def test_trial_load_or_create_loads_on_duplicate_name(message):
    with patch.object(_Trial, "create", side_effect=_validation_error(message)):
        with patch.object(_Trial, "load") as mock_load:
            mock_load.return_value.experiment_name = "exp-1"
            result = _Trial._load_or_create(experiment_name="exp-1", trial_name="trial-1")

    mock_load.assert_called_once_with("trial-1", None)
    assert result is mock_load.return_value


@pytest.mark.parametrize("message", [LEGACY_MESSAGE, NEW_MESSAGE])
def test_trial_component_load_or_create_loads_on_duplicate_name(message):
    with patch.object(_TrialComponent, "create", side_effect=_validation_error(message)):
        with patch.object(_TrialComponent, "load") as mock_load:
            result, is_existed = _TrialComponent._load_or_create(
                trial_component_name="tc-1"
            )

    mock_load.assert_called_once_with("tc-1", None)
    assert result is mock_load.return_value
    assert is_existed is True
