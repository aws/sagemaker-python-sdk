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
"""Status recorded by Run.__exit__."""

from __future__ import absolute_import

from unittest.mock import MagicMock, patch

import pytest

from sagemaker.core.experiments.run import Run
from sagemaker.core.experiments.trial_component import _TrialComponent


def _run_with_trial_component():
    # Skip the network-bound __init__; __exit__ only needs these attributes.
    run = Run.__new__(Run)
    run._trial_component = _TrialComponent(
        sagemaker_session=MagicMock(), trial_component_name="trial-component"
    )
    run._in_load = False
    run._inside_init_context = True
    run._inside_load_context = False
    return run


@pytest.mark.parametrize("message_length, expected_length", [(2000, 1024), (10, 10)])
def test_exit_with_exception_limits_status_message_to_api_maximum(message_length, expected_length):
    """UpdateTrialComponent rejects Status.Message longer than 1024 characters."""
    run = _run_with_trial_component()
    error = RuntimeError("x" * message_length)

    with patch.object(Run, "close") as mock_close:
        run.__exit__(type(error), error, None)

    status = run._trial_component.status
    assert status.primary_status == "Failed"
    assert len(status.message) == expected_length
    mock_close.assert_called_once()
