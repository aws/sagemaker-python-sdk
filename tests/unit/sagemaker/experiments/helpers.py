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
from __future__ import absolute_import

from sagemaker.experiments.trial import _Trial
from sagemaker.experiments.trial_component import _TrialComponent


TEST_EXP_NAME = "my-experiment"
TEST_RUN_GRP_NAME = "my-run-group"
TEST_RUN_NAME = "my-run"


def mock_trial_component_load_func(trial_component_name, sagemaker_session=None):
    return _TrialComponent(
        trial_component_name=trial_component_name, sagemaker_session=sagemaker_session
    )


def mock_tc_load_or_create_func(
    trial_component_name, display_name=None, tags=None, sagemaker_session=None
):
    return _TrialComponent(
        trial_component_name=trial_component_name,
        display_name=display_name,
        tags=tags,
        sagemaker_session=sagemaker_session,
    )


def mock_trial_load_or_create_func(
    experiment_name, trial_name, display_name=None, tags=None, sagemaker_session=None
):
    return _Trial(
        trial_name=trial_name,
        experiment_name=experiment_name,
        display_name=display_name,
        tags=tags,
        sagemaker_session=sagemaker_session,
    )
