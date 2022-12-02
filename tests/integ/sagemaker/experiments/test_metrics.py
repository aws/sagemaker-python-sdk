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
import random
from sagemaker.experiments._metrics import _MetricsManager
from sagemaker.experiments.trial_component import _TrialComponent


def test_epoch(trial_component_obj, sagemaker_session):
    # The fixture creates deletes, just ensure fixture is used at least once
    metric_name = "test-x-step"
    with _MetricsManager(trial_component_obj.trial_component_arn, sagemaker_session) as mm:
        for i in range(100):
            mm.log_metric(metric_name, random.random(), step=i)

    updated_tc = _TrialComponent.load(
        trial_component_name=trial_component_obj.trial_component_name,
        sagemaker_session=sagemaker_session,
    )
    assert len(updated_tc.metrics) == 1
    assert updated_tc.metrics[0].metric_name == metric_name


def test_timestamp(trial_component_obj, sagemaker_session):
    # The fixture creates deletes, just ensure fixture is used at least once
    metric_name = "test-x-timestamp"
    with _MetricsManager(trial_component_obj.trial_component_arn, sagemaker_session) as mm:
        for i in range(100):
            mm.log_metric(metric_name, random.random())

    updated_tc = _TrialComponent.load(
        trial_component_name=trial_component_obj.trial_component_name,
        sagemaker_session=sagemaker_session,
    )
    # the test-x-step data is added in the previous test_epoch test
    assert len(updated_tc.metrics) == 2
    assert updated_tc.metrics[0].metric_name == "test-x-step"
    assert updated_tc.metrics[1].metric_name == "test-x-timestamp"
