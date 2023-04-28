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
from sagemaker.utils import retry_with_backoff


def test_end_to_end(trial_component_obj, sagemaker_session):
    # The fixture creates deletes, just ensure fixture is used at least once
    with _MetricsManager(trial_component_obj.trial_component_name, sagemaker_session) as mm:
        for i in range(100):
            mm.log_metric("test-x-step", random.random(), step=i)
            mm.log_metric("test-x-timestamp", random.random())

    def verify_metrics():
        updated_tc = _TrialComponent.load(
            trial_component_name=trial_component_obj.trial_component_name,
            sagemaker_session=sagemaker_session,
        )
        metrics = updated_tc.metrics
        assert len(metrics) == 2
        assert list(filter(lambda x: x.metric_name == "test-x-step", metrics))
        assert list(filter(lambda x: x.metric_name == "test-x-timestamp", metrics))

    # metrics -> eureka propagation
    retry_with_backoff(verify_metrics)
