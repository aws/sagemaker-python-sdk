# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

import numpy
import pytest

from sagemaker.rl import RLEstimator, RLFramework, RLToolkit
from sagemaker.utils import sagemaker_timestamp
from tests.integ import DATA_DIR, PYTHON_VERSION
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name

CPU_INSTANCE = 'ml.m4.xlarge'


@pytest.mark.continuous_testing
@pytest.mark.skipif(PYTHON_VERSION != 'py3', reason="RL images supports only Python 3.")
def test_coach_mxnet(sagemaker_session, rl_coach_full_version):
    estimator = _test_coach(sagemaker_session, RLFramework.MXNET, rl_coach_full_version)

    with timeout(minutes=15):
        estimator.fit(wait='False')

        estimator = RLEstimator.attach(estimator.latest_training_job.name)

    endpoint_name = 'test-mxnet-coach-deploy-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = estimator.deploy(1, CPU_INSTANCE, entry_point='mxnet_deploy.py',
                                     endpoint_name=endpoint_name)

        observation = numpy.asarray([0, 0, 0, 0])
        action = predictor.predict(observation)

    assert 0 < action[0][0] < 1
    assert 0 < action[0][1] < 1


@pytest.mark.skipif(PYTHON_VERSION != 'py3', reason="RL images supports only Python 3.")
def test_coach_tf(sagemaker_session, rl_coach_full_version):
    estimator = _test_coach(sagemaker_session, RLFramework.TENSORFLOW, rl_coach_full_version)

    with timeout(minutes=15):
        estimator.fit()

    endpoint_name = 'test-tf-coach-deploy-{}'.format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = estimator.deploy(1, CPU_INSTANCE)
        observation = numpy.asarray([0, 0, 0, 0])
        action = predictor.predict(observation)

    assert action == {'predictions': [[0.5, 0.5]]}


def _test_coach(sagemaker_session, rl_framework, rl_coach_version):
    source_dir = os.path.join(DATA_DIR, 'coach_cartpole')
    dependencies = [os.path.join(DATA_DIR, 'sagemaker_rl')]
    cartpole = 'train_coach.py'

    return RLEstimator(toolkit=RLToolkit.COACH,
                       toolkit_version=rl_coach_version,
                       framework=rl_framework,
                       entry_point=cartpole,
                       source_dir=source_dir,
                       role='SageMakerRole',
                       train_instance_count=1,
                       train_instance_type=CPU_INSTANCE,
                       sagemaker_session=sagemaker_session,
                       dependencies=dependencies,
                       hyperparameters={
                           "save_model": 1,
                           "RLCOACH_PRESET": "preset_cartpole_clippedppo",
                           "rl.agent_params.algorithm.discount": 0.9,
                           "rl.evaluation_steps:EnvironmentEpisodes": 1,
                       })


@pytest.mark.continuous_testing
@pytest.mark.skipif(PYTHON_VERSION != 'py3', reason="RL images supports only Python 3.")
def test_ray_tf(sagemaker_session, rl_ray_full_version):
    source_dir = os.path.join(DATA_DIR, 'ray_cartpole')
    cartpole = 'train_ray.py'

    estimator = RLEstimator(entry_point=cartpole,
                            source_dir=source_dir,
                            toolkit=RLToolkit.RAY,
                            framework=RLFramework.TENSORFLOW,
                            toolkit_version=rl_ray_full_version,
                            sagemaker_session=sagemaker_session,
                            role='SageMakerRole',
                            train_instance_type=CPU_INSTANCE,
                            train_instance_count=1)

    with timeout(minutes=15):
        estimator.fit()

    with pytest.raises(NotImplementedError) as e:
        estimator.deploy(1, CPU_INSTANCE)
    assert 'Automatic deployment of Ray models is not currently available' in str(e.value)
