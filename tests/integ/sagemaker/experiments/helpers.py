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

from contextlib import contextmanager
import pytest
import logging

from sagemaker import utils
from sagemaker.experiments.experiment import Experiment
from sagemaker.experiments._run_context import _RunContext

EXP_INTEG_TEST_NAME_PREFIX = "experiments-integ"


def name():
    return utils.unique_name_from_base(EXP_INTEG_TEST_NAME_PREFIX)


def names():
    return [utils.unique_name_from_base(EXP_INTEG_TEST_NAME_PREFIX) for i in range(3)]


def to_seconds(dt):
    return int(dt.timestamp())


@contextmanager
def cleanup_exp_resources(exp_names, sagemaker_session):
    try:
        yield
    finally:
        for exp_name in exp_names:
            exp = Experiment.load(experiment_name=exp_name, sagemaker_session=sagemaker_session)
            exp._delete_all(action="--force")

@pytest.fixture
def clear_run_context():
    current_run = _RunContext.get_current_run()
    if current_run == None:
        return

    logging.info(
        f"RunContext already populated by run {current_run.run_name}"
        f" in experiment {current_run.experiment_name}."
        " Clearing context manually"
    )
    _RunContext.drop_current_run()
