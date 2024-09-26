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

import glob
import logging
import os
import shutil
import tempfile
import time
import uuid

import boto3
import pytest

from sagemaker.experiments import Run
from tests.integ import DATA_DIR

from sagemaker.experiments import trial_component, trial, experiment
from sagemaker.utils import retry_with_backoff, unique_name_from_base
from tests.integ.sagemaker.experiments.helpers import name, names

TAGS = [{"Key": "some-key", "Value": "some-value"}]
EXP_NAME_BASE_IN_LOCAL = "Job-Exp-in-Local"
RUN_NAME_IN_LOCAL = "job-run-in-local"


@pytest.fixture(scope="module")
def run_obj(sagemaker_session):
    run = Run(
        experiment_name=unique_name_from_base(EXP_NAME_BASE_IN_LOCAL),
        run_name=RUN_NAME_IN_LOCAL,
        sagemaker_session=sagemaker_session,
    )
    try:
        yield run
        time.sleep(0.5)
    finally:
        exp = experiment.Experiment.load(
            experiment_name=run.experiment_name, sagemaker_session=sagemaker_session
        )
        exp._delete_all(action="--force")


@pytest.fixture(scope="module")
def trial_component_obj(sagemaker_session):
    trial_component_obj = trial_component._TrialComponent.create(
        trial_component_name=name(),
        sagemaker_session=sagemaker_session,
        tags=TAGS,
    )
    yield trial_component_obj
    time.sleep(0.5)
    _delete_associations(trial_component_obj.trial_component_arn, sagemaker_session)
    retry_with_backoff(trial_component_obj.delete)


@pytest.fixture(scope="module")
def experiment_obj(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    description = "{}-{}".format("description", str(uuid.uuid4()))
    boto3.set_stream_logger("", logging.INFO)
    experiment_name = name()
    experiment_obj = experiment.Experiment.create(
        experiment_name=experiment_name,
        description=description,
        sagemaker_session=sagemaker_session,
        tags=TAGS,
    )
    yield experiment_obj
    time.sleep(0.5)
    experiment_obj.delete()
    with pytest.raises(client.exceptions.ResourceNotFound):
        client.describe_experiment(ExperimentName=experiment_name)


@pytest.fixture(scope="module")
def trial_obj(sagemaker_session, experiment_obj):
    trial_obj = trial._Trial.create(
        trial_name=name(),
        experiment_name=experiment_obj.experiment_name,
        tags=TAGS,
        sagemaker_session=sagemaker_session,
    )
    yield trial_obj
    time.sleep(0.5)
    trial_obj.delete()


@pytest.fixture(scope="module")
def trials(experiment_obj, sagemaker_session):
    trial_objs = []
    for trial_name in names():
        next_trial = trial._Trial.create(
            trial_name=trial_name,
            experiment_name=experiment_obj.experiment_name,
            sagemaker_session=sagemaker_session,
        )
        trial_objs.append(next_trial)
        time.sleep(0.5)
    yield trial_objs
    for trial_obj in trial_objs:
        trial_obj.delete()


@pytest.fixture(scope="module")
def trial_component_with_force_disassociation_obj(trials, sagemaker_session):
    trial_component_obj = trial_component._TrialComponent.create(
        trial_component_name=name(), sagemaker_session=sagemaker_session
    )
    for trial_obj in trials:
        sagemaker_session.sagemaker_client.associate_trial_component(
            TrialName=trial_obj.trial_name,
            TrialComponentName=trial_component_obj.trial_component_name,
        )
    yield trial_component_obj
    time.sleep(0.5)
    trial_component_obj.delete(force_disassociate=True)


@pytest.fixture(scope="module")
def trial_components(sagemaker_session):
    trial_component_objs = [
        trial_component._TrialComponent.create(
            trial_component_name=trial_component_name,
            sagemaker_session=sagemaker_session,
        )
        for trial_component_name in names()
    ]
    yield trial_component_objs
    for trial_component_obj in trial_component_objs:
        trial_component_obj.delete()


@pytest.fixture(scope="module")
def tempdir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


_EXP_PLUS_SDK_TAR = "sagemaker-dev-1.0.tar.gz"


@pytest.fixture(scope="module")
def dev_sdk_tar():
    resource_dir = os.path.join(DATA_DIR, "experiment")
    os.system("python -m build --sdist")
    sdist_path = max(glob.glob("dist/sagemaker-*"), key=os.path.getctime)
    sdk_file = os.path.join(resource_dir, _EXP_PLUS_SDK_TAR)
    shutil.copy(sdist_path, sdk_file)
    return sdk_file


def _delete_associations(arn, sagemaker_session):
    client = sagemaker_session.sagemaker_client
    outgoing_associations = client.list_associations(SourceArn=arn)["AssociationSummaries"]
    incoming_associations = client.list_associations(DestinationArn=arn)["AssociationSummaries"]
    associations = []
    if outgoing_associations:
        associations.extend(outgoing_associations)
    if incoming_associations:
        associations.extend(incoming_associations)
    for association in associations:
        source_arn = association["SourceArn"]
        destination_arn = association["DestinationArn"]
        client.delete_association(SourceArn=source_arn, DestinationArn=destination_arn)
