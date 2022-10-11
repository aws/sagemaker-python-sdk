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

import base64
import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import boto3
import pytest

import docker

from tests.integ import lock
from tests.integ.utils import create_repository
from tests.integ import DATA_DIR

from sagemaker.experiments import trial_component, trial, experiment
from sagemaker.s3 import S3Uploader
from sagemaker.utils import retry_with_backoff
from tests.integ.sagemaker.experiments.helpers import name, names

TAGS = [{"Key": "some-key", "Value": "some-value"}]


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
    experiment_obj = experiment._Experiment.create(
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


@pytest.fixture(scope="module")
def bucket(sagemaker_session):
    return sagemaker_session.default_bucket()


@pytest.fixture(scope="module")
def training_input_s3_uri(sagemaker_session, tempdir, bucket):
    filepath = os.path.join(tempdir, name())
    with open(filepath, "w") as w:
        w.write("Hello World!")
    s3_uri = f"s3://{bucket}/experiments/training-input/{name()}"
    return S3Uploader.upload(
        local_path=filepath, desired_s3_uri=s3_uri, sagemaker_session=sagemaker_session
    )


@pytest.fixture(scope="module")
def training_output_s3_uri(bucket):
    return f"s3://{bucket}/experiments/training-output/"


# TODO we should remove the boto model file once the Run API changes release
BOTO_MODEL_LOCAL_PATH = os.path.join(DATA_DIR, "experiment", "sagemaker-2017-07-24.normal.json")
METRICS_MODEL_LOCAL_PATH = os.path.join(
    DATA_DIR, "experiment", "sagemaker-metrics-2022-09-30.normal.json"
)
IMAGE_REPO_NAME = "sagemaker-experiments-test"
IMAGE_VERSION = "1.0.92"  # We should bump it up if need to update the docker image
SM_SDK_TAR_NAME_IN_IMAGE = "sagemaker-dev.tar.gz"
SM_BOTO_MODEL_PATH_IN_IMAGE = "boto/sagemaker-2017-07-24.normal.json"
SM_METRICS_MODEL_PATH_IN_IMAGE = "boto/sagemaker-metrics-2022-09-30.normal.json"


@pytest.fixture(scope="module")
def docker_image(sagemaker_session):
    # requires docker to be running
    docker_client = docker.from_env()
    ecr_client = sagemaker_session.boto_session.client("ecr")

    token = ecr_client.get_authorization_token()
    username, password = (
        base64.b64decode(token["authorizationData"][0]["authorizationToken"]).decode().split(":")
    )
    registry = token["authorizationData"][0]["proxyEndpoint"]
    repository_name = IMAGE_REPO_NAME
    tag = "{}/{}:{}".format(registry, repository_name, IMAGE_VERSION)[8:]
    docker_dir = os.path.join(DATA_DIR, "experiment", "docker")

    with lock.lock():
        # initialize the docker image repository
        create_repository(ecr_client, repository_name)

        # pull existing image for layer cache
        try:
            docker_client.images.pull(tag, auth_config={"username": username, "password": password})
            print("Docker image with tag {} already exists.".format(tag))
            return tag
        except docker.errors.NotFound:
            print("Docker image with tag {} does not exist. Will create one.".format(tag))

        # copy boto model under docker dir
        os.makedirs(os.path.join(docker_dir, "boto"), exist_ok=True)
        shutil.copy(
            BOTO_MODEL_LOCAL_PATH,
            os.path.join(docker_dir, SM_BOTO_MODEL_PATH_IN_IMAGE),
        )
        shutil.copy(
            METRICS_MODEL_LOCAL_PATH,
            os.path.join(docker_dir, SM_METRICS_MODEL_PATH_IN_IMAGE),
        )

        # generate sdk tar file from package and put it under docker dir
        subprocess.check_call([sys.executable, "setup.py", "sdist"])
        sdist_path = max(glob.glob("dist/sagemaker-*"), key=os.path.getctime)
        shutil.copy(sdist_path, os.path.join(docker_dir, SM_SDK_TAR_NAME_IN_IMAGE))

        docker_client.images.build(
            path=docker_dir,
            dockerfile="Dockerfile",
            tag=tag,
            cache_from=[tag],
            buildargs={
                "library": SM_SDK_TAR_NAME_IN_IMAGE,
                "botomodel": SM_BOTO_MODEL_PATH_IN_IMAGE,
                "script": "scripts/train_job_script_for_run_clz.py",
                "metricsmodel": SM_METRICS_MODEL_PATH_IN_IMAGE,
            },
        )
        docker_client.images.push(tag, auth_config={"username": username, "password": password})
        return tag


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
