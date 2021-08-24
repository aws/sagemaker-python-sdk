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
"""This module contains code to configure Lineage integration tests"""
from __future__ import absolute_import

import time

import boto3
import pytest
import logging
import uuid
from sagemaker.lineage import (
    action,
    context,
    association,
    artifact,
)

from smexperiments import trial_component, trial, experiment

from tests.integ.sagemaker.lineage.helpers import name, names

SLEEP_TIME_SECONDS = 1


@pytest.fixture
def action_obj(sagemaker_session):
    obj = action.Action.create(
        action_name=name(),
        action_type="bar",
        source_uri="bazz",
        status="InProgress",
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete()


@pytest.fixture
def endpoint_deployment_action_obj(sagemaker_session):
    obj = action.Action.create(
        action_name=name(),
        action_type="Action",
        source_uri="bazz",
        status="InProgress",
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete(disassociate=True)


@pytest.fixture
def endpoint_action_obj(sagemaker_session):
    obj = action.Action.create(
        action_name=name(),
        action_type="ModelDeployment",
        source_uri="bazz",
        status="InProgress",
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete(disassociate=True)


@pytest.fixture
def action_obj_with_association(sagemaker_session, artifact_obj):
    obj = action.Action.create(
        action_name=name(),
        action_type="bar",
        source_uri="bazz",
        status="InProgress",
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )
    association.Association.create(
        source_arn=obj.action_arn,
        destination_arn=artifact_obj.artifact_arn,
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete(disassociate=True)


@pytest.fixture
def action_objs(sagemaker_session):
    action_objs = []
    for action_name in names():
        action_objs.append(
            action.Action.create(
                action_name=action_name,
                action_type="SDKIntegrationTest",
                source_uri="foo",
                status="InProgress",
                properties={"k1": "v1"},
                sagemaker_session=sagemaker_session,
            )
        )
        time.sleep(SLEEP_TIME_SECONDS)

    yield action_objs
    for action_obj in action_objs:
        action_obj.delete()


@pytest.fixture
def artifact_obj(sagemaker_session):
    obj = artifact.Artifact.create(
        artifact_name="SDKIntegrationTest",
        artifact_type="SDKIntegrationTest",
        source_uri=name(),
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete()


@pytest.fixture
def artifact_obj_with_association(sagemaker_session, artifact_obj):
    obj = artifact.Artifact.create(
        artifact_name="foo",
        artifact_type="SDKIntegrationTest",
        source_uri=name(),
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )
    association.Association.create(
        source_arn=obj.artifact_arn,
        destination_arn=artifact_obj.artifact_arn,
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete(disassociate=True)


@pytest.fixture
def trial_component_obj(sagemaker_session):
    trial_component_obj = trial_component.TrialComponent.create(
        trial_component_name=name(),
        sagemaker_boto_client=sagemaker_session.sagemaker_client,
    )
    yield trial_component_obj
    time.sleep(0.5)
    trial_component_obj.delete()


@pytest.fixture
def trial_obj(sagemaker_session, experiment_obj):
    trial_obj = trial.Trial.create(
        trial_name=name(),
        experiment_name=experiment_obj.experiment_name,
        sagemaker_boto_client=sagemaker_session.sagemaker_client,
    )
    yield trial_obj
    time.sleep(0.5)
    trial_obj.delete()


@pytest.fixture
def experiment_obj(sagemaker_session):
    description = "{}-{}".format("description", str(uuid.uuid4()))
    boto3.set_stream_logger("", logging.INFO)
    experiment_name = name()
    experiment_obj = experiment.Experiment.create(
        experiment_name=experiment_name,
        description=description,
        sagemaker_boto_client=sagemaker_session.sagemaker_client,
    )
    yield experiment_obj
    time.sleep(0.5)
    experiment_obj.delete()


@pytest.fixture
def trial_associated_artifact(artifact_obj, trial_obj, trial_component_obj, sagemaker_session):
    assntn = association.Association.create(
        source_arn=artifact_obj.artifact_arn,
        destination_arn=trial_component_obj.trial_component_arn,
        association_type="ContributedTo",
        sagemaker_session=sagemaker_session,
    )
    trial_obj.add_trial_component(trial_component_obj)
    yield artifact_obj
    trial_obj.remove_trial_component(trial_component_obj)
    assntn.delete()


@pytest.fixture
def model_artifact_associated_endpoints(
    sagemaker_session, endpoint_deployment_action_obj, endpoint_context_obj
):

    model_artifact_obj = artifact.ModelArtifact.create(
        artifact_name="model-artifact-name",
        artifact_type="model-artifact-type",
        source_uri=name(),
        source_types=None,
        sagemaker_session=sagemaker_session,
    )

    association.Association.create(
        source_arn=model_artifact_obj.artifact_arn,
        destination_arn=endpoint_deployment_action_obj.action_arn,
        sagemaker_session=sagemaker_session,
    )

    association.Association.create(
        source_arn=endpoint_deployment_action_obj.action_arn,
        destination_arn=endpoint_context_obj.context_arn,
        sagemaker_session=sagemaker_session,
    )
    yield model_artifact_obj
    time.sleep(SLEEP_TIME_SECONDS)
    model_artifact_obj.delete(disassociate=True)


@pytest.fixture
def artifact_obj1(sagemaker_session):
    obj = artifact.Artifact.create(
        artifact_name="foo",
        artifact_type="Context",
        source_uri=name(),
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete(disassociate=True)


@pytest.fixture
def dataset_artifact_associated_models(sagemaker_session, trial_component_obj, model_artifact_obj1):
    dataset_artifact_obj = artifact.DatasetArtifact.create(
        artifact_name="dataset-artifact-name",
        artifact_type="Context",
        source_uri=name(),
        source_types=None,
        sagemaker_session=sagemaker_session,
    )

    association.Association.create(
        source_arn=dataset_artifact_obj.artifact_arn,
        destination_arn=trial_component_obj.trial_component_arn,
        sagemaker_session=sagemaker_session,
    )

    association_obj = association.Association.create(
        source_arn=trial_component_obj.trial_component_arn,
        destination_arn=model_artifact_obj1.artifact_arn,
        sagemaker_session=sagemaker_session,
    )
    yield dataset_artifact_obj
    time.sleep(SLEEP_TIME_SECONDS)
    dataset_artifact_obj.delete(disassociate=True)
    association_obj.delete


@pytest.fixture
def model_artifact_obj1(sagemaker_session):
    obj = artifact.Artifact.create(
        artifact_name="foo",
        artifact_type="Context",
        source_uri=name(),
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete(disassociate=True)


@pytest.fixture
def artifact_objs(sagemaker_session):
    artifact_objs = []
    for artifact_name in names():
        artifact_objs.append(
            artifact.Artifact.create(
                artifact_name=artifact_name,
                artifact_type="SDKIntegrationTest",
                source_uri=name(),
                properties={"k1": "v1"},
                sagemaker_session=sagemaker_session,
            )
        )
        time.sleep(SLEEP_TIME_SECONDS)

    artifact_objs.append(
        artifact.Artifact.create(
            artifact_name=name(),
            artifact_type="SDKIntegrationTestType2",
            source_uri=name(),
            properties={"k1": "v1"},
            sagemaker_session=sagemaker_session,
        )
    )

    yield artifact_objs

    for artifact_obj in artifact_objs:
        artifact_obj.delete()


@pytest.fixture
def context_obj(sagemaker_session):
    obj = context.Context.create(
        context_name=name(),
        source_uri="bar",
        source_type="test-source-type",
        context_type="test-context-type",
        description="test-description",
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete()


@pytest.fixture
def endpoint_context_obj(sagemaker_session):
    obj = context.Context.create(
        context_name=name(),
        source_uri="bar",
        source_type="Context",
        context_type="test-context-type",
        description="test-description",
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete(disassociate=True)


@pytest.fixture
def model_obj(sagemaker_session):
    model = context.Context.create(
        context_name=name(),
        source_uri="bar1",
        source_type="test-source-type1",
        context_type="Model",
        description="test-description",
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )

    yield model
    time.sleep(SLEEP_TIME_SECONDS)
    model.delete(disassociate=True)


@pytest.fixture
def context_obj_with_association(sagemaker_session, action_obj):
    obj = context.Context.create(
        context_name=name(),
        source_uri="bar",
        source_type="test-source-type",
        context_type="test-context-type",
        description="test-description",
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )
    association.Association.create(
        source_arn=obj.context_arn,
        destination_arn=action_obj.action_arn,
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete(disassociate=True)


@pytest.fixture
def endpoint_context_associate_with_model(sagemaker_session, endpoint_action_obj, model_obj):
    context_name = name()
    obj = context.EndpointContext.create(
        source_uri="endpontContextWithModel" + context_name,
        context_name=context_name,
        source_type="test-source-type",
        context_type="test-context-type",
        description="test-description",
        properties={"k1": "v1"},
        sagemaker_session=sagemaker_session,
    )

    association.Association.create(
        source_arn=obj.context_arn,
        destination_arn=endpoint_action_obj.action_arn,
        sagemaker_session=sagemaker_session,
    )

    association.Association.create(
        source_arn=endpoint_action_obj.action_arn,
        destination_arn=model_obj.context_arn,
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete(disassociate=True)


@pytest.fixture
def context_objs(sagemaker_session):
    context_objs = []
    for context_name in names():
        context_objs.append(
            context.Context.create(
                context_name=context_name,
                context_type="SDKIntegrationTest",
                source_uri="foo",
                properties={"k1": "v1"},
                sagemaker_session=sagemaker_session,
            )
        )
        time.sleep(SLEEP_TIME_SECONDS)

    yield context_objs
    for context_obj in context_objs:
        context_obj.delete()


@pytest.fixture
def association_obj(sagemaker_session, context_obj, action_obj):
    obj = association.Association.create(
        source_arn=context_obj.context_arn,
        destination_arn=action_obj.action_arn,
        association_type="ContributedTo",
        sagemaker_session=sagemaker_session,
    )
    yield obj
    time.sleep(SLEEP_TIME_SECONDS)
    obj.delete()


@pytest.fixture
def association_objs(sagemaker_session, context_obj, artifact_obj, association_obj):
    obj = association.Association.create(
        source_arn=context_obj.context_arn,
        destination_arn=artifact_obj.artifact_arn,
        association_type="ContributedTo",
        sagemaker_session=sagemaker_session,
    )
    yield [obj, association_obj]
    obj.delete()
