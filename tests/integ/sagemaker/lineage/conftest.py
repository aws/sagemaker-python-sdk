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
from sagemaker.model import ModelPackage
from tests.integ.sagemaker.workflow.test_workflow import (
    test_end_to_end_pipeline_successful_execution,
)
from sagemaker.workflow.pipeline import _PipelineExecution
from sagemaker.session import get_execution_role
from smexperiments import trial_component, trial, experiment
from random import randint
from botocore.exceptions import ClientError
from sagemaker.lineage.query import (
    LineageQuery,
    LineageFilter,
    LineageSourceEnum,
    LineageEntityEnum,
    LineageQueryDirectionEnum,
)
from sagemaker.lineage.lineage_trial_component import LineageTrialComponent

from tests.integ.sagemaker.lineage.helpers import name, names

SLEEP_TIME_SECONDS = 1
SLEEP_TIME_TWO_SECONDS = 2
STATIC_PIPELINE_NAME = "SdkIntegTestStaticPipeline20"
STATIC_ENDPOINT_NAME = "SdkIntegTestStaticEndpoint20"
STATIC_MODEL_PACKAGE_GROUP_NAME = "SdkIntegTestStaticPipeline20ModelPackageGroup"


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
    time.sleep(4)
    yield artifact_obj
    trial_obj.remove_trial_component(trial_component_obj)
    assntn.delete()


@pytest.fixture
def upstream_trial_associated_artifact(
    artifact_obj, trial_obj, trial_component_obj, sagemaker_session
):
    assntn = association.Association.create(
        source_arn=trial_component_obj.trial_component_arn,
        destination_arn=artifact_obj.artifact_arn,
        association_type="ContributedTo",
        sagemaker_session=sagemaker_session,
    )
    trial_obj.add_trial_component(trial_component_obj)
    time.sleep(4)
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
    model = artifact.Artifact.create(
        artifact_name=name(),
        artifact_type="Model",
        source_uri="bar1",
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
        destination_arn=model_obj.artifact_arn,
        sagemaker_session=sagemaker_session,
    )
    yield obj
    # sleep 2 seconds since take longer for lineage injection
    time.sleep(SLEEP_TIME_TWO_SECONDS)
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


@pytest.fixture(scope="module")
def static_pipeline_execution_arn(sagemaker_session):
    # Lineage query tests require several SageMaker resources
    # and their auto-created lineage entities. This method
    # looks for and returns a successful pipeline execution
    # for a static pipeline. If one doesn't exist, it starts
    # an execution and waits for it. This execution takes
    # approximately 25 minutes to run.
    try:
        sagemaker_session.sagemaker_client.describe_pipeline(PipelineName=STATIC_PIPELINE_NAME)
        return _get_static_pipeline_execution_arn(sagemaker_session)
    except sagemaker_session.sagemaker_client.exceptions.ResourceNotFound:
        print("Static pipeline execution not found. Starting one.")
        return create_and_execute_static_pipeline(sagemaker_session)


def _get_static_pipeline_execution_arn(sagemaker_session):
    pipeline_execution_arn = None
    while pipeline_execution_arn is None:
        time.sleep(randint(2, 5))
        pipeline_executions = sagemaker_session.sagemaker_client.list_pipeline_executions(
            PipelineName=STATIC_PIPELINE_NAME,
            SortBy="CreationTime",
            SortOrder="Ascending",
        )

        for pipeline_execution in pipeline_executions["PipelineExecutionSummaries"]:
            if pipeline_execution["PipelineExecutionStatus"] == "Succeeded":
                pipeline_execution_arn = pipeline_execution["PipelineExecutionArn"]
            elif pipeline_execution["PipelineExecutionStatus"] == "Executing":
                # wait on the execution to finish
                _PipelineExecution(
                    arn=pipeline_execution["PipelineExecutionArn"],
                    sagemaker_session=sagemaker_session,
                ).wait()
                pipeline_execution_arn = pipeline_execution["PipelineExecutionArn"]

        _deploy_static_endpoint(
            execution_arn=pipeline_execution_arn, sagemaker_session=sagemaker_session
        )

    return pipeline_execution_arn


@pytest.fixture
def static_approval_action(
    sagemaker_session, static_endpoint_context, static_pipeline_execution_arn
):
    query_filter = LineageFilter(
        entities=[LineageEntityEnum.ACTION], sources=[LineageSourceEnum.APPROVAL]
    )
    query_result = LineageQuery(sagemaker_session).query(
        start_arns=[static_endpoint_context.context_arn],
        query_filter=query_filter,
        direction=LineageQueryDirectionEnum.ASCENDANTS,
        include_edges=False,
    )
    action_name = query_result.vertices[0].arn.split("/")[1]
    yield action.ModelPackageApprovalAction.load(
        action_name=action_name, sagemaker_session=sagemaker_session
    )


@pytest.fixture
def static_model_deployment_action(sagemaker_session, static_processing_job_trial_component):
    query_filter = LineageFilter(
        entities=[LineageEntityEnum.ACTION], sources=[LineageSourceEnum.MODEL_DEPLOYMENT]
    )
    query_result = LineageQuery(sagemaker_session).query(
        start_arns=[static_processing_job_trial_component.trial_component_arn],
        query_filter=query_filter,
        direction=LineageQueryDirectionEnum.DESCENDANTS,
        include_edges=False,
    )
    model_approval_actions = []
    for vertex in query_result.vertices:
        model_approval_actions.append(vertex.to_lineage_object())
    yield model_approval_actions[0]


@pytest.fixture
def static_processing_job_trial_component(
    sagemaker_session, static_dataset_artifact
) -> LineageTrialComponent:
    query_filter = LineageFilter(
        entities=[LineageEntityEnum.TRIAL_COMPONENT], sources=[LineageSourceEnum.PROCESSING_JOB]
    )

    query_result = LineageQuery(sagemaker_session).query(
        start_arns=[static_dataset_artifact.artifact_arn],
        query_filter=query_filter,
        direction=LineageQueryDirectionEnum.ASCENDANTS,
        include_edges=False,
    )
    processing_jobs = []
    for vertex in query_result.vertices:
        processing_jobs.append(vertex.to_lineage_object())

    return processing_jobs[0]


@pytest.fixture
def static_training_job_trial_component(
    sagemaker_session, static_model_artifact
) -> LineageTrialComponent:
    query_filter = LineageFilter(
        entities=[LineageEntityEnum.TRIAL_COMPONENT], sources=[LineageSourceEnum.TRAINING_JOB]
    )

    query_result = LineageQuery(sagemaker_session).query(
        start_arns=[static_model_artifact.artifact_arn],
        query_filter=query_filter,
        direction=LineageQueryDirectionEnum.ASCENDANTS,
        include_edges=False,
    )
    training_jobs = []
    for vertex in query_result.vertices:
        training_jobs.append(vertex.to_lineage_object())

    return training_jobs[0]


@pytest.fixture
def static_transform_job_trial_component(
    static_processing_job_trial_component, sagemaker_session, static_endpoint_context
) -> LineageTrialComponent:
    query_filter = LineageFilter(
        entities=[LineageEntityEnum.TRIAL_COMPONENT], sources=[LineageSourceEnum.TRANSFORM_JOB]
    )
    query_result = LineageQuery(sagemaker_session).query(
        start_arns=[static_processing_job_trial_component.trial_component_arn],
        query_filter=query_filter,
        direction=LineageQueryDirectionEnum.DESCENDANTS,
        include_edges=False,
    )
    transform_jobs = []
    for vertex in query_result.vertices:
        transform_jobs.append(vertex.to_lineage_object())
    yield transform_jobs[0]


@pytest.fixture
def static_endpoint_context(sagemaker_session, static_pipeline_execution_arn):
    endpoint_arn = get_endpoint_arn_from_static_pipeline(sagemaker_session)

    if endpoint_arn is None:
        _deploy_static_endpoint(
            execution_arn=static_pipeline_execution_arn,
            sagemaker_session=sagemaker_session,
        )
        endpoint_arn = get_endpoint_arn_from_static_pipeline(sagemaker_session)

    contexts = sagemaker_session.sagemaker_client.list_contexts(SourceUri=endpoint_arn)[
        "ContextSummaries"
    ]
    if len(contexts) != 1:
        raise (
            Exception(
                f"Got an unexpected number of Contexts for \
                endpoint {STATIC_ENDPOINT_NAME} from pipeline \
                execution {static_pipeline_execution_arn}. \
                Expected 1 but got {len(contexts)}"
            )
        )

    yield context.EndpointContext.load(
        contexts[0]["ContextName"], sagemaker_session=sagemaker_session
    )


@pytest.fixture
def static_model_package_group_context(sagemaker_session, static_pipeline_execution_arn):

    model_package_group_arn = get_model_package_group_arn_from_static_pipeline(sagemaker_session)

    contexts = sagemaker_session.sagemaker_client.list_contexts(SourceUri=model_package_group_arn)[
        "ContextSummaries"
    ]
    if len(contexts) != 1:
        raise (
            Exception(
                f"Got an unexpected number of Contexts for \
                model package group {STATIC_MODEL_PACKAGE_GROUP_NAME} from pipeline \
                execution {static_pipeline_execution_arn}. \
                Expected 1 but got {len(contexts)}"
            )
        )

    yield context.ModelPackageGroup.load(
        contexts[0]["ContextName"], sagemaker_session=sagemaker_session
    )


@pytest.fixture
def static_model_artifact(sagemaker_session, static_pipeline_execution_arn):
    model_package_arn = get_model_package_arn_from_static_pipeline(
        static_pipeline_execution_arn, sagemaker_session
    )

    artifacts = sagemaker_session.sagemaker_client.list_artifacts(SourceUri=model_package_arn)[
        "ArtifactSummaries"
    ]
    if len(artifacts) != 1:
        raise (
            Exception(
                f"Got an unexpected number of Artifacts for \
                    model package {model_package_arn}. Expected 1 but got {len(artifacts)}"
            )
        )

    yield artifact.ModelArtifact.load(
        artifacts[0]["ArtifactArn"], sagemaker_session=sagemaker_session
    )


@pytest.fixture
def static_dataset_artifact(static_model_artifact, sagemaker_session):
    dataset_associations = sagemaker_session.sagemaker_client.list_associations(
        DestinationArn=static_model_artifact.artifact_arn, SourceType="DataSet"
    )
    if len(dataset_associations["AssociationSummaries"]) == 0:
        # no directly associated dataset. work backwards from the model
        model_associations = sagemaker_session.sagemaker_client.list_associations(
            DestinationArn=static_model_artifact.artifact_arn, SourceType="Model"
        )
        training_job_associations = sagemaker_session.sagemaker_client.list_associations(
            DestinationArn=model_associations["AssociationSummaries"][0]["SourceArn"],
            SourceType="SageMakerTrainingJob",
        )
        dataset_associations = sagemaker_session.sagemaker_client.list_associations(
            DestinationArn=training_job_associations["AssociationSummaries"][0]["SourceArn"],
            SourceType="DataSet",
        )

    yield artifact.DatasetArtifact.load(
        dataset_associations["AssociationSummaries"][0]["SourceArn"],
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture
def static_image_artifact(static_dataset_artifact, sagemaker_session):
    query_filter = LineageFilter(
        entities=[LineageEntityEnum.ARTIFACT], sources=[LineageSourceEnum.IMAGE]
    )
    query_result = LineageQuery(sagemaker_session).query(
        start_arns=[static_dataset_artifact.artifact_arn],
        query_filter=query_filter,
        direction=LineageQueryDirectionEnum.ASCENDANTS,
        include_edges=False,
    )
    image_artifact = []
    for vertex in query_result.vertices:
        image_artifact.append(vertex.to_lineage_object())
    return image_artifact[0]


def get_endpoint_arn_from_static_pipeline(sagemaker_session):
    try:
        endpoint_arn = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=STATIC_ENDPOINT_NAME
        )["EndpointArn"]

        return endpoint_arn
    except ClientError as e:
        error = e.response["Error"]
        if error["Code"] == "ValidationException":
            return None
        raise e


def get_model_package_group_arn_from_static_pipeline(sagemaker_session):
    static_model_package_group_arn = (
        sagemaker_session.sagemaker_client.describe_model_package_group(
            ModelPackageGroupName=STATIC_MODEL_PACKAGE_GROUP_NAME
        )["ModelPackageGroupArn"]
    )
    return static_model_package_group_arn


def get_model_package_arn_from_static_pipeline(pipeline_execution_arn, sagemaker_session):
    # get the model package ARN from the pipeline
    pipeline_execution_steps = sagemaker_session.sagemaker_client.list_pipeline_execution_steps(
        PipelineExecutionArn=pipeline_execution_arn
    )["PipelineExecutionSteps"]

    model_package_arn = None
    for step in pipeline_execution_steps:
        if "RegisterModel" in step["Metadata"]:
            model_package_arn = step["Metadata"]["RegisterModel"]["Arn"]

    if model_package_arn is None:
        raise (
            Exception(
                f"Did not find a model package ARN in static pipeline execution {pipeline_execution_arn}"
            )
        )

    return model_package_arn


def create_and_execute_static_pipeline(sagemaker_session):
    # start the execution and wait for success
    print(f"Starting static execution of pipeline '{STATIC_PIPELINE_NAME}'")
    try:
        execution_arn = test_end_to_end_pipeline_successful_execution(
            sagemaker_session=sagemaker_session,
            region_name=sagemaker_session.boto_session.region_name,
            role=get_execution_role(sagemaker_session),
            pipeline_name=STATIC_PIPELINE_NAME,
            wait=True,
        )

        # now deploy the model package to an endpoint
        _deploy_static_endpoint(
            execution_arn=execution_arn,
            sagemaker_session=sagemaker_session,
        )

        return execution_arn
    except Exception:
        # Pipeline already exists, meaning an execution was started by
        # tests in a different thread
        execution_arn = _get_static_pipeline_execution_arn(sagemaker_session)
        _deploy_static_endpoint(
            execution_arn=execution_arn,
            sagemaker_session=sagemaker_session,
        )
        return execution_arn


def _deploy_static_endpoint(execution_arn, sagemaker_session):
    try:
        model_package_arn = get_model_package_arn_from_static_pipeline(
            execution_arn, sagemaker_session
        )

        model_package = ModelPackage(
            role=get_execution_role(sagemaker_session),
            model_package_arn=model_package_arn,
            sagemaker_session=sagemaker_session,
        )
        model_package.deploy(1, "ml.t2.medium", endpoint_name=STATIC_ENDPOINT_NAME)
        time.sleep(120)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Endpoint {STATIC_ENDPOINT_NAME} already exists. Continuing.")
            pass
        else:
            raise (e)
