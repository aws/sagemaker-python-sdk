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
import unittest.mock
from sagemaker.lineage.artifact import DatasetArtifact, ModelArtifact, Artifact
from sagemaker.lineage.context import EndpointContext, Context
from sagemaker.lineage.action import Action
from sagemaker.lineage.lineage_trial_component import LineageTrialComponent
from sagemaker.lineage.query import LineageEntityEnum, LineageSourceEnum, Vertex, LineageQuery
import pytest
import re


def test_lineage_query(sagemaker_session):
    lineage_query = LineageQuery(sagemaker_session)
    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": "arn1", "Type": "Endpoint", "LineageType": "Artifact"},
            {"Arn": "arn2", "Type": "Model", "LineageType": "Context"},
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }

    response = lineage_query.query(
        start_arns=["arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext"]
    )

    assert len(response.edges) == 1
    assert response.edges[0].source_arn == "arn1"
    assert response.edges[0].destination_arn == "arn2"
    assert response.edges[0].association_type == "Produced"
    assert len(response.vertices) == 2

    assert response.vertices[0].arn == "arn1"
    assert response.vertices[0].lineage_source == "Endpoint"
    assert response.vertices[0].lineage_entity == "Artifact"
    assert response.vertices[1].arn == "arn2"
    assert response.vertices[1].lineage_source == "Model"
    assert response.vertices[1].lineage_entity == "Context"


def test_lineage_query_duplication(sagemaker_session):
    lineage_query = LineageQuery(sagemaker_session)
    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": "arn1", "Type": "Endpoint", "LineageType": "Artifact"},
            {"Arn": "arn1", "Type": "Endpoint", "LineageType": "Artifact"},
            {"Arn": "arn2", "Type": "Model", "LineageType": "Context"},
        ],
        "Edges": [
            {"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"},
            {"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"},
        ],
    }

    response = lineage_query.query(
        start_arns=["arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext"]
    )

    assert len(response.edges) == 1
    assert response.edges[0].source_arn == "arn1"
    assert response.edges[0].destination_arn == "arn2"
    assert response.edges[0].association_type == "Produced"
    assert len(response.vertices) == 2
    assert response.vertices[0].arn == "arn1"
    assert response.vertices[0].lineage_source == "Endpoint"
    assert response.vertices[0].lineage_entity == "Artifact"
    assert response.vertices[1].arn == "arn2"
    assert response.vertices[1].lineage_source == "Model"
    assert response.vertices[1].lineage_entity == "Context"


def test_lineage_query_cross_account_same_artifact(sagemaker_session):
    lineage_query = LineageQuery(sagemaker_session)
    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {
                "Arn": "arn:aws:sagemaker:us-east-2:012345678901:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "Type": "Endpoint",
                "LineageType": "Artifact",
            },
            {
                "Arn": "arn:aws:sagemaker:us-east-2:012345678902:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "Type": "Endpoint",
                "LineageType": "Artifact",
            },
        ],
        "Edges": [
            {
                "SourceArn": "arn:aws:sagemaker:us-east-2:012345678901:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "DestinationArn": "arn:aws:sagemaker:us-east-2:012345678902:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "AssociationType": "SAME_AS",
            },
            {
                "SourceArn": "arn:aws:sagemaker:us-east-2:012345678902:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "DestinationArn": "arn:aws:sagemaker:us-east-2:012345678901:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "AssociationType": "SAME_AS",
            },
        ],
    }

    response = lineage_query.query(
        start_arns=["arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext"]
    )
    assert len(response.edges) == 0
    assert len(response.vertices) == 1
    assert (
        response.vertices[0].arn
        == "arn:aws:sagemaker:us-east-2:012345678901:artifact/e1f29799189751939405b0f2b5b9d2a0"
    )
    assert response.vertices[0].lineage_source == "Endpoint"
    assert response.vertices[0].lineage_entity == "Artifact"


def test_lineage_query_cross_account(sagemaker_session):
    lineage_query = LineageQuery(sagemaker_session)
    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {
                "Arn": "arn:aws:sagemaker:us-east-2:012345678901:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "Type": "Endpoint",
                "LineageType": "Artifact",
            },
            {
                "Arn": "arn:aws:sagemaker:us-east-2:012345678902:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "Type": "Endpoint",
                "LineageType": "Artifact",
            },
            {
                "Arn": "arn:aws:sagemaker:us-east-2:012345678903:artifact/e1f29799189751939405b0f2b5b9abcd",
                "Type": "Endpoint",
                "LineageType": "Artifact",
            },
            {
                "Arn": "arn:aws:sagemaker:us-east-2:012345678903:artifact/e1f29799189751939405b0f2b5b9efgh",
                "Type": "Endpoint",
                "LineageType": "Artifact",
            },
        ],
        "Edges": [
            {
                "SourceArn": "arn:aws:sagemaker:us-east-2:012345678901:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "DestinationArn": "arn:aws:sagemaker:us-east-2:012345678902:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "AssociationType": "SAME_AS",
            },
            {
                "SourceArn": "arn:aws:sagemaker:us-east-2:012345678902:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "DestinationArn": "arn:aws:sagemaker:us-east-2:012345678901:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "AssociationType": "SAME_AS",
            },
            {
                "SourceArn": "arn:aws:sagemaker:us-east-2:012345678902:artifact/e1f29799189751939405b0f2b5b9d2a0",
                "DestinationArn": "arn:aws:sagemaker:us-east-2:012345678903:artifact/e1f29799189751939405b0f2b5b9abcd",
                "AssociationType": "ABC",
            },
            {
                "SourceArn": "arn:aws:sagemaker:us-east-2:012345678903:artifact/e1f29799189751939405b0f2b5b9abcd",
                "DestinationArn": "arn:aws:sagemaker:us-east-2:012345678903:artifact/e1f29799189751939405b0f2b5b9efgh",
                "AssociationType": "DEF",
            },
        ],
    }

    response = lineage_query.query(
        start_arns=["arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext"]
    )

    assert len(response.edges) == 2
    assert (
        response.edges[0].source_arn
        == "arn:aws:sagemaker:us-east-2:012345678901:artifact/e1f29799189751939405b0f2b5b9d2a0"
    )
    assert (
        response.edges[0].destination_arn
        == "arn:aws:sagemaker:us-east-2:012345678903:artifact/e1f29799189751939405b0f2b5b9abcd"
    )
    assert response.edges[0].association_type == "ABC"

    assert (
        response.edges[1].source_arn
        == "arn:aws:sagemaker:us-east-2:012345678903:artifact/e1f29799189751939405b0f2b5b9abcd"
    )
    assert (
        response.edges[1].destination_arn
        == "arn:aws:sagemaker:us-east-2:012345678903:artifact/e1f29799189751939405b0f2b5b9efgh"
    )
    assert response.edges[1].association_type == "DEF"

    assert len(response.vertices) == 3
    assert (
        response.vertices[0].arn
        == "arn:aws:sagemaker:us-east-2:012345678901:artifact/e1f29799189751939405b0f2b5b9d2a0"
    )
    assert response.vertices[0].lineage_source == "Endpoint"
    assert response.vertices[0].lineage_entity == "Artifact"
    assert (
        response.vertices[1].arn
        == "arn:aws:sagemaker:us-east-2:012345678903:artifact/e1f29799189751939405b0f2b5b9abcd"
    )
    assert response.vertices[1].lineage_source == "Endpoint"
    assert response.vertices[1].lineage_entity == "Artifact"
    assert (
        response.vertices[2].arn
        == "arn:aws:sagemaker:us-east-2:012345678903:artifact/e1f29799189751939405b0f2b5b9efgh"
    )
    assert response.vertices[2].lineage_source == "Endpoint"
    assert response.vertices[2].lineage_entity == "Artifact"


def test_vertex_to_object_endpoint_context(sagemaker_session):
    vertex = Vertex(
        arn="arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext",
        lineage_entity=LineageEntityEnum.CONTEXT.value,
        lineage_source=LineageSourceEnum.ENDPOINT.value,
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.describe_context.return_value = {
        "ContextName": "MyContext",
        "ContextArn": "arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext",
        "Source": {
            "SourceUri": "arn:aws:sagemaker:us-west-2:0123456789012:endpoint/myendpoint",
            "SourceType": "ARN",
            "SourceId": "Thu Dec 17 17:16:24 UTC 2020",
        },
        "ContextType": "Endpoint",
        "Properties": {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:0123456789012:\
                pipeline/mypipeline/execution/0irnteql64d0",
            "PipelineStepName": "MyStep",
            "Status": "Completed",
        },
        "CreationTime": 1608225384.0,
        "CreatedBy": {},
        "LastModifiedTime": 1608225384.0,
        "LastModifiedBy": {},
    }

    context = vertex.to_lineage_object()

    assert context.context_arn == "arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext"
    assert context.context_name == "MyContext"
    assert isinstance(context, EndpointContext)


def test_vertex_to_object_context(sagemaker_session):
    vertex = Vertex(
        arn="arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext",
        lineage_entity=LineageEntityEnum.CONTEXT.value,
        lineage_source=LineageSourceEnum.MODEL_DEPLOYMENT.value,
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.describe_context.return_value = {
        "ContextName": "MyContext",
        "ContextArn": "arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext",
        "Source": {
            "SourceUri": "arn:aws:sagemaker:us-west-2:0123456789012:model/mymodel",
            "SourceType": "ARN",
            "SourceId": "Thu Dec 17 17:16:24 UTC 2020",
        },
        "ContextType": "ModelDeployment",
        "Properties": {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:0123456789012:\
                pipeline/mypipeline/execution/0irnteql64d0",
            "PipelineStepName": "MyStep",
            "Status": "Completed",
        },
        "CreationTime": 1608225384.0,
        "CreatedBy": {},
        "LastModifiedTime": 1608225384.0,
        "LastModifiedBy": {},
    }

    context = vertex.to_lineage_object()

    assert context.context_arn == "arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext"
    assert context.context_name == "MyContext"
    assert isinstance(context, Context)


def test_vertex_to_object_trial_component(sagemaker_session):

    tc_arn = "arn:aws:sagemaker:us-west-2:963951943925:trial-component/abaloneprocess-ixyt08z3ru-aws-processing-job"
    vertex = Vertex(
        arn=tc_arn,
        lineage_entity=LineageEntityEnum.TRIAL_COMPONENT.value,
        lineage_source=LineageSourceEnum.TRANSFORM_JOB.value,
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.describe_trial_component.return_value = {
        "TrialComponentName": "MyTrialComponent",
        "TrialComponentArn": tc_arn,
        "Source": {
            "SourceUri": "arn:aws:sagemaker:us-west-2:0123456789012:model/my_trial_component",
            "SourceType": "ARN",
            "SourceId": "Thu Dec 17 17:16:24 UTC 2020",
        },
        "TrialComponentType": "ModelDeployment",
        "Properties": {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:0123456789012:\
                pipeline/mypipeline/execution/0irnteql64d0",
            "PipelineStepName": "MyStep",
            "Status": "Completed",
        },
        "CreationTime": 1608225384.0,
        "CreatedBy": {},
        "LastModifiedTime": 1608225384.0,
        "LastModifiedBy": {},
    }

    trial_component = vertex.to_lineage_object()

    expected_calls = [
        unittest.mock.call(TrialComponentName="abaloneprocess-ixyt08z3ru-aws-processing-job"),
    ]
    assert expected_calls == sagemaker_session.sagemaker_client.describe_trial_component.mock_calls

    assert trial_component.trial_component_arn == tc_arn
    assert trial_component.trial_component_name == "MyTrialComponent"
    assert isinstance(trial_component, LineageTrialComponent)


def test_vertex_to_object_model_artifact(sagemaker_session):
    vertex = Vertex(
        arn="arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        lineage_entity=LineageEntityEnum.ARTIFACT.value,
        lineage_source=LineageSourceEnum.MODEL.value,
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "ArtifactArn": "arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        "Source": {
            "SourceUri": "arn:aws:sagemaker:us-west-2:0123456789012:model/mymodel",
            "SourceTypes": [],
        },
        "ArtifactType": "Model",
        "Properties": {},
        "CreationTime": 1608224704.149,
        "CreatedBy": {},
        "LastModifiedTime": 1608224704.149,
        "LastModifiedBy": {},
    }

    artifact = vertex.to_lineage_object()

    assert (
        artifact.artifact_arn
        == "arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f"
    )
    assert isinstance(artifact, ModelArtifact)


def test_vertex_to_object_artifact(sagemaker_session):
    vertex = Vertex(
        arn="arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        lineage_entity=LineageEntityEnum.ARTIFACT.value,
        lineage_source=LineageSourceEnum.MODEL.value,
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "ArtifactArn": "arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        "Source": {
            "SourceUri": "arn:aws:sagemaker:us-west-2:0123456789012:model/mymodel",
            "SourceTypes": [],
        },
        "ArtifactType": None,
        "Properties": {},
        "CreationTime": 1608224704.149,
        "CreatedBy": {},
        "LastModifiedTime": 1608224704.149,
        "LastModifiedBy": {},
    }

    artifact = vertex.to_lineage_object()

    assert (
        artifact.artifact_arn
        == "arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f"
    )
    assert isinstance(artifact, Artifact)


def test_vertex_to_dataset_artifact(sagemaker_session):
    vertex = Vertex(
        arn="arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        lineage_entity=LineageEntityEnum.ARTIFACT.value,
        lineage_source=LineageSourceEnum.DATASET.value,
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "ArtifactArn": "arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        "Source": {
            "SourceUri": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            "SourceTypes": [],
        },
        "ArtifactType": "Image",
        "Properties": {},
        "CreationTime": 1608224704.149,
        "CreatedBy": {},
        "LastModifiedTime": 1608224704.149,
        "LastModifiedBy": {},
    }

    artifact = vertex.to_lineage_object()

    assert (
        artifact.artifact_arn
        == "arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f"
    )
    assert isinstance(artifact, DatasetArtifact)


def test_vertex_to_model_artifact(sagemaker_session):
    vertex = Vertex(
        arn="arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        lineage_entity=LineageEntityEnum.ARTIFACT.value,
        lineage_source=LineageSourceEnum.MODEL.value,
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "ArtifactArn": "arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        "Source": {
            "SourceUri": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            "SourceTypes": [],
        },
        "ArtifactType": "Image",
        "Properties": {},
        "CreationTime": 1608224704.149,
        "CreatedBy": {},
        "LastModifiedTime": 1608224704.149,
        "LastModifiedBy": {},
    }

    artifact = vertex.to_lineage_object()

    assert (
        artifact.artifact_arn
        == "arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f"
    )
    assert isinstance(artifact, ModelArtifact)


def test_vertex_to_object_image_artifact(sagemaker_session):
    vertex = Vertex(
        arn="arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        lineage_entity=LineageEntityEnum.ARTIFACT.value,
        lineage_source=LineageSourceEnum.IMAGE.value,
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.describe_artifact.return_value = {
        "ArtifactArn": "arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        "Source": {
            "SourceUri": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            "SourceTypes": [],
        },
        "ArtifactType": "Image",
        "Properties": {},
        "CreationTime": 1608224704.149,
        "CreatedBy": {},
        "LastModifiedTime": 1608224704.149,
        "LastModifiedBy": {},
    }

    artifact = vertex.to_lineage_object()

    assert (
        artifact.artifact_arn
        == "arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f"
    )
    assert isinstance(artifact, Artifact)


def test_vertex_to_object_action(sagemaker_session):
    vertex = Vertex(
        arn="arn:aws:sagemaker:us-west-2:0123456789012:action/cp-m5-20210424t041405868z-1619237657-1-aws-endpoint",
        lineage_entity=LineageEntityEnum.ACTION.value,
        lineage_source="A",
        sagemaker_session=sagemaker_session,
    )

    sagemaker_session.sagemaker_client.describe_action.return_value = {
        "ActionName": "cp-m5-20210424t041405868z-1619237657-1-aws-endpoint",
        "Source": {
            "SourceUri": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            "SourceTypes": [],
        },
        "ActionType": "A",
        "Properties": {},
        "CreationTime": 1608224704.149,
        "CreatedBy": {},
        "LastModifiedTime": 1608224704.149,
        "LastModifiedBy": {},
    }

    action = vertex.to_lineage_object()

    assert action.action_name == "cp-m5-20210424t041405868z-1619237657-1-aws-endpoint"
    assert isinstance(action, Action)


def test_vertex_to_object_unconvertable(sagemaker_session):
    vertex = Vertex(
        arn="arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        lineage_entity=LineageEntityEnum.TRIAL.value,
        lineage_source=LineageSourceEnum.TENSORBOARD.value,
        sagemaker_session=sagemaker_session,
    )

    with pytest.raises(ValueError):
        vertex.to_lineage_object()


def test_get_visualization_elements(sagemaker_session):
    lineage_query = LineageQuery(sagemaker_session)
    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": "arn1", "Type": "Endpoint", "LineageType": "Artifact"},
            {"Arn": "arn2", "Type": "Model", "LineageType": "Context"},
            {
                "Arn": "arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext",
                "Type": "Model",
                "LineageType": "Context",
            },
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }

    query_response = lineage_query.query(
        start_arns=["arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext"]
    )

    elements = query_response._get_visualization_elements()

    assert elements["nodes"][0] == ("arn1", "Endpoint", "Artifact", False)
    assert elements["nodes"][1] == ("arn2", "Model", "Context", False)
    assert elements["nodes"][2] == (
        "arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext",
        "Model",
        "Context",
        True,
    )
    assert elements["edges"][0] == ("arn1", "arn2", "Produced")


def test_query_lineage_result_str(sagemaker_session):
    lineage_query = LineageQuery(sagemaker_session)
    sagemaker_session.sagemaker_client.query_lineage.return_value = {
        "Vertices": [
            {"Arn": "arn1", "Type": "Endpoint", "LineageType": "Artifact"},
            {"Arn": "arn2", "Type": "Model", "LineageType": "Context"},
        ],
        "Edges": [{"SourceArn": "arn1", "DestinationArn": "arn2", "AssociationType": "Produced"}],
    }

    query_response = lineage_query.query(
        start_arns=["arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext"]
    )

    response_str = query_response.__str__()
    pattern = r"Mock id='\d*'"
    replace = r"Mock id=''"
    response_str = re.sub(pattern, replace, response_str)

    assert (
        response_str
        == "{'edges': [\n\t{'source_arn': 'arn1', 'destination_arn': 'arn2', 'association_type': 'Produced'}],"
        + "\n\n'vertices': [\n\t{'arn': 'arn1', 'lineage_entity': 'Artifact', 'lineage_source': 'Endpoint', "
        + "'_session': <Mock id=''>}, \n\t{'arn': 'arn2', 'lineage_entity': 'Context', 'lineage_source': "
        + "'Model', '_session': <Mock id=''>}],\n\n'startarn': "
        + "['arn:aws:sagemaker:us-west-2:0123456789012:context/mycontext'],\n}"
    )
