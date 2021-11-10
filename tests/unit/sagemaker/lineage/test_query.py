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
from sagemaker.lineage.artifact import DatasetArtifact, ModelArtifact, Artifact
from sagemaker.lineage.context import EndpointContext, Context
from sagemaker.lineage.query import LineageEntityEnum, LineageSourceEnum, Vertex, LineageQuery
import pytest


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


def test_vertex_to_object_artifact(sagemaker_session):
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


def test_vertex_to_object_unconvertable(sagemaker_session):
    vertex = Vertex(
        arn="arn:aws:sagemaker:us-west-2:0123456789012:artifact/e66eef7f19c05e75284089183491bd4f",
        lineage_entity=LineageEntityEnum.ACTION.value,
        lineage_source=LineageSourceEnum.TENSORBOARD.value,
        sagemaker_session=sagemaker_session,
    )

    with pytest.raises(ValueError):
        vertex.to_lineage_object()
