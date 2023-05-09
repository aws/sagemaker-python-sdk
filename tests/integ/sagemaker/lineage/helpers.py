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
"""This module contains helper methods for tests of SageMaker Lineage"""
from __future__ import absolute_import

import uuid
import time
from datetime import datetime


def name():
    return "lineage-integ-{}-{}".format(
        str(uuid.uuid4()), str(datetime.now().timestamp()).split(".")[0]
    )


def names():
    return [
        "lineage-integ-{}-{}".format(
            str(uuid.uuid4()), str(datetime.now().timestamp()).split(".")[0]
        )
        for i in range(3)
    ]


def traverse_graph_back(start_arn, sagemaker_session):
    def visit(arn, visited: set):
        visited.add(arn)
        associations = sagemaker_session.sagemaker_client.list_associations(DestinationArn=arn)[
            "AssociationSummaries"
        ]
        for association in associations:
            if association["SourceArn"] not in visited:
                ret.append(association)
                visit(association["SourceArn"], visited)

        return ret

    ret = []
    return visit(start_arn, set())


def traverse_graph_forward(start_arn, sagemaker_session):
    def visit(arn, visited: set):
        visited.add(arn)
        associations = sagemaker_session.sagemaker_client.list_associations(SourceArn=arn)[
            "AssociationSummaries"
        ]
        for association in associations:
            if association["DestinationArn"] not in visited:
                ret.append(association)
                visit(association["DestinationArn"], visited)

        return ret

    ret = []
    return visit(start_arn, set())


class LineageResourceHelper:
    def __init__(self, sagemaker_session):
        self.client = sagemaker_session.sagemaker_client
        self.artifacts = []
        self.actions = []
        self.contexts = []
        self.associations = []

    def create_artifact(self, artifact_name, artifact_type="Dataset"):
        response = self.client.create_artifact(
            ArtifactName=artifact_name,
            Source={
                "SourceUri": "Test-artifact-" + artifact_name,
                "SourceTypes": [
                    {"SourceIdType": "S3ETag", "Value": "Test-artifact-sourceId-value"},
                ],
            },
            ArtifactType=artifact_type,
        )
        self.artifacts.append(response["ArtifactArn"])

        return response["ArtifactArn"]

    def create_action(self, action_name, action_type="ModelDeployment"):
        response = self.client.create_action(
            ActionName=action_name,
            Source={
                "SourceUri": "Test-action-" + action_name,
                "SourceType": "S3ETag",
                "SourceId": "Test-action-sourceId-value",
            },
            ActionType=action_type,
        )
        self.actions.append(response["ActionArn"])

        return response["ActionArn"]

    def create_context(self, context_name, context_type="Endpoint"):
        response = self.client.create_context(
            ContextName=context_name,
            Source={
                "SourceUri": "Test-context-" + context_name,
                "SourceType": "S3ETag",
                "SourceId": "Test-context-sourceId-value",
            },
            ContextType=context_type,
        )
        self.contexts.append(response["ContextArn"])

        return response["ContextArn"]

    def create_association(self, source_arn, dest_arn, association_type="AssociatedWith"):
        response = self.client.add_association(
            SourceArn=source_arn, DestinationArn=dest_arn, AssociationType=association_type
        )
        if "SourceArn" in response.keys():
            self.associations.append((source_arn, dest_arn))
            return True
        else:
            return False

    def clean_all(self):
        # clean all lineage data created by LineageResourceHelper

        time.sleep(1)  # avoid GSI race condition between create & delete

        for source, dest in self.associations:
            try:
                self.client.delete_association(SourceArn=source, DestinationArn=dest)
            except Exception as e:
                print("skipped " + str(e))

        for artifact_arn in self.artifacts:
            try:
                self.client.delete_artifact(ArtifactArn=artifact_arn)
            except Exception as e:
                print("skipped " + str(e))

        for action_arn in self.actions:
            try:
                self.client.delete_action(ActionArn=action_arn)
            except Exception as e:
                print("skipped " + str(e))

        for context_arn in self.contexts:
            try:
                self.client.delete_context(ContextArn=context_arn)
            except Exception as e:
                print("skipped " + str(e))
