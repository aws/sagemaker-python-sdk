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
"""This module contains code to test SageMaker ``LineageQueryResult.visualize()``"""
from __future__ import absolute_import
import time
import json
import os

import pytest

import sagemaker.lineage.query
from sagemaker.lineage.query import LineageQueryDirectionEnum
from tests.integ.sagemaker.lineage.helpers import name, LineageResourceHelper


def test_LineageResourceHelper(sagemaker_session):
    # check if LineageResourceHelper works properly
    lineage_resource_helper = LineageResourceHelper(sagemaker_session=sagemaker_session)
    try:
        art1 = lineage_resource_helper.create_artifact(artifact_name=name())
        art2 = lineage_resource_helper.create_artifact(artifact_name=name())
        lineage_resource_helper.create_association(source_arn=art1, dest_arn=art2)
        lineage_resource_helper.clean_all()
    except Exception as e:
        print(e)
        assert False


@pytest.mark.skip("visualizer load test")
def test_wide_graph_visualize(sagemaker_session):
    lineage_resource_helper = LineageResourceHelper(sagemaker_session=sagemaker_session)
    wide_graph_root_arn = lineage_resource_helper.create_artifact(artifact_name=name())

    # create wide graph
    # Artifact ----> Artifact
    #        \ \ \-> Artifact
    #         \ \--> Artifact
    #          \--->  ...
    try:
        for i in range(10):
            artifact_arn = lineage_resource_helper.create_artifact(artifact_name=name())
            lineage_resource_helper.create_association(
                source_arn=wide_graph_root_arn, dest_arn=artifact_arn
            )
    except Exception as e:
        print(e)
        lineage_resource_helper.clean_all()
        assert False

    try:
        lq = sagemaker.lineage.query.LineageQuery(sagemaker_session)
        lq_result = lq.query(start_arns=[wide_graph_root_arn])
        lq_result.visualize(path="wideGraph.html")
    except Exception as e:
        print(e)
        lineage_resource_helper.clean_all()
        assert False

    lineage_resource_helper.clean_all()


@pytest.mark.skip("visualizer load test")
def test_long_graph_visualize(sagemaker_session):
    lineage_resource_helper = LineageResourceHelper(sagemaker_session=sagemaker_session)
    long_graph_root_arn = lineage_resource_helper.create_artifact(artifact_name=name())
    last_arn = long_graph_root_arn

    # create long graph
    # Artifact -> Artifact -> ... -> Artifact
    try:
        for i in range(10):
            new_artifact_arn = lineage_resource_helper.create_artifact(artifact_name=name())
            lineage_resource_helper.create_association(
                source_arn=last_arn, dest_arn=new_artifact_arn
            )
            last_arn = new_artifact_arn
    except Exception as e:
        print(e)
        lineage_resource_helper.clean_all()
        assert False

    try:
        lq = sagemaker.lineage.query.LineageQuery(sagemaker_session)
        lq_result = lq.query(
            start_arns=[long_graph_root_arn], direction=LineageQueryDirectionEnum.DESCENDANTS
        )
        # max depth = 10 -> graph rendered only has length of ten (in DESCENDANTS direction)
        lq_result.visualize(path="longGraph.html")
    except Exception as e:
        print(e)
        lineage_resource_helper.clean_all()
        assert False

    lineage_resource_helper.clean_all()


def test_graph_visualize(sagemaker_session):
    lineage_resource_helper = LineageResourceHelper(sagemaker_session=sagemaker_session)

    # create lineage data
    # image artifact ------> model artifact(startarn) -> model deploy action -> endpoint context
    #                    /->
    # dataset artifact -/
    try:
        graph_startarn = lineage_resource_helper.create_artifact(
            artifact_name=name(), artifact_type="Model"
        )
        image_artifact = lineage_resource_helper.create_artifact(
            artifact_name=name(), artifact_type="Image"
        )
        lineage_resource_helper.create_association(
            source_arn=image_artifact, dest_arn=graph_startarn, association_type="ContributedTo"
        )
        dataset_artifact = lineage_resource_helper.create_artifact(
            artifact_name=name(), artifact_type="DataSet"
        )
        lineage_resource_helper.create_association(
            source_arn=dataset_artifact, dest_arn=graph_startarn, association_type="AssociatedWith"
        )
        modeldeploy_action = lineage_resource_helper.create_action(
            action_name=name(), action_type="ModelDeploy"
        )
        lineage_resource_helper.create_association(
            source_arn=graph_startarn, dest_arn=modeldeploy_action, association_type="ContributedTo"
        )
        endpoint_context = lineage_resource_helper.create_context(
            context_name=name(), context_type="Endpoint"
        )
        lineage_resource_helper.create_association(
            source_arn=modeldeploy_action,
            dest_arn=endpoint_context,
            association_type="AssociatedWith",
        )
        time.sleep(1)
    except Exception as e:
        print(e)
        lineage_resource_helper.clean_all()
        assert False

    # visualize
    try:
        lq = sagemaker.lineage.query.LineageQuery(sagemaker_session)
        lq_result = lq.query(start_arns=[graph_startarn])
        lq_result.visualize(path="testGraph.html")
    except Exception as e:
        print(e)
        lineage_resource_helper.clean_all()
        assert False

    # check generated graph info
    try:
        fo = open("testGraph.html", "r")
        lines = fo.readlines()
        for line in lines:
            if "nodes = " in line:
                node = line
            if "edges = " in line:
                edge = line

        # extract node data
        start = node.find("[")
        end = node.find("]")
        res = node[start + 1 : end].split("}, ")
        res = [i + "}" for i in res]
        res[-1] = res[-1][:-1]
        node_dict = [json.loads(i) for i in res]

        # extract edge data
        start = edge.find("[")
        end = edge.find("]")
        res = edge[start + 1 : end].split("}, ")
        res = [i + "}" for i in res]
        res[-1] = res[-1][:-1]
        edge_dict = [json.loads(i) for i in res]

        # check node number
        assert len(node_dict) == 5

        # check startarn
        found_value = next(
            dictionary for dictionary in node_dict if dictionary["id"] == graph_startarn
        )
        assert found_value["color"] == "#146eb4"
        assert found_value["label"] == "Model"
        assert found_value["shape"] == "star"
        assert found_value["title"] == "Artifact"

        # check image artifact
        found_value = next(
            dictionary for dictionary in node_dict if dictionary["id"] == image_artifact
        )
        assert found_value["color"] == "#146eb4"
        assert found_value["label"] == "Image"
        assert found_value["shape"] == "dot"
        assert found_value["title"] == "Artifact"

        # check dataset artifact
        found_value = next(
            dictionary for dictionary in node_dict if dictionary["id"] == dataset_artifact
        )
        assert found_value["color"] == "#146eb4"
        assert found_value["label"] == "DataSet"
        assert found_value["shape"] == "dot"
        assert found_value["title"] == "Artifact"

        # check modeldeploy action
        found_value = next(
            dictionary for dictionary in node_dict if dictionary["id"] == modeldeploy_action
        )
        assert found_value["color"] == "#88c396"
        assert found_value["label"] == "ModelDeploy"
        assert found_value["shape"] == "dot"
        assert found_value["title"] == "Action"

        # check endpoint context
        found_value = next(
            dictionary for dictionary in node_dict if dictionary["id"] == endpoint_context
        )
        assert found_value["color"] == "#ff9900"
        assert found_value["label"] == "Endpoint"
        assert found_value["shape"] == "dot"
        assert found_value["title"] == "Context"

        # check edge number
        assert len(edge_dict) == 4

        # check image_artifact ->  model_artifact(startarn) edge
        found_value = next(
            dictionary for dictionary in edge_dict if dictionary["from"] == image_artifact
        )
        assert found_value["to"] == graph_startarn
        assert found_value["title"] == "ContributedTo"

        # check dataset_artifact ->  model_artifact(startarn) edge
        found_value = next(
            dictionary for dictionary in edge_dict if dictionary["from"] == dataset_artifact
        )
        assert found_value["to"] == graph_startarn
        assert found_value["title"] == "AssociatedWith"

        # check model_artifact(startarn) ->  modeldeploy_action edge
        found_value = next(
            dictionary for dictionary in edge_dict if dictionary["from"] == graph_startarn
        )
        assert found_value["to"] == modeldeploy_action
        assert found_value["title"] == "ContributedTo"

        # check modeldeploy_action ->  endpoint_context edge
        found_value = next(
            dictionary for dictionary in edge_dict if dictionary["from"] == modeldeploy_action
        )
        assert found_value["to"] == endpoint_context
        assert found_value["title"] == "AssociatedWith"

    except Exception as e:
        print(e)
        lineage_resource_helper.clean_all()
        os.remove("testGraph.html")
        assert False

    # delete generated test graph
    os.remove("testGraph.html")
    # clean lineage data
    lineage_resource_helper.clean_all()
