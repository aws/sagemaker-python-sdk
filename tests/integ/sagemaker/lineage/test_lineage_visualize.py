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
import os
import re

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
    except Exception as e:
        print(e)
        assert False
    finally:
        lineage_resource_helper.clean_all()


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
        for i in range(150):
            artifact_arn = lineage_resource_helper.create_artifact(artifact_name=name())
            lineage_resource_helper.create_association(
                source_arn=wide_graph_root_arn, dest_arn=artifact_arn
            )

        lq = sagemaker.lineage.query.LineageQuery(sagemaker_session)
        lq_result = lq.query(start_arns=[wide_graph_root_arn])
        lq_result.visualize(path="wideGraph.html")

    except Exception as e:
        print(e)
        assert False

    finally:
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

        lq = sagemaker.lineage.query.LineageQuery(sagemaker_session)
        lq_result = lq.query(
            start_arns=[long_graph_root_arn], direction=LineageQueryDirectionEnum.DESCENDANTS
        )
        # max depth = 10 -> graph rendered only has length of ten (in DESCENDANTS direction)
        lq_result.visualize(path="longGraph.html")

    except Exception as e:
        print(e)
        assert False

    finally:
        lineage_resource_helper.clean_all()


def test_graph_visualize(sagemaker_session, extract_data_from_html):
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
        time.sleep(3)

        # visualize
        lq = sagemaker.lineage.query.LineageQuery(sagemaker_session)
        lq_result = lq.query(start_arns=[graph_startarn])
        lq_result.visualize(path="testGraph.html")

        # check generated graph info
        with open("testGraph.html", "r") as fo:
            lines = fo.readlines()
        for line in lines:
            if "nodes = " in line:
                node = line
            if "edges = " in line:
                edge = line

        node_dict = extract_data_from_html(node)
        edge_dict = extract_data_from_html(edge)

        # check node number
        assert len(node_dict) == 5

        expected_nodes = {
            graph_startarn: {
                "color": "#146eb4",
                "label": "Model",
                "shape": "star",
                "title": "Entity: Artifact"
                + "\nType: Model"
                + "\nAccount ID: "
                + str(re.search(r":\d{12}:", graph_startarn).group()[1:-1])
                + "\nName: "
                + str(re.search(r"\/.*", graph_startarn).group()[1:]),
            },
            image_artifact: {
                "color": "#146eb4",
                "label": "Image",
                "shape": "dot",
                "title": "Entity: Artifact"
                + "\nType: Image"
                + "\nAccount ID: "
                + str(re.search(r":\d{12}:", image_artifact).group()[1:-1])
                + "\nName: "
                + str(re.search(r"\/.*", image_artifact).group()[1:]),
            },
            dataset_artifact: {
                "color": "#146eb4",
                "label": "Data Set",
                "shape": "dot",
                "title": "Entity: Artifact"
                + "\nType: Data Set"
                + "\nAccount ID: "
                + str(re.search(r":\d{12}:", dataset_artifact).group()[1:-1])
                + "\nName: "
                + str(re.search(r"\/.*", dataset_artifact).group()[1:]),
            },
            modeldeploy_action: {
                "color": "#88c396",
                "label": "Model Deploy",
                "shape": "dot",
                "title": "Entity: Action"
                + "\nType: Model Deploy"
                + "\nAccount ID: "
                + str(re.search(r":\d{12}:", modeldeploy_action).group()[1:-1])
                + "\nName: "
                + str(re.search(r"\/.*", modeldeploy_action).group()[1:]),
            },
            endpoint_context: {
                "color": "#ff9900",
                "label": "Endpoint",
                "shape": "dot",
                "title": "Entity: Context"
                + "\nType: Endpoint"
                + "\nAccount ID: "
                + str(re.search(r":\d{12}:", endpoint_context).group()[1:-1])
                + "\nName: "
                + str(re.search(r"\/.*", endpoint_context).group()[1:]),
            },
        }

        # check node properties
        for node in node_dict:
            for label, val in expected_nodes[node["id"]].items():
                assert node[label] == val

        # check edge number
        assert len(edge_dict) == 4

        expected_edges = {
            image_artifact: {
                "from": image_artifact,
                "to": graph_startarn,
                "title": "ContributedTo",
            },
            dataset_artifact: {
                "from": dataset_artifact,
                "to": graph_startarn,
                "title": "AssociatedWith",
            },
            graph_startarn: {
                "from": graph_startarn,
                "to": modeldeploy_action,
                "title": "ContributedTo",
            },
            modeldeploy_action: {
                "from": modeldeploy_action,
                "to": endpoint_context,
                "title": "AssociatedWith",
            },
        }

        # check edge properties
        for edge in edge_dict:
            for label, val in expected_edges[edge["from"]].items():
                assert edge[label] == val

    except Exception as e:
        print(e)
        assert False

    finally:
        lineage_resource_helper.clean_all()
        os.remove("testGraph.html")
