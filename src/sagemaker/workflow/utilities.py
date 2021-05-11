# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Utilities to support workflow."""
from __future__ import absolute_import

from typing import List, Sequence, Union, Dict

from sagemaker.workflow.entities import (
    Entity,
    RequestType,
)
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.dataset_definition.inputs import (
    RedshiftDatasetDefinition,
    AthenaDatasetDefinition,
)
from uuid import uuid4


def list_to_request(entities: Sequence[Union[Entity, StepCollection]]) -> List[RequestType]:
    """Get the request structure for list of entities.

    Args:
        entities (Sequence[Entity]): A list of entities.
    Returns:
        list: A request structure for a workflow service call.
    """
    request_dicts = []
    for entity in entities:
        if isinstance(entity, Entity):
            request_dicts.append(entity.to_request())
        elif isinstance(entity, StepCollection):
            request_dicts.extend(entity.request_dicts())
    return request_dicts


def generate_data_ingestion_flow_recipe(
    input_name: str,
    s3_uri: str = None,
    s3_content_type: str = "csv",
    s3_has_header: bool = False,
    athena_dataset_definition: AthenaDatasetDefinition = None,
    redshift_dataset_definition: RedshiftDatasetDefinition = None,
) -> Dict:
    """Generate the data ingestion only flow recipe

    Args:
        input_name (str): s3 input to recipe source node
        s3_uri (str): s3 input uri
        s3_content_type (str): s3 input content type
        s3_has_header (bool): flag indicating the input has header or not
        athena_dataset_definition (AthenaDatasetDefinition): athena input to recipe source node
        redshift_dataset_definition (RedshiftDatasetDefinition): redshift input to recipe source node
    Returns:
        dict: A flow recipe only conduct data ingestion with 1-1 mapping
    """
    if s3_uri is None and athena_dataset_definition is None and redshift_dataset_definition is None:
        raise ValueError("One of s3 input, athena dataset definition, or redshift dataset definition need to be given.")

    recipe = {"metadata": {"version": 1, "disable_limits": False}, "nodes": []}

    source_node = {
        "node_id": str(uuid4()),
        "type": "SOURCE",
        "inputs": [],
        "outputs": [
            {
                "name": "default",
                "sampling": {"sampling_method": "sample_by_limit", "limit_rows": 50000},
            }
        ],
    }

    input_definition = None
    operator = None

    if s3_uri is not None:
        operator = "sagemaker.s3_source_0.1"
        input_definition = {
            "__typename": "S3CreateDatasetDefinitionOutput",
            "datasetSourceType": "S3",
            "name": input_name,
            "description": None,
            "s3ExecutionContext": {
                "__typename": "S3ExecutionContext",
                "s3Uri": s3_uri,
                "s3ContentType": s3_content_type,
                "s3HasHeader": s3_has_header,
            },
        }

    if input_definition is None and athena_dataset_definition is not None:
        operator = "sagemaker.athena_source_0.1"
        input_definition = {
            "datasetSourceType": "Athena",
            "name": input_name,
            "catalogName": athena_dataset_definition.catalog,
            "databaseName": athena_dataset_definition.database,
            "queryString": athena_dataset_definition.query_string,
            "s3OutputLocation": athena_dataset_definition.output_s3_uri,
            "outputFormat": athena_dataset_definition.output_format,
        }

    if input_definition is None and redshift_dataset_definition is not None:
        operator = "sagemaker.redshift_source_0.1"
        input_definition = {
            "datasetSourceType": "Redshift",
            "name": input_name,
            "clusterIdentifier": redshift_dataset_definition.cluster_id,
            "database": redshift_dataset_definition.database,
            "dbUser": redshift_dataset_definition.db_user,
            "queryString": redshift_dataset_definition.query_string,
            "unloadIamRole": redshift_dataset_definition.cluster_role_arn,
            "s3OutputLocation": redshift_dataset_definition.output_s3_uri,
            "outputFormat": redshift_dataset_definition.output_format,
        }

    source_node["operator"] = operator
    source_node["parameters"] = {"dataset_definition": input_definition}

    recipe["nodes"].append(source_node)

    type_infer_and_cast_node = {
        "node_id": str(uuid4()),
        "type": "TRANSFORM",
        "operator": "sagemaker.spark.infer_and_cast_type_0.1",
        "parameters": {},
        "inputs": [
            {"name": "default", "node_id": source_node["node_id"], "output_name": "default"}
        ],
        "outputs": [{"name": "default"}],
    }

    recipe["nodes"].append(type_infer_and_cast_node)

    return recipe
