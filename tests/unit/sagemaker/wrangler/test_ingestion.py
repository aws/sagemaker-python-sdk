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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from sagemaker.wrangler.ingestion import (
    generate_data_ingestion_flow_from_s3_input,
    generate_data_ingestion_flow_from_athena_dataset_definition,
    generate_data_ingestion_flow_from_redshift_dataset_definition,
)
from sagemaker.dataset_definition.inputs import (
    RedshiftDatasetDefinition,
    AthenaDatasetDefinition,
)


def test_generate_ingestion_flow_for_s3_input():
    flow, output_name = generate_data_ingestion_flow_from_s3_input(
        "test-s3",
        s3_uri="test_uri",
        s3_content_type="csv",
        s3_has_header=True,
        operator_version="0.2",
        schema={"k1": "string", "k2": "long"},
    )
    assert flow["metadata"] == {"version": 1, "disable_limits": False}
    assert len(flow["nodes"]) == 2
    assert output_name.endswith(flow["nodes"][1]["outputs"][0]["name"])

    # reset node id
    flow["nodes"][0]["node_id"] = "123456"
    flow["nodes"][1]["node_id"] = "7891011"
    flow["nodes"][1]["inputs"][0]["node_id"] = "123456"
    assert flow["nodes"] == [
        {
            "node_id": "123456",
            "type": "SOURCE",
            "operator": "sagemaker.s3_source_0.2",
            "parameters": {
                "dataset_definition": {
                    "datasetSourceType": "S3",
                    "name": "test-s3",
                    "s3ExecutionContext": {
                        "s3Uri": "test_uri",
                        "s3ContentType": "csv",
                        "s3HasHeader": True,
                    },
                }
            },
            "inputs": [],
            "outputs": [{"name": "default"}],
        },
        {
            "node_id": "7891011",
            "type": "TRANSFORM",
            "operator": "sagemaker.spark.infer_and_cast_type_0.2",
            "trained_parameters": {"k1": "string", "k2": "long"},
            "parameters": {},
            "inputs": [{"name": "default", "node_id": "123456", "output_name": "default"}],
            "outputs": [{"name": "default"}],
        },
    ]


def test_generate_ingestion_flow_for_athena_input():
    athena_df = AthenaDatasetDefinition(
        catalog="AwsDataCatalog",
        database="sagemaker_db",
        query_string='SELECT * FROM "table"',
        output_s3_uri="s3://my-bucket/athena/",
        output_format="parquet",
    )

    flow, output_name = generate_data_ingestion_flow_from_athena_dataset_definition(
        "test-athena",
        athena_dataset_definition=athena_df,
        operator_version="0.2",
        schema={"k1": "string", "k2": "long"},
    )

    assert flow["metadata"] == {"version": 1, "disable_limits": False}
    assert len(flow["nodes"]) == 2
    assert output_name.endswith(flow["nodes"][1]["outputs"][0]["name"])
    # reset node id
    flow["nodes"][0]["node_id"] = "123456"
    flow["nodes"][1]["node_id"] = "7891011"
    flow["nodes"][1]["inputs"][0]["node_id"] = "123456"
    assert flow["nodes"] == [
        {
            "node_id": "123456",
            "type": "SOURCE",
            "operator": "sagemaker.athena_source_0.2",
            "parameters": {
                "dataset_definition": {
                    "datasetSourceType": "Athena",
                    "name": "test-athena",
                    "catalogName": athena_df.catalog,
                    "databaseName": athena_df.database,
                    "queryString": athena_df.query_string,
                    "s3OutputLocation": athena_df.output_s3_uri,
                    "outputFormat": athena_df.output_format,
                }
            },
            "inputs": [],
            "outputs": [{"name": "default"}],
        },
        {
            "node_id": "7891011",
            "type": "TRANSFORM",
            "operator": "sagemaker.spark.infer_and_cast_type_0.2",
            "trained_parameters": {"k1": "string", "k2": "long"},
            "parameters": {},
            "inputs": [{"name": "default", "node_id": "123456", "output_name": "default"}],
            "outputs": [{"name": "default"}],
        },
    ]


def test_generate_ingestion_flow_for_redshift_input():
    redshift_df = RedshiftDatasetDefinition(
        cluster_id="cluster",
        database="db",
        db_user="db-user",
        query_string="query_string",
        cluster_role_arn="role",
        output_s3_uri="s3://my-bucket/redshift",
        output_format="parquet",
    )

    flow, output_name = generate_data_ingestion_flow_from_redshift_dataset_definition(
        "test-redshift",
        redshift_dataset_definition=redshift_df,
        operator_version="0.2",
        schema={"k1": "string", "k2": "long"},
    )

    assert flow["metadata"] == {"version": 1, "disable_limits": False}
    assert len(flow["nodes"]) == 2
    assert output_name.endswith(flow["nodes"][1]["outputs"][0]["name"])
    # reset node id
    flow["nodes"][0]["node_id"] = "123456"
    flow["nodes"][1]["node_id"] = "7891011"
    flow["nodes"][1]["inputs"][0]["node_id"] = "123456"

    assert flow["nodes"] == [
        {
            "node_id": "123456",
            "type": "SOURCE",
            "operator": "sagemaker.redshift_source_0.2",
            "parameters": {
                "dataset_definition": {
                    "datasetSourceType": "Redshift",
                    "name": "test-redshift",
                    "clusterIdentifier": redshift_df.cluster_id,
                    "database": redshift_df.database,
                    "dbUser": redshift_df.db_user,
                    "queryString": redshift_df.query_string,
                    "unloadIamRole": redshift_df.cluster_role_arn,
                    "s3OutputLocation": redshift_df.output_s3_uri,
                    "outputFormat": redshift_df.output_format,
                }
            },
            "inputs": [],
            "outputs": [{"name": "default"}],
        },
        {
            "node_id": "7891011",
            "type": "TRANSFORM",
            "operator": "sagemaker.spark.infer_and_cast_type_0.2",
            "trained_parameters": {"k1": "string", "k2": "long"},
            "parameters": {},
            "inputs": [{"name": "default", "node_id": "123456", "output_name": "default"}],
            "outputs": [{"name": "default"}],
        },
    ]
