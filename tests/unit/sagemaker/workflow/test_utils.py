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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
import shutil
import tempfile

import pytest
import sagemaker

from mock import (
    Mock,
    PropertyMock,
)

from sagemaker.dataset_definition.inputs import (
    RedshiftDatasetDefinition,
    AthenaDatasetDefinition,
)
from sagemaker.estimator import Estimator
from sagemaker.workflow._utils import _RepackModelStep
from sagemaker.workflow.utilities import generate_data_ingestion_flow_recipe
from tests.unit import DATA_DIR

REGION = "us-west-2"
BUCKET = "my-bucket"
IMAGE_URI = "fakeimage"
ROLE = "DummyRole"


@pytest.fixture
def boto_session():
    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value=ROLE)

    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock

    session_mock = Mock(region_name=REGION)
    session_mock.resource.return_value = resource_mock

    return session_mock


@pytest.fixture
def client():
    """Mock client.

    Considerations when appropriate:

        * utilize botocore.stub.Stubber
        * separate runtime client from client
    """
    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    return client_mock


@pytest.fixture
def sagemaker_session(boto_session, client):
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=client,
        sagemaker_runtime_client=client,
        default_bucket=BUCKET,
    )


@pytest.fixture
def estimator(sagemaker_session):
    return Estimator(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=1,
        instance_type="c4.4xlarge",
        sagemaker_session=sagemaker_session,
    )


@pytest.fixture
def source_dir(request):
    wf = os.path.join(DATA_DIR, "workflow")
    tmp = tempfile.mkdtemp()
    shutil.copy2(os.path.join(wf, "inference.py"), os.path.join(tmp, "inference.py"))
    shutil.copy2(os.path.join(wf, "foo"), os.path.join(tmp, "foo"))

    def fin():
        shutil.rmtree(tmp)

    request.addfinalizer(fin)

    return tmp


def test_repack_model_step(estimator):
    model_data = f"s3://{BUCKET}/model.tar.gz"
    entry_point = f"{DATA_DIR}/dummy_script.py"
    step = _RepackModelStep(
        name="MyRepackModelStep",
        estimator=estimator,
        model_data=model_data,
        entry_point=entry_point,
        depends_on=["TestStep"],
    )
    request_dict = step.to_request()

    hyperparameters = request_dict["Arguments"]["HyperParameters"]
    assert hyperparameters["inference_script"] == '"dummy_script.py"'
    assert hyperparameters["model_archive"] == '"model.tar.gz"'
    assert hyperparameters["sagemaker_program"] == '"_repack_model.py"'

    del request_dict["Arguments"]["HyperParameters"]
    del request_dict["Arguments"]["AlgorithmSpecification"]["TrainingImage"]
    assert request_dict == {
        "Name": "MyRepackModelStep",
        "Type": "Training",
        "DependsOn": ["TestStep"],
        "Arguments": {
            "AlgorithmSpecification": {"TrainingInputMode": "File"},
            "DebugHookConfig": {"CollectionConfigurations": [], "S3OutputPath": "s3://my-bucket/"},
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataDistributionType": "FullyReplicated",
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{BUCKET}",
                        }
                    },
                }
            ],
            "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/"},
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.large",
                "VolumeSizeInGB": 30,
            },
            "RoleArn": ROLE,
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        },
    }
    assert step.properties.TrainingJobName.expr == {
        "Get": "Steps.MyRepackModelStep.TrainingJobName"
    }


def test_repack_model_step_with_source_dir(estimator, source_dir):
    model_data = f"s3://{BUCKET}/model.tar.gz"
    entry_point = "inference.py"
    step = _RepackModelStep(
        name="MyRepackModelStep",
        estimator=estimator,
        model_data=model_data,
        entry_point=entry_point,
        source_dir=source_dir,
    )
    request_dict = step.to_request()
    assert os.path.isfile(f"{source_dir}/_repack_model.py")

    hyperparameters = request_dict["Arguments"]["HyperParameters"]
    assert hyperparameters["inference_script"] == '"inference.py"'
    assert hyperparameters["model_archive"] == '"model.tar.gz"'
    assert hyperparameters["sagemaker_program"] == '"_repack_model.py"'

    del request_dict["Arguments"]["HyperParameters"]
    del request_dict["Arguments"]["AlgorithmSpecification"]["TrainingImage"]
    assert request_dict == {
        "Name": "MyRepackModelStep",
        "Type": "Training",
        "Arguments": {
            "AlgorithmSpecification": {"TrainingInputMode": "File"},
            "DebugHookConfig": {"CollectionConfigurations": [], "S3OutputPath": "s3://my-bucket/"},
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataDistributionType": "FullyReplicated",
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{BUCKET}",
                        }
                    },
                }
            ],
            "OutputDataConfig": {"S3OutputPath": f"s3://{BUCKET}/"},
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.large",
                "VolumeSizeInGB": 30,
            },
            "RoleArn": ROLE,
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
        },
    }
    assert step.properties.TrainingJobName.expr == {
        "Get": "Steps.MyRepackModelStep.TrainingJobName"
    }


def test_generate_no_op_recipe_for_s3_input():
    recipe = generate_data_ingestion_flow_recipe(
        "test-s3", s3_uri="test_uri", s3_content_type="csv", s3_has_header=True
    )
    assert recipe["metadata"] == {"version": 1, "disable_limits": False}
    assert len(recipe["nodes"]) == 2
    # reset node id
    recipe["nodes"][0]["node_id"] = "123456"
    recipe["nodes"][1]["node_id"] = "7891011"
    recipe["nodes"][1]["inputs"][0]['node_id'] = "123456"
    assert recipe["nodes"] == [
        {
            "node_id": "123456",
            "type": "SOURCE",
            "operator": "sagemaker.s3_source_0.1",
            "parameters": {
                "dataset_definition": {
                    "__typename": "S3CreateDatasetDefinitionOutput",
                    "datasetSourceType": "S3",
                    "name": "test-s3",
                    "description": None,
                    "s3ExecutionContext": {
                        "__typename": "S3ExecutionContext",
                        "s3Uri": "test_uri",
                        "s3ContentType": "csv",
                        "s3HasHeader": True,
                    },
                }
            },
            "inputs": [],
            "outputs": [
                {
                    "name": "default",
                    "sampling": {"sampling_method": "sample_by_limit", "limit_rows": 50000},
                }
            ],
        },
        {
            "node_id": "7891011",
            "type": "TRANSFORM",
            "operator": "sagemaker.spark.infer_and_cast_type_0.1",
            "parameters": {},
            "inputs": [{"name": "default", "node_id": "123456", "output_name": "default"}],
            "outputs": [{"name": "default"}],
        },
    ]


def test_generate_no_op_recipe_for_athena_input():
    athena_df = AthenaDatasetDefinition(
            catalog="AwsDataCatalog",
            database="sagemaker_db",
            query_string='SELECT * FROM "table"',
            output_s3_uri="s3://my-bucket/athena/",
            output_format="parquet"
    )

    recipe = generate_data_ingestion_flow_recipe(
        "test-athena",
        athena_dataset_definition=athena_df
    )

    assert recipe["metadata"] == {"version": 1, "disable_limits": False}
    assert len(recipe["nodes"]) == 2
    # reset node id
    recipe["nodes"][0]["node_id"] = "123456"
    recipe["nodes"][1]["node_id"] = "7891011"
    recipe["nodes"][1]["inputs"][0]['node_id'] = "123456"
    assert recipe["nodes"] == [
        {
            "node_id": "123456",
            "type": "SOURCE",
            "operator": "sagemaker.athena_source_0.1",
            "parameters": {
                'dataset_definition': {
                    'datasetSourceType': 'Athena',
                    'name': 'test-athena',
                    'catalogName': athena_df.catalog,
                    'databaseName': athena_df.database,
                    'queryString': athena_df.query_string,
                    's3OutputLocation': athena_df.output_s3_uri,
                    'outputFormat': athena_df.output_format,
                }
            },
            "inputs": [],
            "outputs": [
                {
                    "name": "default",
                    "sampling": {"sampling_method": "sample_by_limit", "limit_rows": 50000},
                }
            ],
        },
        {
            "node_id": "7891011",
            "type": "TRANSFORM",
            "operator": "sagemaker.spark.infer_and_cast_type_0.1",
            "parameters": {},
            "inputs": [{"name": "default", "node_id": "123456", "output_name": "default"}],
            "outputs": [{"name": "default"}],
        }
    ]


def test_generate_no_op_recipe_for_redshift_input():
    redshift_df = RedshiftDatasetDefinition(
        cluster_id='cluster',
        database='db',
        db_user='db-user',
        query_string='query_string',
        cluster_role_arn='role',
        output_s3_uri='s3://my-bucket/redshift',
        output_format='parquet',
    )

    recipe = generate_data_ingestion_flow_recipe(
        "test-redshift",
        redshift_dataset_definition=redshift_df
    )

    assert recipe["metadata"] == {"version": 1, "disable_limits": False}
    assert len(recipe["nodes"]) == 2
    # reset node id
    recipe["nodes"][0]["node_id"] = "123456"
    recipe["nodes"][1]["node_id"] = "7891011"
    recipe["nodes"][1]["inputs"][0]['node_id'] = "123456"

    assert recipe["nodes"] == [
        {
            "node_id": "123456",
            "type": "SOURCE",
            "operator": "sagemaker.redshift_source_0.1",
            "parameters": {
                'dataset_definition': {
                    'datasetSourceType': 'Redshift',
                    'name': 'test-redshift',
                    'clusterIdentifier': redshift_df.cluster_id,
                    'database': redshift_df.database,
                    'dbUser': redshift_df.db_user,
                    'queryString': redshift_df.query_string,
                    'unloadIamRole': redshift_df.cluster_role_arn,
                    's3OutputLocation': redshift_df.output_s3_uri,
                    'outputFormat': redshift_df.output_format
                }
            },
            "inputs": [],
            "outputs": [
                {
                    "name": "default",
                    "sampling": {"sampling_method": "sample_by_limit", "limit_rows": 50000},
                }
            ],
        },
        {
            "node_id": "7891011",
            "type": "TRANSFORM",
            "operator": "sagemaker.spark.infer_and_cast_type_0.1",
            "parameters": {},
            "inputs": [{"name": "default", "node_id": "123456", "output_name": "default"}],
            "outputs": [{"name": "default"}],
        }
    ]


def test_generate_no_op_recipe_empty_input_source():
    with pytest.raises(ValueError):
        generate_data_ingestion_flow_recipe(input_name="source")


def test_generate_no_op_recipe_s3_input_ignore_athena_df():
    athena_df = AthenaDatasetDefinition(
        catalog="AwsDataCatalog",
        database="sagemaker_db",
        query_string='SELECT * FROM "table"',
        output_s3_uri="s3://my-bucket/athena/",
        output_format="parquet"
    )

    redshift_df = RedshiftDatasetDefinition(
        cluster_id='cluster',
        database='db',
        db_user='db-user',
        query_string='query_string',
        cluster_role_arn='role',
        output_s3_uri='s3://my-bucket/redshift',
        output_format='parquet',
    )

    recipe = generate_data_ingestion_flow_recipe(
        "test-s3",
        s3_uri="test_uri",
        s3_content_type="csv",
        s3_has_header=True,
        athena_dataset_definition=athena_df,
        redshift_dataset_definition=redshift_df
    )

    assert recipe["metadata"] == {"version": 1, "disable_limits": False}
    assert len(recipe["nodes"]) == 2
    # reset node id
    recipe["nodes"][0]["node_id"] = "123456"
    recipe["nodes"][1]["node_id"] = "7891011"
    recipe["nodes"][1]["inputs"][0]['node_id'] = "123456"
    assert recipe["nodes"] == [
        {
            "node_id": "123456",
            "type": "SOURCE",
            "operator": "sagemaker.s3_source_0.1",
            "parameters": {
                "dataset_definition": {
                    "__typename": "S3CreateDatasetDefinitionOutput",
                    "datasetSourceType": "S3",
                    "name": "test-s3",
                    "description": None,
                    "s3ExecutionContext": {
                        "__typename": "S3ExecutionContext",
                        "s3Uri": "test_uri",
                        "s3ContentType": "csv",
                        "s3HasHeader": True,
                    },
                }
            },
            "inputs": [],
            "outputs": [
                {
                    "name": "default",
                    "sampling": {"sampling_method": "sample_by_limit", "limit_rows": 50000},
                }
            ],
        },
        {
            "node_id": "7891011",
            "type": "TRANSFORM",
            "operator": "sagemaker.spark.infer_and_cast_type_0.1",
            "parameters": {},
            "inputs": [{"name": "default", "node_id": "123456", "output_name": "default"}],
            "outputs": [{"name": "default"}],
        },
    ]


def test_generate_no_op_recipe_athena_ignore_redshift_df():
    athena_df = AthenaDatasetDefinition(
        catalog="AwsDataCatalog",
        database="sagemaker_db",
        query_string='SELECT * FROM "table"',
        output_s3_uri="s3://my-bucket/athena/",
        output_format="parquet"
    )

    redshift_df = RedshiftDatasetDefinition(
        cluster_id='cluster',
        database='db',
        db_user='db-user',
        query_string='query_string',
        cluster_role_arn='role',
        output_s3_uri='s3://my-bucket/redshift',
        output_format='parquet',
    )

    recipe = generate_data_ingestion_flow_recipe(
        "test-athena",
        athena_dataset_definition=athena_df,
        redshift_dataset_definition=redshift_df,
    )

    assert recipe["metadata"] == {"version": 1, "disable_limits": False}
    assert len(recipe["nodes"]) == 2
    # reset node id
    recipe["nodes"][0]["node_id"] = "123456"
    recipe["nodes"][1]["node_id"] = "7891011"
    recipe["nodes"][1]["inputs"][0]['node_id'] = "123456"
    assert recipe["nodes"] == [
        {
            "node_id": "123456",
            "type": "SOURCE",
            "operator": "sagemaker.athena_source_0.1",
            "parameters": {
                'dataset_definition': {
                    'datasetSourceType': 'Athena',
                    'name': 'test-athena',
                    'catalogName': athena_df.catalog,
                    'databaseName': athena_df.database,
                    'queryString': athena_df.query_string,
                    's3OutputLocation': athena_df.output_s3_uri,
                    'outputFormat': athena_df.output_format,
                }
            },
            "inputs": [],
            "outputs": [
                {
                    "name": "default",
                    "sampling": {"sampling_method": "sample_by_limit", "limit_rows": 50000},
                }
            ],
        },
        {
            "node_id": "7891011",
            "type": "TRANSFORM",
            "operator": "sagemaker.spark.infer_and_cast_type_0.1",
            "parameters": {},
            "inputs": [{"name": "default", "node_id": "123456", "output_name": "default"}],
            "outputs": [{"name": "default"}],
        }
    ]
