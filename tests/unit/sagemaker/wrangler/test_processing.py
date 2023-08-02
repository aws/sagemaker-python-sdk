#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License"). You
#  may not use this file except in compliance with the License. A copy of
#  the License is located at
#  #
#      http://aws.amazon.com/apache2.0/
#  #
#  or in the "license" file accompanying this file. This file is
#  distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
#  ANY KIND, either express or implied. See the License for the specific
#  language governing permissions and limitations under the License.
from __future__ import absolute_import

import pytest
from mock import Mock, MagicMock
from sagemaker.session_settings import SessionSettings

from sagemaker.wrangler.processing import DataWranglerProcessor
from sagemaker.processing import ProcessingInput

ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
REGION = "us-west-2"
DATA_WRANGLER_RECIPE_SOURCE = "s3://data_wrangler_flows/flow-26-18-43-16-0b48ac2e.flow"
DATA_WRANGLER_CONTAINER_URI = (
    "174368400705.dkr.ecr.us-west-2.amazonaws.com/sagemaker-data-wrangler-container:2.x"
)
MOCK_S3_URI = "s3://mock_data/mock.csv"


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    session_mock.expand_role.return_value = ROLE

    # For tests which doesn't verify config file injection, operate with empty config
    session_mock.sagemaker_config = {}
    return session_mock


def test_data_wrangler_processor_with_required_parameters(sagemaker_session):
    processor = DataWranglerProcessor(
        role=ROLE,
        data_wrangler_flow_source=DATA_WRANGLER_RECIPE_SOURCE,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    processor.run()
    expected_args = _get_expected_args(processor._current_job_name)
    sagemaker_session.process.assert_called_with(**expected_args)


def test_data_wrangler_processor_with_mock_input(sagemaker_session):
    processor = DataWranglerProcessor(
        role=ROLE,
        data_wrangler_flow_source=DATA_WRANGLER_RECIPE_SOURCE,
        instance_count=1,
        instance_type="ml.m4.xlarge",
        sagemaker_session=sagemaker_session,
    )

    mock_input = ProcessingInput(
        source=MOCK_S3_URI,
        destination="/opt/ml/processing/mock_input",
        input_name="mock_input",
        s3_data_type="S3Prefix",
        s3_input_mode="File",
        s3_data_distribution_type="FullyReplicated",
    )
    processor.run(inputs=[mock_input])
    expected_args = _get_expected_args(processor._current_job_name, add_mock_input=True)
    sagemaker_session.process.assert_called_with(**expected_args)


def _get_expected_args(job_name, add_mock_input=False):
    args = {
        "inputs": [
            {
                "InputName": "flow",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": DATA_WRANGLER_RECIPE_SOURCE,
                    "LocalPath": "/opt/ml/processing/flow",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            }
        ],
        "output_config": {"Outputs": []},
        "job_name": job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "stopping_condition": None,
        "app_specification": {
            "ImageUri": DATA_WRANGLER_CONTAINER_URI,
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
        "experiment_config": None,
    }

    if add_mock_input:
        mock_input = {
            "InputName": "mock_input",
            "AppManaged": False,
            "S3Input": {
                "S3Uri": MOCK_S3_URI,
                "LocalPath": "/opt/ml/processing/mock_input",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
                "S3CompressionType": "None",
            },
        }
        args["inputs"].insert(0, mock_input)
    return args
