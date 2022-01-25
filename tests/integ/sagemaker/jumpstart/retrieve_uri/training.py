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

import os
import time
import boto3
from botocore.config import Config

from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from tests.integ.sagemaker.jumpstart.utils import (
    get_test_artifact_bucket,
    get_sm_session,
)
from tests.integ.sagemaker.jumpstart.retrieve_uri.utils import (
    get_full_hyperparameters,
)
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket

from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
)


class TrainingJobLauncher:
    def __init__(
        self,
        image_uri,
        script_uri,
        model_uri,
        hyperparameters,
        instance_type,
        training_dataset_s3_key,
        suffix=time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()),
        region=JUMPSTART_DEFAULT_REGION_NAME,
        boto_config=Config(retries={"max_attempts": 10, "mode": "standard"}),
        base_name="jumpstart-training-job",
        execution_role=None,
    ) -> None:

        self.account_id = boto3.client("sts").get_caller_identity()["Account"]
        self.suffix = suffix
        self.test_suite_id = os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]
        self.region = region
        self.config = boto_config
        self.base_name = base_name
        self.execution_role = execution_role or get_sm_session().get_caller_identity_arn()
        self.image_uri = image_uri
        self.script_uri = script_uri
        self.model_uri = model_uri
        self.hyperparameters = hyperparameters
        self.instance_type = instance_type
        self.training_dataset_s3_key = training_dataset_s3_key
        self.sagemaker_client = self.get_sagemaker_client()

    def get_sagemaker_client(self) -> boto3.client:
        return boto3.client(service_name="sagemaker", config=self.config, region_name=self.region)

    def get_training_job_name(self) -> str:
        timestamp_length = len(self.suffix)
        non_timestamped_name = f"{self.base_name}-training-job-"

        if len(non_timestamped_name) > 63 - timestamp_length:
            non_timestamped_name = non_timestamped_name[: 63 - timestamp_length]

        return f"{non_timestamped_name}{self.suffix}"

    def wait_until_training_job_complete(self):
        print("Waiting for training job to complete...")
        self.sagemaker_client.get_waiter("training_job_completed_or_stopped").wait(
            TrainingJobName=self.training_job_name
        )

    def create_training_job(self) -> None:
        self.training_job_name = self.get_training_job_name()
        self.output_tarball_base_path = (
            f"s3://{get_test_artifact_bucket()}/{self.test_suite_id}/training_model_tarballs"
        )
        training_params = {
            "AlgorithmSpecification": {
                "TrainingImage": self.image_uri,
                "TrainingInputMode": "File",
            },
            "RoleArn": self.execution_role,
            "OutputDataConfig": {
                "S3OutputPath": self.output_tarball_base_path,
            },
            "ResourceConfig": {
                "InstanceCount": 1,
                "InstanceType": self.instance_type,
                "VolumeSizeInGB": 50,
            },
            "TrainingJobName": self.training_job_name,
            "EnableNetworkIsolation": True,
            "HyperParameters": get_full_hyperparameters(
                self.hyperparameters, self.training_job_name, self.model_uri
            ),
            "StoppingCondition": {"MaxRuntimeInSeconds": 86400},
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{get_jumpstart_content_bucket(self.region)}/{self.training_dataset_s3_key}",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "CompressionType": "None",
                },
                {
                    "ChannelName": "model",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": self.model_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "CompressionType": "None",
                },
                {
                    "ChannelName": "code",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": self.script_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "CompressionType": "None",
                },
            ],
        }
        print("Creating training job...")
        self.sagemaker_client.create_training_job(
            **training_params,
        )
