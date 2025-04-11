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

import time
import boto3
import os
from botocore.config import Config

from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from tests.integ.sagemaker.jumpstart.utils import (
    get_test_artifact_bucket,
    get_sm_session,
)

from sagemaker.utils import repack_model
from tests.integ.sagemaker.jumpstart.constants import (
    ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID,
    JUMPSTART_TAG,
)


class InferenceJobLauncher:
    def __init__(
        self,
        image_uri,
        script_uri,
        model_uri,
        instance_type,
        environment_variables,
        suffix=time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()),
        region=JUMPSTART_DEFAULT_REGION_NAME,
        boto_config=Config(retries={"max_attempts": 10, "mode": "standard"}),
        base_name="jumpstart-inference-job",
        execution_role=None,
    ) -> None:

        self.suffix = suffix
        self.test_suite_id = os.environ[ENV_VAR_JUMPSTART_SDK_TEST_SUITE_ID]
        self.region = region
        self.config = boto_config
        self.base_name = base_name
        self.execution_role = execution_role or get_sm_session().get_caller_identity_arn()
        self.account_id = boto3.client("sts").get_caller_identity()["Account"]
        self.image_uri = image_uri
        self.script_uri = script_uri
        self.model_uri = model_uri
        self.instance_type = instance_type
        self.environment_variables = environment_variables
        self.sagemaker_client = self.get_sagemaker_client()

    def launch_inference_job(self):

        print("Packaging artifacts...")
        self.repacked_model_uri = self.package_artifacts()

        print("Creating model...")
        self.create_model()

        print("Creating endpoint config...")
        self.create_endpoint_config()

        print("Creating endpoint...")
        self.create_endpoint()

    def package_artifacts(self):

        self.model_name = self.get_model_name()

        if self.script_uri is None:
            print("No script uri provided. Not performing prepack")
            return self.model_uri

        cache_bucket_uri = f"s3://{get_test_artifact_bucket()}"
        repacked_model_uri = "/".join(
            [
                cache_bucket_uri,
                self.test_suite_id,
                "inference_model_tarballs",
                self.model_name,
                "repacked_model.tar.gz",
            ]
        )

        repack_model(
            inference_script="inference.py",
            source_directory=self.script_uri,
            dependencies=None,
            model_uri=self.model_uri,
            repacked_model_uri=repacked_model_uri,
            sagemaker_session=get_sm_session(),
            kms_key=None,
        )

        return repacked_model_uri

    def wait_until_endpoint_in_service(self):
        print("Waiting for endpoint to get in service...")
        self.sagemaker_client.get_waiter("endpoint_in_service").wait(
            EndpointName=self.endpoint_name
        )

    def get_sagemaker_client(self) -> boto3.client:
        return boto3.client(service_name="sagemaker", config=self.config, region_name=self.region)

    def get_endpoint_config_name(self) -> str:
        timestamp_length = len(self.suffix)
        non_timestamped_name = f"{self.base_name}-endpoint-config-"

        max_endpoint_config_name_length = 63

        if len(non_timestamped_name) > max_endpoint_config_name_length - timestamp_length:
            non_timestamped_name = non_timestamped_name[
                : max_endpoint_config_name_length - timestamp_length
            ]

        return f"{non_timestamped_name}{self.suffix}"

    def get_endpoint_name(self) -> str:
        timestamp_length = len(self.suffix)
        non_timestamped_name = f"{self.base_name}-endpoint-"

        max_endpoint_name_length = 63

        if len(non_timestamped_name) > max_endpoint_name_length - timestamp_length:
            non_timestamped_name = non_timestamped_name[
                : max_endpoint_name_length - timestamp_length
            ]

        return f"{non_timestamped_name}{self.suffix}"

    def get_model_name(self) -> str:
        timestamp_length = len(self.suffix)
        non_timestamped_name = f"{self.base_name}-model-"

        max_model_name_length = 63

        if len(non_timestamped_name) > max_model_name_length - timestamp_length:
            non_timestamped_name = non_timestamped_name[: max_model_name_length - timestamp_length]

        return f"{non_timestamped_name}{self.suffix}"

    def create_model(self) -> None:
        primary_container = {
            "Image": self.image_uri,
            "Mode": "SingleModel",
            "Environment": self.environment_variables,
        }
        if self.repacked_model_uri.endswith(".tar.gz"):
            primary_container["ModelDataUrl"] = self.repacked_model_uri
        else:
            primary_container["ModelDataSource"] = {
                "S3DataSource": {
                    "S3Uri": self.repacked_model_uri,
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            }
        self.sagemaker_client.create_model(
            ModelName=self.model_name,
            EnableNetworkIsolation=True,
            ExecutionRoleArn=self.execution_role,
            PrimaryContainer=primary_container,
        )

    def create_endpoint_config(self) -> None:
        self.endpoint_config_name = self.get_endpoint_config_name()
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=self.endpoint_config_name,
            ProductionVariants=[
                {
                    "InstanceType": self.instance_type,
                    "InitialInstanceCount": 1,
                    "ModelName": self.model_name,
                    "VariantName": "AllTraffic",
                }
            ],
        )

    def create_endpoint(self) -> None:
        self.endpoint_name = self.get_endpoint_name()
        self.sagemaker_client.create_endpoint(
            EndpointName=self.endpoint_name,
            EndpointConfigName=self.endpoint_config_name,
            Tags=[
                {
                    "Key": JUMPSTART_TAG,
                    "Value": self.test_suite_id,
                }
            ],
        )
