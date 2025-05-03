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
import os
import boto3
from botocore.config import Config


class Base:
    def __init__(self, session=None, region=None):
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        profile_name = os.getenv("AWS_PROFILE")

        if session is None:
            if all([aws_access_key_id, aws_secret_access_key, aws_session_token]):
                self.session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                )
            elif profile_name:
                self.session = boto3.Session(profile_name=profile_name)
            else:
                self.session = boto3.Session()

        self.region = region if region else os.getenv("AWS_REGION")

        # Create a custom config with the user agent
        custom_config = Config(region_name=self.region, user_agent_extra="SageMakerSDK/3.0")

        self.client = self.session.client("sagemaker", config=custom_config)
