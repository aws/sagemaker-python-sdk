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

"""CustomAgentLambda — Lambda-based agent environment for Agentic RFT."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Optional

import boto3

S3_URI_PATTERN = re.compile(r"^s3://[^/]+(/.*)?$")


class CustomAgentLambda:
    """Lambda-based agent environment for Agentic RFT.

    Creates and wraps Lambda functions that serve as agent environments or
    bridges between SageMaker and custom agent environments (e.g., LangSmith,
    EKS, Fargate).

    Args:
        lambda_arn: ARN of the Lambda function.
    """

    def __init__(self, lambda_arn: str):
        self.lambda_arn = lambda_arn

    def __repr__(self):
        return f"CustomAgentLambda(lambda_arn={self.lambda_arn!r})"

    @classmethod
    def create(
        cls,
        source: str,
        function_name: Optional[str] = None,
        role: Optional[str] = None,
        runtime: str = "python3.12",
        handler: str = "lambda_function.handler",
        timeout: int = 900,
        memory_size: int = 256,
        environment: Optional[dict] = None,
        sagemaker_session=None,
    ) -> CustomAgentLambda:
        """Create a new Lambda function and return an CustomAgentLambda.

        The ``source`` parameter accepts three formats:

        - **S3 URI** (``s3://bucket/key.zip``): deploys from an S3 artifact.
        - **Local file path**: reads the file, packages it as a zip, and uploads.
        - **Inline code string**: packages the raw code as a zip and uploads.

        Detection order: S3 URI → existing local path → inline code.

        Args:
            source: S3 URI, local file path, or inline Python code string.
            function_name: Lambda function name. If not provided, a unique name
                is generated automatically.
            role: IAM role ARN for the Lambda execution role.
            runtime: Lambda runtime (default: ``"python3.12"``).
            handler: Lambda handler (default: ``"lambda_function.handler"``).
            timeout: Lambda timeout in seconds (default: 900).
            memory_size: Lambda memory in MB (default: 256).
            environment: Dict of environment variables for the Lambda.
            sagemaker_session: Optional SageMaker session for role resolution.

        Returns:
            CustomAgentLambda wrapping the created Lambda ARN.

        Raises:
            ValueError: If ``source`` is empty.
        """
        if not source or not source.strip():
            raise ValueError("'source' must be provided.")

        if not function_name:
            from sagemaker.train.utils import _get_unique_name

            function_name = _get_unique_name("SageMaker-agent-adapter", max_length=64)

        if not role:
            from sagemaker.train.defaults import TrainDefaults

            sagemaker_session = TrainDefaults.get_sagemaker_session(
                sagemaker_session=sagemaker_session
            )
            role = TrainDefaults.get_role(role=role, sagemaker_session=sagemaker_session)

        lambda_client = boto3.client("lambda")

        if S3_URI_PATTERN.match(source):
            bucket, key = _parse_s3_uri(source)
            if key.endswith(".zip"):
                code_param = {"S3Bucket": bucket, "S3Key": key}
            else:
                s3_client = boto3.client("s3")
                response = s3_client.get_object(Bucket=bucket, Key=key)
                code_content = response["Body"].read().decode("utf-8")
                code_param = {"ZipFile": _zip_code(code_content)}
        else:
            code_content = source
            if Path(source).exists():
                with open(source, "r") as f:
                    code_content = f.read()
            code_param = {"ZipFile": _zip_code(code_content)}

        response = lambda_client.create_function(
            FunctionName=function_name,
            Runtime=runtime,
            Role=role,
            Handler=handler,
            Code=code_param,
            Timeout=timeout,
            MemorySize=memory_size,
            Environment={"Variables": environment} if environment else {},
        )
        return cls(lambda_arn=response["FunctionArn"])

    @classmethod
    def get(cls, lambda_arn: str) -> CustomAgentLambda:
        """Wrap an existing Lambda ARN.

        Validates the Lambda exists by calling GetFunction.

        Args:
            lambda_arn: ARN of an existing Lambda function.

        Returns:
            CustomAgentLambda wrapping the Lambda ARN.

        Raises:
            botocore.exceptions.ClientError: If the Lambda does not exist.
        """
        lambda_client = boto3.client("lambda")
        lambda_client.get_function(FunctionName=lambda_arn)
        return cls(lambda_arn=lambda_arn)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an S3 URI into (bucket, key)."""
    path = uri[len("s3://"):]
    bucket, _, key = path.partition("/")
    return bucket, key


def _zip_code(code_content: str) -> bytes:
    """Package code content into a zip archive."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("lambda_function.py", code_content)
    zip_buffer.seek(0)
    return zip_buffer.read()
