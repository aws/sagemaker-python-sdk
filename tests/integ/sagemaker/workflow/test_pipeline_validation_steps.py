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

import pytest
import sagemaker

from mock import (
    Mock,
    PropertyMock,
    patch,
)

from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import Properties, PropertyFile
from sagemaker import get_execution_role, utils

from sagemaker.processing import Processor
from sagemaker.processing import ProcessingInput
from sagemaker.workflow.steps import (
    ProcessingStep,
    CacheConfig,
)

from botocore.exceptions import ClientError, ValidationError

REGION = "us-west-2"
BUCKET = "my-bucket"
IMAGE_URI = "fakeimage"
ROLE = "DummyRole"
MODEL_NAME = "gisele"


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
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


def test_processing_step_inputs_exceeds_limit_on_pipeline_create(sagemaker_session):
    processing_input_data_uri_parameter = ParameterString(
        name="ProcessingInputDataUri", default_value=f"s3://{BUCKET}/processing_manifest"
    )
    instance_type_parameter = ParameterString(name="InstanceType", default_value="ml.m4.4xlarge")
    instance_count_parameter = ParameterInteger(name="InstanceCount", default_value=1)
    processor = Processor(
        image_uri=IMAGE_URI,
        role=ROLE,
        instance_count=instance_count_parameter,
        instance_type=instance_type_parameter,
        sagemaker_session=sagemaker_session,
    )
    # Botocore supports max of 10 processing inputs. Throw error at compile time.
    inputs = []
    for _ in range(11):
        inputs.append(
            ProcessingInput(
                source="",
                destination="processing_manifest"
            )
        )
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    step = ProcessingStep(
        name="OverloadedProcessing Step",
        description="ProcessingStep description",
        display_name="MyProcessingStep",
        depends_on=["TestStep", "SecondTestStep"],
        processor=processor,
        inputs=inputs,
        outputs=[],
        cache_config=cache_config,
        property_files=[evaluation_report],
    )
    step.add_depends_on(["ThirdTestStep"])
    pipeline = Pipeline(
        name="OverloadedProcessingStep_Pipeline",
        parameters=[
            processing_input_data_uri_parameter,
            instance_type_parameter,
            instance_count_parameter,
        ],
        steps=[step],
        sagemaker_session=sagemaker_session,
    )

    try:
        with pytest.raises(ValidationError):
            response = pipeline.create(role)
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass

