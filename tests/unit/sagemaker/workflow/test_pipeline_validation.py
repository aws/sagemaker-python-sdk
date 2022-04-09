from __future__ import absolute_import


import pytest
import sagemaker
import os

from mock import (
    Mock,
    PropertyMock,
)

from sagemaker.processing import (
    Processor,
    ProcessingInput,
    ScriptProcessor,
)

from botocore.exceptions import ValidationError
from sagemaker.network import NetworkConfig
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import Properties, PropertyFile
from sagemaker.workflow.parameters import ParameterString, ParameterInteger

from sagemaker.workflow.steps import (
    ProcessingStep,
    ConfigurableRetryStep,
    StepTypeEnum,
    CacheConfig,
)
from sagemaker.predictor import Predictor
from sagemaker.model import FrameworkModel
from tests.unit import DATA_DIR

DUMMY_SCRIPT_PATH = os.path.join(DATA_DIR, "dummy_script.py")

REGION = "us-west-2"
BUCKET = "my-bucket"
IMAGE_URI = "fakeimage"
ROLE = "DummyRole"
MODEL_NAME = "gisele"


class CustomStep(ConfigurableRetryStep):
    def __init__(self, name, display_name=None, description=None, retry_policies=None):
        super(CustomStep, self).__init__(
            name, StepTypeEnum.TRAINING, display_name, description, None, retry_policies
        )
        self._properties = Properties(path=f"Steps.{name}")

    @property
    def arguments(self):
        return dict()

    @property
    def properties(self):
        return self._properties


class DummyFrameworkModel(FrameworkModel):
    def __init__(self, sagemaker_session, **kwargs):
        super(DummyFrameworkModel, self).__init__(
            "s3://bucket/model_1.tar.gz",
            "mi-1",
            ROLE,
            os.path.join(DATA_DIR, "dummy_script.py"),
            sagemaker_session=sagemaker_session,
            **kwargs,
        )

    def create_predictor(self, endpoint_name):
        return Predictor(endpoint_name, self.sagemaker_session)


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
def script_processor(sagemaker_session):
    return ScriptProcessor(
        role=ROLE,
        image_uri="012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri",
        command=["python3"],
        instance_type="ml.m4.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="my_sklearn_processor",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=NetworkConfig(
            subnets=["my_subnet_id"],
            security_group_ids=["my_security_group_id"],
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
        ),
        sagemaker_session=sagemaker_session,
    )


def test_processing_step_inputs_exceed_max(sagemaker_session):
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
    inputs = []
    for _ in range(11):
        inputs.append(
            ProcessingInput(
                source=processing_input_data_uri_parameter,
                destination="processing_manifest",
            )
        )

    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    step = ProcessingStep(
        name="MyProcessingStep",
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
        name="MyPipeline",
        parameters=[
            processing_input_data_uri_parameter,
            instance_type_parameter,
            instance_count_parameter,
        ],
        steps=[step],
        sagemaker_session=sagemaker_session,
    )
    with pytest.raises(ValidationError) as error:
        pipeline.create(role_arn=ROLE)

    assert (
        str(error.value)
        == f"Invalid value ('{len(inputs)}') for param {inputs} of type {type(inputs)} "
    )
