from operator import index

import pytest
from unittest.mock import Mock, patch
import numpy as np
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.resources import (
    TrainingJob, Model, EndpointConfig, Endpoint,
    AlgorithmSpecification, Channel, DataSource, S3DataSource,
    OutputDataConfig, ResourceConfig, StoppingCondition
)
from sagemaker.core.shapes import InvokeEndpointOutput
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from sagemaker.utils.base_serializers import CSVSerializer, JSONSerializer
from sagemaker.utils.base_deserializers import CSVDeserializer, JSONDeserializer
import pandas as pd
import time
import boto3


@pytest.mark.integration
class TestEndpointInvoke:

    @pytest.fixture(scope="class")
    def sagemaker_session(self):
        """Create a SageMaker session."""
        return Session()

    @pytest.fixture(scope="class")
    def role(self):
        """Get the execution role."""
        return get_execution_role()

    @pytest.fixture(scope="class")
    def region(self, sagemaker_session):
        """Get the AWS region."""
        return sagemaker_session.boto_region_name

    @pytest.fixture(scope="class")
    def simple_data(self):
        # Create a very simple dataset with 5 rows
        # First column is target (0 or 1), followed by two features
        train_df = pd.DataFrame([
            [0, 1.0, 2.0],  # Row 1
            [1, 2.0, 3.0],  # Row 2
            [0, 1.5, 2.5],  # Row 3
        ], columns=['target', 'feature1', 'feature2'])

        test_df = pd.DataFrame([
            [1, 2.5, 3.5],  # Row 4
            [0, 1.2, 2.2],  # Row 5
        ], columns=['target', 'feature1', 'feature2'])

        # Create version of test data without target
        test_df_no_target = test_df.drop('target', axis=1)

        return {
            "train_data": train_df,
            "test_data": test_df,
            "test_data_no_target": test_df_no_target
        }

    @pytest.fixture(scope="class")
    def training_resources(self, sagemaker_session, simple_data):
        # Set up S3 paths
        bucket = "sagemaker-us-west-2-913524917855"
        prefix = f"test-scikit-iris-{int(time.time())}"
        train_path = f"s3://{bucket}/{prefix}/train.csv"
        output_path = f"s3://{bucket}/{prefix}/output"

        simple_data["train_data"].to_csv('train.csv', index=False, header=False)#.encode('utf-8')
        simple_data["test_data"].to_csv('test.csv', index=False, header=False)

        # Upload training data
        sagemaker_session.upload_data(
            "train.csv",
            bucket=bucket,
            key_prefix=f"{prefix}"
        )

        return {
            "bucket": bucket,
            "prefix": prefix,
            "train_path": train_path,
            "output_path": output_path
        }


    @pytest.fixture(scope="class")
    def endpoint(self, sagemaker_session, training_resources, role):
        # Create training job
        image = "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest"
        job_name = f"test-xgboost-{int(time.time())}"

        training_job = TrainingJob.create(
            training_job_name=job_name,
            hyper_parameters={
                "objective": "multi:softmax",
                "num_class": "3",
                "num_round": "10",
                "eval_metric": "merror",
            },
            algorithm_specification=AlgorithmSpecification(
                training_image=image,
                training_input_mode="File"
            ),
            role_arn=role,
            input_data_config=[
                Channel(
                    channel_name="train",
                    content_type="csv",
                    data_source=DataSource(
                        s3_data_source=S3DataSource(
                            s3_data_type="S3Prefix",
                            s3_uri=training_resources["train_path"],
                            s3_data_distribution_type="FullyReplicated",
                        )
                    ),
                )
            ],
            output_data_config=OutputDataConfig(
                s3_output_path=training_resources["output_path"]
            ),
            resource_config=ResourceConfig(
                instance_type="ml.m4.xlarge",
                instance_count=1,
                volume_size_in_gb=30
            ),
            stopping_condition=StoppingCondition(max_runtime_in_seconds=600),
        )
        training_job.wait()

        # Create model, endpoint config, and endpoint
        model_name = f"test-model-{int(time.time())}"
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(
                image=image,
                model_data_url=training_job.model_artifacts.s3_model_artifacts,
            ),
            execution_role_arn=role,
        )

        endpoint_config = EndpointConfig.create(
            endpoint_config_name=model_name,
            production_variants=[
                ProductionVariant(
                    variant_name=model_name,
                    initial_instance_count=1,
                    instance_type="ml.m5.xlarge",
                    model_name=model,
                )
            ],
        )

        endpoint = Endpoint.create(
            endpoint_name=model_name,
            endpoint_config_name=endpoint_config,
        )
        endpoint.wait_for_status("InService")

        yield endpoint

        # Cleanup
        endpoint.delete()
        endpoint_config.delete()
        model.delete()

    def test_endpoint_invoke_with_serializers(self, endpoint, simple_data):
        # Test with serializer and deserializer
        serializer = CSVSerializer()
        deserializer = CSVDeserializer()

        endpoint.serializer = serializer
        endpoint.deserializer = deserializer


        response = endpoint.invoke(
            body=simple_data["test_data_no_target"],
            content_type="text/csv",
            accept="text/csv"
        )

        assert response is not None
        assert isinstance(response, InvokeEndpointOutput)
        assert hasattr(response, 'body')
        assert hasattr(response, 'content_type')

    def test_endpoint_invoke_without_serializers(self, endpoint, simple_data):
        # Test without serializer and deserializer
        endpoint.serializer = None
        endpoint.deserializer = None

        response = endpoint.invoke(
            body=simple_data["test_data_no_target"].to_csv(index=False, header=False),
            content_type="text/csv",
            accept="text/csv"
        )

        assert response is not None
        assert isinstance(response, InvokeEndpointOutput)
        assert hasattr(response, 'body')
        assert hasattr(response, 'content_type')

    def test_endpoint_invoke_with_invalid_serializer_config(self, endpoint):
        # Test with only serializer but no deserializer
        endpoint.serializer = CSVSerializer()
        endpoint.deserializer = None

        with pytest.raises(ValueError) as exc_info:
            endpoint.invoke(
                body="test data",
                content_type="text/csv"
            )
        assert "Both serializer and deserializer must be provided together" in str(exc_info.value)


