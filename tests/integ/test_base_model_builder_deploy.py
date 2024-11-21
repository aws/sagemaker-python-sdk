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
import unittest

from docker.utils import exclude_paths

from sagemaker import Session, get_execution_role

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import pandas as pd

import os

from sagemaker_core.main.shapes import AlgorithmSpecification, Channel, DataSource, S3DataSource, OutputDataConfig, \
    ResourceConfig, StoppingCondition
import uuid
from sagemaker.serve.builder.model_builder import  ModelBuilder, BatchTransformInferenceConfig
import pandas as pd
import numpy as np
from sagemaker.serve import InferenceSpec, SchemaBuilder
from sagemaker_core.main.resources import TrainingJob
from xgboost import XGBClassifier

from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig

from sagemaker.s3_utils import s3_path_join
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker_core.main.resources import Endpoint


class TestBaseModelBuilderDeploy(unittest.TestCase):
    def setUp(self):
        sagemaker_session = Session()
        role = get_execution_role()
        region = sagemaker_session.boto_region_name
        bucket = sagemaker_session.default_bucket()
        self.bucket = bucket
        # Get IRIS Data

        iris = load_iris()
        iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        iris_df['target'] = iris.target

        # Prepare Data

        os.makedirs('./data', exist_ok=True)

        iris_df = iris_df[['target'] + [col for col in iris_df.columns if col != 'target']]

        train_data, test_data = train_test_split(iris_df, test_size=0.2, random_state=42)

        train_data.to_csv('./data/train.csv', index=False, header=False)
        test_data.to_csv('./data/test.csv', index=False, header=False)

        # Remove the target column from the testing data. We will use this to call invoke_endpoint later
        test_data_no_target = test_data.drop('target', axis=1)

        prefix = "DEMO-scikit-iris"
        TRAIN_DATA = "train.csv"
        DATA_DIRECTORY = "data"

        train_input = sagemaker_session.upload_data(
            DATA_DIRECTORY, bucket=bucket, key_prefix="{}/{}".format(prefix, DATA_DIRECTORY)
        )

        s3_input_path = "s3://{}/{}/data/{}".format(bucket, prefix, TRAIN_DATA)
        s3_output_path = "s3://{}/{}/output".format(bucket, prefix)

        print(s3_input_path)
        print(s3_output_path)

        image = "433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest"


        class XGBoostSpec(InferenceSpec):
            def load(self, model_dir: str):
                print(model_dir)
                model = XGBClassifier()
                model.load_model(model_dir + "/xgboost-model")
                return model

            def invoke(self, input_object: object, model: object):
                prediction_probabilities = model.predict_proba(input_object)
                predictions = np.argmax(prediction_probabilities, axis=1)
                return predictions


        data = {
            'Name': ['Alice', 'Bob', 'Charlie']
        }
        df = pd.DataFrame(data)
        training_job_name = str(uuid.uuid4())
        schema_builder = SchemaBuilder(sample_input=df, sample_output=df)

        training_job = TrainingJob.create(
            training_job_name=training_job_name,
            hyper_parameters={
                'objective': 'multi:softmax',
                'num_class': '3',
                'num_round': '10',
                'eval_metric': 'merror'
            },
            algorithm_specification=AlgorithmSpecification(
                training_image=image,
                training_input_mode='File'
            ),
            role_arn=role,
            input_data_config=[
                Channel(
                    channel_name='train',
                    content_type='csv',
                    compression_type='None',
                    record_wrapper_type='None',
                    data_source=DataSource(
                        s3_data_source=S3DataSource(
                            s3_data_type='S3Prefix',
                            s3_uri=s3_input_path,
                            s3_data_distribution_type='FullyReplicated'
                        )
                    )
                )
            ],
            output_data_config=OutputDataConfig(
                s3_output_path=s3_output_path
            ),
            resource_config=ResourceConfig(
                instance_type='ml.m4.xlarge',
                instance_count=1,
                volume_size_in_gb=30
            ),
            stopping_condition=StoppingCondition(
                max_runtime_in_seconds=600
            )
        )
        training_job.wait()

        self.model_builder = ModelBuilder(
            name="ModelBuilderTest",
            model_path=training_job.model_artifacts.s3_model_artifacts,
            role_arn=role,
            inference_spec=XGBoostSpec(),
            image_uri=image,
            schema_builder=schema_builder,
            instance_type="ml.c6i.xlarge"
        )
        self.model_builder.build()

    def test_real_time_deployment(self):
        real_time_predictor = self.model_builder.deploy(
            endpoint_name="test",
            initial_instance_count=1)

        assert real_time_predictor is not None

    def test_serverless_deployment(self):
        serverless_predictor = self.model_builder.deploy(
            endpoint_name="test1",
            inference_config=ServerlessInferenceConfig())

        assert serverless_predictor is not None

    def test_async_deployment(self):
        async_predictor = self.model_builder.deploy(
            endpoint_name="test2",
            inference_config=AsyncInferenceConfig(
                output_path=s3_path_join("s3://", self.bucket, "async_inference/output")),
        )

        assert async_predictor is not None

    def tearDown(self):
        endpoints = Endpoint.get_all()
        for endpoint in endpoints:
            endpoint.delete()
