from __future__ import absolute_import

import unittest
import os
import uuid

import numpy as np
import pandas as pd
from sagemaker_core.main.resources import TrainingJob
from xgboost import XGBClassifier

from sagemaker.serve import ModelBuilder, SchemaBuilder
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker_core.main.shapes import OutputDataConfig, StoppingCondition, Channel, DataSource, \
    S3DataSource, AlgorithmSpecification, ResourceConfig
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sagemaker import Session, get_execution_role, image_uris
from sagemaker.modules.train import ModelTrainer

prefix = "DEMO-scikit-iris"
TRAIN_DATA = "train.csv"
TEST_DATA = "test.csv"
DATA_DIRECTORY = "data"


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


class TestModelBuilderHandshake(unittest.TestCase):

    def setUp(self):
        self.sagemaker_session = Session()
        self.role = get_execution_role()
        self.region = self.sagemaker_session.boto_region_name
        self.bucket = self.sagemaker_session.default_bucket()
        self.setup_data()

    def setup_data(self):
        self.iris = load_iris()
        self.iris_df = pd.DataFrame(self.iris.data, columns=self.iris.feature_names)
        self.iris_df['target'] = self.iris.target

        os.makedirs('./data', exist_ok=True)

        iris_df = self.iris_df[
            ['target'] + [col for col in self.iris_df.columns if col != 'target']]

        self.train_data, self.test_data = train_test_split(iris_df, test_size=0.2, random_state=42)

        self.train_data.to_csv('./data/train.csv', index=False, header=False)
        self.test_data.to_csv('./data/test.csv', index=False, header=False)

        # Remove the target column from the testing data. We will use this to call invoke_endpoint later
        self.test_data_no_target = self.test_data.drop('target', axis=1)

        self.train_input = self.sagemaker_session.upload_data(
            DATA_DIRECTORY, bucket=self.bucket, key_prefix="{}/{}".format(prefix, DATA_DIRECTORY)
        )

        self.s3_input_path = "s3://{}/{}/data/{}".format(self.bucket, prefix, TRAIN_DATA)
        self.s3_output_path = "s3://{}/{}/output".format(self.bucket, prefix)
        self.s3_test_path = "s3://{}/{}/data/{}".format(self.bucket, prefix, TEST_DATA)
        self.xgboost_image = image_uris.retrieve(framework="xgboost", region="us-west-2",
                                                 image_scope="training")
        data = {
            'Name': ['Alice', 'Bob', 'Charlie']
        }
        df = pd.DataFrame(data)
        self.schema_builder = SchemaBuilder(sample_input=df, sample_output=df)

    def test_model_trainer_handshake(self):
        model_trainer = ModelTrainer(
            base_job_name='test-mb-handshake',
            hyperparameters={
                'objective': 'multi:softmax',
                'num_class': '3',
                'num_round': '10',
                'eval_metric': 'merror'
            },
            training_image=self.xgboost_image,
            training_input_mode='File',
            role=self.role,
            output_data_config=OutputDataConfig(
                s3_output_path=self.s3_output_path
            ),
            stopping_condition=StoppingCondition(
                max_runtime_in_seconds=600
            )
        )

        model_trainer.train(
            input_data_config=[
                Channel(
                    channel_name='train',
                    content_type='csv',
                    compression_type='None',
                    record_wrapper_type='None',
                    data_source=DataSource(
                        s3_data_source=S3DataSource(
                            s3_data_type='S3Prefix',
                            s3_uri=self.s3_input_path,
                            s3_data_distribution_type='FullyReplicated'
                        )
                    ))])

        model_builder = ModelBuilder(
            model=model_trainer,  # ModelTrainer object passed onto ModelBuilder directly
            role_arn=self.role,
            image_uri=self.xgboost_image,
            inference_spec=XGBoostSpec(),
            schema_builder=self.schema_builder,
            instance_type="ml.c6i.xlarge"
        )
        model = model_builder.build()
        assert (model.model_data == model_trainer
                ._latest_training_job.model_artifacts.s3_model_artifacts)

    def test_sagemaker_core_handshake(self):
        training_job_name = str(uuid.uuid4())
        training_job = TrainingJob.create(
            training_job_name=training_job_name,
            hyper_parameters={
                'objective': 'multi:softmax',
                'num_class': '3',
                'num_round': '10',
                'eval_metric': 'merror'
            },
            algorithm_specification=AlgorithmSpecification(
                training_image=self.xgboost_image,
                training_input_mode='File'
            ),
            role_arn=self.role,
            input_data_config=[
                Channel(
                    channel_name='train',
                    content_type='csv',
                    compression_type='None',
                    record_wrapper_type='None',
                    data_source=DataSource(
                        s3_data_source=S3DataSource(
                            s3_data_type='S3Prefix',
                            s3_uri=self.s3_input_path,
                            s3_data_distribution_type='FullyReplicated'
                        )
                    )
                )
            ],
            output_data_config=OutputDataConfig(
                s3_output_path=self.s3_output_path
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

        model_builder = ModelBuilder(
            model=training_job,
            role_arn=self.role,
            inference_spec=XGBoostSpec(),
            image_uri=self.xgboost_image,
            schema_builder=self.schema_builder,
            instance_type="ml.c6i.xlarge"
        )
        model = model_builder.build()

        assert model.model_data == training_job.model_artifacts.s3_model_artifacts
