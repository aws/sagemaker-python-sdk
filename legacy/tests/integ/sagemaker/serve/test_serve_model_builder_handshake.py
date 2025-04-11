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
import os
import uuid

import numpy as np
import pandas as pd
from sagemaker_core.main.resources import TrainingJob
from xgboost import XGBClassifier

from sagemaker.serve import ModelBuilder, SchemaBuilder
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker_core.main.shapes import (
    OutputDataConfig,
    StoppingCondition,
    Channel,
    DataSource,
    S3DataSource,
    AlgorithmSpecification,
    ResourceConfig,
)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sagemaker import get_execution_role, image_uris
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


@pytest.fixture(scope="module")
def data_setup(mb_sagemaker_session):
    sagemaker_session = mb_sagemaker_session
    bucket = sagemaker_session.default_bucket()

    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df["target"] = iris.target

    os.makedirs("./data", exist_ok=True)

    iris_df = iris_df[["target"] + [col for col in iris_df.columns if col != "target"]]

    train_data, test_data = train_test_split(iris_df, test_size=0.2, random_state=42)

    train_data.to_csv("./data/train.csv", index=False, header=False)
    test_data.to_csv("./data/test.csv", index=False, header=False)

    data = {"Name": ["Alice", "Bob", "Charlie"]}
    df = pd.DataFrame(data)
    schema_builder = SchemaBuilder(sample_input=df, sample_output=df)

    # Remove the target column from the testing data. We will use this to call invoke_endpoint later
    test_data.drop("target", axis=1)

    sagemaker_session.upload_data(
        DATA_DIRECTORY, bucket=bucket, key_prefix="{}/{}".format(prefix, DATA_DIRECTORY)
    )

    s3_input_path = "s3://{}/{}/data/{}".format(bucket, prefix, TRAIN_DATA)
    s3_output_path = "s3://{}/{}/output".format(bucket, prefix)

    data_setup = {
        "s3_input_path": s3_input_path,
        "s3_output_path": s3_output_path,
        "schema_builder": schema_builder,
    }
    return data_setup


def test_model_trainer_handshake(mb_sagemaker_session, mb_sagemaker_core_session, data_setup):
    sagemaker_session = mb_sagemaker_session
    role = get_execution_role(sagemaker_session=sagemaker_session)
    xgboost_image = image_uris.retrieve(
        framework="xgboost", region="us-west-2", image_scope="training"
    )

    model_trainer = ModelTrainer(
        sagemaker_session=mb_sagemaker_core_session,
        base_job_name="test-mb-handshake",
        hyperparameters={
            "objective": "multi:softmax",
            "num_class": "3",
            "num_round": "10",
            "eval_metric": "merror",
        },
        training_image=xgboost_image,
        training_input_mode="File",
        role=role,
        output_data_config=OutputDataConfig(s3_output_path=data_setup["s3_output_path"]),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=600),
    )

    model_trainer.train(
        input_data_config=[
            Channel(
                channel_name="train",
                content_type="csv",
                compression_type="None",
                record_wrapper_type="None",
                data_source=DataSource(
                    s3_data_source=S3DataSource(
                        s3_data_type="S3Prefix",
                        s3_uri=data_setup["s3_input_path"],
                        s3_data_distribution_type="FullyReplicated",
                    )
                ),
            )
        ]
    )

    model_builder = ModelBuilder(
        model=model_trainer,  # ModelTrainer object passed onto ModelBuilder directly
        sagemaker_session=sagemaker_session,
        role_arn=role,
        image_uri=xgboost_image,
        inference_spec=XGBoostSpec(),
        schema_builder=data_setup["schema_builder"],
        instance_type="ml.c6i.xlarge",
    )
    model = model_builder.build()
    assert model.model_data == model_trainer._latest_training_job.model_artifacts.s3_model_artifacts


def test_sagemaker_core_handshake(mb_sagemaker_session, data_setup):
    sagemaker_session = mb_sagemaker_session
    role = get_execution_role(sagemaker_session=sagemaker_session)
    xgboost_image = image_uris.retrieve(
        framework="xgboost", region="us-west-2", image_scope="training"
    )

    training_job_name = str(uuid.uuid4())
    training_job = TrainingJob.create(
        training_job_name=training_job_name,
        hyper_parameters={
            "objective": "multi:softmax",
            "num_class": "3",
            "num_round": "10",
            "eval_metric": "merror",
        },
        algorithm_specification=AlgorithmSpecification(
            training_image=xgboost_image, training_input_mode="File"
        ),
        role_arn=role,
        input_data_config=[
            Channel(
                channel_name="train",
                content_type="csv",
                compression_type="None",
                record_wrapper_type="None",
                data_source=DataSource(
                    s3_data_source=S3DataSource(
                        s3_data_type="S3Prefix",
                        s3_uri=data_setup["s3_input_path"],
                        s3_data_distribution_type="FullyReplicated",
                    )
                ),
            )
        ],
        output_data_config=OutputDataConfig(s3_output_path=data_setup["s3_output_path"]),
        resource_config=ResourceConfig(
            instance_type="ml.m4.xlarge", instance_count=1, volume_size_in_gb=30
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=600),
    )
    training_job.wait()

    model_builder = ModelBuilder(
        sagemaker_session=sagemaker_session,
        model=training_job,
        role_arn=role,
        inference_spec=XGBoostSpec(),
        image_uri=xgboost_image,
        schema_builder=data_setup["schema_builder"],
        instance_type="ml.c6i.xlarge",
    )
    model = model_builder.build()

    assert model.model_data == training_job.model_artifacts.s3_model_artifacts
