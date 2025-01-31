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
from typing import Union


import os
import re
import pytest
import subprocess
import logging
import numpy as np
import pandas as pd
import sagemaker
import boto3
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path
from sagemaker.local import LocalSession
from sagemaker.processing import (
    ProcessingInput, 
    ProcessingOutput
)
from sagemaker.sklearn import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.deserializers import CSVDeserializer
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import CSVSerializer


# Replace this role ARN with an appropriate role for your environment
ROLE = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"


def ensure_docker_compose_installed():
    """
    Downloads the Docker Compose plugin if not present, and verifies installation
    by checking the output of 'docker compose version' matches the pattern:
    'Docker Compose version vX.Y.Z'
    """

    cli_plugins_path = Path.home() / ".docker" / "cli-plugins"
    cli_plugins_path.mkdir(parents=True, exist_ok=True)

    compose_binary_path = cli_plugins_path / "docker-compose"
    if not compose_binary_path.exists():
        subprocess.run(
            [
                "curl",
                "-SL",
                "https://github.com/docker/compose/releases/download/v2.3.3/docker-compose-linux-x86_64",
                "-o",
                str(compose_binary_path),
            ],
            check=True,
        )
        subprocess.run(["chmod", "+x", str(compose_binary_path)], check=True)

    # Verify Docker Compose version
    try:
        output = subprocess.check_output(["docker", "compose", "version"], stderr=subprocess.STDOUT)
        output_decoded = output.decode("utf-8").strip()
        logging.info(f"'docker compose version' output: {output_decoded}")

        # Example expected format: "Docker Compose version vxxx"
        pattern = r"Docker Compose version+"
        match = re.search(pattern, output_decoded)
        assert (
            match is not None
        ), f"Could not find a Docker Compose version string matching '{pattern}' in: {output_decoded}"

    except subprocess.CalledProcessError as e:
        raise AssertionError(f"Failed to verify Docker Compose: {e}")


"""
Local Model: ProcessingJob
"""
@pytest.mark.local
def test_scikit_learn_local_processing():
    """
    Test local mode processing with a scikit-learn processor.
    This uses the same logic as scikit_learn_local_processing.py but in a pytest test function.

    Requirements/Assumptions:
      - Docker must be installed and running on the local machine.
      - 'processing_script.py' must be in the current working directory (or specify the correct path).
      - There should be some local input data if 'processing_script.py' needs it (see ProcessingInput below).
    """
    ensure_docker_compose_installed()
    
    # 1. Create local session for testing
    sagemaker_session = LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}

    # 2. Define a scikit-learn processor in local mode
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_count=1,
        instance_type="local",
        role=ROLE,
        sagemaker_session=sagemaker_session
    )

    logging.warning("Starting local processing job.")
    logging.warning("Note: the first run may take time to pull the required Docker image.")

    # 3. Run the processing job locally
    #    - Update 'source' and 'destination' paths based on your local folder setup
    processor.run(
        code="sample_processing_script.py",
        inputs=[
            ProcessingInput(
                source="s3://sagemaker-example-files-prod-us-east-1/datasets/tabular/uci_bank_marketing/bank-additional-full.csv",
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train_data", 
                source="/opt/ml/processing/output/train",
                destination="./output_data/train",
            ),
            ProcessingOutput(
                output_name="validation_data", 
                source="/opt/ml/processing/output/validation", 
                destination="./output_data/validation"
            ),
            ProcessingOutput(
                output_name="test_data", 
                source="/opt/ml/processing/output/test", 
                destination="./output_data/test"
            ),
        ],
    )
    assert True


"""
Local Model: Inference
"""
@pytest.mark.local
def test_pytorch_local_model_inference():
    """
    Test local mode inference for a TensorFlow NLP model using PyTorch. 
    This test deploys the model locally via Docker, performs an inference 
    on a sample image URL, and asserts that the output is received.
    """
    ensure_docker_compose_installed()

    # 1. Create a local session for inference
    sagemaker_session = LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}

    # pre created model for inference
    model_dir = 's3://aws-ml-blog/artifacts/pytorch-nlp-script-mode-local-model-inference/model.tar.gz'
    # sample dummy inference
    test_data = [
        "Never allow the same bug to bite you twice.",
        "The best part of Amazon SageMaker is that it makes machine learning easy.",
        "Amazon SageMaker Inference Recommender helps you choose the best available compute instance and configuration to deploy machine learning models for optimal inference performance and cost."
    ]
    logging.warning(f'test_data: {test_data}')

    model = PyTorchModel(
        model_data=model_dir,
        framework_version='1.8',
        # source_dir='inference',
        py_version='py3',
        entry_point='sample_inference_script.py',
        role=ROLE,
        sagemaker_session=sagemaker_session
    )

    logging.warning('Deploying endpoint in local mode')
    logging.warning(
        'Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='local',
        container_startup_health_check_timeout=600
    )

    # create a new CSV serializer and deserializer
    predictor.serializer = CSVSerializer()
    predictor.deserializer = CSVDeserializer()

    predictions = predictor.predict(
        ",".join(test_data)
    )
    logging.warning(f'predictions: {predictions}')
    # delete endpoint, clean up and terminate
    predictor.delete_endpoint(predictor.endpoint)

    # assert model response
    assert type(predictions) == list, "Response return type is a List"
    assert len(predictions) >= 1, "empty list returned"


def download_training_and_eval_data():
    logging.warning('Downloading training dataset')

    # Load California Housing dataset, then join labels and features
    california = datasets.fetch_california_housing()
    dataset = np.insert(california.data, 0, california.target, axis=1)
    # Create directory and write csv
    os.makedirs("./data/train", exist_ok=True)
    os.makedirs("./data/validation", exist_ok=True)
    os.makedirs("./data/test", exist_ok=True)

    train, other = train_test_split(dataset, test_size=0.3)
    validation, test = train_test_split(other, test_size=0.5)

    np.savetxt("./data/train/california_train.csv", train, delimiter=",")
    np.savetxt("./data/validation/california_validation.csv", validation, delimiter=",")
    np.savetxt("./data/test/california_test.csv", test, delimiter=",")

    logging.warning('Downloading completed')


def do_inference_on_local_endpoint(predictor):
    print(f'\nStarting Inference on endpoint (local).')
    test_data = pd.read_csv("data/test/california_test.csv", header=None)
    test_X = test_data.iloc[:, 1:]
    test_y = test_data.iloc[:, 0]
    predictions = predictor.predict(test_X.values)
    logging.warning("Predictions: {}".format(predictions))
    logging.warning("Actual: {}".format(test_y.values))
    logging.warning(f"RMSE: {mean_squared_error(predictions, test_y.values)}")
    return predictions, test_y.values, float(mean_squared_error(predictions, test_y.values))


"""
Local Model: TrainingJob and Inference
"""
@pytest.mark.local
def test_sklearn_local_model_train_inference():

    download_training_and_eval_data()
    
    logging.warning('Starting model training.')
    logging.warning('Note: if launching for the first time in local mode, container image download might take a few minutes to complete.')

    # 1. Create a local session for inference
    sagemaker_session = LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}
    
    sklearn = SKLearn(
        entry_point="sample_training_script.py",
        # source_dir='training',
        framework_version="1.2-1",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        instance_type="local",
        hyperparameters={"max_leaf_nodes": 30},
    )

    train_input = "file://./data/train/california_train.csv"
    validation_input = "file://./data/validation/california_validation.csv"

    sklearn.fit({"train": train_input, "validation": validation_input})
    logging.warning('Completed model training')
    logging.warning('Deploying endpoint in local mode')

    predictor = sklearn.deploy(
        initial_instance_count=1, 
        instance_type="local",
        container_startup_health_check_timeout=600
    )

    # get predictions from local endpoint
    test_preds, test_y, test_mse = do_inference_on_local_endpoint(predictor)
    
    logging.warning('About to delete the endpoint')
    predictor.delete_endpoint()

    assert type(test_preds) == np.ndarray, f"predictions are not in a np.ndarray format: {test_preds}"
    assert type(test_y) == np.ndarray, f"Y ground truth are not in a np.ndarray format: {test_y}"
    assert type(test_mse) == float, f"MSE is not a number: {test_mse}"
