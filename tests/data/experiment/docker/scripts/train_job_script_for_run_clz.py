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
"""This script file runs on SageMaker training job"""
from __future__ import absolute_import

import logging
import time

import os

import boto3

from sagemaker import Session
from sagemaker.experiments.run import Run

for key, value in os.environ.items():
    logging.info("OS env var - {}: {}".format(key, value))

boto_session = boto3.Session(region_name=os.environ["AWS_REGION"])
sagemaker_session = Session(boto_session=boto_session)

with Run.init(
    experiment_name="my-train-job-exp-in-script",
    run_name="my-train-job-run-in-script",
    sagemaker_session=sagemaker_session,
) as run:
    logging.info(f"Run name: {run.run_name}")
    logging.info(f"Experiment name: {run.experiment_name}")
    logging.info(f"Trial component name: {run._trial_component.trial_component_name}")
    run.log_parameter("p1", 1.0)
    run.log_parameter("p2", 2.0)
    if "TRAINING_JOB_ARN" in os.environ:
        for i in range(2):
            run.log_metric("A", i)
        for i in range(2):
            run.log_metric("B", i)
        for i in range(2):
            run.log_metric("C", i)
        for i in range(2):
            time.sleep(0.003)
            run.log_metric("D", i)
        for i in range(2):
            time.sleep(0.003)
            run.log_metric("E", i)
        time.sleep(15)
