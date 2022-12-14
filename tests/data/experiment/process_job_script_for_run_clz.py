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
"""This script file runs on SageMaker processing job"""
from __future__ import absolute_import

import logging
import os
import boto3

sdk_file = "sagemaker-dev-1.0.tar.gz"
os.system(f"pip install {sdk_file}")


from sagemaker import Session
from sagemaker.experiments import load_run


boto_session = boto3.Session(region_name=os.environ["AWS_REGION"])
sagemaker_session = Session(boto_session=boto_session)


with load_run(sagemaker_session=sagemaker_session) as run:
    logging.info(f"Run name: {run.run_name}")
    logging.info(f"Experiment name: {run.experiment_name}")
    logging.info(f"Trial component name: {run._trial_component.trial_component_name}")
    run.log_parameters({"p3": 3.0, "p4": 4.0})
    run.log_metric("test-job-load-log-metric", 0.1)
