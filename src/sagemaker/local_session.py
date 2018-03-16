# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import datetime
import logging
import time

import urllib3
from botocore.exceptions import ClientError

from sagemaker.image import SageMakerContainer
from sagemaker.session import Session

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class LocalSagemakerClient(object):
    def __init__(self, sagemaker_session=None):
        self.train_container = None
        self.serve_container = None
        self.sagemaker_session = sagemaker_session
        self.s3_model_artifacts = None
        self.model_name = None
        self.primary_container = None
        self.role_arn = None
        self.created_endpoint = False

    def create_training_job(self, TrainingJobName, AlgorithmSpecification, RoleArn, InputDataConfig, OutputDataConfig,
                            ResourceConfig, StoppingCondition, HyperParameters, Tags=None):

        self.train_container = SageMakerContainer(ResourceConfig['InstanceType'], ResourceConfig['InstanceCount'],
                                                  AlgorithmSpecification['TrainingImage'], self.sagemaker_session)

        self.s3_model_artifacts = self.train_container.train(InputDataConfig, HyperParameters)

    def describe_training_job(self, TrainingJobName):
        response = {'ResourceConfig': {'InstanceCount': self.train_container.instance_count},
                    'TrainingJobStatus': 'Completed',
                    'TrainingStartTime': datetime.datetime.now(),
                    'TrainingEndTime': datetime.datetime.now(),
                    'ModelArtifacts': {'S3ModelArtifacts': self.s3_model_artifacts}
                    }
        return response

    def create_model(self, ModelName, PrimaryContainer, ExecutionRoleArn):
        self.model_name = ModelName
        self.primary_container = PrimaryContainer
        self.role_arn = ExecutionRoleArn

    def describe_endpoint_config(self, EndpointConfigName):
        if self.created_endpoint:
            return True
        else:
            error_response = {'Error': {'Code': 'ValidationException', 'Message': 'Could not find endpoint'}}
            raise ClientError(error_response, 'describe_endpoint_config')

    def create_endpoint_config(self, EndpointConfigName, ProductionVariants):
        self.variants = ProductionVariants

    def describe_endpoint(self, EndpointName):
        return {'EndpointStatus': 'InService'}

    def create_endpoint(self, EndpointName, EndpointConfigName):
        instance_type = self.variants[0]['InstanceType']
        instance_count = self.variants[0]['InitialInstanceCount']
        self.serve_container = SageMakerContainer(instance_type, instance_count,
                                                  self.primary_container['Image'], self.sagemaker_session)
        self.serve_container.serve(self.primary_container)
        self.created_endpoint = True

        i = 0
        http = urllib3.PoolManager()
        while True:
            i += 1

            if i >= 10:
                raise RuntimeError("Giving up, endpoint: %s didn't launch correctly" % EndpointName)

            logger.info("Checking if endpoint is up, attempt: %s" % i)
            try:
                r = http.request('GET', "http://localhost:8080/ping")
                if r.status != 200:
                    logger.info("Container still not up, got: %s" % r.status)
                else:
                    return
            except urllib3.exceptions.RequestError:
                logger.info("Container still not up")

            time.sleep(1)

    def delete_endpoint(self, EndpointName):
        self.serve_container.stop_serving()


class LocalSagemakerRuntimeClient(object):
    def __init__(self):
        self.http = urllib3.PoolManager()

    def invoke_endpoint(self, Body, EndpointName, ContentType, Accept):
        r = self.http.request('POST', "http://localhost:8080/invocations", body=Body, preload_content=False,
                              headers={'Content-type': ContentType, 'Accept': Accept})

        return {'Body': r, 'ContentType': Accept}


class LocalSession(Session):

    def __init__(self, boto_session=None):
        super(LocalSession, self).__init__(boto_session)

        self.sagemaker_client = LocalSagemakerClient(self)
        self.sagemaker_runtime_client = LocalSagemakerRuntimeClient()
