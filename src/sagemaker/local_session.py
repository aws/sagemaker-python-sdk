# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import time

import urllib3
from botocore.exceptions import ClientError

from sagemaker.image import train, serve
from sagemaker.session import Session


class LocalSagemakerClient(object):
    def __init__(self, boto_session=None):
        self.train_instance_count = None
        self.boto_session = boto_session
        self.s3_model_artifacts = None
        self.model_name = None
        self.primary_container = None
        self.role_arn = None
        self.created_endpoint = False

    def create_training_job(self, TrainingJobName, AlgorithmSpecification, RoleArn, InputDataConfig, OutputDataConfig,
                            ResourceConfig, StoppingCondition, HyperParameters, Tags=None):
        self.train_instance_count = ResourceConfig['InstanceCount']

        self.s3_model_artifacts = train(AlgorithmSpecification, InputDataConfig, ResourceConfig, HyperParameters,
                                        self.boto_session)

    def describe_training_job(self, TrainingJobName):
        response = {'ResourceConfig': {'InstanceCount': self.train_instance_count},
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

        self.container = serve(self.primary_container, self.variants[0])
        self.container.up()
        self.created_endpoint = True

        i = 0
        http = urllib3.PoolManager()
        while True:
            i += 1

            if i > 1:
                time.sleep(1)

            if i >= 10:
                print("Giving up, endpoint didn't launch correctly")
                return

            print("Checking if endpoint is up, attempt: %s" % i)
            try:
                r = http.request('GET', "http://localhost:8080/ping")
                if r.status != 200:
                    print("Container still not up, got: %s" % r.status)
                    continue
            except:
                print("Container still not up")
                continue
            print("Container is up")
            return


    def delete_endpoint(self, EndpointName):
        self.container.down()


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

        self.sagemaker_client = LocalSagemakerClient(boto_session)
        self.sagemaker_runtime_client = LocalSagemakerRuntimeClient()
