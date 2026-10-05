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


class BatchTestResourceManager:

    def __init__(
        self,
        batch_client,
        queue_name="pysdk-test-queue",
        service_env_name="pysdk-test-queue-service-environment",
    ):
        self.batch_client = batch_client
        self.queue_name = queue_name
        self.service_environment_name = service_env_name

    def _create_or_get_service_environment(self, service_environment_name):
        print(f"Creating service environment: {service_environment_name}")
        try:
            response = self.batch_client.create_service_environment(
                serviceEnvironmentName=service_environment_name,
                serviceEnvironmentType="SAGEMAKER_TRAINING",
                capacityLimits=[{"maxCapacity": 10, "capacityUnit": "NUM_INSTANCES"}],
            )
            print(f"Service environment {service_environment_name} created successfully.")
            return response
        except Exception as e:
            if "Object already exists" in str(e):
                print("Resource already exists. Fetching existing resource.")
                response = self.batch_client.describe_service_environments(
                    serviceEnvironments=[service_environment_name]
                )
                return response["serviceEnvironments"][0]
            else:
                print(f"Error creating service environment: {e}")
                raise

    def _create_or_get_queue(self, queue_name, service_environment_arn):

        print(f"Creating job queue: {queue_name}")
        try:
            response = self.batch_client.create_job_queue(
                jobQueueName=queue_name,
                priority=1,
                computeEnvironmentOrder=[],
                serviceEnvironmentOrder=[
                    {
                        "order": 1,
                        "serviceEnvironment": service_environment_arn,
                    },
                ],
                jobQueueType="SAGEMAKER_TRAINING",
            )
            print(f"Job queue {queue_name} created successfully.")
            return response
        except Exception as e:
            if "Object already exists" in str(e):
                print("Resource already exists. Fetching existing resource.")
                response = self.batch_client.describe_job_queues(jobQueues=[queue_name])
                return response["jobQueues"][0]
            else:
                print(f"Error creating job queue: {e}")
                raise

    def _update_queue_state(self, queue_name, state):
        try:
            print(f"Updating queue {queue_name} to state {state}")
            response = self.batch_client.update_job_queue(jobQueue=queue_name, state=state)
            return response
        except Exception as e:
            print(f"Error updating queue: {e}")

    def _update_service_environment_state(self, service_environment_name, state):
        print(f"Updating service environment {service_environment_name} to state {state}")
        try:
            response = self.batch_client.update_service_environment(
                serviceEnvironment=service_environment_name, state=state
            )
            return response
        except Exception as e:
            print(f"Error updating service environment: {e}")

    def _wait_for_queue_state(self, queue_name, state):
        print(f"Waiting for queue {queue_name} to be {state}...")
        while True:
            response = self.batch_client.describe_job_queues(jobQueues=[queue_name])
            print(f"Current state: {response}")
            if response["jobQueues"][0]["state"] == state:
                break
            time.sleep(5)
        print(f"Queue {queue_name} is now {state}.")

    def _wait_for_service_environment_state(self, service_environment_name, state):
        print(f"Waiting for service environment {service_environment_name} to be {state}...")
        while True:
            response = self.batch_client.describe_service_environments(
                serviceEnvironments=[service_environment_name]
            )
            print(f"Current state: {response}")
            if response["serviceEnvironments"][0]["state"] == state:
                break
            time.sleep(5)
        print(f"Service environment {service_environment_name} is now {state}.")

    def get_or_create_resources(self, queue_name=None, service_environment_name=None):
        queue_name = queue_name or self.queue_name
        service_environment_name = service_environment_name or self.service_environment_name

        service_environment = self._create_or_get_service_environment(service_environment_name)
        if service_environment.get("state") != "ENABLED":
            self._update_service_environment_state(service_environment_name, "ENABLED")
            self._wait_for_service_environment_state(service_environment_name, "ENABLED")
        time.sleep(10)

        queue = self._create_or_get_queue(queue_name, service_environment["serviceEnvironmentArn"])
        if queue.get("state") != "ENABLED":
            self._update_queue_state(queue_name, "ENABLED")
            self._wait_for_queue_state(queue_name, "ENABLED")
        time.sleep(10)
        return queue, service_environment
