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
    CAPACITY_UNIT = "ml.m5.2xlarge"

    def __init__(
        self,
        batch_client,
        queue_name="pysdk-test-qm-queue",
        service_env_name="pysdk-test-qm-queue-service-environment",
        scheduling_policy_name="pysdk-test-qm-scheduling-policy",
        quota_share_name="pysdk-test-quota-share",
    ):
        self.batch_client = batch_client
        self.queue_name = queue_name
        self.service_environment_name = service_env_name
        self.scheduling_policy_name = scheduling_policy_name
        self.quota_share_name = quota_share_name

    def _create_or_get_service_environment(self, service_environment_name):
        print(f"Creating service environment: {service_environment_name}")
        try:
            response = self.batch_client.create_service_environment(
                serviceEnvironmentName=service_environment_name,
                serviceEnvironmentType="SAGEMAKER_TRAINING",
                capacityLimits=[{"maxCapacity": 10, "capacityUnit": BatchTestResourceManager.CAPACITY_UNIT}],
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

    def _create_or_get_queue(self, queue_name, service_environment_arn, scheduling_policy_arn=None):
        print(f"Creating job queue: {queue_name}")
        try:
            create_params = {
                "jobQueueName": queue_name,
                "priority": 1,
                "computeEnvironmentOrder": [],
                "serviceEnvironmentOrder": [
                    {
                        "order": 1,
                        "serviceEnvironment": service_environment_arn,
                    },
                ],
                "jobQueueType": "SAGEMAKER_TRAINING",
            }
            if scheduling_policy_arn:
                create_params["schedulingPolicyArn"] = scheduling_policy_arn
            response = self.batch_client.create_job_queue(**create_params)
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

    def _find_scheduling_policy(self, scheduling_policy_name):
        paginator = self.batch_client.get_paginator("list_scheduling_policies")
        for page in paginator.paginate():
            for sp in page.get("schedulingPolicies", []):
                if scheduling_policy_name in sp["arn"]:
                    return sp
        return None

    def _create_or_get_scheduling_policy(self, scheduling_policy_name):
        print(f"Creating scheduling policy: {scheduling_policy_name}")
        try:
            response = self.batch_client.create_scheduling_policy(
                name=scheduling_policy_name,
                quotaSharePolicy={"idleResourceAssignmentStrategy": "FIFO"},
            )
            print(f"Scheduling policy {scheduling_policy_name} created successfully.")
            return response
        except Exception as e:
            if "Object already exists" in str(e):
                print("Resource already exists. Fetching existing resource.")
                sp = self._find_scheduling_policy(scheduling_policy_name)
                if not sp:
                    raise
                return sp
            else:
                print(f"Error creating scheduling policy: {e}")
                raise

    def _create_or_get_quota_share(self, quota_share_name, queue_name):
        print(f"Creating quota share: {quota_share_name}")
        try:
            response = self.batch_client.create_quota_share(
                quotaShareName=quota_share_name,
                jobQueue=queue_name,
                capacityLimits=[{"maxCapacity": 10, "capacityUnit": BatchTestResourceManager.CAPACITY_UNIT}],
                resourceSharingConfiguration={"strategy": "RESERVE"},
                preemptionConfiguration={"inSharePreemption": "DISABLED"},
                state="ENABLED",
            )
            print(f"Quota share {quota_share_name} created successfully.")
            return response
        except Exception as e:
            if "already exists" in str(e):
                print("Resource already exists. Fetching existing resource.")
                desc_jq = self.batch_client.describe_job_queues(jobQueues=[queue_name])
                jq_arn = desc_jq["jobQueues"][0]["jobQueueArn"]
                return self.batch_client.describe_quota_share(quotaShareArn=f"{jq_arn}/quota-share/{quota_share_name}")
            else:
                print(f"Error creating quota share: {e}")
                raise

    def _update_quota_share_state(self, quota_share_arn, state):
        print(f"Updating quota share {quota_share_arn} to state {state}")
        try:
            response = self.batch_client.update_quota_share(quotaShareArn=quota_share_arn, state=state)
            return response
        except Exception as e:
            print(f"Error updating quota share: {e}")

    def _wait_for_quota_share_state(self, quota_share_arn, expected_status, expected_state, timeout=300):
        print(f"Waiting for quota share to be {expected_status}...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = self.batch_client.describe_quota_share(quotaShareArn=quota_share_arn)
            except Exception as e:
                if expected_status == "DELETED" and "does not exist" in str(e):
                    return
                raise e

            state = response.get("state")
            status = response.get("status")

            if status == expected_status and state == expected_state:
                print(f"Quota share is now {expected_state}.")
                return
            if status == "INVALID":
                raise ValueError(f"Something went wrong!")

            time.sleep(5)
        raise TimeoutError(f"Quota share did not reach {expected_state} within {timeout}s")

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

    def _wait_for_queue_state(self, job_queue_name, expected_status, expected_state, timeout=300):
        print(f"Waiting for queue {job_queue_name} to be {expected_status}...")
        start = time.time()
        while time.time() - start < timeout:
            describe_jq_response = self.batch_client.describe_job_queues(
                jobQueues=[job_queue_name]
            )
            if describe_jq_response["jobQueues"]:
                jq = describe_jq_response["jobQueues"][0]

                state = jq["state"]
                status = jq["status"]

                if status == expected_status and state == expected_state:
                    print(f"Queue {job_queue_name} is now {state}.")
                    return
                if status == "INVALID":
                    raise ValueError(f"Something went wrong!")
            elif expected_status == "DELETED":
                print(f"JobQueue {job_queue_name} has been deleted")
                return

            time.sleep(5)
        raise TimeoutError(f"Queue {job_queue_name} did not reach {expected_state} within {timeout}s")

    def _wait_for_service_environment_state(self, service_environment_name, expected_status, expected_state, timeout=300):
        print(f"Waiting for service environment {service_environment_name} to be {expected_status}...")
        start = time.time()
        while time.time() - start < timeout:
            describe_response = self.batch_client.describe_service_environments(
                serviceEnvironments=[service_environment_name]
            )
            if describe_response["serviceEnvironments"]:
                se = describe_response["serviceEnvironments"][0]

                state = se["state"]
                status = se["status"]

                if status == expected_status and state == expected_state:
                    print(f"Service environment {service_environment_name} is now {expected_state}.")
                    return
                if status == "INVALID":
                    raise ValueError(f"Something went wrong!")
            elif expected_status == "DELETED":
                print(f"ServiceEnvironment {service_environment_name} has been deleted")
                return

            time.sleep(5)
        raise TimeoutError(f"Service environment {service_environment_name} did not reach {expected_state} within {timeout}s")

    def _delete_service_environment(self, service_environment_name: str):
        print(f"Setting ServiceEnvironment {service_environment_name} to DISABLED")
        self.batch_client.update_service_environment(
            serviceEnvironment=service_environment_name, state="DISABLED"
        )

        print("Waiting for ServiceEnvironment update to finish...")
        self._wait_for_service_environment_state(service_environment_name, "VALID", "DISABLED")

        print(f"Deleting ServiceEnvironment {service_environment_name}")
        self.batch_client.delete_service_environment(serviceEnvironment=service_environment_name)

        print("Waiting for ServiceEnvironment update to finish...")
        self._wait_for_service_environment_state(service_environment_name, "DELETED", "DISABLED")

    def _delete_job_queue(self, job_queue_name: str):
        print(f"Disabling JobQueue {job_queue_name}")
        self.batch_client.update_job_queue(jobQueue=job_queue_name, state="DISABLED")

        print("Waiting for JobQueue update to finish...")
        self._wait_for_queue_state(job_queue_name, "VALID", "DISABLED")

        print(f"Deleting JobQueue {job_queue_name}")
        self.batch_client.delete_job_queue(jobQueue=job_queue_name)

        print("Waiting for JobQueue update to finish...")
        self._wait_for_queue_state(job_queue_name, "DELETED", "DISABLED")

    def _delete_scheduling_policy(self, scheduling_policy_arn: str):
        print(f"Deleting SchedulingPolicy {scheduling_policy_arn}")
        self.batch_client.delete_scheduling_policy(arn=scheduling_policy_arn)

    def _delete_quota_share(self, quota_share_arn: str):
        print(f"Disabling QuotaShare {quota_share_arn}")
        self.batch_client.update_quota_share(quotaShareArn=quota_share_arn, state="DISABLED")

        print("Waiting for QuotaShare update to finish...")
        self._wait_for_quota_share_state(quota_share_arn, "VALID", "DISABLED")

        print(f"Deleting QuotaShare {quota_share_arn}")
        self.batch_client.delete_quota_share(quotaShareArn=quota_share_arn)

        print("Waiting for QuotaShare deletion to finish...")
        self._wait_for_quota_share_state(quota_share_arn, "DELETED", "DISABLED")

    def get_or_create_resources(
        self,
        queue_name=None,
        service_environment_name=None,
        scheduling_policy_name=None,
        quota_share_name=None
    ):
        queue_name = queue_name or self.queue_name
        service_environment_name = service_environment_name or self.service_environment_name
        scheduling_policy_name = scheduling_policy_name or self.scheduling_policy_name
        quota_share_name = quota_share_name or self.quota_share_name

        service_environment = self._create_or_get_service_environment(service_environment_name)
        if service_environment.get("state") != "ENABLED":
            self._update_service_environment_state(service_environment_name, "ENABLED")
            self._wait_for_service_environment_state(service_environment_name, "VALID", "ENABLED")
        time.sleep(10)

        scheduling_policy = self._create_or_get_scheduling_policy(scheduling_policy_name)
        scheduling_policy_arn = scheduling_policy.get("arn")

        queue = self._create_or_get_queue(queue_name, service_environment["serviceEnvironmentArn"],
                                          scheduling_policy_arn)
        if queue.get("state") != "ENABLED":
            self._update_queue_state(queue_name, "ENABLED")
            self._wait_for_queue_state(queue_name, "VALID", "ENABLED")
        time.sleep(10)

        quota_share = self._create_or_get_quota_share(quota_share_name, queue_name)
        if quota_share.get("state") != "ENABLED":
            self._update_quota_share_state(quota_share["quotaShareArn"], "ENABLED")
            self._wait_for_quota_share_state(quota_share["quotaShareArn"], "VALID", "ENABLED")
        time.sleep(10)

        return queue, service_environment, scheduling_policy, quota_share

    def delete_resources(
        self,
        queue_name=None,
        service_environment_name=None,
        scheduling_policy_name=None,
        quota_share_name=None
    ):
        queue_name = queue_name or self.queue_name
        service_environment_name = service_environment_name or self.service_environment_name
        scheduling_policy_name = scheduling_policy_name or self.scheduling_policy_name
        quota_share_name = quota_share_name or self.quota_share_name

        # Get ARNs needed for deletion
        desc_jq = self.batch_client.describe_job_queues(jobQueues=[queue_name])
        if desc_jq["jobQueues"]:
            jq_arn = desc_jq["jobQueues"][0]["jobQueueArn"]
            quota_share_arn = f"{jq_arn}/quota-share/{quota_share_name}"
            self._delete_quota_share(quota_share_arn)

        self._delete_job_queue(queue_name)

        sp = self._find_scheduling_policy(scheduling_policy_name)
        if sp:
            self._delete_scheduling_policy(sp["arn"])

        self._delete_service_environment(service_environment_name)
