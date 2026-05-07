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

import pytest

from sagemaker.core.experiments.trial_component import _TrialComponent
from sagemaker.core.remote_function import RemoteExecutor
from sagemaker.core.remote_function.client import get_future, list_futures
from sagemaker.core.remote_function.core.serialization import CloudpickleSerializer
from sagemaker.core.remote_function.errors import DeserializationError
from sagemaker.core.s3 import S3Uploader
from sagemaker.core.s3 import s3_path_join
from sagemaker.core.common_utils import unique_name_from_base

ROLE = "SageMakerRole"


def test_executor_submit(sagemaker_session, dummy_container_without_error, cpu_instance_type):
    def square(x):
        return x * x

    def cube(x):
        return x * x * x

    timestamp = int(time.time())
    job_prefix = f"test-submit-{timestamp}"
    with RemoteExecutor(
        max_parallel_jobs=1,
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=30,
        job_name_prefix=job_prefix,
    ) as e:
        future_1 = e.submit(square, 10)
        future_2 = e.submit(cube, 10)

    assert future_1.result() == 100
    assert future_2.result() == 1000

    assert get_future(future_1._job.job_name, sagemaker_session).result() == 100
    assert get_future(future_2._job.job_name, sagemaker_session).result() == 1000

    assert next(
        list_futures(job_name_prefix=job_prefix, sagemaker_session=sagemaker_session)
    )._job.job_name.startswith(job_prefix)


def test_executor_map(sagemaker_session, dummy_container_without_error, cpu_instance_type):
    def power(a, b):
        return a**b

    timestamp = int(time.time())
    job_prefix = f"test-map-{timestamp}"
    with RemoteExecutor(
        max_parallel_jobs=1,
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=30,
        job_name_prefix=job_prefix,
    ) as e:
        results = e.map(power, [5, 6], [2, 3])

    assert len(results) == 2
    assert results[0] == 25
    assert results[1] == 216

    assert next(
        list_futures(job_name_prefix=job_prefix, sagemaker_session=sagemaker_session)
    )._job.job_name.startswith(job_prefix)


def test_executor_submit_using_spot_instances(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    def square_on_spot_instance(x):
        return x * x

    def cube_on_spot_instance(x):
        return x * x * x

    timestamp = int(time.time())
    job_prefix = f"test-submit-spot-instances-{timestamp}"
    with RemoteExecutor(
        max_parallel_jobs=1,
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        use_spot_instances=True,
        max_wait_time_in_seconds=48 * 60 * 60,
        job_name_prefix=job_prefix,
    ) as e:
        future_1 = e.submit(square_on_spot_instance, 10)
        future_2 = e.submit(cube_on_spot_instance, 10)

    assert future_1.result() == 100
    assert future_2.result() == 1000

    assert get_future(future_1._job.job_name, sagemaker_session).result() == 100
    assert get_future(future_2._job.job_name, sagemaker_session).result() == 1000

    assert next(
        list_futures(job_name_prefix=job_prefix, sagemaker_session=sagemaker_session)
    )._job.job_name.startswith(job_prefix)

    describe_job_1 = next(
        list_futures(job_name_prefix=job_prefix, sagemaker_session=sagemaker_session)
    )._job.describe()

    assert describe_job_1["EnableManagedSpotTraining"] is True
    assert describe_job_1["StoppingCondition"]["MaxWaitTimeInSeconds"] == 172800

    describe_job_2 = next(
        list_futures(job_name_prefix=job_prefix, sagemaker_session=sagemaker_session)
    )._job.describe()
    assert describe_job_2["EnableManagedSpotTraining"] is True
    assert describe_job_2["StoppingCondition"]["MaxWaitTimeInSeconds"] == 172800


def test_executor_submit_checksum_mismatch(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    def original_func(x):
        return x * x

    def updated_func(x):
        return x + 25

    def wait_for_future_running(future, timeout_in_seconds):
        polling_timeout = time.time() + timeout_in_seconds
        while time.time() < polling_timeout:
            if future.running():
                return True
            time.sleep(1)
        return False

    pickled_updated_func = CloudpickleSerializer.serialize(updated_func)

    with RemoteExecutor(
        max_parallel_jobs=1,
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=30,
        pre_execution_commands=["sleep 60"],
    ) as e:
        future = e.submit(original_func, 10)
        if wait_for_future_running(future, 60):
            S3Uploader.upload_bytes(
                pickled_updated_func,
                s3_path_join(future._job.s3_uri, "function", "payload.pkl"),
                kms_key=None,
                sagemaker_session=sagemaker_session,
            )
        else:
            assert False, "Expected future to be in running state, but it timed-out."

    with pytest.raises(DeserializationError) as ex:
        future.result()
    assert "Integrity check for the serialized function or data failed" in str(ex)
