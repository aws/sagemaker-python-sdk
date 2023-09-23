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

from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.remote_function import RemoteExecutor
from sagemaker.remote_function.client import get_future, list_futures
from sagemaker.experiments.run import Run, load_run
from tests.integ.sagemaker.experiments.helpers import cleanup_exp_resources
from sagemaker.utils import unique_name_from_base

ROLE = "SageMakerRole"


def test_executor_submit(sagemaker_session, dummy_container_without_error, cpu_instance_type):
    def square(x):
        return x * x

    def cube(x):
        return x * x * x

    with RemoteExecutor(
        max_parallel_jobs=1,
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=30,
    ) as e:
        future_1 = e.submit(square, 10)
        future_2 = e.submit(cube, 10)

    assert future_1.result() == 100
    assert future_2.result() == 1000

    assert get_future(future_1._job.job_name, sagemaker_session).result() == 100
    assert get_future(future_2._job.job_name, sagemaker_session).result() == 1000

    assert next(
        list_futures(job_name_prefix="square", sagemaker_session=sagemaker_session)
    )._job.job_name.startswith("square")
    assert next(
        list_futures(job_name_prefix="cube", sagemaker_session=sagemaker_session)
    )._job.job_name.startswith("cube")


def test_executor_map(sagemaker_session, dummy_container_without_error, cpu_instance_type):
    def power(a, b):
        return a**b

    with RemoteExecutor(
        max_parallel_jobs=1,
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=30,
    ) as e:
        results = e.map(power, [5, 6], [2, 3])

    assert len(results) == 2
    assert results[0] == 25
    assert results[1] == 216

    assert next(
        list_futures(job_name_prefix="power", sagemaker_session=sagemaker_session)
    )._job.job_name.startswith("power")


def test_executor_submit_with_run_inside(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    def square(x):
        with load_run() as run:
            result = x * x
            run.log_metric("x", result)
        return result

    def cube(x):
        with load_run() as run:
            result = x * x * x
            run.log_metric("x", result)
        return result

    exp_name = unique_name_from_base("my-test-exp")
    run_name = "my-test-run"
    tc_name = Run._generate_trial_component_name(experiment_name=exp_name, run_name=run_name)

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with RemoteExecutor(
            max_parallel_jobs=1,
            role=ROLE,
            image_uri=dummy_container_without_error,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            keep_alive_period_in_seconds=30,
        ) as e:
            with Run(
                experiment_name=exp_name,
                run_name=run_name,
                sagemaker_session=sagemaker_session,
            ):
                future_1 = e.submit(square, 10)
                future_2 = e.submit(cube, 10)

        assert future_1.result() == 100
        assert future_2.result() == 1000

        assert get_future(future_1._job.job_name, sagemaker_session).result() == 100
        assert get_future(future_2._job.job_name, sagemaker_session).result() == 1000

        assert next(
            list_futures(job_name_prefix="square", sagemaker_session=sagemaker_session)
        )._job.job_name.startswith("square")
        assert next(
            list_futures(job_name_prefix="cube", sagemaker_session=sagemaker_session)
        )._job.job_name.startswith("cube")

        tc = _TrialComponent.load(trial_component_name=tc_name, sagemaker_session=sagemaker_session)
        assert len(tc.metrics) > 0
        for metric_summary in tc.metrics:
            if metric_summary.metric_name != "x":
                continue
            assert metric_summary.max == 1000
            assert metric_summary.min == 100
            assert metric_summary.avg == 550


def test_executor_submit_with_run_outside(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    def square(x):
        with load_run() as run:
            result = x * x
            run.log_metric("x", result)
        return result

    def cube(x):
        with load_run() as run:
            result = x * x * x
            run.log_metric("x", result)
        return result

    exp_name = unique_name_from_base("my-test-exp")
    run_name = "my-test-run"
    tc_name = Run._generate_trial_component_name(experiment_name=exp_name, run_name=run_name)

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with Run(
            experiment_name=exp_name,
            run_name=run_name,
            sagemaker_session=sagemaker_session,
        ):
            with RemoteExecutor(
                max_parallel_jobs=1,
                role=ROLE,
                image_uri=dummy_container_without_error,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
                keep_alive_period_in_seconds=30,
            ) as e:
                future_1 = e.submit(square, 10)
                future_2 = e.submit(cube, 10)

        assert future_1.result() == 100
        assert future_2.result() == 1000

        assert get_future(future_1._job.job_name, sagemaker_session).result() == 100
        assert get_future(future_2._job.job_name, sagemaker_session).result() == 1000

        assert next(
            list_futures(job_name_prefix="square", sagemaker_session=sagemaker_session)
        )._job.job_name.startswith("square")
        assert next(
            list_futures(job_name_prefix="cube", sagemaker_session=sagemaker_session)
        )._job.job_name.startswith("cube")

        tc = _TrialComponent.load(trial_component_name=tc_name, sagemaker_session=sagemaker_session)
        assert len(tc.metrics) > 0
        for metric_summary in tc.metrics:
            if metric_summary.metric_name != "x":
                continue
            assert metric_summary.max == 1000
            assert metric_summary.min == 100
            assert metric_summary.avg == 550


def test_executor_submit_using_spot_instances(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    def square_on_spot_instance(x):
        return x * x

    def cube_on_spot_instance(x):
        return x * x * x

    with RemoteExecutor(
        max_parallel_jobs=1,
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        use_spot_instances=True,
        max_wait_time_in_seconds=48 * 60 * 60,
    ) as e:
        future_1 = e.submit(square_on_spot_instance, 10)
        future_2 = e.submit(cube_on_spot_instance, 10)

    assert future_1.result() == 100
    assert future_2.result() == 1000

    assert get_future(future_1._job.job_name, sagemaker_session).result() == 100
    assert get_future(future_2._job.job_name, sagemaker_session).result() == 1000

    describe_job_1 = next(
        list_futures(job_name_prefix="square-on-spot-instance", sagemaker_session=sagemaker_session)
    )._job.describe()
    assert describe_job_1["EnableManagedSpotTraining"] is True
    assert describe_job_1["StoppingCondition"]["MaxWaitTimeInSeconds"] == 172800

    describe_job_2 = next(
        list_futures(job_name_prefix="cube-on-spot-instance", sagemaker_session=sagemaker_session)
    )._job.describe()
    assert describe_job_2["EnableManagedSpotTraining"] is True
    assert describe_job_2["StoppingCondition"]["MaxWaitTimeInSeconds"] == 172800


def test_executor_map_with_run(sagemaker_session, dummy_container_without_error, cpu_instance_type):
    def square(x):
        with load_run() as run:
            result = x * x
            run.log_metric("x", result)
        return result

    exp_name = unique_name_from_base("my-test-exp")
    run_name = "my-test-run"
    tc_name = Run._generate_trial_component_name(experiment_name=exp_name, run_name=run_name)

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with Run(
            experiment_name=exp_name,
            run_name=run_name,
            sagemaker_session=sagemaker_session,
        ):
            with RemoteExecutor(
                max_parallel_jobs=2,
                role=ROLE,
                image_uri=dummy_container_without_error,
                instance_type=cpu_instance_type,
                sagemaker_session=sagemaker_session,
                keep_alive_period_in_seconds=30,
            ) as e:
                results = e.map(square, [2, 4])

        assert len(results) == 2
        assert results[0] == 4
        assert results[1] == 16

        with RemoteExecutor(
            max_parallel_jobs=2,
            role=ROLE,
            image_uri=dummy_container_without_error,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            keep_alive_period_in_seconds=30,
        ) as e:
            with Run(
                experiment_name=exp_name,
                run_name=run_name,
                sagemaker_session=sagemaker_session,
            ):
                results = e.map(square, [6, 8])

        assert len(results) == 2
        assert results[0] == 36
        assert results[1] == 64

        tc = _TrialComponent.load(trial_component_name=tc_name, sagemaker_session=sagemaker_session)
        assert len(tc.metrics) > 0
        for metric_summary in tc.metrics:
            if metric_summary.metric_name != "x":
                continue
            assert metric_summary.max == 64
            assert metric_summary.min == 4
            assert metric_summary.avg == 30
