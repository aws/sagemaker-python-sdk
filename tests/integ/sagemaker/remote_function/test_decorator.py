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


import pytest
import os
import logging
import random
import string
import numpy as np
import pandas as pd
import subprocess
import shlex
from sagemaker.experiments.run import Run, load_run
from sagemaker.remote_function import CheckpointLocation
from tests.integ.sagemaker.experiments.helpers import cleanup_exp_resources
from sagemaker.experiments.trial_component import _TrialComponent
from sagemaker.experiments._api_types import _TrialComponentStatusType

from sagemaker.remote_function import remote
from sagemaker.remote_function.spark_config import SparkConfig
from sagemaker.remote_function.custom_file_filter import CustomFileFilter
from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentError,
)
from sagemaker.remote_function.errors import (
    DeserializationError,
    SerializationError,
)
from sagemaker.utils import unique_name_from_base

from tests.integ.kms_utils import get_or_create_kms_key
from tests.integ import DATA_DIR
from tests.integ.s3_utils import assert_s3_files_exist

ROLE = "SageMakerRole"
CHECKPOINT_FILE_CONTENT = "test checkpoint file"


@pytest.fixture(scope="module")
def s3_kms_key(sagemaker_session):
    return get_or_create_kms_key(sagemaker_session=sagemaker_session)


@pytest.fixture(scope="module")
def checkpoint_s3_location(sagemaker_session):
    def random_s3_uri():
        return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))

    return "s3://{}/rm-func-checkpoints/{}".format(
        sagemaker_session.default_bucket(), random_s3_uri()
    )


def test_decorator(sagemaker_session, dummy_container_without_error, cpu_instance_type):
    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=60,
    )
    def divide(x, y):
        return x / y

    assert divide(10, 2) == 5
    assert divide(20, 2) == 10


def test_decorated_function_raises_exception(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
    )
    def divide(x, y):
        logging.warning(f"{x}/{y}")
        return x / y

    with pytest.raises(ZeroDivisionError):
        divide(10, 0)


def test_remote_python_runtime_is_incompatible(
    sagemaker_session, dummy_container_incompatible_python_runtime, cpu_instance_type
):
    @remote(
        role=ROLE,
        image_uri=dummy_container_incompatible_python_runtime,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
    )
    def divide(x, y):
        return x / y

    with pytest.raises(
        RuntimeEnvironmentError,
        match=(
            "Please make sure that the python version used in the training container "
            "is same as the local python version."
        ),
    ):
        divide(10, 2)


# TODO: add VPC settings, update SageMakerRole with KMS permissions
@pytest.mark.skip
def test_advanced_job_setting(
    sagemaker_session, dummy_container_without_error, cpu_instance_type, s3_kms_key
):
    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        s3_kms_key=s3_kms_key,
        sagemaker_session=sagemaker_session,
    )
    def divide(x, y):
        return x / y

    assert divide(10, 2) == 5


def test_with_custom_file_filter(
    sagemaker_session, dummy_container_without_error, cpu_instance_type, monkeypatch
):
    source_dir_path = os.path.join(os.path.dirname(__file__))
    with monkeypatch.context() as m:
        m.chdir(source_dir_path)
        dependencies_path = os.path.join(DATA_DIR, "remote_function", "requirements.txt")

        @remote(
            role=ROLE,
            image_uri=dummy_container_without_error,
            dependencies=dependencies_path,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            include_local_workdir=True,
            custom_file_filter=CustomFileFilter(),
            keep_alive_period_in_seconds=300,
        )
        def train(x):
            from helpers import local_module
            from helpers.nested_helper import local_module2

            return local_module.square(x) + local_module2.cube(x)

        assert train(2) == 12


def test_with_misconfigured_custom_file_filter(
    sagemaker_session, dummy_container_without_error, cpu_instance_type, monkeypatch
):
    source_dir_path = os.path.join(os.path.dirname(__file__))
    with monkeypatch.context() as m:
        m.chdir(source_dir_path)
        dependencies_path = os.path.join(DATA_DIR, "remote_function", "requirements.txt")

        @remote(
            role=ROLE,
            image_uri=dummy_container_without_error,
            dependencies=dependencies_path,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            include_local_workdir=True,
            # exclude critical modules
            custom_file_filter=CustomFileFilter(ignore_name_patterns=["helpers"]),
            keep_alive_period_in_seconds=300,
        )
        def train(x):
            from helpers import local_module
            from helpers.nested_helper import local_module2

            return local_module.square(x) + local_module2.cube(x)

        with pytest.raises(ModuleNotFoundError):
            train(2)


def test_with_additional_dependencies(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    dependencies_path = os.path.join(DATA_DIR, "remote_function", "requirements.txt")

    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        dependencies=dependencies_path,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
    )
    def cuberoot(x):
        from scipy.special import cbrt

        return cbrt(x)

    assert cuberoot(27) == 3


def test_additional_dependencies_with_job_conda_env(
    sagemaker_session, dummy_container_with_conda, cpu_instance_type
):
    dependencies_path = os.path.join(DATA_DIR, "remote_function", "requirements.txt")

    @remote(
        role=ROLE,
        image_uri=dummy_container_with_conda,
        dependencies=dependencies_path,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        job_conda_env="integ_test_env",
    )
    def cuberoot(x):
        from scipy.special import cbrt

        return cbrt(x)

    assert cuberoot(27) == 3


def test_additional_dependencies_with_default_conda_env(
    sagemaker_session, dummy_container_with_conda, cpu_instance_type
):
    dependencies_path = os.path.join(DATA_DIR, "remote_function", "requirements.txt")

    @remote(
        role=ROLE,
        image_uri=dummy_container_with_conda,
        dependencies=dependencies_path,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
    )
    def cuberoot(x):
        from scipy.special import cbrt

        return cbrt(x)

    assert cuberoot(27) == 3


def test_additional_dependencies_with_non_existent_conda_env(
    sagemaker_session, dummy_container_with_conda, cpu_instance_type
):
    dependencies_path = os.path.join(DATA_DIR, "remote_function", "requirements.txt")

    @remote(
        role=ROLE,
        image_uri=dummy_container_with_conda,
        dependencies=dependencies_path,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        job_conda_env="non_existent_env",
    )
    def cuberoot(x):
        from scipy.special import cbrt

        return cbrt(x)

    with pytest.raises(RuntimeEnvironmentError):
        cuberoot(27) == 3


def test_additional_dependencies_with_conda_yml_file(
    sagemaker_session, dummy_container_with_conda, cpu_instance_type, conda_env_yml
):
    @remote(
        role=ROLE,
        image_uri=dummy_container_with_conda,
        dependencies=conda_env_yml,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        job_conda_env="integ_test_env",
        keep_alive_period_in_seconds=120,
    )
    def cuberoot(x):
        from scipy.special import cbrt

        return cbrt(x)

    assert cuberoot(27) == 3


def test_with_non_existent_dependencies(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    dependencies_path = os.path.join(DATA_DIR, "remote_function", "non_existent_requirements.txt")

    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        dependencies=dependencies_path,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=30,
    )
    def divide(x, y):
        return x / y

    with pytest.raises(RuntimeEnvironmentError):
        divide(10, 2)


def test_with_incompatible_dependencies(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    """
    This test is limited by the python version it is run with.
    It is currently working with python 3.8+. However, running it with older versions
    or versions in the future may require changes to 'old_deps_requirements.txt'
    to fulfill testing scenario.

    """

    dependencies_path = os.path.join(DATA_DIR, "remote_function", "old_deps_requirements.txt")

    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        dependencies=dependencies_path,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=30,
    )
    def mul_ten(df: pd.DataFrame):
        return df.mul(10)

    df1 = pd.DataFrame(
        {
            "A": [14, 4, 5, 4, 1],
            "B": [5, 2, 54, 3, 2],
            "C": [20, 20, 7, 3, 8],
            "D": [14, 3, 6, 2, 6],
        }
    )

    with pytest.raises(DeserializationError):
        mul_ten(df1)


def test_decorator_with_exp_and_run_names_passed_to_remote_function(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=30,
    )
    def train(exp_name, run_name):

        with Run(experiment_name=exp_name, run_name=run_name) as run:
            print(f"Experiment name: {run.experiment_name}")
            print(f"Run name: {run.run_name}")
            print(f"Trial component name: {run._trial_component.trial_component_name}")

            run.log_parameter("p1", 1.0)
            run.log_parameter("p2", 2)

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

    exp_name = unique_name_from_base("my-test-exp")
    run_name = "my-test-run"
    tc_name = Run._generate_trial_component_name(experiment_name=exp_name, run_name=run_name)

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        train(exp_name, run_name)

        tc = _TrialComponent.load(trial_component_name=tc_name, sagemaker_session=sagemaker_session)

        assert tc.start_time
        assert tc.end_time
        assert tc.status.primary_status == _TrialComponentStatusType.Completed.value
        assert tc.parameters["p1"] == 1.0
        assert tc.parameters["p2"] == 2.0
        assert len(tc.metrics) == 5
        for metric_summary in tc.metrics:
            # metrics deletion is not supported at this point
            # so its count would accumulate
            assert metric_summary.count > 0
            assert metric_summary.min == 0.0
            assert metric_summary.max == 1.0


def test_decorator_load_run_inside_remote_function(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=30,
    )
    def train():
        with load_run() as run:
            run.log_parameters({"p3": 3.0, "p4": 4})
            run.log_metric("test-job-load-log-metric", 0.1)

    exp_name = unique_name_from_base("my-test-exp")
    run_name = "my-test-run"
    tc_name = Run._generate_trial_component_name(experiment_name=exp_name, run_name=run_name)

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with Run(
            experiment_name=exp_name,
            run_name=run_name,
            sagemaker_session=sagemaker_session,
        ):
            train()

        tc = _TrialComponent.load(trial_component_name=tc_name, sagemaker_session=sagemaker_session)

        assert tc.parameters["p3"] == 3.0
        assert tc.parameters["p4"] == 4.0
        assert len(tc.metrics) > 0
        for metric_summary in tc.metrics:
            if metric_summary.metric_name != "test-job-load-log-metric":
                continue
            assert metric_summary.last == 0.1
            assert metric_summary.max == 0.1
            assert metric_summary.min == 0.1


def test_decorator_with_nested_exp_run(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=30,
    )
    def train(exp_name, run_name):
        with Run(
            experiment_name=exp_name,
            run_name=run_name,
        ) as run:
            print(f"Experiment name: {run.experiment_name}")
            print(f"Run name: {run.run_name}")
            print(f"Trial component name: {run._trial_component.trial_component_name}")

            run.log_parameter("p1", 1.0)
            run.log_parameter("p2", 2)

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

    exp_name = unique_name_from_base("my-test-exp")
    run_name = "my-test-run"

    with cleanup_exp_resources(exp_names=[exp_name], sagemaker_session=sagemaker_session):
        with pytest.raises(
            RuntimeError, match="It is not allowed to use nested 'with' statements on the Run."
        ):
            with Run(
                experiment_name=exp_name,
                run_name=run_name,
                sagemaker_session=sagemaker_session,
            ):
                train(
                    exp_name=exp_name,
                    run_name=run_name,
                )


def test_decorator_function_defined_in_with_run(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    exp_name = unique_name_from_base("my-test-exp")
    run_name = "my-test-run"
    with Run(
        experiment_name=exp_name,
        run_name=run_name,
        sagemaker_session=sagemaker_session,
    ) as run:

        @remote(
            role=ROLE,
            image_uri=dummy_container_without_error,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )
        def train(metric_1, value_1, metric_2, value_2):
            run.log_parameter(metric_1, value_1)
            run.log_parameter(metric_2, value_2)

        with pytest.raises(SerializationError) as e:
            train("p1", 1.0, "p2", 0.5)
            assert isinstance(e.__cause__, NotImplementedError)


def test_decorator_pre_execution_command(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):

    random_str_1 = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    random_str_2 = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    random_str_3 = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))

    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        pre_execution_commands=[
            f"echo {random_str_1} > {random_str_1}",
            f"echo {random_str_2} > {random_str_2}",
            f"echo {random_str_3} > {random_str_3}",
            f"rm ./{random_str_2}",
        ],
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=60,
    )
    def get_file_content(file_names):
        joined_content = ""
        for file_name in file_names:
            if os.path.exists(file_name):
                with open(f"{file_name}", "r") as f:
                    joined_content += f.read()
        return joined_content

    assert (
        get_file_content([random_str_1, random_str_2, random_str_3])
        == random_str_1 + "\n" + random_str_3 + "\n"
    )


def test_decorator_pre_execution_script(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):

    pre_execution_script_path = os.path.join(DATA_DIR, "remote_function", "pre_exec_commands")

    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        pre_execution_script=pre_execution_script_path,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=60,
    )
    def get_file_content(file_names):
        joined_content = ""
        for file_name in file_names:
            if os.path.exists(file_name):
                with open(f"{file_name}", "r") as f:
                    joined_content += f.read()
        return joined_content

    assert (
        get_file_content(["test_file_1", "test_file_2", "test_file_3"])
        == "test-content-1" + "\n" + "test-content-3" + "\n"
    )


def test_decorator_pre_execution_command_error(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):

    random_str_1 = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    random_str_2 = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
    random_str_3 = "".join(random.choices(string.ascii_uppercase + string.digits, k=10))

    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        pre_execution_commands=[
            f"echo {random_str_1} > {random_str_1}",
            "aws sagemaker describe-training-job",
            f"echo {random_str_3} > {random_str_3}",
        ],
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=60,
    )
    def get_file_content(file_names):
        joined_content = ""
        for file_name in file_names:
            if os.path.exists(file_name):
                with open(f"{file_name}", "r") as f:
                    joined_content += f.read()
        return joined_content

    with pytest.raises(RuntimeEnvironmentError) as e:
        get_file_content([random_str_1, random_str_2, random_str_3])
        assert "aws: error: the following arguments are required: --training-job-name" in str(e)


def test_decorator_pre_execution_script_error(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):

    pre_execution_script_path = os.path.join(
        DATA_DIR, "remote_function", "pre_exec_commands_bad_cmd"
    )

    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        pre_execution_script=pre_execution_script_path,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=60,
    )
    def get_file_content(file_names):
        joined_content = ""
        for file_name in file_names:
            if os.path.exists(file_name):
                with open(f"{file_name}", "r") as f:
                    joined_content += f.read()
        return joined_content

    with pytest.raises(RuntimeEnvironmentError) as e:
        get_file_content(["test_file_1", "test_file_2", "test_file_3"])
        assert "line 2: bws: command not found" in str(e)


def test_decorator_with_spot_instances(
    sagemaker_session, dummy_container_without_error, cpu_instance_type
):
    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        use_spot_instances=True,
        max_wait_time_in_seconds=48 * 60 * 60,
    )
    def divide(x, y):
        return x / y

    assert divide(10, 2) == 5
    assert divide(20, 2) == 10


def test_decorator_with_spot_instances_save_and_load_checkpoints(
    sagemaker_session,
    dummy_container_without_error,
    cpu_instance_type,
    checkpoint_s3_location,
):
    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        use_spot_instances=True,
        max_wait_time_in_seconds=48 * 60 * 60,
    )
    def save_checkpoints(checkpoint_path: Union[str, os.PathLike]):
        file_path_1 = os.path.join(checkpoint_path, "checkpoint_1.json")
        with open(file_path_1, "w") as f:
            f.write(CHECKPOINT_FILE_CONTENT)

        file_path_2 = os.path.join(checkpoint_path, "checkpoint_2.json")
        with open(file_path_2, "w") as f:
            f.write(CHECKPOINT_FILE_CONTENT)

        return CHECKPOINT_FILE_CONTENT

    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        use_spot_instances=True,
        max_wait_time_in_seconds=48 * 60 * 60,
    )
    def load_checkpoints(checkpoint_path: Union[str, os.PathLike]):
        file_path_1 = os.path.join(checkpoint_path, "checkpoint_1.json")
        with open(file_path_1, "r") as file:
            file_content_1 = file.read()

        file_path_2 = os.path.join(checkpoint_path, "checkpoint_2.json")
        with open(file_path_2, "r") as file:
            file_content_2 = file.read()

        return file_content_1 + file_content_2

    assert save_checkpoints(CheckpointLocation(checkpoint_s3_location)) == CHECKPOINT_FILE_CONTENT
    assert_s3_files_exist(
        sagemaker_session, checkpoint_s3_location, ["checkpoint_1.json", "checkpoint_2.json"]
    )

    assert (
        load_checkpoints(CheckpointLocation(checkpoint_s3_location))
        == CHECKPOINT_FILE_CONTENT + CHECKPOINT_FILE_CONTENT
    )


def test_with_user_and_workdir_set_in_the_image(
    sagemaker_session, dummy_container_with_user_and_workdir, cpu_instance_type
):
    dependencies_path = os.path.join(DATA_DIR, "remote_function", "requirements.txt")

    @remote(
        role=ROLE,
        image_uri=dummy_container_with_user_and_workdir,
        dependencies=dependencies_path,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
    )
    def cuberoot(x):
        from scipy.special import cbrt

        return cbrt(x)

    assert cuberoot(27) == 3


def test_with_user_and_workdir_set_in_the_image_client_error_case(
    sagemaker_session, dummy_container_with_user_and_workdir, cpu_instance_type
):
    client_error_message = "Testing client error in job."

    @remote(
        role=ROLE,
        image_uri=dummy_container_with_user_and_workdir,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
    )
    def my_func():
        raise RuntimeError(client_error_message)

    with pytest.raises(RuntimeError) as error:
        my_func()
    assert client_error_message in str(error)


@pytest.mark.skip
def test_decorator_with_spark_job(sagemaker_session, cpu_instance_type):
    @remote(
        role=ROLE,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=60,
        spark_config=SparkConfig(
            configuration=[
                {
                    "Classification": "spark-defaults",
                    "Properties": {"spark.app.name", "remote-spark-test"},
                }
            ]
        ),
    )
    def test_spark_transform():
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        assert spark.conf.get(spark.app.name) == "remote-spark-test"

    test_spark_transform()


@pytest.mark.skip
def test_decorator_auto_capture(sagemaker_session, auto_capture_test_container):
    """
    This test runs a docker container. The Container invocation will execute a python script
    with remote function to test auto_capture scenario. The test requires conda to be
    installed on the client side which is not available in the code build image. Hence we need
    to run the test in another docker container with conda installed.

    Any assertion is not needed because if remote function execution fails, docker run comand
    will throw an error thus failing this test.
    """
    creds = sagemaker_session.boto_session.get_credentials()
    region = sagemaker_session.boto_session.region_name
    env = {
        "AWS_ACCESS_KEY_ID": str(creds.access_key),
        "AWS_SECRET_ACCESS_KEY": str(creds.secret_key),
        "AWS_SESSION_TOKEN": str(creds.token),
    }
    cmd = (
        f"docker run -e AWS_ACCESS_KEY_ID={env['AWS_ACCESS_KEY_ID']} "
        f"-e AWS_SECRET_ACCESS_KEY={env['AWS_SECRET_ACCESS_KEY']} "
        f"-e AWS_SESSION_TOKEN={env['AWS_SESSION_TOKEN']} "
        f"-e AWS_DEFAULT_REGION={region} "
        f"--rm {auto_capture_test_container}"
    )
    subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT).decode("utf-8")


def test_decorator_torchrun(
    sagemaker_session,
    dummy_container_without_error,
    gpu_instance_type,
    use_torchrun=False,
    use_mpirun=False,
):
    @remote(
        role=ROLE,
        image_uri=dummy_container_without_error,
        instance_type=gpu_instance_type,
        sagemaker_session=sagemaker_session,
        keep_alive_period_in_seconds=60,
        use_torchrun=use_torchrun,
        use_mpirun=use_mpirun,
    )
    def divide(x, y):
        return x / y

    assert divide(10, 2) == 5
    assert divide(20, 2) == 10
