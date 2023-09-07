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

import os
import time
import uuid

import pytest

from sagemaker.debugger import (
    DebuggerHookConfig,
    FrameworkProfile,
    ProfilerConfig,
    ProfilerRule,
    Rule,
    rule_configs,
)
from sagemaker.debugger.metrics_config import DetailedProfilingConfig
from sagemaker.mxnet.estimator import MXNet
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.timeout import timeout


TRAINING_STATUS = "Training"
ALGO_PULL_FINISHED_MESSAGE = "Training image download completed. Training in progress."


def _wait_until_training_can_be_updated(sagemaker_client, job_name, poll=5):
    ready_for_updating = _check_secondary_status(sagemaker_client, job_name)
    while not ready_for_updating:
        time.sleep(poll)
        ready_for_updating = _check_secondary_status(sagemaker_client, job_name)


def _check_secondary_status(sagemaker_client, job_name):
    desc = sagemaker_client.describe_training_job(TrainingJobName=job_name)
    secondary_status_transitions = desc.get("SecondaryStatusTransitions")
    if not secondary_status_transitions:
        return False

    latest_secondary_status_transition = secondary_status_transitions[-1]
    secondary_status = latest_secondary_status_transition.get("Status")
    status_message = latest_secondary_status_transition.get("StatusMessage")
    return TRAINING_STATUS == secondary_status and ALGO_PULL_FINISHED_MESSAGE == status_message


def test_mxnet_with_default_profiler_config_and_profiler_rule(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_training_latest_version,
            py_version=mxnet_training_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        training_job_name = unique_name_from_base("test-profiler-mxnet-training")
        mx.fit(
            inputs={"train": train_input, "test": test_input},
            job_name=training_job_name,
            wait=False,
        )

        job_description = mx.latest_training_job.describe()
        assert (
            job_description["ProfilerConfig"]
            == ProfilerConfig(
                s3_output_path=mx.output_path, system_monitor_interval_millis=500
            )._to_request_dict()
        )
        assert job_description.get("ProfilingStatus") == "Enabled"

        with pytest.raises(ValueError) as error:
            mx.enable_default_profiling()
        assert "Debugger monitoring is already enabled." in str(error)


def test_mxnet_with_custom_profiler_config_then_update_rule_and_config(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        profiler_config = ProfilerConfig(
            s3_output_path=f"s3://{sagemaker_session.default_bucket()}/{str(uuid.uuid4())}/system",
            system_monitor_interval_millis=1000,
        )
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_training_latest_version,
            py_version=mxnet_training_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            profiler_config=profiler_config,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        training_job_name = unique_name_from_base("test-profiler-mxnet-training")
        mx.fit(
            inputs={"train": train_input, "test": test_input},
            job_name=training_job_name,
            wait=False,
        )

        job_description = mx.latest_training_job.describe()
        assert job_description.get("ProfilerConfig") == profiler_config._to_request_dict()
        assert job_description.get("ProfilingStatus") == "Enabled"

        _wait_until_training_can_be_updated(sagemaker_session.sagemaker_client, training_job_name)

        mx.update_profiler(
            rules=[ProfilerRule.sagemaker(rule_configs.CPUBottleneck())],
            system_monitor_interval_millis=500,
        )

        job_description = mx.latest_training_job.describe()
        assert job_description["ProfilerConfig"]["S3OutputPath"] == profiler_config.s3_output_path
        assert job_description["ProfilerConfig"]["ProfilingIntervalInMilliseconds"] == 500


def test_mxnet_with_built_in_profiler_rule_with_custom_parameters(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        custom_profiler_report_rule = ProfilerRule.sagemaker(
            rule_configs.ProfilerReport(CPUBottleneck_threshold=90), name="CustomProfilerReportRule"
        )
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_training_latest_version,
            py_version=mxnet_training_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            rules=[custom_profiler_report_rule],
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        training_job_name = unique_name_from_base("test-profiler-mxnet-training")
        mx.fit(
            inputs={"train": train_input, "test": test_input},
            job_name=training_job_name,
            wait=False,
        )

        job_description = mx.latest_training_job.describe()
        assert job_description.get("ProfilingStatus") == "Enabled"
        assert (
            job_description.get("ProfilerConfig")
            == ProfilerConfig(
                s3_output_path=mx.output_path, system_monitor_interval_millis=500
            )._to_request_dict()
        )

        profiler_rule_configuration = job_description.get("ProfilerRuleConfigurations")[0]
        assert profiler_rule_configuration["RuleConfigurationName"] == "CustomProfilerReportRule"
        assert profiler_rule_configuration["RuleEvaluatorImage"] == mx.rules[0].image_uri
        assert profiler_rule_configuration["RuleParameters"] == {
            "rule_to_invoke": "ProfilerReport",
            "CPUBottleneck_threshold": "90",
        }


def test_mxnet_with_profiler_and_debugger_then_disable_framework_metrics(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        rules = [
            Rule.sagemaker(rule_configs.vanishing_gradient()),
            Rule.sagemaker(
                base_config=rule_configs.all_zero(), rule_parameters={"tensor_regex": ".*"}
            ),
            ProfilerRule.sagemaker(rule_configs.ProfilerReport(), name="CustomProfilerReportRule"),
        ]
        debugger_hook_config = DebuggerHookConfig(
            s3_output_path=f"s3://{sagemaker_session.default_bucket()}/{str(uuid.uuid4())}/tensors",
        )
        profiler_config = ProfilerConfig(
            s3_output_path=f"s3://{sagemaker_session.default_bucket()}/{str(uuid.uuid4())}/system",
            system_monitor_interval_millis=1000,
            framework_profile_params=FrameworkProfile(),
        )

        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_training_latest_version,
            py_version=mxnet_training_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            rules=rules,
            debugger_hook_config=debugger_hook_config,
            profiler_config=profiler_config,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        training_job_name = unique_name_from_base("test-profiler-mxnet-training")
        mx.fit(
            inputs={"train": train_input, "test": test_input},
            job_name=training_job_name,
            wait=False,
        )

        job_description = mx.latest_training_job.describe()
        assert job_description["ProfilerConfig"] == profiler_config._to_request_dict()
        assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()
        assert job_description.get("ProfilingStatus") == "Enabled"

        profiler_rule_configuration = job_description.get("ProfilerRuleConfigurations")[0]
        assert profiler_rule_configuration["RuleConfigurationName"] == "CustomProfilerReportRule"
        assert profiler_rule_configuration["RuleEvaluatorImage"] == mx.rules[0].image_uri
        assert profiler_rule_configuration["RuleParameters"] == {
            "rule_to_invoke": "ProfilerReport",
        }

        for index, rule in enumerate(mx.debugger_rules):
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleConfigurationName"]
                == rule.name
            )
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleEvaluatorImage"]
                == rule.image_uri
            )

        _wait_until_training_can_be_updated(sagemaker_session.sagemaker_client, training_job_name)

        mx.update_profiler(disable_framework_metrics=True)
        job_description = mx.latest_training_job.describe()
        assert job_description["ProfilerConfig"]["ProfilingParameters"] == {}


def test_mxnet_with_enable_framework_metrics_then_update_framework_metrics(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        profiler_config = ProfilerConfig(
            framework_profile_params=FrameworkProfile(start_step=1, num_steps=5)
        )

        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_training_latest_version,
            py_version=mxnet_training_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            profiler_config=profiler_config,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        training_job_name = unique_name_from_base("test-profiler-mxnet-training")
        mx.fit(
            inputs={"train": train_input, "test": test_input},
            job_name=training_job_name,
            wait=False,
        )

        job_description = mx.latest_training_job.describe()
        assert (
            job_description["ProfilerConfig"]["ProfilingParameters"]
            == profiler_config._to_request_dict()["ProfilingParameters"]
        )
        assert job_description.get("ProfilingStatus") == "Enabled"

        _wait_until_training_can_be_updated(sagemaker_session.sagemaker_client, training_job_name)

        updated_framework_profile = FrameworkProfile(
            detailed_profiling_config=DetailedProfilingConfig(profile_default_steps=True)
        )
        mx.update_profiler(framework_profile_params=updated_framework_profile)

        job_description = mx.latest_training_job.describe()
        assert (
            job_description["ProfilerConfig"]["ProfilingParameters"]
            == updated_framework_profile.profiling_parameters
        )


def test_mxnet_with_disable_profiler_then_enable_default_profiling(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_training_latest_version,
            py_version=mxnet_training_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            disable_profiler=True,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        training_job_name = unique_name_from_base("test-profiler-mxnet-training")
        mx.fit(
            inputs={"train": train_input, "test": test_input},
            job_name=training_job_name,
            wait=False,
        )

        job_description = mx.latest_training_job.describe()
        assert job_description.get("ProfilerRuleConfigurations") is None
        assert job_description.get("ProfilingStatus") == "Disabled"

        _wait_until_training_can_be_updated(sagemaker_session.sagemaker_client, training_job_name)
        mx.enable_default_profiling()

        job_description = mx.latest_training_job.describe()
        assert job_description["ProfilerConfig"]["S3OutputPath"] == mx.output_path


def test_mxnet_profiling_with_disable_debugger_hook(
    sagemaker_session,
    mxnet_training_latest_version,
    mxnet_training_latest_py_version,
    cpu_instance_type,
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_training_latest_version,
            py_version=mxnet_training_latest_py_version,
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            debugger_hook_config=False,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        training_job_name = unique_name_from_base("test-profiler-mxnet-training")
        mx.fit(
            inputs={"train": train_input, "test": test_input},
            job_name=training_job_name,
            wait=False,
        )

        job_description = mx.latest_training_job.describe()
        # setting debugger_hook_config to false  would not disable profiling
        # https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-turn-off.html
        assert job_description.get("ProfilingStatus") == "Enabled"
