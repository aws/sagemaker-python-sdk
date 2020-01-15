# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import uuid

import pytest

from sagemaker.debugger import Rule
from sagemaker.debugger import DebuggerHookConfig
from sagemaker.debugger import TensorBoardOutputConfig

from sagemaker.debugger import rule_configs
from sagemaker.mxnet.estimator import MXNet
from tests.integ import DATA_DIR, PYTHON_VERSION, TRAINING_DEFAULT_TIMEOUT_MINUTES
from tests.integ.retry import retries
from tests.integ.timeout import timeout


_NON_ERROR_TERMINAL_RULE_JOB_STATUSES = ["NoIssuesFound", "IssuesFound", "Stopped"]

CUSTOM_RULE_REPO_WITH_PLACEHOLDERS = (
    "{}.dkr.ecr.{}.amazonaws.com/sagemaker-debugger-rule-evaluator:latest"
)

CUSTOM_RULE_CONTAINERS_ACCOUNTS_MAP = {
    "ap-east-1": "645844755771",
    "ap-northeast-1": "670969264625",
    "ap-northeast-2": "326368420253",
    "ap-south-1": "552407032007",
    "ap-southeast-1": "631532610101",
    "ap-southeast-2": "445670767460",
    "ca-central-1": "105842248657",
    "eu-central-1": "691764027602",
    "eu-north-1": "091235270104",
    "eu-west-1": "606966180310",
    "eu-west-2": "074613877050",
    "eu-west-3": "224335253976",
    "me-south-1": "050406412588",
    "sa-east-1": "466516958431",
    "us-east-1": "864354269164",
    "us-east-2": "840043622174",
    "us-west-1": "952348334681",
    "us-west-2": "759209512951",
}

# TODO-reinvent-2019: test get_debugger_artifacts_path and get_tensorboard_artifacts_path


def test_mxnet_with_rules(sagemaker_session, mxnet_full_version, cpu_instance_type):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        rules = [
            Rule.sagemaker(rule_configs.vanishing_gradient()),
            Rule.sagemaker(
                base_config=rule_configs.all_zero(), rule_parameters={"tensor_regex": ".*"}
            ),
            Rule.sagemaker(rule_configs.loss_not_decreasing()),
        ]

        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_full_version,
            py_version=PYTHON_VERSION,
            train_instance_count=1,
            train_instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            rules=rules,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input})

        job_description = mx.latest_training_job.describe()

        for index, rule in enumerate(rules):
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleConfigurationName"]
                == rule.name
            )
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleEvaluatorImage"]
                == rule.image_uri
            )
            assert job_description["DebugRuleConfigurations"][index]["VolumeSizeInGB"] == 0
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleParameters"][
                    "rule_to_invoke"
                ]
                == rule.rule_parameters["rule_to_invoke"]
            )
        assert (
            job_description["DebugRuleEvaluationStatuses"]
            == mx.latest_training_job.rule_job_summary()
        )

        _wait_and_assert_that_no_rule_jobs_errored(training_job=mx.latest_training_job)


def test_mxnet_with_custom_rule(sagemaker_session, mxnet_full_version, cpu_instance_type):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        rules = [_get_custom_rule(sagemaker_session)]

        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_full_version,
            py_version=PYTHON_VERSION,
            train_instance_count=1,
            train_instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            rules=rules,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input})

        job_description = mx.latest_training_job.describe()

        for index, rule in enumerate(rules):
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleConfigurationName"]
                == rule.name
            )
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleEvaluatorImage"]
                == rule.image_uri
            )
            assert job_description["DebugRuleConfigurations"][index]["VolumeSizeInGB"] == 30
        assert (
            job_description["DebugRuleEvaluationStatuses"]
            == mx.latest_training_job.rule_job_summary()
        )

        _wait_and_assert_that_no_rule_jobs_errored(training_job=mx.latest_training_job)


def test_mxnet_with_debugger_hook_config(sagemaker_session, mxnet_full_version, cpu_instance_type):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        debugger_hook_config = DebuggerHookConfig(
            s3_output_path=os.path.join(
                "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensors"
            )
        )

        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_full_version,
            py_version=PYTHON_VERSION,
            train_instance_count=1,
            train_instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            debugger_hook_config=debugger_hook_config,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input})

        job_description = mx.latest_training_job.describe()
        assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()

        _wait_and_assert_that_no_rule_jobs_errored(training_job=mx.latest_training_job)


def test_mxnet_with_rules_and_debugger_hook_config(
    sagemaker_session, mxnet_full_version, cpu_instance_type
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        rules = [
            Rule.sagemaker(rule_configs.vanishing_gradient()),
            Rule.sagemaker(
                base_config=rule_configs.all_zero(), rule_parameters={"tensor_regex": ".*"}
            ),
            Rule.sagemaker(rule_configs.loss_not_decreasing()),
        ]
        debugger_hook_config = DebuggerHookConfig(
            s3_output_path=os.path.join(
                "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensors"
            )
        )

        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_full_version,
            py_version=PYTHON_VERSION,
            train_instance_count=1,
            train_instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            rules=rules,
            debugger_hook_config=debugger_hook_config,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input})

        job_description = mx.latest_training_job.describe()

        for index, rule in enumerate(rules):
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleConfigurationName"]
                == rule.name
            )
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleEvaluatorImage"]
                == rule.image_uri
            )
            assert job_description["DebugRuleConfigurations"][index]["VolumeSizeInGB"] == 0
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleParameters"][
                    "rule_to_invoke"
                ]
                == rule.rule_parameters["rule_to_invoke"]
            )
        assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()
        assert (
            job_description["DebugRuleEvaluationStatuses"]
            == mx.latest_training_job.rule_job_summary()
        )

        _wait_and_assert_that_no_rule_jobs_errored(training_job=mx.latest_training_job)


def test_mxnet_with_custom_rule_and_debugger_hook_config(
    sagemaker_session, mxnet_full_version, cpu_instance_type
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        rules = [_get_custom_rule(sagemaker_session)]
        debugger_hook_config = DebuggerHookConfig(
            s3_output_path=os.path.join(
                "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensors"
            )
        )

        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_full_version,
            py_version=PYTHON_VERSION,
            train_instance_count=1,
            train_instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            rules=rules,
            debugger_hook_config=debugger_hook_config,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input})

        job_description = mx.latest_training_job.describe()

        for index, rule in enumerate(rules):
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleConfigurationName"]
                == rule.name
            )
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleEvaluatorImage"]
                == rule.image_uri
            )
            assert job_description["DebugRuleConfigurations"][index]["VolumeSizeInGB"] == 30
        assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()
        assert (
            job_description["DebugRuleEvaluationStatuses"]
            == mx.latest_training_job.rule_job_summary()
        )

        _wait_and_assert_that_no_rule_jobs_errored(training_job=mx.latest_training_job)


def test_mxnet_with_tensorboard_output_config(
    sagemaker_session, mxnet_full_version, cpu_instance_type
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        tensorboard_output_config = TensorBoardOutputConfig(
            s3_output_path=os.path.join(
                "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensorboard"
            )
        )

        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_full_version,
            py_version=PYTHON_VERSION,
            train_instance_count=1,
            train_instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            tensorboard_output_config=tensorboard_output_config,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input})

        job_description = mx.latest_training_job.describe()
        assert (
            job_description["TensorBoardOutputConfig"]
            == tensorboard_output_config._to_request_dict()
        )

        _wait_and_assert_that_no_rule_jobs_errored(training_job=mx.latest_training_job)


@pytest.mark.canary_quick
def test_mxnet_with_all_rules_and_configs(sagemaker_session, mxnet_full_version, cpu_instance_type):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        rules = [
            Rule.sagemaker(rule_configs.vanishing_gradient()),
            Rule.sagemaker(
                base_config=rule_configs.all_zero(), rule_parameters={"tensor_regex": ".*"}
            ),
            Rule.sagemaker(rule_configs.loss_not_decreasing()),
            _get_custom_rule(sagemaker_session),
        ]
        debugger_hook_config = DebuggerHookConfig(
            s3_output_path=os.path.join(
                "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensors"
            )
        )
        tensorboard_output_config = TensorBoardOutputConfig(
            s3_output_path=os.path.join(
                "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensorboard"
            )
        )

        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_full_version,
            py_version=PYTHON_VERSION,
            train_instance_count=1,
            train_instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            rules=rules,
            debugger_hook_config=debugger_hook_config,
            tensorboard_output_config=tensorboard_output_config,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input})

        job_description = mx.latest_training_job.describe()

        for index, rule in enumerate(rules):
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleConfigurationName"]
                == rule.name
            )
            assert (
                job_description["DebugRuleConfigurations"][index]["RuleEvaluatorImage"]
                == rule.image_uri
            )
        assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()
        assert (
            job_description["TensorBoardOutputConfig"]
            == tensorboard_output_config._to_request_dict()
        )
        assert (
            job_description["DebugRuleEvaluationStatuses"]
            == mx.latest_training_job.rule_job_summary()
        )

        _wait_and_assert_that_no_rule_jobs_errored(training_job=mx.latest_training_job)


def test_mxnet_with_debugger_hook_config_disabled(
    sagemaker_session, mxnet_full_version, cpu_instance_type
):
    with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
        script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist_gluon.py")
        data_path = os.path.join(DATA_DIR, "mxnet_mnist")

        mx = MXNet(
            entry_point=script_path,
            role="SageMakerRole",
            framework_version=mxnet_full_version,
            py_version=PYTHON_VERSION,
            train_instance_count=1,
            train_instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            debugger_hook_config=False,
        )

        train_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
        )
        test_input = mx.sagemaker_session.upload_data(
            path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
        )

        mx.fit({"train": train_input, "test": test_input})

        job_description = mx.latest_training_job.describe()

        assert job_description.get("DebugHookConfig") is None


def _get_custom_rule(session):
    script_path = os.path.join(DATA_DIR, "mxnet_mnist", "my_custom_rule.py")

    return Rule.custom(
        name="test-custom-rule",
        source=script_path,
        rule_to_invoke="CustomGradientRule",
        instance_type="ml.m5.xlarge",
        volume_size_in_gb=30,
        image_uri=CUSTOM_RULE_REPO_WITH_PLACEHOLDERS.format(
            CUSTOM_RULE_CONTAINERS_ACCOUNTS_MAP[session.boto_region_name], session.boto_region_name
        ),
    )


def _wait_and_assert_that_no_rule_jobs_errored(training_job):
    # Wait for all rule jobs to complete.
    # Training job completion takes takes ~5min after training job ends
    # 120 retries * 10s sleeps = 20min timeout
    for _ in retries(
        max_retry_count=120,
        exception_message_prefix="Waiting for all jobs to be in success status or any to be in error",
        seconds_to_sleep=10,
    ):
        job_description = training_job.describe()
        debug_rule_evaluation_statuses = job_description.get("DebugRuleEvaluationStatuses")
        if not debug_rule_evaluation_statuses:
            break
        incomplete_rule_job_found = False
        for debug_rule_evaluation_status in debug_rule_evaluation_statuses:
            assert debug_rule_evaluation_status["RuleEvaluationStatus"] != "Error"
            if (
                debug_rule_evaluation_status["RuleEvaluationStatus"]
                not in _NON_ERROR_TERMINAL_RULE_JOB_STATUSES
            ):
                incomplete_rule_job_found = True
        if not incomplete_rule_job_found:
            break
