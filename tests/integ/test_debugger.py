# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# import logging
# import os
# import uuid
#
# from sagemaker.debugger import Rule, rule_configs, DebuggerHookConfig, TensorBoardOutputConfig
# from sagemaker.mxnet.estimator import MXNet
# from tests.integ import DATA_DIR, PYTHON_VERSION, TRAINING_DEFAULT_TIMEOUT_MINUTES
# from tests.integ.timeout import timeout
#
#
# # TODO-reinvent-2019 [akarpur]: uncomment this test once API changes are in Prod
# def test_mxnet_with_rules(sagemaker_session, mxnet_full_version, cpu_instance_type):
#     logging.getLogger().setLevel(logging.DEBUG)  # TODO-reinvent-2019: REMOVE THIS
#
#     with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
#         rules = [
#             Rule.sagemaker(rule_configs.vanishing_gradient()),
#             Rule.sagemaker(rule_configs.all_zero()),
#             Rule.sagemaker(rule_configs.loss_not_decreasing()),
#         ]
#
#         script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
#         data_path = os.path.join(DATA_DIR, "mxnet_mnist")
#
#         mx = MXNet(
#             entry_point=script_path,
#             role="SageMakerRole",
#             framework_version=mxnet_full_version,
#             py_version=PYTHON_VERSION,
#             train_instance_count=1,
#             train_instance_type=cpu_instance_type,
#             sagemaker_session=sagemaker_session,
#             rules=rules,
#         )
#
#         train_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
#         )
#         test_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
#         )
#
#         mx.fit({"train": train_input, "test": test_input})
#
#         job_description = mx.latest_training_job.describe()
#
#         for index, rule in enumerate(rules):
#             assert (
#                 job_description["DebugRuleConfigurations"][index]
#                 == rule.to_debugger_rule_config_dict()
#             )
#             # TODO-reinvent-2019 [akarpur]: uncomment this assert statement once the service
#             #  is stable and Rules can successfully complete
#             # assert job_description["DebugRuleEvaluationStatuses"][index][
#             #     "RuleEvaluationStatus"
#             # ] not in {"NotStarted", "Error"}
#         assert (
#             job_description["DebugRuleEvaluationStatuses"]
#             == mx.latest_training_job.rule_job_summary()
#         )
#
#
# # TODO-reinvent-2019 [akarpur]: uncomment this test once API changes are in Prod
# def test_mxnet_with_custom_rule(sagemaker_session, mxnet_full_version, cpu_instance_type):
#     logging.getLogger().setLevel(logging.DEBUG)  # TODO-reinvent-2019: REMOVE
#
#     with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
#         rules = [
#             Rule.custom(
#                 name="test-custom-rule",
#                 instance_type="t2.micro",
#                 image_uri="453379255795.dkr.ecr.{}.amazonaws.com/script-rule-executor:latest".format(
#                     sagemaker_session.boto_region_name
#                 ),
#             )
#         ]
#
#         script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
#         data_path = os.path.join(DATA_DIR, "mxnet_mnist")
#
#         mx = MXNet(
#             entry_point=script_path,
#             role="SageMakerRole",
#             framework_version=mxnet_full_version,
#             py_version=PYTHON_VERSION,
#             train_instance_count=1,
#             train_instance_type=cpu_instance_type,
#             sagemaker_session=sagemaker_session,
#             rules=rules,
#         )
#
#         train_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
#         )
#         test_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
#         )
#
#         mx.fit({"train": train_input, "test": test_input})
#
#         job_description = mx.latest_training_job.describe()
#
#         for index, rule in enumerate(rules):
#             assert (
#                 job_description["DebugRuleConfigurations"][index]
#                 == rule.to_debugger_rule_config_dict()
#             )
#             # TODO-reinvent-2019 [akarpur]: uncomment this assert statement once the service
#             #  is stable and Rules can successfully complete
#             # assert job_description["DebugRuleEvaluationStatuses"][index][
#             #     "RuleEvaluationStatus"
#             # ] not in {"NotStarted", "Error"}
#         assert (
#             job_description["DebugRuleEvaluationStatuses"]
#             == mx.latest_training_job.rule_job_summary()
#         )
#
#
# # TODO-reinvent-2019 [akarpur]: uncomment this test once API changes are in Prod
# def test_mxnet_with_debugger_hook_config(sagemaker_session, mxnet_full_version, cpu_instance_type):
#     logging.getLogger().setLevel(logging.DEBUG)  # TODO-reinvent-2019: REMOVE
#
#     with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
#         debugger_hook_config = DebuggerHookConfig(
#             s3_output_path=os.path.join(
#                 "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensors"
#             )
#         )
#
#         script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
#         data_path = os.path.join(DATA_DIR, "mxnet_mnist")
#
#         mx = MXNet(
#             entry_point=script_path,
#             role="SageMakerRole",
#             framework_version=mxnet_full_version,
#             py_version=PYTHON_VERSION,
#             train_instance_count=1,
#             train_instance_type=cpu_instance_type,
#             sagemaker_session=sagemaker_session,
#             debugger_hook_config=debugger_hook_config,
#         )
#
#         train_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
#         )
#         test_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
#         )
#
#         mx.fit({"train": train_input, "test": test_input})
#
#         job_description = mx.latest_training_job.describe()
#         assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()
#
#
# # TODO-reinvent-2019 [akarpur]: uncomment this test once API changes are in Prod
# def test_mxnet_with_rules_and_debugger_hook_config(
#     sagemaker_session, mxnet_full_version, cpu_instance_type
# ):
#     logging.getLogger().setLevel(logging.DEBUG)  # TODO-reinvent-2019: REMOVE
#
#     with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
#         rules = [
#             Rule.sagemaker(rule_configs.vanishing_gradient()),
#             Rule.sagemaker(rule_configs.all_zero()),
#             Rule.sagemaker(rule_configs.loss_not_decreasing()),
#         ]
#         debugger_hook_config = DebuggerHookConfig(
#             s3_output_path=os.path.join(
#                 "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensors"
#             )
#         )
#
#         script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
#         data_path = os.path.join(DATA_DIR, "mxnet_mnist")
#
#         mx = MXNet(
#             entry_point=script_path,
#             role="SageMakerRole",
#             framework_version=mxnet_full_version,
#             py_version=PYTHON_VERSION,
#             train_instance_count=1,
#             train_instance_type=cpu_instance_type,
#             sagemaker_session=sagemaker_session,
#             rules=rules,
#             debugger_hook_config=debugger_hook_config,
#         )
#
#         train_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
#         )
#         test_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
#         )
#
#         mx.fit({"train": train_input, "test": test_input})
#
#         job_description = mx.latest_training_job.describe()
#
#         for index, rule in enumerate(rules):
#             assert (
#                 job_description["DebugRuleConfigurations"][index]
#                 == rule.to_debugger_rule_config_dict()
#             )
#             # TODO-reinvent-2019 [akarpur]: uncomment this assert statement once the service
#             #  is stable and Rules can successfully complete
#             # assert job_description["DebugRuleEvaluationStatuses"][index][
#             #     "RuleEvaluationStatus"
#             # ] not in {"NotStarted", "Error"}
#         assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()
#         assert (
#             job_description["DebugRuleEvaluationStatuses"]
#             == mx.latest_training_job.rule_job_summary()
#         )
#
#
# # TODO-reinvent-2019 [akarpur]: uncomment this test once API changes are in Prod
# def test_mxnet_with_custom_rule_and_debugger_hook_config(
#     sagemaker_session, mxnet_full_version, cpu_instance_type
# ):
#     logging.getLogger().setLevel(logging.DEBUG)  # TODO-reinvent-2019: REMOVE
#
#     with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
#         rules = [
#             Rule.custom(
#                 name="test-custom-rule",
#                 instance_type="t2.micro",
#                 image_uri="453379255795.dkr.ecr.{}.amazonaws.com/script-rule-executor:latest".format(
#                     sagemaker_session.boto_region_name
#                 ),
#             )
#         ]
#         debugger_hook_config = DebuggerHookConfig(
#             s3_output_path=os.path.join(
#                 "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensors"
#             )
#         )
#
#         script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
#         data_path = os.path.join(DATA_DIR, "mxnet_mnist")
#
#         mx = MXNet(
#             entry_point=script_path,
#             role="SageMakerRole",
#             framework_version=mxnet_full_version,
#             py_version=PYTHON_VERSION,
#             train_instance_count=1,
#             train_instance_type=cpu_instance_type,
#             sagemaker_session=sagemaker_session,
#             rules=rules,
#             debugger_hook_config=debugger_hook_config,
#         )
#
#         train_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
#         )
#         test_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
#         )
#
#         mx.fit({"train": train_input, "test": test_input})
#
#         job_description = mx.latest_training_job.describe()
#
#         for index, rule in enumerate(rules):
#             assert (
#                 job_description["DebugRuleConfigurations"][index]
#                 == rule.to_debugger_rule_config_dict()
#             )
#             # TODO-reinvent-2019 [akarpur]: uncomment this assert statement once the service
#             #  is stable and Rules can successfully complete
#             # assert job_description["DebugRuleEvaluationStatuses"][index][
#             #     "RuleEvaluationStatus"
#             # ] not in {"NotStarted", "Error"}
#         assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()
#         assert (
#             job_description["DebugRuleEvaluationStatuses"]
#             == mx.latest_training_job.rule_job_summary()
#         )
#
#
# # TODO-reinvent-2019 [akarpur]: uncomment this test once API changes are in Prod
# def test_mxnet_with_tensorboard_output_config(
#     sagemaker_session, mxnet_full_version, cpu_instance_type
# ):
#     logging.getLogger().setLevel(logging.DEBUG)  # TODO-reinvent-2019: REMOVE
#
#     with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
#         tensorboard_output_config = TensorBoardOutputConfig(
#             s3_output_path=os.path.join(
#                 "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensorboard"
#             )
#         )
#
#         script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
#         data_path = os.path.join(DATA_DIR, "mxnet_mnist")
#
#         mx = MXNet(
#             entry_point=script_path,
#             role="SageMakerRole",
#             framework_version=mxnet_full_version,
#             py_version=PYTHON_VERSION,
#             train_instance_count=1,
#             train_instance_type=cpu_instance_type,
#             sagemaker_session=sagemaker_session,
#             tensorboard_output_config=tensorboard_output_config,
#         )
#
#         train_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
#         )
#         test_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
#         )
#
#         mx.fit({"train": train_input, "test": test_input})
#
#         job_description = mx.latest_training_job.describe()
#         assert (
#             job_description["TensorBoardOutputConfig"]
#             == tensorboard_output_config._to_request_dict()
#         )
#
#
# # TODO-reinvent-2019 [akarpur]: uncomment this test once API changes are in Prod
# def test_mxnet_with_all_rules_and_configs(sagemaker_session, mxnet_full_version, cpu_instance_type):
#     logging.getLogger().setLevel(logging.DEBUG)  # TODO-reinvent-2019: REMOVE
#
#     with timeout(minutes=TRAINING_DEFAULT_TIMEOUT_MINUTES):
#         rules = [
#             Rule.sagemaker(rule_configs.vanishing_gradient()),
#             Rule.sagemaker(rule_configs.all_zero()),
#             Rule.sagemaker(rule_configs.similar_across_runs()),
#             Rule.sagemaker(rule_configs.weight_update_ratio()),
#             Rule.sagemaker(rule_configs.exploding_tensor()),
#             Rule.sagemaker(rule_configs.unchanged_tensor()),
#             Rule.sagemaker(rule_configs.loss_not_decreasing()),
#             Rule.sagemaker(rule_configs.check_input_images()),
#             Rule.sagemaker(rule_configs.dead_relu()),
#             Rule.sagemaker(rule_configs.confusion()),
#             Rule.custom(
#                 name="test-custom-rule",
#                 instance_type="t2.micro",
#                 image_uri="453379255795.dkr.ecr.{}.amazonaws.com/script-rule-executor:latest".format(
#                     sagemaker_session.boto_region_name
#                 ),
#             ),
#         ]
#         debugger_hook_config = DebuggerHookConfig(
#             s3_output_path=os.path.join(
#                 "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensors"
#             )
#         )
#         tensorboard_output_config = TensorBoardOutputConfig(
#             s3_output_path=os.path.join(
#                 "s3://", sagemaker_session.default_bucket(), str(uuid.uuid4()), "tensorboard"
#             )
#         )
#
#         script_path = os.path.join(DATA_DIR, "mxnet_mnist", "mnist.py")
#         data_path = os.path.join(DATA_DIR, "mxnet_mnist")
#
#         mx = MXNet(
#             entry_point=script_path,
#             role="SageMakerRole",
#             framework_version=mxnet_full_version,
#             py_version=PYTHON_VERSION,
#             train_instance_count=1,
#             train_instance_type=cpu_instance_type,
#             sagemaker_session=sagemaker_session,
#             rules=rules,
#             debugger_hook_config=debugger_hook_config,
#             tensorboard_output_config=tensorboard_output_config,
#         )
#
#         train_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "train"), key_prefix="integ-test-data/mxnet_mnist/train"
#         )
#         test_input = mx.sagemaker_session.upload_data(
#             path=os.path.join(data_path, "test"), key_prefix="integ-test-data/mxnet_mnist/test"
#         )
#
#         mx.fit({"train": train_input, "test": test_input})
#
#         job_description = mx.latest_training_job.describe()
#
#         for index, rule in enumerate(rules):
#             assert (
#                 job_description["DebugRuleConfigurations"][index]
#                 == rule.to_debugger_rule_config_dict()
#             )
#             # TODO-reinvent-2019 [akarpur]: uncomment this assert statement once the service
#             #  is stable and Rules can successfully complete
#             # assert job_description["DebugRuleEvaluationStatuses"][index][
#             #     "RuleEvaluationStatus"
#             # ] not in {"NotStarted", "Error"}
#         assert job_description["DebugHookConfig"] == debugger_hook_config._to_request_dict()
#         assert (
#             job_description["TensorBoardOutputConfig"]
#             == tensorboard_output_config._to_request_dict()
#         )
#         assert (
#             job_description["DebugRuleEvaluationStatuses"]
#             == mx.latest_training_job.rule_job_summary()
#         )
