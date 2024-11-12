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

import pasta

from sagemaker.cli.compatibility.v2.modifiers import airflow
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call

MODEL_CONFIG_CALL_TEMPLATES = (
    "model_config({})",
    "airflow.model_config({})",
    "workflow.airflow.model_config({})",
    "sagemaker.workflow.airflow.model_config({})",
    "model_config_from_estimator({})",
    "airflow.model_config_from_estimator({})",
    "workflow.airflow.model_config_from_estimator({})",
    "sagemaker.workflow.airflow.model_config_from_estimator({})",
)


def test_arg_order_node_should_be_modified_model_config_with_args():
    modifier = airflow.ModelConfigArgModifier()

    for template in MODEL_CONFIG_CALL_TEMPLATES:
        node = ast_call(template.format("instance_type, model"))
        assert modifier.node_should_be_modified(node) is True


def test_arg_order_node_should_be_modified_model_config_without_args():
    modifier = airflow.ModelConfigArgModifier()

    for template in MODEL_CONFIG_CALL_TEMPLATES:
        node = ast_call(template.format(""))
        assert modifier.node_should_be_modified(node) is False


def test_arg_order_node_should_be_modified_random_function_call():
    node = ast_call("sagemaker.workflow.airflow.prepare_framework_container_def()")
    modifier = airflow.ModelConfigArgModifier()
    assert modifier.node_should_be_modified(node) is False


def test_arg_order_modify_node():
    model_config_calls = (
        ("model_config(instance_type, model)", "model_config(model, instance_type=instance_type)"),
        (
            "model_config('ml.m4.xlarge', 'my-model')",
            "model_config('my-model', instance_type='ml.m4.xlarge')",
        ),
        (
            "model_config('ml.m4.xlarge', model='my-model')",
            "model_config(instance_type='ml.m4.xlarge', model='my-model')",
        ),
        (
            "model_config_from_estimator(instance_type, estimator, task_id, task_type)",
            "model_config_from_estimator(estimator, task_id, task_type, instance_type=instance_type)",
        ),
        (
            "model_config_from_estimator(instance_type, estimator, task_id=task_id, task_type=task_type)",
            "model_config_from_estimator(estimator, instance_type=instance_type, task_id=task_id, task_type=task_type)",
        ),
    )

    modifier = airflow.ModelConfigArgModifier()

    for call, expected in model_config_calls:
        node = ast_call(call)
        modifier.modify_node(node)
        assert expected == pasta.dump(node)


def test_image_arg_node_should_be_modified_model_config_with_arg():
    modifier = airflow.ModelConfigImageURIRenamer()

    for template in MODEL_CONFIG_CALL_TEMPLATES:
        node = ast_call(template.format("image=my_image"))
        assert modifier.node_should_be_modified(node) is True


def test_image_arg_node_should_be_modified_model_config_without_arg():
    modifier = airflow.ModelConfigImageURIRenamer()

    for template in MODEL_CONFIG_CALL_TEMPLATES:
        node = ast_call(template.format(""))
        assert modifier.node_should_be_modified(node) is False


def test_image_arg_node_should_be_modified_random_function_call():
    node = ast_call("sagemaker.workflow.airflow.prepare_framework_container_def()")
    modifier = airflow.ModelConfigImageURIRenamer()
    assert modifier.node_should_be_modified(node) is False


def test_image_arg_modify_node():
    model_config_calls = (
        ("model_config(image='image:latest')", "model_config(image_uri='image:latest')"),
        (
            "model_config_from_estimator(image=my_image)",
            "model_config_from_estimator(image_uri=my_image)",
        ),
    )

    modifier = airflow.ModelConfigImageURIRenamer()

    for call, expected in model_config_calls:
        node = ast_call(call)
        modifier.modify_node(node)
        assert expected == pasta.dump(node)
