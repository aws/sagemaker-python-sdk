# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


def test_node_should_be_modified_model_config_with_args():
    model_config_calls = (
        "model_config(instance_type, model)",
        "airflow.model_config(instance_type, model)",
        "workflow.airflow.model_config(instance_type, model)",
        "sagemaker.workflow.airflow.model_config(instance_type, model)",
        "model_config_from_estimator(instance_type, model)",
        "airflow.model_config_from_estimator(instance_type, model)",
        "workflow.airflow.model_config_from_estimator(instance_type, model)",
        "sagemaker.workflow.airflow.model_config_from_estimator(instance_type, model)",
    )

    modifier = airflow.ModelConfigArgModifier()

    for call in model_config_calls:
        node = ast_call(call)
        assert modifier.node_should_be_modified(node) is True


def test_node_should_be_modified_model_config_without_args():
    model_config_calls = (
        "model_config()",
        "airflow.model_config()",
        "workflow.airflow.model_config()",
        "sagemaker.workflow.airflow.model_config()",
        "model_config_from_estimator()",
        "airflow.model_config_from_estimator()",
        "workflow.airflow.model_config_from_estimator()",
        "sagemaker.workflow.airflow.model_config_from_estimator()",
    )

    modifier = airflow.ModelConfigArgModifier()

    for call in model_config_calls:
        node = ast_call(call)
        assert modifier.node_should_be_modified(node) is False


def test_node_should_be_modified_random_function_call():
    node = ast_call("sagemaker.workflow.airflow.prepare_framework_container_def()")
    modifier = airflow.ModelConfigArgModifier()
    assert modifier.node_should_be_modified(node) is False


def test_modify_node():
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
