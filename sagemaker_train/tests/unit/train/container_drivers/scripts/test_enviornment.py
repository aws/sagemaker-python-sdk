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
"""Enviornment Variable Script Unit Tests."""
from __future__ import absolute_import

import os
import io
import logging

from unittest.mock import patch

from sagemaker.train.container_drivers.scripts.environment import (
    set_env,
    log_env_variables,
    HIDDEN_VALUE,
)
from sagemaker.train.container_drivers.common.utils import safe_serialize, safe_deserialize

RESOURCE_CONFIG = dict(
    current_host="algo-1",
    hosts=["algo-1", "algo-2", "algo-3"],
    current_group_name="train1",
    current_instance_type="ml.p3.16xlarge",
    instance_groups=[
        dict(
            instance_group_name="train1",
            instance_type="ml.p3.16xlarge",
            hosts=["algo-1", "algo-2"],
        ),
        dict(
            instance_group_name="train2",
            instance_type="ml.p3.8xlarge",
            hosts=["algo-3"],
        ),
    ],
    network_interface_name="eth0",
)

INPUT_DATA_CONFIG = {
    "train": {
        "ContentType": "trainingContentType",
        "TrainingInputMode": "File",
        "S3DistributionType": "FullyReplicated",
        "RecordWrapperType": "None",
    },
    "validation": {
        "TrainingInputMode": "File",
        "S3DistributionType": "FullyReplicated",
        "RecordWrapperType": "None",
    },
}

USER_HYPERPARAMETERS = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "hosts": ["algo-1", "algo-2"],
    "mp_parameters": {
        "microbatches": 2,
        "partitions": 2,
        "pipeline": "interleaved",
        "optimize": "memory",
        "horovod": True,
    },
}

SOURCE_CODE = {
    "source_dir": "code",
    "entry_script": "train.py",
}

DISTRIBUTED_CONFIG = {
    "process_count_per_node": 2,
}

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "sm_training.env")

# flake8: noqa
EXPECTED_ENV = """
export SM_MODEL_DIR='/opt/ml/model'
export SM_INPUT_DIR='/opt/ml/input'
export SM_INPUT_DATA_DIR='/opt/ml/input/data'
export SM_INPUT_CONFIG_DIR='/opt/ml/input/config'
export SM_OUTPUT_DIR='/opt/ml/output'
export SM_OUTPUT_FAILURE='/opt/ml/output/failure'
export SM_OUTPUT_DATA_DIR='/opt/ml/output/data'
export SM_LOG_LEVEL='20'
export SM_MASTER_ADDR='algo-1'
export SM_MASTER_PORT='7777'
export SM_SOURCE_DIR='/opt/ml/input/data/code'
export SM_ENTRY_SCRIPT='train.py'
export SM_DISTRIBUTED_DRIVER_DIR='/opt/ml/input/data/sm_drivers/distributed_drivers'
export SM_DISTRIBUTED_CONFIG='{"process_count_per_node": 2}'
export SM_CHANNEL_TRAIN='/opt/ml/input/data/train'
export SM_CHANNEL_VALIDATION='/opt/ml/input/data/validation'
export SM_CHANNELS='["train", "validation"]'
export SM_HP_BATCH_SIZE='32'
export SM_HP_LEARNING_RATE='0.001'
export SM_HP_HOSTS='["algo-1", "algo-2"]'
export SM_HP_MP_PARAMETERS='{"microbatches": 2, "partitions": 2, "pipeline": "interleaved", "optimize": "memory", "horovod": true}'
export SM_HPS='{"batch_size": 32, "learning_rate": 0.001, "hosts": ["algo-1", "algo-2"], "mp_parameters": {"microbatches": 2, "partitions": 2, "pipeline": "interleaved", "optimize": "memory", "horovod": true}}'
export SM_CURRENT_HOST='algo-1'
export SM_CURRENT_INSTANCE_TYPE='ml.p3.16xlarge'
export SM_HOSTS='["algo-1", "algo-2", "algo-3"]'
export SM_NETWORK_INTERFACE_NAME='eth0'
export SM_HOST_COUNT='3'
export SM_CURRENT_HOST_RANK='0'
export SM_NUM_CPUS='8'
export SM_NUM_GPUS='0'
export SM_NUM_NEURONS='0'
export SM_RESOURCE_CONFIG='{"current_host": "algo-1", "hosts": ["algo-1", "algo-2", "algo-3"], "current_group_name": "train1", "current_instance_type": "ml.p3.16xlarge", "instance_groups": [{"instance_group_name": "train1", "instance_type": "ml.p3.16xlarge", "hosts": ["algo-1", "algo-2"]}, {"instance_group_name": "train2", "instance_type": "ml.p3.8xlarge", "hosts": ["algo-3"]}], "network_interface_name": "eth0"}'
export SM_INPUT_DATA_CONFIG='{"train": {"ContentType": "trainingContentType", "TrainingInputMode": "File", "S3DistributionType": "FullyReplicated", "RecordWrapperType": "None"}, "validation": {"TrainingInputMode": "File", "S3DistributionType": "FullyReplicated", "RecordWrapperType": "None"}}'
export SM_TRAINING_ENV='{"channel_input_dirs": {"train": "/opt/ml/input/data/train", "validation": "/opt/ml/input/data/validation"}, "current_host": "algo-1", "current_instance_type": "ml.p3.16xlarge", "hosts": ["algo-1", "algo-2", "algo-3"], "master_addr": "algo-1", "master_port": 7777, "hyperparameters": {"batch_size": 32, "learning_rate": 0.001, "hosts": ["algo-1", "algo-2"], "mp_parameters": {"microbatches": 2, "partitions": 2, "pipeline": "interleaved", "optimize": "memory", "horovod": true}}, "input_data_config": {"train": {"ContentType": "trainingContentType", "TrainingInputMode": "File", "S3DistributionType": "FullyReplicated", "RecordWrapperType": "None"}, "validation": {"TrainingInputMode": "File", "S3DistributionType": "FullyReplicated", "RecordWrapperType": "None"}}, "input_config_dir": "/opt/ml/input/config", "input_data_dir": "/opt/ml/input/data", "input_dir": "/opt/ml/input", "job_name": "test-job", "log_level": 20, "model_dir": "/opt/ml/model", "network_interface_name": "eth0", "num_cpus": 8, "num_gpus": 0, "num_neurons": 0, "output_data_dir": "/opt/ml/output/data", "resource_config": {"current_host": "algo-1", "hosts": ["algo-1", "algo-2", "algo-3"], "current_group_name": "train1", "current_instance_type": "ml.p3.16xlarge", "instance_groups": [{"instance_group_name": "train1", "instance_type": "ml.p3.16xlarge", "hosts": ["algo-1", "algo-2"]}, {"instance_group_name": "train2", "instance_type": "ml.p3.8xlarge", "hosts": ["algo-3"]}], "network_interface_name": "eth0"}}'
"""


@patch(
    "sagemaker.train.container_drivers.scripts.environment.read_source_code_json",
    return_value=SOURCE_CODE,
)
@patch(
    "sagemaker.train.container_drivers.scripts.environment.read_distributed_json",
    return_value=DISTRIBUTED_CONFIG,
)
@patch("sagemaker.train.container_drivers.scripts.environment.num_cpus", return_value=8)
@patch("sagemaker.train.container_drivers.scripts.environment.num_gpus", return_value=0)
@patch("sagemaker.train.container_drivers.scripts.environment.num_neurons", return_value=0)
@patch(
    "sagemaker.train.container_drivers.scripts.environment.safe_serialize",
    side_effect=safe_serialize,
)
@patch(
    "sagemaker.train.container_drivers.scripts.environment.safe_deserialize",
    side_effect=safe_deserialize,
)
def test_set_env(
    mock_safe_deserialize,
    mock_safe_serialize,
    mock_num_neurons,
    mock_num_gpus,
    mock_num_cpus,
    mock_read_distributed_json,
    mock_read_source_code_json,
):
    with patch.dict(os.environ, {"TRAINING_JOB_NAME": "test-job"}):
        set_env(
            resource_config=RESOURCE_CONFIG,
            input_data_config=INPUT_DATA_CONFIG,
            hyperparameters_config=USER_HYPERPARAMETERS,
            output_file=OUTPUT_FILE,
        )

        mock_num_cpus.assert_called_once()
        mock_num_gpus.assert_called_once()
        mock_num_neurons.assert_called_once()
        mock_read_distributed_json.assert_called_once()
        mock_read_source_code_json.assert_called_once()

        with open(OUTPUT_FILE, "r") as f:
            env_file = f.read().strip()
            expected_env = _remove_extra_lines(EXPECTED_ENV)
            env_file = _remove_extra_lines(env_file)

            assert env_file == expected_env
        os.remove(OUTPUT_FILE)
        assert not os.path.exists(OUTPUT_FILE)


@patch.dict(os.environ, {"SECRET_TOKEN": "122345678", "CLEAR_DATA": "123456789"}, clear=True)
def test_log_env_variables():
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)

    logger = logging.getLogger("sagemaker.train.container_drivers.scripts.environment")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    env_vars = {
        "SM_MODEL_DIR": "/opt/ml/model",
        "SM_INPUT_DIR": "/opt/ml/input",
        "SM_HPS": {"batch_size": 32, "learning_rate": 0.001, "access_token": "123456789"},
        "SM_HP_BATCH_SIZE": 32,
        "SM_HP_LEARNING_RATE": 0.001,
        "SM_HP_ACCESS_TOKEN": "123456789",
    }
    log_env_variables(env_vars_dict=env_vars)

    log_output = log_stream.getvalue()

    assert f"SECRET_TOKEN={HIDDEN_VALUE}" in log_output
    assert "CLEAR_DATA=123456789" in log_output
    assert "SM_MODEL_DIR=/opt/ml/model" in log_output
    assert (
        f'SM_HPS={{"batch_size": 32, "learning_rate": 0.001, "access_token": "{HIDDEN_VALUE}"}}'
        in log_output
    )
    assert "SM_HP_BATCH_SIZE=32" in log_output
    assert "SM_HP_LEARNING_RATE=0.001" in log_output
    assert f"SM_HP_ACCESS_TOKEN={HIDDEN_VALUE}" in log_output


def _remove_extra_lines(string):
    """Removes extra blank lines from a string."""
    return "\n".join([line for line in string.splitlines() if line.strip()])
