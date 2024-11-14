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
"""LocalContainer Tests."""
from __future__ import absolute_import
import os
import shutil

import pytest

from sagemaker.modules.configs import Channel, FileSystemDataSource
from sagemaker.modules.local_core.local_container import DOCKER_COMPOSE_FILENAME, _LocalContainer
from sagemaker_core.shapes import DataSource

TRAINING_JOB_NAME = "job_name"
INSTANCE_TYPE = "ml.m5.xlarge"
TEST_IMAGE_NAME = "test_image"
CONTAINER_ROOT = os.getcwd()
CONTAINER_ENTRYPOINT = ["/bin/bash"]
CONTAINER_ARGUMENTS = [
    "-c",
    "chmod +x /opt/ml/input/data/sm_code/train.sh && /opt/ml/input/data/sm_code/train.sh",
]


@pytest.fixture
def input_data_config():
    return [
        Channel(
            channel_name="local_input_channel",
            data_source=DataSource(
                file_system_data_source=FileSystemDataSource.model_construct(
                    directory_path=CONTAINER_ROOT,
                    file_system_type="EFS",
                ),
            ),
            input_mode="File",
        )
    ]


@pytest.fixture
def hyper_parameters():
    return {
        "epochs": "1",
        "optimizer": "adamw_torch",
    }


@pytest.fixture
def shared_volumes():
    return [
        f"{CONTAINER_ROOT}/model:/opt/ml/model",
        f"{CONTAINER_ROOT}:/opt/ml/input/data/local_input_channel",
    ]


@pytest.fixture
def environment():
    return {
        "SM_OUTPUT_DIR": "/opt/ml/output",
        "SM_INPUT_CONFIG_DIR": "/opt/ml/input/config",
        "SM_OUTPUT_DATA_DIR": "/opt/ml/output/data",
    }


@pytest.fixture
def local_container(input_data_config, hyper_parameters, environment):
    container = _LocalContainer(
        training_job_name=TRAINING_JOB_NAME,
        instance_type=INSTANCE_TYPE,
        instance_count=2,
        image=TEST_IMAGE_NAME,
        container_root=CONTAINER_ROOT,
        is_studio=False,
        input_data_config=input_data_config,
        hyper_parameters=hyper_parameters,
        environment=environment,
        sagemaker_session=None,
        container_entrypoint=CONTAINER_ENTRYPOINT,
        container_arguments=CONTAINER_ARGUMENTS,
    )
    return container


def expected_host_config(shared_volumes, host):
    return {
        "entrypoint": [
            "/bin/bash",
            "-c",
            "chmod +x /opt/ml/input/data/sm_code/train.sh && "
            "/opt/ml/input/data/sm_code/train.sh",
        ],
        "environment": [
            "SM_OUTPUT_DIR=/opt/ml/output",
            "SM_INPUT_CONFIG_DIR=/opt/ml/input/config",
            "SM_OUTPUT_DATA_DIR=/opt/ml/output/data",
        ],
        "image": "test_image",
        "networks": {
            "sagemaker-local": {
                "aliases": [
                    host,
                ],
            },
        },
        "volumes": shared_volumes
        + [
            f"{CONTAINER_ROOT}/{host}/output:/opt/ml/output",
            f"{CONTAINER_ROOT}/{host}/output/data:/opt/ml/output/data",
            f"{CONTAINER_ROOT}/{host}/input:/opt/ml/input",
        ],
    }


def expected_compose_file(shared_volumes, hosts):
    return {
        "networks": {
            "sagemaker-local": {
                "name": "sagemaker-local",
            },
        },
        "services": {host: expected_host_config(shared_volumes, host) for host in hosts},
    }


def test_write_config_files(local_container, input_data_config, hyper_parameters):
    config_path = os.path.join(local_container.container_root, "algo-1", "input", "config")
    os.makedirs(config_path, exist_ok=True)
    local_container._write_config_files(
        host="algo-1",
        input_data_config=input_data_config,
        hyper_parameters=hyper_parameters,
    )

    assert os.path.exists(os.path.join(config_path, "hyperparameters.json"))
    assert os.path.exists(os.path.join(config_path, "resourceconfig.json"))
    assert os.path.exists(os.path.join(config_path, "inputdataconfig.json"))

    shutil.rmtree(config_path)


def test_prepare_training_volumes(
    local_container, input_data_config, hyper_parameters, shared_volumes
):
    data_dir = os.path.join(local_container.container_root, "input", "data")
    output = local_container._prepare_training_volumes(
        data_dir, input_data_config, hyper_parameters
    )

    assert output == shared_volumes


def test_create_docker_host(local_container, environment, shared_volumes):
    host = "algo-1"
    output = local_container._create_docker_host(host, environment, shared_volumes)
    assert output == expected_host_config(shared_volumes, host)


def test_generate_compose_file(local_container, environment, shared_volumes):
    output = local_container._generate_compose_file(environment, shared_volumes)

    assert output == expected_compose_file(shared_volumes, local_container.hosts)

    docker_compose_path = os.path.join(local_container.container_root, DOCKER_COMPOSE_FILENAME)
    assert os.path.exists(docker_compose_path)
    os.remove(docker_compose_path)
