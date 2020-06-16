# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import random
import string

from botocore.credentials import Credentials

import base64
import json
import os
import subprocess
import tarfile

import pytest
import yaml
from mock import patch, Mock, MagicMock

import sagemaker
from sagemaker.local.image import _SageMakerContainer, _aws_credentials

REGION = "us-west-2"
BUCKET_NAME = "mybucket"
EXPANDED_ROLE = "arn:aws:iam::111111111111:role/ExpandedRole"
TRAINING_JOB_NAME = "my-job"
INPUT_DATA_CONFIG = [
    {
        "ChannelName": "a",
        "DataUri": "file:///tmp/source1",
        "DataSource": {
            "FileDataSource": {
                "FileDataDistributionType": "FullyReplicated",
                "FileUri": "file:///tmp/source1",
            }
        },
    },
    {
        "ChannelName": "b",
        "DataUri": "s3://my-own-bucket/prefix",
        "DataSource": {
            "S3DataSource": {
                "S3DataDistributionType": "FullyReplicated",
                "S3DataType": "S3Prefix",
                "S3Uri": "s3://my-own-bucket/prefix",
            }
        },
    },
]

OUTPUT_DATA_CONFIG = {"S3OutputPath": ""}

HYPERPARAMETERS = {
    "a": 1,
    "b": json.dumps("bee"),
    "sagemaker_submit_directory": json.dumps("s3://my_bucket/code"),
}

LOCAL_CODE_HYPERPARAMETERS = {
    "a": 1,
    "b": 2,
    "sagemaker_submit_directory": json.dumps("file:///tmp/code"),
}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    boto_mock.client("sts").get_caller_identity.return_value = {"Account": "123"}
    boto_mock.resource("s3").Bucket(BUCKET_NAME).objects.filter.return_value = []

    sms = sagemaker.Session(boto_session=boto_mock, sagemaker_client=Mock())

    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    sms.expand_role = Mock(return_value=EXPANDED_ROLE)

    return sms


def test_sagemaker_container_hosts_should_have_lowercase_names():
    random.seed(a=42)

    def assert_all_lowercase(hosts):
        for host in hosts:
            assert host.lower() == host

    sagemaker_container = _SageMakerContainer("local", 2, "my-image", sagemaker_session=Mock())
    assert_all_lowercase(sagemaker_container.hosts)

    sagemaker_container = _SageMakerContainer("local", 10, "my-image", sagemaker_session=Mock())
    assert_all_lowercase(sagemaker_container.hosts)

    sagemaker_container = _SageMakerContainer("local", 1, "my-image", sagemaker_session=Mock())
    assert_all_lowercase(sagemaker_container.hosts)


@patch("sagemaker.local.local_session.LocalSession")
def test_write_config_file(LocalSession, tmpdir):
    sagemaker_container = _SageMakerContainer("local", 2, "my-image")
    sagemaker_container.container_root = str(tmpdir.mkdir("container-root"))
    host = "algo-1"

    sagemaker.local.image._create_config_file_directories(sagemaker_container.container_root, host)

    container_root = sagemaker_container.container_root
    config_file_root = os.path.join(container_root, host, "input", "config")

    hyperparameters_file = os.path.join(config_file_root, "hyperparameters.json")
    resource_config_file = os.path.join(config_file_root, "resourceconfig.json")
    input_data_config_file = os.path.join(config_file_root, "inputdataconfig.json")

    # write the config files, and then lets check they exist and have the right content.
    sagemaker_container.write_config_files(host, HYPERPARAMETERS, INPUT_DATA_CONFIG)

    assert os.path.exists(hyperparameters_file)
    assert os.path.exists(resource_config_file)
    assert os.path.exists(input_data_config_file)

    hyperparameters_data = json.load(open(hyperparameters_file))
    resource_config_data = json.load(open(resource_config_file))
    input_data_config_data = json.load(open(input_data_config_file))

    # Validate HyperParameters
    for k, v in HYPERPARAMETERS.items():
        assert k in hyperparameters_data
        assert hyperparameters_data[k] == v

    # Validate Resource Config
    assert resource_config_data["current_host"] == host
    assert resource_config_data["hosts"] == sagemaker_container.hosts

    # Validate Input Data Config
    for channel in INPUT_DATA_CONFIG:
        assert channel["ChannelName"] in input_data_config_data


@patch("sagemaker.local.local_session.LocalSession")
def test_write_config_files_input_content_type(LocalSession, tmpdir):
    sagemaker_container = _SageMakerContainer("local", 1, "my-image")
    sagemaker_container.container_root = str(tmpdir.mkdir("container-root"))
    host = "algo-1"

    sagemaker.local.image._create_config_file_directories(sagemaker_container.container_root, host)

    container_root = sagemaker_container.container_root
    config_file_root = os.path.join(container_root, host, "input", "config")

    input_data_config_file = os.path.join(config_file_root, "inputdataconfig.json")

    # write the config files, and then lets check they exist and have the right content.
    input_data_config = [
        {
            "ChannelName": "channel_a",
            "DataUri": "file:///tmp/source1",
            "ContentType": "text/csv",
            "DataSource": {
                "FileDataSource": {
                    "FileDataDistributionType": "FullyReplicated",
                    "FileUri": "file:///tmp/source1",
                }
            },
        },
        {
            "ChannelName": "channel_b",
            "DataUri": "s3://my-own-bucket/prefix",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://my-own-bucket/prefix",
                }
            },
        },
    ]
    sagemaker_container.write_config_files(host, HYPERPARAMETERS, input_data_config)

    assert os.path.exists(input_data_config_file)
    parsed_input_config = json.load(open(input_data_config_file))
    # Validate Input Data Config
    for channel in input_data_config:
        assert channel["ChannelName"] in parsed_input_config

    # Channel A has a content type
    assert "ContentType" in parsed_input_config["channel_a"]
    assert parsed_input_config["channel_a"]["ContentType"] == "text/csv"

    # Channel B does not have content type
    assert "ContentType" not in parsed_input_config["channel_b"]


@patch("sagemaker.local.local_session.LocalSession")
def test_retrieve_artifacts(LocalSession, tmpdir):
    sagemaker_container = _SageMakerContainer("local", 2, "my-image")
    sagemaker_container.hosts = ["algo-1", "algo-2"]  # avoid any randomness
    sagemaker_container.container_root = str(tmpdir.mkdir("container-root"))

    volume1 = os.path.join(sagemaker_container.container_root, "algo-1")
    volume2 = os.path.join(sagemaker_container.container_root, "algo-2")
    os.mkdir(volume1)
    os.mkdir(volume2)

    compose_data = {
        "services": {
            "algo-1": {
                "volumes": [
                    "%s:/opt/ml/model" % os.path.join(volume1, "model"),
                    "%s:/opt/ml/output" % os.path.join(volume1, "output"),
                ]
            },
            "algo-2": {
                "volumes": [
                    "%s:/opt/ml/model" % os.path.join(volume2, "model"),
                    "%s:/opt/ml/output" % os.path.join(volume2, "output"),
                ]
            },
        }
    }

    dirs = [
        ("model", volume1),
        ("model/data", volume1),
        ("model", volume2),
        ("model/data", volume2),
        ("model/tmp", volume2),
        ("output", volume1),
        ("output/data", volume1),
        ("output", volume2),
        ("output/data", volume2),
        ("output/log", volume2),
    ]

    files = [
        ("model/data/model.json", volume1),
        ("model/data/variables.csv", volume1),
        ("model/data/model.json", volume2),
        ("model/data/variables2.csv", volume2),
        ("model/tmp/something-else.json", volume2),
        ("output/data/loss.json", volume1),
        ("output/data/accuracy.json", volume1),
        ("output/data/loss.json", volume2),
        ("output/data/accuracy2.json", volume2),
        ("output/log/warnings.txt", volume2),
    ]

    expected_model = [
        "data",
        "data/model.json",
        "data/variables.csv",
        "data/variables2.csv",
        "tmp/something-else.json",
    ]
    expected_output = [
        "data",
        "log",
        "data/loss.json",
        "data/accuracy.json",
        "data/accuracy2.json",
        "log/warnings.txt",
    ]

    for d, volume in dirs:
        os.mkdir(os.path.join(volume, d))

    # create all the files
    for f, volume in files:
        open(os.path.join(volume, f), "a").close()

    output_path = str(tmpdir.mkdir("exported_files"))
    output_data_config = {"S3OutputPath": "file://%s" % output_path}

    model_artifacts = sagemaker_container.retrieve_artifacts(
        compose_data, output_data_config, sagemaker_session
    ).replace("file://", "")
    artifacts = os.path.dirname(model_artifacts)

    # we have both the tar files
    assert set(os.listdir(artifacts)) == {"model.tar.gz", "output.tar.gz"}

    # check that the tar files contain what we expect
    tar = tarfile.open(os.path.join(output_path, "model.tar.gz"))
    model_tar_files = [m.name for m in tar.getmembers()]
    for f in expected_model:
        assert f in model_tar_files

    tar = tarfile.open(os.path.join(output_path, "output.tar.gz"))
    output_tar_files = [m.name for m in tar.getmembers()]
    for f in expected_output:
        assert f in output_tar_files


def test_stream_output():
    # it should raise an exception if the command fails
    with pytest.raises(RuntimeError):
        p = subprocess.Popen(
            ["ls", "/some/unknown/path"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        sagemaker.local.image._stream_output(p)

    p = subprocess.Popen(["echo", "hello"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    exit_code = sagemaker.local.image._stream_output(p)
    assert exit_code == 0


def test_check_output():
    with pytest.raises(Exception):
        sagemaker.local.image._check_output(["ls", "/some/unknown/path"])

    msg = "hello!"

    output = sagemaker.local.image._check_output(["echo", msg]).strip()
    assert output == msg

    output = sagemaker.local.image._check_output("echo %s" % msg).strip()
    assert output == msg


@patch("sagemaker.local.local_session.LocalSession", Mock())
@patch("sagemaker.local.image._stream_output", Mock())
@patch("sagemaker.local.image._SageMakerContainer._cleanup")
@patch("sagemaker.local.image._SageMakerContainer.retrieve_artifacts")
@patch("sagemaker.local.data.get_data_source_instance")
@patch("subprocess.Popen")
def test_train(
    popen, get_data_source_instance, retrieve_artifacts, cleanup, tmpdir, sagemaker_session
):
    data_source = Mock()
    data_source.get_root_dir.return_value = "foo"
    get_data_source_instance.return_value = data_source

    directories = [str(tmpdir.mkdir("container-root")), str(tmpdir.mkdir("data"))]
    with patch(
        "sagemaker.local.image._SageMakerContainer._create_tmp_folder", side_effect=directories
    ):

        instance_count = 2
        image = "my-image"
        sagemaker_container = _SageMakerContainer(
            "local", instance_count, image, sagemaker_session=sagemaker_session
        )
        sagemaker_container.train(
            INPUT_DATA_CONFIG, OUTPUT_DATA_CONFIG, HYPERPARAMETERS, TRAINING_JOB_NAME
        )

        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )
        call_args = popen.call_args[0][0]
        assert call_args is not None

        expected = [
            "docker-compose",
            "-f",
            docker_compose_file,
            "up",
            "--build",
            "--abort-on-container-exit",
        ]
        for i, v in enumerate(expected):
            assert call_args[i] == v

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f)
            assert len(config["services"]) == instance_count
            for h in sagemaker_container.hosts:
                assert config["services"][h]["image"] == image
                assert config["services"][h]["command"] == "train"
                # TODO-reinvent-2019 [akarpur]: uncomment the below assert statement
                # assert "AWS_REGION={}".format(REGION) in config["services"][h]["environment"]
                assert (
                    "TRAINING_JOB_NAME={}".format(TRAINING_JOB_NAME)
                    in config["services"][h]["environment"]
                )

        # assert that expected by sagemaker container output directories exist
        assert os.path.exists(os.path.join(sagemaker_container.container_root, "output"))
        assert os.path.exists(os.path.join(sagemaker_container.container_root, "output/data"))

    retrieve_artifacts.assert_called_once()
    cleanup.assert_called_once()


@patch("sagemaker.local.local_session.LocalSession", Mock())
@patch("sagemaker.local.image._stream_output", Mock())
@patch("sagemaker.local.image._SageMakerContainer._cleanup", Mock())
@patch("sagemaker.local.data.get_data_source_instance")
def test_train_with_hyperparameters_without_job_name(
    get_data_source_instance, tmpdir, sagemaker_session
):
    data_source = Mock()
    data_source.get_root_dir.return_value = "foo"
    get_data_source_instance.return_value = data_source

    directories = [str(tmpdir.mkdir("container-root")), str(tmpdir.mkdir("data"))]
    with patch(
        "sagemaker.local.image._SageMakerContainer._create_tmp_folder", side_effect=directories
    ):
        instance_count = 2
        image = "my-image"
        sagemaker_container = _SageMakerContainer(
            "local", instance_count, image, sagemaker_session=sagemaker_session
        )
        sagemaker_container.train(
            INPUT_DATA_CONFIG, OUTPUT_DATA_CONFIG, HYPERPARAMETERS, TRAINING_JOB_NAME
        )

        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f)
            for h in sagemaker_container.hosts:
                assert (
                    "TRAINING_JOB_NAME={}".format(TRAINING_JOB_NAME)
                    in config["services"][h]["environment"]
                )


@patch("sagemaker.local.local_session.LocalSession", Mock())
@patch("sagemaker.local.image._stream_output", side_effect=RuntimeError("this is expected"))
@patch("sagemaker.local.image._SageMakerContainer._cleanup")
@patch("sagemaker.local.image._SageMakerContainer.retrieve_artifacts")
@patch("sagemaker.local.data.get_data_source_instance")
@patch("subprocess.Popen", Mock())
def test_train_error(
    get_data_source_instance, retrieve_artifacts, cleanup, _stream_output, tmpdir, sagemaker_session
):
    data_source = Mock()
    data_source.get_root_dir.return_value = "foo"
    get_data_source_instance.return_value = data_source

    directories = [str(tmpdir.mkdir("container-root")), str(tmpdir.mkdir("data"))]
    with patch(
        "sagemaker.local.image._SageMakerContainer._create_tmp_folder", side_effect=directories
    ):
        instance_count = 2
        image = "my-image"
        sagemaker_container = _SageMakerContainer(
            "local", instance_count, image, sagemaker_session=sagemaker_session
        )

        with pytest.raises(RuntimeError) as e:
            sagemaker_container.train(
                INPUT_DATA_CONFIG, OUTPUT_DATA_CONFIG, HYPERPARAMETERS, TRAINING_JOB_NAME
            )

        assert "this is expected" in str(e)

    retrieve_artifacts.assert_called_once()
    cleanup.assert_called_once()


@patch("sagemaker.local.local_session.LocalSession", Mock())
@patch("sagemaker.local.image._stream_output", Mock())
@patch("sagemaker.local.image._SageMakerContainer._cleanup", Mock())
@patch("sagemaker.local.data.get_data_source_instance")
@patch("subprocess.Popen", Mock())
def test_train_local_code(get_data_source_instance, tmpdir, sagemaker_session):
    data_source = Mock()
    data_source.get_root_dir.return_value = "foo"
    get_data_source_instance.return_value = data_source

    directories = [str(tmpdir.mkdir("container-root")), str(tmpdir.mkdir("data"))]
    with patch(
        "sagemaker.local.image._SageMakerContainer._create_tmp_folder", side_effect=directories
    ):
        instance_count = 2
        image = "my-image"
        sagemaker_container = _SageMakerContainer(
            "local", instance_count, image, sagemaker_session=sagemaker_session
        )

        sagemaker_container.train(
            INPUT_DATA_CONFIG, OUTPUT_DATA_CONFIG, LOCAL_CODE_HYPERPARAMETERS, TRAINING_JOB_NAME
        )

        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )
        shared_folder_path = os.path.join(sagemaker_container.container_root, "shared")

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f)
            assert len(config["services"]) == instance_count

        for h in sagemaker_container.hosts:
            assert config["services"][h]["image"] == image
            assert config["services"][h]["command"] == "train"
            volumes = config["services"][h]["volumes"]
            assert "%s:/opt/ml/code" % "/tmp/code" in volumes
            assert "%s:/opt/ml/shared" % shared_folder_path in volumes

            config_file_root = os.path.join(
                sagemaker_container.container_root, h, "input", "config"
            )
            hyperparameters_file = os.path.join(config_file_root, "hyperparameters.json")
            hyperparameters_data = json.load(open(hyperparameters_file))
            assert hyperparameters_data["sagemaker_submit_directory"] == json.dumps("/opt/ml/code")


@patch("sagemaker.local.local_session.LocalSession", Mock())
@patch("sagemaker.local.image._stream_output", Mock())
@patch("sagemaker.local.image._SageMakerContainer._cleanup", Mock())
@patch("sagemaker.local.data.get_data_source_instance")
@patch("subprocess.Popen", Mock())
def test_train_local_intermediate_output(get_data_source_instance, tmpdir, sagemaker_session):
    data_source = Mock()
    data_source.get_root_dir.return_value = "foo"
    get_data_source_instance.return_value = data_source

    directories = [str(tmpdir.mkdir("container-root")), str(tmpdir.mkdir("data"))]
    with patch(
        "sagemaker.local.image._SageMakerContainer._create_tmp_folder", side_effect=directories
    ):
        instance_count = 2
        image = "my-image"
        sagemaker_container = _SageMakerContainer(
            "local", instance_count, image, sagemaker_session=sagemaker_session
        )

        output_path = str(tmpdir.mkdir("customer_intermediate_output"))
        output_data_config = {"S3OutputPath": "file://%s" % output_path}
        hyperparameters = {"sagemaker_s3_output": output_path}

        sagemaker_container.train(
            INPUT_DATA_CONFIG, output_data_config, hyperparameters, TRAINING_JOB_NAME
        )

        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )
        intermediate_folder_path = os.path.join(output_path, "output/intermediate")

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f)
            assert len(config["services"]) == instance_count
            for h in sagemaker_container.hosts:
                assert config["services"][h]["image"] == image
                assert config["services"][h]["command"] == "train"
                volumes = config["services"][h]["volumes"]
                assert "%s:/opt/ml/output/intermediate" % intermediate_folder_path in volumes


def test_container_has_gpu_support(tmpdir, sagemaker_session):
    instance_count = 1
    image = "my-image"
    sagemaker_container = _SageMakerContainer(
        "local_gpu", instance_count, image, sagemaker_session=sagemaker_session
    )

    docker_host = sagemaker_container._create_docker_host("host-1", {}, set(), "train", [])
    assert "runtime" in docker_host
    assert docker_host["runtime"] == "nvidia"


def test_container_does_not_enable_nvidia_docker_for_cpu_containers(sagemaker_session):
    instance_count = 1
    image = "my-image"
    sagemaker_container = _SageMakerContainer(
        "local", instance_count, image, sagemaker_session=sagemaker_session
    )

    docker_host = sagemaker_container._create_docker_host("host-1", {}, set(), "train", [])
    assert "runtime" not in docker_host


@patch("sagemaker.local.image._HostingContainer.run", Mock())
@patch("sagemaker.local.image._SageMakerContainer._prepare_serving_volumes", Mock(return_value=[]))
@patch("shutil.copy", Mock())
@patch("shutil.copytree", Mock())
def test_serve(tmpdir, sagemaker_session):
    with patch(
        "sagemaker.local.image._SageMakerContainer._create_tmp_folder",
        return_value=str(tmpdir.mkdir("container-root")),
    ):
        image = "my-image"
        sagemaker_container = _SageMakerContainer(
            "local", 1, image, sagemaker_session=sagemaker_session
        )
        environment = {"env1": 1, "env2": "b", "SAGEMAKER_SUBMIT_DIRECTORY": "s3://some/path"}

        sagemaker_container.serve("/some/model/path", environment)
        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f)

            for h in sagemaker_container.hosts:
                assert config["services"][h]["image"] == image
                assert config["services"][h]["command"] == "serve"


@patch("sagemaker.local.image._HostingContainer.run", Mock())
@patch("sagemaker.local.image._SageMakerContainer._prepare_serving_volumes", Mock(return_value=[]))
@patch("shutil.copy", Mock())
@patch("shutil.copytree", Mock())
def test_serve_local_code(tmpdir, sagemaker_session):
    with patch(
        "sagemaker.local.image._SageMakerContainer._create_tmp_folder",
        return_value=str(tmpdir.mkdir("container-root")),
    ):
        image = "my-image"
        sagemaker_container = _SageMakerContainer(
            "local", 1, image, sagemaker_session=sagemaker_session
        )
        environment = {"env1": 1, "env2": "b", "SAGEMAKER_SUBMIT_DIRECTORY": "file:///tmp/code"}

        sagemaker_container.serve("/some/model/path", environment)
        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f)

            for h in sagemaker_container.hosts:
                assert config["services"][h]["image"] == image
                assert config["services"][h]["command"] == "serve"

                volumes = config["services"][h]["volumes"]
                assert "%s:/opt/ml/code" % "/tmp/code" in volumes
                assert (
                    "SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code"
                    in config["services"][h]["environment"]
                )


@patch("sagemaker.local.image._HostingContainer.run", Mock())
@patch("sagemaker.local.image._SageMakerContainer._prepare_serving_volumes", Mock(return_value=[]))
@patch("shutil.copy", Mock())
@patch("shutil.copytree", Mock())
def test_serve_local_code_no_env(tmpdir, sagemaker_session):
    with patch(
        "sagemaker.local.image._SageMakerContainer._create_tmp_folder",
        return_value=str(tmpdir.mkdir("container-root")),
    ):
        image = "my-image"
        sagemaker_container = _SageMakerContainer(
            "local", 1, image, sagemaker_session=sagemaker_session
        )
        sagemaker_container.serve("/some/model/path", {})
        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f)

            for h in sagemaker_container.hosts:
                assert config["services"][h]["image"] == image
                assert config["services"][h]["command"] == "serve"


@patch("sagemaker.local.data.get_data_source_instance")
@patch("tarfile.is_tarfile")
@patch("tarfile.open", MagicMock())
@patch("os.makedirs", Mock())
def test_prepare_serving_volumes_with_s3_model(
    is_tarfile, get_data_source_instance, sagemaker_session
):
    sagemaker_container = _SageMakerContainer(
        "local", 1, "some-image", sagemaker_session=sagemaker_session
    )
    sagemaker_container.container_root = "/tmp/container_root"

    s3_data_source = Mock()
    s3_data_source.get_root_dir.return_value = "/tmp/downloaded/data/"
    s3_data_source.get_file_list.return_value = ["/tmp/downloaded/data/my_model.tar.gz"]
    get_data_source_instance.return_value = s3_data_source
    is_tarfile.return_value = True

    volumes = sagemaker_container._prepare_serving_volumes("s3://bucket/my_model.tar.gz")
    is_tarfile.assert_called_with("/tmp/downloaded/data/my_model.tar.gz")

    assert len(volumes) == 1
    assert volumes[0].container_dir == "/opt/ml/model"
    assert volumes[0].host_dir == "/tmp/downloaded/data/"


@patch("sagemaker.local.data.get_data_source_instance")
@patch("tarfile.is_tarfile", Mock(return_value=False))
@patch("os.makedirs", Mock())
def test_prepare_serving_volumes_with_local_model(get_data_source_instance, sagemaker_session):
    sagemaker_container = _SageMakerContainer(
        "local", 1, "some-image", sagemaker_session=sagemaker_session
    )
    sagemaker_container.container_root = "/tmp/container_root"

    local_file_data_source = Mock()
    local_file_data_source.get_root_dir.return_value = "/path/to/my_model"
    local_file_data_source.get_file_list.return_value = ["/path/to/my_model/model"]
    get_data_source_instance.return_value = local_file_data_source

    volumes = sagemaker_container._prepare_serving_volumes("file:///path/to/my_model")

    assert len(volumes) == 1
    assert volumes[0].container_dir == "/opt/ml/model"
    assert volumes[0].host_dir == "/path/to/my_model"


def test_ecr_login_non_ecr():
    session_mock = Mock()
    result = sagemaker.local.image._ecr_login_if_needed(session_mock, "ubuntu")

    session_mock.assert_not_called()
    assert result is False


@patch("sagemaker.local.image._check_output", return_value="123451324")
@pytest.mark.parametrize(
    "image",
    [
        "520713654638.dkr.ecr.us-east-1.amazonaws.com/image-i-have:1.0",
        "520713654638.dkr.ecr.us-iso-east-1.c2s.ic.gov/image-i-have:1.0",
    ],
)
def test_ecr_login_image_exists(_check_output, image):
    session_mock = Mock()

    result = sagemaker.local.image._ecr_login_if_needed(session_mock, image)

    session_mock.assert_not_called()
    _check_output.assert_called()
    assert result is False


@patch("subprocess.check_output", return_value="".encode("utf-8"))
def test_ecr_login_needed(check_output):
    session_mock = Mock()

    token = "very-secure-token"
    token_response = "AWS:%s" % token
    b64_token = base64.b64encode(token_response.encode("utf-8"))
    response = {
        u"authorizationData": [
            {
                u"authorizationToken": b64_token,
                u"proxyEndpoint": u"https://520713654638.dkr.ecr.us-east-1.amazonaws.com",
            }
        ],
        "ResponseMetadata": {
            "RetryAttempts": 0,
            "HTTPStatusCode": 200,
            "RequestId": "25b2ac63-36bf-11e8-ab6a-e5dc597d2ad9",
        },
    }
    session_mock.client("ecr").get_authorization_token.return_value = response
    image = "520713654638.dkr.ecr.us-east-1.amazonaws.com/image-i-need:1.1"
    result = sagemaker.local.image._ecr_login_if_needed(session_mock, image)

    expected_command = (
        "docker login -u AWS -p %s https://520713654638.dkr.ecr.us-east-1.amazonaws.com" % token
    )

    check_output.assert_called_with(expected_command, shell=True)
    session_mock.client("ecr").get_authorization_token.assert_called_with(
        registryIds=["520713654638"]
    )

    assert result is True


@patch("subprocess.check_output", return_value="".encode("utf-8"))
def test_pull_image(check_output):
    image = "520713654638.dkr.ecr.us-east-1.amazonaws.com/image-i-need:1.1"

    sagemaker.local.image._pull_image(image)

    expected_command = "docker pull %s" % image

    check_output.assert_called_once_with(expected_command, shell=True)


def test__aws_credentials_with_long_lived_credentials():
    credentials = Credentials(access_key=_random_string(), secret_key=_random_string(), token=None)
    session = Mock()
    session.get_credentials.return_value = credentials

    aws_credentials = _aws_credentials(session)

    assert aws_credentials == [
        "AWS_ACCESS_KEY_ID=%s" % credentials.access_key,
        "AWS_SECRET_ACCESS_KEY=%s" % credentials.secret_key,
    ]


@patch("sagemaker.local.image._aws_credentials_available_in_metadata_service")
def test__aws_credentials_with_short_lived_credentials_and_ec2_metadata_service_having_credentials(
    mock,
):
    credentials = Credentials(
        access_key=_random_string(), secret_key=_random_string(), token=_random_string()
    )
    session = Mock()
    session.get_credentials.return_value = credentials
    mock.return_value = True
    aws_credentials = _aws_credentials(session)

    assert aws_credentials is None


@patch("sagemaker.local.image._aws_credentials_available_in_metadata_service")
def test__aws_credentials_with_short_lived_credentials_and_ec2_metadata_service_having_no_credentials(
    mock,
):
    credentials = Credentials(
        access_key=_random_string(), secret_key=_random_string(), token=_random_string()
    )
    session = Mock()
    session.get_credentials.return_value = credentials
    mock.return_value = False
    aws_credentials = _aws_credentials(session)

    assert aws_credentials == [
        "AWS_ACCESS_KEY_ID=%s" % credentials.access_key,
        "AWS_SECRET_ACCESS_KEY=%s" % credentials.secret_key,
        "AWS_SESSION_TOKEN=%s" % credentials.token,
    ]


def _random_string(size=6, chars=string.ascii_uppercase):
    return "".join(random.choice(chars) for x in range(size))
