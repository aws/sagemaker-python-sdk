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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import random
import string

from botocore.credentials import Credentials

import base64
import logging
import json
import os
import subprocess
import tarfile

import pytest
import yaml
from mock import patch, Mock, MagicMock, mock_open, call
import sagemaker
from sagemaker.local.image import _SageMakerContainer, _Volume, _aws_credentials
from sagemaker.local.utils import STUDIO_APP_TYPES

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

ENVIRONMENT = {"MYVAR": "HELLO_WORLD"}

LOCAL_STUDIO_METADATA_BASE = '{{"AppType":"{app_type}","DomainId":"d-1234567890","UserProfileName": \
    "dummy-profile","ResourceArn":"arn:aws:sagemaker:us-west-2:123456789012:app/arn", \
    "ResourceName":"datascience-1-0-ml-t3-medium-1234567890"}}'

LOCAL_STUDIO_METADATA_WITH_SPACE = '{"AppType":"KernelGateway","DomainId":"d-1234567890","SpaceName": \
    "dummy-space","ResourceArn":"arn:aws:sagemaker:us-west-2:123456789012:app/arn", \
    "ResourceName":"datascience-1-0-ml-t3-medium-1234567890"}'

DUMMY_APPTYPE_METADATA = '{"AppType":"DUMMY"}'

LOCAL_STUDIO_INCOMPLETE_METADATA = '{"AppType":"KernelGateway"}'

CLASSIC_METADATA = '{"ResourceArn": \
    "arn:aws:sagemaker:us-west-2:616250812882:notebook-instance/test", \
    "ResourceName": "test"}'


@pytest.fixture()
def sagemaker_session():
    boto_mock = MagicMock(name="boto_session", region_name=REGION)
    boto_mock.client("sts").get_caller_identity.return_value = {"Account": "123"}
    boto_mock.resource("s3").Bucket(BUCKET_NAME).objects.filter.return_value = []

    sms = sagemaker.Session(boto_session=boto_mock, sagemaker_client=MagicMock())

    sms.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    sms.expand_role = Mock(return_value=EXPANDED_ROLE)

    return sms


@patch("os.path.exists", return_value=True)
def test_check_for_studio(patch_os_exists, sagemaker_session):
    for app_type in STUDIO_APP_TYPES:
        metadata = LOCAL_STUDIO_METADATA_BASE.format(app_type=app_type)
        print(metadata)
        with patch("sagemaker.local.utils.open", mock_open(read_data=metadata)):
            with pytest.raises(
                NotImplementedError,
                match="Multi instance Local Mode execution is currently not supported in SageMaker Studio.",
            ):
                _SageMakerContainer("local", 2, "my-image", sagemaker_session=sagemaker_session)

            sagemaker_container = _SageMakerContainer(
                "local", 1, "my-image", sagemaker_session=sagemaker_session
            )
            assert sagemaker_container.is_studio

    with patch("sagemaker.local.utils.open", mock_open(read_data=LOCAL_STUDIO_METADATA_WITH_SPACE)):
        with pytest.raises(
            NotImplementedError,
            match="Multi instance Local Mode execution is currently not supported in SageMaker Studio.",
        ):
            _SageMakerContainer("local", 2, "my-image", sagemaker_session=sagemaker_session)

        sagemaker_container = _SageMakerContainer(
            "local", 1, "my-image", sagemaker_session=sagemaker_session
        )
        assert sagemaker_container.is_studio

    with patch("sagemaker.local.utils.open", mock_open(read_data=CLASSIC_METADATA)):
        sagemaker_container = _SageMakerContainer(
            "local", 1, "my-image", sagemaker_session=sagemaker_session
        )
        assert not sagemaker_container.is_studio

    with patch("sagemaker.local.utils.open", mock_open(read_data=DUMMY_APPTYPE_METADATA)):
        with pytest.raises(
            NotImplementedError,
            match="AppType DUMMY in Studio does not support Local Mode.",
        ):
            _SageMakerContainer("local", 2, "my-image", sagemaker_session=sagemaker_session)


@patch("subprocess.check_output", Mock(return_value="Docker Compose version v2.0.0-rc.3"))
def test_get_compose_cmd_prefix_with_docker_cli():
    compose_cmd_prefix = _SageMakerContainer._get_compose_cmd_prefix()
    assert compose_cmd_prefix == ["docker", "compose"]


@patch(
    "subprocess.check_output",
    side_effect=subprocess.CalledProcessError(returncode=1, cmd="docker compose version"),
)
@patch("sagemaker.local.image.find_executable", Mock(return_value="/usr/bin/docker-compose"))
def test_get_compose_cmd_prefix_with_docker_compose_cli(check_output):
    compose_cmd_prefix = _SageMakerContainer._get_compose_cmd_prefix()
    assert compose_cmd_prefix == ["docker-compose"]


@patch(
    "subprocess.check_output",
    side_effect=subprocess.CalledProcessError(returncode=1, cmd="docker compose version"),
)
@patch("sagemaker.local.image.find_executable", Mock(return_value=None))
def test_get_compose_cmd_prefix_raises_import_error(check_output):
    with pytest.raises(ImportError) as e:
        _SageMakerContainer._get_compose_cmd_prefix()
    assert (
        "Docker Compose is not installed. "
        "Local Mode features will not work without docker compose. "
        "For more information on how to install 'docker compose', please, see "
        "https://docs.docker.com/compose/install/" in str(e)
    )


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
    tar.close()


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
@patch(
    "sagemaker.local.image._SageMakerContainer._get_compose_cmd_prefix",
    Mock(return_value=["docker-compose"]),
)
@patch("sagemaker.local.data.get_data_source_instance")
@patch("subprocess.Popen")
def test_train(
    popen, get_data_source_instance, retrieve_artifacts, cleanup, tmpdir, sagemaker_session, caplog
):
    data_source = Mock()
    data_source.get_root_dir.return_value = "foo"
    get_data_source_instance.return_value = data_source

    caplog.set_level(logging.INFO)

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
            INPUT_DATA_CONFIG, OUTPUT_DATA_CONFIG, HYPERPARAMETERS, ENVIRONMENT, TRAINING_JOB_NAME
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
            config = yaml.load(f, Loader=yaml.SafeLoader)
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
    assert "[Masked]" in caplog.text


@patch("sagemaker.local.local_session.LocalSession", Mock())
@patch("sagemaker.local.image._stream_output", Mock())
@patch("sagemaker.local.image._SageMakerContainer._cleanup")
@patch("sagemaker.local.image._SageMakerContainer.retrieve_artifacts")
@patch(
    "sagemaker.local.image._SageMakerContainer._get_compose_cmd_prefix",
    Mock(return_value=["docker-compose"]),
)
@patch("sagemaker.local.data.get_data_source_instance")
@patch("subprocess.Popen")
def test_train_for_studio(
    popen, get_data_source_instance, retrieve_artifacts, cleanup, tmpdir, sagemaker_session, caplog
):
    data_source = Mock()
    data_source.get_root_dir.return_value = "foo"
    get_data_source_instance.return_value = data_source

    caplog.set_level(logging.INFO)

    directories = [str(tmpdir.mkdir("container-root")), str(tmpdir.mkdir("data"))]
    with patch(
        "sagemaker.local.image._SageMakerContainer._create_tmp_folder", side_effect=directories
    ):
        instance_count = 1
        image = "my-image"
        metadata = LOCAL_STUDIO_METADATA_BASE.format(app_type="KernelGateway")
        with patch("sagemaker.local.utils.open", mock_open(read_data=metadata)):
            with patch("os.path.exists", return_value=True):
                sagemaker_container = _SageMakerContainer(
                    "local", instance_count, image, sagemaker_session=sagemaker_session
                )

        sagemaker_container.train(
            INPUT_DATA_CONFIG,
            OUTPUT_DATA_CONFIG,
            HYPERPARAMETERS,
            ENVIRONMENT,
            TRAINING_JOB_NAME,
        )

        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )

        expected_up_cmd = [
            "docker-compose",
            "-f",
            docker_compose_file,
            "up",
            "--build",
            "--abort-on-container-exit",
        ]

        popen.assert_has_calls(
            [
                call(expected_up_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT),
            ]
        )

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            assert len(config["services"]) == instance_count
            for h in sagemaker_container.hosts:
                assert config["services"][h]["image"] == image
                assert config["services"][h]["command"] == "train"
                assert (
                    "TRAINING_JOB_NAME={}".format(TRAINING_JOB_NAME)
                    in config["services"][h]["environment"]
                )
                assert "SM_STUDIO_LOCAL_MODE=True" in config["services"][h]["environment"]
                assert config["services"][h]["network_mode"] == "sagemaker"

        # assert that expected by sagemaker container output directories exist
        assert os.path.exists(os.path.join(sagemaker_container.container_root, "output"))
        assert os.path.exists(os.path.join(sagemaker_container.container_root, "output/data"))

    retrieve_artifacts.assert_called_once()
    cleanup.assert_called_once()
    assert "[Masked]" in caplog.text


@patch("sagemaker.local.local_session.LocalSession", Mock())
@patch("sagemaker.local.image._stream_output", Mock())
@patch("sagemaker.local.image._SageMakerContainer._cleanup")
@patch("sagemaker.local.image._SageMakerContainer.retrieve_artifacts")
@patch(
    "sagemaker.local.image._SageMakerContainer._get_compose_cmd_prefix",
    Mock(return_value=["docker-compose"]),
)
@patch("sagemaker.local.data.get_data_source_instance")
@patch("subprocess.Popen")
def test_train_with_entry_point(
    popen, get_data_source_instance, retrieve_artifacts, cleanup, tmpdir, sagemaker_session, caplog
):
    # This is to test the case of Pipeline function step,
    # which is translated into a training job, with container_entrypoint configured
    data_source = Mock()
    data_source.get_root_dir.return_value = "foo"
    get_data_source_instance.return_value = data_source

    caplog.set_level(logging.INFO)

    directories = [str(tmpdir.mkdir("container-root")), str(tmpdir.mkdir("data"))]
    with patch(
        "sagemaker.local.image._SageMakerContainer._create_tmp_folder", side_effect=directories
    ):

        instance_count = 2
        image = "my-image"
        container_entrypoint = [
            "/bin/bash",
            "/opt/ml/input/data/sagemaker_remote_function_bootstrap/job_driver.sh",
        ]
        container_args = ["--region", "us-west-2", "--client_python_version", "3.10"]
        sagemaker_container = _SageMakerContainer(
            instance_type="local",
            instance_count=instance_count,
            image=image,
            sagemaker_session=sagemaker_session,
            container_entrypoint=container_entrypoint,
            container_arguments=container_args,
        )
        sagemaker_container.train(
            INPUT_DATA_CONFIG, OUTPUT_DATA_CONFIG, HYPERPARAMETERS, ENVIRONMENT, TRAINING_JOB_NAME
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
            config = yaml.load(f, Loader=yaml.SafeLoader)
            assert len(config["services"]) == instance_count
            for h in sagemaker_container.hosts:
                host_config = config["services"][h]
                assert host_config["image"] == image
                assert not host_config.get("command", None)
                assert (
                    "TRAINING_JOB_NAME={}".format(TRAINING_JOB_NAME) in host_config["environment"]
                )
                assert host_config["entrypoint"] == container_entrypoint + container_args

        # assert that expected by sagemaker container output directories exist
        assert os.path.exists(os.path.join(sagemaker_container.container_root, "output"))
        assert os.path.exists(os.path.join(sagemaker_container.container_root, "output/data"))

    retrieve_artifacts.assert_called_once()
    cleanup.assert_called_once()
    assert "[Masked]" in caplog.text


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
            INPUT_DATA_CONFIG, OUTPUT_DATA_CONFIG, HYPERPARAMETERS, ENVIRONMENT, TRAINING_JOB_NAME
        )

        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            for h in sagemaker_container.hosts:
                assert (
                    "TRAINING_JOB_NAME={}".format(TRAINING_JOB_NAME)
                    in config["services"][h]["environment"]
                )


@patch("sagemaker.local.local_session.LocalSession", Mock())
@patch("sagemaker.local.image._stream_output", side_effect=RuntimeError("this is expected"))
@patch("sagemaker.local.image._SageMakerContainer._cleanup")
@patch("sagemaker.local.image._SageMakerContainer.retrieve_artifacts")
@patch(
    "sagemaker.local.image._SageMakerContainer._get_compose_cmd_prefix",
    Mock(return_value=["docker-compose"]),
)
@patch("sagemaker.local.data.get_data_source_instance")
@patch("subprocess.Popen", Mock())
def test_train_error(
    get_data_source_instance,
    retrieve_artifacts,
    cleanup,
    _stream_output,
    tmpdir,
    sagemaker_session,
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
                INPUT_DATA_CONFIG,
                OUTPUT_DATA_CONFIG,
                HYPERPARAMETERS,
                ENVIRONMENT,
                TRAINING_JOB_NAME,
            )

        assert "this is expected" in str(e)

    retrieve_artifacts.assert_called_once()
    cleanup.assert_called_once()


@patch("sagemaker.local.local_session.LocalSession", Mock())
@patch("sagemaker.local.image._stream_output", Mock())
@patch("sagemaker.local.image._SageMakerContainer._cleanup", Mock())
@patch(
    "sagemaker.local.image._SageMakerContainer._get_compose_cmd_prefix",
    Mock(return_value=["docker-compose"]),
)
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
            INPUT_DATA_CONFIG,
            OUTPUT_DATA_CONFIG,
            LOCAL_CODE_HYPERPARAMETERS,
            ENVIRONMENT,
            TRAINING_JOB_NAME,
        )

        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )
        shared_folder_path = os.path.join(sagemaker_container.container_root, "shared")

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            assert len(config["services"]) == instance_count

        for h in sagemaker_container.hosts:
            assert config["services"][h]["image"] == image
            assert config["services"][h]["command"] == "train"
            volumes = config["services"][h]["volumes"]
            volumes = [v[:-2] if v.endswith(":z") else v for v in volumes]
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
@patch(
    "sagemaker.local.image._SageMakerContainer._get_compose_cmd_prefix",
    Mock(return_value=["docker-compose"]),
)
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
            INPUT_DATA_CONFIG, output_data_config, hyperparameters, ENVIRONMENT, TRAINING_JOB_NAME
        )

        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )
        intermediate_folder_path = os.path.join(output_path, "output/intermediate")

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
            assert len(config["services"]) == instance_count
            for h in sagemaker_container.hosts:
                assert config["services"][h]["image"] == image
                assert config["services"][h]["command"] == "train"
                volumes = config["services"][h]["volumes"]
                volumes = [v[:-2] if v.endswith(":z") else v for v in volumes]
                assert "%s:/opt/ml/output/intermediate" % intermediate_folder_path in volumes


@patch("platform.system", Mock(return_value="Linux"))
@patch("sagemaker.local.image.SELINUX_ENABLED", Mock(return_value=True))
def test_container_selinux_has_label(tmpdir):
    volume = _Volume(str(tmpdir), "/opt/ml/model")

    assert volume.map.endswith(":z")


@patch("platform.system", Mock(return_value="Darwin"))
@patch("sagemaker.local.image.SELINUX_ENABLED", Mock(return_value=True))
def test_container_has_selinux_no_label(tmpdir):
    volume = _Volume(str(tmpdir), "/opt/ml/model")

    assert not volume.map.endswith(":z")


def test_container_has_gpu_support(tmpdir, sagemaker_session):
    instance_count = 1
    image = "my-image"
    sagemaker_container = _SageMakerContainer(
        "local_gpu", instance_count, image, sagemaker_session=sagemaker_session
    )

    docker_host = sagemaker_container._create_docker_host("host-1", {}, set(), "train", [])
    assert "deploy" in docker_host
    assert docker_host["deploy"] == {
        "resources": {"reservations": {"devices": [{"count": "all", "capabilities": ["gpu"]}]}}
    }


def test_container_does_not_enable_nvidia_docker_for_cpu_containers(sagemaker_session):
    instance_count = 1
    image = "my-image"
    sagemaker_container = _SageMakerContainer(
        "local", instance_count, image, sagemaker_session=sagemaker_session
    )

    docker_host = sagemaker_container._create_docker_host("host-1", {}, set(), "train", [])
    assert "runtime" not in docker_host


def test_container_with_custom_config(sagemaker_session):
    custom_config = {
        "local": {
            "container_config": {"shm_size": "128M"},
        }
    }
    sagemaker_session.config = custom_config
    instance_count = 1
    image = "my-image"
    sagemaker_container = _SageMakerContainer(
        "local", instance_count, image, sagemaker_session=sagemaker_session
    )

    docker_host = sagemaker_container._create_docker_host("host-1", {}, set(), "train", [])
    assert "shm_size" in docker_host


@patch("sagemaker.local.image._HostingContainer.run", Mock())
@patch("sagemaker.local.image._SageMakerContainer._prepare_serving_volumes", Mock(return_value=[]))
@patch("shutil.copy", Mock())
@patch("shutil.copytree", Mock())
def test_serve(tmpdir, sagemaker_session, caplog):
    caplog.set_level(logging.INFO)
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
            config = yaml.load(f, Loader=yaml.SafeLoader)

            for h in sagemaker_container.hosts:
                assert config["services"][h]["image"] == image
                assert config["services"][h]["command"] == "serve"
    assert "[Masked]" in caplog.text


@patch("sagemaker.local.image._stream_output", Mock())
@patch("sagemaker.local.image._SageMakerContainer._prepare_serving_volumes", Mock(return_value=[]))
@patch("shutil.copy", Mock())
@patch("shutil.copytree", Mock())
@patch(
    "sagemaker.local.image._SageMakerContainer._get_compose_cmd_prefix",
    Mock(return_value=["docker-compose"]),
)
@patch("subprocess.Popen")
def test_serve_for_studio(popen, tmpdir, sagemaker_session, caplog):
    caplog.set_level(logging.INFO)
    with patch(
        "sagemaker.local.image._SageMakerContainer._create_tmp_folder",
        return_value=str(tmpdir.mkdir("container-root")),
    ):
        instance_count = 1
        image = "my-image"
        metadata = LOCAL_STUDIO_METADATA_BASE.format(app_type="KernelGateway")
        with patch("sagemaker.local.utils.open", mock_open(read_data=metadata)):
            with patch("os.path.exists", return_value=True):
                sagemaker_container = _SageMakerContainer(
                    "local", instance_count, image, sagemaker_session=sagemaker_session
                )

        environment = {"env1": 1, "env2": "b", "SAGEMAKER_SUBMIT_DIRECTORY": "s3://some/path"}

        sagemaker_container.serve("/some/model/path", environment)
        docker_compose_file = os.path.join(
            sagemaker_container.container_root, "docker-compose.yaml"
        )

        expected_up_cmd = [
            "docker-compose",
            "-f",
            docker_compose_file,
            "up",
            "--build",
            "--abort-on-container-exit",
        ]

        popen.assert_has_calls(
            [
                call(expected_up_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
            ]
        )

        with open(docker_compose_file, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

            for h in sagemaker_container.hosts:
                assert config["services"][h]["image"] == image
                assert config["services"][h]["command"] == "serve"
                assert "SM_STUDIO_LOCAL_MODE=True" in config["services"][h]["environment"]
                assert config["services"][h]["network_mode"] == "sagemaker"
    assert "[Masked]" in caplog.text


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
            config = yaml.load(f, Loader=yaml.SafeLoader)

            for h in sagemaker_container.hosts:
                assert config["services"][h]["image"] == image
                assert config["services"][h]["command"] == "serve"

                volumes = config["services"][h]["volumes"]
                volumes = [v[:-2] if v.endswith(":z") else v for v in volumes]
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
            config = yaml.load(f, Loader=yaml.SafeLoader)

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
        "520713654638.dkr.ecr.us-isob-east-1.sc2s.sgov.gov/image-i-have:1.0",
    ],
)
def test_ecr_login_image_exists(_check_output, image):
    session_mock = Mock()

    result = sagemaker.local.image._ecr_login_if_needed(session_mock, image)

    session_mock.assert_not_called()
    _check_output.assert_called()
    assert result is False


@patch("subprocess.Popen", return_value=Mock(autospec=subprocess.Popen))
@patch("sagemaker.local.image._check_output", return_value="")
def test_ecr_login_needed(mock_check_output, popen):
    session_mock = Mock()

    token = "very-secure-token"
    token_response = "AWS:%s" % token
    b64_token = base64.b64encode(token_response.encode("utf-8"))
    response = {
        "authorizationData": [
            {
                "authorizationToken": b64_token,
                "proxyEndpoint": "https://520713654638.dkr.ecr.us-east-1.amazonaws.com",
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
    # What a sucessful login would look like
    popen.return_value.communicate.return_value = (None, None)

    result = sagemaker.local.image._ecr_login_if_needed(session_mock, image)

    mock_check_output.assert_called_with(f"docker images -q {image}")
    expected_command = [
        "docker",
        "login",
        "https://520713654638.dkr.ecr.us-east-1.amazonaws.com",
        "-u",
        "AWS",
        "--password-stdin",
    ]
    popen.assert_called_with(expected_command, stdin=subprocess.PIPE)
    popen.return_value.communicate.assert_called_with(input=token.encode())
    session_mock.client("ecr").get_authorization_token.assert_called_with(
        registryIds=["520713654638"]
    )

    assert result is True


@patch("subprocess.check_output", return_value="".encode("utf-8"))
def test_pull_image(check_output):
    image = "520713654638.dkr.ecr.us-east-1.amazonaws.com/image-i-need:1.1"

    sagemaker.local.image._pull_image(image)

    expected_command = "docker pull %s" % image

    check_output.assert_called_once_with(expected_command.split())


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


@patch("sagemaker.local.image._aws_credentials_available_in_metadata_service")
def test__aws_credentials_with_short_lived_credentials_and_ec2_metadata_service_having_credentials_override(
    mock,
):
    os.environ["USE_SHORT_LIVED_CREDENTIALS"] = "1"
    credentials = Credentials(
        access_key=_random_string(), secret_key=_random_string(), token=_random_string()
    )
    session = Mock()
    session.get_credentials.return_value = credentials
    mock.return_value = True
    aws_credentials = _aws_credentials(session)

    assert aws_credentials == [
        "AWS_ACCESS_KEY_ID=%s" % credentials.access_key,
        "AWS_SECRET_ACCESS_KEY=%s" % credentials.secret_key,
        "AWS_SESSION_TOKEN=%s" % credentials.token,
    ]


def _random_string(size=6, chars=string.ascii_uppercase):
    return "".join(random.choice(chars) for x in range(size))
