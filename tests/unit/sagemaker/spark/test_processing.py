from __future__ import absolute_import

import pathlib
import subprocess
import tempfile

from io import BytesIO
from typing import Generator
from unittest.case import TestCase
from unittest.mock import (
    MagicMock,
    Mock,
    call,
    mock_open,
    patch,
)

import pytest

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.session_settings import SessionSettings
from sagemaker.spark.processing import (
    PySparkProcessor,
    SparkJarProcessor,
    _SparkProcessorBase,
    _HistoryServer,
    FileType,
)

SPARK_EVENT_LOGS_S3_URI = "s3://bucket/spark-events"
REGION = "us-east-1"
BUCKET_NAME = "bucket"


@pytest.fixture
def processing_output():
    return ProcessingOutput(
        source="/opt/ml/processing/spark-events/",
        destination=SPARK_EVENT_LOGS_S3_URI,
        s3_upload_mode="Continuous",
    )


@pytest.fixture
def processing_input():
    return ProcessingInput(
        source="s3_uri",
        destination="destination",
    )


@pytest.fixture()
def sagemaker_session():
    boto_mock = MagicMock(name="boto_session", region_name=REGION)
    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
        settings=SessionSettings(),
        default_bucket_prefix=None,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)
    session_mock.sagemaker_config = {}

    return session_mock


@pytest.fixture
def spark_processor_base(sagemaker_session) -> _SparkProcessorBase:
    spark_processor_base = _SparkProcessorBase(
        base_job_name="sm-spark",
        role="AmazonSageMaker-ExecutionRole",
        framework_version="2.4",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        image_uri="790336243319.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark:0.1",
        sagemaker_session=sagemaker_session,
    )

    return spark_processor_base


@pytest.fixture
def py_spark_processor(sagemaker_session) -> PySparkProcessor:
    spark = PySparkProcessor(
        base_job_name="sm-spark",
        role="AmazonSageMaker-ExecutionRole",
        framework_version="2.4",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        image_uri="790336243319.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark:0.1",
        sagemaker_session=sagemaker_session,
    )

    return spark


@pytest.fixture
def tempdir() -> Generator[tempfile.TemporaryDirectory, None, None]:
    dir = tempfile.TemporaryDirectory()
    yield dir
    dir.cleanup()


@pytest.fixture
def empty_tempdir_path() -> Generator[str, None, None]:
    dir = tempfile.TemporaryDirectory()
    yield pathlib.Path(dir.name).resolve()
    dir.cleanup()


@pytest.fixture
def jars_dir(tempdir: tempfile.TemporaryDirectory) -> str:
    return tempdir.name


@pytest.fixture
def jar_file(jars_dir: str) -> Generator[str, None, None]:
    tmp = tempfile.NamedTemporaryFile(dir=jars_dir, prefix="1", suffix=".jar")
    yield tmp.name
    tmp.close()


def test_pyspark_processor_instantiation(sagemaker_session):
    # This just tests that the import is right and that the processor can be instantiated
    # Functionality is tested in project root container directory.
    PySparkProcessor(
        base_job_name="sm-spark",
        role="AmazonSageMaker-ExecutionRole",
        framework_version="2.4",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=sagemaker_session,
    )


happy_config_dict = {
    "Classification": "core-site",
    "Properties": {"hadoop.security.groups.cache.secs": "250"},
}
happy_config_list = [
    {"Classification": "core-site", "Properties": {"hadoop.security.groups.cache.secs": "250"}},
    {"Classification": "spark-defaults", "Properties": {"spark.driver.memory": "2"}},
]
nested_config = [
    {
        "Classification": "yarn-env",
        "Properties": {},
        "Configurations": [
            {
                "Classification": "export",
                "Properties": {
                    "YARN_RESOURCEMANAGER_OPTS": "-Xdebug -Xrunjdwp:transport=dt_socket"
                },
                "Configurations": [],
            }
        ],
    }
]
invalid_classification_dict = {"Classification": "invalid-site", "Properties": {}}
invalid_classification_list = [invalid_classification_dict]
missing_classification_dict = {"Properties": {}}
missing_classification_list = [missing_classification_dict]
missing_properties_dict = {"Classification": "core-site"}
missing_properties_list = [missing_properties_dict]


@pytest.mark.parametrize(
    "config,expected",
    [
        (happy_config_dict, None),
        (invalid_classification_dict, ValueError),
        (happy_config_list, None),
        (invalid_classification_list, ValueError),
        (nested_config, None),
        (missing_classification_dict, ValueError),
        (missing_classification_list, ValueError),
        (missing_properties_dict, ValueError),
        (missing_properties_list, ValueError),
    ],
)
def test_configuration_validation(config, expected, sagemaker_session) -> None:
    # This just tests that the import is right and that the processor can be instantiated
    # Functionality is tested in project root container directory.
    spark = PySparkProcessor(
        base_job_name="sm-spark",
        role="AmazonSageMaker-ExecutionRole",
        framework_version="2.4",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        sagemaker_session=sagemaker_session,
    )

    if expected is None:
        spark._validate_configuration(config)
    else:
        with pytest.raises(expected):
            spark._validate_configuration(config)


@patch("sagemaker.processing.ScriptProcessor.run")
def test_spark_processor_base_run(mock_super_run, spark_processor_base):
    spark_processor_base.run(submit_app="app")

    mock_super_run.assert_called_with("app", None, None, None, True, True, None, None, None)


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "spark_event_logs_s3_uri": None,
                "configuration": None,
                "inputs": None,
                "outputs": None,
            },
            {"inputs": None, "outputs": None},
        ),
        (
            {
                "spark_event_logs_s3_uri": SPARK_EVENT_LOGS_S3_URI,
                "configuration": None,
                "inputs": None,
                "outputs": None,
            },
            {"inputs": None, "outputs": [processing_output]},
        ),
        (
            {
                "spark_event_logs_s3_uri": None,
                "configuration": happy_config_dict,
                "inputs": None,
                "outputs": None,
            },
            {"inputs": [processing_input], "outputs": None},
        ),
        (
            {
                "spark_event_logs_s3_uri": None,
                "configuration": happy_config_dict,
                "inputs": [],
                "outputs": None,
            },
            {"inputs": [processing_input], "outputs": None},
        ),
    ],
)
@patch("sagemaker.spark.processing.ProcessingOutput")
@patch("sagemaker.spark.processing._SparkProcessorBase._stage_configuration")
def test_spark_processor_base_extend_processing_args(
    mock_stage_configuration,
    mock_processing_output,
    spark_processor_base,
    config,
    expected,
    sagemaker_session,
):
    mock_stage_configuration.return_value = processing_input
    mock_processing_output.return_value = processing_output

    extended_inputs, extended_outputs = spark_processor_base._extend_processing_args(
        inputs=config["inputs"],
        outputs=config["outputs"],
        configuration=config["configuration"],
        spark_event_logs_s3_uri=config["spark_event_logs_s3_uri"],
    )

    assert extended_inputs == expected["inputs"]
    assert extended_outputs == expected["outputs"]


serialized_configuration = BytesIO("test".encode("utf-8"))


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "spark_processor_type": "py_spark_processor",
                "configuration_location": None,
            },
            "s3://bucket/None/input/conf/configuration.json",
        ),
        (
            {
                "spark_processor_type": "py_spark_processor",
                "configuration_location": "s3://configbucket/someprefix/",
            },
            "s3://configbucket/someprefix/None/input/conf/configuration.json",
        ),
        (
            {
                "spark_processor_type": "spark_jar_processor",
                "configuration_location": None,
            },
            "s3://bucket/None/input/conf/configuration.json",
        ),
        (
            {
                "spark_processor_type": "spark_jar_processor",
                "configuration_location": "s3://configbucket/someprefix",
            },
            "s3://configbucket/someprefix/None/input/conf/configuration.json",
        ),
    ],
)
@patch("sagemaker.spark.processing.BytesIO")
@patch("sagemaker.spark.processing.S3Uploader.upload_string_as_file_body")
def test_stage_configuration(mock_s3_upload, mock_bytesIO, config, expected, sagemaker_session):
    spark_processor_type = {
        "py_spark_processor": PySparkProcessor,
        "spark_jar_processor": SparkJarProcessor,
    }[config["spark_processor_type"]]
    spark_processor = spark_processor_type(
        base_job_name="sm-spark",
        role="AmazonSageMaker-ExecutionRole",
        framework_version="2.4",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        image_uri="790336243319.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark:0.1",
        configuration_location=config["configuration_location"],
        sagemaker_session=sagemaker_session,
    )

    desired_s3_uri = expected
    mock_bytesIO.return_value = serialized_configuration

    result = spark_processor._stage_configuration({})

    mock_s3_upload.assert_called_with(
        body=serialized_configuration,
        desired_s3_uri=desired_s3_uri,
        sagemaker_session=sagemaker_session,
    )
    assert result.source == desired_s3_uri


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "spark_processor_type": "py_spark_processor",
                "dependency_location": None,
                "submit_deps": None,
                "input_channel_name": "channelName",
            },
            ValueError,
        ),
        (
            {
                "spark_processor_type": "py_spark_processor",
                "dependency_location": None,
                "submit_deps": ["s3"],
                "input_channel_name": None,
            },
            ValueError,
        ),
        (
            {
                "spark_processor_type": "py_spark_processor",
                "dependency_location": None,
                "submit_deps": ["other"],
                "input_channel_name": "channelName",
            },
            ValueError,
        ),
        (
            {
                "spark_processor_type": "py_spark_processor",
                "dependency_location": None,
                "submit_deps": ["file"],
                "input_channel_name": "channelName",
            },
            ValueError,
        ),
        (
            {
                "spark_processor_type": "py_spark_processor",
                "dependency_location": None,
                "submit_deps": ["file"],
                "input_channel_name": "channelName",
            },
            ValueError,
        ),
        (
            {
                "spark_processor_type": "py_spark_processor",
                "dependency_location": None,
                "submit_deps": ["s3", "s3"],
                "input_channel_name": "channelName",
            },
            (None, "s3://bucket,s3://bucket"),
        ),
        (
            {
                "spark_processor_type": "py_spark_processor",
                "dependency_location": None,
                "submit_deps": ["jar"],
                "input_channel_name": "channelName",
            },
            ("s3://bucket/None/input/channelName", "/opt/ml/processing/input/channelName"),
        ),
        (
            {
                "spark_processor_type": "py_spark_processor",
                "dependency_location": "s3://codebucket/someprefix/",
                "submit_deps": ["jar"],
                "input_channel_name": "channelName",
            },
            (
                "s3://codebucket/someprefix/None/input/channelName",
                "/opt/ml/processing/input/channelName",
            ),
        ),
        (
            {
                "spark_processor_type": "spark_jar_processor",
                "dependency_location": None,
                "submit_deps": ["jar"],
                "input_channel_name": "channelName",
            },
            ("s3://bucket/None/input/channelName", "/opt/ml/processing/input/channelName"),
        ),
        (
            {
                "spark_processor_type": "spark_jar_processor",
                "dependency_location": "s3://codebucket/someprefix",
                "submit_deps": ["jar"],
                "input_channel_name": "channelName",
            },
            (
                "s3://codebucket/someprefix/None/input/channelName",
                "/opt/ml/processing/input/channelName",
            ),
        ),
    ],
)
@patch("sagemaker.spark.processing.S3Uploader")
def test_stage_submit_deps(mock_s3_uploader, jar_file, config, expected, sagemaker_session):
    spark_processor_type = {
        "py_spark_processor": PySparkProcessor,
        "spark_jar_processor": SparkJarProcessor,
    }[config["spark_processor_type"]]
    spark_processor = spark_processor_type(
        base_job_name="sm-spark",
        role="AmazonSageMaker-ExecutionRole",
        framework_version="2.4",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        image_uri="790336243319.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark:0.1",
        dependency_location=config["dependency_location"],
        sagemaker_session=sagemaker_session,
    )

    submit_deps_dict = {
        None: None,
        "s3": "s3://bucket",
        "jar": jar_file,
        "file": "file://test",
        "other": "test://",
    }
    submit_deps = None
    if config["submit_deps"] is not None:
        submit_deps = [submit_deps_dict[submit_dep] for submit_dep in config["submit_deps"]]

    if expected is ValueError:
        with pytest.raises(expected) as e:
            spark_processor._stage_submit_deps(submit_deps, config["input_channel_name"])

        assert isinstance(e.value, expected)
    else:
        input_channel, spark_opt = spark_processor._stage_submit_deps(
            submit_deps, config["input_channel_name"]
        )

        if expected[0] is None:
            assert input_channel is None
            assert spark_opt == expected[1]
        else:
            assert input_channel.source == expected[0]
            assert spark_opt == expected[1]


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "image_uri": None,
                "framework_version": "2.4",
                "py_version": "py37",
                "container_version": "1",
                "instance_type": "ml.c5.xlarge",
            },
            "153931337802.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark-processing:2.4-cpu-py37-v1",
        ),
        (
            {
                "image_uri": None,
                "framework_version": "2.4",
                "py_version": None,
                "container_version": None,
                "instance_type": "ml.c5.xlarge",
            },
            "153931337802.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark-processing:2.4-cpu",
        ),
        (
            {
                "image_uri": None,
                "framework_version": "3.0",
                "py_version": None,
                "container_version": None,
                "instance_type": "ml.c5.xlarge",
            },
            "153931337802.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark-processing:3.0-cpu",
        ),
        (
            {
                "image_uri": "image_uri",
                "framework_version": "2.4",
                "py_version": None,
                "container_version": None,
                "instance_type": "ml.c5.xlarge",
            },
            "image_uri",
        ),
        (
            {
                "image_uri": None,
                "framework_version": "invalidFrameworkVersion",
                "py_version": None,
                "container_version": None,
                "instance_type": "ml.c5.xlarge",
            },
            ValueError,
        ),
        (
            {
                "image_uri": None,
                "framework_version": "2.4",
                "py_version": None,
                "container_version": "1",
                "instance_type": "ml.c5.xlarge",
            },
            ValueError,
        ),
        (
            {
                "image_uri": None,
                "framework_version": "2.4",
                "py_version": "py37",
                "container_version": None,
                "instance_type": "ml.c5.xlarge",
            },
            ValueError,
        ),
    ],
)
def test_retrieve_image_uri(py_spark_processor, config, expected):

    if expected is ValueError:
        with pytest.raises(expected):
            py_spark_processor._retrieve_image_uri(
                config["image_uri"],
                config["framework_version"],
                config["py_version"],
                config["container_version"],
                "us-west-2",
                config["instance_type"],
            )
    else:
        assert expected == py_spark_processor._retrieve_image_uri(
            config["image_uri"],
            config["framework_version"],
            config["py_version"],
            config["container_version"],
            "us-west-2",
            config["instance_type"],
        )


@patch("sagemaker.spark.processing._HistoryServer")
def test_terminate_history_server(mock_history_server, py_spark_processor):
    py_spark_processor.history_server = mock_history_server

    py_spark_processor.terminate_history_server()
    mock_history_server.down.assert_called_once()


@patch("sagemaker.spark.processing._ecr_login_if_needed")
@patch("sagemaker.spark.processing._pull_image")
@patch("sagemaker.spark.processing._SparkProcessorBase._prepare_history_server_env_variables")
@patch("sagemaker.spark.processing._HistoryServer.run")
@patch("sagemaker.spark.processing._SparkProcessorBase._check_history_server")
def test_start_history_server(
    mock_check_history_server,
    mock_history_server_run,
    mock_prepare_history_server_env_variables,
    mock_pull_image,
    mock_ecr_login,
    py_spark_processor,
):
    mock_ecr_login.return_value = True
    py_spark_processor.start_history_server()

    mock_pull_image.assert_called_once()
    mock_prepare_history_server_env_variables.assert_called_once()
    mock_history_server_run.assert_called_once()
    mock_check_history_server.assert_called_once()


@patch("sagemaker.spark.processing._SparkProcessorBase._get_notebook_instance_domain")
@patch("sagemaker.spark.processing._SparkProcessorBase._config_aws_credentials")
@patch("sagemaker.spark.processing._SparkProcessorBase._is_notebook_instance")
def test_prepare_history_server_env_variables(
    mock_is_notebook_instance,
    mock_config_aws_credentials,
    mock_get_notebook_instance_domain,
    py_spark_processor,
):
    mock_is_notebook_instance.return_value = True
    mock_get_notebook_instance_domain.return_value = "domain"

    result = py_spark_processor._prepare_history_server_env_variables(SPARK_EVENT_LOGS_S3_URI)
    assert len(result) == 3
    assert result[_HistoryServer.arg_remote_domain_name] == "domain"
    assert result[_HistoryServer.arg_event_logs_s3_uri] == SPARK_EVENT_LOGS_S3_URI
    assert result["AWS_REGION"] == REGION

    mock_is_notebook_instance.return_value = False
    mock_get_notebook_instance_domain.return_value = "domain"
    mock_config_aws_credentials.return_value = {
        "AWS_ACCESS_KEY_ID": "123",
        "AWS_SECRET_ACCESS_KEY": "456",
        "AWS_SESSION_TOKEN": "789",
    }
    result = py_spark_processor._prepare_history_server_env_variables(SPARK_EVENT_LOGS_S3_URI)
    assert len(result) == 5
    assert result[_HistoryServer.arg_event_logs_s3_uri] == SPARK_EVENT_LOGS_S3_URI
    assert result["AWS_ACCESS_KEY_ID"] == "123"
    assert result["AWS_SECRET_ACCESS_KEY"] == "456"
    assert result["AWS_SESSION_TOKEN"] == "789"
    assert result["AWS_REGION"] == REGION

    py_spark_processor._spark_event_logs_s3_uri = SPARK_EVENT_LOGS_S3_URI
    mock_is_notebook_instance.return_value = True
    mock_get_notebook_instance_domain.return_value = "domain"
    result = py_spark_processor._prepare_history_server_env_variables(None)
    assert len(result) == 6
    assert result[_HistoryServer.arg_remote_domain_name] == "domain"
    assert result[_HistoryServer.arg_event_logs_s3_uri] == SPARK_EVENT_LOGS_S3_URI
    assert result["AWS_REGION"] == REGION


@patch("os.path.isfile")
def test_is_notebook_instance(mock_is_file, py_spark_processor):
    mock_is_file.return_value = True

    assert py_spark_processor._is_notebook_instance()


@pytest.mark.parametrize(
    "config, expected",
    [(True, "--network host"), (False, "-p 80:80 -p 15050:15050")],
)
@patch("sagemaker.spark.processing._SparkProcessorBase._is_notebook_instance")
def test_get_network_config(mock_is_notebook_instance, py_spark_processor, config, expected):
    mock_is_notebook_instance.return_value = config

    assert py_spark_processor._get_network_config() == expected


@patch(
    "sagemaker.spark.processing.open", new_callable=mock_open, read_data='{"ResourceName":"abc"}'
)
@patch("sagemaker.spark.processing._SparkProcessorBase._is_notebook_instance")
def test_get_notebook_instance_domain(
    mock_is_notebook_instance, mock_open_file, py_spark_processor
):
    mock_is_notebook_instance.return_value = True

    assert (
        py_spark_processor._get_notebook_instance_domain()
        == "https://abc.notebook.us-east-1.sagemaker.aws"
    )


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "is_history_server_started": True,
                "is_note_book_instance": True,
                "instance_domain": "http://test",
            },
            "History server is up on http://test/proxy/15050",
        ),
        (
            {"is_history_server_started": True, "is_note_book_instance": False},
            "History server is up on http://0.0.0.0/proxy/15050",
        ),
        (
            {"is_history_server_started": False, "is_note_book_instance": False},
            "History server failed to start. Please run 'docker logs history_server' to see logs",
        ),
    ],
)
@patch("sagemaker.spark.processing._SparkProcessorBase._get_notebook_instance_domain")
@patch("sagemaker.spark.processing._SparkProcessorBase._is_history_server_started")
@patch("sagemaker.spark.processing._SparkProcessorBase._is_notebook_instance")
def test_check_history_server(
    mock_is_notebook_instance,
    mock_is_history_server_started,
    mock_get_notebook_instance_domain,
    py_spark_processor,
    config,
    expected,
):

    mock_is_notebook_instance.return_value = config["is_note_book_instance"]
    mock_is_history_server_started.return_value = config["is_history_server_started"]

    if "instance_domain" in config:
        mock_get_notebook_instance_domain.return_value = config["instance_domain"]

    with TestCase.assertLogs("sagemaker", level="INFO") as cm:
        py_spark_processor._check_history_server(ping_timeout=3)

    assert expected in cm.output[0]


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "inputs": None,
                "submit_files": None,
                "files_input": [processing_input],
                "files_opt": "opt",
                "file_type": FileType.JAR,
            },
            {"command": ["smspark-submit"], "inputs": None},
        ),
        (
            {
                "inputs": None,
                "submit_files": ["file1"],
                "files_input": processing_input,
                "files_opt": "opt",
                "file_type": FileType.JAR,
            },
            {"command": ["smspark-submit", "--jars", "opt"], "inputs": [processing_input]},
        ),
        (
            {
                "inputs": [processing_input],
                "submit_files": ["file1"],
                "files_input": processing_input,
                "files_opt": "opt",
                "file_type": FileType.PYTHON,
            },
            {
                "command": ["smspark-submit", "--py-files", "opt"],
                "inputs": [processing_input, processing_input],
            },
        ),
        (
            {
                "inputs": [processing_input],
                "submit_files": ["file1"],
                "files_input": None,
                "files_opt": "",
                "file_type": FileType.PYTHON,
            },
            {"command": ["smspark-submit"], "inputs": [processing_input]},
        ),
    ],
)
@patch("sagemaker.spark.processing._SparkProcessorBase._stage_submit_deps")
def test_handle_script_dependencies(mock_stage_submit_deps, py_spark_processor, config, expected):
    mock_stage_submit_deps.return_value = (config["files_input"], config["files_opt"])
    inputs = py_spark_processor._handle_script_dependencies(
        config["inputs"], config["submit_files"], config["file_type"]
    )

    assert py_spark_processor.command == expected["command"]
    assert inputs == expected["inputs"]


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"response": MagicMock(status=200)}, True),
        ({"response": MagicMock(status=500)}, False),
        ({"response": ValueError}, False),
    ],
)
@patch("urllib.request.urlopen")
def test_is_history_server_started(mock_urlopen, py_spark_processor, config, expected):
    mock_urlopen.return_value = config["response"]
    assert py_spark_processor._is_history_server_started() == expected


def test_validate_s3_uri(py_spark_processor):
    with pytest.raises(ValueError) as e:
        py_spark_processor._validate_s3_uri("http")

    assert isinstance(e.value, ValueError)


def test_config_aws_credentials(py_spark_processor):
    expected_result = {
        "AWS_ACCESS_KEY_ID": "123",
        "AWS_SECRET_ACCESS_KEY": "456",
        "AWS_SESSION_TOKEN": "789",
    }

    creds = MagicMock(access_key="123", secret_key="456", token="789")
    py_spark_processor.sagemaker_session.boto_session.get_credentials = MagicMock(
        name="get_credentials", return_value=creds
    )
    assert py_spark_processor._config_aws_credentials() == expected_result

    py_spark_processor.sagemaker_session.boto_session.get_credentials = MagicMock(
        name="get_credentials", side_effect=ValueError
    )
    assert not py_spark_processor._config_aws_credentials()


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"submit_app": None, "files": ["test"], "inputs": [], "opt": None}, ValueError),
        (
            {"submit_app": "test.py", "files": None, "inputs": [processing_input], "opt": None},
            [processing_input],
        ),
        (
            {
                "submit_app": "test.py",
                "files": ["test"],
                "inputs": [processing_input],
                "opt": None,
            },
            [processing_input, processing_input, processing_input, processing_input],
        ),
        (
            {"submit_app": "test.py", "files": ["test"], "inputs": None, "opt": None},
            [processing_input, processing_input, processing_input],
        ),
        (
            {"submit_app": "test.py", "files": ["test"], "inputs": None, "opt": "opt"},
            [processing_input, processing_input, processing_input],
        ),
    ],
)
@patch("sagemaker.spark.processing._SparkProcessorBase.run")
@patch("sagemaker.spark.processing._SparkProcessorBase._stage_submit_deps")
@patch("sagemaker.spark.processing._SparkProcessorBase._generate_current_job_name")
def test_py_spark_processor_run(
    mock_generate_current_job_name,
    mock_stage_submit_deps,
    mock_super_run,
    py_spark_processor,
    config,
    expected,
):
    mock_stage_submit_deps.return_value = (processing_input, "opt")
    mock_generate_current_job_name.return_value = "jobName"

    if expected is ValueError:
        with pytest.raises(expected):
            py_spark_processor.run(
                submit_app=config["submit_app"],
                submit_py_files=config["files"],
                submit_jars=config["files"],
                submit_files=config["files"],
                inputs=config["inputs"],
            )
    else:
        py_spark_processor.run(
            submit_app=config["submit_app"],
            submit_py_files=config["files"],
            submit_jars=config["files"],
            submit_files=config["files"],
            inputs=config["inputs"],
        )

        mock_super_run.assert_called_with(
            submit_app=config["submit_app"],
            inputs=expected,
            outputs=None,
            arguments=None,
            wait=True,
            logs=True,
            job_name="jobName",
            experiment_config=None,
            kms_key=None,
        )


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "submit_app": None,
                "files": ["test"],
                "inputs": [],
                "opt": None,
                "arguments": ["arg1"],
            },
            ValueError,
        ),
        (
            {
                "submit_app": "test.py",
                "files": None,
                "inputs": [processing_input],
                "opt": None,
                "arguments": ["arg1"],
            },
            [processing_input],
        ),
        (
            {
                "submit_app": "test.py",
                "files": ["test"],
                "inputs": [processing_input],
                "opt": None,
                "arguments": ["arg1"],
            },
            [processing_input, processing_input, processing_input, processing_input],
        ),
        (
            {
                "submit_app": "test.py",
                "files": ["test"],
                "inputs": None,
                "opt": None,
                "arguments": ["arg1"],
            },
            [processing_input, processing_input, processing_input],
        ),
        (
            {
                "submit_app": "test.py",
                "files": ["test"],
                "inputs": None,
                "opt": "opt",
                "arguments": ["arg1"],
            },
            [processing_input, processing_input, processing_input],
        ),
    ],
)
@patch("sagemaker.spark.processing._SparkProcessorBase.get_run_args")
@patch("sagemaker.spark.processing._SparkProcessorBase._stage_submit_deps")
@patch("sagemaker.spark.processing._SparkProcessorBase._generate_current_job_name")
def test_py_spark_processor_get_run_args(
    mock_generate_current_job_name,
    mock_stage_submit_deps,
    mock_super_get_run_args,
    py_spark_processor,
    config,
    expected,
):
    mock_stage_submit_deps.return_value = (processing_input, "opt")
    mock_generate_current_job_name.return_value = "jobName"

    if expected is ValueError:
        with pytest.raises(expected):
            py_spark_processor.get_run_args(
                submit_app=config["submit_app"],
                submit_py_files=config["files"],
                submit_jars=config["files"],
                submit_files=config["files"],
                inputs=config["inputs"],
                arguments=config["arguments"],
            )
    else:
        py_spark_processor.get_run_args(
            submit_app=config["submit_app"],
            submit_py_files=config["files"],
            submit_jars=config["files"],
            submit_files=config["files"],
            inputs=config["inputs"],
            arguments=config["arguments"],
        )

        mock_super_get_run_args.assert_called_with(
            code=config["submit_app"],
            inputs=expected,
            outputs=None,
            arguments=config["arguments"],
        )


@patch("sagemaker.spark.processing._SparkProcessorBase.run")
@patch("sagemaker.spark.processing._SparkProcessorBase._stage_submit_deps")
@patch("sagemaker.spark.processing._SparkProcessorBase._generate_current_job_name")
def test_py_spark_processor_run_twice(
    mock_generate_current_job_name, mock_stage_submit_deps, mock_super_run, py_spark_processor
):
    mock_stage_submit_deps.return_value = (processing_input, "opt")
    mock_generate_current_job_name.return_value = "jobName"
    expected_command = ["smspark-submit", "--py-files", "opt", "--jars", "opt", "--files", "opt"]
    py_spark_processor.run(
        submit_app="submit_app",
        submit_py_files="files",
        submit_jars="test",
        submit_files="test",
        inputs=[],
    )

    py_spark_processor.run(
        submit_app="submit_app",
        submit_py_files="files",
        submit_jars="test",
        submit_files="test",
        inputs=[],
    )

    assert py_spark_processor.command == expected_command


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "submit_app": None,
                "submit_class": "_class",
                "files": ["test"],
                "inputs": [],
                "opt": None,
            },
            ValueError,
        ),
        (
            {
                "submit_app": "test.py",
                "submit_class": None,
                "files": ["test"],
                "inputs": [],
                "opt": None,
            },
            ValueError,
        ),
        (
            {
                "submit_app": "test.py",
                "submit_class": "_class",
                "files": None,
                "inputs": [processing_input],
                "opt": None,
            },
            [processing_input],
        ),
        (
            {
                "submit_app": "test.py",
                "submit_class": "_class",
                "files": ["test"],
                "inputs": [processing_input],
                "opt": None,
            },
            [processing_input, processing_input, processing_input],
        ),
        (
            {
                "submit_app": "test.py",
                "submit_class": "_class",
                "files": ["test"],
                "inputs": None,
                "opt": None,
            },
            [processing_input, processing_input],
        ),
        (
            {
                "submit_app": "test.py",
                "submit_class": "_class",
                "files": ["test"],
                "inputs": None,
                "opt": "opt",
            },
            [processing_input, processing_input],
        ),
    ],
)
@patch("sagemaker.spark.processing._SparkProcessorBase.run")
@patch("sagemaker.spark.processing._SparkProcessorBase._stage_submit_deps")
@patch("sagemaker.spark.processing._SparkProcessorBase._generate_current_job_name")
def test_spark_jar_processor_run(
    mock_generate_current_job_name,
    mock_stage_submit_deps,
    mock_super_run,
    config,
    expected,
    sagemaker_session,
):
    mock_stage_submit_deps.return_value = (processing_input, "opt")
    mock_generate_current_job_name.return_value = "jobName"

    spark_jar_processor = SparkJarProcessor(
        base_job_name="sm-spark",
        role="AmazonSageMaker-ExecutionRole",
        framework_version="2.4",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        image_uri="790336243319.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark:0.1",
        sagemaker_session=sagemaker_session,
    )

    if expected is ValueError:
        with pytest.raises(expected):
            spark_jar_processor.run(
                submit_app=config["submit_app"],
                submit_class=config["submit_class"],
                submit_jars=config["files"],
                submit_files=config["files"],
                inputs=config["inputs"],
            )
    else:
        spark_jar_processor.run(
            submit_app=config["submit_app"],
            submit_class=config["submit_class"],
            submit_jars=config["files"],
            submit_files=config["files"],
            inputs=config["inputs"],
        )

        mock_super_run.assert_called_with(
            submit_app=config["submit_app"],
            inputs=expected,
            outputs=None,
            arguments=None,
            wait=True,
            logs=True,
            job_name="jobName",
            experiment_config=None,
            kms_key=None,
        )


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "submit_app": None,
                "submit_class": "_class",
                "files": ["test"],
                "inputs": [],
                "opt": None,
                "arguments": ["arg1"],
                "kms_key": "test_kms_key",
            },
            ValueError,
        ),
        (
            {
                "submit_app": "test.py",
                "submit_class": None,
                "files": ["test"],
                "inputs": [],
                "opt": None,
                "arguments": ["arg1"],
                "kms_key": "test_kms_key",
            },
            ValueError,
        ),
        (
            {
                "submit_app": "test.py",
                "submit_class": "_class",
                "files": None,
                "inputs": [processing_input],
                "opt": None,
                "arguments": ["arg1"],
                "kms_key": "test_kms_key",
            },
            [processing_input],
        ),
        (
            {
                "submit_app": "test.py",
                "submit_class": "_class",
                "files": ["test"],
                "inputs": [processing_input],
                "opt": None,
                "arguments": ["arg1"],
                "kms_key": "test_kms_key",
            },
            [processing_input, processing_input, processing_input],
        ),
        (
            {
                "submit_app": "test.py",
                "submit_class": "_class",
                "files": ["test"],
                "inputs": None,
                "opt": None,
                "arguments": ["arg1"],
                "kms_key": "test_kms_key",
            },
            [processing_input, processing_input],
        ),
        (
            {
                "submit_app": "test.py",
                "submit_class": "_class",
                "files": ["test"],
                "inputs": None,
                "opt": "opt",
                "arguments": ["arg1"],
                "kms_key": "test_kms_key",
            },
            [processing_input, processing_input],
        ),
    ],
)
@patch("sagemaker.spark.processing._SparkProcessorBase.get_run_args")
@patch("sagemaker.spark.processing._SparkProcessorBase._stage_submit_deps")
@patch("sagemaker.spark.processing._SparkProcessorBase._generate_current_job_name")
def test_spark_jar_processor_get_run_args(
    mock_generate_current_job_name,
    mock_stage_submit_deps,
    mock_super_get_run_args,
    config,
    expected,
    sagemaker_session,
):
    mock_stage_submit_deps.return_value = (processing_input, "opt")
    mock_generate_current_job_name.return_value = "jobName"

    spark_jar_processor = SparkJarProcessor(
        base_job_name="sm-spark",
        role="AmazonSageMaker-ExecutionRole",
        framework_version="2.4",
        instance_count=1,
        instance_type="ml.c5.xlarge",
        image_uri="790336243319.dkr.ecr.us-west-2.amazonaws.com/sagemaker-spark:0.1",
        sagemaker_session=sagemaker_session,
    )

    if expected is ValueError:
        with pytest.raises(expected):
            spark_jar_processor.get_run_args(
                submit_app=config["submit_app"],
                submit_class=config["submit_class"],
                submit_jars=config["files"],
                submit_files=config["files"],
                inputs=config["inputs"],
                arguments=config["arguments"],
            )
    else:
        spark_jar_processor.get_run_args(
            submit_app=config["submit_app"],
            submit_class=config["submit_class"],
            submit_jars=config["files"],
            submit_files=config["files"],
            inputs=config["inputs"],
            arguments=config["arguments"],
        )

        mock_super_get_run_args.assert_called_with(
            code=config["submit_app"],
            inputs=expected,
            outputs=None,
            arguments=config["arguments"],
        )


@patch("sagemaker.spark.processing._HistoryServer.down")
@patch("sagemaker.spark.processing._HistoryServer._get_run_history_server_cmd")
@patch("subprocess.Popen")
def test_history_server_run(mock_subprocess_popen, mock_get_run_history_server_cmd, mock_down):
    mock_get_run_history_server_cmd.return_value = (
        "docker run --env AWS_ACCESS_KEY_ID=123 "
        "--name history_server --network host --entrypoint "
        "smspark-history-server image_uri"
    )
    history_server = _HistoryServer({}, "image_uri", "--network host")

    history_server.run()

    mock_down.assert_called_once()
    mock_subprocess_popen.assert_called_with(
        [
            "docker",
            "run",
            "--env",
            "AWS_ACCESS_KEY_ID=123",
            "--name",
            "history_server",
            "--network",
            "host",
            "--entrypoint",
            "smspark-history-server",
            "image_uri",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


@patch("subprocess.call")
def test_history_server_down(mock_subprocess_call):
    history_server = _HistoryServer({}, "image_uri", "--network host")

    history_server.down()

    mock_subprocess_call.assert_has_calls(
        [call(["docker", "stop", "history_server"]), call(["docker", "rm", "history_server"])]
    )


def test_get_run_history_server_cmd():
    args = {
        "AWS_ACCESS_KEY_ID": "123",
        "AWS_SECRET_ACCESS_KEY": "456",
        "AWS_SESSION_TOKEN": "789",
        "event_logs_s3_uri": "s3://bucket",
        "remote_domain_name": "domain",
    }

    history_server = _HistoryServer(args, "image_uri", "--network host")
    result = history_server._get_run_history_server_cmd()
    expected_cmd = (
        "docker run --env AWS_ACCESS_KEY_ID=123 --env AWS_SECRET_ACCESS_KEY=456 "
        "--env AWS_SESSION_TOKEN=789 --name history_server --network host --entrypoint "
        "smspark-history-server image_uri --event-logs-s3-uri s3://bucket --remote-domain-name domain"
    )

    assert result == expected_cmd
