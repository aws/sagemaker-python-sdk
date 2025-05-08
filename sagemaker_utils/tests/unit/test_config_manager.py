import os
from distutils.command.config import config

import pytest
import yaml
import logging
from unittest.mock import Mock, MagicMock, patch, call
from jsonschema import exceptions
from yaml.constructor import ConstructorError

from sagemaker.utils.config.config_manager import SageMakerConfig

from sagemaker.utils.config.config_utils import non_repeating_log_factory, get_sagemaker_config_logger, _log_sagemaker_config_single_substitution, _log_sagemaker_config_merge
logger = get_sagemaker_config_logger()
log_info_function = non_repeating_log_factory(logger, "info")

@pytest.fixture()
def config_file_as_yaml(get_data_dir):
    config_file_path = os.path.join(get_data_dir, "config.yaml")
    with open(config_file_path, "r") as f:
        content = f.read()
    return content


@pytest.fixture()
def expected_merged_config(get_data_dir):
    expected_merged_config_file_path = os.path.join(
        get_data_dir, "expected_output_config_after_merge.yaml"
    )
    with open(expected_merged_config_file_path, "r") as f:
        content = yaml.safe_load(f.read())
    return content

class TestSageMakerConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = SageMakerConfig()
        yield
        logger.propagate = False

    def test_config_when_default_config_file_and_user_config_file_is_not_found(self):
        assert self.config.load_sagemaker_config(repeat_log=True) == {}

    def test_config_when_overriden_default_config_file_is_not_found(self, get_data_dir):
        fake_config_file_path = os.path.join(get_data_dir, "config-not-found.yaml")
        os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = fake_config_file_path
        try:
            with pytest.raises(ValueError):
                self.config.load_sagemaker_config(repeat_log=True)
        finally:
            del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]

    def test_invalid_config_file_which_has_python_code(self, get_data_dir):
        invalid_config_file_path = os.path.join(get_data_dir, "config_file_with_code.yaml")
        with open(invalid_config_file_path, "r") as f:
            yaml.unsafe_load(f)
        with pytest.raises(ConstructorError) as exception_info:
            self.config.load_sagemaker_config(additional_config_paths=[invalid_config_file_path], repeat_log=True)
        assert "python/object/apply:eval" in str(exception_info.value)

    def test_config_when_additional_config_file_path_is_not_found(self, get_data_dir):
        fake_config_file_path = os.path.join(get_data_dir, "config-not-found.yaml")
        with pytest.raises(ValueError):
            self.config.load_sagemaker_config(additional_config_paths=[fake_config_file_path], repeat_log=True)

    def test_config_factory_when_override_user_config_file_is_not_found(self, get_data_dir):
        fake_additional_override_config_file_path = os.path.join(
            get_data_dir, "additional-config-not-found.yaml"
        )
        os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = fake_additional_override_config_file_path
        try:
            with pytest.raises(ValueError):
                self.config.load_sagemaker_config(repeat_log=True)
        finally:
            del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]

    def test_default_config_file_with_invalid_schema(self, get_data_dir):
        config_file_path = os.path.join(get_data_dir, "invalid_config_file.yaml")
        os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = config_file_path
        try:
            with pytest.raises(exceptions.ValidationError):
                self.config.load_sagemaker_config(repeat_log=True)
        finally:
            del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]

    def test_default_config_file_when_directory_is_provided_as_the_path(
        self, get_data_dir, valid_config_with_all_the_scopes, base_config_with_schema
    ):
        expected_config = base_config_with_schema
        expected_config["SageMaker"] = valid_config_with_all_the_scopes
        os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = get_data_dir
        try:
            assert expected_config == self.config.load_sagemaker_config(repeat_log=True)
        finally:
            del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]

    def test_additional_config_paths_when_directory_is_provided(
        self, get_data_dir, valid_config_with_all_the_scopes, base_config_with_schema
    ):
        expected_config = base_config_with_schema
        expected_config["SageMaker"] = valid_config_with_all_the_scopes
        assert expected_config == self.config.load_sagemaker_config(
            additional_config_paths=[get_data_dir], repeat_log=True
        )

    def test_default_config_file_when_path_is_provided_as_environment_variable(
        self, get_data_dir, valid_config_with_all_the_scopes, base_config_with_schema
    ):
        os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = get_data_dir
        expected_config = base_config_with_schema
        expected_config["SageMaker"] = valid_config_with_all_the_scopes
        try:
            assert expected_config == self.config.load_sagemaker_config(repeat_log=True)
        finally:
            del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]

    def test_merge_behavior_when_additional_config_file_path_is_not_found(
        self, get_data_dir, valid_config_with_all_the_scopes, base_config_with_schema
    ):
        valid_config_file_path = os.path.join(get_data_dir, "config.yaml")
        fake_additional_override_config_file_path = os.path.join(
            get_data_dir, "additional-config-not-found.yaml"
        )
        os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = valid_config_file_path
        try:
            with pytest.raises(ValueError):
                self.config.load_sagemaker_config(
                    additional_config_paths=[fake_additional_override_config_file_path], repeat_log=True
                )
        finally:
            del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]

    def test_merge_behavior(self, get_data_dir, expected_merged_config):
        valid_config_file_path = os.path.join(get_data_dir, "sample_config_for_merge.yaml")
        additional_override_config_file_path = os.path.join(
            get_data_dir, "sample_additional_config_for_merge.yaml"
        )
        os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = valid_config_file_path
        try:
            assert expected_merged_config == self.config.load_sagemaker_config(
                additional_config_paths=[additional_override_config_file_path], repeat_log=True
            )
            os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = additional_override_config_file_path
            assert expected_merged_config == self.config.load_sagemaker_config(repeat_log=True)
        finally:
            del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]
            del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]

    def test_s3_config_file(
        self, config_file_as_yaml, valid_config_with_all_the_scopes, base_config_with_schema, s3_resource_mock
    ):
        config_file_bucket = "config-file-bucket"
        config_file_s3_prefix = "config/config.yaml"
        list_file_entry_mock = Mock()
        list_file_entry_mock.key = config_file_s3_prefix
        s3_resource_mock.Bucket(name=config_file_bucket).objects.filter(
            Prefix=config_file_s3_prefix
        ).all.return_value = [list_file_entry_mock]
        response_body_mock = MagicMock()
        response_body_mock.read.return_value = config_file_as_yaml.encode("utf-8")
        s3_resource_mock.Object(config_file_bucket, config_file_s3_prefix).get.return_value = {
            "Body": response_body_mock
        }
        config_file_s3_uri = "s3://{}/{}".format(config_file_bucket, config_file_s3_prefix)
        expected_config = base_config_with_schema
        expected_config["SageMaker"] = valid_config_with_all_the_scopes
        assert expected_config == self.config.load_sagemaker_config(
            additional_config_paths=[config_file_s3_uri], s3_resource=s3_resource_mock, repeat_log=True
        )

    def test_config_factory_when_default_s3_config_file_is_not_found(self, s3_resource_mock):
        config_file_bucket = "config-file-bucket"
        config_file_s3_prefix = "config/config.yaml"
        s3_resource_mock.Bucket(name=config_file_bucket).objects.filter(
            Prefix=config_file_s3_prefix
        ).all.return_value = []
        config_file_s3_uri = "s3://{}/{}".format(config_file_bucket, config_file_s3_prefix)
        with pytest.raises(ValueError):
            self.config.load_sagemaker_config(
                additional_config_paths=[config_file_s3_uri],
                s3_resource=s3_resource_mock,
                repeat_log=True,
            )

    def test_s3_config_file_when_uri_provided_corresponds_to_a_path(
        self, config_file_as_yaml, valid_config_with_all_the_scopes, base_config_with_schema, s3_resource_mock,
    ):
        config_file_bucket = "config-file-bucket"
        config_file_s3_prefix = "config"
        list_of_files = ["/config.yaml", "/something.txt", "/README.MD"]
        list_s3_files_mock = []
        for file in list_of_files:
            entry_mock = Mock()
            entry_mock.key = config_file_s3_prefix + file
            list_s3_files_mock.append(entry_mock)
        s3_resource_mock.Bucket(name=config_file_bucket).objects.filter(
            Prefix=config_file_s3_prefix
        ).all.return_value = list_s3_files_mock
        response_body_mock = MagicMock()
        response_body_mock.read.return_value = config_file_as_yaml.encode("utf-8")
        s3_resource_mock.Object(config_file_bucket, config_file_s3_prefix).get.return_value = {
            "Body": response_body_mock
        }
        config_file_s3_uri = "s3://{}/{}".format(config_file_bucket, config_file_s3_prefix)
        expected_config = base_config_with_schema
        expected_config["SageMaker"] = valid_config_with_all_the_scopes
        assert expected_config == self.config.load_sagemaker_config(
            additional_config_paths=[config_file_s3_uri], s3_resource=s3_resource_mock, repeat_log=True
        )

    def test_merge_of_s3_default_config_file_and_regular_config_file(
        self, get_data_dir, expected_merged_config, s3_resource_mock
    ):
        config_file_content_path = os.path.join(get_data_dir, "sample_config_for_merge.yaml")
        with open(config_file_content_path, "r") as f:
            config_file_as_yaml = f.read()
        config_file_bucket = "config-file-bucket"
        config_file_s3_prefix = "config/config.yaml"
        config_file_s3_uri = "s3://{}/{}".format(config_file_bucket, config_file_s3_prefix)
        list_file_entry_mock = Mock()
        list_file_entry_mock.key = config_file_s3_prefix
        s3_resource_mock.Bucket(name=config_file_bucket).objects.filter(
            Prefix=config_file_s3_prefix
        ).all.return_value = [list_file_entry_mock]
        response_body_mock = MagicMock()
        response_body_mock.read.return_value = config_file_as_yaml.encode("utf-8")
        s3_resource_mock.Object(config_file_bucket, config_file_s3_prefix).get.return_value = {
            "Body": response_body_mock
        }
        additional_override_config_file_path = os.path.join(
            get_data_dir, "sample_additional_config_for_merge.yaml"
        )
        os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = config_file_s3_uri
        try:
            assert expected_merged_config == self.config.load_sagemaker_config(
                additional_config_paths=[additional_override_config_file_path],
                s3_resource=s3_resource_mock,
                repeat_log=True,
            )
        finally:
            del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]

    def test_logging_when_overridden_admin_is_found_and_overridden_user_config_is_found(
        self, get_data_dir, caplog
    ):
        logger.propagate = True
        os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = get_data_dir
        os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = get_data_dir
        try:
            self.config.load_sagemaker_config(repeat_log=True)
            assert "Fetched defaults config from location: {}".format(get_data_dir) in caplog.text
            assert (
                "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_ADMIN_CONFIG_FILE_PATH)
                not in caplog.text
            )
            assert (
                "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_USER_CONFIG_FILE_PATH)
                not in caplog.text
            )
        finally:
            del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]
            del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]
            logger.propagate = False

    def test_logging_when_overridden_admin_is_found_and_default_user_config_not_found(
        self, get_data_dir, caplog
    ):
        logger.propagate = True
        caplog.set_level(logging.DEBUG, logger=logger.name)
        os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = get_data_dir
        try:
            self.config.load_sagemaker_config(repeat_log=True)
            assert "Fetched defaults config from location: {}".format(get_data_dir) in caplog.text
            assert (
                "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_USER_CONFIG_FILE_PATH)
                in caplog.text
            )
            assert (
                "Unable to load the config file from the location: {}".format(
                    self.config._DEFAULT_USER_CONFIG_FILE_PATH
                )
                in caplog.text
            )
        finally:
            del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]
            logger.propagate = False

    def test_logging_when_default_admin_not_found_and_overriden_user_config_is_found(
            self, get_data_dir, caplog
    ):
        logger.propagate = True
        caplog.set_level(logging.DEBUG, logger=logger.name)
        os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = get_data_dir
        try:
            self.config.load_sagemaker_config(repeat_log=True)
            assert "Fetched defaults config from location: {}".format(get_data_dir) in caplog.text
            assert (
                    "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_ADMIN_CONFIG_FILE_PATH)
                    in caplog.text
            )
            assert (
                    "Unable to load the config file from the location: {}".format(
                        self.config._DEFAULT_ADMIN_CONFIG_FILE_PATH
                    )
                    in caplog.text
            )
        finally:
            del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]
            logger.propagate = False

    def test_logging_when_default_admin_not_found_and_default_user_config_not_found(self, caplog):
        logger.propagate = True
        caplog.set_level(logging.DEBUG, logger=logger.name)
        self.config.load_sagemaker_config(repeat_log=True)
        assert (
                "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_ADMIN_CONFIG_FILE_PATH)
                in caplog.text
        )
        assert (
                "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_USER_CONFIG_FILE_PATH)
                in caplog.text
        )
        assert (
                "Unable to load the config file from the location: {}".format(
                    self.config._DEFAULT_ADMIN_CONFIG_FILE_PATH
                )
                in caplog.text
        )
        assert (
                "Unable to load the config file from the location: {}".format(
                    self.config._DEFAULT_USER_CONFIG_FILE_PATH
                )
                in caplog.text
        )
        logger.propagate = False

    def test_load_config_without_repeating_log(self):
        config = self.config
        with patch.object(config, 'log_info_function') as mock_log_info:
            config.load_sagemaker_config(repeat_log=False)
            assert mock_log_info.call_count == 2
            mock_log_info.assert_has_calls(
                [
                    call(
                        "Not applying SDK defaults from location: %s",
                        self.config._DEFAULT_ADMIN_CONFIG_FILE_PATH,
                    ),
                    call(
                        "Not applying SDK defaults from location: %s",
                        self.config._DEFAULT_USER_CONFIG_FILE_PATH,
                    ),
                ],
                any_order=True,
            )

    def test_logging_when_default_admin_not_found_and_overriden_user_config_not_found(
            self, get_data_dir, caplog
    ):
        logger.propagate = True
        fake_config_file_path = os.path.join(get_data_dir, "config-not-found.yaml")
        os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = fake_config_file_path
        try:
            with pytest.raises(ValueError):
                self.config.load_sagemaker_config(repeat_log=True)
            assert (
                    "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_ADMIN_CONFIG_FILE_PATH)
                    in caplog.text
            )
            assert (
                    "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_USER_CONFIG_FILE_PATH)
                    not in caplog.text
            )
        finally:
            del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]
            logger.propagate = False

    def test_logging_when_overriden_admin_not_found_and_overridden_user_config_not_found(
            self, get_data_dir, caplog
    ):
        logger.propagate = True
        fake_config_file_path = os.path.join(get_data_dir, "config-not-found.yaml")
        os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"] = fake_config_file_path
        os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"] = fake_config_file_path
        try:
            with pytest.raises(ValueError):
                self.config.load_sagemaker_config(repeat_log=True)
            assert (
                    "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_ADMIN_CONFIG_FILE_PATH)
                    not in caplog.text
            )
            assert (
                    "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_USER_CONFIG_FILE_PATH)
                    not in caplog.text
            )
        finally:
            del os.environ["SAGEMAKER_USER_CONFIG_OVERRIDE"]
            del os.environ["SAGEMAKER_ADMIN_CONFIG_OVERRIDE"]
            logger.propagate = False

    def test_logging_with_additional_configs_and_none_are_found(self, caplog):
        logger.propagate = True
        with pytest.raises(ValueError):
            self.config.load_sagemaker_config(additional_config_paths=["fake-path"], repeat_log=True)
        assert (
                "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_ADMIN_CONFIG_FILE_PATH)
                in caplog.text
        )
        assert (
                "Not applying SDK defaults from location: {}".format(self.config._DEFAULT_USER_CONFIG_FILE_PATH)
                in caplog.text
        )
        logger.propagate = False

    def test_load_local_mode_config(self):
        with patch('sagemaker.utils.config.config_manager.SageMakerConfig._load_config_from_file') as mock_load:
            self.config.load_local_mode_config()
            mock_load.assert_called_with(self.config._DEFAULT_LOCAL_MODE_CONFIG_FILE_PATH)

    def test_load_local_mode_config_when_config_file_is_not_found(self):
        with patch(
                'sagemaker.utils.config.config_manager.SageMakerConfig._load_config_from_file',
                side_effect=ValueError
        ):
            assert self.config.load_local_mode_config() is None

    @pytest.mark.parametrize(
        "method_name",
        ["info", "warning", "debug"],
    )
    def test_non_repeating_log_factory(self, method_name):
        tmp_logger = logging.getLogger("test-logger")
        mock = MagicMock()
        setattr(tmp_logger, method_name, mock)

        log_function = non_repeating_log_factory(tmp_logger, method_name)
        log_function("foo")
        log_function("foo")

        mock.assert_called_once()

    @pytest.mark.parametrize(
        "method_name",
        ["info", "warning", "debug"],
    )
    def test_non_repeating_log_factory_cache_size(self, method_name):
        tmp_logger = logging.getLogger("test-logger")
        mock = MagicMock()
        setattr(tmp_logger, method_name, mock)

        log_function = non_repeating_log_factory(tmp_logger, method_name, cache_size=2)
        log_function("foo")
        log_function("bar")
        log_function("foo2")
        log_function("foo")

        assert mock.call_count == 4

    # Tests for static methods
    def test_get_config_value(self):
        # Test with valid nested config
        test_config = {"level1": {"level2": {"value": "test"}}}
        assert self.config.get_config_value("level1.level2.value", test_config) == "test"

        # Test with non-existent path
        assert self.config.get_config_value("level1.nonexistent", test_config) is None

        # Test with None config
        assert self.config.get_config_value("any.path", None) is None

    def test_get_nested_value(self):
        # Test with valid nested dictionary
        test_dict = {"level1": {"level2": {"value": "test"}}}
        assert self.config.get_nested_value(test_dict, ["level1", "level2", "value"]) == "test"

        # Test with non-existent path
        assert self.config.get_nested_value(test_dict, ["level1", "nonexistent"]) is None

        # Test with invalid structure
        invalid_dict = {"level1": "not_a_dict"}
        with pytest.raises(ValueError):
            self.config.get_nested_value(invalid_dict, ["level1", "anything"])

        # Test with None inputs
        assert self.config.get_nested_value(None, ["any"]) is None
        assert self.config.get_nested_value({}, None) is None

    def test_set_nested_value(self):
        # Test creating new nested structure
        result = self.config.set_nested_value({}, ["level1", "level2", "value"], "test")
        assert result["level1"]["level2"]["value"] == "test"

        # Test overwriting existing value
        existing = {"level1": {"level2": {"value": "old"}}}
        result = self.config.set_nested_value(existing, ["level1", "level2", "value"], "new")
        assert result["level1"]["level2"]["value"] == "new"

        # Test with None dictionary
        result = self.config.set_nested_value(None, ["key"], "value")
        assert result == {"key": "value"}

    # Tests for instance methods
    def test_resolve_value_from_config(self):
        mock_config_logger = Mock()

        mock_info_logger = Mock()
        mock_config_logger.info = mock_info_logger
        # using a shorter name for inside the test
        sagemaker_session = MagicMock()
        sagemaker_session.sagemaker_config = {"SchemaVersion": "1.0"}
        config_key_path = "SageMaker.EndpointConfig.KmsKeyId"
        sagemaker_session.sagemaker_config.update(
            {"SageMaker": {"EndpointConfig": {"KmsKeyId": "CONFIG_VALUE"}}}
        )
        sagemaker_config = {
            "SchemaVersion": "1.0",
            "SageMaker": {"EndpointConfig": {"KmsKeyId": "CONFIG_VALUE"}},
        }

        # direct_input should be respected
        assert (
                self.config.resolve_value_from_config("INPUT", config_key_path, "DEFAULT_VALUE", sagemaker_session)
                == "INPUT"
        )

        assert self.config.resolve_value_from_config("INPUT", config_key_path, None, sagemaker_session) == "INPUT"

        assert (
                self.config.resolve_value_from_config("INPUT", "SageMaker.EndpointConfig.Tags", None, sagemaker_session)
                == "INPUT"
        )

        # Config or default values should be returned if no direct_input
        assert (
                self.config.resolve_value_from_config(None, None, "DEFAULT_VALUE", sagemaker_session) == "DEFAULT_VALUE"
        )

        assert (
                self.config.resolve_value_from_config(
                    None, "SageMaker.EndpointConfig.Tags", "DEFAULT_VALUE", sagemaker_session
                )
                == "DEFAULT_VALUE"
        )

        assert (
                self.config.resolve_value_from_config(None, config_key_path, "DEFAULT_VALUE", sagemaker_session)
                == "CONFIG_VALUE"
        )

        assert self.config.resolve_value_from_config(None, None, None, sagemaker_session) is None

        # Config value from sagemaker_config should be returned
        # if no direct_input and sagemaker_session is None
        assert (
                self.config.resolve_value_from_config(None, config_key_path, None, None, sagemaker_config)
                == "CONFIG_VALUE"
        )

        # Different falsy direct_inputs
        assert self.config.resolve_value_from_config("", config_key_path, None, sagemaker_session) == ""

        assert self.config.resolve_value_from_config([], config_key_path, None, sagemaker_session) == []

        assert self.config.resolve_value_from_config(False, config_key_path, None, sagemaker_session) is False

        assert self.config.resolve_value_from_config({}, config_key_path, None, sagemaker_session) == {}

        # Different falsy config_values
        sagemaker_session.sagemaker_config.update({"SageMaker": {"EndpointConfig": {"KmsKeyId": ""}}})
        assert self.config.resolve_value_from_config(None, config_key_path, None, sagemaker_session) == ""

        mock_info_logger.reset_mock()

    def test_get_sagemaker_config_value(self):
        mock_config_logger = Mock()

        mock_info_logger = Mock()
        mock_config_logger.info = mock_info_logger
        # using a shorter name for inside the test
        sagemaker_session = MagicMock()
        sagemaker_session.sagemaker_config = {"SchemaVersion": "1.0"}
        config_key_path = "SageMaker.EndpointConfig.KmsKeyId"
        sagemaker_session.sagemaker_config.update(
            {"SageMaker": {"EndpointConfig": {"KmsKeyId": "CONFIG_VALUE"}}}
        )
        sagemaker_config = {
            "SchemaVersion": "1.0",
            "SageMaker": {"EndpointConfig": {"KmsKeyId": "CONFIG_VALUE"}},
        }

        # Tests that the function returns the correct value when the key exists in the sagemaker_session configuration.
        assert (
                self.config.get_sagemaker_config_value(
                    sagemaker_session=sagemaker_session, key=config_key_path, sagemaker_config=None
                )
                == "CONFIG_VALUE"
        )

        # Tests that the function correctly uses the sagemaker_config to get value for the requested
        # config_key_path when sagemaker_session is None.
        assert (
                self.config.get_sagemaker_config_value(
                    sagemaker_session=None, key=config_key_path, sagemaker_config=sagemaker_config
                )
                == "CONFIG_VALUE"
        )

        # Tests that the function returns None when the key does not exist in the configuration.
        invalid_key = "inavlid_key"
        assert (
                self.config.get_sagemaker_config_value(
                    sagemaker_session=sagemaker_session, key=invalid_key, sagemaker_config=sagemaker_config
                )
                is None
        )

        # Tests that the function returns None when sagemaker_session and sagemaker_config are None.
        assert (
                self.config.get_sagemaker_config_value(
                    sagemaker_session=None, key=config_key_path, sagemaker_config=None
                )
                is None
        )

    @patch("jsonschema.validate")
    def test_resolve_class_attribute_from_config(self, mock_validate):
        class TestClass:
            def __init__(self):
                self.attribute = None

        # Test setting attribute from config
        mock_session = Mock()
        mock_session.sagemaker_config = {"path": "config_value"}

        result = self.config.resolve_class_attribute_from_config(
            TestClass,
            None,
            "attribute",
            "path",
            sagemaker_session=mock_session
        )
        assert result.attribute == "config_value"

        # Test existing value priority
        instance = TestClass()
        instance.attribute = "existing"
        result = self.config.resolve_class_attribute_from_config(
            TestClass,
            instance,
            "attribute",
            "path",
            sagemaker_session=mock_session
        )
        assert result.attribute == "existing"

    @patch("jsonschema.validate")
    def test_resolve_nested_dict_value_from_config(self, mock_validate):
        # Test setting nested value from config
        mock_session = Mock()
        mock_session.sagemaker_config = {"path": "config_value"}

        dictionary = {}
        result = self.config.resolve_nested_dict_value_from_config(
            dictionary,
            ["key1", "key2"],
            "path",
            sagemaker_session=mock_session
        )
        assert result["key1"]["key2"] == "config_value"

        # Test with existing value
        dictionary = {"key1": {"key2": "existing"}}
        result = self.config.resolve_nested_dict_value_from_config(
            dictionary,
            ["key1", "key2"],
            "path",
            sagemaker_session=mock_session
        )
        assert result["key1"]["key2"] == "existing"

    @patch("jsonschema.validate")
    def test_update_list_of_dicts_with_values_from_config(self, mock_validate):
        # Test updating list with config values
        input_list = [{"key1": "value1"}]
        mock_session = Mock()
        mock_session.sagemaker_config = {
            "path": [{"key2": "value2"}]
        }

        self.config.update_list_of_dicts_with_values_from_config(
            input_list,
            "path",
            sagemaker_session=mock_session
        )
        assert input_list[0] == {"key1": "value1", "key2": "value2"}

    @patch("jsonschema.validate")
    def test_validate_required_paths(self, mock_validate):
        # Test with all required paths present
        test_dict = {
            "required1": "value1",
            "nested": {"required2": "value2"}
        }
        assert self.config._validate_required_paths_in_a_dict(
            test_dict,
            ["required1", "nested.required2"]
        )

        # Test with missing required path
        assert not self.config._validate_required_paths_in_a_dict(
            test_dict,
            ["required1", "nonexistent"]
        )

    @patch("jsonschema.validate")
    def test_validate_union_paths(self, mock_validate):
        # Test valid union (only one option present)
        test_dict = {"option1": "value1"}
        assert self.config._validate_union_key_paths_in_a_dict(
            test_dict,
            [["option1", "option2"]]
        )

        # Test invalid union (both options present)
        test_dict["option2"] = "value2"
        assert not self.config._validate_union_key_paths_in_a_dict(
            test_dict,
            [["option1", "option2"]]
        )

    @patch("jsonschema.validate")
    def test_update_nested_dictionary_with_values_from_config(self, mock_validate):
        # Test updating nested dictionary
        source_dict = {"level1": {"existing": "value"}}
        mock_session = Mock()
        mock_session.sagemaker_config = {
            "path": {"level1": {"config": "value"}}
        }

        result = self.config.update_nested_dictionary_with_values_from_config(
            source_dict,
            "path",
            sagemaker_session=mock_session
        )
        assert result == {
            "level1": {
                "existing": "value",
                "config": "value"
            }
        }

    @patch.object(SageMakerConfig, 'load_sagemaker_config')
    def test_load_default_configs_empty(self, mock_load_config):
        """Test with empty config"""
        mock_load_config.return_value = {}
        result = self.config.load_default_configs_for_resource_name("test_resource")
        assert result == {}

    @patch.object(SageMakerConfig, 'load_sagemaker_config')
    def test_load_default_configs_valid(self, mock_load_config):
        """Test with valid config"""
        mock_config = {
            "SageMaker": {
                "PythonSDK": {
                    "Resources": {
                        "test_resource": {"key": "value"}
                    }
                }
            }
        }
        mock_load_config.return_value = mock_config
        result = self.config.load_default_configs_for_resource_name("test_resource")
        assert result == {"key": "value"}

    @patch.object(SageMakerConfig, 'load_sagemaker_config')
    def test_load_default_configs_nonexistent(self, mock_load_config):
        """Test with non-existent resource"""
        mock_config = {
            "SageMaker": {
                "PythonSDK": {
                    "Resources": {
                        "test_resource": {"key": "value"}
                    }
                }
            }
        }
        mock_load_config.return_value = mock_config
        result = self.config.load_default_configs_for_resource_name("non_existent_resource")
        assert result is None

    @patch.object(SageMakerConfig, 'load_sagemaker_config')
    def test_load_default_configs_caching(self, mock_load_config):
        """Test caching behavior"""
        mock_config = {
            "SageMaker": {
                "PythonSDK": {
                    "Resources": {
                        "test_resource": {"key": "value"}
                    }
                }
            }
        }
        mock_load_config.return_value = mock_config

        # First call
        self.config.load_default_configs_for_resource_name("test_resource")
        # Second call with same resource name
        self.config.load_default_configs_for_resource_name("test_resource")

        # Should be calling load_sagemaker_config once due to caching
        mock_load_config.assert_called_once()

    def test_get_resolved_config_value(self):
        # Test when value exists in resource_defaults
        resource_defaults = {"attribute": "resource_value"}
        global_defaults = {"attribute": "global_value"}
        result = self.config.get_resolved_config_value("attribute", resource_defaults, global_defaults)
        assert result == "resource_value"

        # Test when value exists only in global_defaults
        resource_defaults = {}
        global_defaults = {"attribute": "global_value"}
        result = self.config.get_resolved_config_value("attribute", resource_defaults, global_defaults)
        assert result == "global_value"

        # Test when value doesn't exist in either defaults
        resource_defaults = {}
        global_defaults = {}
        result = self.config.get_resolved_config_value("attribute", resource_defaults, global_defaults)
        assert result is None

        # Test with None defaults
        result = self.config.get_resolved_config_value("attribute", None, None)
        assert result is None

        # Test when attribute exists in both defaults (resource should take precedence)
        resource_defaults = {"attribute": "resource_value"}
        global_defaults = {"attribute": "global_value"}
        result = self.config.get_resolved_config_value("attribute", resource_defaults, global_defaults)
        assert result == "resource_value"

        # Test logging behavior
        with patch('sagemaker.utils.config.config_manager.logger') as mock_logger:
            self.config.get_resolved_config_value("missing_attribute", {}, {})
            mock_logger.debug.assert_called_once_with(
                "Configurable value missing_attribute not entered in parameters or present in the Config")

