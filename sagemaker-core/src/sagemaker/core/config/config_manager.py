# sagemaker_config.py

import pathlib
import copy
import inspect
import os
from typing import List, Optional
import boto3
import yaml
import jsonschema
from platformdirs import site_config_dir, user_config_dir
from botocore.utils import merge_dicts
from six.moves.urllib.parse import urlparse
from sagemaker.core.config.config_schema import SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA
from sagemaker.core.config.config_utils import (
    non_repeating_log_factory,
    get_sagemaker_config_logger,
    _log_sagemaker_config_single_substitution,
    _log_sagemaker_config_merge,
)
from functools import lru_cache

logger = get_sagemaker_config_logger()
log_info_function = non_repeating_log_factory(logger, "info")


class SageMakerConfig:
    _APP_NAME = "sagemaker"
    _CONFIG_FILE_NAME = "config.yaml"
    _DEFAULT_ADMIN_CONFIG_FILE_PATH = os.path.join(site_config_dir(_APP_NAME), _CONFIG_FILE_NAME)
    _DEFAULT_USER_CONFIG_FILE_PATH = os.path.join(user_config_dir(_APP_NAME), _CONFIG_FILE_NAME)
    _DEFAULT_LOCAL_MODE_CONFIG_FILE_PATH = os.path.join(
        os.path.expanduser("~"), ".sagemaker", _CONFIG_FILE_NAME
    )
    ENV_VARIABLE_ADMIN_CONFIG_OVERRIDE = "SAGEMAKER_ADMIN_CONFIG_OVERRIDE"
    ENV_VARIABLE_USER_CONFIG_OVERRIDE = "SAGEMAKER_USER_CONFIG_OVERRIDE"
    S3_PREFIX = "s3://"

    def __init__(self):
        self.logger = get_sagemaker_config_logger()
        self.log_info_function = non_repeating_log_factory(self.logger, "info")

    def load_sagemaker_config(
        self,
        additional_config_paths: Optional[List[str]] = None,
        s3_resource=None,
        repeat_log: bool = False,
    ) -> dict:
        default_config_path = os.getenv(
            self.ENV_VARIABLE_ADMIN_CONFIG_OVERRIDE, self._DEFAULT_ADMIN_CONFIG_FILE_PATH
        )
        user_config_path = os.getenv(
            self.ENV_VARIABLE_USER_CONFIG_OVERRIDE, self._DEFAULT_USER_CONFIG_FILE_PATH
        )
        config_paths = [default_config_path, user_config_path]
        if additional_config_paths:
            config_paths += additional_config_paths
        config_paths = list(filter(lambda item: item is not None, config_paths))
        merged_config = {}

        log_info = self.log_info_function
        if repeat_log:
            log_info = self.logger.info

        for file_path in config_paths:
            config_from_file = {}
            if file_path.startswith(self.S3_PREFIX):
                config_from_file = self._load_config_from_s3(file_path, s3_resource)
            else:
                try:
                    config_from_file = self._load_config_from_file(file_path)
                except ValueError as error:
                    if file_path not in (
                        self._DEFAULT_ADMIN_CONFIG_FILE_PATH,
                        self._DEFAULT_USER_CONFIG_FILE_PATH,
                    ):
                        raise
                    self.logger.debug(error)
            if config_from_file:
                self.validate_sagemaker_config(config_from_file)
                merge_dicts(merged_config, config_from_file)
                log_info("Fetched defaults config from location: %s", file_path)
            else:
                log_info("Not applying SDK defaults from location: %s", file_path)

        return merged_config

    @staticmethod
    def validate_sagemaker_config(sagemaker_config: Optional[dict] = None):
        jsonschema.validate(sagemaker_config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)

    def load_local_mode_config(self) -> Optional[dict]:
        try:
            content = self._load_config_from_file(self._DEFAULT_LOCAL_MODE_CONFIG_FILE_PATH)
        except ValueError:
            content = None
        return content

    def _load_config_from_file(self, file_path: str) -> dict:
        inferred_file_path = file_path
        if os.path.isdir(file_path):
            inferred_file_path = os.path.join(file_path, self._CONFIG_FILE_NAME)
        if not os.path.exists(inferred_file_path):
            raise ValueError(
                f"Unable to load the config file from the location: {file_path}"
                f"Provide a valid file path"
            )
        self.logger.debug("Fetching defaults config from location: %s", file_path)
        with open(inferred_file_path, "r") as f:
            content = yaml.safe_load(f)
        return content

    def _load_config_from_s3(self, s3_uri, s3_resource_for_config) -> dict:
        if not s3_resource_for_config:
            boto_session = boto3.DEFAULT_SESSION or boto3.Session()
            boto_region_name = boto_session.region_name
            if boto_region_name is None:
                raise ValueError(
                    "Must setup local AWS configuration with a region supported by SageMaker."
                )
            s3_resource_for_config = boto_session.resource("s3", region_name=boto_region_name)

        self.logger.debug("Fetching defaults config from location: %s", s3_uri)
        inferred_s3_uri = self._get_inferred_s3_uri(s3_uri, s3_resource_for_config)
        parsed_url = urlparse(inferred_s3_uri)
        bucket, key_prefix = parsed_url.netloc, parsed_url.path.lstrip("/")
        s3_object = s3_resource_for_config.Object(bucket, key_prefix)
        s3_file_content = s3_object.get()["Body"].read()
        return yaml.safe_load(s3_file_content.decode("utf-8"))

    def _get_inferred_s3_uri(self, s3_uri, s3_resource_for_config):
        parsed_url = urlparse(s3_uri)
        bucket, key_prefix = parsed_url.netloc, parsed_url.path.lstrip("/")
        s3_bucket = s3_resource_for_config.Bucket(name=bucket)
        s3_objects = s3_bucket.objects.filter(Prefix=key_prefix).all()
        s3_files_with_same_prefix = [
            f"{self.S3_PREFIX}{bucket}/{s3_object.key}" for s3_object in s3_objects
        ]
        if len(s3_files_with_same_prefix) == 0:
            raise ValueError(f"Provide a valid S3 path instead of {s3_uri}")
        if len(s3_files_with_same_prefix) > 1:
            inferred_s3_uri = str(pathlib.PurePosixPath(s3_uri, self._CONFIG_FILE_NAME)).replace(
                "s3:/", "s3://"
            )
            if inferred_s3_uri not in s3_files_with_same_prefix:
                raise ValueError(
                    f"Provide an S3 URI of a directory that has a {self._CONFIG_FILE_NAME} file."
                )
            return inferred_s3_uri
        return s3_uri

    @staticmethod
    def get_config_value(key_path, config):
        """Placeholder Docstring"""
        if config is None:
            return None

        current_section = config
        for key in key_path.split("."):
            if key in current_section:
                current_section = current_section[key]
            else:
                return None

        return current_section

    @staticmethod
    def get_nested_value(dictionary: dict, nested_keys: List[str]):
        """Returns a nested value from the given dictionary, and None if none present.

        Raises
            ValueError if the dictionary structure does not match the nested_keys
        """
        if (
            dictionary is not None
            and isinstance(dictionary, dict)
            and nested_keys is not None
            and len(nested_keys) > 0
        ):

            current_section = dictionary

            for key in nested_keys[:-1]:
                current_section = current_section.get(key, None)
                if current_section is None:
                    # means the full path of nested_keys doesnt exist in the dictionary
                    # or the value was set to None
                    return None
                if not isinstance(current_section, dict):
                    raise ValueError(
                        "Unexpected structure of dictionary.",
                        "Expected value of type dict at key '{}' but got '{}' for dict '{}'".format(
                            key, current_section, dictionary
                        ),
                    )
            return current_section.get(nested_keys[-1], None)

        return None

    @staticmethod
    def set_nested_value(dictionary: dict, nested_keys: List[str], value_to_set: object):
        """Sets a nested value in a dictionary.

        This sets a nested value inside the given dictionary and returns the new dictionary. Note: if
        provided an unintended list of nested keys, this can overwrite an unexpected part of the dict.
        Recommended to use after a check with get_nested_value first
        """

        if dictionary is None:
            dictionary = {}

        if (
            dictionary is not None
            and isinstance(dictionary, dict)
            and nested_keys is not None
            and len(nested_keys) > 0
        ):
            current_section = dictionary
            for key in nested_keys[:-1]:
                if (
                    key not in current_section
                    or current_section[key] is None
                    or not isinstance(current_section[key], dict)
                ):
                    current_section[key] = {}
                current_section = current_section[key]

            current_section[nested_keys[-1]] = value_to_set
        return dictionary

    def resolve_value_from_config(
        self,
        direct_input=None,
        config_path: str = None,
        default_value=None,
        sagemaker_session=None,
        sagemaker_config: dict = None,
    ):
        """Decides which value for the caller to use.

        Note: This method incorporates information from the sagemaker config.

        Uses this order of prioritization:
        1. direct_input
        2. config value
        3. default_value
        4. None

        Args:
            direct_input: The value that the caller of this method starts with. Usually this is an
                input to the caller's class or method.
            config_path (str): A string denoting the path used to lookup the value in the
                sagemaker config.
            default_value: The value used if not present elsewhere.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
                SageMaker interactions (default: None).
            sagemaker_config (dict): The sdk defaults config that is normally accessed through a
                Session object by doing `session.sagemaker_config`. (default: None) This parameter will
                be checked for the config value if (and only if) sagemaker_session is None. This
                parameter exists for the rare cases where the user provided no Session but a default
                Session cannot be initialized before config injection is needed. In that case,
                the config dictionary may be loaded and passed here before a default Session object
                is created.

        Returns:
            The value that should be used by the caller
        """

        config_value = (
            self.get_sagemaker_config_value(
                sagemaker_session, config_path, sagemaker_config=sagemaker_config
            )
            if config_path
            else None
        )
        _log_sagemaker_config_single_substitution(direct_input, config_value, config_path)

        if direct_input is not None:
            return direct_input

        if config_value is not None:
            return config_value

        return default_value

    def get_sagemaker_config_value(self, sagemaker_session, key, sagemaker_config: dict = None):
        """Returns the value that corresponds to the provided key from the configuration file.

        Args:
            key: Key Path of the config file entry.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
                SageMaker interactions.
            sagemaker_config (dict): The sdk defaults config that is normally accessed through a
                Session object by doing `session.sagemaker_config`. (default: None) This parameter will
                be checked for the config value if (and only if) sagemaker_session is None. This
                parameter exists for the rare cases where no Session provided but a default Session
                cannot be initialized before config injection is needed. In that case, the config
                dictionary may be loaded and passed here before a default Session object is created.

        Returns:
            object: The corresponding default value in the configuration file.
        """
        if sagemaker_session and hasattr(sagemaker_session, "sagemaker_config"):
            config_to_check = sagemaker_session.sagemaker_config
        else:
            config_to_check = sagemaker_config

        if not config_to_check:
            return None

        self.validate_sagemaker_config(config_to_check)
        config_value = self.get_config_value(key, config_to_check)
        # Copy the value so any modifications to the output will not modify the source config
        return copy.deepcopy(config_value)

    def resolve_class_attribute_from_config(
        self,
        clazz: Optional[type],
        instance: Optional[object],
        attribute: str,
        config_path: str,
        default_value=None,
        sagemaker_session=None,
    ):
        """Utility method that merges config values to data classes.

        Takes an instance of a class and, if not already set, sets the instance's attribute to a
        value fetched from the sagemaker_config or the default_value.

        Uses this order of prioritization to determine what the value of the attribute should be:
        1. current value of attribute
        2. config value
        3. default_value
        4. does not set it

        Args:
            clazz (Optional[type]): Class of 'instance'. Used to generate a new instance if the
                   instance is None. If None is provided here, no new object will be created
                   if 'instance' doesnt exist. Note: if provided, the constructor should set default
                   values to None; Otherwise, the constructor's non-None default will be left
                   as-is even if a config value was defined.
            instance (Optional[object]): instance of the Class 'clazz' that has an attribute
                     of 'attribute' to set
            attribute (str): attribute of the instance to set if not already set
            config_path (str): a string denoting the path to use to lookup the config value in the
                               sagemaker config
            default_value: the value to use if not present elsewhere
            sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
                    SageMaker interactions (default: None).

        Returns:
            The updated class instance that should be used by the caller instead of the
            'instance' parameter that was passed in.
        """
        config_value = self.get_sagemaker_config_value(sagemaker_session, config_path)

        if config_value is None and default_value is None:
            # return instance unmodified. Could be None or populated
            return instance

        if instance is None:
            if clazz is None or not inspect.isclass(clazz):
                return instance
            # construct a new instance if the instance does not exist
            instance = clazz()

        if not hasattr(instance, attribute):
            raise TypeError(
                "Unexpected structure of object.",
                "Expected attribute {} to be present inside instance {} of class {}".format(
                    attribute, instance, clazz
                ),
            )

        current_value = getattr(instance, attribute)
        if current_value is None:
            # only set value if object does not already have a value set
            if config_value is not None:
                setattr(instance, attribute, config_value)
            elif default_value is not None:
                setattr(instance, attribute, default_value)

        _log_sagemaker_config_single_substitution(current_value, config_value, config_path)

        return instance

    def resolve_nested_dict_value_from_config(
        self,
        dictionary: dict,
        nested_keys: List[str],
        config_path: str,
        default_value: object = None,
        sagemaker_session=None,
    ):
        """Utility method that sets the value of a key path in a nested dictionary .

        This method takes a dictionary and, if not already set, sets the value for the provided
        list of nested keys to the value fetched from the sagemaker_config or the default_value.

        Uses this order of prioritization to determine what the value of the attribute should be:
        (1) current value of nested key, (2) config value, (3) default_value, (4) does not set it

        Args:
            dictionary: The dict to update.
            nested_keys: The paths of keys where the value should be checked and set if needed.
            config_path (str): A string denoting the path used to find the config value in the
            sagemaker config.
            default_value: The value to use if not present elsewhere.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
                SageMaker interactions (default: None).

        Returns:
            The updated dictionary that should be used by the caller instead of the
            'dictionary' parameter that was passed in.
        """
        config_value = self.get_sagemaker_config_value(sagemaker_session, config_path)

        if config_value is None and default_value is None:
            # if there is nothing to set, return early. And there is no need to traverse through
            # the dictionary or add nested dicts to it
            return dictionary

        try:
            current_nested_value = self.get_nested_value(dictionary, nested_keys)
        except ValueError as e:
            logger.error("Failed to check dictionary for applying sagemaker config: %s", e)
            return dictionary

        if current_nested_value is None:
            # only set value if not already set
            if config_value is not None:
                dictionary = self.set_nested_value(dictionary, nested_keys, config_value)
            elif default_value is not None:
                dictionary = self.set_nested_value(dictionary, nested_keys, default_value)

        _log_sagemaker_config_single_substitution(current_nested_value, config_value, config_path)

        return dictionary

    def update_list_of_dicts_with_values_from_config(
        self,
        input_list,
        config_key_path,
        required_key_paths: List[str] = None,
        union_key_paths: List[List[str]] = None,
        sagemaker_session=None,
    ):
        """Updates a list of dictionaries with missing values that are present in Config.

        In some cases, config file might introduce new parameters which requires certain other
        parameters to be provided as part of the input list. Without those parameters, the underlying
        service will throw an exception. This method provides the capability to specify required key
        paths.

        In some other cases, config file might introduce new parameters but the service API requires
        either an existing parameter or the new parameter that was supplied by config but not both

        Args:
            input_list: The input list that was provided as a method parameter.
            config_key_path: The Key Path in the Config file that corresponds to the input_list
            parameter.
            required_key_paths (List[str]): List of required key paths that should be verified in the
            merged output. If a required key path is missing, we will not perform the merge for that
            item.
            union_key_paths (List[List[str]]): List of List of Key paths for which we need to verify
            whether exactly zero/one of the parameters exist.
            For example: If the resultant dictionary can have either 'X1' or 'X2' as parameter or
            neither but not both, then pass [['X1', 'X2']]
            sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
                SageMaker interactions (default: None).

        Returns:
            No output. In place merge happens.
        """
        if not input_list:
            return
        inputs_copy = copy.deepcopy(input_list)
        inputs_from_config = (
            self.get_sagemaker_config_value(sagemaker_session, config_key_path) or []
        )
        unmodified_inputs_from_config = copy.deepcopy(inputs_from_config)

        for i in range(min(len(input_list), len(inputs_from_config))):
            dict_from_inputs = input_list[i]
            dict_from_config = inputs_from_config[i]
            merge_dicts(dict_from_config, dict_from_inputs)
            # Check if required key paths are present in merged dict (dict_from_config)
            required_key_path_check_passed = self._validate_required_paths_in_a_dict(
                dict_from_config, required_key_paths
            )
            if not required_key_path_check_passed:
                # Don't do the merge, config is introducing a new parameter which needs a
                # corresponding required parameter.
                continue
            union_key_path_check_passed = self._validate_union_key_paths_in_a_dict(
                dict_from_config, union_key_paths
            )
            if not union_key_path_check_passed:
                # Don't do the merge, Union parameters are not obeyed.
                continue
            input_list[i] = dict_from_config

        _log_sagemaker_config_merge(
            source_value=inputs_copy,
            config_value=unmodified_inputs_from_config,
            merged_source_and_config_value=input_list,
            config_key_path=config_key_path,
        )

    def _validate_required_paths_in_a_dict(
        self, source_dict, required_key_paths: List[str] = None
    ) -> bool:
        """Placeholder docstring"""
        if not required_key_paths:
            return True
        for required_key_path in required_key_paths:
            if self.get_config_value(required_key_path, source_dict) is None:
                return False
        return True

    def _validate_union_key_paths_in_a_dict(
        self, source_dict, union_key_paths: List[List[str]] = None
    ) -> bool:
        """Placeholder docstring"""
        if not union_key_paths:
            return True
        for union_key_path in union_key_paths:
            union_parameter_present = False
            for key_path in union_key_path:
                if self.get_config_value(key_path, source_dict):
                    if union_parameter_present:
                        return False
                    union_parameter_present = True
        return True

    def update_nested_dictionary_with_values_from_config(
        self, source_dict, config_key_path, sagemaker_session=None
    ) -> dict:
        """Updates a nested dictionary with missing values that are present in Config.

        Args:
            source_dict: The input nested dictionary that was provided as method parameter.
            config_key_path: The Key Path in the Config file which corresponds to this
            source_dict parameter.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session object, used for
                SageMaker interactions (default: None).

        Returns:
            dict: The merged nested dictionary that is updated with missing values that are present
            in the Config file.
        """
        inferred_config_dict = (
            self.get_sagemaker_config_value(sagemaker_session, config_key_path) or {}
        )
        original_config_dict_value = copy.deepcopy(inferred_config_dict)
        merge_dicts(inferred_config_dict, source_dict or {})

        if original_config_dict_value == {}:
            # The config value is empty. That means either
            # (1) inferred_config_dict equals source_dict, or
            # (2) if source_dict was None, inferred_config_dict equals {}
            # We should return whatever source_dict was to be safe. Because if for example,
            # a VpcConfig is set to {} instead of None, some boto calls will fail due to
            # ParamValidationError (because a VpcConfig was specified but required parameters for
            # the VpcConfig were missing.)

            # Don't need to print because no config value was used or defined
            return source_dict

        _log_sagemaker_config_merge(
            source_value=source_dict,
            config_value=original_config_dict_value,
            merged_source_and_config_value=inferred_config_dict,
            config_key_path=config_key_path,
        )

        return inferred_config_dict

    @lru_cache(maxsize=None)
    def load_default_configs_for_resource_name(self, resource_name: str):
        configs_data = self.load_sagemaker_config()
        if not configs_data:
            logger.debug("No default configurations found for resource: %s", resource_name)
            return {}
        return configs_data["SageMaker"]["PythonSDK"]["Resources"].get(resource_name)

    def get_resolved_config_value(self, attribute, resource_defaults, global_defaults):
        if resource_defaults and attribute in resource_defaults:
            return resource_defaults[attribute]
        if global_defaults and attribute in global_defaults:
            return global_defaults[attribute]
        logger.debug(
            f"Configurable value {attribute} not entered in parameters or present in the Config"
        )
        return None
