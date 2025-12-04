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

import datetime
import logging
import os
import re
import subprocess

from boto3.session import Session
from botocore.config import Config
from rich import reconfigure
from rich.console import Console
from rich.logging import RichHandler
from rich.style import Style
from rich.theme import Theme
from rich.traceback import install
from typing import Any, Dict, List, TypeVar, Generic, Type
from sagemaker.core.utils.code_injection.codec import transform
from sagemaker.core.utils.code_injection.constants import Color
from sagemaker.core.utils.user_agent import get_user_agent_extra_suffix


def add_indent(text, num_spaces=4):
    """
    Add customizable indent spaces to a given text.
    Parameters:
        text (str): The text to which the indent spaces will be added.
        num_spaces (int): Number of spaces to be added for each level of indentation. Default is 4.
    Returns:
        str: The text with added indent spaces.
    """
    indent = " " * num_spaces
    lines = text.split("\n")
    indented_text = "\n".join(indent + line for line in lines)
    return indented_text.rstrip(" ")


def clean_documentaion(documentation):
    documentation = re.sub(r"<\/?p>", "", documentation)
    documentation = re.sub(r"<\/?code>", "'", documentation)
    return documentation


def convert_to_snake_case(entity_name):
    """
    Convert a string to snake_case.
    Args:
        entity_name (str): The string to convert.
    Returns:
        str: The converted string in snake_case.
    """
    snake_case = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", entity_name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake_case).lower()


def snake_to_pascal(snake_str):
    """
    Convert a snake_case string to PascalCase.
    Args:
        snake_str (str): The snake_case string to be converted.
    Returns:
        str: The PascalCase string.
    """
    components = snake_str.split("_")
    return "".join(x.title() for x in components[0:])


def reformat_file_with_black(filename):
    try:
        # Run black with specific options using subprocess
        subprocess.run(["black", "-l", "100", filename], check=True)
        print(f"File '{filename}' reformatted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while reformatting '{filename}': {e}")


def remove_html_tags(text):
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def escape_special_rst_characters(text):
    # List of special characters that need to be escaped in reStructuredText
    special_characters = ["*", "|"]

    for char in special_characters:
        # Use a regex to find the special character if preceded by a space
        pattern = rf"(?<=\s){re.escape(char)}"
        text = re.sub(pattern, rf"\\{char}", text)

    return text


def get_textual_rich_theme() -> Theme:
    """
    Get a textual rich theme with customized styling.

    Returns:
        Theme: A textual rich theme
    """
    return Theme(
        {
            "logging.level.info": Style(color=Color.BLUE.value, bold=True),
            "logging.level.debug": Style(color=Color.GREEN.value, bold=True),
            "logging.level.warning": Style(color=Color.YELLOW.value, bold=True),
            "logging.level.error": Style(color=Color.RED.value, bold=True),
            "logging.keyword": Style(color=Color.YELLOW.value, bold=True),
            "repr.attrib_name": Style(color=Color.YELLOW.value, italic=False),
            "repr.attrib_value": Style(color=Color.PURPLE.value, italic=False),
            "repr.bool_true": Style(color=Color.GREEN.value, italic=True),
            "repr.bool_false": Style(color=Color.RED.value, italic=True),
            "repr.call": Style(color=Color.PURPLE.value, bold=True),
            "repr.none": Style(color=Color.PURPLE.value, italic=True),
            "repr.str": Style(color=Color.GREEN.value),
            "repr.path": Style(color=Color.PURPLE.value),
            "repr.filename": Style(color=Color.PURPLE.value),
            "repr.url": Style(color=Color.BLUE.value, underline=True),
            "repr.tag_name": Style(color=Color.PURPLE.value, bold=True),
            "repr.ipv4": Style.null(),
            "repr.ipv6": Style.null(),
            "repr.eui48": Style.null(),
            "repr.eui64": Style.null(),
            "json.bool_true": Style(color=Color.GREEN.value, italic=True),
            "json.bool_false": Style(color=Color.RED.value, italic=True),
            "json.null": Style(color=Color.PURPLE.value, italic=True),
            "json.str": Style(color=Color.GREEN.value),
            "json.key": Style(color=Color.BLUE.value, bold=True),
            "traceback.error": Style(color=Color.BRIGHT_RED.value, italic=True),
            "traceback.border": Style(color=Color.BRIGHT_RED.value),
            "traceback.title": Style(color=Color.BRIGHT_RED.value, bold=True),
        }
    )


textual_rich_console_and_traceback_enabled = False


def enable_textual_rich_console_and_traceback():
    """
    Reconfigure the global textual rich console with the customized theme
        and enable textual rich error traceback
    """
    global textual_rich_console_and_traceback_enabled
    if not textual_rich_console_and_traceback_enabled:
        theme = get_textual_rich_theme()
        reconfigure(theme=theme)
        console = Console(theme=theme)
        install(console=console)
        textual_rich_console_and_traceback_enabled = True


def get_rich_handler():
    handler = RichHandler(markup=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def get_textual_rich_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Get a logger with textual rich handler.

    Args:
        name (str): The name of the logger
        log_level (str): The log level to set.
            Accepted values are: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
            Defaults to the value of "INFO".

    Return:
        logging.Logger: A textial rich logger.

    """
    enable_textual_rich_console_and_traceback()
    handler = get_rich_handler()
    logging.basicConfig(level=getattr(logging, log_level), handlers=[handler])
    logger = logging.getLogger(name)

    return logger


logger = get_textual_rich_logger(__name__)

T = TypeVar("T")

SPECIAL_SNAKE_TO_PASCAL_MAPPINGS = {
    "volume_size_in_g_b": "VolumeSizeInGB",
    "volume_size_in_gb": "VolumeSizeInGB",
}


def configure_logging(log_level=None):
    """Configure the logging configuration based on log level.

    Usage:
        Set Environment Variable LOG_LEVEL to DEBUG to see debug logs
        configure_logging()
        configure_logging("DEBUG")

    Args:
        log_level (str): The log level to set.
            Accepted values are: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
            Defaults to the value of the LOG_LEVEL environment variable.
            If argument/environment variable is not set, defaults to "INFO".

    Raises:
        AttributeError: If the log level is invalid.
    """

    if not log_level:
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    _logger = logging.getLogger()
    _logger.setLevel(getattr(logging, log_level))
    # reset any currently associated handlers with log level
    for handler in _logger.handlers:
        _logger.removeHandler(handler)
    rich_handler = get_rich_handler()
    _logger.addHandler(rich_handler)


def is_snake_case(s: str):
    if not s:
        return False
    if s[0].isupper():
        return False
    if not s.islower() and not s.isalnum():
        return False
    if s.startswith("_") or s.endswith("_"):
        return False
    if "__" in s:
        return False
    return True


def snake_to_pascal(snake_str):
    """
    Convert a snake_case string to PascalCase.

    Args:
        snake_str (str): The snake_case string to be converted.

    Returns:
        str: The PascalCase string.

    """
    if pascal_str := SPECIAL_SNAKE_TO_PASCAL_MAPPINGS.get(snake_str):
        return pascal_str
    components = snake_str.split("_")
    return "".join(x.title() for x in components[0:])


def pascal_to_snake(pascal_str):
    """
    Converts a PascalCase string to snake_case.

    Args:
        pascal_str (str): The PascalCase string to be converted.

    Returns:
        str: The converted snake_case string.
    """
    snake_case = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", pascal_str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake_case).lower()


def is_not_primitive(obj):
    return not isinstance(obj, (int, float, str, bool, datetime.datetime))


def is_not_str_dict(obj):
    return not isinstance(obj, dict) or not all(isinstance(k, str) for k in obj.keys())


def is_primitive_list(obj):
    return all(not is_not_primitive(s) for s in obj)


def is_primitive_class(cls):
    return cls in (str, int, bool, float, datetime.datetime)


class Unassigned:
    """A custom type used to signify an undefined optional argument."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


class SingletonMeta(type):
    """
    Singleton metaclass. Ensures that a single instance of a class using this metaclass is created.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Overrides the call method to return an existing instance of the class if it exists,
        or create a new one if it doesn't.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class SageMakerClient(metaclass=SingletonMeta):
    """
    A singleton class for creating a SageMaker client.
    """

    def __init__(
        self,
        session: Session = None,
        region_name: str = None,
        config: Config = None,
    ):
        """
        Initializes the SageMakerClient with a boto3 session, region name, and service name.
        Creates a boto3 client using the provided session, region, and service.
        """
        if session is None:
            logger.debug("No boto3 session provided. Creating a new session.")
            session = Session(region_name=region_name)

        if region_name is None:
            logger.warning("No region provided. Using default region.")
            region_name = session.region_name

        if config is None:
            logger.debug("No config provided. Using default config.")
            config = Config(retries={"max_attempts": 10, "mode": "standard"})

        self.config = Config(user_agent_extra=get_user_agent_extra_suffix())
        self.session = session
        self.region_name = region_name
        # Read region from environment variable, default to us-west-2
        import os
        env_region = os.environ.get('SAGEMAKER_REGION', region_name)
        env_stage = os.environ.get('SAGEMAKER_STAGE', 'prod')  # default to gamma
        logger.info(f"Runs on sagemaker {env_stage}, region:{env_region}")


        self.sagemaker_client = session.client(
            "sagemaker",
            region_name=env_region,
            config=self.config,
        )
        
        self.sagemaker_runtime_client = session.client(
            "sagemaker-runtime", region_name, config=self.config
        )
        self.sagemaker_featurestore_runtime_client = session.client(
            "sagemaker-featurestore-runtime", region_name, config=self.config
        )
        self.sagemaker_metrics_client = session.client(
            "sagemaker-metrics", region_name, config=self.config
        )

    def get_client(self, service_name: str) -> Any:
        """
        Get the client of corresponding service

        Args:
            service_name (str): the service name

        Returns:
            Any: the client of that service
        """
        service_name = service_name.replace("-", "_")
        return getattr(self, service_name + "_client")


class ResourceIterator(Generic[T]):
    """ResourceIterator class to iterate over a list of resources."""

    def __init__(
        self,
        client: SageMakerClient,
        summaries_key: str,
        summary_name: str,
        resource_cls: Type[T],
        list_method: str,
        list_method_kwargs: dict = {},
        custom_key_mapping: dict = None,
    ):
        """Initialize a ResourceIterator object

        Args:
            client (SageMakerClient): The sagemaker client object used to make list method calls.
            summaries_key (str): The summaries key string used to access the list of summaries in the response.
            summary_name (str): The summary name used to transform list response data.
            resource_cls (Type[T]): The resource class to be instantiated for each resource object.
            list_method (str): The list method string used to make list calls to the client.
            list_method_kwargs (dict, optional): The kwargs used to make list method calls. Defaults to {}.
            custom_key_mapping (dict, optional): The custom key mapping used to map keys from summary object to those expected from resource object during initialization. Defaults to None.
        """
        self.summaries_key = summaries_key
        self.summary_name = summary_name
        self.client = client
        self.list_method = list_method
        self.list_method_kwargs = list_method_kwargs
        self.custom_key_mapping = custom_key_mapping

        self.resource_cls = resource_cls
        self.index = 0
        self.summary_list = []
        self.next_token = None

    def __iter__(self):
        return self

    def __next__(self) -> T:

        # If there are summaries in the summary_list, return the next summary
        if len(self.summary_list) > 0 and self.index < len(self.summary_list):
            # Get the next summary from the resource summary_list
            summary = self.summary_list[self.index]
            self.index += 1

            # Initialize the resource object
            if is_primitive_class(self.resource_cls):
                # If the resource class is a primitive class, there will be only one element in the summary
                resource_object = list(summary.values())[0]
            else:
                # Transform the resource summary into format to initialize object
                init_data = transform(summary, self.summary_name)

                if self.custom_key_mapping:
                    init_data = {self.custom_key_mapping.get(k, k): v for k, v in init_data.items()}

                # Filter out the fields that are not in the resource class
                fields = self.resource_cls.__annotations__
                init_data = {k: v for k, v in init_data.items() if k in fields}

                resource_object = self.resource_cls(**init_data)

            # If the resource object has refresh method, refresh and return it
            if hasattr(resource_object, "refresh"):
                resource_object.refresh()
            return resource_object

        # If index reached the end of summary_list, and there is no next token, raise StopIteration
        elif (
            len(self.summary_list) > 0
            and self.index >= len(self.summary_list)
            and (not self.next_token)
        ):
            raise StopIteration

        # Otherwise, get the next page of summaries by calling the list method with the next token if available
        else:
            if self.next_token:
                response = getattr(self.client, self.list_method)(
                    NextToken=self.next_token, **self.list_method_kwargs
                )
            else:
                response = getattr(self.client, self.list_method)(**self.list_method_kwargs)

            self.summary_list = response.get(self.summaries_key, [])
            self.next_token = response.get("NextToken", None)
            self.index = 0

            # If list_method returned an empty list, raise StopIteration
            if len(self.summary_list) == 0:
                raise StopIteration

            return self.__next__()


def serialize(value: Any) -> Any:
    """
    Serialize an object recursively by converting all objects to JSON-serializable types

    Args:
       value (Any): The object to be serialized

    Returns:
        Any: The serialized object
    """
    from sagemaker.core.helper.pipeline_variable import PipelineVariable

    if value is None or isinstance(value, type(Unassigned())):
        return None
    elif isinstance(value, PipelineVariable):
        # Return PipelineVariables as-is (Join, ExecutionVariables, etc.)
        return value
    elif isinstance(value, Dict):
        # if the value is a dict, use _serialize_dict() to serialize it recursively
        return _serialize_dict(value)
    elif isinstance(value, List):
        # if the value is a dict, use _serialize_list() to serialize it recursively
        return _serialize_list(value)
    elif is_not_primitive(value):
        # if the value is a dict, use _serialize_shape() to serialize it recursively
        return _serialize_shape(value)
    else:
        return value


def _serialize_dict(value: Dict) -> dict:
    """
    Serialize all values in a dict recursively

    Args:
       value (dict): The dict to be serialized

    Returns:
        dict: The serialized dict
    """
    serialized_dict = {}
    for k, v in value.items():
        if serialize_result := serialize(v):
            serialized_dict.update({k: serialize_result})
    return serialized_dict


def _serialize_list(value: List) -> list:
    """
    Serialize all objects in a list

    Args:
       value (list): The dict to be serialized

    Returns:
        list: The serialized list
    """
    serialized_list = []
    for v in value:
        if serialize_result := serialize(v):
            serialized_list.append(serialize_result)
    return serialized_list


def _serialize_shape(value: Any) -> dict:
    """
    Serialize a shape object defined in resource.py or shape.py to a dict

    Args:
       value (Any): The shape to be serialized

    Returns:
        dict: The dict of serialized shape
    """
    serialized_dict = {}
    for k, v in vars(value).items():
        if serialize_result := serialize(v):
            key = snake_to_pascal(k) if is_snake_case(k) else k
            serialized_dict.update({key[0].upper() + key[1:]: serialize_result})
    return serialized_dict
