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
"""Constants used in the code_generator modules."""
import os

CLASS_METHODS = set(["create", "add", "register", "import", "list", "get"])
OBJECT_METHODS = set(
    ["refresh", "delete", "update", "start", "stop", "deregister", "wait", "wait_for_status"]
)

TERMINAL_STATES = set(["Completed", "Stopped", "Deleted", "Failed", "Succeeded", "Cancelled"])

RESOURCE_WITH_LOGS = set(["TrainingJob", "ProcessingJob", "TransformJob"])

CONFIGURABLE_ATTRIBUTE_SUBSTRINGS = [
    "kms",
    "s3",
    "subnet",
    "tags",
    "role",
    "security_group",
]

BASIC_JSON_TYPES_TO_PYTHON_TYPES = {
    "string": "StrPipeVar",
    "integer": "int",
    "boolean": "bool",
    "long": "int",
    "float": "float",
    "map": "dict",
    "double": "float",
    "list": "list",
    "timestamp": "datetime.datetime",
    "blob": "Any",
}

BASIC_RETURN_TYPES = {"str", "int", "bool", "float", "datetime.datetime"}

SHAPE_DAG_FILE_PATH = os.getcwd() + "/src/sagemaker/core/utils/code_injection/shape_dag.py"
PYTHON_TYPES_TO_BASIC_JSON_TYPES = {
    "str": "string",
    "StrPipeVar": "string",
    "int": "integer",
    "bool": "boolean",
    "float": "double",
    "datetime.datetime": "timestamp",
}

LICENCES_STRING = """
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
"""

LOGGER_STRING = """
logger = get_textual_rich_logger(__name__)

"""

# TODO: The file name should be injected, we should update it to be more generic
ADDITIONAL_OPERATION_FILE_PATH = (
    os.getcwd() + "/src/sagemaker/core/tools/additional_operations.json"
)

# Get the package root directory (sagemaker-core/)
# When installed, __file__ points to site-packages, so we need to find the actual source/sample location
_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

SERVICE_JSON_FILE_PATH = os.path.join(_PACKAGE_ROOT, "sample/sagemaker/2017-07-24/service-2.json")
RUNTIME_SERVICE_JSON_FILE_PATH = os.path.join(
    _PACKAGE_ROOT, "sample/sagemaker-runtime/2017-05-13/service-2.json"
)
FEATURE_STORE_SERVICE_JSON_FILE_PATH = os.path.join(
    _PACKAGE_ROOT, "sample/sagemaker-featurestore-runtime/2020-07-01/service-2.json"
)
METRICS_SERVICE_JSON_FILE_PATH = os.path.join(
    _PACKAGE_ROOT, "sample/sagemaker-metrics/2022-09-30/service-2.json"
)

GENERATED_CLASSES_LOCATION = os.getcwd() + "/src/sagemaker/core"
UTILS_CODEGEN_FILE_NAME = "utils.py"
INTELLIGENT_DEFAULTS_HELPER_CODEGEN_FILE_NAME = "intelligent_defaults_helper.py"

RESOURCES_CODEGEN_FILE_NAME = "resources.py"

SHAPES_CODEGEN_FILE_NAME = "shapes.py"

CONFIG_SCHEMA_FILE_NAME = "config_schema.py"

API_COVERAGE_JSON_FILE_PATH = os.getcwd() + "/src/sagemaker/core/tools/api_coverage.json"
