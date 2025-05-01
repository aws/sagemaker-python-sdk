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
"""Constants used in the code_injection modules."""
from enum import Enum

BASIC_TYPES = ["string", "boolean", "integer", "long", "double", "timestamp", "float"]
STRUCTURE_TYPE = "structure"
MAP_TYPE = "map"
LIST_TYPE = "list"


class Color(Enum):
    RED = "rgb(215,0,0)"
    GREEN = "rgb(0,135,0)"
    BLUE = "rgb(0,105,255)"
    YELLOW = "rgb(215,175,0)"
    PURPLE = "rgb(225,0,225)"
    BRIGHT_RED = "rgb(255,0,0)"
