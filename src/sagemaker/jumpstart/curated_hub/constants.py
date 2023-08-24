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
"""This module accessors for the SageMaker JumpStart Curated Hub."""
from __future__ import absolute_import
from enum import Enum


CURATED_HUB_DEFAULT_DESCRIPTION = "This is a curated hub."
CURATED_HUB_CONTENT_TYPE = "Model"
CURATED_HUB_DEFAULT_DOCUMENT_SCHEMA_VERSION = "1.0.0"

class HubContentType(str, Enum):
    """Hub hub content types"""

    MODEL = "Model"
    NOTEBOOK = "Notebook"
