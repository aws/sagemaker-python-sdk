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
from __future__ import absolute_import

import logging

from sagemaker.remote_function.logging_config import get_logger


def test_logger_config():
    logging.basicConfig(level=logging.INFO)

    logger_1 = get_logger()
    assert len(logger_1.handlers) == 1

    logger_2 = get_logger()
    assert logger_2 is logger_1
    assert len(logger_2.handlers) == 1
