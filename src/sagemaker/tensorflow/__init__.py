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
"""Placeholder docstring"""
from __future__ import absolute_import

import sys
import os

# Hack to use our local copy of tensorflow_serving.apis, which contains the protobuf-generated
# classes for tensorflow serving. Currently tensorflow_serving_api can only be pip-installed for
# python 2.
sys.path.append(os.path.dirname(__file__))

from sagemaker.tensorflow.estimator import (  # noqa: E402, F401 # pylint: disable=wrong-import-position
    TensorFlow,
)
from sagemaker.tensorflow.model import (  # noqa: E402, F401 # pylint: disable=wrong-import-position
    TensorFlowModel,
    TensorFlowPredictor,
)
