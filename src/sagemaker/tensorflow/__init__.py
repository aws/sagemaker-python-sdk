# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import sys

import os

# Hack to use our local copy of tensorflow_serving.apis, which contains the protobuf-generated
# classes for tensorflow serving. Currently tensorflow_serving_api can only be pip-installed for python 2.
sys.path.append(os.path.dirname(__file__))

from distutils.version import LooseVersion  # noqa: E402
import tensorflow  # noqa: E402

if LooseVersion(tensorflow.__version__) < LooseVersion("1.3.0"):
    message = 'Tensorflow version must be >= 1.3.0. Current version: {}'.format(tensorflow.__version__)
    raise AssertionError(message)

from sagemaker.tensorflow.estimator import TensorFlow  # noqa: E402
from sagemaker.tensorflow.model import TensorFlowModel, TensorFlowPredictor  # noqa: E402

__all__ = [TensorFlow, TensorFlowModel, TensorFlowPredictor]
