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
"""DJL Handler Template

Getting Started DJL Handle provided via ModelBuilder.
Feel free to re-purpose this script for your DJL usecase
and re-deploy via ModelBuilder().deploy().
"""
from __future__ import absolute_import

from djl_python.inputs import Input
from djl_python.outputs import Output


class HandleTemplate:
    """A DJL Handler class template that uses the default DeepSpeed, FasterTransformer, and HuggingFaceAccelerate Handlers

    Reference the default handlers here:
    - https://github.com/deepjavalibrary/djl-serving/blob/master/engines/python/setup/djl_python/deepspeed.py
    - https://github.com/deepjavalibrary/djl-serving/blob/master/engines/python/setup/djl_python/fastertransformer.py
    - https://github.com/deepjavalibrary/djl-serving/blob/master/engines/python/setup/djl_python/huggingface.py
    """

    def __init__(self):
        self.initialized = False
        self.handle = None

    def initialize(self, inputs: Input):
        """Template method to load you model with specified engine."""
        self.initialized = True

        if "DeepSpeed" == inputs.get_property("engine"):
            from djl_python.deepspeed import handle
        elif "FasterTransformer" == inputs.get_property("engine"):
            from djl_python.fastertransformer import handle
        else:
            from djl_python.huggingface import handle

        self._handle = handle

    def inference(self, inputs: Input):
        """Template method used to invoke the model. Please implement this if you'd like to construct your own script"""


_handle_template = HandleTemplate()


def handle(inputs: Input) -> Output:
    """Driver function required by djl-serving"""
    if not _handle_template.initialized:
        _handle_template.initialize(inputs)

    return _handle_template._handle(inputs)
