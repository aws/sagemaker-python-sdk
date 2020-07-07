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
from __future__ import absolute_import

import io

from sagemaker.deserializers import StringDeserializer


def test_string_deserializer_plain_text():
    deserializer = StringDeserializer()

    result = deserializer.deserialize("Hello, world!", "text/plain")

    assert result == "Hello, world!"


def test_string_deserializer_octet_stream():
    deserializer = StringDeserializer()

    result = deserializer.deserialize(io.BytesIO(b"Hello, world!"), "application/octet-stream")

    assert result == "Hello, world!"
