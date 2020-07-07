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

from sagemaker.deserializers import BytesDeserializer, StreamDeserializer


def test_bytes_deserializer():
    deserializer = BytesDeserializer()

    result = deserializer.deserialize(io.BytesIO(b"[1, 2, 3]"), "application/json")

    assert result == b"[1, 2, 3]"


def test_stream_deserializer():
    deserializer = StreamDeserializer()

    stream, content_type = deserializer.deserialize(io.BytesIO(b"[1, 2, 3]"), "application/json")
    result = stream.read()

    assert result == b"[1, 2, 3]"
    assert content_type == "application/json"
