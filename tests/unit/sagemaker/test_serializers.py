# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest

from sagemaker import serializers


class BaseSerializerTest:
    def test_instantiate_serializer_without_content_type_attribute(self):
        class StubSerializer(serializers.BaseSerializer):
            def serialize(self, data):
                pass

        with pytest.raises(TypeError):
            StubSerializer()

    def test_instantiate_serializer_without_serialize_method(self):
        class StubSerializer(serializers.BaseSerializer):
            CONTENT_TYPE = "application/json"

        with pytest.raises(TypeError):
            StubSerializer()
