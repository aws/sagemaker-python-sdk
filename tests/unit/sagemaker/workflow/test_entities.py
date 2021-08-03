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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import pytest

from enum import Enum

from sagemaker.workflow.entities import (
    DefaultEnumMeta,
    Entity,
)


class CustomEntity(Entity):
    def __init__(self, foo):
        self.foo = foo

    def to_request(self):
        return {"foo": self.foo}


class CustomEnum(Enum, metaclass=DefaultEnumMeta):
    A = 1
    B = 2


@pytest.fixture
def custom_entity():
    return CustomEntity(1)


@pytest.fixture
def custom_entity_list():
    return [CustomEntity(1), CustomEntity(2)]


def test_entity(custom_entity):
    request_struct = {"foo": 1}
    assert custom_entity.to_request() == request_struct


def test_default_enum_meta():
    assert CustomEnum().value == 1
