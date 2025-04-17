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

import pytest
from sagemaker.amazon.hyperparameter import Hyperparameter


class Test(object):

    blank = Hyperparameter(name="some-name", data_type=int)
    elizabeth = Hyperparameter(name="elizabeth")
    validated = Hyperparameter(name="validated", validate=lambda value: value > 55, data_type=int)


def test_blank_access():
    x = Test()
    # blank isn't set yet, so accessing it is an error
    with pytest.raises(AttributeError):
        x.blank


def test_blank():
    x = Test()
    x.blank = 82
    assert x.blank == 82


def test_delete():
    x = Test()
    x.blank = 97
    assert x.blank == 97
    del x.blank
    with pytest.raises(AttributeError):
        x.blank


def test_name():
    x = Test()
    with pytest.raises(AttributeError) as excinfo:
        x.elizabeth
        assert "elizabeth" in excinfo


def test_validated():
    x = Test()
    x.validated = 66
    with pytest.raises(ValueError):
        x.validated = 23


def test_data_type():
    x = Test()
    x.validated = 66
    assert isinstance(x.validated, Test.__dict__["validated"].data_type)


def test_from_string():
    x = Test()
    value = 65

    x.validated = value
    from_api = str(value)

    x.validated = from_api
    assert x.validated == value
