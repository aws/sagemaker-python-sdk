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

from sagemaker.cli.compatibility.v2.modifiers import parsing
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call


def test_arg_from_keywords():
    kw_name = "framework_version"
    kw_value = "1.6.0"

    call = ast_call("MXNet({}='{}', py_version='py3', entry_point='run')".format(kw_name, kw_value))
    returned_kw = parsing.arg_from_keywords(call, kw_name)

    assert kw_name == returned_kw.arg
    assert kw_value == returned_kw.value.s


def test_arg_from_keywords_absent_keyword():
    call = ast_call("MXNet(entry_point='run')")
    assert parsing.arg_from_keywords(call, "framework_version") is None


def test_arg_value():
    call = ast_call("MXNet(framework_version='1.6.0')")
    assert "1.6.0" == parsing.arg_value(call, "framework_version")

    call = ast_call("MXNet(framework_version=mxnet_version)")
    assert "mxnet_version" == parsing.arg_value(call, "framework_version")

    call = ast_call("MXNet(instance_count=1)")
    assert 1 == parsing.arg_value(call, "instance_count")

    call = ast_call("MXNet(enable_network_isolation=True)")
    assert parsing.arg_value(call, "enable_network_isolation") is True

    call = ast_call("MXNet(source_dir=None)")
    assert parsing.arg_value(call, "source_dir") is None


def test_arg_value_absent_keyword():
    code = "MXNet(entry_point='run')"

    with pytest.raises(KeyError) as e:
        parsing.arg_value(ast_call(code), "framework_version")

    assert "arg 'framework_version' not found in call: {}".format(code) in str(e.value)
