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

from sagemaker.deprecations import (
    deprecated_class,
    deprecated_deserialize,
    deprecated_function,
    deprecated_serialize,
    removed_arg,
    removed_function,
    removed_kwargs,
    renamed_kwargs,
    deprecation_warning,
    deprecated,
)


def test_renamed_kwargs():
    kwargs, c = {"a": 1}, 2
    val = renamed_kwargs("b", new_name="c", value=c, kwargs=kwargs)
    assert val == 2

    kwargs, c = {"a": 1, "c": 2}, 2
    val = renamed_kwargs("b", new_name="c", value=c, kwargs=kwargs)
    assert val == 2

    with pytest.warns(DeprecationWarning):
        kwargs, c = {"a": 1, "b": 3}, 2
        val = renamed_kwargs("b", new_name="c", value=c, kwargs=kwargs)
        assert val == 3
        assert kwargs == {"a": 1, "b": 3, "c": 3}


def test_removed_arg():
    arg = None
    removed_arg("b", arg)

    with pytest.warns(DeprecationWarning):
        arg = "it's here"
        removed_arg("b", arg)


def test_removed_kwargs():
    kwarg = {"a": 1}
    removed_kwargs("b", kwarg)

    with pytest.warns(DeprecationWarning):
        kwarg = {"a": 1, "b": 3}
        removed_kwargs("b", kwarg)


def test_deprecation_warning_for_function():
    @deprecation_warning(msg="message", date="date")
    def sample_function():
        return "xxxx...."

    with pytest.warns(DeprecationWarning) as w:
        output = sample_function()
        assert output == "xxxx...."
        msg = (
            "sample_function will be deprecated on date.message in sagemaker>=2.\n"
            "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details."
        )
        assert str(w[-1].message) == msg


def test_deprecation_warning_for_class():
    @deprecation_warning(msg="message", date="date")
    class SampleClass:
        def __init__(self):
            pass

    with pytest.warns(DeprecationWarning) as w:
        SampleClass()
        msg = (
            "SampleClass will be deprecated on date.message in sagemaker>=2.\n"
            "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details."
        )
        assert str(w[-1].message) == msg


def test_deprecation_warning_for_class_method():
    class SampleClass:
        def __init__(self):
            pass

        @deprecation_warning(msg="message", date="date")
        def sample_method(self):
            return "xxxx...."

    s = SampleClass()
    with pytest.warns(DeprecationWarning) as w:
        output = s.sample_method()
        assert output == "xxxx...."
        msg = (
            "sample_method will be deprecated on date.message in sagemaker>=2.\n"
            "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details."
        )
        assert str(w[-1].message) == msg


def test_deprecated_for_function():
    @deprecated()
    def sample_function():
        return "xxxx...."

    with pytest.warns(DeprecationWarning) as w:
        output = sample_function()
        assert output == "xxxx...."
        msg = (
            "sample_function is a no-op in sagemaker>=2.\n"
            "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details."
        )
        assert str(w[-1].message) == msg


def test_deprecated_for_class():
    @deprecated(sdk_version="2.66.6")
    class SampleClass:
        def __init__(self):
            pass

    with pytest.warns(DeprecationWarning) as w:
        SampleClass()
        msg = (
            "SampleClass is a no-op in sagemaker>=2.66.6.\n"
            "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details."
        )
        assert str(w[-1].message) == msg


def test_deprecated_for_class_method():
    class SampleClass:
        def __init__(self):
            pass

        @deprecated()
        def sample_method(self):
            return "xxxx...."

    s = SampleClass()
    with pytest.warns(DeprecationWarning) as w:
        output = s.sample_method()
        assert output == "xxxx...."
        msg = (
            "sample_method is a no-op in sagemaker>=2.\n"
            "See: https://sagemaker.readthedocs.io/en/stable/v2.html for details."
        )
        assert str(w[-1].message) == msg


def test_removed_function():
    removed = removed_function("foo")
    with pytest.warns(DeprecationWarning):
        removed()


def test_removed_function_from_instance():
    class A:
        def func(self):
            return "a"

    a = A()
    a.func = removed_function("foo")
    with pytest.warns(DeprecationWarning):
        a.func()


def test_removed_function_from_class():
    class A:
        func = removed_function("foo")

    a = A()
    with pytest.warns(DeprecationWarning):
        a.func()


def test_deprecated_function():
    def func(a, b):
        return a + b

    deprecated = deprecated_function(func, "foo")
    with pytest.warns(DeprecationWarning):
        assert deprecated(1, 2) == 3


def test_deprecated_serialize():
    class A:
        def serialize(self):
            return 1

    a = deprecated_serialize(A(), "foo")
    with pytest.warns(DeprecationWarning):
        assert a.serialize() == 1


def test_deprecated_deserialize():
    class A:
        def deserialize(self):
            return 1

    a = deprecated_deserialize(A(), "foo")
    with pytest.warns(DeprecationWarning):
        assert a.deserialize() == 1


def test_deprecated_class():
    class A:
        pass

    B = deprecated_class(A, "foo")
    with pytest.warns(DeprecationWarning):
        B()
