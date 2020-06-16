# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import sys

import pasta
import pytest

from sagemaker.cli.compatibility.v2.modifiers import framework_version
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call


class Template:
    """Essentially a data class with a parametrized format method

    Helper to interpolate various combinations of framework_version, py_version, and image
    as expected by framework and model classes

    TODO: use attrs package and eliminate the boilerplate...
    """

    def __init__(
        self, framework, framework_version, py_version, py_version_for_model=True,
    ):
        self.framework = framework
        self.framework_version = framework_version
        self.py_version = py_version
        self.py_version_for_model = py_version_for_model

    def constructors(self, versions=False, image=False):
        return self._frameworks(versions, image) + self._models(versions, image)

    def _templates(self, model=False):
        module = self.framework.lower()
        submodule = "model" if model else "estimator"
        suffix = "Model" if model else ""
        classname = "{framework}{suffix}".format(framework=self.framework, suffix=suffix)
        templates = (
            "{classname}({{}})",
            "sagemaker.{module}.{classname}({{}})",
            "sagemaker.{module}.{submodule}.{classname}({{}})",
        )
        return tuple(
            template.format(classname=classname, module=module, submodule=submodule)
            for template in templates
        )

    def _frameworks(self, versions=False, image=False):
        keywords = dict()
        if image:
            keywords["image_name"] = "my:image"
        if versions:
            keywords["framework_version"] = self.framework_version
            keywords["py_version"] = self.py_version
        return _format_templates(keywords, self._templates())

    def _models(self, versions=False, image=False):
        keywords = dict()
        if image:
            keywords["image"] = "my:image"
        if versions:
            keywords["framework_version"] = self.framework_version
            if self.py_version_for_model:
                keywords["py_version"] = self.py_version
        return _format_templates(keywords, self._templates(model=True))


def _format_templates(keywords, templates):
    args = ", ".join(
        "{key}='{value}'".format(key=key, value=value) for key, value in keywords.items()
    )
    return [template.format(args) for template in templates]


TEMPLATES = [
    Template(
        framework="TensorFlow",
        framework_version="1.11.0",
        py_version="py2",
        py_version_for_model=False,
    ),
    Template(framework="MXNet", framework_version="1.2.0", py_version="py2",),
    Template(framework="Chainer", framework_version="4.1.0", py_version="py3",),
    Template(framework="PyTorch", framework_version="0.4.0", py_version="py3",),
    Template(
        framework="SKLearn",
        framework_version="0.20.0",
        py_version="py3",
        py_version_for_model=False,
    ),
]


def constructors(versions=False, image=False):
    return [ctr for template in TEMPLATES for ctr in template.constructors(versions, image)]


@pytest.fixture(autouse=True)
def skip_if_py2():
    # Remove once https://github.com/aws/sagemaker-python-sdk/issues/1461 is addressed.
    if sys.version_info.major < 3:
        pytest.skip("v2 migration script doesn't support Python 2.")


@pytest.fixture
def constructors_empty():
    return constructors()


@pytest.fixture
def constructors_with_versions():
    return constructors(versions=True)


@pytest.fixture
def constructors_with_image():
    return constructors(image=True)


@pytest.fixture
def constructors_with_both():
    return constructors(versions=True, image=True)


def _test_node_should_be_modified(ctrs, should_modify=True):
    modifier = framework_version.FrameworkVersionEnforcer()
    for ctr in ctrs:
        node = ast_call(ctr)
        if should_modify:
            assert modifier.node_should_be_modified(node), "{} wasn't modified.".format(ctr)
        else:
            assert not modifier.node_should_be_modified(node), "{} was modified.".format(ctr)


def test_node_should_be_modified_empty(constructors_empty):
    _test_node_should_be_modified(constructors_empty, should_modify=True)


def test_node_should_be_modified_with_versions(constructors_with_versions):
    _test_node_should_be_modified(constructors_with_versions, should_modify=False)


def test_node_should_be_modified_with_image(constructors_with_image):
    _test_node_should_be_modified(constructors_with_image, should_modify=False)


def test_node_should_be_modified_random_function_call():
    _test_node_should_be_modified(["sagemaker.session.Session()"], should_modify=False)


def _test_modify_node(ctrs_before, ctrs_expected):
    modifier = framework_version.FrameworkVersionEnforcer()
    for before, expected in zip(ctrs_before, ctrs_expected):
        node = ast_call(before)
        modifier.modify_node(node)
        # NOTE: this type of equality with pasta depends on ordering of args...
        assert expected == pasta.dump(node)


def test_modify_node_empty(constructors_empty, constructors_with_versions):
    _test_modify_node(constructors_empty, constructors_with_versions)


def test_modify_node_with_versions(constructors_with_versions):
    _test_modify_node(constructors_with_versions, constructors_with_versions)


def test_modify_node_with_image(constructors_with_image, constructors_with_both):
    _test_modify_node(constructors_with_image, constructors_with_both)
