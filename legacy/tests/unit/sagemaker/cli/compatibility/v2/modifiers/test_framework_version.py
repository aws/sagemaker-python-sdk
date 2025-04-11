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

from sagemaker.cli.compatibility.v2.modifiers import framework_version
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call


class Template:
    """Essentially a data class with a parametrized format method

    Helper to interpolate various combinations of framework_version, py_version, and image
    as expected by framework and model classes

    TODO: use attrs package and eliminate the boilerplate...
    """

    def __init__(
        self,
        framework,
        framework_version,
        py_version,
        py_version_for_model=True,
    ):
        self.framework = framework
        self.framework_version = framework_version
        self.py_version = py_version
        self.py_version_for_model = py_version_for_model

    def constructors(self, fw_version=False, py_version=False, image=False):
        return self._frameworks(fw_version, py_version, image) + self._models(
            fw_version, py_version, image
        )

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

    def _frameworks(self, fw_version=False, py_version=False, image=False):
        keywords = self._base_keywords(fw_version, image)
        if py_version:
            keywords["py_version"] = (
                "py_version" if py_version == "named" else "'{}'".format(self.py_version)
            )
        return _format_templates(keywords, self._templates())

    def _models(self, fw_version=False, py_version=False, image=False):
        keywords = self._base_keywords(fw_version, image)
        if py_version and self.py_version_for_model:
            keywords["py_version"] = (
                "py_version" if py_version == "named" else "'{}'".format(self.py_version)
            )
        return _format_templates(keywords, self._templates(model=True))

    def _base_keywords(self, fw_version=False, image=False):
        keywords = dict()
        if image:
            keywords["image_uri"] = "'my:image'"
        if fw_version:
            keywords["framework_version"] = (
                "fw_version" if fw_version == "named" else "'{}'".format(self.framework_version)
            )
        return keywords


def _format_templates(keywords, templates):
    args = ", ".join(
        "{key}={value}".format(key=key, value=value) for key, value in keywords.items()
    )

    return [template.format(args) for template in templates]


TEMPLATES = [
    Template(
        framework="TensorFlow",
        framework_version="1.11.0",
        py_version="py2",
        py_version_for_model=False,
    ),
    Template(
        framework="MXNet",
        framework_version="1.2.0",
        py_version="py2",
    ),
    Template(
        framework="Chainer",
        framework_version="4.1.0",
        py_version="py3",
    ),
    Template(
        framework="PyTorch",
        framework_version="0.4.0",
        py_version="py3",
    ),
    Template(
        framework="SKLearn",
        framework_version="0.20.0",
        py_version="py3",
        py_version_for_model=False,
    ),
]


def constructors(fw_version=False, py_version=False, image=False):
    return [
        ctr
        for template in TEMPLATES
        for ctr in template.constructors(fw_version, py_version, image)
    ]


@pytest.fixture
def constructors_empty():
    return constructors()


@pytest.fixture
def constructors_with_only_fw_version_that_need_py_version():
    ctrs = []
    for template in TEMPLATES:
        if template.py_version_for_model:
            ctrs.extend(template.constructors(fw_version=True))
        else:
            ctrs.extend(template._frameworks(fw_version=True))
    return ctrs


@pytest.fixture
def constructors_with_only_fw_version():
    return constructors(fw_version=True)


@pytest.fixture
def constructors_with_only_py_version():
    return constructors(py_version=True)


@pytest.fixture
def constructors_with_both_versions():
    return constructors(fw_version=True, py_version=True)


@pytest.fixture
def constructors_with_image():
    return constructors(image=True)


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


def test_node_should_be_modified_with_only_fw_versions(
    constructors_with_only_fw_version_that_need_py_version,
):
    _test_node_should_be_modified(
        constructors_with_only_fw_version_that_need_py_version, should_modify=True
    )


def test_node_should_be_modified_with_only_py_versions(constructors_with_only_py_version):
    _test_node_should_be_modified(constructors_with_only_py_version, should_modify=True)


def test_node_should_be_modified_with_versions(constructors_with_both_versions):
    _test_node_should_be_modified(constructors_with_both_versions, should_modify=False)


def test_node_should_be_modified_with_image(constructors_with_image):
    _test_node_should_be_modified(constructors_with_image, should_modify=False)


def test_node_should_be_modified_random_function_call():
    _test_node_should_be_modified(["sagemaker.session.Session()"], should_modify=False)


def _test_modify_node(ctrs_before, ctrs_expected):
    modifier = framework_version.FrameworkVersionEnforcer()
    for before, expected in zip(ctrs_before, ctrs_expected):
        node = ast_call(before)
        modifier.modify_node(node)
        _assert_equal_kwargs(ast_call(expected), node)


def _assert_equal_kwargs(expected, actual):
    assert _keywords_for_node(expected) == _keywords_for_node(actual)


def _keywords_for_node(node):
    return {kw.arg: getattr(kw.value, kw.value._fields[0]) for kw in node.keywords}


def test_modify_node_empty(constructors_empty, constructors_with_both_versions):
    _test_modify_node(constructors_empty, constructors_with_both_versions)


def test_modify_node_only_fw_version(
    constructors_with_only_fw_version, constructors_with_both_versions
):
    _test_modify_node(constructors_with_only_fw_version, constructors_with_both_versions)


def test_modify_node_only_py_version(
    constructors_with_only_py_version, constructors_with_both_versions
):
    _test_modify_node(constructors_with_only_py_version, constructors_with_both_versions)


def test_modify_node_only_named_fw_version():
    _test_modify_node(
        constructors(fw_version="named"), constructors(fw_version="named", py_version="literal")
    )


def test_modify_node_only_named_py_version():
    _test_modify_node(
        constructors(py_version="named"), constructors(fw_version="literal", py_version="named")
    )
