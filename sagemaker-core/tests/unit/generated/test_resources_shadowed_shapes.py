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
"""Guard resource annotations against shape names shadowed by resource classes.

resources.py star-imports every shape and then defines resource classes, some of which
share a shape's name (``Tag``). Before Python 3.14 an annotation is evaluated when its
function or class is defined, so a bare ``Tag`` written above ``class Tag`` means the
shape. Python 3.14 evaluates annotations lazily (PEP 649), so the same text resolves to
the resource class and rejects shape instances (issue #6554).
"""

import ast
import inspect
import typing

import pytest
from pydantic import TypeAdapter

from sagemaker.core import resources, shapes


def _resource_methods_with_tags():
    for _, cls in inspect.getmembers(resources, inspect.isclass):
        if cls.__module__ != resources.__name__:
            continue
        for method_name in ("create", "add_tags"):
            method = cls.__dict__.get(method_name)
            if method is None:
                continue
            func = method.__func__ if isinstance(method, classmethod) else method
            func = inspect.unwrap(func)
            if "tags" in inspect.signature(func).parameters:
                yield f"{cls.__name__}.{method_name}", func


TAGGED_METHODS = sorted(_resource_methods_with_tags())


def test_no_shape_name_used_before_its_shadowing_resource_class():
    """A bare annotation name must not change meaning between eager and lazy evaluation."""
    tree = ast.parse(inspect.getsource(resources))
    shape_names = {name for name, obj in vars(shapes).items() if inspect.isclass(obj)}
    shadowing_class_line = {
        node.name: node.lineno
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in shape_names and node.name != "Base"
    }

    def annotations(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs:
                yield arg.annotation
            yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation

    offenders = set()
    for node in ast.walk(tree):
        for annotation in annotations(node):
            if annotation is None:
                continue
            for name in ast.walk(annotation):
                if (
                    isinstance(name, ast.Name)
                    and name.id in shadowing_class_line
                    and name.lineno < shadowing_class_line[name.id]
                ):
                    offenders.add((name.id, name.lineno))

    assert not offenders, (
        "Annotations reference a shape name that a later resource class shadows; add the "
        f"name to RESOURCE_SHADOWED_SHAPE_NAMES and regenerate: {sorted(offenders)[:10]}"
    )


def test_tagged_methods_found():
    assert len(TAGGED_METHODS) > 50
    assert "HyperParameterTuningJob.create" in dict(TAGGED_METHODS)


@pytest.mark.parametrize("name,func", TAGGED_METHODS, ids=[name for name, _ in TAGGED_METHODS])
def test_tags_annotation_resolves_to_shape_tag(name, func):
    """Every generated ``tags`` parameter takes the shape Tag on every Python version."""
    tags_type = typing.get_type_hints(func)["tags"]
    adapter = TypeAdapter(tags_type)

    validated = adapter.validate_python([shapes.Tag(key="project", value="p1")])
    assert type(validated[0]) is shapes.Tag

    validated = adapter.validate_python([{"key": "project", "value": "p1"}])
    assert type(validated[0]) is shapes.Tag
