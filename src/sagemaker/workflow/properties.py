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
"""The properties definitions for workflow."""
from __future__ import absolute_import

from typing import Dict, Union

import attr

import botocore.loaders

from sagemaker.workflow.entities import Expression


class PropertiesMeta(type):
    """Load an internal shapes attribute from the botocore sagemaker service model."""

    _shapes = None
    _primitive_types = {"string", "boolean", "integer", "float"}

    def __new__(mcs, *args, **kwargs):
        """Loads up the shapes from the botocore sagemaker service model."""
        if mcs._shapes is None:
            loader = botocore.loaders.Loader()
            model = loader.load_service_model("sagemaker", "service-2")
            mcs._shapes = model["shapes"]
        return super().__new__(mcs, *args, **kwargs)


class Properties(metaclass=PropertiesMeta):
    """Properties for use in workflow expressions."""

    def __init__(self, path: str, shape_name: str = None):
        """Create a Properties instance representing the given shape.

        Args:
            path (str): The parent path of the Properties instance.
            shape_name (str): The botocore sagemaker service model shape name.
        """
        self._path = path
        self._shape_name = shape_name

        shape = Properties._shapes.get(self._shape_name, {})
        shape_type = shape.get("type")
        if shape_type in Properties._primitive_types:
            self.__str__ = shape_name
        elif shape_type == "structure":
            members = shape["members"]
            for key, info in members.items():
                if Properties._shapes.get(info["shape"], {}).get("type") == "list":
                    self.__dict__[key] = PropertiesList(f"{path}.{key}", info["shape"])
                else:
                    self.__dict__[key] = Properties(f"{path}.{key}", info["shape"])

    @property
    def expr(self):
        """The 'Get' expression dict for a `Properties`."""
        return {"Get": self._path}


class PropertiesList(Properties):
    """PropertiesList for use in workflow expressions."""

    def __init__(self, path: str, shape_name: str = None):
        """Create a PropertiesList instance representing the given shape.

        Args:
            path (str): The parent path of the PropertiesList instance.
            shape_name (str): The botocore sagemaker service model shape name.
        """
        super(PropertiesList, self).__init__(path, shape_name)
        self._items: Dict[Union[int, str], Properties] = dict()

    def __getitem__(self, item: Union[int, str]):
        """Populate the indexing item with a Property, for both lists and dictionaries.

        Args:
            item (Union[int, str]): The index of the item in sequence.
        """
        if item not in self._items.keys():
            shape = Properties._shapes.get(self._shape_name)
            member = shape["member"]["shape"]
            if isinstance(item, str):
                property_item = Properties(f"{self._path}['{item}']", member)
            else:
                property_item = Properties(f"{self._path}[{item}]", member)
            self._items[item] = property_item

        return self._items.get(item)


@attr.s
class PropertyFile(Expression):
    """Provides a property file struct.

    Attributes:
        name: The name of the property file for reference with `JsonGet` functions.
        output_name: The name of the processing job output channel.
        path: The path to the file at the output channel location.
    """

    name: str = attr.ib()
    output_name: str = attr.ib()
    path: str = attr.ib()

    @property
    def expr(self) -> Dict[str, str]:
        """The expression dict for a `PropertyFile`."""
        return {
            "PropertyFileName": self.name,
            "OutputName": self.output_name,
            "FilePath": self.path,
        }
