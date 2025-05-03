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
"""Generates the code for the service model."""
from sagemaker.core.utils.utils import reformat_file_with_black
from sagemaker.core.tools.shapes_codegen import ShapesCodeGen
from sagemaker.core.tools.resources_codegen import ResourcesCodeGen
from typing import Optional

from sagemaker.core.tools.data_extractor import ServiceJsonData, load_service_jsons


def generate_code(
    shapes_code_gen: Optional[ShapesCodeGen] = None,
    resources_code_gen: Optional[ShapesCodeGen] = None,
) -> None:
    """
    Generates the code for the given code generators. If any code generator is not
    provided when calling this function, the function will initiate the generator.

    Note ordering is important, generate the utils and lower level classes first
    then generate the higher level classes.

    Args:
        shapes_code_gen (ShapesCodeGen): The code generator for shape classes.
        resources_code_gen (ResourcesCodeGen): The code generator for resource classes.

    Returns:
        None
    """
    service_json_data: ServiceJsonData = load_service_jsons()

    shapes_code_gen = shapes_code_gen or ShapesCodeGen()
    resources_code_gen = resources_code_gen or ResourcesCodeGen(
        service_json=service_json_data.sagemaker
    )

    shapes_code_gen.generate_shapes()
    reformat_file_with_black(".")


"""
Initializes all the code generator classes and triggers generator.
"""
if __name__ == "__main__":
    generate_code()
