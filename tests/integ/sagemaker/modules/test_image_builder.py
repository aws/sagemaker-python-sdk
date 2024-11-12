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
"""This module contains code to test image builder"""
from __future__ import absolute_import

import os

from sagemaker.modules.image_builder import build_image_from_base, build_image_from_dockerfile


def test_build_public_image_locally():
    build_image_from_base(image_name="python_310", base_image="python:3.10")


def test_build_with_dependency_file():
    dependency_file_path = os.getcwd() + "/tests/integ/sagemaker/modules/requirements.txt"
    build_image_from_base(
        image_name="ubuntu_with_dependencies", dependency_file=dependency_file_path
    )


def test_build_image_and_push_to_ecr():
    dependency_file_path = os.getcwd() + "/tests/integ/sagemaker/modules/environment.yml"
    build_image_from_base(
        image_name="ecr_test_image",
        dependency_file=dependency_file_path,
        base_image="debian",
        deploy_to_ecr=True,
        ecr_repo_name="image_builder_integ_test",
    )


def test_build_image_from_dockerfile():
    dockerfile_path = os.getcwd() + "/tests/integ/sagemaker/modules/Dockerfile"
    build_image_from_dockerfile(
        image_name="image_from_dockerfile", dockerfile=dockerfile_path, deploy_to_ecr=True
    )
