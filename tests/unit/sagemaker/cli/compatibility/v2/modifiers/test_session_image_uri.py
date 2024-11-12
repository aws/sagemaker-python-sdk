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

import pasta

from sagemaker.cli.compatibility.v2.modifiers import renamed_params
from tests.unit.sagemaker.cli.compatibility.v2.modifiers.ast_converter import ast_call

CREATE_MODEL_TEMPLATES = (
    "sagemaker_session.create_model_from_job({})",
    "sess.create_model_from_job({})",
)

CREATE_ENDPOINT_TEMPLATES = (
    "sagemaker_session.endpoint_from_job({})",
    "sagemaker_session.endpoint_from_model_data({})",
    "sess.endpoint_from_job({})",
    "sess.endpoint_from_model_data({})",
)


def test_create_model_node_should_be_modified():
    modifier = renamed_params.SessionCreateModelImageURIRenamer()

    for template in CREATE_MODEL_TEMPLATES:
        call = ast_call(template.format("primary_container_image=my_image"))
        assert modifier.node_should_be_modified(call)


def test_create_model_node_should_be_modified_no_image():
    modifier = renamed_params.SessionCreateModelImageURIRenamer()

    for template in CREATE_MODEL_TEMPLATES:
        call = ast_call(template.format(""))
        assert not modifier.node_should_be_modified(call)


def test_create_model_node_should_be_modified_random_function_call():
    modifier = renamed_params.SessionCreateModelImageURIRenamer()
    assert not modifier.node_should_be_modified(ast_call("create_model()"))


def test_create_model_modify_node():
    modifier = renamed_params.SessionCreateModelImageURIRenamer()

    for template in CREATE_MODEL_TEMPLATES:
        call = ast_call(template.format("primary_container_image=my_image"))
        modifier.modify_node(call)

        expected = template.format("image_uri=my_image")
        assert expected == pasta.dump(call)


def test_create_endpoint_node_should_be_modified():
    modifier = renamed_params.SessionCreateEndpointImageURIRenamer()

    for template in CREATE_ENDPOINT_TEMPLATES:
        call = ast_call(template.format("deployment_image=my_image"))
        assert modifier.node_should_be_modified(call)


def test_create_endpoint_node_should_be_modified_no_image():
    modifier = renamed_params.SessionCreateEndpointImageURIRenamer()

    for template in CREATE_ENDPOINT_TEMPLATES:
        call = ast_call(template.format(""))
        assert not modifier.node_should_be_modified(call)


def test_create_endpoint_node_should_be_modified_random_function_call():
    modifier = renamed_params.SessionCreateEndpointImageURIRenamer()
    assert not modifier.node_should_be_modified(ast_call("create_endpoint()"))


def test_create_endpoint_modify_node():
    modifier = renamed_params.SessionCreateEndpointImageURIRenamer()

    for template in CREATE_ENDPOINT_TEMPLATES:
        call = ast_call(template.format("deployment_image=my_image"))
        modifier.modify_node(call)

        expected = template.format("image_uri=my_image")
        assert expected == pasta.dump(call)
