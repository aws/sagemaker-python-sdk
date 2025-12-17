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
"""Test for JumpStart search_public_hub_models function."""
from __future__ import absolute_import

import pytest
from sagemaker.core.jumpstart.search import search_public_hub_models
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import HubContent


@pytest.mark.serial
@pytest.mark.integ
def test_search_public_hub_models_default_args():
    # Only query, uses default hub name and session
    query = "@task:text-generation OR @framework:huggingface"
    results = search_public_hub_models(query)

    assert isinstance(results, list)
    assert all(isinstance(m, HubContent) for m in results)
    assert len(results) > 0, "Expected at least one matching model from the public hub"


@pytest.mark.serial
@pytest.mark.integ
def test_search_public_hub_models_custom_session():
    # Provide a custom SageMaker session
    session = Session()
    query = "@task:text-generation"
    results = search_public_hub_models(query, sagemaker_session=session)

    assert isinstance(results, list)
    assert all(isinstance(m, HubContent) for m in results)


@pytest.mark.serial
@pytest.mark.integ
def test_search_public_hub_models_custom_hub_name():
    # Using the default public hub but provided explicitly
    query = "@framework:huggingface"
    results = search_public_hub_models(query, hub_name="SageMakerPublicHub")

    assert isinstance(results, list)
    assert all(isinstance(m, HubContent) for m in results)


@pytest.mark.serial
@pytest.mark.integ
def test_search_public_hub_models_all_args():
    # Provide both hub_name and session explicitly
    session = Session()
    query = "@task:natural-language-processing"
    results = search_public_hub_models(
        query, hub_name="SageMakerPublicHub", sagemaker_session=session
    )

    assert isinstance(results, list)
    assert all(isinstance(m, HubContent) for m in results)
