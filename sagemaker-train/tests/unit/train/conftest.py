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
"""Shared fixtures for trainer unit tests."""
import pytest


@pytest.fixture(autouse=True)
def _skip_hub_model_validation(request, monkeypatch):
    """Skip the live Hub availability check during trainer construction.

    Trainers resolve a raw base model name through ``_resolve_model_and_name``,
    which validates that the model exists in the SageMaker Hub via a
    DescribeHubContent call. Trainer unit tests build trainers with placeholder
    model names against mock sessions and must not reach the network, so this
    autouse fixture turns the check into a no-op for these tests.

    The check's own behavior is covered directly in
    ``tests/unit/train/common_utils/test_finetune_utils.py``; that directory is
    excluded here so those tests exercise the real function.
    """
    if "common_utils" in str(request.node.fspath):
        return
    monkeypatch.setattr(
        "sagemaker.train.common_utils.finetune_utils._validate_model_in_hub",
        lambda *args, **kwargs: None,
    )
