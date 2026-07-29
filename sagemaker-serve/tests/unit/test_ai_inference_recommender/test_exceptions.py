# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for AI inference recommender exceptions."""
from __future__ import absolute_import

from sagemaker.core.utils.exceptions import SageMakerCoreError, ValidationError

from sagemaker.serve.ai_inference_recommender.exceptions import (
    FeatureGatedError,
    WorkloadValidationError,
)


def test_feature_gated_error_includes_runbook_url():
    err = FeatureGatedError(message="account not enrolled")
    assert "account not enrolled" in str(err)
    assert "generative-ai-inference-recommendations" in err.runbook_url
    assert isinstance(err, SageMakerCoreError)


def test_workload_validation_error_chains_message():
    err = WorkloadValidationError(message="missing required field 'tokenizer'")
    assert "missing required field" in str(err)
    assert isinstance(err, ValidationError)
