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
"""Integration tests for _extract_evaluator_arn in finetune_utils.

These tests exercise the three input paths of _extract_evaluator_arn:
1. Evaluator object  -> returns evaluator.arn directly
2. Evaluator ARN string -> validates and returns the string
3. Lambda ARN string -> auto-creates an Evaluator in AI Registry and returns its ARN
"""

import os
import re
import logging
import pytest
from sagemaker.ai_registry.evaluator import Evaluator
from sagemaker.ai_registry.air_constants import REWARD_FUNCTION
from sagemaker.train.common_utils.finetune_utils import _extract_evaluator_arn

logger = logging.getLogger(__name__)


# Test resource names (ARNs are constructed dynamically from account/region)
EVALUATOR_NAME = "rlvr-eval-lambda-arn-integ-test"
LAMBDA_FUNCTION_NAME = "rlvr-oss-reward-function"
# _extract_evaluator_arn sanitizes the function name (replaces non-alphanumeric/hyphen with -)
SANITIZED_LAMBDA_FUNCTION_NAME = re.sub(r"[^a-zA-Z0-9-]", "-", LAMBDA_FUNCTION_NAME)[:63]


@pytest.fixture(scope="module")
def account_id(sagemaker_session):
    """Resolve the AWS account ID from the current session."""
    return sagemaker_session.boto_session.client("sts").get_caller_identity()["Account"]


@pytest.fixture(scope="module")
def region(sagemaker_session):
    """Resolve the AWS region from the current session."""
    return sagemaker_session.boto_session.region_name


@pytest.fixture(scope="module")
def evaluator(sagemaker_session, lambda_arn):
    """Ensure the evaluator exists in the AI Registry and return its ARN."""
    try:
        evaluator = Evaluator.get(EVALUATOR_NAME, sagemaker_session=sagemaker_session)
    except Exception as e:
        logger.info(
            f"Evaluator '{EVALUATOR_NAME}' not found ({e}). Creating a new one from Lambda: {lambda_arn}"
        )
        evaluator = Evaluator.create(
            name=EVALUATOR_NAME,
            type=REWARD_FUNCTION,
            source=lambda_arn,
            wait=True,
            sagemaker_session=sagemaker_session,
        )
    return evaluator


@pytest.fixture(scope="module")
def lambda_arn(region, account_id):
    """Construct the Lambda function ARN from account and region."""
    return f"arn:aws:lambda:{region}:{account_id}:function:{LAMBDA_FUNCTION_NAME}"


def test_extract_evaluator_arn_with_evaluator_object(evaluator):

    result = _extract_evaluator_arn(evaluator, "custom_reward_function")

    # Should return the evaluator's ARN directly
    assert result == evaluator.arn
    assert result.startswith("arn:aws:sagemaker:")
    assert "/JsonDoc/" in result


def test_extract_evaluator_arn_with_evaluator_string(sagemaker_session, evaluator):
    """Test _extract_evaluator_arn with a valid evaluator hub-content ARN string.

    Verifies that passing a valid SageMaker hub-content ARN string passes
    validation and is returned as-is.
    """
    result = _extract_evaluator_arn(evaluator.arn, "custom_reward_function")

    # Should return the ARN string unchanged
    assert result == evaluator.arn

