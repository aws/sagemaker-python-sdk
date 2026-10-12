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
"""Tags passed to the HyperParameterTuningJob created by HyperparameterTuner."""

from __future__ import absolute_import

from unittest.mock import MagicMock, patch

import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.parameter import ContinuousParameter
from sagemaker.core.shapes import Tag
from sagemaker.train.configs import Compute
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.train.tuner import HyperparameterTuner

ROLE = "arn:aws:iam::123456789012:role/role"
IMAGE = "123456789012.dkr.ecr.us-west-2.amazonaws.com/image:latest"


@pytest.fixture
def session():
    session = MagicMock(spec=Session)
    session.default_bucket.return_value = "bucket"
    session.default_bucket_prefix = None
    session.boto_region_name = "us-west-2"
    session.boto_session = MagicMock()
    session.sagemaker_config = None
    session.settings = MagicMock(include_jumpstart_tags=False)
    return session


@patch("sagemaker.train.tuner.HyperParameterTuningJob.create")
@patch("sagemaker.train.defaults.resolve_and_validate_role", return_value=ROLE)
def test_tune_accepts_model_trainer_tag_objects_and_tag_dicts(_, mock_create, session):
    model_trainer = ModelTrainer(
        training_image=IMAGE,
        role=ROLE,
        sagemaker_session=session,
        compute=Compute(instance_type="ml.m5.xlarge", instance_count=1),
        hyperparameters={"lr": 0.1},
        tags=[Tag(key="team", value="ml")],
    )
    tuner = HyperparameterTuner(
        model_trainer,
        "accuracy",
        {"lr": ContinuousParameter(0.001, 0.1)},
        max_jobs=2,
        tags=[{"Key": "project", "Value": "hpo"}],
    )

    tuner.tune(inputs="s3://bucket/train", wait=False)

    assert mock_create.call_args.kwargs["tags"] == [
        Tag(key="project", value="hpo"),
        Tag(key="team", value="ml"),
    ]
