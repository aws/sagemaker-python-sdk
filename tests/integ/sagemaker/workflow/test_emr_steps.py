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

import re

import pytest

from sagemaker import get_execution_role, utils
from sagemaker.workflow.emr_step import EMRStep, EMRStepConfig
from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.retry import StepRetryPolicy, StepExceptionTypeEnum


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def pipeline_name():
    return utils.unique_name_from_base("my-pipeline-emr")


@pytest.fixture
def region_name(sagemaker_session):
    return sagemaker_session.boto_session.region_name


def test_two_steps_emr_pipeline(sagemaker_session, role, pipeline_name, region_name):
    instance_count = ParameterInteger(name="InstanceCount", default_value=2)

    emr_step_config = EMRStepConfig(
        jar="s3://us-west-2.elasticmapreduce/libs/script-runner/script-runner.jar",
        args=["dummy_emr_script_path"],
    )

    step_emr_1 = EMRStep(
        name="emr-step-1",
        cluster_id="j-1YONHTCP3YZKC",
        display_name="emr_step_1",
        description="MyEMRStepDescription",
        step_config=emr_step_config,
    )

    step_emr_2 = EMRStep(
        name="emr-step-2",
        cluster_id=step_emr_1.properties.ClusterId,
        display_name="emr_step_2",
        description="MyEMRStepDescription",
        step_config=emr_step_config,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[instance_count],
        steps=[step_emr_1, step_emr_2],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_emr_with_cluster_config(sagemaker_session, role, pipeline_name, region_name):

    emr_step_config = EMRStepConfig(
        jar="s3://us-west-2.elasticmapreduce/libs/script-runner/script-runner.jar",
        args=["dummy_emr_script_path"],
    )

    cluster_config = {
        "Instances": {
            "InstanceGroups": [
                {
                    "Name": "Master Instance Group",
                    "InstanceRole": "MASTER",
                    "InstanceCount": 1,
                    "InstanceType": "m1.small",
                    "Market": "ON_DEMAND",
                }
            ],
            "InstanceCount": 1,
            "HadoopVersion": "MyHadoopVersion",
        },
        "AmiVersion": "3.8.0",
        "AdditionalInfo": "MyAdditionalInfo",
    }

    step_emr_with_cluster_config = EMRStep(
        name="MyEMRStep-name",
        display_name="MyEMRStep-display_name",
        description="MyEMRStepDescription",
        cluster_id=None,
        step_config=emr_step_config,
        cluster_config=cluster_config,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_emr_with_cluster_config],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_emr_with_retry_policies(sagemaker_session, role, pipeline_name, region_name):
    """Test EMR steps with retry policies in both cluster_id and cluster_config scenarios."""
    emr_step_config = EMRStepConfig(
        jar="s3://us-west-2.elasticmapreduce/libs/script-runner/script-runner.jar",
        args=["dummy_emr_script_path"],
    )

    retry_policies = [
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.SERVICE_FAULT],
            interval_seconds=1,
            max_attempts=3,
            backoff_rate=2.0,
        )
    ]

    # Step with existing cluster and retry policies
    step_emr_1 = EMRStep(
        name="emr-step-1",
        cluster_id="j-1YONHTCP3YZKC",
        display_name="emr_step_1",
        description="EMR Step with retry policies",
        step_config=emr_step_config,
        retry_policies=retry_policies,
    )

    # Step with cluster config and retry policies
    cluster_config = {
        "Instances": {
            "InstanceGroups": [
                {
                    "Name": "Master Instance Group",
                    "InstanceRole": "MASTER",
                    "InstanceCount": 1,
                    "InstanceType": "m1.small",
                    "Market": "ON_DEMAND",
                }
            ],
            "InstanceCount": 1,
            "HadoopVersion": "MyHadoopVersion",
        },
        "AmiVersion": "3.8.0",
        "AdditionalInfo": "MyAdditionalInfo",
    }

    step_emr_2 = EMRStep(
        name="emr-step-2",
        display_name="emr_step_2",
        description="EMR Step with cluster config and retry policies",
        cluster_id=None,
        step_config=emr_step_config,
        cluster_config=cluster_config,
        retry_policies=retry_policies,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_emr_1, step_emr_2],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_emr_with_expire_after_retry_policy(sagemaker_session, role, pipeline_name, region_name):
    """Test EMR step with retry policy using expire_after_mins."""
    emr_step_config = EMRStepConfig(
        jar="s3://us-west-2.elasticmapreduce/libs/script-runner/script-runner.jar",
        args=["dummy_emr_script_path"],
    )

    retry_policies = [
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.SERVICE_FAULT],
            interval_seconds=1,
            expire_after_mins=30,
            backoff_rate=2.0,
        )
    ]

    step_emr = EMRStep(
        name="emr-step-expire",
        cluster_id="j-1YONHTCP3YZKC",
        display_name="emr_step_expire",
        description="EMR Step with expire after retry policy",
        step_config=emr_step_config,
        retry_policies=retry_policies,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_emr],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_emr_with_multiple_exception_types(sagemaker_session, role, pipeline_name, region_name):
    """Test EMR step with multiple exception types in retry policy."""
    retry_policies = [
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.SERVICE_FAULT, StepExceptionTypeEnum.THROTTLING],
            interval_seconds=1,
            max_attempts=3,
            backoff_rate=2.0,
        )
    ]

    step_emr = EMRStep(
        name="emr-step-multi-except",
        cluster_id="j-1YONHTCP3YZKC",
        display_name="emr_step_multi_except",
        description="EMR Step with multiple exception types",
        step_config=EMRStepConfig(
            jar="s3://us-west-2.elasticmapreduce/libs/script-runner/script-runner.jar",
            args=["dummy_emr_script_path"],
        ),
        retry_policies=retry_policies,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_emr],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass


def test_emr_with_multiple_retry_policies(sagemaker_session, role, pipeline_name, region_name):
    """Test EMR step with multiple retry policies."""
    retry_policies = [
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.SERVICE_FAULT],
            interval_seconds=1,
            max_attempts=3,
            backoff_rate=2.0,
        ),
        StepRetryPolicy(
            exception_types=[StepExceptionTypeEnum.THROTTLING],
            interval_seconds=5,
            expire_after_mins=60,
            backoff_rate=1.5,
        ),
    ]

    step_emr = EMRStep(
        name="emr-step-multi-policy",
        cluster_id="j-1YONHTCP3YZKC",
        display_name="emr_step_multi_policy",
        description="EMR Step with multiple retry policies",
        step_config=EMRStepConfig(
            jar="s3://us-west-2.elasticmapreduce/libs/script-runner/script-runner.jar",
            args=["dummy_emr_script_path"],
        ),
        retry_policies=retry_policies,
    )

    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_emr],
        sagemaker_session=sagemaker_session,
    )

    try:
        response = pipeline.create(role)
        create_arn = response["PipelineArn"]
        assert re.match(
            rf"arn:aws:sagemaker:{region_name}:\d{{12}}:pipeline/{pipeline_name}",
            create_arn,
        )
    finally:
        try:
            pipeline.delete()
        except Exception:
            pass
