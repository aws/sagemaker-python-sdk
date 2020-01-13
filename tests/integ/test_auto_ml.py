# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os
import time

import pytest
import tests.integ
from sagemaker import AutoML, CandidateEstimator, AutoMLInput

from sagemaker.exceptions import UnexpectedStatusException
from sagemaker.utils import unique_name_from_base
from tests.integ import DATA_DIR, AUTO_ML_DEFAULT_TIMEMOUT_MINUTES, auto_ml_utils
from tests.integ.timeout import timeout

ROLE = "SageMakerRole"
PREFIX = "sagemaker/beta-automl-xgboost"
AUTO_ML_INSTANCE_TYPE = "ml.m5.2xlarge"
INSTANCE_COUNT = 1
RESOURCE_POOLS = [{"InstanceType": AUTO_ML_INSTANCE_TYPE, "PoolSize": INSTANCE_COUNT}]
TARGET_ATTRIBUTE_NAME = "virginica"
DATA_DIR = os.path.join(DATA_DIR, "automl", "data")
TRAINING_DATA = os.path.join(DATA_DIR, "iris_training.csv")
TEST_DATA = os.path.join(DATA_DIR, "iris_test.csv")
PROBLEM_TYPE = "MultiClassClassification"
JOB_NAME = "auto-ml-{}".format(time.strftime("%y%m%d-%H%M%S"))

# use a succeeded AutoML job to test describe and list candidates method, otherwise tests will run too long
AUTO_ML_JOB_NAME = "python-sdk-integ-test-base-job"

EXPECTED_DEFAULT_JOB_CONFIG = {
    "CompletionCriteria": {"MaxCandidates": 3},
    "SecurityConfig": {"EnableInterContainerTrafficEncryption": False},
}


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
@pytest.mark.canary_quick
def test_auto_ml_fit(sagemaker_session):
    auto_ml = AutoML(
        role=ROLE,
        target_attribute_name=TARGET_ATTRIBUTE_NAME,
        sagemaker_session=sagemaker_session,
        max_candidates=3,
    )

    job_name = unique_name_from_base("auto-ml", max_length=32)
    inputs = sagemaker_session.upload_data(path=TRAINING_DATA, key_prefix=PREFIX + "/input")
    with timeout(minutes=AUTO_ML_DEFAULT_TIMEMOUT_MINUTES):
        auto_ml.fit(inputs, job_name=job_name)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
def test_auto_ml_fit_local_input(sagemaker_session):
    auto_ml = AutoML(
        role=ROLE,
        target_attribute_name=TARGET_ATTRIBUTE_NAME,
        sagemaker_session=sagemaker_session,
        max_candidates=1,
    )

    inputs = TRAINING_DATA
    job_name = unique_name_from_base("auto-ml", max_length=32)
    with timeout(minutes=AUTO_ML_DEFAULT_TIMEMOUT_MINUTES):
        auto_ml.fit(inputs, job_name=job_name)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
def test_auto_ml_input_object_fit(sagemaker_session):
    auto_ml = AutoML(
        role=ROLE,
        target_attribute_name=TARGET_ATTRIBUTE_NAME,
        sagemaker_session=sagemaker_session,
        max_candidates=1,
    )
    job_name = unique_name_from_base("auto-ml", max_length=32)
    s3_input = sagemaker_session.upload_data(path=TRAINING_DATA, key_prefix=PREFIX + "/input")
    inputs = AutoMLInput(inputs=s3_input, target_attribute_name=TARGET_ATTRIBUTE_NAME)
    with timeout(minutes=AUTO_ML_DEFAULT_TIMEMOUT_MINUTES):
        auto_ml.fit(inputs, job_name=job_name)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
def test_auto_ml_fit_optional_args(sagemaker_session):
    output_path = "s3://{}/{}".format(sagemaker_session.default_bucket(), "specified_ouput_path")
    problem_type = "MulticlassClassification"
    job_objective = {"MetricName": "Accuracy"}
    auto_ml = AutoML(
        role=ROLE,
        target_attribute_name=TARGET_ATTRIBUTE_NAME,
        sagemaker_session=sagemaker_session,
        max_candidates=1,
        output_path=output_path,
        problem_type=problem_type,
        job_objective=job_objective,
    )
    inputs = TRAINING_DATA
    with timeout(minutes=AUTO_ML_DEFAULT_TIMEMOUT_MINUTES):
        auto_ml.fit(inputs, job_name=JOB_NAME)

    auto_ml_desc = auto_ml.describe_auto_ml_job(job_name=JOB_NAME)
    assert auto_ml_desc["AutoMLJobStatus"] == "Completed"
    assert auto_ml_desc["AutoMLJobName"] == JOB_NAME
    assert auto_ml_desc["AutoMLJobObjective"] == job_objective
    assert auto_ml_desc["ProblemType"] == problem_type
    assert auto_ml_desc["OutputDataConfig"]["S3OutputPath"] == output_path


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
def test_auto_ml_invalid_target_attribute(sagemaker_session):
    auto_ml = AutoML(
        role=ROLE, target_attribute_name="y", sagemaker_session=sagemaker_session, max_candidates=1
    )
    job_name = unique_name_from_base("auto-ml", max_length=32)
    inputs = sagemaker_session.upload_data(path=TRAINING_DATA, key_prefix=PREFIX + "/input")
    with pytest.raises(
        UnexpectedStatusException, match="Could not complete the data builder processing job."
    ):
        auto_ml.fit(inputs, job_name=job_name)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
def test_auto_ml_describe_auto_ml_job(sagemaker_session):
    expected_default_input_config = [
        {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": "s3://{}/{}/input/iris_training.csv".format(
                        sagemaker_session.default_bucket(), PREFIX
                    ),
                }
            },
            "TargetAttributeName": TARGET_ATTRIBUTE_NAME,
        }
    ]
    expected_default_output_config = {
        "S3OutputPath": "s3://{}/".format(sagemaker_session.default_bucket())
    }

    auto_ml_utils.create_auto_ml_job_if_not_exist(sagemaker_session)
    auto_ml = AutoML(
        role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
    )

    desc = auto_ml.describe_auto_ml_job(job_name=AUTO_ML_JOB_NAME)
    assert desc["AutoMLJobName"] == AUTO_ML_JOB_NAME
    assert desc["AutoMLJobStatus"] == "Completed"
    assert isinstance(desc["BestCandidate"], dict)
    assert desc["InputDataConfig"] == expected_default_input_config
    assert desc["AutoMLJobConfig"] == EXPECTED_DEFAULT_JOB_CONFIG
    assert desc["OutputDataConfig"] == expected_default_output_config


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
def test_list_candidates(sagemaker_session):
    auto_ml_utils.create_auto_ml_job_if_not_exist(sagemaker_session)

    auto_ml = AutoML(
        role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
    )

    candidates = auto_ml.list_candidates(job_name=AUTO_ML_JOB_NAME)
    assert len(candidates) == 3


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
def test_best_candidate(sagemaker_session):
    auto_ml_utils.create_auto_ml_job_if_not_exist(sagemaker_session)

    auto_ml = AutoML(
        role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
    )
    best_candidate = auto_ml.best_candidate(job_name=AUTO_ML_JOB_NAME)
    assert len(best_candidate["InferenceContainers"]) == 3
    assert len(best_candidate["CandidateSteps"]) == 4
    assert best_candidate["CandidateStatus"] == "Completed"


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
@pytest.mark.canary_quick
def test_deploy_best_candidate(sagemaker_session, cpu_instance_type):
    auto_ml_utils.create_auto_ml_job_if_not_exist(sagemaker_session)

    auto_ml = AutoML(
        role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
    )
    best_candidate = auto_ml.best_candidate(job_name=AUTO_ML_JOB_NAME)
    endpoint_name = unique_name_from_base("sagemaker-auto-ml-best-candidate-test")

    with timeout(minutes=AUTO_ML_DEFAULT_TIMEMOUT_MINUTES):
        auto_ml.deploy(
            candidate=best_candidate,
            initial_instance_count=INSTANCE_COUNT,
            instance_type=cpu_instance_type,
            endpoint_name=endpoint_name,
        )

    endpoint_status = sagemaker_session.sagemaker_client.describe_endpoint(
        EndpointName=endpoint_name
    )["EndpointStatus"]
    assert endpoint_status == "InService"
    sagemaker_session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
def test_candidate_estimator_default_rerun_and_deploy(sagemaker_session, cpu_instance_type):
    auto_ml_utils.create_auto_ml_job_if_not_exist(sagemaker_session)

    auto_ml = AutoML(
        role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
    )

    candidates = auto_ml.list_candidates(job_name=AUTO_ML_JOB_NAME)
    candidate = candidates[1]

    candidate_estimator = CandidateEstimator(candidate, sagemaker_session)
    inputs = sagemaker_session.upload_data(path=TEST_DATA, key_prefix=PREFIX + "/input")
    endpoint_name = unique_name_from_base("sagemaker-auto-ml-rerun-candidate-test")
    with timeout(minutes=AUTO_ML_DEFAULT_TIMEMOUT_MINUTES):
        candidate_estimator.fit(inputs)
        auto_ml.deploy(
            initial_instance_count=INSTANCE_COUNT,
            instance_type=cpu_instance_type,
            candidate=candidate,
            endpoint_name=endpoint_name,
        )

    endpoint_status = sagemaker_session.sagemaker_client.describe_endpoint(
        EndpointName=endpoint_name
    )["EndpointStatus"]
    assert endpoint_status == "InService"
    sagemaker_session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
def test_candidate_estimator_rerun_with_optional_args(sagemaker_session, cpu_instance_type):
    auto_ml_utils.create_auto_ml_job_if_not_exist(sagemaker_session)

    auto_ml = AutoML(
        role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
    )

    candidates = auto_ml.list_candidates(job_name=AUTO_ML_JOB_NAME)
    candidate = candidates[1]

    candidate_estimator = CandidateEstimator(candidate, sagemaker_session)
    inputs = sagemaker_session.upload_data(path=TEST_DATA, key_prefix=PREFIX + "/input")
    endpoint_name = unique_name_from_base("sagemaker-auto-ml-rerun-candidate-test")
    with timeout(minutes=AUTO_ML_DEFAULT_TIMEMOUT_MINUTES):
        candidate_estimator.fit(inputs, encrypt_inter_container_traffic=True)
        auto_ml.deploy(
            initial_instance_count=INSTANCE_COUNT,
            instance_type=cpu_instance_type,
            candidate=candidate,
            endpoint_name=endpoint_name,
        )

    endpoint_status = sagemaker_session.sagemaker_client.describe_endpoint(
        EndpointName=endpoint_name
    )["EndpointStatus"]
    assert endpoint_status == "InService"
    sagemaker_session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
def test_candidate_estimator_get_steps(sagemaker_session):
    auto_ml_utils.create_auto_ml_job_if_not_exist(sagemaker_session)

    auto_ml = AutoML(
        role=ROLE, target_attribute_name=TARGET_ATTRIBUTE_NAME, sagemaker_session=sagemaker_session
    )
    candidates = auto_ml.list_candidates(job_name=AUTO_ML_JOB_NAME)
    candidate = candidates[1]

    candidate_estimator = CandidateEstimator(candidate, sagemaker_session)
    steps = candidate_estimator.get_steps()
    assert len(steps) == 3
