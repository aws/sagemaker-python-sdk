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

import pytest
from sagemaker import AutoMLV2, CandidateEstimator
from sagemaker.utils import unique_name_from_base

import tests.integ
from tests.conftest import CUSTOM_S3_OBJECT_KEY_PREFIX
from tests.integ import AUTO_ML_DEFAULT_TIMEMOUT_MINUTES, auto_ml_v2_utils
from tests.integ.timeout import timeout

ROLE = "SageMakerRole"
INSTANCE_COUNT = 1


@pytest.fixture(scope="module")
def test_tabular_session_job_name():
    return unique_name_from_base("tabular-job", max_length=32)


@pytest.fixture(scope="module")
def test_image_classification_session_job_name():
    return unique_name_from_base("image-clf-job", max_length=32)


@pytest.fixture(scope="module")
def test_text_classification_session_job_name():
    return unique_name_from_base("text-clf-job", max_length=32)


@pytest.fixture(scope="module")
def test_text_generation_session_job_name():
    return unique_name_from_base("text-gen-job", max_length=32)


@pytest.fixture(scope="module")
def test_time_series_forecasting_session_job_name():
    return unique_name_from_base("ts-forecast-job", max_length=32)


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
@pytest.mark.parametrize(
    "problem_type,job_name_fixture_key",
    [
        pytest.param(
            auto_ml_v2_utils.TABULAR_PROBLEM_TYPE,
            "test_tabular_session_job_name",
            marks=pytest.mark.dependency(name=auto_ml_v2_utils.TABULAR_PROBLEM_TYPE),
        ),
        pytest.param(
            auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE,
            "test_image_classification_session_job_name",
            marks=pytest.mark.dependency(name=auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE,
            "test_text_classification_session_job_name",
            marks=pytest.mark.dependency(name=auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE,
            "test_text_generation_session_job_name",
            marks=pytest.mark.dependency(name=auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE),
        ),
        pytest.param(
            auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE,
            "test_time_series_forecasting_session_job_name",
            marks=pytest.mark.dependency(
                name=auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE
            ),
        ),
    ],
)
def test_auto_ml_v2_describe_auto_ml_job(
    problem_type, job_name_fixture_key, sagemaker_session, request
):
    # Use the request fixture to dynamically get the values of the fixtures
    job_name = request.getfixturevalue(job_name_fixture_key)
    expected_default_output_config = {
        "S3OutputPath": "s3://{}/{}/".format(
            sagemaker_session.default_bucket(), CUSTOM_S3_OBJECT_KEY_PREFIX
        )
    }

    auto_ml_v2_utils.create_auto_ml_job_v2_if_not_exist(sagemaker_session, job_name, problem_type)
    auto_ml = AutoMLV2(
        base_job_name="automl_v2_test",
        role=ROLE,
        sagemaker_session=sagemaker_session,
        problem_config=auto_ml_v2_utils.PROBLEM_CONFIGS[problem_type],
    )

    desc = auto_ml.describe_auto_ml_job(job_name=job_name)
    assert desc["AutoMLJobName"] == job_name
    assert desc["AutoMLJobStatus"] in ["InProgress", "Completed"]
    assert desc["AutoMLJobSecondaryStatus"] != "Failed"
    assert (
        desc["AutoMLProblemTypeConfig"]
        == auto_ml_v2_utils.PROBLEM_CONFIGS[problem_type].to_request_dict()
    )
    assert desc["OutputDataConfig"] == expected_default_output_config


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
@pytest.mark.parametrize(
    "problem_type,job_name_fixture_key",
    [
        pytest.param(
            auto_ml_v2_utils.TABULAR_PROBLEM_TYPE,
            "test_tabular_session_job_name",
            marks=pytest.mark.dependency(depends=[auto_ml_v2_utils.TABULAR_PROBLEM_TYPE]),
        ),
        pytest.param(
            auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE,
            "test_image_classification_session_job_name",
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE]
            ),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE,
            "test_text_classification_session_job_name",
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE]
            ),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE,
            "test_text_generation_session_job_name",
            marks=pytest.mark.dependency(depends=[auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE]),
        ),
        pytest.param(
            auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE,
            "test_time_series_forecasting_session_job_name",
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE]
            ),
        ),
    ],
)
def test_auto_ml_v2_attach(problem_type, job_name_fixture_key, sagemaker_session, request):
    # Use the request fixture to dynamically get the values of the fixtures
    job_name = request.getfixturevalue(job_name_fixture_key)
    expected_default_output_config = {
        "S3OutputPath": "s3://{}/{}/".format(
            sagemaker_session.default_bucket(), CUSTOM_S3_OBJECT_KEY_PREFIX
        )
    }

    auto_ml_v2_utils.create_auto_ml_job_v2_if_not_exist(sagemaker_session, job_name, problem_type)
    attached_automl_job = AutoMLV2.attach(
        auto_ml_job_name=job_name, sagemaker_session=sagemaker_session
    )

    desc = attached_automl_job.describe_auto_ml_job(job_name=job_name)
    assert desc["AutoMLJobName"] == job_name
    assert desc["AutoMLJobStatus"] in ["InProgress", "Completed"]
    assert desc["AutoMLJobSecondaryStatus"] != "Failed"
    assert (
        desc["AutoMLProblemTypeConfig"]
        == auto_ml_v2_utils.PROBLEM_CONFIGS[problem_type].to_request_dict()
    )
    assert desc["OutputDataConfig"] == expected_default_output_config


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
@pytest.mark.parametrize(
    "problem_type,job_name_fixture_key,num_candidates",
    [
        pytest.param(
            auto_ml_v2_utils.TABULAR_PROBLEM_TYPE,
            "test_tabular_session_job_name",
            auto_ml_v2_utils.PROBLEM_CONFIGS[auto_ml_v2_utils.TABULAR_PROBLEM_TYPE].max_candidates,
            marks=pytest.mark.dependency(depends=[auto_ml_v2_utils.TABULAR_PROBLEM_TYPE]),
        ),
        pytest.param(
            auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE,
            "test_image_classification_session_job_name",
            auto_ml_v2_utils.PROBLEM_CONFIGS[
                auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE
            ].max_candidates,
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE]
            ),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE,
            "test_text_classification_session_job_name",
            auto_ml_v2_utils.PROBLEM_CONFIGS[
                auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE
            ].max_candidates,
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE]
            ),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE,
            "test_text_generation_session_job_name",
            auto_ml_v2_utils.PROBLEM_CONFIGS[
                auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE
            ].max_candidates,
            marks=pytest.mark.dependency(depends=[auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE]),
        ),
        pytest.param(
            auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE,
            "test_time_series_forecasting_session_job_name",
            7,
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE]
            ),
        ),
    ],
)
def test_list_candidates(
    problem_type, job_name_fixture_key, num_candidates, sagemaker_session, request
):
    # Use the request fixture to dynamically get the values of the fixtures
    job_name = request.getfixturevalue(job_name_fixture_key)
    auto_ml_v2_utils.create_auto_ml_job_v2_if_not_exist(sagemaker_session, job_name, problem_type)

    attached_automl_job = AutoMLV2.attach(
        auto_ml_job_name=job_name, sagemaker_session=sagemaker_session
    )

    desc = attached_automl_job.describe_auto_ml_job(job_name=job_name)
    if desc["AutoMLJobStatus"] == "Completed":
        auto_ml = AutoMLV2(
            base_job_name="automl_v2_test",
            role=ROLE,
            sagemaker_session=sagemaker_session,
            problem_config=auto_ml_v2_utils.PROBLEM_CONFIGS[problem_type],
        )

        candidates = auto_ml.list_candidates(job_name=job_name)
        assert len(candidates) == num_candidates
    else:
        pytest.skip("The job hasn't finished yet")


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
@pytest.mark.parametrize(
    "problem_type,job_name_fixture_key,num_containers",
    [
        pytest.param(
            auto_ml_v2_utils.TABULAR_PROBLEM_TYPE,
            "test_tabular_session_job_name",
            3,
            marks=pytest.mark.dependency(depends=[auto_ml_v2_utils.TABULAR_PROBLEM_TYPE]),
        ),
        pytest.param(
            auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE,
            "test_image_classification_session_job_name",
            1,
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE]
            ),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE,
            "test_text_classification_session_job_name",
            1,
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE]
            ),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE,
            "test_text_generation_session_job_name",
            1,
            marks=pytest.mark.dependency(depends=[auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE]),
        ),
        pytest.param(
            auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE,
            "test_time_series_forecasting_session_job_name",
            1,
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE]
            ),
        ),
    ],
)
def test_best_candidate(
    problem_type, job_name_fixture_key, num_containers, sagemaker_session, request
):
    # Use the request fixture to dynamically get the values of the fixtures
    job_name = request.getfixturevalue(job_name_fixture_key)
    auto_ml_v2_utils.create_auto_ml_job_v2_if_not_exist(sagemaker_session, job_name, problem_type)

    attached_automl_job = AutoMLV2.attach(
        auto_ml_job_name=job_name, sagemaker_session=sagemaker_session
    )

    desc = attached_automl_job.describe_auto_ml_job(job_name=job_name)
    if desc["AutoMLJobStatus"] == "Completed":
        auto_ml = AutoMLV2(
            base_job_name="automl_v2_test",
            role=ROLE,
            sagemaker_session=sagemaker_session,
            problem_config=auto_ml_v2_utils.PROBLEM_CONFIGS[problem_type],
        )
        best_candidate = auto_ml.best_candidate(job_name=job_name)
        assert len(best_candidate["InferenceContainers"]) == num_containers
        assert best_candidate["CandidateStatus"] == "Completed"
    else:
        pytest.skip("The job hasn't finished yet")


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS
    or tests.integ.test_region() in tests.integ.NO_CANVAS_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
@pytest.mark.release
@pytest.mark.parametrize(
    "problem_type,job_name_fixture_key,instance_type_fixture_key",
    [
        pytest.param(
            auto_ml_v2_utils.TABULAR_PROBLEM_TYPE,
            "test_tabular_session_job_name",
            "cpu_instance_type",
            marks=pytest.mark.dependency(depends=[auto_ml_v2_utils.TABULAR_PROBLEM_TYPE]),
        ),
        pytest.param(
            auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE,
            "test_image_classification_session_job_name",
            "gpu_instance_type",
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE]
            ),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE,
            "test_text_classification_session_job_name",
            "gpu_instance_type",
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE]
            ),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE,
            "test_text_generation_session_job_name",
            "gpu_instance_type",
            marks=pytest.mark.dependency(depends=[auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE]),
        ),
        pytest.param(
            auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE,
            "test_time_series_forecasting_session_job_name",
            "cpu_instance_type",
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE]
            ),
        ),
    ],
)
def test_deploy_best_candidate(
    problem_type,
    job_name_fixture_key,
    instance_type_fixture_key,
    sagemaker_session,
    cpu_instance_type,
    gpu_instance_type,
    request,
):
    # Use the request fixture to dynamically get the values of the fixtures
    job_name = request.getfixturevalue(job_name_fixture_key)
    instance_type = request.getfixturevalue(instance_type_fixture_key)

    auto_ml_v2_utils.create_auto_ml_job_v2_if_not_exist(sagemaker_session, job_name, problem_type)

    attached_automl_job = AutoMLV2.attach(
        auto_ml_job_name=job_name, sagemaker_session=sagemaker_session
    )

    desc = attached_automl_job.describe_auto_ml_job(job_name=job_name)
    if desc["AutoMLJobStatus"] == "Completed":
        auto_ml = AutoMLV2(
            base_job_name="automl_v2_test",
            role=ROLE,
            sagemaker_session=sagemaker_session,
            problem_config=auto_ml_v2_utils.PROBLEM_CONFIGS[problem_type],
        )
        best_candidate = auto_ml.best_candidate(job_name=job_name)
        endpoint_name = unique_name_from_base("sagemaker-auto-ml-best-candidate-test")

        with timeout(minutes=AUTO_ML_DEFAULT_TIMEMOUT_MINUTES):
            auto_ml.deploy(
                candidate=best_candidate,
                initial_instance_count=INSTANCE_COUNT,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
            )

        endpoint_status = sagemaker_session.sagemaker_client.describe_endpoint(
            EndpointName=endpoint_name
        )["EndpointStatus"]
        assert endpoint_status == "InService"
        sagemaker_session.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    else:
        pytest.skip("The job hasn't finished yet")


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_AUTO_ML_REGIONS,
    reason="AutoML is not supported in the region yet.",
)
@pytest.mark.parametrize(
    "problem_type,job_name_fixture_key,num_steps",
    [
        pytest.param(
            auto_ml_v2_utils.TABULAR_PROBLEM_TYPE,
            "test_tabular_session_job_name",
            3,
            marks=pytest.mark.dependency(depends=[auto_ml_v2_utils.TABULAR_PROBLEM_TYPE]),
        ),
        pytest.param(
            auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE,
            "test_image_classification_session_job_name",
            1,
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.IMAGE_CLASSIFICATION_PROBLEM_TYPE]
            ),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE,
            "test_text_classification_session_job_name",
            1,
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.TEXT_CLASSIFICATION_PROBLEM_TYPE]
            ),
        ),
        pytest.param(
            auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE,
            "test_text_generation_session_job_name",
            1,
            marks=pytest.mark.dependency(depends=[auto_ml_v2_utils.TEXT_GENERATION_PROBLEM_TYPE]),
        ),
        pytest.param(
            auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE,
            "test_time_series_forecasting_session_job_name",
            1,
            marks=pytest.mark.dependency(
                depends=[auto_ml_v2_utils.TIME_SERIES_FORECASTING_PROBLEM_TYPE]
            ),
        ),
    ],
)
def test_candidate_estimator_get_steps(
    problem_type, job_name_fixture_key, num_steps, sagemaker_session, request
):
    # Use the request fixture to dynamically get the values of the fixtures
    job_name = request.getfixturevalue(job_name_fixture_key)
    auto_ml_v2_utils.create_auto_ml_job_v2_if_not_exist(sagemaker_session, job_name, problem_type)

    attached_automl_job = AutoMLV2.attach(
        auto_ml_job_name=job_name, sagemaker_session=sagemaker_session
    )
    desc = attached_automl_job.describe_auto_ml_job(job_name=job_name)
    if desc["AutoMLJobStatus"] == "Completed":
        auto_ml = AutoMLV2(
            base_job_name="automl_v2_test",
            role=ROLE,
            sagemaker_session=sagemaker_session,
            problem_config=auto_ml_v2_utils.PROBLEM_CONFIGS[problem_type],
        )
        candidates = auto_ml.list_candidates(job_name=job_name)
        candidate = candidates[0]

        candidate_estimator = CandidateEstimator(candidate, sagemaker_session)
        steps = candidate_estimator.get_steps()
        assert len(steps) == num_steps
    else:
        pytest.skip("The job hasn't finished yet")
