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

import itertools
import os
import time

import pandas
import pytest

import sagemaker
import tests.integ
from sagemaker import AlgorithmEstimator, ModelPackage
from sagemaker.serializers import CSVSerializer
from sagemaker.tuner import IntegerParameter, HyperparameterTuner
from sagemaker.utils import sagemaker_timestamp, _aws_partition, unique_name_from_base
from tests.integ import DATA_DIR
from tests.integ.timeout import timeout, timeout_and_delete_endpoint_by_name
from tests.integ.marketplace_utils import REGION_ACCOUNT_MAP


# All these tests require a manual 1 time subscription to the following Marketplace items:
# Algorithm: Scikit Decision Trees
# https://aws.amazon.com/marketplace/pp/prodview-ha4f3kqugba3u
#
# Pre-Trained Model: Scikit Decision Trees - Pretrained Model
# https://aws.amazon.com/marketplace/pp/prodview-7qop4x5ahrdhe
#
# Both are written by Amazon and are free to subscribe.

ALGORITHM_ARN = (
    "arn:{partition}:sagemaker:{region}:{account}:algorithm/scikit-decision-trees-"
    "15423055-57b73412d2e93e9239e4e16f83298b8f"
)

MODEL_PACKAGE_ARN = (
    "arn:{partition}:sagemaker:{region}:{account}:model-package/scikit-iris-detector-"
    "154230595-8f00905c1f927a512b73ea29dd09ae30"
)


@pytest.mark.release
@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
@pytest.mark.skip(
    reason="This test has always failed, but the failure was masked by a bug. "
    "This test should be fixed. Details in https://github.com/aws/sagemaker-python-sdk/pull/968"
)
def test_marketplace_estimator(sagemaker_session, cpu_instance_type):
    with timeout(minutes=15):
        data_path = os.path.join(DATA_DIR, "marketplace", "training")
        region = sagemaker_session.boto_region_name
        account = REGION_ACCOUNT_MAP[region]
        algorithm_arn = ALGORITHM_ARN.format(
            partition=_aws_partition(region), region=region, account=account
        )

        algo = AlgorithmEstimator(
            algorithm_arn=algorithm_arn,
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
        )

        train_input = algo.sagemaker_session.upload_data(
            path=data_path, key_prefix="integ-test-data/marketplace/train"
        )

        algo.fit({"training": train_input})

    endpoint_name = "test-marketplace-estimator{}".format(sagemaker_timestamp())
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        predictor = algo.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        shape = pandas.read_csv(os.path.join(data_path, "iris.csv"), header=None)

        a = [50 * i for i in range(3)]
        b = [40 + i for i in range(10)]
        indices = [i + j for i, j in itertools.product(a, b)]

        test_data = shape.iloc[indices[:-1]]
        test_x = test_data.iloc[:, 1:]

        print(predictor.predict(test_x.values).decode("utf-8"))


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
def test_marketplace_attach(sagemaker_session, cpu_instance_type):
    with timeout(minutes=15):
        data_path = os.path.join(DATA_DIR, "marketplace", "training")
        region = sagemaker_session.boto_region_name
        account = REGION_ACCOUNT_MAP[region]
        algorithm_arn = ALGORITHM_ARN.format(
            partition=_aws_partition(region), region=region, account=account
        )

        mktplace = AlgorithmEstimator(
            algorithm_arn=algorithm_arn,
            role="SageMakerRole",
            instance_count=1,
            instance_type=cpu_instance_type,
            sagemaker_session=sagemaker_session,
            base_job_name=unique_name_from_base("test-marketplace"),
        )

        train_input = mktplace.sagemaker_session.upload_data(
            path=data_path, key_prefix="integ-test-data/marketplace/train"
        )

        mktplace.fit({"training": train_input}, wait=False)
        training_job_name = mktplace.latest_training_job.name

        print("Waiting to re-attach to the training job: %s" % training_job_name)
        time.sleep(20)
        endpoint_name = "test-marketplace-estimator{}".format(sagemaker_timestamp())

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        print("Re-attaching now to: %s" % training_job_name)
        estimator = AlgorithmEstimator.attach(
            training_job_name=training_job_name, sagemaker_session=sagemaker_session
        )
        predictor = estimator.deploy(
            1, cpu_instance_type, endpoint_name=endpoint_name, serializer=CSVSerializer()
        )
        shape = pandas.read_csv(os.path.join(data_path, "iris.csv"), header=None)
        a = [50 * i for i in range(3)]
        b = [40 + i for i in range(10)]
        indices = [i + j for i, j in itertools.product(a, b)]

        test_data = shape.iloc[indices[:-1]]
        test_x = test_data.iloc[:, 1:]

        print(predictor.predict(test_x.values).decode("utf-8"))


@pytest.mark.release
@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
def test_marketplace_model(sagemaker_session, cpu_instance_type):
    region = sagemaker_session.boto_region_name
    account = REGION_ACCOUNT_MAP[region]
    model_package_arn = MODEL_PACKAGE_ARN.format(
        partition=_aws_partition(region), region=region, account=account
    )

    def predict_wrapper(endpoint, session):
        return sagemaker.Predictor(endpoint, session, serializer=CSVSerializer())

    model = ModelPackage(
        role="SageMakerRole",
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session,
        predictor_cls=predict_wrapper,
    )

    endpoint_name = "test-marketplace-model-endpoint{}".format(sagemaker_timestamp())
    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=20):
        predictor = model.deploy(1, cpu_instance_type, endpoint_name=endpoint_name)
        data_path = os.path.join(DATA_DIR, "marketplace", "training")
        shape = pandas.read_csv(os.path.join(data_path, "iris.csv"), header=None)
        a = [50 * i for i in range(3)]
        b = [40 + i for i in range(10)]
        indices = [i + j for i, j in itertools.product(a, b)]

        test_data = shape.iloc[indices[:-1]]
        test_x = test_data.iloc[:, 1:]

        print(predictor.predict(test_x.values).decode("utf-8"))


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
def test_marketplace_tuning_job(sagemaker_session, cpu_instance_type):
    data_path = os.path.join(DATA_DIR, "marketplace", "training")
    region = sagemaker_session.boto_region_name
    account = REGION_ACCOUNT_MAP[region]
    algorithm_arn = ALGORITHM_ARN.format(
        partition=_aws_partition(region), region=region, account=account
    )

    mktplace = AlgorithmEstimator(
        algorithm_arn=algorithm_arn,
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name=unique_name_from_base("test-marketplace"),
    )

    train_input = mktplace.sagemaker_session.upload_data(
        path=data_path, key_prefix="integ-test-data/marketplace/train"
    )

    mktplace.set_hyperparameters(max_leaf_nodes=10)

    hyperparameter_ranges = {"max_leaf_nodes": IntegerParameter(1, 100000)}

    tuner = HyperparameterTuner(
        estimator=mktplace,
        base_tuning_job_name=unique_name_from_base("byo"),
        objective_metric_name="validation:accuracy",
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=2,
        max_parallel_jobs=2,
    )

    tuner.fit({"training": train_input}, include_cls_metadata=False)
    time.sleep(15)
    tuner.wait()


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
def test_marketplace_transform_job(sagemaker_session, cpu_instance_type):
    data_path = os.path.join(DATA_DIR, "marketplace", "training")
    region = sagemaker_session.boto_region_name
    account = REGION_ACCOUNT_MAP[region]
    algorithm_arn = ALGORITHM_ARN.format(
        partition=_aws_partition(region), region=region, account=account
    )

    algo = AlgorithmEstimator(
        algorithm_arn=algorithm_arn,
        role="SageMakerRole",
        instance_count=1,
        instance_type=cpu_instance_type,
        sagemaker_session=sagemaker_session,
        base_job_name=unique_name_from_base("test-marketplace"),
    )

    train_input = algo.sagemaker_session.upload_data(
        path=data_path, key_prefix="integ-test-data/marketplace/train"
    )

    shape = pandas.read_csv(data_path + "/iris.csv", header=None).drop([0], axis=1)

    transform_workdir = DATA_DIR + "/marketplace/transform"
    shape.to_csv(transform_workdir + "/batchtransform_test.csv", index=False, header=False)
    transform_input = algo.sagemaker_session.upload_data(
        transform_workdir, key_prefix="integ-test-data/marketplace/transform"
    )

    algo.fit({"training": train_input})

    transformer = algo.transformer(1, cpu_instance_type)
    transformer.transform(transform_input, content_type="text/csv")
    transformer.wait()


@pytest.mark.skipif(
    tests.integ.test_region() in tests.integ.NO_MARKET_PLACE_REGIONS,
    reason="Marketplace is not available in {}".format(tests.integ.test_region()),
)
def test_marketplace_transform_job_from_model_package(sagemaker_session, cpu_instance_type):
    data_path = os.path.join(DATA_DIR, "marketplace", "training")
    shape = pandas.read_csv(data_path + "/iris.csv", header=None).drop([0], axis=1)

    TRANSFORM_WORKDIR = DATA_DIR + "/marketplace/transform"
    shape.to_csv(TRANSFORM_WORKDIR + "/batchtransform_test.csv", index=False, header=False)
    transform_input = sagemaker_session.upload_data(
        TRANSFORM_WORKDIR, key_prefix="integ-test-data/marketplace/transform"
    )

    region = sagemaker_session.boto_region_name
    account = REGION_ACCOUNT_MAP[region]
    model_package_arn = MODEL_PACKAGE_ARN.format(
        partition=_aws_partition(region), region=region, account=account
    )

    model = ModelPackage(
        role="SageMakerRole",
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session,
    )

    transformer = model.transformer(1, cpu_instance_type)
    transformer.transform(transform_input, content_type="text/csv")
    transformer.wait()
